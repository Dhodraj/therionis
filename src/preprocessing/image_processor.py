"""
Image preprocessing utilities for medical imaging
"""
import io
import numpy as np
from PIL import Image
import cv2
import pydicom
from typing import Tuple, Optional
import torch
from torchvision import transforms
from config.settings import settings


class ImagePreprocessor:
    """
    Class for preprocessing medical images for model inference
    """
    
    def __init__(self):
        self.max_size = settings.max_image_size
        self.supported_formats = settings.supported_formats
        
        # Standard transforms for X-ray images (torchxrayvision expects specific normalization)
        self.xray_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # Normalize to the range expected by torchxrayvision (-1024 to 1024)
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Maps [0,1] to [-1,1]
            transforms.Lambda(lambda x: x * 1024)  # Scale to [-1024, 1024]
        ])
    
    def load_image(self, file_content: bytes, filename: str) -> np.ndarray:
        """
        Load image from bytes, supporting multiple formats including DICOM
        
        Args:
            file_content: Raw file bytes
            filename: Original filename for format detection
            
        Returns:
            numpy array of the image
        """
        file_extension = self._get_file_extension(filename)
        
        if file_extension in ['.dcm', '.dicom']:
            return self._load_dicom(file_content)
        else:
            return self._load_standard_image(file_content)
    
    def _load_dicom(self, file_content: bytes) -> np.ndarray:
        """
        Load DICOM image from bytes
        """
        try:
            # Create a file-like object from bytes
            dicom_file = io.BytesIO(file_content)
            dicom_data = pydicom.dcmread(dicom_file)
            
            # Extract pixel array
            image_array = dicom_data.pixel_array
            
            # Normalize to 0-255 range
            if image_array.dtype != np.uint8:
                image_min = image_array.min()
                image_max = image_array.max()
                if image_max > image_min:
                    image_array = ((image_array - image_min) / (image_max - image_min) * 255).astype(np.uint8)
                else:
                    image_array = np.zeros_like(image_array, dtype=np.uint8)
            
            # Convert to RGB if grayscale
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Failed to load DICOM image: {str(e)}")
    
    def _load_standard_image(self, file_content: bytes) -> np.ndarray:
        """
        Load standard image formats (JPG, PNG) from bytes
        """
        try:
            # Load image using PIL
            image = Image.open(io.BytesIO(file_content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    def preprocess_for_inference(self, image_array: np.ndarray, model_type: str = "xray") -> torch.Tensor:
        """
        Preprocess image for model inference
        
        Args:
            image_array: Input image as numpy array
            model_type: Type of model ("xray", "ct", "mri")
            
        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)
        
        image = Image.fromarray(image_array)
        
        if model_type == "xray":
            # Apply X-ray specific preprocessing
            tensor = self.xray_transforms(image)
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            return tensor
        else:
            # Generic preprocessing for other modalities
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            tensor = transform(image)
            tensor = tensor.unsqueeze(0)
            return tensor
    
    def resize_image(self, image_array: np.ndarray, max_size: Optional[int] = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image_array: Input image
            max_size: Maximum dimension size
            
        Returns:
            Resized image array
        """
        if max_size is None:
            max_size = self.max_size
        
        height, width = image_array.shape[:2]
        
        if max(height, width) <= max_size:
            return image_array
        
        # Calculate new dimensions
        if height > width:
            new_height = max_size
            new_width = int((width * max_size) / height)
        else:
            new_width = max_size
            new_height = int((height * max_size) / width)
        
        # Resize using OpenCV
        resized = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def validate_image(self, file_content: bytes, filename: str) -> Tuple[bool, str]:
        """
        Validate image format and size
        
        Args:
            file_content: Raw file bytes
            filename: Original filename
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if len(file_content) > settings.max_file_size:
            return False, f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
        
        # Check file extension
        file_extension = self._get_file_extension(filename)
        if file_extension not in self.supported_formats:
            return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
        
        # Try to load the image
        try:
            self.load_image(file_content, filename)
            return True, ""
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def _get_file_extension(self, filename: str) -> str:
        """
        Get file extension in lowercase with dot prefix
        """
        if '.' not in filename:
            return ""
        extension = filename.lower().split('.')[-1]
        return f".{extension}"
    
    def get_image_info(self, image_array: np.ndarray) -> dict:
        """
        Get basic information about the image
        
        Args:
            image_array: Input image array
            
        Returns:
            Dictionary with image information
        """
        return {
            "shape": image_array.shape,
            "dtype": str(image_array.dtype),
            "min_value": float(image_array.min()),
            "max_value": float(image_array.max()),
            "mean_value": float(image_array.mean()),
        }