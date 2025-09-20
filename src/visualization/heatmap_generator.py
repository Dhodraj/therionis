"""
Visualization module for generating heatmaps and annotated medical images
"""
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchcam import methods as cam_methods
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from skimage import measure
from skimage.morphology import binary_opening, disk


class MedicalImageVisualizer:
    """
    Class for creating heatmaps and annotated visualizations of medical images
    """
    
    def __init__(self):
        self.cam_extractor = None
        self.colormap = cm.get_cmap('jet')
        
    def generate_gradcam_heatmap(
        self, 
        model: torch.nn.Module, 
        input_tensor: torch.Tensor,
        target_layer: str = None,
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for model predictions
        
        Args:
            model: PyTorch model
            input_tensor: Input image tensor (1, C, H, W)
            target_layer: Name of target layer for CAM (if None, uses last conv layer)
            target_class: Target class index (if None, uses predicted class)
            
        Returns:
            Heatmap as numpy array
        """
        try:
            # Ensure model is in eval mode but enable gradients for CAM
            model.eval()
            
            # Enable gradients for the input tensor
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            
            # Ensure model parameters have gradients enabled
            for param in model.parameters():
                param.requires_grad_(True)
            
            # Set up CAM extractor
            if target_layer is None:
                # Find the last convolutional layer
                target_layer = self._find_last_conv_layer(model)
            
            cam_extractor = cam_methods.GradCAM(model, target_layer)
            
            # Get model prediction with gradients enabled
            outputs = model(input_tensor)
            probabilities = torch.sigmoid(outputs)
            
            # Use the class with highest probability if target_class not specified
            if target_class is None:
                target_class = probabilities.argmax().item()
            
            # Generate activation map - this needs gradients
            activation_map = cam_extractor(target_class, outputs)
            
            # Convert to numpy and resize to match input image
            heatmap = activation_map[0].cpu().numpy()
            
            return heatmap
            
        except Exception as e:
            print(f"Error generating Grad-CAM: {e}")
            # Return a meaningful empty heatmap based on input size
            input_size = input_tensor.shape[-2:]  # Get H, W from (N, C, H, W)
            return np.zeros(input_size)
    
    def _find_last_conv_layer(self, model: torch.nn.Module) -> str:
        """
        Find the name of the last convolutional layer in the model
        """
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(name)
        
        if conv_layers:
            return conv_layers[-1]
        else:
            # Fallback for models where we can't find conv layers
            return "features"
    
    def create_overlay_visualization(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Create overlay visualization combining original image with heatmap
        
        Args:
            original_image: Original image as numpy array (H, W, 3)
            heatmap: Heatmap as numpy array (H, W)
            alpha: Transparency of heatmap overlay (0-1)
            colormap: Matplotlib colormap name
            
        Returns:
            Overlaid image as numpy array
        """
        # Ensure images are in the correct format
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif len(original_image.shape) == 3 and original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match original image size
        if heatmap.shape != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Normalize heatmap to 0-1 range
        if heatmap.max() > heatmap.min():
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            # Handle case where heatmap is all zeros or constant
            heatmap_norm = np.zeros_like(heatmap)
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_rgba = cmap(heatmap_norm)  # This gives (H, W, 4) with RGBA
        heatmap_colored = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)  # Take only RGB channels
        
        # Ensure both images have the same shape for cv2.addWeighted
        if original_image.shape != heatmap_colored.shape:
            # Resize heatmap_colored to match original_image dimensions but keep 3 channels
            heatmap_colored = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))
            # Ensure it has 3 channels
            if len(heatmap_colored.shape) == 2:
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2RGB)
        
        # Create overlay
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def create_annotated_image(
        self,
        original_image: np.ndarray,
        predictions: List[Dict],
        heatmap: Optional[np.ndarray] = None,
        case_id: str = "",
        max_predictions: int = 10
    ) -> np.ndarray:
        """
        Create comprehensive annotated image with detailed medical annotations
        
        Args:
            original_image: Original image as numpy array
            predictions: List of prediction dictionaries
            heatmap: Optional heatmap to overlay
            case_id: Case identifier for annotation
            max_predictions: Maximum number of predictions to display
            
        Returns:
            Annotated image as numpy array
        """
        # Convert to PIL Image for easier text rendering
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Create base image with compact layout
        img_height, img_width = original_image.shape[:2]
        
        # Create a compact canvas for professional annotations
        canvas_width = img_width + 350  # Compact annotation space
        canvas_height = max(img_height, 650)  # Minimal height for annotations
        
        # Create white canvas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # Place original image on the left side
        if heatmap is not None:
            # Create overlay with enhanced opacity settings
            overlay_img = self.create_enhanced_overlay(original_image, heatmap)
            # Find and highlight regions of interest
            annotated_with_boxes = self._add_region_boxes(overlay_img, heatmap, predictions)
            canvas[:img_height, :img_width] = annotated_with_boxes
        else:
            canvas[:img_height, :img_width] = original_image
        
        # Convert to PIL for detailed annotations
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)
        
        # Load professional fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
        except:
            title_font = header_font = body_font = small_font = ImageFont.load_default()
        
        # Create detailed annotation panel
        self._create_annotation_panel(draw, img_width, canvas_height, predictions, case_id, 
                                    title_font, header_font, body_font, small_font)
        
        # Convert back to numpy array
        return np.array(img_pil)
    
    def _get_confidence_color(self, confidence: float) -> Tuple[int, int, int]:
        """
        Get color based on confidence level
        """
        if confidence >= 0.7:
            return (255, 100, 100)  # Red for high confidence
        elif confidence >= 0.5:
            return (255, 255, 100)  # Yellow for medium confidence
        else:
            return (255, 255, 255)  # White for low confidence
    
    def _get_timestamp(self) -> str:
        """
        Get current timestamp string
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _get_short_timestamp(self) -> str:
        """
        Get shorter timestamp string for compact display
        """
        from datetime import datetime
        return datetime.now().strftime("%m/%d %H:%M")
    
    def _get_short_assessment(self, assessment: str) -> str:
        """
        Get short assessment text for compact display
        """
        if "URGENT" in assessment:
            return "⚠️ URGENT - Seek Medical Attention"
        elif "MODERATE CONCERN" in assessment:
            return "⚡ Moderate Concern - Consult Doctor"
        elif "FINDINGS DETECTED" in assessment:
            return "ℹ️ Findings Detected - Follow Up"
        else:
            return "✅ Likely Safe - Routine Monitoring"
    
    def _get_safety_assessment(self, predictions: List[Dict]) -> str:
        """
        Determine overall safety assessment based on realistic predictions
        """
        if not predictions:
            return "✅ LIKELY SAFE - ROUTINE MONITORING"
            
        # Check for critical conditions with high confidence
        critical_conditions = ['Pneumonia', 'Pneumothorax', 'Lung Lesion', 'Mass', 'Nodule']
        high_conf_critical = [p for p in predictions if p['confidence'] >= 0.75 and 
                             any(crit in p['condition'] for crit in critical_conditions)]
        
        # Check for serious conditions with medium-high confidence  
        serious_conditions = ['Cardiomegaly', 'Effusion', 'Edema', 'Consolidation']
        medium_high_serious = [p for p in predictions if p['confidence'] >= 0.65 and 
                              any(serious in p['condition'] for serious in serious_conditions)]
        
        # Check for moderate findings
        moderate_findings = [p for p in predictions if 0.6 <= p['confidence'] < 0.75]
        
        if high_conf_critical:
            return "⚠️ URGENT - SEEK MEDICAL ATTENTION"
        elif medium_high_serious:
            return "⚡ MODERATE CONCERN - CONSULT DOCTOR"
        elif moderate_findings:
            return "ℹ️ FINDINGS DETECTED - FOLLOW UP"
        else:
            return "✅ LIKELY SAFE - ROUTINE MONITORING"
    
    def _get_safety_color(self, assessment: str) -> Tuple[int, int, int]:
        """
        Get background color for safety assessment
        """
        if "URGENT" in assessment:
            return (255, 200, 200)  # Light red
        elif "MODERATE CONCERN" in assessment:
            return (255, 235, 200)  # Light orange
        elif "FINDINGS DETECTED" in assessment:
            return (255, 255, 200)  # Light yellow
        else:
            return (200, 255, 200)  # Light green
    
    def _is_serious_condition(self, condition: str) -> bool:
        """
        Determine if a condition is considered serious and requires immediate attention
        """
        serious_conditions = [
            'pneumonia', 'pneumothorax', 'pleural effusion', 'cardiomegaly',
            'lung lesion', 'consolidation', 'atelectasis', 'edema',
            'infiltration', 'mass', 'nodule'
        ]
        return any(serious in condition.lower() for serious in serious_conditions)
    
    def _get_recommended_action(self, assessment: str) -> str:
        """
        Get recommended action based on safety assessment
        """
        if "URGENT" in assessment:
            return "Contact emergency services or visit ER immediately. Do not delay medical care."
        elif "MODERATE CONCERN" in assessment:
            return "Schedule appointment with doctor within 24-48 hours for evaluation."
        elif "FINDINGS DETECTED" in assessment:
            return "Follow up with healthcare provider within 1-2 weeks for review."
        else:
            return "Continue routine monitoring. Discuss with doctor at next regular visit."
    
    def _get_condition_explanation(self, condition: str) -> str:
        """
        Get plain language explanation of medical conditions
        """
        explanations = {
            'pneumonia': 'Infection or inflammation in lung tissue',
            'pneumothorax': 'Air leak causing lung collapse',
            'pleural effusion': 'Fluid buildup around lungs',
            'cardiomegaly': 'Enlarged heart',
            'lung lesion': 'Abnormal spot or growth in lung',
            'consolidation': 'Dense area in lung (often infection)',
            'atelectasis': 'Collapsed or partially collapsed lung',
            'edema': 'Fluid buildup in lungs',
            'infiltration': 'Abnormal substance in lung tissue',
            'mass': 'Large abnormal growth',
            'nodule': 'Small growth or spot in lung',
            'effusion': 'Fluid accumulation',
            'opacity': 'Dense area visible on scan',
            'no finding': 'No abnormalities detected'
        }
        
        condition_lower = condition.lower()
        for key, explanation in explanations.items():
            if key in condition_lower:
                return explanation
        
        return "Abnormal finding requiring medical evaluation"
    
    def _wrap_text(self, text: str, max_chars: int) -> List[str]:
        """
        Wrap text to fit within specified character limit
        """
        if len(text) <= max_chars:
            return [text]
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def save_visualization(
        self,
        annotated_image: np.ndarray,
        save_path: Path,
        quality: int = 95
    ) -> bool:
        """
        Save annotated image to file
        
        Args:
            annotated_image: Annotated image array
            save_path: Path to save the image
            quality: JPEG quality (1-100)
            
        Returns:
            Success status
        """
        try:
            # Convert to PIL and save
            img_pil = Image.fromarray(annotated_image)
            
            # Create directory if it doesn't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with appropriate format
            if save_path.suffix.lower() in ['.jpg', '.jpeg']:
                img_pil.save(save_path, 'JPEG', quality=quality)
            else:
                img_pil.save(save_path, 'PNG')
            
            return True
            
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return False
    
    def create_comparison_grid(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        annotated_image: np.ndarray,
        predictions: List[Dict]
    ) -> np.ndarray:
        """
        Create a comparison grid showing original, heatmap, and annotated versions
        
        Args:
            original_image: Original image
            heatmap: Generated heatmap
            annotated_image: Annotated image with predictions
            predictions: List of predictions
            
        Returns:
            Grid image as numpy array
        """
        # Resize all images to same size
        target_size = (512, 512)
        
        orig_resized = cv2.resize(original_image, target_size)
        
        # Create heatmap visualization
        heatmap_vis = self.create_overlay_visualization(original_image, heatmap)
        heatmap_resized = cv2.resize(heatmap_vis, target_size)
        
        annotated_resized = cv2.resize(annotated_image, target_size)
        
        # Create grid (1x3)
        grid = np.hstack([orig_resized, heatmap_resized, annotated_resized])
        
        # Add labels
        grid_pil = Image.fromarray(grid)
        draw = ImageDraw.Draw(grid_pil)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add labels for each section
        labels = ["Original", "Heatmap", "Predictions"]
        for i, label in enumerate(labels):
            x_pos = i * target_size[0] + 10
            draw.text((x_pos, 10), label, fill=(255, 255, 255), font=font, 
                     stroke_width=2, stroke_fill=(0, 0, 0))
        
        return np.array(grid_pil)
    
    def create_enhanced_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.3,
        colormap: str = 'jet'
    ) -> np.ndarray:
        """
        Create enhanced overlay with better visualization
        """
        # Ensure images are in the correct format
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Convert grayscale to RGB if needed
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        elif len(original_image.shape) == 3 and original_image.shape[2] == 1:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Resize heatmap to match original image size
        if heatmap.shape != original_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        
        # Normalize heatmap with better contrast
        if heatmap.max() > heatmap.min():
            heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap_norm = np.zeros_like(heatmap)
        
        # Apply threshold to focus on high-activation areas
        threshold = 0.5
        heatmap_norm[heatmap_norm < threshold] = 0
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_rgba = cmap(heatmap_norm)  # This gives (H, W, 4) with RGBA
        heatmap_colored = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)  # Take only RGB channels
        
        # Ensure both images have the same shape for cv2.addWeighted
        if original_image.shape != heatmap_colored.shape:
            # Resize heatmap_colored to match original_image dimensions but keep 3 channels
            heatmap_colored = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))
            # Ensure it has 3 channels
            if len(heatmap_colored.shape) == 2:
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2RGB)
        
        # Create overlay with enhanced visibility
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def _add_region_boxes(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        predictions: List[Dict],
        threshold: float = 0.6
    ) -> np.ndarray:
        """
        Add bounding boxes around regions of interest in the heatmap
        """
        try:
            # Resize heatmap to match image size
            if heatmap.shape != image.shape[:2]:
                heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            else:
                heatmap_resized = heatmap.copy()
            
            # Ensure heatmap is single channel for processing
            if len(heatmap_resized.shape) > 2:
                # If heatmap has multiple channels, take the first channel or convert to grayscale
                if heatmap_resized.shape[2] == 3:
                    heatmap_resized = cv2.cvtColor(heatmap_resized, cv2.COLOR_BGR2GRAY)
                else:
                    heatmap_resized = heatmap_resized[:, :, 0]  # Take first channel
            
            # Normalize heatmap
            heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
            
            # Create binary mask for high-activation regions
            binary_mask = heatmap_norm > threshold
            
            # Clean up the mask
            try:
                binary_mask = binary_opening(binary_mask, disk(3))
            except:
                # Fallback if skimage is not available
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            
            # Find contours
            binary_uint8 = binary_mask.astype(np.uint8) * 255
            
            # Ensure binary_uint8 is single channel
            if len(binary_uint8.shape) > 2:
                binary_uint8 = cv2.cvtColor(binary_uint8, cv2.COLOR_BGR2GRAY)
            
            contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw bounding boxes for significant regions
            result_image = image.copy()
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Color based on the highest confidence prediction
                    color = self._get_box_color(predictions, i)
                    
                    # Draw bounding box
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                    
                    # Add small label if there's a corresponding prediction
                    if i < len(predictions) and predictions[i]['confidence'] > 0.5:
                        label = f"{predictions[i]['condition'][:8]}..."
                        cv2.putText(result_image, label, (x, y - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return result_image
            
        except Exception as e:
            print(f"Error adding region boxes: {e}")
            return image
    
    def _get_box_color(self, predictions: List[Dict], index: int) -> Tuple[int, int, int]:
        """Get color for bounding box based on prediction confidence"""
        if index < len(predictions):
            confidence = predictions[index]['confidence']
            if confidence >= 0.7:
                return (0, 0, 255)  # Red for high confidence (BGR format)
            elif confidence >= 0.5:
                return (0, 165, 255)  # Orange for medium confidence
            else:
                return (255, 255, 0)  # Cyan for low confidence
        return (128, 128, 128)  # Gray for unknown
    
    def _create_annotation_panel(
        self,
        draw: ImageDraw.Draw,
        img_width: int,
        canvas_height: int,
        predictions: List[Dict],
        case_id: str,
        title_font,
        header_font,
        body_font,
        small_font
    ):
        """Create compact professional annotation panel"""
        panel_x = img_width + 10
        panel_width = 330  # Compact panel width
        max_y = canvas_height - 20  # Minimal bottom margin
        
        # Draw panel background
        panel_bg = (245, 245, 245)  # Light gray background
        draw.rectangle([panel_x - 10, 0, panel_x + panel_width, canvas_height], fill=panel_bg)
        
        # Compact Assessment Header
        y_pos = 15
        safety_assessment = self._get_safety_assessment(predictions)
        safety_color = self._get_safety_color(safety_assessment)
        
        # Modern compact assessment box
        draw.rectangle([panel_x, y_pos, panel_x + panel_width - 10, y_pos + 35], 
                      fill=safety_color, outline=(120, 120, 120), width=1)
        
        # Single line assessment with icon
        assessment_short = self._get_short_assessment(safety_assessment)
        draw.text((panel_x + 8, y_pos + 8), assessment_short, fill=(0, 0, 0), font=body_font)
        y_pos += 45
        
        # Skip Next Steps section as requested - removed entirely
        
        # Compact case info - single line
        info_text = f"ID: {case_id[:6].upper()} | {self._get_short_timestamp()}" if case_id else f"Generated: {self._get_short_timestamp()}"
        draw.text((panel_x + 5, y_pos), info_text, fill=(120, 120, 120), font=small_font)
        y_pos += 20
        
        # Minimal divider and findings header
        draw.line([panel_x, y_pos, panel_x + panel_width - 10, y_pos], fill=(200, 200, 200), width=1)
        y_pos += 10
        draw.text((panel_x + 5, y_pos), "FINDINGS", fill=(60, 60, 60), font=header_font)
        y_pos += 20
        
        # Group predictions by severity (exclude empty conditions)
        valid_predictions = [p for p in predictions if p.get('condition') and p['condition'].strip()]
        urgent_findings = [p for p in valid_predictions if p['confidence'] >= 0.7]
        moderate_findings = [p for p in valid_predictions if 0.4 <= p['confidence'] < 0.7]
        low_findings = [p for p in valid_predictions if 0.1 <= p['confidence'] < 0.4]
        
        # Compact critical findings
        if urgent_findings and y_pos + 30 < max_y:
            for i, pred in enumerate(urgent_findings[:3]):  # Show up to 3
                if y_pos + 25 > max_y:
                    break
                
                # Skip predictions with empty or invalid condition names
                if not pred.get('condition') or not pred['condition'].strip():
                    continue
                    
                confidence_percent = pred['confidence'] * 100
                condition_text = pred['condition'][:18] + "..." if len(pred['condition']) > 18 else pred['condition']
                
                # Compact finding card
                draw.rectangle([panel_x, y_pos, panel_x + panel_width - 10, y_pos + 20], 
                              fill=(255, 240, 240), outline=(220, 120, 120), width=1)
                
                # Condition and confidence in single line
                draw.text((panel_x + 6, y_pos + 3), f"⚠️ {condition_text}", fill=(160, 0, 0), font=small_font)
                draw.text((panel_x + 250, y_pos + 3), f"{confidence_percent:.0f}%", fill=(160, 0, 0), font=small_font)
                y_pos += 25
        
        # Compact moderate findings
        if moderate_findings and y_pos + 25 < max_y:
            for pred in moderate_findings[:2]:  # Show up to 2
                if y_pos + 25 > max_y:
                    break
                
                # Skip predictions with empty or invalid condition names
                if not pred.get('condition') or not pred['condition'].strip():
                    continue
                    
                confidence_percent = pred['confidence'] * 100
                condition_text = pred['condition'][:18] + "..." if len(pred['condition']) > 18 else pred['condition']
                
                # Compact finding card
                draw.rectangle([panel_x, y_pos, panel_x + panel_width - 10, y_pos + 20], 
                              fill=(255, 250, 235), outline=(220, 180, 120), width=1)
                
                # Condition and confidence in single line
                draw.text((panel_x + 6, y_pos + 3), f"⚡ {condition_text}", fill=(140, 90, 0), font=small_font)
                draw.text((panel_x + 250, y_pos + 3), f"{confidence_percent:.0f}%", fill=(140, 90, 0), font=small_font)
                y_pos += 25
        
        # Compact minor findings or clear results
        if not urgent_findings and not moderate_findings:
            if low_findings and y_pos + 20 < max_y:
                # Find first low finding with valid condition name
                valid_low_finding = None
                for finding in low_findings:
                    if finding.get('condition') and finding['condition'].strip():
                        valid_low_finding = finding
                        break
                
                if valid_low_finding:
                    pred = valid_low_finding
                    confidence_percent = pred['confidence'] * 100
                    condition_text = pred['condition'][:22] + "..." if len(pred['condition']) > 22 else pred['condition']
                    
                    draw.rectangle([panel_x, y_pos, panel_x + panel_width - 10, y_pos + 18], 
                                  fill=(248, 248, 248), outline=(180, 180, 180), width=1)
                    draw.text((panel_x + 6, y_pos + 2), f"ℹ️ {condition_text}: {confidence_percent:.0f}%", 
                             fill=(100, 100, 100), font=small_font)
            elif y_pos + 20 < max_y:
                # Clear results
                draw.rectangle([panel_x, y_pos, panel_x + panel_width - 10, y_pos + 18], 
                              fill=(245, 255, 245), outline=(150, 200, 150), width=1)
                draw.text((panel_x + 6, y_pos + 2), "✅ No significant findings detected", 
                         fill=(0, 120, 0), font=small_font)