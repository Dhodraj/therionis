"""
Proper GradCAM implementation for PyTorch models
Based on research from PyTorch documentation and best practices
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


class GradCAM:
    """
    Implementation of Grad-CAM (Gradient-weighted Class Activation Mapping)
    for PyTorch models, specifically designed for medical imaging with torchxrayvision.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer_name: str = None):
        """
        Initialize GradCAM
        
        Args:
            model: PyTorch model (should be in eval mode)
            target_layer_name: Name of target layer for CAM extraction
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.target_layer_name = target_layer_name
        
        # Storage for gradients and feature maps
        self.gradients = None
        self.feature_maps = None
        self.hooks = []
        
        # Auto-detect target layer if not specified
        if target_layer_name is None:
            self.target_layer_name = self._find_target_layer()
        
        print(f"GradCAM initialized with target layer: {self.target_layer_name}")
    
    def _find_target_layer(self) -> str:
        """
        Automatically find the last convolutional layer for torchxrayvision DenseNet models
        """
        # For torchxrayvision DenseNet models, we want the last convolutional layer
        # before the global average pooling and classifier
        
        # Check if this is a torchxrayvision DenseNet model
        if hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            # This is likely a DenseNet architecture
            # The last conv layer is typically in features[-2] or features[-3]
            features = self.model.features
            
            # Look for the last layer that has conv2d
            for i in range(len(features) - 1, -1, -1):
                layer = features[i]
                if hasattr(layer, 'conv2') or 'conv' in str(type(layer)).lower():
                    return f"features.{i}"
            
            # Fallback: use features[-2] (usually norm layer) or features[-3]
            return "features.-3"
        else:
            # Generic fallback for other architectures
            return "features.-1"
    
    def _register_hooks(self):
        """Register forward and backward hooks for gradient extraction"""
        # Clear any existing hooks
        self._remove_hooks()
        
        # Find the target layer
        target_layer = self._get_layer_by_name(self.target_layer_name)
        
        if target_layer is None:
            raise ValueError(f"Could not find layer: {self.target_layer_name}")
        
        # Register forward hook to capture feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()
        
        # Register backward hook to capture gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        self.hooks = [forward_handle, backward_handle]
        print(f"Registered hooks on layer: {self.target_layer_name}")
    
    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _get_layer_by_name(self, name: str):
        """Get layer by name (e.g., 'features.11' or 'features.-1')"""
        parts = name.split('.')
        layer = self.model
        
        for part in parts:
            if part.startswith('-'):
                # Handle negative indexing
                index = int(part)
                layer = layer[index]
            elif part.isdigit():
                index = int(part)
                layer = layer[index]
            else:
                layer = getattr(layer, part)
        
        return layer
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        target_class: Optional[int] = None,
        retain_graph: bool = False
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W) 
            target_class: Target class index (if None, uses predicted class)
            retain_graph: Whether to retain the computation graph
            
        Returns:
            GradCAM heatmap as numpy array (H, W)
        """
        # Ensure input requires gradients
        input_tensor = input_tensor.clone().detach().to(self.device)
        input_tensor.requires_grad_(True)
        
        # Register hooks
        self._register_hooks()
        
        try:
            # Forward pass
            self.model.eval()
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            
            # Compute gradients
            self.model.zero_grad()
            
            # Backward pass for the target class
            class_score = output[0, target_class]
            class_score.backward(retain_graph=retain_graph)
            
            # Generate CAM
            if self.gradients is None or self.feature_maps is None:
                raise RuntimeError("Failed to capture gradients or feature maps")
            
            # Compute channel-wise weights by global average pooling of gradients
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            
            # Weighted combination of feature maps
            cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)  # (1, 1, H, W)
            
            # Apply ReLU to get positive values only
            cam = F.relu(cam)
            
            # Convert to numpy and normalize
            cam_np = cam.squeeze().cpu().numpy()
            
            # Normalize to 0-1 range
            if cam_np.max() > 0:
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
            
            return cam_np
            
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            # Return fallback empty heatmap
            return np.zeros((224, 224))
            
        finally:
            # Clean up
            self._remove_hooks()
            self.gradients = None
            self.feature_maps = None
    
    def generate_cam_for_pathology(
        self, 
        input_tensor: torch.Tensor, 
        pathology_name: str,
        pathologies_list: list
    ) -> np.ndarray:
        """
        Generate GradCAM for a specific pathology
        
        Args:
            input_tensor: Input image tensor
            pathology_name: Name of the pathology
            pathologies_list: List of pathology names from the model
            
        Returns:
            GradCAM heatmap as numpy array
        """
        try:
            # Find pathology index
            if pathology_name in pathologies_list:
                target_class = pathologies_list.index(pathology_name)
                return self.generate_cam(input_tensor, target_class)
            else:
                print(f"Pathology '{pathology_name}' not found in model pathologies")
                return np.zeros((224, 224))
        except Exception as e:
            print(f"Error generating GradCAM for pathology {pathology_name}: {e}")
            return np.zeros((224, 224))


class TorchXRayVisionGradCAM(GradCAM):
    """
    Specialized GradCAM implementation for torchxrayvision DenseNet models
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize GradCAM specifically for torchxrayvision models
        """
        # For torchxrayvision DenseNet, the target layer is typically 
        # the last convolutional layer before global average pooling
        super().__init__(model, target_layer_name="features.-1")
    
    def _find_target_layer(self) -> str:
        """
        Find the optimal target layer for torchxrayvision DenseNet models
        """
        # For DenseNet in torchxrayvision:
        # - features[0-11]: Various DenseBlocks and transitions  
        # - features[-1]: Usually ReLU or BatchNorm
        # - features[-2]: Usually the last actual conv layer or dense block
        
        # Try features[-2] first (usually the best for DenseNet)
        try:
            test_layer = self._get_layer_by_name("features.-2")
            if test_layer is not None:
                return "features.-2"
        except:
            pass
        
        # Fallback to features[-1]
        return "features.-1"