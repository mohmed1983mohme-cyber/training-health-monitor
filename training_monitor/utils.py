"""
Utility functions for Training Health Monitor
"""

import numpy as np


def detect_framework(model):
    """Auto-detect deep learning framework"""
    model_type = str(type(model))
    
    if 'torch' in model_type:
        return 'pytorch'
    elif 'tensorflow' in model_type or 'keras' in model_type:
        return 'tensorflow'
    else:
        return 'unknown'


def extract_gradients_pytorch(model):
    """Extract gradients from PyTorch model"""
    try:
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.cpu().detach().numpy().flatten())
        return np.concatenate(gradients) if gradients else None
    except Exception as e:
        print(f"Error extracting gradients: {e}")
        return None


def extract_gradients_tensorflow(model):
    """Extract gradients from TensorFlow model"""
    try:
        gradients = []
        for var in model.trainable_variables:
            if hasattr(var, 'gradient') and var.gradient is not None:
                gradients.append(var.gradient.numpy().flatten())
        return np.concatenate(gradients) if gradients else None
    except Exception as e:
        print(f"Error extracting gradients: {e}")
        return None
