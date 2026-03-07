"""
VisionAI - Computer Vision Module
Furniture condition detection using YOLO
"""

from .utils import (
    load_params,
    get_class_names,
    FURNITURE_CLASSES,
    CONDITION_COLORS
)

__version__ = "1.0.0"
__all__ = [
    "load_params",
    "get_class_names", 
    "FURNITURE_CLASSES",
    "CONDITION_COLORS"
]

