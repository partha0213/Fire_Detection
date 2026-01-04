"""
Singleton YOLO Model Loader
Ensures model is loaded ONCE and reused forever.
Never loads at startup - only on first inference.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_device = None
_detector_system = None  # FireDetectionSystem wrapper if needed

def get_model():
    """
    Get YOLO model (singleton pattern).
    Loads ONCE on first call, then reuses forever.
    Safe for concurrent requests.
    """
    global _model
    
    if _model is not None:
        return _model  # Already loaded, return cached instance
    
    # Load model ONCE
    print("🔥 Loading YOLO model ONCE (this may take 10-30 seconds on first request)...")
    
    try:
        from ultralytics import YOLO
        import torch
        
        # Set device to CPU (Render has no GPU)
        device = "cpu"
        
        # Find model path
        model_path = os.path.join(
            os.path.dirname(__file__),
            "../..",  # Go up to project root
            "models/yolo/weights/best.pt"
        )
        
        # Resolve absolute path
        model_path = os.path.abspath(model_path)
        logger.info(f"Loading YOLO model from: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            # Try alternate path
            model_path = os.path.join(
                os.path.dirname(__file__),
                "models/yolo/weights/best.pt"
            )
            logger.info(f"Trying alternate path: {model_path}")
        
        # Load model
        _model = YOLO(model_path)
        _model.to(device)
        
        logger.info("✅ YOLO model loaded successfully (singleton - will be reused)")
        return _model
        
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def get_device():
    """Get torch device (CPU for Render)."""
    global _device
    
    if _device is not None:
        return _device
    
    try:
        import torch
        _device = "cpu"  # Render always CPU
        logger.info(f"Using device: {_device}")
        return _device
    except Exception as e:
        logger.error(f"Failed to initialize device: {e}")
        return "cpu"


def get_detector_system():
    """
    Get FireDetectionSystem wrapper (singleton).
    Wraps the YOLO model with notification system.
    Only create this if needed for notification features.
    """
    global _detector_system
    
    if _detector_system is not None:
        return _detector_system
    
    logger.info("Creating FireDetectionSystem wrapper...")
    
    try:
        from src.detection.fire_detector import FireDetectionSystem
        _detector_system = FireDetectionSystem()
        logger.info("✅ FireDetectionSystem wrapper created")
        return _detector_system
    except Exception as e:
        logger.error(f"❌ Failed to create FireDetectionSystem: {e}")
        return None

