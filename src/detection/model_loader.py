"""
Singleton YOLO Model Loader
Ensures model is loaded ONCE and reused forever.
Never loads at startup - only on first inference.
"""

from pathlib import Path
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_model = None
_device = None
_detector_system = None  # FireDetectionSystem wrapper if needed


def _resolve_model_path() -> Path:
    """Resolve model path robustly relative to project root."""
    BASE_DIR = Path(__file__).resolve().parents[2]
    model_path = BASE_DIR / "models" / "yolo" / "weights" / "best.pt"
    logger.info(f"Resolved model path: {model_path}")
    return model_path


def get_model():
    """
    Get YOLO model (singleton pattern).
    Loads ONCE on first call, then reuses forever.
    """
    global _model

    if _model is not None:
        return _model

    logger.info("🔥 Loading YOLO model ONCE (this may take 10-30 seconds)...")
    try:
        from ultralytics import YOLO

        device = "cpu"

        model_path = _resolve_model_path()
        if not model_path.exists():
            logger.error(f"Model file does not exist: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        _model = YOLO(str(model_path))
        try:
            _model.to(device)
        except Exception:
            # Not critical; some YOLO wrappers ignore .to when CPU only
            pass

        logger.info("✅ YOLO model loaded successfully (singleton)")
        return _model

    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise


def get_device():
    """Return device string (cpu)."""
    global _device
    if _device is not None:
        return _device
    _device = "cpu"
    logger.info(f"Using device: {_device}")
    return _device


def get_detector_system():
    """
    Return FireDetectionSystem wrapper (singleton) for notification features.
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
        logger.error(f"Failed to create FireDetectionSystem: {e}")
        return None

