"""
Fire Detection API - Railway-Compatible
FastAPI backend using YOLO-Only model for fire detection
- Fast startup: NO ML models loaded at import time
- Lazy loading: Models load only on first API/WebSocket call
- CPU-only: Defaults to CPU, no CUDA requirement
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import logging
import sys
from pathlib import Path
import json
from io import BytesIO
import base64
import asyncio

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Setup logging FIRST, before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# FAST STARTUP: Initialize FastAPI app immediately (NO heavy ML imports yet)
# ============================================================================
app = FastAPI(
    title="Fire Detection API",
    description="YOLO-Only Fire Detection System (Railway-Compatible)",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("✅ FastAPI app initialized (ready to bind to $PORT)")

# ============================================================================
# DATABASE INITIALIZATION (lightweight, happens at import time)
# ============================================================================
try:
    from src.utils.database import init_db, DetectionEvent, SystemConfig, SessionLocal
    logger.info("✅ Database module imported")
    SessionLocal = init_db()
    logger.info("✅ Database initialized")
except Exception as e:
    logger.error(f"⚠️  Database initialization failed: {e}")
    SessionLocal = None

# ============================================================================
# ============================================================================
# MODEL LOADING - Now handled by src/detection/model_loader.py
# Using singleton pattern - model loads ONCE on first use
# ============================================================================
from src.detection.model_loader import get_model, get_detector_system

# Simple state tracker for health checks
class SimpleState:
    model_loaded = False
    confidence_threshold = 0.5

state = SimpleState()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class DetectionResult(BaseModel):
    fire_detected: bool
    confidence: float
    fire_type: Optional[str] = None
    detections: List[Dict] = []
    timestamp: str
    model_type: str = "yolo"
    threshold_used: float = 0.5


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


# ============================================================================
# LIGHTWEIGHT HEALTH CHECK (fast startup, no ML)
# ============================================================================
@app.get("/health")
async def health_check() -> HealthResponse:
    """
    Lightweight health check - returns immediately.
    Does NOT load ML models.
    """
    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded,
        timestamp=datetime.now().isoformat()
    )


# ============================================================================
# DETECTION HELPER FUNCTIONS
# ============================================================================
def save_detection_to_db(
    fire_detected: bool,
    confidence: float,
    location: str = "Unknown",
    detections_list: List[Dict] = None
):
    """Save detection event to database with error handling."""
    if not fire_detected or not SessionLocal:
        return

    try:
        db = SessionLocal()
        event = DetectionEvent(
            timestamp=datetime.utcnow(),
            confidence=float(confidence),
            class_name="Fire",
            location=location,
            metadata_json=json.dumps(detections_list or [])
        )
        db.add(event)
        db.commit()
        db.close()
        logger.debug(f"💾 Saved detection to DB: confidence={confidence:.2%}, location={location}")
    except Exception as e:
        logger.error(f"❌ Failed to save to DB: {e}")


def process_inference(image_np, threshold: float = 0.5) -> DetectionResult:
    """
    Run fire detection inference on image.
    Uses singleton YOLO model from model_loader.
    """
    model = get_model()
    state.model_loaded = True  # Mark model as loaded after first successful call
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Import numpy and PIL only when needed
        import numpy as np
        
        # Convert RGB to BGR for OpenCV/YOLO
        image_bgr = image_np[:, :, ::-1]
        
        # Run YOLO prediction directly
        yolo_results = model.predict(image_bgr, conf=threshold, verbose=False)
        
        if not yolo_results or len(yolo_results) == 0:
            return DetectionResult(
                fire_detected=False,
                confidence=0.0,
                fire_type=None,
                detections=[],
                timestamp=datetime.now().isoformat(),
                model_type="yolo",
                threshold_used=threshold
            )
        
        result = yolo_results[0]
        fire_detected = result.boxes is not None and len(result.boxes) > 0
        confidence = 0.0
        yolo_boxes = result.boxes if fire_detected else []
        
        detections_list = []
        if yolo_boxes:
            h, w = image_np.shape[:2]
            for box in yolo_boxes:
                try:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    confidence = max(confidence, conf)
                    
                    # Normalize bbox for frontend
                    norm_bbox = [
                        float(xyxy[0]) / w,
                        float(xyxy[1]) / h,
                        float(xyxy[2] - xyxy[0]) / w,
                        float(xyxy[3] - xyxy[1]) / h
                    ]
                    
                    detections_list.append({
                        'class': 'fire',
                        'confidence': conf,
                        'bbox': norm_bbox,
                        'bbox_xyxy': [int(x) for x in xyxy]
                    })
                except Exception as e:
                    logger.error(f"❌ Error processing box: {e}")
                    continue
        
        # Save to database if fire detected
        if fire_detected:
            save_detection_to_db(fire_detected, confidence, "Upload", detections_list)
        
        return DetectionResult(
            fire_detected=fire_detected,
            confidence=float(confidence),
            fire_type="Fire" if fire_detected else None,
            detections=detections_list,
            timestamp=datetime.now().isoformat(),
            model_type="yolo",
            threshold_used=threshold
        )
        
    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


# ============================================================================
# REST API ENDPOINTS
# ============================================================================
@app.post("/detect", response_model=DetectionResult)
async def detect_fire(
    file: UploadFile = File(...),
    threshold: float = 0.5
):
    """
    Upload image for fire detection.
    Lazy-loads model on first call.
    """
    try:
        from PIL import Image
        import numpy as np
        
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        return process_inference(image_np, threshold)
        
    except Exception as e:
        logger.error(f"❌ Upload detection failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/detect/base64", response_model=DetectionResult)
async def detect_fire_base64(data: Dict):
    """Upload base64-encoded image for fire detection."""
    try:
        from PIL import Image
        import numpy as np
        
        image_data = data.get('image', '')
        threshold = float(data.get('threshold', 0.5))
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        return process_inference(image_np, threshold)
        
    except Exception as e:
        logger.error(f"❌ Base64 detection failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")


@app.post("/test/thresholds")
async def test_thresholds(file: UploadFile = File(...)):
    """Test detection with multiple confidence thresholds."""
    try:
        from PIL import Image
        import numpy as np
        
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')
        image_np = np.array(image)
        
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = {}
        
        for thresh in thresholds:
            res = process_inference(image_np, threshold=thresh)
            results[f"threshold_{thresh}"] = {
                "fire_detected": res.fire_detected,
                "confidence": res.confidence,
                "status": "🔥 FIRE" if res.fire_detected else "✅ SAFE"
            }
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"❌ Threshold test failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/alerts")
async def get_alerts():
    """Get detection history from database."""
    if not SessionLocal:
        return {"alerts": []}
    
    db = SessionLocal()
    try:
        events = db.query(DetectionEvent).order_by(
            DetectionEvent.timestamp.desc()
        ).limit(50).all()
        
        alerts = []
        for e in events:
            alerts.append({
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "confidence": e.confidence,
                "location": e.location,
                "status": "active"
            })
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch alerts: {e}")
        return {"alerts": []}
    finally:
        db.close()


@app.get("/detection-threshold")
async def get_detection_threshold():
    """Get current fire detection confidence threshold."""
    return {
        "threshold": state.confidence_threshold,
        "description": "Minimum confidence (0.0-1.0) for fire detection"
    }


@app.post("/detection-threshold")
async def set_detection_threshold(data: Dict):
    """Set fire detection confidence threshold."""
    try:
        threshold = float(data.get('threshold', 0.5))
        
        if threshold < 0.0 or threshold > 1.0:
            raise HTTPException(status_code=400, detail="Threshold must be 0.0-1.0")
        
        state.confidence_threshold = threshold
        logger.info(f"🔧 Detection threshold updated to {threshold}")
        
        return {
            "status": "success",
            "threshold": state.confidence_threshold
        }
        
    except Exception as e:
        logger.error(f"❌ Threshold update failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/notification-status")
async def get_notification_status():
    """Get notification system status."""
    detector = get_detector_system()
    
    if not detector or not hasattr(detector, 'notifier') or not detector.notifier:
        return {
            "status": "not_configured",
            "message": "Notification system not available"
        }
    
    try:
        status = detector.notifier.get_status()
        return {
            "status": "ok",
            "notifications": status
        }
    except Exception as e:
        logger.error(f"❌ Notification status failed: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/test-notification")
async def test_notification():
    """Send test notification."""
    detector = get_detector_system()
    
    if not detector or not hasattr(detector, 'notifier') or not detector.notifier:
        raise HTTPException(status_code=503, detail="Notification system not available")
    
    try:
        detector.notifier.send_fire_alert(
            confidence=0.95,
            location="TEST - Fire Detection System",
            image_path=None
        )
        
        status = detector.notifier.get_status()
        return {
            "status": "success",
            "message": "Test notification sent",
            "details": {
                "email_configured": status.get("email_configured", False),
                "sms_configured": status.get("sms_configured", False)
            }
        }
    except Exception as e:
        logger.error(f"❌ Test notification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WEBSOCKET ENDPOINTS - LAZY LOADING SAFE
# ============================================================================
@app.websocket("/ws/video")
async def client_stream_detection(websocket: WebSocket):
    """
    Client-side WebSocket (Laptop Camera Mode).
    Receives base64 frames → Runs inference → Returns detections.
    
    Model is lazy-loaded on first frame.
    """
    await websocket.accept()
    
    frame_count = 0
    is_connected = True
    detector = None
    
    logger.info("📡 Client WebSocket connected")
    
    try:
        while is_connected:
            try:
                # Receive base64 frame from client
                data = await websocket.receive_text()
                
                if not data:
                    continue
                
                # Lazy-load detector on first frame
                if detector is None:
                    logger.info("⏳ Loading model for client stream...")
                    detector = get_model()
                    if not detector:
                        await websocket.send_text(json.dumps({
                            "error": "Model failed to load"
                        }))
                        break
                
                try:
                    # Decode base64 frame
                    import cv2
                    import numpy as np
                    
                    if "base64," in data:
                        data = data.split("base64,")[1]
                    
                    img_bytes = base64.b64decode(data)
                    img_arr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        logger.debug("⚠️  Failed to decode frame")
                        continue
                    
                    # Run inference
                    fire_detected = False
                    confidence = 0.0
                    detection_boxes = []
                    
                    yolo_results = detector.predict(frame, conf=0.5, verbose=False)
                    
                    if yolo_results and len(yolo_results) > 0:
                        result = yolo_results[0]
                        if result.boxes is not None and len(result.boxes) > 0:
                            fire_detected = True
                            h, w = frame.shape[:2]
                            for box in result.boxes:
                                try:
                                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                                    conf_val = float(box.conf[0])
                                    confidence = max(confidence, conf_val)
                                    detection_boxes.append({
                                        'xyxy': [
                                            int(xyxy[0]), int(xyxy[1]),
                                            int(xyxy[2]), int(xyxy[3])
                                        ],
                                        'confidence': conf_val
                                    })
                                except Exception as e:
                                    logger.debug(f"Box extraction error: {e}")
                    
                    # Save to database (rate-limited)
                    if fire_detected and frame_count % 5 == 0:
                        save_detection_to_db(
                            fire_detected, confidence,
                            "Laptop Camera", detection_boxes
                        )
                    
                    # Send response
                    response = {
                        "fire_detected": fire_detected,
                        "confidence": float(confidence),
                        "fire_type": "Fire" if fire_detected else None,
                        "detections": detection_boxes,
                        "timestamp": datetime.now().isoformat(),
                        "frame_id": frame_count
                    }
                    
                    await websocket.send_text(json.dumps(response))
                    frame_count += 1
                    
                except json.JSONDecodeError:
                    logger.debug("⚠️  Invalid JSON in frame data")
                    continue
                except Exception as inference_err:
                    logger.error(f"❌ Inference error: {inference_err}")
                    continue
                    
            except WebSocketDisconnect:
                logger.info("📡 Client WebSocket disconnected")
                is_connected = False
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info("📡 Client WebSocket runtime disconnect")
                    is_connected = False
                    break
                logger.error(f"❌ Runtime error: {e}")
                is_connected = False
                break
            except Exception as e:
                logger.error(f"❌ Client frame error: {e}")
                if "disconnect" in str(e).lower():
                    is_connected = False
                    break
                
    except WebSocketDisconnect:
        logger.info("📡 Client WebSocket closed (outer)")
    except Exception as e:
        logger.error(f"❌ Client socket error: {e}")
    finally:
        logger.info("📡 Client WebSocket cleanup complete")


@app.websocket("/ws/stream/{source_type}")
async def server_stream_feed(websocket: WebSocket, source_type: str):
    """
    Server-side WebSocket (IP/USB/RTSP Camera Mode).
    Opens camera on server → Runs inference → Sends annotated frames.
    
    Model is lazy-loaded on first frame.
    """
    await websocket.accept()
    
    # Get camera source from query params
    url_param = websocket.query_params.get('url')
    device_param = websocket.query_params.get('device')
    
    logger.info(f"📡 Server stream connected: type={source_type}, url={url_param}")
    
    # Send handshake
    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "connected",
        "message": f"Connected to {source_type} stream"
    }))
    
    # Determine camera source
    camera_source = None
    if source_type == 'usb':
        camera_source = int(device_param) if device_param else 0
    elif source_type == 'ip':
        if url_param:
            camera_source = url_param
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "IP camera URL required"
            }))
            return
    elif source_type == 'rtsp':
        if url_param:
            camera_source = url_param
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "RTSP stream URL required"
            }))
            return
    else:
        camera_source = 0
    
    import cv2
    import numpy as np
    
    camera = None
    frame_count = 0
    detector = None
    
    try:
        # Open camera
        if camera_source is not None:
            camera = cv2.VideoCapture(camera_source)
            
            if isinstance(camera_source, str):
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not camera.isOpened():
                logger.error(f"❌ Could not open {source_type}: {camera_source}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": f"Failed to connect to {source_type} camera"
                }))
                # Keep connection open
                while True:
                    await asyncio.sleep(1)
                return
            
            logger.info(f"✅ {source_type.upper()} camera opened")
        
        # Stream loop
        while True:
            fire_detected = False
            confidence = 0.0
            detection_boxes = []
            frame = None
            
            if camera and camera.isOpened():
                success, frame = camera.read()
                if not success:
                    logger.warning(f"⚠️  Failed to read frame from {source_type}")
                    break
                
                # Lazy-load detector on first frame
                if detector is None:
                    logger.info("⏳ Loading model for server stream...")
                    detector = get_model()
                    if not detector:
                        logger.error("❌ Failed to load model")
                        break
                
                # Run inference
                yolo_results = detector.predict(frame, conf=0.5, verbose=False)
                
                if yolo_results and len(yolo_results) > 0:
                    result = yolo_results[0]
                    if result.boxes is not None and len(result.boxes) > 0:
                        fire_detected = True
                        for box in result.boxes:
                            try:
                                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                                conf = float(box.conf[0])
                                confidence = max(confidence, conf)
                                detection_boxes.append({
                                    'xyxy': [int(x) for x in xyxy],
                                    'confidence': conf
                                })
                            except Exception as e:
                                logger.error(f"Error processing box: {e}")
                
                # Draw annotations on frame
                color = (0, 0, 255) if fire_detected else (0, 255, 0)
                if detection_boxes:
                    for box_info in detection_boxes:
                        xyxy = box_info['xyxy']
                        conf_val = box_info['confidence']
                        cv2.rectangle(
                            frame,
                            (xyxy[0], xyxy[1]),
                            (xyxy[2], xyxy[3]),
                            color, 2
                        )
                        label = f"FIRE {conf_val:.0%}"
                        cv2.putText(
                            frame, label,
                            (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )
                
                # Status overlay
                status_text = "🔥 FIRE DETECTED" if fire_detected else "✅ Area Clear"
                status_color = (0, 0, 255) if fire_detected else (0, 255, 0)
                cv2.putText(
                    frame, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
                )
                cv2.putText(
                    frame, f"Confidence: {confidence:.1%}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                
                # Save to database (rate-limited)
                if fire_detected and frame_count % 150 == 0:
                    save_detection_to_db(
                        fire_detected, confidence,
                        f"{source_type.upper()} Stream",
                        detection_boxes
                    )
                
            else:
                # No camera: send placeholder
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame, f"NO SOURCE FOR {source_type.upper()}",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
                )
                cv2.putText(
                    frame, "Check camera configuration",
                    (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1
                )
                await asyncio.sleep(0.5)
            
            # Encode frame
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret or buffer is None:
                    logger.error("❌ Frame encoding failed")
                    continue
                
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{frame_b64}"
            else:
                continue
            
            # Send payload
            payload = {
                "type": "frame",
                "data": image_data,
                "timestamp": datetime.now().isoformat(),
                "frame_id": frame_count,
                "detection": {
                    "fire_detected": fire_detected,
                    "confidence": float(confidence),
                    "fire_type": "Fire" if fire_detected else None,
                    "boxes": detection_boxes
                }
            }
            
            try:
                await websocket.send_text(json.dumps(payload))
                frame_count += 1
            except WebSocketDisconnect:
                logger.info(f"📡 Stream {source_type} client disconnected")
                break
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info(f"📡 Stream {source_type} disconnected")
                    break
                logger.error(f"❌ Send error: {e}")
                break
            except Exception as e:
                logger.error(f"❌ Send error: {e}")
                break
            
            # Control frame rate (~30 FPS)
            await asyncio.sleep(0.033)
            
    except WebSocketDisconnect:
        logger.info(f"📡 Stream {source_type} disconnected")
    except Exception as e:
        logger.error(f"❌ Stream error: {e}")
    finally:
        if camera:
            camera.release()
        logger.info(f"📡 Stream {source_type} closed")


# ============================================================================
# NO app.run() or if __name__ == "__main__" block
# Railway will run: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
# ============================================================================

logger.info("🚀 Fire Detection API ready for Railway deployment")
