"""
Fire Detection API
FastAPI backend using YOLO-Only model for fire detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
from io import BytesIO
from PIL import Image
import asyncio
import base64
import torch
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.detection.fire_detector import FireDetectionSystem
from src.utils.database import init_db, DetectionEvent, SystemConfig, SessionLocal
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize DB
logger.info("Connecting to Database...")
SessionLocal = init_db()

# Initialize FastAPI app
app = FastAPI(
    title="Fire Detection API",
    description="YOLO-Only Fire Detection System",
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

# ==== Pydantic Models ====

class DetectionResult(BaseModel):
    fire_detected: bool
    confidence: float
    fire_type: Optional[str] = None
    fire_type_probs: Optional[Dict[str, float]] = None
    detections: List[Dict] = []
    timestamp: str
    model_type: str = "yolo"
    threshold_used: float = 0.5
    fusion_weights: Optional[Dict[str, float]] = None # Deprecated/Empty

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    device: str
    timestamp: str

# ==== State Management ====
class AppState:
    def __init__(self):
        self.detector = None
        self.is_loaded = False
        self.device = "cpu"
        self.confidence_threshold = 0.5  # Default threshold for fire detection (50% minimum)

state = AppState()

# ==== Startup Event ====
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Fire Detection API...")
    try:
        state.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {state.device}")
        
        # Initialize the refactored YOLO-only system
        # It handles its own config loading and model initialization
        state.detector = FireDetectionSystem()
        state.is_loaded = True
        logger.info("✅ YOLO-Only FireDetectionSystem loaded successfully!")
        
    except Exception as e:
        logger.error(f"❌ Startup Error: {e}")
        state.is_loaded = False

# ==== Helper Functions ====
def process_inference(image_np: np.ndarray, threshold: float = 0.5) -> DetectionResult:
    """Run inference using the loaded detector."""
    if not state.detector:
         raise HTTPException(status_code=503, detail="Model not initialized")
         
    # Run prediction
    # Run prediction
    # detector.predict() expects file path, numpy array, or tensor
    # IMPORTANT: Ultralytics assumes BGR for numpy arrays, but we have RGB from PIL.
    # We must convert to BGR.
    image_bgr = image_np[:, :, ::-1] # Flip channels RGB -> BGR
    results = state.detector.predict(image_bgr, conf=threshold)
    
    # Extract data from standardized result dict
    fire_detected = results['fire_detected']
    confidence = results['confidence']
    yolo_boxes = results.get('yolo_boxes')
    
    detections_list = []
    if yolo_boxes is not None:
         for box in yolo_boxes:
             # Basic info for frontend
             xyxy = box.xyxy[0].cpu().tolist()
             conf = float(box.conf[0])
             cls_id = int(box.cls[0])
             
             # Normalize bbox for frontend [x, y, w, h] normalized
             h, w, _ = image_np.shape
             norm_bbox = [
                 xyxy[0] / w,
                 xyxy[1] / h,
                 (xyxy[2] - xyxy[0]) / w,
                 (xyxy[3] - xyxy[1]) / h
             ]
             
             detections_list.append({
                 'class': 'fire', # Simplified
                 'confidence': conf,
                 'bbox': norm_bbox,
                 'bbox_xyxy': xyxy
             })
    
    
    # Save to Database if Fire Detected
    if fire_detected and SessionLocal:
        try:
            db = SessionLocal()
            event = DetectionEvent(
                timestamp=datetime.utcnow(),
                confidence=float(confidence),
                class_name="Fire",
                location="Camera 1", # Could come from config
                metadata_json=json.dumps(detections_list)
            )
            db.add(event)
            db.commit()
            db.close()
            # print("DEBUG: Saved detection to DB")
        except Exception as e:
            logger.error(f"Failed to save to DB: {e}")

    return DetectionResult(
        fire_detected=fire_detected,
        confidence=confidence,
        fire_type="Wildfire / Hazard" if fire_detected else None,
        # Legacy/Compatibility fields
        fire_type_probs={'Class A': 1.0} if fire_detected else None, 
        detections=detections_list,
        timestamp=datetime.now().isoformat(),
        model_type="yolo",
        threshold_used=threshold,
        fusion_weights=None
    )

# ==== API Endpoints ====

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if state.is_loaded else "error",
        model_loaded=state.is_loaded,
        model_type="yolo-only",
        device=state.device,
        timestamp=datetime.now().isoformat()
    )

@app.post("/detect", response_model=DetectionResult)
async def detect_fire(
    file: UploadFile = File(...),
    threshold: float = 0.5  # Confidence threshold for fire detection (50% minimum)
):
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    
    # Simulate processing time slightly if needed for UI smoothness, or remove
    # await asyncio.sleep(0.5) 
    
    return process_inference(image_np, threshold)

@app.post("/detect/base64", response_model=DetectionResult)
async def detect_fire_base64(data: Dict):
    if not state.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    image_data = data.get('image', '')
    threshold = float(data.get('threshold', 0.5))
    
    if ',' in image_data:
        image_data = image_data.split(',')[1]
        
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)
        
        return process_inference(image_np, threshold)
    except Exception as e:
        logger.error(f"Base64 error: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

@app.post("/test/thresholds")
async def test_thresholds(file: UploadFile = File(...)):
    """Test with multiple thresholds."""
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB')
    image_np = np.array(image)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    results = {}
    
    for thresh in thresholds:
        res = process_inference(image_np, threshold=thresh)
        results[f"threshold_{thresh}"] = {
            "fire_detected": res.fire_detected,
            "status": "🔥 FIRE" if res.fire_detected else "✅ SAFE"
        }
        
    return {"results": results}

@app.get("/alerts")
async def get_alerts():
    """Get detection history."""
    if not SessionLocal:
        return {"alerts": []}
    
    db = SessionLocal()
    try:
        events = db.query(DetectionEvent).order_by(DetectionEvent.timestamp.desc()).limit(50).all()
        alerts = []
        for e in events:
            alerts.append({
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "confidence": e.confidence,
                "location": e.location,
                "status": "active" # placeholder
            })
        return {"alerts": alerts}
    finally:
        db.close()

class ConfigUpdate(BaseModel):
    key: str
    value: str

@app.post("/config")
async def update_config(config: ConfigUpdate):
    """Update system configuration (e.g. camera IP)."""
    if not SessionLocal:
         raise HTTPException(status_code=503, detail="Database not available")
    
    db = SessionLocal()
    try:
        # Check if exists
        item = db.query(SystemConfig).filter(SystemConfig.key == config.key).first()
        if item:
            item.value = config.value
        else:
            item = SystemConfig(key=config.key, value=config.value)
            db.add(item)
        db.commit()
        return {"status": "success", "key": config.key, "value": config.value}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/config/{key}")
async def get_config_item(key: str):
    if not SessionLocal:
        return {"value": None}
    db = SessionLocal()
    try:
        item = db.query(SystemConfig).filter(SystemConfig.key == key).first()
        return {"value": item.value if item else None}
    finally:
        db.close()

@app.get("/detection-threshold")
async def get_detection_threshold():
    """Get current fire detection confidence threshold."""
    return {
        "threshold": state.confidence_threshold,
        "description": "Minimum confidence score (0.0-1.0) for fire detection. Higher = fewer false positives, lower = more sensitive"
    }

@app.post("/detection-threshold")
async def set_detection_threshold(data: Dict):
    """Set fire detection confidence threshold."""
    threshold = float(data.get('threshold', 0.5))
    
    # Validate range
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.0 and 1.0")
    
    state.confidence_threshold = threshold
    logger.info(f"Detection threshold updated to {threshold}")
    
    return {
        "status": "success",
        "threshold": state.confidence_threshold,
        "message": f"Fire detection threshold set to {threshold:.0%}"
    }

@app.get("/notification-status")
async def get_notification_status():
    """Get status of notification system (email and SMS)."""
    if not state.detector or not state.detector.notifier:
        return {
            "status": "not_initialized",
            "message": "Notification system not initialized"
        }
    
    status = state.detector.notifier.get_status()
    return {
        "status": "ok",
        "notifications": status,
        "message": "Notification system status retrieved"
    }

@app.post("/test-notification")
async def test_notification():
    """Send a test notification to configured email and SMS."""
    if not state.detector or not state.detector.notifier:
        raise HTTPException(status_code=503, detail="Notification system not initialized")
    
    try:
        # Send test email and SMS
        state.detector.notifier.send_fire_alert(
            confidence=0.95,
            location="TEST - Fire Detection System",
            image_path=None
        )
        
        status = state.detector.notifier.get_status()
        return {
            "status": "success",
            "message": "Test notification sent",
            "details": {
                "email_sent": status["email_configured"],
                "sms_sent": status["sms_configured"],
                "email_recipients": status["email_recipients"],
                "sms_recipients": status["sms_recipients"]
            }
        }
    except Exception as e:
        logger.error(f"Test notification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send test notification: {str(e)}")

@app.websocket("/ws/video")
async def client_stream_detection(websocket: WebSocket):
    """
    Client-side streaming endpoint (Laptop Mode).
    Receives base64 frames from client -> Runs Inference -> Returns detections.
    """
    await websocket.accept()
    import cv2
    
    frame_count = 0
    is_connected = True
    
    try:
        while is_connected:
            try:
                # Receive base64 image from client
                data = await websocket.receive_text()
                
                # Decode
                if "base64," in data:
                    data = data.split("base64,")[1]
                
                # Decode from base64
                img_bytes = base64.b64decode(data)
                img_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.debug("Failed to decode frame")
                    continue
                
                # Inference with fire detection
                fire_detected = False
                confidence = 0.0
                detection_boxes = []
                
                # Using 0.5 confidence threshold to reduce false positives from sunlight
                if state.detector:
                    try:
                        results = state.detector.predict(frame, conf=0.5)
                        fire_detected = results.get('fire_detected', False)
                        confidence = results.get('confidence', 0.0)
                        yolo_boxes = results.get('yolo_boxes', [])
                        
                        # Extract box coordinates for response
                        if yolo_boxes:
                            h, w = frame.shape[:2]
                            for box in yolo_boxes:
                                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                                conf_val = float(box.conf[0])
                                detection_boxes.append({
                                    'xyxy': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                                    'confidence': conf_val
                                })
                    except Exception as inference_err:
                        logger.error(f"Inference error: {inference_err}")
                        continue
                
                # Return complete detection results (Frontend draws boxes for Laptop mode)
                try:
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
                except Exception as send_err:
                    logger.error(f"Failed to send response: {send_err}")
                    is_connected = False
                    break
                    
            except WebSocketDisconnect:
                logger.info("Client detection disconnected (receive loop)")
                is_connected = False
                break
            except RuntimeError as rt_err:
                # Catch "Cannot call receive once a disconnect message has been received"
                if "disconnect" in str(rt_err).lower():
                    logger.info("WebSocket already disconnected")
                    is_connected = False
                    break
                else:
                    logger.error(f"Runtime error: {rt_err}")
                    is_connected = False
                    break
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                # Only continue if it's not a disconnect error
                if "disconnect" in str(e).lower():
                    is_connected = False
                    break
                # For other errors, skip frame and continue
                
    except WebSocketDisconnect:
        logger.info("Client detection disconnected (outer)")
    except Exception as e:
        logger.error(f"Client socket error: {e}")
    finally:
        logger.info("Client detection WebSocket closed")


@app.websocket("/ws/stream/{source_type}")
async def server_stream_feed(websocket: WebSocket, source_type: str):
    """
    Server-side streaming endpoint (IP/USB/RTSP Mode).
    Opens camera on server -> Runs Inference -> Sends annotated frames to client.
    Query Parameters:
    - url: Camera URL (for IP/RTSP sources)
    - device: Device index (for USB sources)
    """
    await websocket.accept()
    
    # Get query parameters
    url_param = websocket.query_params.get('url')
    device_param = websocket.query_params.get('device')
    
    logger.info(f"Stream connection: type={source_type}, url={url_param}, device={device_param}")
    
    # HANDSHAKE: Frontend needs this to know stream is ready and set isStreaming=true
    await websocket.send_text(json.dumps({
        "type": "status",
        "status": "connected",
        "message": f"Connected to {source_type} stream"
    }))
    
    # Determine Source
    camera_source = None
    
    if source_type == 'usb':
        # Use device parameter if provided, otherwise default to 0
        camera_source = int(device_param) if device_param else 0
        logger.info(f"Opening USB camera at index: {camera_source}")
    elif source_type == 'ip':
        # Use URL parameter for IP camera
        if url_param:
            camera_source = url_param
            logger.info(f"Opening IP camera at URL: {camera_source}")
        else:
            logger.error("IP camera URL not provided")
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "IP camera URL is required. Please provide a valid camera URL."
            }))
            return
    elif source_type == 'rtsp':
        # Use URL parameter for RTSP stream
        if url_param:
            camera_source = url_param
            logger.info(f"Opening RTSP stream at: {camera_source}")
        else:
            logger.error("RTSP stream URL not provided")
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": "RTSP stream URL is required. Please provide a valid stream URL."
            }))
            return
    else:
        # Fallback
        camera_source = 0

    import cv2
    camera = None
    frame_count = 0
    
    try:
        if camera_source is not None:
            camera = cv2.VideoCapture(camera_source)
            
            # Set timeout for connection attempts
            if isinstance(camera_source, str):
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            if not camera.isOpened():
                 logger.error(f"Could not open source: {source_type} - {camera_source}")
                 await websocket.send_text(json.dumps({
                     "type": "error",
                     "error": f"Failed to connect to {source_type} camera. Check URL and network connection.",
                     "details": str(camera_source)
                 }))
                 # Use dummy loop to keep connection open so UI doesn't spasm
                 while True:
                     await asyncio.sleep(1)
                 return
            
            logger.info(f"✅ Successfully opened {source_type} camera")
        
        while True:
            fire_detected = False
            confidence = 0.0
            detection_boxes = []
            
            if camera and camera.isOpened():
                success, frame = camera.read()
                if not success:
                    break
                
                # Run Inference - Fire detection on every frame
                # Using 0.5 confidence threshold to reduce false positives from sunlight
                if state.detector:
                    results = state.detector.predict(frame, conf=0.5)
                    fire_detected = results['fire_detected']
                    confidence = results.get('confidence', 0.0)
                    yolo_boxes = results.get('yolo_boxes', [])
                    
                    # Store box info for response
                    if yolo_boxes:
                        h, w = frame.shape[:2]
                        for box in yolo_boxes:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            conf_val = float(box.conf[0])
                            detection_boxes.append({
                                'xyxy': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                                'confidence': conf_val
                            })
                    
                    # Draw annotations (Server side drawing)
                    color = (0, 0, 255) if fire_detected else (0, 255, 0)
                    if yolo_boxes:
                        for box in yolo_boxes:
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            conf_val = float(box.conf[0])
                            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                            label = f"FIRE {conf_val:.0%}"
                            cv2.putText(frame, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add detection status in corner
                    status_text = "🔥 FIRE DETECTED" if fire_detected else "✅ Area Clear"
                    status_color = (0, 0, 255) if fire_detected else (0, 255, 0)
                    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Encode with optimized compression
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret or buffer is None:
                    logger.error("Failed to encode frame")
                    continue
                    
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{frame_b64}"
                
            else:
                # No camera source (e.g. unconfigured IP) - Send Placeholder
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"NO SOURCE FOR {source_type.upper()}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank, "Select a camera source from the UI", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                ret, buffer = cv2.imencode('.jpg', blank)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{frame_b64}"
                await asyncio.sleep(0.5) # Slow blink
            
            # Send to Client with complete detection data
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
                logger.info(f"Stream {source_type} client disconnected during send")
                break
            except RuntimeError as rt_err:
                if "disconnect" in str(rt_err).lower():
                    logger.info(f"Stream {source_type} disconnected (runtime error)")
                    break
                else:
                    logger.error(f"Failed to send frame: {rt_err}")
                    break
            except Exception as e:
                logger.error(f"Failed to send frame: {e}")
                break
            
            # Control frame rate - ~15-30 FPS
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except WebSocketDisconnect:
        logger.info(f"Stream {source_type} disconnected")
    except Exception as e:
        logger.error(f"Stream error: {e}")
    finally:
        if camera:
            camera.release()
        logger.info(f"Stream {source_type} closed")




if __name__ == "__main__":
    import uvicorn
    # run on port 8000 to match frontend expectation
    uvicorn.run(app, host="0.0.0.0", port=8000)
