from ultralytics import YOLO
import sys

try:
    print("Loading model...")
    model = YOLO("models/yolo/weights/best.pt")
    print("\n✅ Model Loaded Successfully")
    print("\n--- MODEL CLASS NAMES ---")
    print(model.names)
    print("-------------------------")
    
    # Check what Class 0 is explicitly
    class_0 = model.names.get(0, "Unknown")
    print(f"\nClass 0 is: '{class_0}'")
    
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
