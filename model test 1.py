import os
import cv2
import time
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from ultralytics import YOLO

class FakeFaceDetector:
    def __init__(self, model_path=None):
        # Initialize model - try default YOLOv8n if custom model not found
        try:
            if model_path and os.path.exists(model_path):
                self.model = YOLO(model_path)
                print(f"Loaded custom model from {model_path}")
            else:
                self.model = YOLO("yolov8n.pt")  # Fallback to default model
                print("Using default YOLOv8n model")
        except Exception as e:
            print(f"Model loading error: {e}")
            raise

        self.class_names = ["fake", "real"]
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        # Detection parameters
        self.texture_threshold = 20
        self.color_variation_thresh = 25
        self.min_face_area = 15000

    def detect_fake_faces(self):
        prev_time = 0
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture frame")
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            try:
                results = self.model(img, stream=True, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        conf = box.conf[0]
                        if conf > 0.6:  # Confidence threshold
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            face_img = img[y1:y2, x1:x2]
                            
                            is_fake = self._verify_fake(face_img)
                            color = (0, 0, 255) if is_fake else (0, 255, 0)
                            label = "FAKE" if is_fake else "REAL"
                            
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"Detection error: {e}")

            # Show FPS
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow("Fake Face Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def _verify_fake(self, face_img):
        """Enhanced fake face verification"""
        if face_img.size == 0:
            return True
            
        h, w = face_img.shape[:2]
        if w * h < self.min_face_area:
            return True
            
        blur = cv2.Laplacian(face_img, cv2.CV_64F).var()
        if blur < 30:
            return True
            
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        if np.std(hsv[:,:,1]) < self.color_variation_thresh:
            return True
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < self.texture_threshold:
            return True
            
        return False

if __name__ == "__main__":
    # Try to find the model in common locations
    possible_paths = [
        "../models/l_version_1_300.pt",
        "models/l_version_1_300.pt",
        "l_version_1_300.pt",
        os.path.join(os.path.dirname(__file__), "models/l_version_1_300.pt")
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        print(f"Found model at: {model_path}")
    else:
        print("Custom model not found, using default YOLOv8n")
    
    detector = FakeFaceDetector(model_path)
    detector.detect_fake_faces()