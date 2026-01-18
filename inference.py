"""
NavAssist: Main Inference Script
Clean, efficient object detection with visualization
"""

import logging
from pathlib import Path
from typing import Tuple, List, Dict
import cv2
import numpy as np
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """Efficient object detection wrapper"""
    
    def __init__(self, model_path: str = 'best.pt', conf_threshold: float = 0.5):
        """Initialize detector with specified model"""
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.class_names = self.model.names
            logger.info(f"✓ Model loaded: {model_path}")
            logger.info(f"  Classes: {len(self.class_names)}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, source, conf: float = None) -> List[Dict]:
        """Run inference on image/video/camera"""
        try:
            conf = conf or self.conf_threshold
            results = self.model(source, conf=conf, verbose=False)
            
            detections = []
            for result in results:
                for box in result.boxes:
                    detection = {
                        'class_id': int(box.cls[0]),
                        'class_name': self.class_names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    }
                    detections.append(detection)
            
            return detections
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []
    
    def detect_on_image(self, image_path: str) -> np.ndarray:
        """Detect objects on image and return annotated frame"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            results = self.model(image, verbose=False)
            annotated = results[0].plot()
            
            logger.info(f"✓ Detections found on {Path(image_path).name}")
            return annotated
        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            return None
    
    def detect_on_video(self, video_path: str, output_path: str = None) -> bool:
        """Detect objects on video and save annotated output"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup writer if output path provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame, verbose=False)
                annotated = results[0].plot()
                
                # Write output
                if writer:
                    writer.write(annotated)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"  Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            if writer:
                writer.release()
            
            logger.info(f"✓ Video processing complete: {frame_count} frames")
            return True
        
        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            return False
    
    def detect_on_camera(self, duration: int = 60) -> None:
        """Real-time detection on webcam"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Camera not available")
            
            logger.info(f"Starting camera detection ({duration}s)...")
            
            start_time = cv2.getTickCount()
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame, verbose=False)
                annotated = results[0].plot()
                
                # Calculate FPS
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Display FPS
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('NavAssist - Real-time Detection', annotated)
                frame_count += 1
                
                # Check duration and exit key
                if elapsed >= duration or cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"✓ Camera session complete: {frame_count} frames, {fps:.1f} FPS avg")
        
        except Exception as e:
            logger.error(f"Camera detection failed: {e}")


def main():
    """Example usage"""
    detector = ObjectDetector(model_path='best.pt', conf_threshold=0.5)
    
    # Example: Image detection
    # annotated_image = detector.detect_on_image('image.jpg')
    # if annotated_image is not None:
    #     cv2.imwrite('output.jpg', annotated_image)
    
    # Example: Camera detection
    detector.detect_on_camera(duration=30)


if __name__ == '__main__':
    main()
