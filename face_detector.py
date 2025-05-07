import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Dict, Optional
from collections import deque

# Import config
from config import FACE_DETECTION_THRESHOLD, MODEL_DIR, logger
from utils.helpers import norm_crop_image, draw_bbox_info


class FaceDetector:
    """Face detection and processing module using SCRFD model."""
    
    def __init__(
        self, 
        preloaded_model=None,        
        detection_threshold=FACE_DETECTION_THRESHOLD,
        frame_skip=2,
        verbose=True
    ):
        """Initialize the FaceDetector."""
        self.model = preloaded_model
        self.detection_threshold = detection_threshold
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.model_loaded = preloaded_model is not None
        
        # Cache for last detection results
        self.last_results = {
            'boxes': [],
            'processed_faces': [],
            'landmarks': None,
        }
        
        if verbose:
            logger.info(f"FaceDetector initialized: threshold={detection_threshold}, frame_skip={frame_skip}")
    
    @staticmethod
    def load_models(model_path=None, return_models=True):
        """Load SCRFD detection model."""
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, "face_detection/det_34g.onnx")
            
        from static.models.face_detection.scrfd import SCRFD
        
        try:
            scrfd_model = SCRFD(
                model_path=model_path,
                input_size=(640, 640),
                conf_thres=FACE_DETECTION_THRESHOLD
            )
            logger.info(f"SCRFD model loaded: {model_path}")
            return scrfd_model if return_models else True
        except Exception as e:
            logger.error(f"Failed to load SCRFD model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _extract_faces_from_detections(self, frame, face_boxes, landmarks):
        """Extract aligned face images from detections."""
        processed_faces = []
        
        # Use landmarks for alignment when available
        if landmarks is not None and landmarks.shape[0] > 0:
            for i, landmark in enumerate(landmarks):
                face_landmarks = landmark.reshape(5, 2)
                face_img = norm_crop_image(
                    frame, 
                    face_landmarks, 
                    image_size=112,
                    mode='arcface'
                )
                
                if face_img is not None and face_img.size > 0:
                    processed_faces.append(face_img)
        
        # Fallback to simple crop if needed
        if not processed_faces and face_boxes:
            for box in face_boxes:
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    face_img = cv2.resize(face_img, (112, 112))
                    processed_faces.append(face_img)
                    
        return processed_faces
    
    def process_frame(self, frame, force_detect=False):
        """Process a frame for face detection and alignment."""
        # Check if model is loaded
        if not self.model_loaded:
            logger.error("Face detection model not loaded")
            return [], [], None
        
        self.frame_count += 1
        
        # Skip processing if not needed
        should_process = force_detect or (self.frame_count % self.frame_skip == 0)
        if not should_process:
            return (
                self.last_results['boxes'],
                self.last_results['processed_faces'], 
                self.last_results['landmarks']
            )
        
        try:
            # Detect faces using SCRFD
            detections, landmarks = self.model.detect(frame)
            
            # Handle no detections case
            if len(detections) == 0:
                self.last_results = {
                    'boxes': [],
                    'processed_faces': [],
                    'landmarks': None
                }
                return [], [], None
            
            # Extract bounding boxes from detections
            face_boxes = [(int(x1), int(y1), int(x2), int(y2)) 
                         for x1, y1, x2, y2, _ in detections]
            
            # Process faces for recognition
            processed_faces = self._extract_faces_from_detections(frame, face_boxes, landmarks)
            
            # Update cache
            self.last_results = {
                'boxes': face_boxes,
                'processed_faces': processed_faces,
                'landmarks': landmarks
            }
            
            return face_boxes, processed_faces, landmarks
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return (
                self.last_results['boxes'],
                self.last_results['processed_faces'], 
                self.last_results['landmarks']
            )

    def draw_detection_results(self, image, face_boxes, names=None, similarities=None):
        """Draw face detection results on the image."""
        result = image.copy()
        
        for i, box in enumerate(face_boxes):
            # Default values
            name = "Face"
            color = (0, 255, 0)  # Green
            similarity = 1.0
            
            # Use provided name and similarity if available
            if names and i < len(names):
                name = names[i]
                if name == "Unknown":
                    color = (255, 0, 0)  # Blue in BGR
            
            if similarities and i < len(similarities):
                similarity = similarities[i]
            
            draw_bbox_info(result, bbox=box, similarity=similarity, name=name, color=color)
        
        return result
    
    def reset(self):
        """Reset the detector state."""
        self.frame_count = 0
        self.last_results = {
            'boxes': [],
            'processed_faces': [],
            'landmarks': None
        }


def test_face_detector():
    """Test the face detector module."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Face Detector')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution')
    parser.add_argument('--frame-skip', type=int, default=2, help='Process every nth frame')
    parser.add_argument('--fps', type=int, default=30, help='Target camera FPS')
    args = parser.parse_args()

    # Setup system
    scrfd_model = FaceDetector.load_models()
    face_detector = FaceDetector(preloaded_model=scrfd_model, frame_skip=args.frame_skip)
    
    # Setup camera
    width, height = map(int, args.resolution.split('x'))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}")
        return

    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # Get actual camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera: {actual_width}x{actual_height} at {actual_fps:.1f}fps")
    
    # Setup display and metrics
    window_name = "Face Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    force_detect_interval = max(1, int(actual_fps))
    
    # Performance tracking
    frame_count = 0
    fps = 0
    fps_avg_frame_count = 10
    start_time = time.time()
    
    try:
        while True:
            # Get and process frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break

            frame_count += 1
            force_detect = (frame_count % force_detect_interval == 0)
            
            # Detect faces
            detection_start = time.time()
            face_boxes, processed_faces, landmarks = face_detector.process_frame(
                frame, force_detect=force_detect
            )
            detection_time = time.time() - detection_start
            
            # Calculate FPS
            if frame_count % fps_avg_frame_count == 0:
                fps = fps_avg_frame_count / (time.time() - start_time)
                start_time = time.time()
            
            # Create display
            display = frame.copy()
            if face_boxes:
                display = face_detector.draw_detection_results(display, face_boxes)
            
            # Add performance stats
            display_stats(display, fps, actual_fps, detection_time, 
                         len(face_boxes), force_detect, frame_count)
            
            # Show detected faces
            if processed_faces:
                display = draw_extracted_faces(display, processed_faces, 
                                              actual_height, max_faces=4)
            
            # Show frame and handle key presses
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                face_detector.reset()
                logger.info("Detector reset")
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Final FPS: {fps:.2f}")


def display_stats(frame, fps, camera_fps, detection_time, face_count, force_detect, frame_count):
    """Display performance statistics on frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Camera: {camera_fps:.1f}fps", (10, 60), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Detection: {detection_time*1000:.1f}ms", (10, 90),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 120),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Force detect: {force_detect}", (10, 150),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 180),
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def draw_extracted_faces(frame, faces, frame_height, max_faces=4):
    """Draw extracted face thumbnails on the frame."""
    face_size = 112  # Standard size for ArcFace
    margin = 10
    y_pos = frame_height - face_size - margin
    
    for i, face in enumerate(faces):
        if i >= max_faces:
            break
            
        x_pos = margin + i * (face_size + margin)
        
        # Draw white border
        cv2.rectangle(frame, 
                     (x_pos-2, y_pos-2), 
                     (x_pos+face_size+2, y_pos+face_size+2), 
                     (255, 255, 255), 2)
        
        try:
            # Place face thumbnail
            resized_face = cv2.resize(face, (face_size, face_size))
            frame[y_pos:y_pos+face_size, x_pos:x_pos+face_size] = resized_face
        except Exception as e:
            logger.error(f"Error displaying face: {e}")
    
    return frame


if __name__ == "__main__":
    test_face_detector()