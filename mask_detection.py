import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from collections import deque

# Import config and utilities
from config import MASK_DETECTION_THRESHOLD, logger
from preprocessing import preprocess_for_mask_detection

# Class labels (order matters)
MASK_CLASSES = ["incorrect_mask", "with_mask", "without_mask"]

class MaskDetector:
    """Mask detection module for face recognition pipeline."""
    
    def __init__(
        self,
        model=None,
        threshold=MASK_DETECTION_THRESHOLD,
        cache_size=50,
        cache_ttl=1.0,
        verbose=True
    ):
        """Initialize the mask detector with model and caching parameters."""
        self.model = model
        self.model_loaded = model is not None
        self.threshold = threshold
        
        # Cache system
        self.track_cache = {}          # track_id -> result
        self.image_cache = {}          # image_hash -> result (fallback)
        self.cache_timestamps = {}     # key -> timestamp
        self.cache_ttl = cache_ttl     # Time-to-live for cache entries
        self.cache_size = cache_size   # Maximum cache entries
        
        # Performance tracking
        self.process_times = deque(maxlen=30)
        
        if verbose:
            logger.info(f"MaskDetector initialized with threshold={threshold}")
    
    def _generate_face_hash(self, face: np.ndarray) -> str:
        """Generate a simple hash for a face image."""
        if face is None or face.size == 0:
            return "none"
            
        try:
            # Downsize for efficiency and extract color means
            resized = cv2.resize(face, (32, 32))
            means = np.mean(resized, axis=(0, 1))
            return f"{means[0]:.2f}_{means[1]:.2f}_{means[2]:.2f}"
        except Exception as e:
            logger.error(f"Error generating face hash: {e}")
            return f"error_{time.time()}"
    
    def _clean_cache(self):
        """Remove expired entries from caches efficiently."""
        current_time = time.time()
        
        # Find and remove expired entries
        expired_track_ids = [tid for tid, ts in self.cache_timestamps.items() 
                           if current_time - ts > self.cache_ttl and tid.startswith("track_")]
        
        expired_img_hashes = [img for img, ts in self.cache_timestamps.items() 
                            if current_time - ts > self.cache_ttl and img.startswith("img_")]
        
        # Remove from track cache
        for key in expired_track_ids:
            track_id = int(key[6:])  # Remove "track_" prefix
            self.track_cache.pop(track_id, None)
            self.cache_timestamps.pop(key, None)
            
        # Remove from image cache
        for key in expired_img_hashes:
            img_hash = key[4:]  # Remove "img_" prefix
            self.image_cache.pop(img_hash, None)
            self.cache_timestamps.pop(key, None)
        
        # If still too many entries, remove oldest ones
        total_entries = len(self.track_cache) + len(self.image_cache)
        if total_entries > self.cache_size:
            # Sort by timestamp (oldest first)
            oldest_entries = sorted(self.cache_timestamps.items(), key=lambda x: x[1])[:total_entries - self.cache_size]
            
            # Remove oldest entries
            for key, _ in oldest_entries:
                if key.startswith("track_"):
                    track_id = int(key[6:])
                    self.track_cache.pop(track_id, None)
                elif key.startswith("img_"):
                    img_hash = key[4:]
                    self.image_cache.pop(img_hash, None)
                self.cache_timestamps.pop(key, None)
    
    def detect_mask(self, face: np.ndarray, track_id: int = -1, force_refresh: bool = False) -> Dict:
        """Detect mask on a single face image."""
        # Handle invalid input
        if face is None or face.size == 0:
            return {"class": "unknown", "confidence": 0.0}
        
        # Check track_id cache first (if valid track_id provided)
        if not force_refresh and track_id >= 0 and track_id in self.track_cache:
            # Update timestamp
            self.cache_timestamps[f"track_{track_id}"] = time.time()
            return self.track_cache[track_id]
            
        # Fallback to image hash cache
        face_hash = self._generate_face_hash(face)
        if not force_refresh and face_hash in self.image_cache:
            self.cache_timestamps[f"img_{face_hash}"] = time.time()
            result = self.image_cache[face_hash]
            
            # Update track cache if valid track_id
            if track_id >= 0:
                self.track_cache[track_id] = result
                self.cache_timestamps[f"track_{track_id}"] = time.time()
                
            return result
            
        # Check model availability
        if not self.model_loaded:
            logger.error("Mask detection model not loaded")
            return {"class": "unknown", "confidence": 0.0}
            
        # Perform detection
        try:
            # Preprocess face for model input
            processed_face = preprocess_for_mask_detection(face)
            input_data = np.expand_dims(processed_face, axis=0)
            
            # Run inference
            outputs = self.model.run(None, {"input_1": input_data})
            scores = outputs[0][0]
            
            # Get prediction
            max_idx = np.argmax(scores)
            confidence = float(scores[max_idx])
            mask_class = MASK_CLASSES[max_idx]
            
            # Create result
            result = {
                "class": mask_class,
                "confidence": confidence
            }
            
            # Update caches
            if track_id >= 0:
                self.track_cache[track_id] = result
                self.cache_timestamps[f"track_{track_id}"] = time.time()
                
            self.image_cache[face_hash] = result
            self.cache_timestamps[f"img_{face_hash}"] = time.time()
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting mask: {e}")
            return {"class": "error", "confidence": 0.0}
    
    def batch_detect(self, faces: List[np.ndarray], track_ids: List[int] = None, force_refresh: bool = False) -> List[Dict]:
        """Detect masks on multiple faces efficiently."""
        if not faces:
            return []
            
        # Clean cache periodically
        self._clean_cache()
        
        # Use track_ids if provided, otherwise use -1
        if track_ids is None or len(track_ids) != len(faces):
            track_ids = [-1] * len(faces)
        
        # Measure processing time
        start_time = time.time()
        
        # Process each face
        results = [self.detect_mask(face, tid, force_refresh) 
                  for face, tid in zip(faces, track_ids)]
        
        # Track processing time
        process_time = time.time() - start_time
        if faces:
            self.process_times.append(process_time / len(faces))
        
        return results
    
    def visualize_results(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]], 
                         mask_results: List[Dict]) -> np.ndarray:
        """Visualize mask detection results on the frame."""
        from utils.helpers import draw_bbox_info
        
        result_frame = frame.copy()
        
        # Color and label definitions
        colors = {
            "with_mask": (0, 255, 0),      # Green - proper mask
            "incorrect_mask": (0, 165, 255), # Orange - incorrect mask
            "without_mask": (0, 0, 255),   # Red - no mask
            "unknown": (128, 128, 128),    # Gray - unknown
            "error": (255, 0, 255)         # Magenta - error
        }
        
        display_names = {
            "with_mask": "Mask Worn Properly",
            "incorrect_mask": "Wear Mask Properly!",
            "without_mask": "No Mask! (please wear)",
            "unknown": "Unknown",
            "error": "Error"
        }
        
        # Draw results for each face
        for box, result in zip(face_boxes, mask_results):
            mask_class = result.get("class", "unknown")
            confidence = result.get("confidence", 0.0)
            
            color = colors.get(mask_class, colors["unknown"])
            display_name = display_names.get(mask_class, "Unknown")
            
            # Draw bounding box and info
            draw_bbox_info(
                result_frame,
                bbox=box,
                similarity=confidence,
                name=display_name,
                color=color
            )
            
        return result_frame
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics about mask detection."""
        avg_time = np.mean(self.process_times) if self.process_times else 0
        
        return {
            "avg_process_time_ms": avg_time * 1000,
            "track_cache_size": len(self.track_cache),
            "image_cache_size": len(self.image_cache),
            "model_loaded": self.model_loaded
        }
    
    def reset_cache(self):
        """Clear all caches."""
        self.track_cache.clear()
        self.image_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Mask detection caches cleared")


def test_mask_detector():
    """Test function for the mask detection module."""
    import argparse
    import onnxruntime as ort
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Mask Detector Module')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution')
    parser.add_argument('--frame-skip', type=int, default=2, help='Frame skip for face detection')
    parser.add_argument('--fps', type=int, default=30, help='Target camera FPS')
    parser.add_argument('--threshold', type=float, default=MASK_DETECTION_THRESHOLD, help='Mask detection threshold')
    parser.add_argument('--model', type=str, default=None, help='Path to mask detector model')
    args = parser.parse_args()
    
    # Setup models and camera
    from face_detector import FaceDetector
    import os
    from config import MODEL_DIR
    
    # Load models
    model_path = args.model or os.path.join(MODEL_DIR, "mask_detection", "mask_detector.onnx")
    print(f"Loading models...")
    
    # Face detection model
    scrfd_model = FaceDetector.load_models()
    face_detector = FaceDetector(preloaded_model=scrfd_model, frame_skip=args.frame_skip)
    
    # Mask detection model
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    mask_model = ort.InferenceSession(
        model_path, 
        sess_options=sess_options, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    mask_detector = MaskDetector(model=mask_model, threshold=args.threshold)
    
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
    
    # Get actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera: {actual_width}x{actual_height} at {actual_fps:.1f}fps")
    
    # Setup display
    window_name = "Mask Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    force_detect_interval = max(1, int(actual_fps / 2))
    
    # Performance tracking
    frame_count = 0
    fps = 0
    fps_avg_frame_count = 10
    start_time = time.time()
    
    try:
        while True:
            # Get frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
                
            frame_count += 1
            force_detect = (frame_count % force_detect_interval == 0)
            
            # Process frame
            face_boxes, processed_faces, _ = face_detector.process_frame(
                frame, force_detect=force_detect
            )
            
            # Mask detection
            mask_results = []
            if processed_faces:
                mask_results = mask_detector.batch_detect(
                    processed_faces, force_refresh=force_detect
                )
            
            # Create visualization
            if face_boxes and mask_results:
                display = mask_detector.visualize_results(frame, face_boxes, mask_results)
            else:
                display = frame.copy()
            
            # Calculate FPS
            if frame_count % fps_avg_frame_count == 0:
                fps = fps_avg_frame_count / (time.time() - start_time)
                start_time = time.time()
            
            # Add performance stats to display
            stats = mask_detector.get_performance_stats()
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display, f"Mask detection: {stats['avg_process_time_ms']:.1f}ms", (10, 70),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Faces: {len(face_boxes)}", (10, 110),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow(window_name, display)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                face_detector.reset()
                mask_detector.reset_cache()
                logger.info("All caches reset")
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Final FPS: {fps:.2f}")


if __name__ == "__main__":
    test_mask_detector()