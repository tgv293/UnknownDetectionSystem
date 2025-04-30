import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union
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
        """Initialize the mask detector."""
        self.model = model
        self.model_loaded = model is not None
        self.threshold = threshold
        
        # Cache system - track_id based
        self.track_cache = {}  # track_id -> result
        self.image_cache = {}  # image_hash -> result (fallback)
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self.cache_timestamps = {}  # Key -> timestamp
        
        # Performance tracking
        self.process_times = deque(maxlen=30)
        
        if verbose:
            logger.info(f"MaskDetector initialized with threshold={threshold}")
    
    def _generate_face_hash(self, face: np.ndarray) -> str:
        """Generate a hash for a face image (fallback when no track_id)."""
        if face is None or face.size == 0:
            return "none"
            
        try:
            resized = cv2.resize(face, (32, 32))
            means = np.mean(resized, axis=(0, 1))
            return f"{means[0]:.2f}_{means[1]:.2f}_{means[2]:.2f}"
        except Exception as e:
            logger.error(f"Error generating face hash: {e}")
            return f"error_{time.time()}"
    
    def _clean_cache(self):
        """Remove expired entries from caches."""
        current_time = time.time()
        expired_keys = []
        
        # Find expired entries
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            if key in self.track_cache:
                del self.track_cache[key]
            if key in self.image_cache:
                del self.image_cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
        
        # If still too many entries, remove oldest ones
        total_entries = len(self.track_cache) + len(self.image_cache)
        if total_entries > self.cache_size:
            sorted_items = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            to_remove = total_entries - self.cache_size
            for i in range(min(to_remove, len(sorted_items))):
                key = sorted_items[i][0]
                if key in self.track_cache:
                    del self.track_cache[key]
                if key in self.image_cache:
                    del self.image_cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
    
    def detect_mask(self, face: np.ndarray, track_id: int = -1, force_refresh: bool = False) -> Dict:
        """Detect mask on a single face."""
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
            
            # If we have a valid track_id, update track cache too
            if track_id >= 0:
                self.track_cache[track_id] = result
                self.cache_timestamps[f"track_{track_id}"] = time.time()
                
            return result
            
        # Check if model is loaded
        if not self.model_loaded:
            logger.error("Mask detection model not loaded")
            return {"class": "unknown", "confidence": 0.0}
            
        try:
            # Process face for mask detection
            processed_face = preprocess_for_mask_detection(face)
            
            # Convert to batch format for inference
            input_data = np.expand_dims(processed_face, axis=0)
            
            # Run inference with ONNX model
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
            
            # Store in both caches if applicable
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
        """Detect masks on multiple faces."""
        if not faces:
            return []
            
        # Clean cache periodically
        self._clean_cache()
        
        results = []
        start_time = time.time()
        
        # Use track_ids if provided, otherwise use -1
        if track_ids is None or len(track_ids) != len(faces):
            track_ids = [-1] * len(faces)
        
        # Process each face individually
        for i, face in enumerate(faces):
            result = self.detect_mask(face, track_ids[i], force_refresh)
            results.append(result)
        
        # Track processing time
        process_time = time.time() - start_time
        if len(faces) > 0:
            self.process_times.append(process_time / len(faces))
        
        return results
    
    def visualize_results(
        self, 
        frame: np.ndarray, 
        face_boxes: List[Tuple[int, int, int, int]], 
        mask_results: List[Dict]
    ) -> np.ndarray:
        """Visualize mask detection results on the frame.
        
        Args:
            frame: Original frame
            face_boxes: List of face bounding boxes (x1, y1, x2, y2)
            mask_results: Results from batch_detect method
            
        Returns:
            Frame with annotations
        """
        from utils.helpers import draw_bbox_info
        
        result_frame = frame.copy()
        
        for i, (box, result) in enumerate(zip(face_boxes, mask_results)):
            mask_class = result.get("class", "unknown")
            confidence = result.get("confidence", 0.0)
            
            # Define colors for different mask states
            colors = {
                "with_mask": (0, 255, 0),       # Green - đeo đúng
                "incorrect_mask": (0, 165, 255), # Orange - đeo sai
                "without_mask": (0, 0, 255),    # Red - không đeo
                "unknown": (128, 128, 128),     # Gray - không biết
                "error": (255, 0, 255)          # Magenta - lỗi
            }
            color = colors.get(mask_class, colors["unknown"])
            
            # Format display name based on mask class
            display_names = {
                "with_mask": "Mask Worn Properly",
                "incorrect_mask": "Wear Mask Properly!",
                "without_mask": "No Mask! (please wear)",
                "unknown": "Unknown",
                "error": "Error"
            }
            display_name = display_names.get(mask_class, "Unknown")
            
            # Chỉ gọi draw_bbox_info (nó sẽ vẽ cả bbox)
            draw_bbox_info(
                result_frame,
                bbox=box,
                similarity=confidence,  # Sử dụng confidence làm similarity score
                name=display_name,
                color=color
            )
            
        return result_frame
    
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        avg_time = sum(self.process_times) / len(self.process_times) if self.process_times else 0
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
    
    # Import face detector
    from face_detector import FaceDetector
    
    # Set model path
    model_path = args.model
    if model_path is None:
        import os
        from config import MODEL_DIR
        model_path = os.path.join(MODEL_DIR, "mask_detector.onnx")
    
    # Load models (as would be done in recognize.py)
    print("Loading face detection models...")
    scrfd_model = FaceDetector.load_models()
    
    print(f"Loading mask detection model from {model_path}...")
    # Create ONNX session (would be done in recognize.py)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    mask_model = ort.InferenceSession(
        model_path, 
        sess_options=sess_options, 
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    
    # Initialize modules with pre-loaded models
    face_detector = FaceDetector(
        preloaded_model=scrfd_model,
        frame_skip=args.frame_skip
    )
    
    # Initialize mask detector with pre-loaded model
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
    cap.set(cv2.CAP_PROP_FPS, args.fps)  # Set target FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # Get actual camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Camera initialized: {actual_width}x{actual_height} at {actual_fps:.1f}fps")
    
    # Create window
    window_name = "Mask Detection Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Calculate reasonable force detection interval based on FPS
    force_detect_interval = max(1, int(actual_fps / 2))  # Check every half second
    
    # Cải tiến cách tính FPS
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
            
            # Determine when to force detection
            force_detect = (frame_count % force_detect_interval == 0)
            
            # Process frame để phát hiện khuôn mặt
            face_boxes, processed_faces, _ = face_detector.process_frame(
                frame, 
                force_detect=force_detect
            )
            
            # Check for masks if faces found
            mask_results = []
            if processed_faces:
                mask_results = mask_detector.batch_detect(
                    processed_faces, 
                    force_refresh=force_detect
                )
            
            # Visualize results
            if face_boxes and mask_results:
                display = mask_detector.visualize_results(frame, face_boxes, mask_results)
            else:
                display = frame.copy()
            
            # Tính FPS
            if frame_count % fps_avg_frame_count == 0:
                end_time = time.time()
                fps = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()
            
            # Add performance information
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Get mask detection stats
            stats = mask_detector.get_performance_stats()
            cv2.putText(
                display, 
                f"Mask detection time: {stats['avg_process_time_ms']:.1f}ms", 
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.putText(
                display, 
                f"Faces: {len(face_boxes)}", 
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                display, 
                f"Force detect: {force_detect}", 
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Show frame
            cv2.imshow(window_name, display)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset caches
                face_detector.reset()
                mask_detector.reset_cache()
                logger.info("All caches reset")
            
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Hiển thị thông tin FPS cuối cùng
        logger.info(f"Final FPS: {fps:.2f}")

if __name__ == "__main__":
    test_mask_detector()