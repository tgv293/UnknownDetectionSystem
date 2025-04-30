import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union
from collections import deque
import os

# Import config
from config import logger, MODEL_DIR
os.environ["DEEPFACE_HOME"] = MODEL_DIR

class AntiSpoofing:
    """Anti-spoofing detection module to detect fake/real faces."""
    
    def __init__(
        self, 
        model=None,
        spoof_threshold=0.5,
        cache_size=50, 
        cache_ttl=1.0,
        verbose=True
    ):
        """Initialize the anti-spoofing detector."""
        self.model = model
        self.model_loaded = model is not None
        self.spoof_threshold = spoof_threshold
        
        # Dual cache system - track_id based and image hash based
        self.track_cache = {}  # track_id -> result
        self.image_cache = {}  # image_hash -> result (fallback)
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self.cache_timestamps = {}
        
        # Performance tracking
        self.process_times = deque(maxlen=30)
        
        if verbose:
            logger.info(f"AntiSpoofing initialized with spoof_threshold={spoof_threshold}")
    
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
    
    def check_face(self, face: np.ndarray, track_id: int = -1, force_refresh: bool = False) -> Dict:
        """Check a single face for spoofing."""
        if face is None or face.size == 0:
            return {"is_real": False, "confidence": 0.0}
        
        # Check track_id cache first (if valid track_id provided)
        if not force_refresh and track_id >= 0 and track_id in self.track_cache:
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
            
        # Use DeepFace for anti-spoofing
        try:
            from deepface import DeepFace
            
            face_extraction = DeepFace.extract_faces(
                img_path=face,
                detector_backend="skip",
                enforce_detection=False,
                align=False,
                anti_spoofing=True
            )
            
            # Parse results
            if face_extraction and len(face_extraction) > 0:
                is_real = face_extraction[0].get("is_real", True)
                score = face_extraction[0].get("antispoof_score", 0.8)
            else:
                is_real = True
                score = 0.7
            
            # Create result
            result = {"is_real": is_real, "confidence": score}
            
            # Store in both caches if applicable
            if track_id >= 0:
                self.track_cache[track_id] = result
                self.cache_timestamps[f"track_{track_id}"] = time.time()
                
            self.image_cache[face_hash] = result
            self.cache_timestamps[f"img_{face_hash}"] = time.time()
            
            return result
                
        except Exception as e:
            logger.error(f"Error in anti-spoofing check: {e}")
            return {"is_real": True, "confidence": 0.5}  # Default to real on error
    
    def check_faces(self, faces: List[np.ndarray], track_ids: List[int] = None, force_refresh: bool = False) -> List[Dict]:
        """Check multiple face images for spoofing."""
        if not faces:
            return []
            
        # Clean cache periodically
        self._clean_cache()
        
        # Use track_ids if provided, otherwise use -1
        if track_ids is None or len(track_ids) != len(faces):
            track_ids = [-1] * len(faces)
        
        results = []
        start_time = time.time()
        
        for i, face in enumerate(faces):
            result = self.check_face(face, track_ids[i], force_refresh)
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
        spoof_results: List[Dict]
    ) -> np.ndarray:
        """Visualize anti-spoofing results on the frame.
        
        Args:
            frame: Original frame
            face_boxes: List of face bounding boxes (x1, y1, x2, y2)
            spoof_results: Results from check_faces method
            
        Returns:
            Frame with annotations
        """
        from utils.helpers import draw_bbox_info
        
        result_frame = frame.copy()
        
        for i, (box, result) in enumerate(zip(face_boxes, spoof_results)):
            # Set color and label based on result
            if result["is_real"]:
                color = (0, 255, 0)  # Green for real
                label = "REAL"
            else:
                color = (0, 0, 255)  # Red for fake
                label = "FAKE"
            
            # Get confidence score
            confidence = result["confidence"]
            
            # Use helper function to draw bounding box and info
            draw_bbox_info(
                result_frame,
                bbox=box,
                similarity=confidence,
                name=label,
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
        logger.info("Anti-spoofing caches cleared")

def test_anti_spoofing():
    """Test function for the anti-spoofing module."""
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Anti-Spoofing Module')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution')
    parser.add_argument('--frame-skip', type=int, default=2, help='Process every nth frame')
    parser.add_argument('--fps', type=int, default=30, help='Target camera FPS')
    parser.add_argument('--threshold', type=float, default=0.5, help='Spoofing detection threshold')
    args = parser.parse_args()
    
    # Import face detector
    from face_detector import FaceDetector
    
    # Load models (as would be done in recognize.py)
    print("Loading face detection models...")
    scrfd_model = FaceDetector.load_models()
    
    # Initialize modules with pre-loaded models
    face_detector = FaceDetector(
        preloaded_model=scrfd_model,
        frame_skip=args.frame_skip
    )
    
    # Initialize anti-spoofing with None model (DeepFace will be used directly)
    anti_spoof = AntiSpoofing(model=None, spoof_threshold=args.threshold)
    
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
    
    # Get actual camera settings (may differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Camera initialized: {actual_width}x{actual_height} at {actual_fps:.1f}fps")
    
    # Create window
    window_name = "Anti-Spoofing Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Calculate reasonable force detection interval based on camera FPS
    force_detect_interval = max(1, int(actual_fps / 2))  # Check every half second
    
    # Track FPS
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
            
            # Process frame to detect faces
            face_boxes, processed_faces, landmarks = face_detector.process_frame(
                frame, 
                force_detect=force_detect
            )
            
            # Check for spoofing if faces found
            spoof_results = []
            if processed_faces:
                spoof_results = anti_spoof.check_faces(
                    processed_faces, 
                    force_refresh=force_detect
                )
            
            # Calculate FPS
            if frame_count % fps_avg_frame_count == 0:
                end_time = time.time()
                fps = fps_avg_frame_count / (end_time - start_time)
                start_time = time.time()
            
            # Visualize results
            if face_boxes and spoof_results:
                display = anti_spoof.visualize_results(frame, face_boxes, spoof_results)
            else:
                display = frame.copy()
            
            # Add performance information
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Get anti-spoofing stats
            stats = anti_spoof.get_performance_stats()
            cv2.putText(
                display, 
                f"Anti-spoofing time: {stats['avg_process_time_ms']:.1f}ms", 
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
                anti_spoof.reset_cache()
                logger.info("All caches reset")
            
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Show final FPS
        logger.info(f"Final FPS: {fps:.2f}")


if __name__ == "__main__":
    test_anti_spoofing()