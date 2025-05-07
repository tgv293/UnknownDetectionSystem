import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Union
from collections import deque
import os

# Set environment variables before imports
from config import logger, MODEL_DIR
os.environ["DEEPFACE_HOME"] = MODEL_DIR


class AntiSpoofing:
    """Module to detect fake/real faces using anti-spoofing techniques."""
    
    def __init__(
        self, 
        model=None,
        spoof_threshold=0.5,
        cache_size=50, 
        cache_ttl=1.0,
        verbose=True
    ):
        """Initialize the anti-spoofing detector.
        
        Args:
            model: Pre-loaded anti-spoofing model (if None, uses DeepFace)
            spoof_threshold: Threshold for classifying real vs fake faces
            cache_size: Maximum number of results to cache
            cache_ttl: Time-to-live for cached results (seconds)
            verbose: Whether to print detailed logs
        """
        self.model = model
        self.model_loaded = model is not None
        self.spoof_threshold = spoof_threshold
        
        # Caching system
        self.track_cache = {}         # track_id -> result
        self.image_cache = {}         # image_hash -> result
        self.cache_timestamps = {}    # key -> timestamp
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        
        # Performance tracking
        self.process_times = deque(maxlen=30)
        
        if verbose:
            logger.info(f"AntiSpoofing initialized with threshold={spoof_threshold}")
    
    def _generate_face_hash(self, face: np.ndarray) -> str:
        """Generate a simple hash for a face image."""
        if face is None or face.size == 0:
            return "none"
            
        try:
            # Efficient downsampling and hash generation
            resized = cv2.resize(face, (32, 32))
            means = np.mean(resized, axis=(0, 1))
            return f"{means[0]:.1f}_{means[1]:.1f}_{means[2]:.1f}"
        except Exception as e:
            logger.error(f"Error generating face hash: {e}")
            return f"error_{time.time():.2f}"
    
    def _clean_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        
        # Find and remove expired entries in one pass
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if current_time - timestamp > self.cache_ttl
        ]
        
        for key in expired_keys:
            track_key = key.startswith("track_")
            img_key = key.startswith("img_")
            
            if track_key:
                track_id = int(key[6:])
                self.track_cache.pop(track_id, None)
            elif img_key:
                img_hash = key[4:]
                self.image_cache.pop(img_hash, None)
                
            self.cache_timestamps.pop(key, None)
        
        # If still too many entries, remove oldest ones
        total_entries = len(self.track_cache) + len(self.image_cache)
        if total_entries > self.cache_size:
            sorted_timestamps = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            to_remove = min(total_entries - self.cache_size, len(sorted_timestamps))
            
            for i in range(to_remove):
                key = sorted_timestamps[i][0]
                
                if key.startswith("track_"):
                    track_id = int(key[6:])
                    self.track_cache.pop(track_id, None)
                elif key.startswith("img_"):
                    img_hash = key[4:]
                    self.image_cache.pop(img_hash, None)
                    
                self.cache_timestamps.pop(key, None)
    
    def check_face(self, face: np.ndarray, track_id: int = -1, force_refresh: bool = False) -> Dict:
        """Check if a face is real or fake.
        
        Args:
            face: Face image (RGB/BGR format)
            track_id: Tracking ID for the face (-1 if not tracked)
            force_refresh: Force recalculation ignoring cache
            
        Returns:
            Dict with 'is_real' boolean and 'confidence' float
        """
        # Early return for invalid faces
        if face is None or face.size == 0:
            return {"is_real": False, "confidence": 0.0}
        
        # Check track_id cache first
        if not force_refresh and track_id >= 0 and track_id in self.track_cache:
            self.cache_timestamps[f"track_{track_id}"] = time.time()
            return self.track_cache[track_id]
            
        # Check image hash cache
        face_hash = self._generate_face_hash(face)
        if not force_refresh and face_hash in self.image_cache:
            self.cache_timestamps[f"img_{face_hash}"] = time.time()
            result = self.image_cache[face_hash]
            
            # Update track cache if valid track_id
            if track_id >= 0:
                self.track_cache[track_id] = result
                self.cache_timestamps[f"track_{track_id}"] = time.time()
                
            return result
            
        # Perform anti-spoofing check using DeepFace
        try:
            from deepface import DeepFace
            
            # Extract faces with anti-spoofing enabled
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
                is_real = True  # Default to real if detection fails
                score = 0.7
            
            # Create and cache result
            result = {"is_real": is_real, "confidence": score}
            
            # Store in appropriate caches
            if track_id >= 0:
                self.track_cache[track_id] = result
                self.cache_timestamps[f"track_{track_id}"] = time.time()
                
            self.image_cache[face_hash] = result
            self.cache_timestamps[f"img_{face_hash}"] = time.time()
            
            return result
                
        except Exception as e:
            logger.error(f"Error in anti-spoofing check: {e}")
            return {"is_real": True, "confidence": 0.5}  # Default to real on error
    
    def check_faces(self, faces: List[np.ndarray], track_ids: List[int] = None, 
                   force_refresh: bool = False) -> List[Dict]:
        """Check multiple faces for spoofing in batch.
        
        Args:
            faces: List of face images
            track_ids: Optional list of tracking IDs
            force_refresh: Force recalculation ignoring cache
            
        Returns:
            List of results dicts with 'is_real' and 'confidence' for each face
        """
        # Handle empty input
        if not faces:
            return []
            
        # Clean cache periodically
        self._clean_cache()
        
        # Use -1 as default track_id if not provided
        if track_ids is None or len(track_ids) != len(faces):
            track_ids = [-1] * len(faces)
        
        start_time = time.time()
        results = [self.check_face(face, track_id, force_refresh) 
                  for face, track_id in zip(faces, track_ids)]
        
        # Track processing time
        process_time = time.time() - start_time
        if faces:
            self.process_times.append(process_time / len(faces))
        
        return results
    
    def visualize_results(self, frame: np.ndarray, face_boxes: List[Tuple[int, int, int, int]], 
                         spoof_results: List[Dict]) -> np.ndarray:
        """Visualize anti-spoofing results on the frame.
        
        Args:
            frame: Original video frame
            face_boxes: List of face bounding boxes (x1, y1, x2, y2)
            spoof_results: Results from check_faces method
            
        Returns:
            Annotated frame with anti-spoofing results
        """
        from utils.helpers import draw_bbox_info
        
        result_frame = frame.copy()
        
        for box, result in zip(face_boxes, spoof_results):
            # Set color and label based on result
            if result["is_real"]:
                color = (0, 255, 0)  # Green for real
                label = "REAL"
            else:
                color = (0, 0, 255)  # Red for fake
                label = "FAKE"
            
            confidence = result["confidence"]
            draw_bbox_info(
                result_frame,
                bbox=box,
                similarity=confidence,
                name=label,
                color=color
            )
            
        return result_frame
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics about the anti-spoofing module."""
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
        logger.info("Anti-spoofing caches cleared")


def test_anti_spoofing():
    """Test the anti-spoofing module with live camera feed."""
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
    
    # Initialize components
    print("Loading face detection models...")
    scrfd_model = FaceDetector.load_models()
    face_detector = FaceDetector(preloaded_model=scrfd_model, frame_skip=args.frame_skip)
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
    
    # Create window and calculate parameters
    window_name = "Anti-Spoofing Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    force_detect_interval = max(1, int(cap.get(cv2.CAP_PROP_FPS) / 2))
    
    # FPS tracking
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
            face_boxes, processed_faces, _ = face_detector.process_frame(
                frame, force_detect=force_detect
            )
            
            # Check for spoofing
            spoof_results = []
            if processed_faces:
                spoof_results = anti_spoof.check_faces(
                    processed_faces, force_refresh=force_detect
                )
            
            # Calculate FPS
            if frame_count % fps_avg_frame_count == 0:
                fps = fps_avg_frame_count / (time.time() - start_time)
                start_time = time.time()
            
            # Create visualization
            if face_boxes and spoof_results:
                display = anti_spoof.visualize_results(frame, face_boxes, spoof_results)
            else:
                display = frame.copy()
            
            # Add performance information
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            stats = anti_spoof.get_performance_stats()
            cv2.putText(
                display, 
                f"Processing time: {stats['avg_process_time_ms']:.1f}ms", 
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            cv2.putText(
                display, 
                f"Faces: {len(face_boxes)}", 
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            # Show frame
            cv2.imshow(window_name, display)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                face_detector.reset()
                anti_spoof.reset_cache()
                logger.info("All caches reset")
            
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_anti_spoofing()