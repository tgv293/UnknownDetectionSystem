from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import cv2
from bytetracker import BYTETracker


class FaceTracker:
    """Tracks faces across video frames using ByteTrack algorithm."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize face tracker with specified configuration.
        
        Args:
            config: Configuration dictionary with tracking parameters
        """
        # Default configuration with clear parameter naming
        self.config = {
            "track_thresh": 0.5,       # Confidence threshold for tracking
            "track_buffer": 30,        # Frames to keep track after disappearing
            "match_thresh": 0.8,       # IoU threshold for matching
            "frame_rate": 30,          # Video frame rate
            "min_box_area": 10,        # Minimum bbox area to track
            "aspect_ratio_thresh": 1.6 # Maximum width/height ratio
        }
        
        # Update with user provided config
        if config:
            self.config.update(config)
        
        # Initialize ByteTracker with configuration
        self.tracker = BYTETracker(
            track_thresh=self.config["track_thresh"],
            track_buffer=self.config["track_buffer"],
            match_thresh=self.config["match_thresh"],
            frame_rate=self.config["frame_rate"]
        )
        
        # State tracking
        self.frame_id = 0

    def update(self, faces_info: List[Dict], frame_size: Tuple[int, int]) -> List[Dict]:
        """Update tracker with new face detections.
        
        Args:
            faces_info: List of face information dictionaries from detector
            frame_size: Tuple of (height, width) of the frame
            
        Returns:
            List of tracked faces with tracking IDs and metadata
        """
        self.frame_id += 1
        height, width = frame_size
        
        # Convert face detections to ByteTracker format
        detections = self._convert_faces_to_detections(faces_info)
        
        # Update tracker with new detections
        online_targets = self.tracker.update(detections, [height, width])
        
        # Process tracking results
        tracked_faces = []
        for target in online_targets:
            # Filter out invalid bounding boxes
            tlwh = self._tlbr_to_tlwh(target[:4])
            track_id = int(target[4])
            score = float(target[6])
            
            # Skip if box has unreasonable aspect ratio or too small
            if tlwh[2] * tlwh[3] <= self.config["min_box_area"]:
                continue
                
            if tlwh[2] / tlwh[3] > self.config["aspect_ratio_thresh"]:
                continue
            
            # Create tracked face info
            tracked_face = {
                'bbox': [int(target[0]), int(target[1]), int(target[2]), int(target[3])],
                'track_id': track_id,
                'score': score
            }
            
            # Find original detection that matches this track
            self._add_detection_metadata(tracked_face, faces_info)
            tracked_faces.append(tracked_face)
                
        return tracked_faces
    
    def _convert_faces_to_detections(self, faces_info: List[Dict]) -> np.ndarray:
        """Convert face info dictionaries to detection array for ByteTracker.
        
        Args:
            faces_info: List of face information dictionaries
            
        Returns:
            NumPy array with format [x1, y1, x2, y2, score, class_id]
        """
        if not faces_info:
            return np.zeros((0, 6))
            
        # Extract bbox and confidence for each face
        detections = [
            [face['bbox'][0], face['bbox'][1], face['bbox'][2], face['bbox'][3], 
             face.get('confidence', 1.0), 0]  # Class 0 for all faces
            for face in faces_info
        ]
        
        return np.array(detections)
    
    def _add_detection_metadata(self, tracked_face: Dict, faces_info: List[Dict]) -> None:
        """Add original detection metadata to tracked face.
        
        Args:
            tracked_face: Face tracking result to update
            faces_info: Original face detections
        """
        # Find best matching detection
        best_iou = 0.7  # Minimum IoU threshold
        best_face = None
        
        for face in faces_info:
            iou = self._calculate_iou(tracked_face['bbox'], face['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_face = face
        
        # Add metadata from detection
        if best_face:
            for key in ['landmarks', 'landmarks_dict', 'face_image']:
                if key in best_face:
                    tracked_face[key] = best_face[key]

    def _tlbr_to_tlwh(self, tlbr: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, width, height] format.
        
        Args:
            tlbr: Array with [x1, y1, x2, y2] coordinates
            
        Returns:
            Array with [x, y, width, height] coordinates
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
        
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: First box [x1, y1, x2, y2]
            box2: Second box [x1, y1, x2, y2]
            
        Returns:
            IoU value between 0 and 1
        """
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate areas
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Handle division by zero
        return intersection / union if union > 0 else 0.0
        
    def draw_tracks(self, frame: np.ndarray, tracked_faces: List[Dict]) -> np.ndarray:
        """Draw tracked faces with IDs and landmarks on the frame.
        
        Args:
            frame: Source video frame
            tracked_faces: List of tracked faces from update method
            
        Returns:
            Frame with visualization of tracking results
        """
        img = frame.copy()
        
        for face in tracked_faces:
            bbox = face['bbox']
            track_id = face['track_id']
            
            # Generate unique color based on track ID
            color = self._get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw track ID label
            text = f"ID: {track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
            
            # Background for text
            cv2.rectangle(img, 
                         (bbox[0], bbox[1] - text_size[1] - 10), 
                         (bbox[0] + text_size[0], bbox[1]), 
                         color, -1)
                         
            # Text
            cv2.putText(img, text, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            
            # Draw landmarks if available
            if 'landmarks' in face and face['landmarks'] is not None:
                for landmark_point in face['landmarks']:
                    cv2.circle(img, 
                              (int(landmark_point[0]), int(landmark_point[1])), 
                              2, (0, 255, 0), -1)
                    
        return img
    
    def _get_color(self, idx: int) -> Tuple[int, int, int]:
        """Generate a unique color based on track ID.
        
        Args:
            idx: Track ID
            
        Returns:
            BGR color tuple
        """
        idx = abs(int(idx)) * 3
        return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)