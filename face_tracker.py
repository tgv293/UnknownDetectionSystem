import numpy as np
from bytetracker import BYTETracker
import cv2

class FaceTracker:
    """
    Class theo dõi khuôn mặt sử dụng thuật toán ByteTrack.
    """
    def __init__(self, config=None):
        """
        Khởi tạo face tracker với cấu hình được cung cấp
        
        Args:
            config: Dictionary cấu hình với các tham số tracking
        """
        if config is None:
            config = {}
            
        # Tham số mặc định cho ByteTracker
        self.track_thresh = config.get("track_thresh", 0.5)
        self.track_buffer = config.get("track_buffer", 30)
        self.match_thresh = config.get("match_thresh", 0.8)
        self.frame_rate = config.get("frame_rate", 30)
        self.min_box_area = config.get("min_box_area", 10)
        self.aspect_ratio_thresh = config.get("aspect_ratio_thresh", 1.6)
        
        # Khởi tạo tracker
        self.tracker = BYTETracker(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            frame_rate=self.frame_rate
        )
        
        # Track IDs và thông tin
        self.tracks = {}
        self.frame_id = 0

    def convert_faces_to_detections(self, faces_info):
        """
        Chuyển đổi thông tin khuôn mặt từ detector sang định dạng phù hợp cho tracker
        
        Args:
            faces_info: Danh sách khuôn mặt từ detector
            
        Returns:
            Numpy array với định dạng [x1, y1, x2, y2, score, class_id]
        """
        if not faces_info:
            return np.zeros((0, 6))
            
        detections = []
        for face in faces_info:
            bbox = face['bbox']
            confidence = face['confidence']
            # Format: [x1, y1, x2, y2, score, class_id]
            # Class_id = 0 cho tất cả khuôn mặt
            detections.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence, 0])
            
        return np.array(detections)

    def update(self, faces_info, frame_size):
        """
        Cập nhật tracker với thông tin khuôn mặt mới từ detector
        
        Args:
            faces_info: Danh sách thông tin khuôn mặt từ detector
            frame_size: Tuple (height, width) của frame
            
        Returns:
            List các face đang được theo dõi với ID
        """
        self.frame_id += 1
        height, width = frame_size
        
        # Chuyển đổi đầu vào sang định dạng phù hợp cho ByteTracker
        detections = self.convert_faces_to_detections(faces_info)
        
        # Cập nhật tracker
        online_targets = self.tracker.update(detections, [height, width])
        
        # Xử lý kết quả tracking
        tracked_faces = []
        for t in online_targets:
            # Kết quả tracking có định dạng: [x1, y1, x2, y2, track_id, class_id, score]
            tlwh = self._tlbr_to_tlwh(t[:4])  # Chuyển từ [x1, y1, x2, y2] sang [x, y, w, h]
            track_id = int(t[4])
            score = float(t[6])
            
            # Lọc các bbox không hợp lệ
            vertical = tlwh[2] / tlwh[3] > self.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                # Tạo thông tin face đã được theo dõi
                tracked_face = {
                    'bbox': [int(t[0]), int(t[1]), int(t[2]), int(t[3])],  # [x1, y1, x2, y2] 
                    'track_id': track_id,
                    'score': score,
                    'tlwh': tlwh  # [x, y, w, h]
                }
                
                # Kết hợp thông tin từ detection ban đầu
                for face in faces_info:
                    if self._calculate_iou(tracked_face['bbox'], face['bbox']) > 0.7:
                        if 'landmarks' in face:
                            tracked_face['landmarks'] = face['landmarks']
                        if 'landmarks_dict' in face:
                            tracked_face['landmarks_dict'] = face['landmarks_dict']
                        break
                        
                tracked_faces.append(tracked_face)
                
        return tracked_faces

    def _tlbr_to_tlwh(self, tlbr):
        """
        Chuyển đổi từ [x1, y1, x2, y2] sang [x, y, w, h]
        """
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret
        
    def _calculate_iou(self, box1, box2):
        """
        Tính toán IoU giữa hai bounding box
        """
        # Tọa độ giao nhau
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Diện tích giao nhau
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        intersection = w * h
        
        # Diện tích từng box
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # IoU
        union = box1_area + box2_area - intersection
        if union == 0:
            return 0
        return intersection / union
        
    def draw_tracks(self, frame, tracked_faces):
        """
        Vẽ các tracked face lên frame kèm ID
        
        Args:
            frame: Ảnh gốc
            tracked_faces: Danh sách khuôn mặt được theo dõi
            
        Returns:
            Frame với thông tin tracking
        """
        img = frame.copy()
        
        for face in tracked_faces:
            bbox = face['bbox']
            track_id = face['track_id']
            
            # Vẽ bounding box
            color = self._get_color(track_id)
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Vẽ ID
            text = f"ID: {track_id}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
            cv2.rectangle(img, (bbox[0], bbox[1] - text_size[1] - 10), 
                         (bbox[0] + text_size[0], bbox[1]), color, -1)
            cv2.putText(img, text, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
            
            # Vẽ landmarks nếu có
            if 'landmarks' in face and face['landmarks'] is not None:
                landmarks = face['landmarks']
                for i in range(5):
                    cv2.circle(img, (int(landmarks[i][0]), int(landmarks[i][1])), 
                              2, (0, 255, 0), -1)
                    
        return img
    
    def _get_color(self, idx):
        """
        Sinh màu sắc dựa trên ID
        """
        idx = abs(int(idx)) * 3
        return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)