import cv2
import numpy as np
import time
import os
import argparse
import json
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque

# Import các module
from config import MODEL_DIR, FACE_DETECTION_THRESHOLD, MASK_DETECTION_THRESHOLD, logger
from config import COSINE_SIMILARITY_THRESHOLD, EMAILCONFIG_DIR
from face_detector import FaceDetector
from anti_spoofing import AntiSpoofing
from mask_detection import MaskDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker  # Thêm import face tracker
from notification_service import NotificationService  # THÊM MỚI: Import NotificationService

# Đọc email config từ file
def load_email_config(config_path=EMAILCONFIG_DIR):
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config
    except Exception as e:
        logger.error(f"Failed to load email config: {e}")
        return None

class RecognizerSystem:
    """Hệ thống nhận diện khuôn mặt tích hợp các module xử lý khuôn mặt.
    
    Kết hợp các module:
    - Face detection (SCRFD)
    - Face tracking (ByteTrack)
    - Anti spoofing (DeepFace)
    - Mask detection (ONNX)
    - Face recognition (ArcFace)
    - Notification service (Email alerts)
    """
    
    def __init__(
        self,
        frame_skip=2,              # Số frame bỏ qua (áp dụng cho face_detector)
        detection_threshold=FACE_DETECTION_THRESHOLD,
        use_anti_spoofing=True,
        use_mask_detection=True,
        use_recognition=True,
        use_tracking=True,         # Thêm tham số bật/tắt tracking
        use_notification=False,    # THÊM MỚI: Bật/tắt thông báo
        recognition_threshold=COSINE_SIMILARITY_THRESHOLD,
        model_dir=MODEL_DIR,
        email_config=None,         # THÊM MỚI: Cấu hình email
        verbose=False
    ):
        """Khởi tạo hệ thống nhận diện.
        
        Args:
            frame_skip: Số frame bỏ qua khi detect (chỉ detect 1/(skip+1) frames)
            detection_threshold: Ngưỡng phát hiện khuôn mặt
            use_anti_spoofing: Bật/tắt anti-spoofing
            use_mask_detection: Bật/tắt mask detection
            use_recognition: Bật/tắt nhận diện khuôn mặt
            use_tracking: Bật/tắt tracking khuôn mặt
            use_notification: Bật/tắt thông báo
            recognition_threshold: Ngưỡng nhận diện khuôn mặt
            model_dir: Thư mục chứa models
            email_config: Cấu hình email cho notification service
            verbose: Bật/tắt log chi tiết
        """
        self.frame_skip = frame_skip
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.model_dir = model_dir
        self.verbose = verbose
        
        # Cờ bật/tắt các tính năng
        self.use_anti_spoofing = use_anti_spoofing
        self.use_mask_detection = use_mask_detection
        self.use_recognition = use_recognition
        self.use_tracking = use_tracking
        self.use_notification = use_notification  # THÊM MỚI: Cờ bật/tắt thông báo
        
        # Đếm frame
        self.frame_count = 0
        
        # Khởi tạo models
        self._load_models(model_dir)
        
        # Khởi tạo các modules (không log lặp lại)
        self.face_detector = FaceDetector(
            preloaded_model=self.scrfd_model,
            detection_threshold=detection_threshold,
            frame_skip=frame_skip,
            verbose=False
        )
        
        self.anti_spoofing = AntiSpoofing(
            model=None,
            verbose=False
        ) if use_anti_spoofing else None
        
        self.mask_detector = MaskDetector(
            model=self.mask_model,
            threshold=MASK_DETECTION_THRESHOLD,
            verbose=False
        ) if use_mask_detection else None
        
        # Khởi tạo Face Recognizer
        self.face_recognizer = FaceRecognizer(
            model_path=os.path.join(model_dir, "r100_arcface_glint.onnx"),
            threshold=recognition_threshold, 
            verbose=False
        ) if use_recognition else None
        
        # Khởi tạo Face Tracker
        self.face_tracker = FaceTracker(
            config={
                "track_thresh": self.detection_threshold,
                "track_buffer": 30,
                "match_thresh": 0.8,
                "frame_rate": 30
            }
        ) if use_tracking else None
        
        # THÊM MỚI: Khởi tạo Notification Service
        self.notification_service = NotificationService(
            email_config=email_config,
            detection_threshold=3,  # Mặc định 3 lần phát hiện liên tiếp
            cooldown_period=300,   # 5 phút giữa các thông báo định kỳ
            save_unknown_faces=True,
            recognition_memory_time=60  # 60 giây ghi nhớ khuôn mặt đã nhận diện
        ) if use_notification else None
        
        # THÊM MỚI: ID của camera (mặc định là '0')
        self.camera_id = "0"
        
        # Track history cho các đường đi của khuôn mặt
        self.track_history = {}  # Dict lưu lịch sử đường đi cho mỗi track_id
        self.track_history_length = 30  # Số điểm tối đa trong lịch sử
        
        # Cache cho kết quả
        self.last_face_boxes = []
        self.last_results = []
        
        # Cải tiến cách tính FPS
        self.fps_avg_frame_count = 10
        self.fps = 0
        self.fps_start_time = time.time()
        
        # Thêm các thuộc tính cho cơ chế voting danh tính
        self.identity_history = {}  # {track_id: [(name, similarity, timestamp), ...]}
        self.identity_voting_size = 10  # Lưu trữ 10 nhận diện gần nhất cho mỗi track_id
        self.identity_min_votes = 3     # Cần ít nhất 3 votes để xác nhận danh tính
        self.identity_threshold = 0.4   # Ngưỡng tối thiểu để đưa vào voting
        self.identity_max_age = 5.0     # Thời gian tối đa để xem xét một kết quả nhận diện (giây)
        
        # Log thông tin khởi tạo
        logger.info("RecognizerSystem initialized successfully")
        logger.info(f"Configuration: frame_skip={frame_skip}, "
                f"tracking={use_tracking}, anti-spoofing={use_anti_spoofing}, "
                f"mask-detection={use_mask_detection}, recognition={use_recognition}")
        logger.info(f"Recognition threshold: {recognition_threshold}")
    
    # Thêm phương thức để đặt camera ID
    def set_camera_id(self, camera_id):
        """Đặt ID cho camera hiện tại."""
        self.camera_id = str(camera_id)
        logger.info(f"Set camera ID to {self.camera_id}")
    
    def _update_identity_history(self, track_id, name, similarity):
        """Cập nhật lịch sử nhận diện cho một track_id cụ thể."""
        if track_id < 0:  # Bỏ qua track_id không hợp lệ
            return
        
        current_time = time.time()
        
        # Khởi tạo lịch sử cho track_id mới
        if track_id not in self.identity_history:
            self.identity_history[track_id] = []
        
        # Thêm kết quả nhận diện mới vào lịch sử
        # Chỉ nhận các kết quả có độ tương đồng đủ cao
        if similarity >= self.identity_threshold or name != "unknown":
            self.identity_history[track_id].append((name, similarity, current_time))
            
            # Giới hạn kích thước lịch sử
            if len(self.identity_history[track_id]) > self.identity_voting_size:
                self.identity_history[track_id].pop(0)  # Loại bỏ kết quả cũ nhất
    
    def _get_voted_identity(self, track_id):
        """Lấy danh tính được voting nhiều nhất cho một track_id."""
        if track_id < 0 or track_id not in self.identity_history:
            return "unknown", 0.0
        
        history = self.identity_history[track_id]
        if not history:
            return "unknown", 0.0
        
        # Đếm số lần xuất hiện của mỗi danh tính
        identity_counts = {}
        identity_scores = {}
        current_time = time.time()
        
        for name, similarity, timestamp in history:
            # Bỏ qua các kết quả quá cũ
            if current_time - timestamp > self.identity_max_age:
                continue
                
            if name not in identity_counts:
                identity_counts[name] = 0
                identity_scores[name] = 0
                
            identity_counts[name] += 1
            identity_scores[name] += similarity
        
        # Tìm danh tính có số votes cao nhất
        max_votes = 0
        best_name = "unknown"
        best_score = 0
        
        for name, count in identity_counts.items():
            avg_score = identity_scores[name] / count if count > 0 else 0
            
            # Chỉ xét các danh tính có đủ votes và không phải "unknown"
            if name != "unknown" and count >= self.identity_min_votes and count > max_votes:
                max_votes = count
                best_name = name
                best_score = avg_score
                
        # Nếu không có danh tính nào đủ votes, vẫn có thể trả về một danh tính đáng tin cậy
        if best_name == "unknown" and identity_counts:
            # Tìm danh tính không phải "unknown" có độ tương đồng cao nhất
            for name, count in identity_counts.items():
                if name != "unknown":
                    avg_score = identity_scores[name] / count
                    if avg_score >= self.recognition_threshold and avg_score > best_score:
                        best_name = name
                        best_score = avg_score
        
        return best_name, best_score
    
    def _load_models(self, model_dir):
        """Load tất cả các models cần thiết.
        
        Args:
            model_dir: Thư mục chứa models
        """
        if self.verbose:
            logger.info("Loading models...")
        
        # Load face detection model (SCRFD only, không còn trả về 2 models)
        self.scrfd_model = FaceDetector.load_models()
        
        # Load mask detection model nếu cần
        self.mask_model = None
        if self.use_mask_detection:
            try:
                mask_model_path = os.path.join(model_dir, "mask_detector.onnx")
                if os.path.exists(mask_model_path):
                    # Create ONNX session
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.mask_model = ort.InferenceSession(
                        mask_model_path, 
                        sess_options=sess_options, 
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                    if self.verbose:
                        logger.info(f"Mask detection model loaded from {mask_model_path}")
                else:
                    logger.error(f"Mask detection model not found at {mask_model_path}")
            except Exception as e:
                logger.error(f"Failed to load mask detection model: {e}")
           
    
    def process_frame(self, frame):
        """Xử lý một frame từ video stream."""
        if frame is None:
            return None, []
        
        # Tăng biến đếm frame
        self.frame_count += 1
        
        # Tạo bản sao frame để hiển thị
        display = frame.copy()
        
        # Phát hiện và xử lý khuôn mặt - frame_skip đã được xử lý trong face_detector
        # Nếu đã có tracking, chỉ thực hiện phát hiện vài frame một lần
        if self.use_tracking and self.face_tracker:
            # Force detection chỉ mỗi khoảng thời gian nhất định
            force_detect = (self.frame_count % (self.frame_skip * 4)) == 0
        else:
            # Nếu không có tracking, xử lý bình thường
            force_detect = False
            
        face_boxes, processed_faces, landmarks = self.face_detector.process_frame(frame, force_detect)
        
        # Tạo faces_info để sử dụng với tracker
        faces_info = []
        for i, box in enumerate(face_boxes):
            face_info = {'bbox': box, 'confidence': 1.0}
            if landmarks is not None and i < len(landmarks):
                face_info['landmarks'] = landmarks[i]
            faces_info.append(face_info)
        
        # Cập nhật tracker với kết quả detection mới
        tracked_faces = []
        if self.use_tracking and self.face_tracker:
            tracked_faces = self.face_tracker.update(faces_info, frame.shape[:2])
            
            # Cập nhật track history cho tracking (vẫn giữ lại phần này để dùng cho logic)
            for tracked_face in tracked_faces:
                track_id = tracked_face['track_id']
                
                # Lấy điểm giữa bounding box
                bbox = tracked_face['bbox']
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                center = (center_x, center_y)
                
                # Cập nhật lịch sử track
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(center)
                
                # Giới hạn kích thước lịch sử
                if len(self.track_history[track_id]) > self.track_history_length:
                    self.track_history[track_id].pop(0)
        
        # Chuẩn bị dữ liệu cho xử lý - kết hợp kết quả từ detector và tracker
        if self.use_tracking and tracked_faces:
            # Map dữ liệu từ tracked_faces và processed_faces
            face_boxes_to_process = []
            faces_to_process = []
            landmarks_to_process = []
            track_ids_to_process = []
            
            # Xử lý mỗi tracked face
            for tracked_face in tracked_faces:
                track_id = tracked_face['track_id']
                bbox = tracked_face['bbox']
                
                # Thêm bounding box từ tracking
                face_boxes_to_process.append(bbox)
                track_ids_to_process.append(track_id)
                
                # Tìm processed face tương ứng nếu có
                best_match_idx = -1
                best_iou = 0
                
                for i, face_box in enumerate(face_boxes):
                    # Tính IoU để tìm khuôn mặt phù hợp
                    iou = self.face_tracker._calculate_iou(bbox, face_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                
                # Nếu tìm được processed face phù hợp
                if best_match_idx >= 0 and best_iou > 0.7:
                    faces_to_process.append(processed_faces[best_match_idx])
                    
                    # Lấy landmarks nếu có
                    if landmarks is not None and best_match_idx < len(landmarks):
                        landmarks_to_process.append(landmarks[best_match_idx])
                    else:
                        landmarks_to_process.append(None)
                else:
                    # Không tìm thấy processed face phù hợp
                    # Sử dụng cache từ frame trước nếu có
                    face_found = False
                    
                    for prev_result in self.last_results:
                        if prev_result.get("track_id") == track_id:
                            # Sử dụng face_image từ kết quả trước
                            faces_to_process.append(prev_result.get("face_image"))
                            landmarks_to_process.append(None)
                            face_found = True
                            break
                    
                    if not face_found:
                        # Vẫn không tìm được, sử dụng None
                        faces_to_process.append(None)
                        landmarks_to_process.append(None)
        else:
            # Không tracking, sử dụng trực tiếp kết quả từ detector
            face_boxes_to_process = face_boxes
            faces_to_process = processed_faces
            landmarks_to_process = landmarks if landmarks is not None else [None] * len(processed_faces)
            track_ids_to_process = [-1] * len(face_boxes)  # ID âm khi không tracking
        
        # Xử lý từng bước dựa trên các khuôn mặt đã được chuẩn bị
        face_results = []
        
        # Nếu có khuôn mặt để xử lý
        if face_boxes_to_process:
            # Chỉ xử lý tiếp nếu có khuôn mặt đã được trích xuất
            faces_for_processing = []
            valid_indices = []
            valid_track_ids = []
            
            # Lọc các khuôn mặt hợp lệ để xử lý
            for i, face in enumerate(faces_to_process):
                if face is not None and face.size > 0:
                    faces_for_processing.append(face)
                    valid_indices.append(i)
                    valid_track_ids.append(track_ids_to_process[i])
            
            # Xử lý các bước tiếp theo chỉ nếu có khuôn mặt hợp lệ
            if faces_for_processing:
                # Anti spoofing check với track_id
                spoof_results = []
                if self.use_anti_spoofing and self.anti_spoofing:
                    spoof_results = self.anti_spoofing.check_faces(
                        faces_for_processing, 
                        track_ids=valid_track_ids,
                        force_refresh=force_detect
                    )
                else:
                    # Mặc định tất cả là thật
                    spoof_results = [{"is_real": True, "confidence": 1.0}] * len(faces_for_processing)
                
                # Lọc các khuôn mặt thật cho mask detection
                real_faces = []
                real_indices = []
                real_track_ids = []
                
                for i, result in enumerate(spoof_results):
                    if result["is_real"]:
                        real_faces.append(faces_for_processing[i])
                        real_indices.append(valid_indices[i])
                        real_track_ids.append(valid_track_ids[i])
                
                # Mask detection chỉ cho khuôn mặt thật
                mask_results_map = {}  # Map index -> result
                if self.use_mask_detection and self.mask_detector and real_faces:
                    mask_results = self.mask_detector.batch_detect(
                        real_faces, 
                        track_ids=real_track_ids,
                        force_refresh=force_detect
                    )
                    
                    # Lưu kết quả cho từng index gốc
                    for i, orig_idx in enumerate(real_indices):
                        if i < len(mask_results):
                            mask_results_map[orig_idx] = mask_results[i]
                
                # Nhận diện khuôn mặt thật
                recog_results_map = {}  # Map index -> result
                if self.use_recognition and self.face_recognizer and real_faces:
                    # Lấy mask status cho từng khuôn mặt thật
                    mask_statuses = []
                    for i, orig_idx in enumerate(real_indices):
                        if orig_idx in mask_results_map:
                            mask_statuses.append(mask_results_map[orig_idx]["class"])
                        else:
                            mask_statuses.append("unknown")
                    
                    # Tiến hành nhận diện với track IDs
                    recognition_results = self.face_recognizer.batch_recognize(
                        real_faces,
                        mask_statuses,
                        track_ids=real_track_ids,
                        force_refresh=force_detect
                    )
                    
                    # Lưu kết quả cho từng index gốc
                    for i, orig_idx in enumerate(real_indices):
                        if i < len(recognition_results):
                            recog_results_map[orig_idx] = recognition_results[i]
                            
                            # Cập nhật lịch sử nhận diện cho voting
                            if i < len(real_track_ids):
                                track_id = real_track_ids[i]
                                name = recognition_results[i]["name"]
                                similarity = recognition_results[i]["similarity"]
                                self._update_identity_history(track_id, name, similarity)
                
                # Kết hợp kết quả cho mỗi khuôn mặt ban đầu
                for i, bbox in enumerate(face_boxes_to_process):
                    result = {
                        "bbox": bbox,
                        "face_image": faces_to_process[i] if i < len(faces_to_process) else None,
                        "track_id": track_ids_to_process[i] if i < len(track_ids_to_process) else -1
                    }
                    
                    # Tìm index trong danh sách đã xử lý
                    processed_idx = valid_indices.index(i) if i in valid_indices else -1
                    
                    # Thêm thông tin anti-spoofing
                    if processed_idx >= 0 and processed_idx < len(spoof_results):
                        result["is_real"] = spoof_results[processed_idx]["is_real"]
                        result["spoof_confidence"] = spoof_results[processed_idx]["confidence"]
                    else:
                        result["is_real"] = True  # Mặc định là thật nếu không có kết quả
                        result["spoof_confidence"] = 1.0
                    
                    # Thêm thông tin mask detection nếu có
                    if i in mask_results_map:
                        result["mask_class"] = mask_results_map[i]["class"]
                        result["mask_confidence"] = mask_results_map[i]["confidence"]
                    else:
                        result["mask_class"] = "unknown"
                        result["mask_confidence"] = 0.0
                    
                    # Thêm thông tin nhận diện nếu có
                    if i in recog_results_map:
                        result["name"] = recog_results_map[i]["name"]
                        result["recognition_confidence"] = recog_results_map[i]["confidence"]
                        result["recognition_similarity"] = recog_results_map[i]["similarity"]
                    else:
                        result["name"] = "unknown"
                        result["recognition_confidence"] = 0.0
                        result["recognition_similarity"] = 0.0
                    
                    # THÊM MỚI: Lấy kết quả từ voting để hiển thị
                    track_id = result["track_id"]
                    if track_id >= 0:
                        voted_name, voted_score = self._get_voted_identity(track_id)
                        
                        # Chỉ sử dụng kết quả voting nếu có tên không phải "unknown"
                        if voted_name != "unknown":
                            # Thêm trường mới cho display
                            result["display_name"] = voted_name
                            result["display_similarity"] = voted_score
                    
                    face_results.append(result)
        
        # Lưu kết quả cho frame tiếp theo
        self.last_results = face_results
        self.last_face_boxes = face_boxes_to_process
        
        # THÊM MỚI: Gửi thông báo nếu có khuôn mặt lạ
        if self.use_notification and self.notification_service:
            self.notification_service.process_detection(self.camera_id, face_results)
        
        # Vẽ kết quả lên frame
        display = self._draw_results(display, face_results)
        
        # Tính FPS
        if self.frame_count % self.fps_avg_frame_count == 0:
            end_time = time.time()
            self.fps = self.fps_avg_frame_count / (end_time - self.fps_start_time)
            self.fps_start_time = time.time()
        
        # Hiển thị thông tin hiệu suất
        self._draw_performance_stats(display)
        
        return display, face_results
    
    def _draw_results(self, frame, face_results):
        """Vẽ kết quả lên frame."""
        from utils.helpers import draw_bbox_info
        
        result_frame = frame.copy()
        
        for result in face_results:
            bbox = result.get("bbox", None)
            if not bbox:
                continue
            
            # Lấy thông tin về khuôn mặt
            is_real = result.get("is_real", False)
            # Ưu tiên sử dụng display_name từ voting nếu có
            name = result.get("display_name", result.get("name", "unknown"))
            similarity = result.get("display_similarity", result.get('recognition_similarity', 0.0))
            mask_class = result.get("mask_class", "unknown")
            track_id = result.get("track_id", -1)  # Lấy track ID
            
            # Áp dụng ngưỡng khác nhau dựa trên trạng thái mask
            if mask_class in ["with_mask", "incorrect_mask"]:
                threshold = self.recognition_threshold * 0.83  # Giảm 17% cho khẩu trang
            else:
                threshold = self.recognition_threshold
            
            # Mặc định nhãn là unknown với màu đỏ
            label = f"unknown ({similarity:.2f})"
            color = (0, 0, 255)  # BGR: Đỏ
            
            if is_real:
                # Sử dụng threshold đã điều chỉnh để quyết định
                if name != "unknown" and similarity >= threshold:
                    # Nếu nhận diện được và vượt ngưỡng, hiển thị tên người dùng
                    label = f"{name} ({similarity:.2f})"
                    color = (0, 255, 0)  # BGR: Xanh lá
                
                # Thêm thông tin về trạng thái mask
                if mask_class == "with_mask":
                    label += " [mask]"
                elif mask_class == "incorrect_mask":
                    label += " [incorrect]"
                elif mask_class == "without_mask":
                    label += " [no mask]"
            
            # Thêm track ID nếu có
            if track_id >= 0:
                label += f" ID:{track_id}"
            
            # Vẽ kết quả
            draw_bbox_info(
                result_frame,
                bbox=bbox,
                similarity=similarity,
                name=label,
                color=color
            )
        
        return result_frame
    
    def _draw_performance_stats(self, frame):
        """Vẽ thông tin hiệu suất lên frame."""
        # Hiển thị FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị thông tin frame skip
        cv2.putText(frame, f"Frame skip: {self.frame_skip}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Hiển thị số lượng khuôn mặt
        cv2.putText(frame, f"Faces: {len(self.last_face_boxes)}", (10, 85),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Hiển thị trạng thái của các module
        cv2.putText(frame, f"Tracking: {self.use_tracking}", (10, 110),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Anti-spoofing: {self.use_anti_spoofing}", (10, 135),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Mask detection: {self.use_mask_detection}", (10, 160),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Face recognition: {self.use_recognition}", (10, 185),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # THÊM MỚI: Hiện thị trạng thái notification
        cv2.putText(frame, f"Notification: {self.use_notification}", (10, 210),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Hiển thị thông tin threshold
        cv2.putText(frame, f"Threshold: {self.recognition_threshold:.2f}", 
                  (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def reset_caches(self):
        """Reset tất cả cache của các module."""
        logger.info("Resetting all caches")
        if self.face_detector:
            self.face_detector.reset()
        if self.anti_spoofing:
            self.anti_spoofing.reset_cache()
        if self.mask_detector:
            self.mask_detector.reset_cache()
        if self.face_recognizer:
            self.face_recognizer.reset_cache()
        if self.notification_service:
            # Reset notification service nếu cần
            pass
            
        # Reset track history
        self.track_history = {}
        
        # Reset identity history
        self.identity_history = {}
        
        # Reset các biến khác
        self.last_face_boxes = []
        self.last_results = []


def main():
    """Hàm chính để chạy hệ thống nhận diện."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution')
    parser.add_argument('--skip', type=int, default=2, help='Frame skip rate for face detection')
    parser.add_argument('--no-tracking', action='store_true', help='Disable face tracking')
    parser.add_argument('--no-spoofing', action='store_true', help='Disable anti-spoofing')
    parser.add_argument('--no-mask', action='store_true', help='Disable mask detection')
    parser.add_argument('--no-recognition', action='store_true', help='Disable face recognition')
    parser.add_argument('--notification', action='store_true', help='Enable email notifications')  # THÊM MỚI
    parser.add_argument('--threshold', type=float, default=COSINE_SIMILARITY_THRESHOLD, 
                        help='Face recognition similarity threshold')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup camera
    width, height = map(int, args.resolution.split('x'))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {args.camera}")
        return

    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # THAY ĐỔI: Đọc cấu hình email từ file JSON
    email_config = None
    if args.notification:
        email_config = load_email_config()
        if email_config:
            logger.info(f"Email notifications configured: {email_config.get('sender')} -> {email_config.get('receiver')}")
        else:
            logger.warning("Email configuration incomplete. Notifications will be disabled.")
            args.notification = False
    
    # Create recognizer system
    recognizer = RecognizerSystem(
        frame_skip=args.skip,
        use_tracking=not args.no_tracking,
        use_anti_spoofing=not args.no_spoofing,
        use_mask_detection=not args.no_mask,
        use_recognition=not args.no_recognition,
        use_notification=args.notification,
        recognition_threshold=args.threshold,
        email_config=email_config,
        verbose=args.verbose
    )
    
    # Đặt ID camera
    recognizer.set_camera_id(args.camera)
    
    # Create window
    window_name = "Face Recognition System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Log thông tin khởi động
    logger.info(f"Starting recognition system with camera {args.camera}")
    logger.info(f"Recognition threshold: {args.threshold}")
    logger.info(f"Face tracking: {not args.no_tracking}")
    logger.info(f"Notifications: {args.notification}")
    
    try:
        while True:
            # Get frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Process frame
            start_time = time.time()
            display, results = recognizer.process_frame(frame)
            process_time = time.time() - start_time
            
            # Add processing time info
            cv2.putText(
                display, 
                f"Process: {process_time*1000:.1f}ms", 
                (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
            # Show frame
            cv2.imshow(window_name, display)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset caches
                recognizer.reset_caches()
                logger.info("All caches reset")
                
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Recognition system stopped")

if __name__ == "__main__":
    main()