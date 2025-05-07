import cv2
import numpy as np
import time
import os
import argparse
import json
import onnxruntime as ort
import sys
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import deque

# Import các module
from config import MODEL_DIR, FACE_DETECTION_THRESHOLD, MASK_DETECTION_THRESHOLD, logger
from config import COSINE_SIMILARITY_THRESHOLD, EMAILCONFIG_DIR
from face_detector import FaceDetector
from anti_spoofing import AntiSpoofing
from mask_detection import MaskDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker
from notification_service import NotificationService


def load_email_config(config_path=EMAILCONFIG_DIR):
    """Load email configuration from JSON file."""
    try:
        # Tạo thư mục config nếu chưa tồn tại
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        if not os.path.exists(config_path):
            # Tạo file config mẫu nếu chưa tồn tại
            default_config = {
                "sender": "",
                "receiver": "",
                "smtp_server": "",
                "smtp_port": 587,
                "username": "",
                "password": ""
            }
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
            
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading email config: {e}")
        return {}

class RecognizerSystem:
    """Hệ thống nhận diện khuôn mặt tích hợp các module xử lý khuôn mặt."""
    
    def __init__(
        self,
        frame_skip=2,
        detection_threshold=FACE_DETECTION_THRESHOLD,
        use_anti_spoofing=True,
        use_mask_detection=True,
        use_recognition=True,
        use_tracking=True,
        use_notification=False,
        recognition_threshold=COSINE_SIMILARITY_THRESHOLD,
        model_dir=MODEL_DIR,
        email_config=None,
        verbose=False
    ):
        """Khởi tạo hệ thống nhận diện."""
        self.frame_skip = frame_skip
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.model_dir = model_dir
        self.verbose = verbose
        self.email_config = email_config
        
        # Cờ bật/tắt các tính năng
        self.use_anti_spoofing = use_anti_spoofing
        self.use_mask_detection = use_mask_detection
        self.use_recognition = use_recognition
        self.use_tracking = use_tracking
        self.use_notification = use_notification
        
        # Đếm frame và camera ID
        self.frame_count = 0
        self.camera_id = "0"
        
        # Khởi tạo models
        self._load_models(model_dir)
        
        # Khởi tạo các modules
        self._initialize_modules()
        
        # Cache cho kết quả
        self.last_face_boxes = []
        self.last_results = []
        
        # Cải tiến cách tính FPS
        self.fps_avg_frame_count = 10
        self.fps = 0
        self.fps_start_time = time.time()
        
        # Cấu hình voting danh tính - CẢI TIẾN: Tăng thời gian lưu lịch sử nhận diện
        self.identity_history = {}  # {track_id: [(name, similarity, timestamp), ...]}
        self.identity_voting_size = 5  # Tăng từ 10 lên 20 votes
        self.identity_min_votes = 3     # Cần ít nhất 3 votes để xác nhận danh tính
        self.identity_threshold = 0.4   # Ngưỡng tối thiểu để đưa vào voting
        self.identity_max_age = 15.0    # Tăng từ 5.0 lên 15.0 giây
        
        # THÊM MỚI: Danh tính ổn định cho mỗi track_id
        self.stable_identities = {}     # {track_id: {"name": name, "similarity": score, "time": timestamp}}
        self.stable_identity_timeout = 1  # 1 giây trước khi xem xét thay đổi danh tính ổn định
        self.stable_identity_threshold = 0.5  # Ngưỡng để thiết lập danh tính ổn định
        
        logger.info("RecognizerSystem initialized successfully")
        
    def _initialize_modules(self):
        """Khởi tạo các modules của hệ thống."""
        # Khởi tạo face detector
        try:
            self.face_detector = FaceDetector(
                preloaded_model=self.scrfd_model,
                detection_threshold=self.detection_threshold,
                frame_skip=self.frame_skip,
                verbose=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise RuntimeError("Cannot initialize system: Face detector initialization failed")
        
        # Khởi tạo anti-spoofing module
        self.anti_spoofing = None
        if self.use_anti_spoofing:
            try:
                self.anti_spoofing = AntiSpoofing(model=None, verbose=False)
            except Exception as e:
                logger.error(f"Failed to initialize anti-spoofing: {e}")
                self.use_anti_spoofing = False
        
        # Khởi tạo mask detector
        self.mask_detector = None
        if self.use_mask_detection and self.mask_model:
            try:
                self.mask_detector = MaskDetector(
                    model=self.mask_model,
                    threshold=MASK_DETECTION_THRESHOLD,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Failed to initialize mask detector: {e}")
                self.use_mask_detection = False
        else:
            self.use_mask_detection = False
        
        # Khởi tạo face recognizer
        self.face_recognizer = None
        if self.use_recognition:
            try:
                model_path = os.path.join(self.model_dir, "face_recognition", "r100_arcface_glint.onnx")
                if not os.path.exists(model_path):
                    logger.error(f"ArcFace model not found at: {model_path}")
                    raise FileNotFoundError(f"ArcFace model file not found: {model_path}")
                    
                self.face_recognizer = FaceRecognizer(
                    model_path=model_path,
                    threshold=self.recognition_threshold, 
                    verbose=False
                )
                
                if not self.face_recognizer.model_loaded:
                    logger.error("Failed to load face recognition model")
                    raise RuntimeError("Face recognition model not loaded properly")
            except Exception as e:
                logger.error(f"Failed to initialize face recognizer: {e}")
                raise RuntimeError(f"Cannot initialize system: {e}")
        
        # Khởi tạo face tracker
        self.face_tracker = None
        if self.use_tracking:
            try:
                self.face_tracker = FaceTracker(
                    config={
                        "track_thresh": self.detection_threshold,
                        "track_buffer": 60,
                        "match_thresh": 0.7,
                        "frame_rate": 30
                    }
                )
            except Exception as e:
                logger.error(f"Failed to initialize face tracker: {e}")
                self.use_tracking = False
        
        # Khởi tạo notification service
        self.notification_service = None
        if self.use_notification and hasattr(self, 'email_config') and self.email_config:
            try:
                self.notification_service = NotificationService(
                    email_config=self.email_config,
                    detection_threshold=3,
                    cooldown_period=300,
                    save_unknown_faces=True,
                    recognition_memory_time=60
                )
            except Exception as e:
                logger.error(f"Failed to initialize notification service: {e}")
                self.use_notification = False
    
    def set_camera_id(self, camera_id):
        """Đặt ID cho camera hiện tại."""
        self.camera_id = str(camera_id)
        logger.info(f"Set camera ID to {self.camera_id}")
    
    def _update_identity_history(self, track_id, name, similarity):
        """Cập nhật lịch sử nhận diện cho một track_id."""
        if track_id < 0:  # Bỏ qua track_id không hợp lệ
            return
        
        current_time = time.time()
        
        # Khởi tạo lịch sử cho track_id mới
        if track_id not in self.identity_history:
            self.identity_history[track_id] = []
        
        # Thêm kết quả nhận diện mới vào lịch sử
        if similarity >= self.identity_threshold or name != "unknown":
            self.identity_history[track_id].append((name, similarity, current_time))
            
            # Giới hạn kích thước lịch sử
            if len(self.identity_history[track_id]) > self.identity_voting_size:
                self.identity_history[track_id].pop(0)
    
    def _clean_stable_identities(self):
        """Dọn dẹp các danh tính ổn định đã quá hạn."""
        current_time = time.time()
        to_remove = []
        
        for track_id, data in self.stable_identities.items():
            if current_time - data["time"] > self.stable_identity_timeout * 2:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            self.stable_identities.pop(track_id, None)
    
    def _get_voted_identity(self, track_id):
        """Lấy danh tính được voting nhiều nhất cho một track_id."""
        if track_id < 0:
            return "unknown", 0.0
        
        current_time = time.time()
        
        # Kiểm tra nếu đã có danh tính ổn định cho track_id này
        if track_id in self.stable_identities:
            stable_data = self.stable_identities[track_id]
            stable_name = stable_data["name"]
            stable_score = stable_data["similarity"]
            stable_time = stable_data["time"]
            
            # Nếu danh tính ổn định gần đây, sử dụng luôn
            if current_time - stable_time < self.stable_identity_timeout:
                # Chỉ xác nhận lại danh tính ổn định nếu có lịch sử danh tính mới tốt hơn
                if track_id in self.identity_history:
                    history = self.identity_history[track_id]
                    if history:
                        # Đếm số lần xuất hiện của danh tính khác
                        different_names = [item for item in history 
                                         if item[0] != stable_name 
                                         and item[0] != "unknown" 
                                         and item[1] >= self.recognition_threshold
                                         and current_time - item[2] < 5.0]
                        
                        # Nếu không có đủ bằng chứng mạnh để thay đổi, giữ nguyên danh tính ổn định
                        if len(different_names) < 5:  # Cần ít nhất 5 nhận diện khác mới xem xét thay đổi
                            return stable_name, stable_score
        
        # Xử lý voting nếu không có danh tính ổn định hoặc cần cập nhật
        if track_id not in self.identity_history:
            return "unknown", 0.0
        
        history = self.identity_history[track_id]
        if not history:
            return "unknown", 0.0
        
        # Đếm số lần xuất hiện và tính điểm trung bình
        identity_counts = {}
        identity_scores = {}
        
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
            best_non_unknown_name = "unknown"
            best_non_unknown_score = 0
            
            for name, count in identity_counts.items():
                if name != "unknown":
                    avg_score = identity_scores[name] / count if count > 0 else 0
                    if avg_score >= self.recognition_threshold and avg_score > best_non_unknown_score:
                        best_non_unknown_name = name
                        best_non_unknown_score = avg_score
            
            if best_non_unknown_name != "unknown":
                best_name = best_non_unknown_name
                best_score = best_non_unknown_score
        
        # THÊM MỚI: Cập nhật danh tính ổn định khi có kết quả tin cậy cao
        if best_name != "unknown" and best_score >= self.stable_identity_threshold:
            self.stable_identities[track_id] = {
                "name": best_name,
                "similarity": best_score,
                "time": current_time
            }
        
        return best_name, best_score
    
    def _load_models(self, model_dir):
        """Load tất cả các models cần thiết."""
        # Load face detection model (SCRFD)
        try:
            scrfd_model_path = os.path.join(model_dir, "face_detection", "det_34g.onnx")
            if not os.path.exists(scrfd_model_path):
                raise FileNotFoundError(f"Face detection model not found: {scrfd_model_path}")
                
            self.scrfd_model = FaceDetector.load_models(model_path=scrfd_model_path)
            if self.scrfd_model is None:
                raise RuntimeError("Face detection model could not be loaded")
        except Exception as e:
            logger.error(f"Critical error loading SCRFD model: {e}")
            raise RuntimeError(f"Cannot initialize system: {e}")
        
        # Load mask detection model
        self.mask_model = None
        if self.use_mask_detection:
            try:
                mask_model_path = os.path.join(model_dir, "mask_detection", "mask_detector.onnx")
                if os.path.exists(mask_model_path):
                    sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.mask_model = ort.InferenceSession(
                        mask_model_path, 
                        sess_options=sess_options, 
                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                    )
                else:
                    logger.error(f"Mask detection model not found at {mask_model_path}")
                    self.use_mask_detection = False
            except Exception as e:
                logger.error(f"Failed to load mask detection model: {e}")
                self.use_mask_detection = False
    
    def _prepare_tracking_data(self, face_boxes, processed_faces, landmarks, frame_shape):
        """Chuẩn bị dữ liệu tracking từ kết quả phát hiện khuôn mặt."""
        # Tạo faces_info để sử dụng với tracker
        faces_info = []
        for i, box in enumerate(face_boxes):
            face_info = {'bbox': box, 'confidence': 1.0}
            if landmarks is not None and i < len(landmarks):
                face_info['landmarks'] = landmarks[i]
            faces_info.append(face_info)
        
        # Cập nhật tracker với kết quả detection mới
        tracked_faces = []
        if self.use_tracking and self.face_tracker and faces_info:
            try:
                tracked_faces = self.face_tracker.update(faces_info, frame_shape[:2])
            except Exception as e:
                logger.error(f"Error updating tracker: {e}")
                
        return tracked_faces
    
    def _map_faces_to_tracks(self, tracked_faces, face_boxes, processed_faces, landmarks):
        """Map kết quả nhận diện khuôn mặt với các track."""
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
            
            # Tìm processed face tương ứng
            best_match_idx = -1
            best_iou = 0
            
            for i, face_box in enumerate(face_boxes):
                # Tính IoU để tìm khuôn mặt phù hợp
                iou = self.face_tracker._calculate_iou(bbox, face_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match_idx = i
            
            # Xử lý nếu tìm được hoặc không tìm được processed face
            if best_match_idx >= 0 and best_iou > 0.4 and best_match_idx < len(processed_faces):
                faces_to_process.append(processed_faces[best_match_idx])
                
                if landmarks is not None and best_match_idx < len(landmarks):
                    landmarks_to_process.append(landmarks[best_match_idx])
                else:
                    landmarks_to_process.append(None)
            else:
                # Tìm face_image từ kết quả trước
                face_found = False
                for prev_result in self.last_results:
                    if prev_result.get("track_id") == track_id:
                        faces_to_process.append(prev_result.get("face_image"))
                        landmarks_to_process.append(None)
                        face_found = True
                        break
                
                if not face_found:
                    faces_to_process.append(None)
                    landmarks_to_process.append(None)
                    
        return face_boxes_to_process, faces_to_process, landmarks_to_process, track_ids_to_process
    
    def _process_face_features(self, faces_to_process, track_ids_to_process, force_detect=False):
        """Xử lý các tính năng khuôn mặt (anti-spoof, mask, recognition)."""
        # Lọc các khuôn mặt hợp lệ
        faces_for_processing = []
        valid_indices = []
        valid_track_ids = []
        
        for i, face in enumerate(faces_to_process):
            if face is not None and face.size > 0 and i < len(track_ids_to_process):
                faces_for_processing.append(face)
                valid_indices.append(i)
                valid_track_ids.append(track_ids_to_process[i])
        
        # FIX: Return 4 values instead of 3 when no faces are found
        if not faces_for_processing:
            return [], [], {}, {}
            
        # Anti-spoofing check
        spoof_results = []
        if self.use_anti_spoofing and self.anti_spoofing:
            try:
                spoof_results = self.anti_spoofing.check_faces(
                    faces_for_processing,
                    track_ids=valid_track_ids,
                    force_refresh=force_detect
                )
            except Exception as e:
                logger.error(f"Error in anti-spoofing: {e}")
                spoof_results = [{"is_real": True, "confidence": 1.0}] * len(faces_for_processing)
        else:
            spoof_results = [{"is_real": True, "confidence": 1.0}] * len(faces_for_processing)
        
        # Lọc khuôn mặt thật cho mask detection và recognition
        real_faces = []
        real_indices = []
        real_track_ids = []
        
        for i, result in enumerate(spoof_results):
            if result["is_real"]:
                real_faces.append(faces_for_processing[i])
                real_indices.append(valid_indices[i])
                real_track_ids.append(valid_track_ids[i])
        
        # Mask detection
        mask_results_map = {}
        if self.use_mask_detection and self.mask_detector and real_faces:
            try:
                mask_results = self.mask_detector.batch_detect(
                    real_faces,
                    track_ids=real_track_ids,
                    force_refresh=force_detect
                )
                
                for i, orig_idx in enumerate(real_indices):
                    if i < len(mask_results):
                        mask_results_map[orig_idx] = mask_results[i]
            except Exception as e:
                logger.error(f"Error in mask detection: {e}")
        
        # Face recognition
        recog_results_map = {}
        if self.use_recognition and self.face_recognizer and real_faces:
            try:
                mask_statuses = []
                for i, orig_idx in enumerate(real_indices):
                    if orig_idx in mask_results_map:
                        mask_statuses.append(mask_results_map[orig_idx]["class"])
                    else:
                        mask_statuses.append("unknown")
                
                recognition_results = self.face_recognizer.batch_recognize(
                    real_faces,
                    mask_statuses,
                    track_ids=real_track_ids,
                    force_refresh=force_detect
                )
                
                for i, orig_idx in enumerate(real_indices):
                    if i < len(recognition_results):
                        recog_results_map[orig_idx] = recognition_results[i]
                        
                        # Cập nhật lịch sử nhận diện
                        if i < len(real_track_ids):
                            track_id = real_track_ids[i]
                            name = recognition_results[i]["name"]
                            similarity = recognition_results[i]["similarity"]
                            self._update_identity_history(track_id, name, similarity)
            except Exception as e:
                logger.error(f"Error in face recognition: {e}")
        
        return valid_indices, spoof_results, mask_results_map, recog_results_map
    
    def process_frame(self, frame):
        """Xử lý một frame từ video stream."""
        if frame is None:
            return None, []
        
        # Tăng biến đếm frame và tạo bản sao để hiển thị
        self.frame_count += 1
        display = frame.copy()
        
        # Dọn dẹp các danh tính ổn định đã quá hạn
        self._clean_stable_identities()
        
        # Phát hiện và xử lý khuôn mặt
        force_detect = (self.frame_count % (self.frame_skip * 2)) == 0 if self.use_tracking else False
            
        try:
            face_boxes, processed_faces, landmarks = self.face_detector.process_frame(frame, force_detect)
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return display, self.last_results
        
        # Chuẩn bị dữ liệu cho processing
        if self.use_tracking and self.face_tracker:
            tracked_faces = self._prepare_tracking_data(face_boxes, processed_faces, landmarks, frame.shape)
            if tracked_faces:
                face_boxes_to_process, faces_to_process, landmarks_to_process, track_ids_to_process = \
                    self._map_faces_to_tracks(tracked_faces, face_boxes, processed_faces, landmarks)
            else:
                # Nếu không có track, sử dụng kết quả detector
                face_boxes_to_process = face_boxes
                faces_to_process = processed_faces
                landmarks_to_process = landmarks if landmarks is not None else [None] * len(processed_faces)
                track_ids_to_process = [-1] * len(face_boxes)
        else:
            # Không tracking, sử dụng trực tiếp kết quả detector
            face_boxes_to_process = face_boxes
            faces_to_process = processed_faces
            landmarks_to_process = landmarks if landmarks is not None else [None] * len(processed_faces)
            track_ids_to_process = [-1] * len(face_boxes)
        
        # Xử lý các khuôn mặt đã được chuẩn bị
        face_results = []
        
        if face_boxes_to_process:
            try:
                # Xử lý anti-spoofing, mask detection và recognition
                valid_indices, spoof_results, mask_results_map, recog_results_map = \
                    self._process_face_features(faces_to_process, track_ids_to_process, force_detect)
                
                # Kết hợp kết quả
                for i, bbox in enumerate(face_boxes_to_process):
                    result = {
                        "bbox": bbox,
                        "face_image": faces_to_process[i] if i < len(faces_to_process) else None,
                        "track_id": track_ids_to_process[i] if i < len(track_ids_to_process) else -1,
                        "is_real": True,
                        "name": "unknown",
                        "mask_class": "unknown",
                        "recognition_similarity": 0.0
                    }
                    
                    # Chỉ xử lý nếu có kết quả hợp lệ
                    if valid_indices:
                        # Tìm index trong danh sách đã xử lý
                        try:
                            processed_idx = valid_indices.index(i) if i in valid_indices else -1
                            
                            # Thêm thông tin anti-spoofing
                            if processed_idx >= 0 and processed_idx < len(spoof_results):
                                result["is_real"] = spoof_results[processed_idx]["is_real"]
                            
                            # Chỉ xử lý mask và recognition cho khuôn mặt thật
                            if result["is_real"]:
                                # Thêm thông tin mask detection
                                if i in mask_results_map:
                                    result["mask_class"] = mask_results_map[i]["class"]
                                
                                # Thêm thông tin nhận diện 
                                if i in recog_results_map:
                                    result["name"] = recog_results_map[i]["name"]
                                    result["recognition_similarity"] = recog_results_map[i]["similarity"]
                                
                                # Lấy kết quả từ voting
                                track_id = result["track_id"]
                                if track_id >= 0:
                                    voted_name, voted_score = self._get_voted_identity(track_id)
                                    
                                    if voted_name != "unknown":
                                        result["display_name"] = voted_name
                                        result["display_similarity"] = voted_score
                        except ValueError:
                            # Xử lý trường hợp i không có trong valid_indices
                            pass
                    
                    face_results.append(result)
            except Exception as e:
                logger.error(f"Error processing face features: {e}")
                # Trường hợp lỗi, trả về kết quả trống
        
        # Lưu kết quả cho frame tiếp theo
        self.last_results = face_results
        self.last_face_boxes = face_boxes_to_process
        
        # Gửi thông báo nếu có khuôn mặt lạ
        if self.use_notification and self.notification_service:
            try:
                self.notification_service.process_detection(self.camera_id, face_results)
            except Exception as e:
                logger.error(f"Error sending notifications: {e}")
        
        # Vẽ kết quả và hiệu suất
        display = self._draw_results(display, face_results)
        
        # Tính FPS
        if self.frame_count % self.fps_avg_frame_count == 0:
            end_time = time.time()
            self.fps = self.fps_avg_frame_count / (end_time - self.fps_start_time)
            self.fps_start_time = time.time()
        
        display = self._draw_performance_stats(display)
        
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
            name = result.get("display_name", result.get("name", "unknown"))
            similarity = result.get("display_similarity", result.get('recognition_similarity', 0.0))
            mask_class = result.get("mask_class", "unknown")
            track_id = result.get("track_id", -1)
            
            # Áp dụng ngưỡng tùy theo trạng thái mask
            threshold = self.recognition_threshold * 0.83 if mask_class in ["with_mask", "incorrect_mask"] else self.recognition_threshold
            
            # Xác định màu hiển thị
            color = (0, 0, 255)  # Đỏ cho unknown/fake
            
            if is_real and name != "unknown" and similarity >= threshold:
                color = (0, 255, 0)  # Xanh lá cho khuôn mặt đã biết
            
            # Format nhãn ngắn gọn: ID: X Name: Y (không thêm score vì draw_bbox_info sẽ thêm)
            track_text = f"{track_id}" if track_id >= 0 else "-"
            label = f"ID:{track_text} {name}"  # Bỏ score ra khỏi label
            
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
        # Hiển thị FPS và số khuôn mặt
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Faces: {len(self.last_face_boxes)}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def reset_caches(self):
        """Reset tất cả cache của các module."""
        logger.info("Resetting all caches")
        
        # Reset module caches
        if self.face_detector:
            self.face_detector.reset()
            
        if self.anti_spoofing:
            self.anti_spoofing.reset_cache()
            
        if self.mask_detector:
            self.mask_detector.reset_cache()
            
        if self.face_recognizer:
            self.face_recognizer.reset_cache()
        
        # Reset notification counters
        if self.notification_service and hasattr(self.notification_service, "reset_counters"):
            self.notification_service.reset_counters()
            
        # Reset tracking data
        self.last_face_boxes = []
        self.last_results = []
        self.identity_history = {}
        self.stable_identities = {}  # Reset stable identities
        
        logger.info("All caches reset successfully")


def main():
    """Hàm chính để chạy hệ thống nhận diện."""
    parser = argparse.ArgumentParser(description='Face Recognition System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution')
    parser.add_argument('--skip', type=int, default=2, help='Frame skip rate for detection')
    parser.add_argument('--no-tracking', action='store_true', help='Disable face tracking')
    parser.add_argument('--no-spoofing', action='store_true', help='Disable anti-spoofing')
    parser.add_argument('--no-mask', action='store_true', help='Disable mask detection')
    parser.add_argument('--no-recognition', action='store_true', help='Disable face recognition')
    parser.add_argument('--notification', action='store_true', help='Enable email notifications')
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
    
    # Đọc cấu hình email nếu cần
    email_config = None
    if args.notification:
        email_config = load_email_config()
        if not email_config:
            logger.warning("Email configuration incomplete. Notifications disabled.")
            args.notification = False
    
    try:
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
        
        recognizer.set_camera_id(args.camera)
        
        # Create window
        window_name = "Face Recognition System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        logger.info(f"Starting recognition system with camera {args.camera}")
        
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
            
            # Add processing time
            cv2.putText(
                display, 
                f"Process: {process_time*1000:.1f}ms", 
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            
            # Show frame
            cv2.imshow(window_name, display)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                recognizer.reset_caches()
                
    except KeyboardInterrupt:
        logger.info("Recognition system interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error running recognition system: {e}", exc_info=True)
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Recognition system stopped")


if __name__ == "__main__":
    main()