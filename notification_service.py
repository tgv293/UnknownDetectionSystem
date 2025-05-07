import os
import cv2
import time
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from config import logger

class NotificationService:
    """Dịch vụ gửi thông báo khi phát hiện người lạ."""

    def __init__(
        self,
        email_config: Optional[Dict] = None,
        detection_threshold: int = 5,
        cooldown_period: int = 300,  # 5 phút
        save_unknown_faces: bool = True,
        unknown_dir: str = "temp_captures",
        periodic_notification: bool = True,
        track_timeout: int = 10,  # Thời gian chờ trước khi xóa track_id (giây)
        recognition_memory_time: int = 60,  # Thời gian ghi nhớ khuôn mặt đã nhận diện (giây)
        auto_cleanup: bool = True,
        max_file_age_days: int = 3  # Tự động dọn dẹp ảnh cũ hơn 3 ngày
    ):
        """Khởi tạo notification service."""
        # Cấu hình cơ bản
        self.email_config = email_config or {}
        self.detection_threshold = detection_threshold
        self.cooldown_period = cooldown_period
        self.save_unknown_faces = save_unknown_faces
        self.unknown_dir = unknown_dir
        self.periodic_notification = periodic_notification
        self.track_timeout = track_timeout
        self.recognition_memory_time = recognition_memory_time
        self.auto_cleanup = auto_cleanup
        self.max_file_age_days = max_file_age_days
        
        # Tạo thư mục lưu ảnh nếu cần
        if self.save_unknown_faces:
            os.makedirs(self.unknown_dir, exist_ok=True)
        
        # Theo dõi người lạ
        self.unknown_counter = {}  # {track_id: số lần phát hiện}
        self.current_unknown_faces = {}  # {track_id: thông tin khuôn mặt}
        self.track_last_seen = {}  # {track_id: thời gian xuất hiện cuối}
        self.last_notification_time = 0
        
        # Theo dõi người quen
        self.detected_persons = set()  # Tên những người quen đã phát hiện
        self.person_count = 0
        self.recognized_tracks = {}  # {track_id: {"name", "last_seen", "similarity"}}
        
        # Quản lý email
        self.email_threads = []
        self._last_cleanup_time = time.time()
        
        logger.info("Notification service initialized")
    
    def process_detection(self, camera_id: str, face_results: List[Dict]) -> None:
        """Xử lý kết quả nhận diện và gửi thông báo khi phát hiện người lạ."""
        if not face_results:
            return
            
        current_time = time.time()
        new_unknown_faces = []
        current_frame_tracks = set()
        
        # Bảo trì hệ thống
        self._clean_inactive_tracks(current_time)
        self._clean_completed_email_threads()
        
        # Tự động dọn dẹp ảnh cũ theo định kỳ (1 ngày)
        if self.auto_cleanup and current_time - self._last_cleanup_time > 86400:
            self.cleanup_temp_captures(self.max_file_age_days)
            self._last_cleanup_time = current_time
        
        # Xử lý từng khuôn mặt trong kết quả
        for face in face_results:
            # Bỏ qua khuôn mặt không thật (anti-spoofing)
            if not face.get("is_real", False):
                continue
                
            # Lấy thông tin cơ bản
            name = face.get("display_name", face.get("name", "unknown"))
            similarity = face.get("display_similarity", face.get("recognition_similarity", 0.0))
            face_image = face.get("face_image")
            mask_class = face.get("mask_class", "unknown")
            track_id = face.get("track_id", -1)
            
            # Bỏ qua nếu không có track_id hợp lệ
            if track_id < 0:
                continue
            
            # Cập nhật trạng thái theo dõi
            self.track_last_seen[track_id] = current_time
            current_frame_tracks.add(track_id)
            
            # Kiểm tra và xử lý khuôn mặt (người quen hoặc người lạ)
            self._process_single_face(track_id, name, similarity, face_image, 
                                     mask_class, current_time, new_unknown_faces)
        
        # Lọc các khuôn mặt đang xuất hiện trong frame hiện tại
        active_faces = {track_id: info for track_id, info in self.current_unknown_faces.items() 
                        if track_id in current_frame_tracks}
        
        # Gửi thông báo khi cần
        self._handle_notifications(camera_id, active_faces, new_unknown_faces, current_time)
    
    def _process_single_face(
        self, 
        track_id: int, 
        name: str, 
        similarity: float, 
        face_image: Any, 
        mask_class: str,
        current_time: float, 
        new_unknown_faces: List[Dict]
    ) -> None:
        """Xử lý một khuôn mặt đơn lẻ, phân loại là người quen hay người lạ."""
        # Điều chỉnh ngưỡng dựa trên trạng thái khẩu trang
        recognition_threshold = 0.6  # Ngưỡng mặc định
        if mask_class in ["with_mask", "incorrect_mask"]:
            recognition_threshold *= 0.83  # Giảm ngưỡng cho người đeo khẩu trang
        
        # Xác định là người quen hay người lạ
        is_known = (name != "unknown") and (similarity >= recognition_threshold)
        
        if is_known:
            # Xử lý người quen đã nhận diện được
            self._handle_known_person(track_id, name, similarity, current_time)
        else:
            # Kiểm tra xem có phải người đã nhận diện gần đây không
            if self._is_recently_recognized(track_id, current_time):
                return
            
            # Xử lý người lạ
            self._process_unknown_person(track_id, face_image, mask_class, 
                                        similarity, current_time, new_unknown_faces)
    
    def _handle_known_person(self, track_id: int, name: str, similarity: float, current_time: float) -> None:
        """Cập nhật thông tin cho người quen đã nhận diện được."""
        # Thêm vào danh sách người đã phát hiện
        self.detected_persons.add(name)
        self.person_count = len(self.detected_persons)
        
        # Cập nhật thông tin track_id đã nhận diện
        self.recognized_tracks[track_id] = {
            "name": name,
            "last_seen": current_time,
            "similarity": similarity
        }
        
        # Xóa khỏi danh sách người lạ nếu có
        self.unknown_counter.pop(track_id, None)
        self.current_unknown_faces.pop(track_id, None)
    
    def _is_recently_recognized(self, track_id: int, current_time: float) -> bool:
        """Kiểm tra xem track_id có phải là người đã nhận diện gần đây không."""
        if track_id not in self.recognized_tracks:
            return False
            
        recognized_info = self.recognized_tracks[track_id]
        if current_time - recognized_info["last_seen"] < self.recognition_memory_time:
            # Khuôn mặt đã được nhận diện gần đây
            return True
        else:
            # Đã quá thời gian ghi nhớ, xóa khỏi danh sách
            self.recognized_tracks.pop(track_id, None)
            return False
    
    def _process_unknown_person(
        self, 
        track_id: int, 
        face_image: Any, 
        mask_class: str,
        similarity: float, 
        current_time: float,
        new_unknown_faces: List[Dict]
    ) -> None:
        """Xử lý khi phát hiện người lạ."""
        # Tăng bộ đếm phát hiện
        if track_id not in self.unknown_counter:
            self.unknown_counter[track_id] = 0
        self.unknown_counter[track_id] += 1
        
        # Kiểm tra ngưỡng phát hiện
        current_count = self.unknown_counter[track_id]
        if current_count >= self.detection_threshold:
            # Lưu ảnh người lạ nếu cần
            image_path = self._save_unknown_face_image(track_id, face_image)
            
            # Cập nhật thông tin người lạ
            is_new = track_id not in self.current_unknown_faces
            face_info = {
                "track_id": track_id,
                "count": current_count,
                "image_path": image_path,
                "time": current_time,
                "mask_class": mask_class,
                "similarity": similarity,
                "is_new": is_new
            }
            
            # Thêm vào danh sách thông báo nếu là người lạ mới
            if is_new:
                new_unknown_faces.append(face_info)
            
            # Cập nhật thông tin vào danh sách hiện tại
            self.current_unknown_faces[track_id] = face_info
    
    def _save_unknown_face_image(self, track_id: int, face_image: Any) -> Optional[str]:
        """Lưu ảnh khuôn mặt người lạ vào thư mục tạm."""
        if not self.save_unknown_faces or face_image is None:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.unknown_dir, f"unknown_track{track_id}_{timestamp}.jpg")
        cv2.imwrite(image_path, face_image)
        return image_path
    
    def _handle_notifications(
        self, 
        camera_id: str, 
        active_faces: Dict[int, Dict], 
        new_unknown_faces: List[Dict],
        current_time: float
    ) -> None:
        """Quyết định khi nào gửi thông báo dựa trên các điều kiện."""
        if not active_faces:
            return
            
        # Gửi thông báo ngay lập tức khi có người lạ mới
        if new_unknown_faces:
            self._send_notification(camera_id, active_faces, has_new_face=True)
            self.last_notification_time = current_time
            logger.info(f"Đã thông báo về {len(new_unknown_faces)} người lạ mới tại camera {camera_id}")
            
        # Gửi thông báo định kỳ về người lạ đã phát hiện trước đó
        elif (self.periodic_notification and 
              current_time - self.last_notification_time > self.cooldown_period):
            self._send_notification(camera_id, active_faces, has_new_face=False)
            self.last_notification_time = current_time
            logger.info(f"Đã gửi thông báo định kỳ về {len(active_faces)} người lạ tại camera {camera_id}")
    
    def _clean_inactive_tracks(self, current_time: float) -> None:
        """Dọn dẹp các track_id không còn hoạt động."""
        tracks_to_remove = [
            track_id for track_id, last_seen in self.track_last_seen.items()
            if current_time - last_seen > self.track_timeout
        ]
        
        for track_id in tracks_to_remove:
            self.track_last_seen.pop(track_id, None)
            self.unknown_counter.pop(track_id, None)
            self.current_unknown_faces.pop(track_id, None)
            self.recognized_tracks.pop(track_id, None)
            
        if tracks_to_remove:
            logger.debug(f"Đã xóa {len(tracks_to_remove)} track không hoạt động")
    
    def _clean_completed_email_threads(self) -> None:
        """Dọn dẹp các thread gửi email đã hoàn thành."""
        self.email_threads = [thread for thread in self.email_threads if thread.is_alive()]
    
    def _send_notification(self, camera_id: str, active_faces: Dict[int, Dict], has_new_face: bool) -> None:
        """Gửi thông báo về các khuôn mặt người lạ."""
        if not active_faces:
            return
            
        # Thu thập thông tin cho email
        images = []
        face_count = len(active_faces)
        new_face_count = sum(1 for face_info in active_faces.values() if face_info.get("is_new", False))
        
        # Lấy đường dẫn đến các ảnh
        for face_info in active_faces.values():
            image_path = face_info.get("image_path")
            if image_path and os.path.exists(image_path):
                images.append(image_path)
                face_info["is_new"] = False  # Đánh dấu đã thông báo
        
        # Xác định loại thông báo
        notification_type = "new_face" if has_new_face else "periodic"
        
        # Gửi email trong thread riêng để không làm lag hệ thống chính
        self._send_email_in_thread(
            camera_id=camera_id,
            image_path=images[0] if images else None,
            count=face_count,
            notification_type=notification_type,
            group_data={
                "total_people": face_count,
                "new_people": new_face_count,
                "all_images": images,
                "face_details": list(active_faces.values())
            }
        )
        
        # Thêm: Gửi thông báo qua WebSocket cho tất cả client đang kết nối
        try:
            from app import active_websockets
            import json
            import asyncio
            
            current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            
            notification_data = {
                "type": notification_type,
                "camera_id": camera_id,
                "new_count": new_face_count,
                "total_count": face_count,
                "timestamp": current_time
            }
            
            # Gửi thông báo đến tất cả WebSocket đang kết nối
            for websocket in active_websockets:
                try:
                    asyncio.create_task(websocket.send_json({
                        "type": "security_notification",
                        "notification": notification_data
                    }))
                except Exception as e:
                    logger.error(f"Error sending WebSocket notification: {e}")
        except Exception as e:
            logger.error(f"Error creating WebSocket notification: {e}")
    
    def _send_email_in_thread(self, **kwargs) -> threading.Thread:
        """Gửi email trong một thread riêng để tránh lag."""
        email_thread = threading.Thread(
            target=self.send_email_notification,
            kwargs=kwargs
        )
        email_thread.daemon = True
        email_thread.start()
        self.email_threads.append(email_thread)
        return email_thread
    
    def send_email_notification(
        self, 
        camera_id: str, 
        image_path: Optional[str] = None, 
        count: int = 0,
        notification_type: str = "new_face",
        face_info: Optional[Dict] = None,
        group_data: Optional[Dict] = None
    ) -> bool:
        """Gửi email thông báo với hình ảnh đính kèm."""
        if not self._is_email_config_valid():
            return False
            
        try:
            # Lấy thông tin email
            sender = self.email_config.get('sender', '')
            receiver = self.email_config.get('receiver', '')
            smtp_server = self.email_config.get('smtp_server', '')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username', '')
            password = self.email_config.get('password', '')
            
            # Tạo email
            msg = MIMEMultipart()
            self._set_email_headers(msg, camera_id, notification_type, count, group_data)
            self._add_email_content(msg, camera_id, notification_type, count, group_data)
            self._attach_images(msg, image_path, group_data)
            
            # Gửi email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Đã gửi email thông báo cho camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi gửi email: {e}")
            return False
    
    def _is_email_config_valid(self) -> bool:
        """Kiểm tra cấu hình email có hợp lệ không."""
        if not self.email_config:
            logger.warning("Không có cấu hình email, bỏ qua thông báo")
            return False
            
        required_fields = ['sender', 'receiver', 'smtp_server', 'username', 'password']
        if not all(field in self.email_config for field in required_fields):
            logger.warning("Cấu hình email không đầy đủ")
            return False
            
        return True
    
    def _set_email_headers(
        self, 
        msg: MIMEMultipart, 
        camera_id: str,
        notification_type: str,
        count: int,
        group_data: Optional[Dict]
    ) -> None:
        """Thiết lập tiêu đề email."""
        sender = self.email_config.get('sender', '')
        receiver = self.email_config.get('receiver', '')
        
        # Thiết lập người gửi và người nhận
        msg['From'] = sender
        msg['To'] = receiver
        
        # Thiết lập tiêu đề email
        if notification_type == "new_face":
            new_count = group_data.get("new_people", 1) if group_data else 1
            if new_count > 1:
                msg['Subject'] = f'CẢNH BÁO AN NINH: Phát hiện {new_count} người lạ mới (Camera {camera_id})'
            else:
                msg['Subject'] = f'CẢNH BÁO AN NINH: Phát hiện người lạ mới (Camera {camera_id})'
        else:
            msg['Subject'] = f'CẬP NHẬT AN NINH: {count} người lạ đang có mặt (Camera {camera_id})'
    
    def _add_email_content(
        self, 
        msg: MIMEMultipart, 
        camera_id: str,
        notification_type: str,
        count: int,
        group_data: Optional[Dict]
    ) -> None:
        """Thêm nội dung vào email."""
        current_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        
        if notification_type == "new_face":
            new_count = group_data.get("new_people", 1) if group_data else 1
            total_count = group_data.get("total_people", count) if group_data else count
            
            text = self._create_new_face_email_content(camera_id, new_count, total_count, current_time)
        else:
            text = self._create_periodic_email_content(camera_id, count, current_time)
            
        msg.attach(MIMEText(text, 'html'))
    
    def _create_new_face_email_content(
        self, 
        camera_id: str, 
        new_count: int, 
        total_count: int, 
        current_time: str
    ) -> str:
        """Tạo nội dung email thông báo người lạ mới."""
        if new_count > 1:
            return f"""
            <html>
            <body>
                <h2>CẢNH BÁO AN NINH</h2>
                <p>Hệ thống đã phát hiện <strong>{new_count} người lạ mới</strong> tại camera {camera_id}.</p>
                <p>Tổng cộng có <strong>{total_count} người lạ</strong> đang xuất hiện.</p>
                <p>Thời gian: {current_time}</p>
                <p>Vui lòng kiểm tra!</p>
            </body>
            </html>
            """
        else:
            return f"""
            <html>
            <body>
                <h2>CẢNH BÁO AN NINH</h2>
                <p>Hệ thống đã phát hiện <strong>người lạ mới</strong> tại camera {camera_id}.</p>
                <p>Tổng cộng có <strong>{total_count} người lạ</strong> đang xuất hiện.</p>
                <p>Thời gian: {current_time}</p>
                <p>Vui lòng kiểm tra!</p>
            </body>
            </html>
            """
    
    def _create_periodic_email_content(self, camera_id: str, count: int, current_time: str) -> str:
        """Tạo nội dung email cập nhật định kỳ."""
        return f"""
        <html>
        <body>
            <h2>CẬP NHẬT AN NINH</h2>
            <p>Hiện có <strong>{count} người lạ</strong> đang xuất hiện tại camera {camera_id}.</p>
            <p>Thời gian: {current_time}</p>
            <p>Đây là thông báo định kỳ.</p>
        </body>
        </html>
        """
    
    def _attach_images(
        self, 
        msg: MIMEMultipart, 
        main_image_path: Optional[str],
        group_data: Optional[Dict]
    ) -> None:
        """Đính kèm hình ảnh vào email."""
        # Đính kèm hình ảnh chính
        if main_image_path and os.path.exists(main_image_path):
            self._attach_single_image(msg, main_image_path)
        
        # Đính kèm các hình ảnh phụ
        if group_data and group_data.get("all_images"):
            # Bỏ qua ảnh đầu tiên vì đã đính kèm là ảnh chính
            for img_path in group_data["all_images"][1:]:
                if os.path.exists(img_path):
                    self._attach_single_image(msg, img_path)
    
    def _attach_single_image(self, msg: MIMEMultipart, image_path: str) -> None:
        """Đính kèm một hình ảnh vào email."""
        with open(image_path, 'rb') as f:
            img_data = f.read()
            image = MIMEImage(img_data)
            image.add_header('Content-Disposition', 
                            f'attachment; filename="{os.path.basename(image_path)}"')
            msg.attach(image)
    
    def update_email_config(self, new_config):
        """Cập nhật cấu hình email."""
        self.email_config = new_config
        logger.info("Email configuration updated")
    
    def cleanup_temp_captures(self, max_age_days: int = 3) -> Dict[str, int]:
        """Dọn dẹp thư mục temp_captures, xóa các tệp cũ hơn max_age_days ngày.
        
        Args:
            max_age_days: Số ngày tối đa để giữ lại tệp (mặc định: 3 ngày)
        
        Returns:
            Dict: Thông tin về số lượng tệp đã xóa và còn lại
        """
        if not self.save_unknown_faces or not os.path.exists(self.unknown_dir):
            return {"deleted": 0, "remaining": 0}
        
        try:
            # Lấy danh sách tất cả các tệp trong thư mục
            all_files = [os.path.join(self.unknown_dir, f) for f in os.listdir(self.unknown_dir) 
                        if os.path.isfile(os.path.join(self.unknown_dir, f))]
            
            # Lọc các tệp hình ảnh
            image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                return {"deleted": 0, "remaining": 0}
                
            # Xác định thời gian hiện tại và thời gian tối đa
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 60 * 60
            deleted_count = 0
            
            # Xóa các tệp cũ
            for file_path in image_files:
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)
                    deleted_count += 1
            
            # Đếm số tệp còn lại
            remaining_files = len([f for f in os.listdir(self.unknown_dir) 
                                if os.path.isfile(os.path.join(self.unknown_dir, f)) 
                                and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # Ghi log kết quả
            logger.info(f"Dọn dẹp thành công: Đã xóa {deleted_count} ảnh cũ, còn lại {remaining_files} ảnh.")
            
            return {
                "deleted": deleted_count,
                "remaining": remaining_files
            }
        except Exception as e:
            logger.error(f"Lỗi khi dọn dẹp thư mục ảnh tạm: {e}")
            return {"error": str(e), "deleted": 0, "remaining": 0}
    
    def reset_counters(self) -> None:
        """Reset các bộ đếm và trạng thái theo dõi."""
        self.unknown_counter.clear()
        self.current_unknown_faces.clear()
        self.track_last_seen.clear()
        self.recognized_tracks.clear()
        self.detected_persons.clear()
        self.person_count = 0
        self.last_notification_time = 0
        logger.info("Đã reset tất cả các bộ đếm và trạng thái theo dõi")