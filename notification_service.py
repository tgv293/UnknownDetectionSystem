import os
import cv2
import time
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from typing import Dict, List, Optional
from config import logger

class NotificationService:
    """Dịch vụ gửi thông báo khi phát hiện người lạ."""

    def __init__(
        self,
        email_config: Dict = None,
        detection_threshold: int = 3,
        cooldown_period: int = 300,  # 5 phút = 300 giây
        save_unknown_faces: bool = True,
        unknown_dir: str = "temp_captures",
        periodic_notification: bool = True,  # Bật/tắt thông báo định kỳ
        track_timeout: int = 10,  # Thời gian chờ trước khi xóa track_id (giây)
        recognition_memory_time: int = 60  # Thời gian ghi nhớ khuôn mặt đã nhận diện (giây)
    ):
        """Khởi tạo notification service."""
        self.email_config = email_config or {}
        self.detection_threshold = detection_threshold
        self.cooldown_period = cooldown_period
        self.save_unknown_faces = save_unknown_faces
        self.unknown_dir = unknown_dir
        self.periodic_notification = periodic_notification
        self.track_timeout = track_timeout
        self.recognition_memory_time = recognition_memory_time
        
        # Đảm bảo thư mục tồn tại
        if self.save_unknown_faces:
            os.makedirs(self.unknown_dir, exist_ok=True)
        
        # Theo dõi người lạ - sử dụng track_id
        self.unknown_counter = {}  # {track_id: counter} - Đếm số lần phát hiện liên tục
        
        # Danh sách người lạ đang có mặt
        self.current_unknown_faces = {}  # {track_id: {face_info}}
        
        # Thêm theo dõi thời gian xuất hiện cuối cùng của mỗi track_id
        self.track_last_seen = {}  # {track_id: timestamp}
        
        # Thời gian thông báo cuối cùng
        self.last_notification_time = 0
        
        # Danh sách người đã nhận diện
        self.detected_persons = set()
        self.person_count = 0
        
        # THÊM MỚI: Lưu trữ những track_id đã được nhận diện
        self.recognized_tracks = {}  # {track_id: {"name": name, "last_seen": timestamp}}
        
        # THÊM MỚI: Email thread pool
        self.email_threads = []
        
        logger.info("Notification service initialized")
        
    def process_detection(self, camera_id: str, face_results: List[Dict]):
        """Xử lý kết quả nhận diện và gửi thông báo nếu cần."""
        if not face_results:
            return
            
        current_time = time.time()
        new_unknown_faces = []  # Danh sách người lạ mới phát hiện trong frame này
        current_frame_tracks = set()  # Tập hợp các track_id đang trong frame hiện tại
        
        # Dọn dẹp các track_id đã quá thời gian timeout
        self._clean_inactive_tracks(current_time)
        
        # Dọn dẹp các email thread đã hoàn thành
        self._clean_completed_email_threads()
            
        # Cập nhật danh sách người quen và theo dõi người lạ
        for result in face_results:
            is_real = result.get("is_real", False)
            if not is_real:
                continue
                
            # Sử dụng kết quả từ cơ chế voting nếu có
            name = result.get("display_name", result.get("name", "unknown"))
            similarity = result.get("display_similarity", result.get("recognition_similarity", 0.0))
            
            face_image = result.get("face_image")
            mask_class = result.get("mask_class", "unknown")
            track_id = result.get("track_id", -1)
            
            # Bỏ qua nếu không có track_id hợp lệ
            if track_id < 0:
                continue
            
            # Cập nhật thời gian xuất hiện cuối cùng và đánh dấu đang trong khung hình hiện tại
            self.track_last_seen[track_id] = current_time
            current_frame_tracks.add(track_id)
            
            # Áp dụng ngưỡng phù hợp với trạng thái mask
            threshold = 0.6  # Ngưỡng mặc định
            if mask_class in ["with_mask", "incorrect_mask"]:
                threshold *= 0.83  # Giảm 17% cho khẩu trang
            
            # Xác định là người quen hay người lạ
            is_known = (name != "unknown") and (similarity >= threshold)
            
            # Nếu là người quen, thêm vào danh sách và xóa khỏi unknown nếu có
            if is_known:
                self.detected_persons.add(name)
                self.person_count = len(self.detected_persons)
                
                # Cập nhật thông tin track_id đã nhận diện
                self.recognized_tracks[track_id] = {
                    "name": name,
                    "last_seen": current_time,
                    "similarity": similarity
                }
                
                # Xóa khỏi danh sách unknown nếu có
                if track_id in self.unknown_counter:
                    self.unknown_counter.pop(track_id, None)
                if track_id in self.current_unknown_faces:
                    self.current_unknown_faces.pop(track_id, None)
                continue
            
            # THÊM MỚI: Kiểm tra xem khuôn mặt này có phải là người đã từng nhận diện trong khoảng thời gian gần đây không
            if track_id in self.recognized_tracks:
                recognized_info = self.recognized_tracks[track_id]
                if current_time - recognized_info["last_seen"] < self.recognition_memory_time:
                    # Khuôn mặt đã được nhận diện gần đây, bỏ qua không xử lý như unknown
                    logger.debug(f"Skipping track_id {track_id} as it was recognized as {recognized_info['name']} recently")
                    continue
                else:
                    # Đã quá thời gian ghi nhớ, xóa khỏi danh sách đã nhận diện
                    self.recognized_tracks.pop(track_id, None)
            
            # Xử lý người lạ (tăng counter)
            if track_id not in self.unknown_counter:
                self.unknown_counter[track_id] = 0
            self.unknown_counter[track_id] += 1
            
            # Kiểm tra ngưỡng phát hiện
            current_count = self.unknown_counter[track_id]
            if current_count >= self.detection_threshold:
                # Lưu hình ảnh nếu có
                image_path = None
                if self.save_unknown_faces and face_image is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(self.unknown_dir, f"unknown_track{track_id}_{timestamp}.jpg")
                    cv2.imwrite(image_path, face_image)
                
                # Cập nhật thông tin người lạ đang có mặt
                face_info = {
                    "track_id": track_id,
                    "count": current_count,
                    "image_path": image_path,
                    "time": current_time,
                    "mask_class": mask_class,
                    "similarity": similarity,
                    "is_new": track_id not in self.current_unknown_faces
                }
                
                # Kiểm tra xem đây có phải là người lạ mới không
                if track_id not in self.current_unknown_faces:
                    # Người lạ mới - thêm vào danh sách thông báo
                    new_unknown_faces.append(face_info)
                
                # Cập nhật thông tin người lạ này trong danh sách hiện tại
                self.current_unknown_faces[track_id] = face_info
        
        # Lọc CHÍNH XÁC các khuôn mặt đang xuất hiện trong khung hình hiện tại
        active_faces = {track_id: info for track_id, info in self.current_unknown_faces.items() 
                        if track_id in current_frame_tracks}
        
        # Gửi thông báo nếu có gương mặt lạ mới - GỬI NGAY LẬP TỨC không phụ thuộc cooldown
        if new_unknown_faces and active_faces:
            self._send_notification(camera_id, active_faces, has_new_face=True)
            # Reset thời gian cooldown sau khi gửi
            self.last_notification_time = current_time
            logger.info(f"Reset cooldown after sending new face notification - next periodic in {self.cooldown_period} seconds")
            
        # Gửi thông báo định kỳ nếu đến thời gian và không có gương mặt mới
        elif self.periodic_notification and active_faces and current_time - self.last_notification_time > self.cooldown_period:
            self._send_notification(camera_id, active_faces, has_new_face=False)
            self.last_notification_time = current_time
            logger.info(f"Sent periodic notification - next in {self.cooldown_period} seconds")
    
    def _clean_inactive_tracks(self, current_time):
        """Dọn dẹp các track_id không còn hoạt động trong thời gian dài."""
        # Danh sách các track_id cần xóa
        tracks_to_remove = []
        
        # Kiểm tra thời gian mỗi track_id
        for track_id, last_seen in list(self.track_last_seen.items()):
            if current_time - last_seen > self.track_timeout:
                tracks_to_remove.append(track_id)
        
        # Xóa các track_id hết hạn
        for track_id in tracks_to_remove:
            self.track_last_seen.pop(track_id, None)
            self.unknown_counter.pop(track_id, None)
            self.current_unknown_faces.pop(track_id, None)
            
            # THÊM MỚI: Cũng xóa khỏi danh sách đã nhận diện nếu có
            if track_id in self.recognized_tracks:
                self.recognized_tracks.pop(track_id, None)
            
        if tracks_to_remove:
            logger.debug(f"Removed {len(tracks_to_remove)} inactive tracks")
    
    # THÊM MỚI: Dọn dẹp các email thread đã hoàn thành
    def _clean_completed_email_threads(self):
        """Dọn dẹp các thread gửi email đã hoàn thành."""
        active_threads = []
        for thread in self.email_threads:
            if thread.is_alive():
                active_threads.append(thread)
        self.email_threads = active_threads
    
    def _send_notification(self, camera_id: str, active_faces: Dict, has_new_face: bool = False):
        """Gửi thông báo về các gương mặt lạ."""
        if not active_faces:
            return
            
        # Chuẩn bị dữ liệu
        images = []
        face_count = len(active_faces)
        new_face_count = sum(1 for face_info in active_faces.values() if face_info.get("is_new", False))
        
        for face_info in active_faces.values():
            if face_info["image_path"] and os.path.exists(face_info["image_path"]):
                images.append(face_info["image_path"])
                # Đánh dấu không còn là mặt mới nữa
                face_info["is_new"] = False
        
        # Loại thông báo
        notification_type = "new_face" if has_new_face else "periodic"
        
        # THAY ĐỔI: Gửi email trong một thread riêng để tránh lag
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
        
        if has_new_face:
            logger.info(f"New face notification queued with {face_count} unknown faces currently in frame at camera {camera_id}")
        else:
            logger.info(f"Periodic notification queued for {face_count} unknown faces currently in frame at camera {camera_id}")
    
    # THÊM MỚI: Phương thức gửi email trong thread riêng
    def _send_email_in_thread(self, camera_id, image_path, count, notification_type, group_data=None):
        """Gửi email trong một thread riêng để tránh lag."""
        email_thread = threading.Thread(
            target=self.send_email_notification,
            args=(camera_id, image_path, count, notification_type),
            kwargs={"group_data": group_data}
        )
        email_thread.daemon = True  # Đảm bảo thread sẽ kết thúc khi chương trình chính kết thúc
        email_thread.start()
        self.email_threads.append(email_thread)
        return email_thread
    
    def send_email_notification(
        self, 
        camera_id: str, 
        image_path: Optional[str], 
        count: int,
        notification_type: str = "new_face",  # "new_face" hoặc "periodic"
        face_info: Optional[Dict] = None,
        group_data: Optional[Dict] = None
    ):
        """Gửi email thông báo với hình ảnh đính kèm"""
        if not self.email_config:
            logger.warning("Email configuration not provided, skipping notification")
            return False
            
        try:
            sender = self.email_config.get('sender', '')
            receiver = self.email_config.get('receiver', '')
            smtp_server = self.email_config.get('smtp_server', '')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username', '')
            password = self.email_config.get('password', '')
            
            if not all([sender, receiver, smtp_server, username, password]):
                logger.warning("Incomplete email configuration")
                return False
            
            msg = MIMEMultipart()
            
            # Tiêu đề email dựa vào loại thông báo
            if notification_type == "new_face":
                new_count = group_data.get("new_people", 1) if group_data else 1
                total_count = group_data.get("total_people", count) if group_data else count
                if new_count > 1:
                    msg['Subject'] = f'CẢNH BÁO AN NINH: Phát hiện {new_count} người lạ mới (Camera {camera_id})'
                else:
                    msg['Subject'] = f'CẢNH BÁO AN NINH: Phát hiện người lạ mới (Camera {camera_id})'
            else:  # periodic
                msg['Subject'] = f'CẬP NHẬT AN NINH: {count} người lạ đang có mặt (Camera {camera_id})'
                
            msg['From'] = sender
            msg['To'] = receiver
            
            # Thêm nội dung
            current_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
            
            if notification_type == "new_face":
                # Cập nhật nội dung để hiển thị tất cả người lạ
                new_count = group_data.get("new_people", 1) if group_data else 1
                total_count = group_data.get("total_people", count) if group_data else count
                
                if new_count > 1:
                    text = f"""
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
                    text = f"""
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
            else:  # periodic
                text = f"""
                <html>
                <body>
                    <h2>CẬP NHẬT AN NINH</h2>
                    <p>Hiện có <strong>{count} người lạ</strong> đang xuất hiện tại camera {camera_id}.</p>
                    <p>Thời gian: {current_time}</p>
                    <p>Đây là thông báo định kỳ.</p>
                </body>
                </html>
                """
            msg.attach(MIMEText(text, 'html'))
            
            # Đính kèm hình ảnh chính
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(image_path)}"')
                    msg.attach(image)
            
            # Đính kèm các hình ảnh phụ nếu có
            if group_data and group_data.get("all_images"):
                for i, img_path in enumerate(group_data["all_images"][1:]):  # Bỏ qua ảnh đầu tiên đã đính kèm
                    if os.path.exists(img_path):
                        with open(img_path, 'rb') as f:
                            img_data = f.read()
                            image = MIMEImage(img_data)
                            image.add_header('Content-Disposition', 
                                            f'attachment; filename="{os.path.basename(img_path)}"')
                            msg.attach(image)
            
            # Gửi email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent for camera {camera_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False