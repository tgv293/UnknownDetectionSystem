import cv2
import numpy as np
from typing import Optional, Tuple, Union
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

# Kích thước chuẩn
FACE_RECOGNITION_SIZE = 112  # ArcFace
MASK_DETECTION_SIZE = 224    # MobileNetV2

def smart_resize(image: np.ndarray, target_size: Union[Tuple[int, int], int],
                 force_rgb: bool = False) -> Optional[np.ndarray]:
    """Resize ảnh với phương pháp nội suy thích ứng và đảm bảo định dạng màu"""
    if image is None:
        return None

    # Xử lý đầu vào
    target_width, target_height = target_size if isinstance(
        target_size, tuple) else (target_size, target_size)

    # Convert to RGB if requested
    if force_rgb and image.shape[2] >= 3:
        if image[0, 0, 0] > image[0, 0, 2]:  # Simple BGR check
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Kiểm tra nếu không cần resize
    img_h, img_w = image.shape[:2]
    if img_h == target_height and img_w == target_width:
        return image

    # Chọn phương pháp nội suy dựa trên kích thước
    interp = cv2.INTER_CUBIC if img_w < target_width or img_h < target_height else cv2.INTER_AREA

    # Thực hiện resize
    return cv2.resize(image, (target_width, target_height), interpolation=interp)

def resize_for_face_recognition(image: np.ndarray) -> Optional[np.ndarray]:
    """Resize ảnh cho nhận dạng khuôn mặt (ArcFace)"""
    return smart_resize(image, FACE_RECOGNITION_SIZE) if image is not None else None

def resize_for_mask_detection(image: np.ndarray, for_mobilenet: bool = True) -> Optional[np.ndarray]:
    """Resize và chuyển sang RGB ảnh cho phát hiện khẩu trang
    
    Args:
        image: Ảnh đầu vào
        for_mobilenet: Nếu True, chỉ resize và chuyển RGB, không chuẩn hóa
                      (chuẩn hóa sẽ được xử lý bởi preprocess_input sau này)
    
    Returns:
        Ảnh đã resize và chuyển RGB, chưa chuẩn hóa
    """
    if image is None:
        return None
    
    # Resize và chuyển sang RGB
    resized = smart_resize(image, MASK_DETECTION_SIZE, force_rgb=True)
    
    # Không chuẩn hóa ở đây, để xử lý bằng preprocess_input sau này
    return resized

def preprocess_for_mask_detection(image: np.ndarray) -> Optional[np.ndarray]:
    """Tiền xử lý đầy đủ cho model phát hiện khẩu trang MobileNetV2
    
    Hàm này thực hiện toàn bộ quy trình tiền xử lý cần thiết cho MobileNetV2:
    1. Resize và chuyển RGB
    2. Chuyển sang array
    3. Chuẩn hóa với preprocess_input
    
    Args:
        image: Ảnh khuôn mặt đầu vào (BGR format)
        
    Returns:
        Ảnh đã qua tiền xử lý, sẵn sàng cho batch processing
    """
    if image is None:
        return None
        
    # Resize và chuyển sang RGB
    resized = resize_for_mask_detection(image)
    
    # Chuyển sang array (tương đương với img_to_array)
    image_array = np.asarray(resized, dtype=np.float32)
    
    # Sử dụng preprocess_input từ MobileNetV2 để chuẩn hóa đúng cách
    preprocessed = preprocess_input(image_array)
    
    return preprocessed