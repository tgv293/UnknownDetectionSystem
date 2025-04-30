import os
import sys
import tensorflow as tf
import torch
import logging

# Thêm thư mục gốc dự án vào path để giải quyết vấn đề import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Thiết lập logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("security_system")

# Đường dẫn thư mục
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
EMAILCONFIG_DIR = os.path.join(BASE_DIR, "email_config.json")

# Đường dẫn cho dataset hình ảnh
DATASET_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
os.makedirs(DATASET_IMAGES_DIR, exist_ok=True)

# Tạo thư mục embeddings
EMBEDDINGS_DIR = os.path.join(DATASET_DIR, "embeddings")
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Đường dẫn file embedding và chỉ mục
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
EMBEDDINGS_SOURCE_INFO_PATH = os.path.join(EMBEDDINGS_DIR, "face_embeddings_source_info.pkl")
EMBEDDINGS_HNSW_PATH = os.path.join(EMBEDDINGS_DIR, "face_embeddings_hnsw.bin")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Ngưỡng
COSINE_SIMILARITY_THRESHOLD = 0.6  # Ngưỡng nhận diện khuôn mặt
FACE_DETECTION_THRESHOLD = 0.5     # Ngưỡng phát hiện khuôn mặt
MASK_DETECTION_THRESHOLD = 0.5      # Ngưỡng phát hiện khẩu trang

# Cấu hình GPU cho cả TensorFlow và PyTorch
def configure_gpu():
    """Cấu hình GPU cho hiệu suất tối ưu với TensorFlow và PyTorch"""
    try:
        # Cấu hình TensorFlow GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logger.info("TensorFlow GPU memory growth enabled")
        else:
            logger.info("No TensorFlow GPU found, using CPU")

        # Kiểm tra PyTorch GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            logger.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("PyTorch using CPU")

        return device
    except Exception as e:
        logger.warning(f"GPU configuration error: {e}")
        return torch.device('cpu')

# Khởi tạo thiết bị mặc định
DEVICE = configure_gpu()