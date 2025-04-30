import streamlit as st
import cv2
import os
import time
import pickle
import shutil
import subprocess
from PIL import Image
import numpy as np
import tempfile

# First Streamlit command must be set_page_config
st.set_page_config(page_title="Face Recognition System", layout="wide")

# Import your existing modules
from face_detector import FaceDetector
import face_recognizer
import face_tracker
import anti_spoofing
import mask_detection
import preprocessing
from utils.helpers import draw_bbox, draw_bbox_info
from config import *

# Khởi tạo Face Detector một lần khi app khởi động
@st.cache_resource
def load_face_detector():
    scrfd_model = FaceDetector.load_models()
    detector = FaceDetector(
        preloaded_model=scrfd_model,
        detection_threshold=FACE_DETECTION_THRESHOLD
    )
    return detector

# Tải face detector
face_detector_instance = load_face_detector()

st.title("Face Recognition System")

# Khởi tạo session state cho quá trình chụp ảnh
if 'current_pose_idx' not in st.session_state:
    st.session_state.current_pose_idx = 0

if 'captured_images' not in st.session_state:
    st.session_state.captured_images = {}

if 'person_name' not in st.session_state:
    st.session_state.person_name = ""

def main():
    st.sidebar.title("Options")
    app_mode = st.sidebar.selectbox("Choose the app mode", 
        ["About", "Recognize Face", "Add New Person", "View Database"])
    
    if app_mode == "About":
        st.markdown("# Face Recognition System")
        st.markdown("This application demonstrates face recognition with mask detection and anti-spoofing capabilities.")
        
    elif app_mode == "Recognize Face":
        st.markdown("## Face Recognition")
        run_recognition()
        
    elif app_mode == "Add New Person":
        st.markdown("## Add New Person")
        add_person()
        
    elif app_mode == "View Database":
        st.markdown("## Database")
        view_database()

def run_recognition():
    # Camera input options
    source = st.sidebar.radio("Select Source", ["Webcam", "Upload Image"])
    
    if source == "Webcam":
        run_webcam_recognition()
    else:
        run_image_recognition()

def run_webcam_recognition():
    stframe = st.empty()
    start_btn = st.button("Start Recognition")
    stop_btn = st.button("Stop")
    
    if start_btn:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.error("Cannot read from webcam")
                break
                
            # Process frame using your existing modules
            face_boxes, _, _ = face_detector_instance.process_frame(frame)
            
            # Draw bounding boxes and info
            result_img = frame.copy()
            for box in face_boxes:
                draw_bbox_info(
                    result_img,
                    bbox=box,
                    similarity=1.0,
                    name="Face",
                    color=(0, 255, 0)
                )
            
            # Display the frame
            stframe.image(result_img, channels="BGR", use_column_width=True)
            
        cap.release()

def run_image_recognition():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Convert PIL Image to OpenCV format
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Process with your modules
        face_boxes, _, _ = face_detector_instance.process_frame(frame)
        
        # Draw detection results
        result_img = frame.copy()
        for box in face_boxes:
            draw_bbox_info(
                result_img,
                bbox=box,
                similarity=1.0,
                name="Face",
                color=(0, 255, 0)
            )
            
        # Convert back to RGB for display
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img_rgb, caption="Detection Results", use_column_width=True)
        
        st.write(f"Detected {len(face_boxes)} faces")

def add_person():
    # Define pose information
    poses = ["front", "left", "right", "down"]
    pose_instructions = {
        "front": "Look straight at the camera",
        "left": "Turn your face to the left",
        "right": "Turn your face to the right",
        "down": "Tilt your head down"
    }
    
    # Get person name if not already set
    if not st.session_state.person_name:
        st.subheader("Thêm người mới vào cơ sở dữ liệu")
        person_name = st.text_input("Nhập tên người")
        
        if person_name:
            # Check if person already exists
            person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
            if os.path.exists(person_dir):
                st.warning(f"Người dùng '{person_name}' đã tồn tại. Dữ liệu hiện tại sẽ bị ghi đè!")
                overwrite = st.radio("Bạn có muốn tiếp tục?", ["Không", "Có"])
                if overwrite == "Có":
                    st.session_state.person_name = person_name
            else:
                st.session_state.person_name = person_name
    
    # If name is set, proceed with image capture
    if st.session_state.person_name:
        st.subheader(f"Thêm ảnh cho người dùng: {st.session_state.person_name}")
        
        # Get current pose
        current_pose = poses[st.session_state.current_pose_idx]
        
        # Show instructions for current pose
        with st.container():
            st.subheader(f"Góc chụp hiện tại: **{current_pose.upper()}**")
            st.info(f"Hướng dẫn: {pose_instructions[current_pose]}")
            
            # Upload image for current pose
            uploaded_file = st.file_uploader(f"Tải lên ảnh cho góc chụp {current_pose}", 
                                            type=["jpg", "jpeg", "png"], 
                                            key=f"upload_{current_pose}")
            
            # Process uploaded image
            if uploaded_file:
                # Create temp directory
                temp_dir = os.path.join("temp_captures", st.session_state.person_name, "no_mask")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Convert and save uploaded image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Convert RGB to BGR for OpenCV if necessary
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Save image
                save_path = os.path.join(temp_dir, f"{current_pose}.jpg")
                cv2.imwrite(save_path, img_array, [cv2.IMWRITE_JPEG_QUALITY, 95])
                st.session_state.captured_images[current_pose] = save_path
                
                # Display uploaded image
                st.image(image, caption=f"{current_pose} image", width=300)
                
                # Add buttons to confirm or retake
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Chụp lại", key=f"retake_{current_pose}"):
                        if current_pose in st.session_state.captured_images:
                            del st.session_state.captured_images[current_pose]
                
                with col2:
                    if st.button("Tiếp tục", key=f"next_{current_pose}"):
                        if st.session_state.current_pose_idx < len(poses) - 1:
                            st.session_state.current_pose_idx += 1
                            st.experimental_rerun()
        
        # Display progress
        progress_text = f"Đã chụp: {len(st.session_state.captured_images)}/{len(poses)} ảnh"
        st.progress(len(st.session_state.captured_images) / len(poses))
        st.caption(progress_text)
        
        # Display captured images
        if st.session_state.captured_images:
            st.subheader("Ảnh đã chụp:")
            cols = st.columns(len(poses))
            
            for i, pose in enumerate(poses):
                if pose in st.session_state.captured_images:
                    img_path = st.session_state.captured_images[pose]
                    if os.path.exists(img_path):
                        with cols[i]:
                            img = Image.open(img_path)
                            st.image(img, caption=f"{pose}", width=150)
        
        # Process all images when complete
        if len(st.session_state.captured_images) == len(poses):
            if st.button("Hoàn thành và xử lý tất cả ảnh"):
                with st.spinner("Đang xử lý..."):
                    # Process captured faces
                    process_status = process_captured_faces(st.session_state.person_name, st.session_state.captured_images)
                    
                    if process_status:
                        # Reset state for new entry
                        st.session_state.current_pose_idx = 0
                        st.session_state.captured_images = {}
                        st.session_state.person_name = ""
                        
                        st.balloons()
                        st.success("Đã thêm người dùng mới thành công!")
                        
                        # Add option to create embeddings
                        if st.button("Tạo Embeddings Ngay"):
                            try:
                                with st.spinner("Đang tạo embeddings..."):
                                    result = subprocess.run(["python", "create_embeddings.py"], 
                                                         capture_output=True, text=True, check=True)
                                    st.success("Đã tạo embeddings thành công!")
                                    st.code(result.stdout)
                            except subprocess.CalledProcessError as e:
                                st.error(f"Lỗi khi tạo embeddings: {e}")
                                st.code(e.stderr)
                    else:
                        st.error("Có lỗi xảy ra trong quá trình xử lý ảnh!")

def process_captured_faces(person_name, captured_images):
    try:
        # Create directories
        person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
        no_mask_dir = os.path.join(person_dir, "no_mask")
        
        # Check if person exists
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
        
        # Create directories
        os.makedirs(no_mask_dir, exist_ok=True)
        
        # Copy captured images
        for pose, src_path in captured_images.items():
            if os.path.exists(src_path):
                dst_path = os.path.join(no_mask_dir, f"{pose}.jpg")
                shutil.copy2(src_path, dst_path)
        
        # Create masked versions using ImageProcessor from add_person.py
        with_mask_dir = os.path.join(person_dir, "with_mask")
        os.makedirs(with_mask_dir, exist_ok=True)
        
        # Import ImageProcessor from add_person
        from add_person import ImageProcessor
        
        # Create masked versions
        mask_types = ["surgical", "N95", "cloth"]
        success, mask_count = ImageProcessor.create_masked_dataset(person_name, no_mask_dir, mask_types)
        
        # Show results
        if success:
            st.success(f"""
            ### Thông tin Dataset:
            - Người dùng: {person_name}
            - Số ảnh không khẩu trang: {len(captured_images)}
            - Số ảnh có khẩu trang: {mask_count}
            - Loại khẩu trang: {', '.join(mask_types)}
            
            Bạn cần chạy create_embeddings.py để hoàn thành quá trình.
            """)
            return True
        else:
            st.error("Không thể tạo ảnh có khẩu trang.")
            return False
            
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
        return False

def view_database():
    """Hiển thị cơ sở dữ liệu khuôn mặt."""
    st.subheader("Face Recognition Database")
    
    # Kiểm tra các file embedding tồn tại
    if not os.path.exists(EMBEDDINGS_PATH):
        st.warning(f"Không tìm thấy file embedding tại {EMBEDDINGS_PATH}")
        
        # Thêm kiểm tra thư mục hình ảnh
        if os.path.exists(DATASET_IMAGES_DIR):
            persons = [d for d in os.listdir(DATASET_IMAGES_DIR) if os.path.isdir(os.path.join(DATASET_IMAGES_DIR, d))]
            if persons:
                st.info(f"Tìm thấy {len(persons)} người trong thư mục hình ảnh nhưng chưa tạo embeddings.")
                st.write("Danh sách người: " + ", ".join(persons))
                
                with st.expander("Xem hướng dẫn tạo embeddings"):
                    st.code("python create_embeddings.py", language="bash")
        return
        
    try:
        # Tải dữ liệu embeddings
        with open(EMBEDDINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            
        # Tải thêm thông tin về nguồn ảnh nếu có
        source_info = {}
        if os.path.exists(EMBEDDINGS_SOURCE_INFO_PATH):
            try:
                with open(EMBEDDINGS_SOURCE_INFO_PATH, 'rb') as f:
                    source_info = pickle.load(f)
            except Exception as e:
                st.warning(f"Không thể đọc file source_info: {e}")
        
        # Xác định cấu trúc dữ liệu
        if isinstance(data, dict) and "embeddings_db" in data:
            embeddings_db = data["embeddings_db"]
            model_info = data.get("model", "Unknown")
            augmentation = "Có" if data.get("with_augmentation", False) else "Không"
        else:
            embeddings_db = data
            model_info = "Unknown"
            augmentation = "Không xác định"
            
        # Hiển thị thông tin chung về database
        st.info(f"""
        **Thông tin Database:**
        - Model sử dụng: {model_info}
        - Sử dụng augmentation: {augmentation}
        - Số người trong database: {len(embeddings_db)}
        - File embeddings: {os.path.basename(EMBEDDINGS_PATH)}
        - Index embeddings: {os.path.basename(EMBEDDINGS_HNSW_PATH)}
        """)
        
        # Hiển thị thông tin chi tiết của từng người trong database
        st.subheader("Danh sách người trong database:")
        
        # Tạo bảng thông tin tóm tắt
        summary_data = {
            "Người": [],
            "Ảnh không khẩu trang": [],
            "Ảnh có khẩu trang": [],
            "Combined embeddings": []
        }
        
        for person, data in embeddings_db.items():
            no_mask_count = len([p for p in data["no_mask"] if p != "combined"])
            with_mask_count = len([p for p in data["with_mask"] if p != "combined"])
            
            combined_count = 0
            if "combined" in data["no_mask"]:
                combined_count += 1
            if "combined" in data["with_mask"]:
                combined_count += 1
                
            summary_data["Người"].append(person)
            summary_data["Ảnh không khẩu trang"].append(no_mask_count)
            summary_data["Ảnh có khẩu trang"].append(with_mask_count) 
            summary_data["Combined embeddings"].append(combined_count)
        
        st.dataframe(summary_data)
        
        # Chức năng xem chi tiết cho từng người
        st.subheader("Chi tiết theo người:")
        selected_person = st.selectbox("Chọn người để xem chi tiết:", list(embeddings_db.keys()))
        
        if selected_person:
            person_data = embeddings_db[selected_person]
            
            # Hiển thị thông tin chi tiết về các pose
            st.markdown(f"### {selected_person}")
            
            # Tab cho không khẩu trang và có khẩu trang
            tab1, tab2 = st.tabs(["Không khẩu trang", "Có khẩu trang"])
            
            with tab1:
                if person_data["no_mask"]:
                    st.write(f"Số lượng embeddings: {len(person_data['no_mask'])}")
                    
                    # Hiển thị các pose cụ thể
                    poses = list(person_data["no_mask"].keys())
                    st.json(poses)
                    
                    # Hiển thị ảnh gốc nếu có
                    if os.path.exists(DATASET_IMAGES_DIR):
                        person_dir = os.path.join(DATASET_IMAGES_DIR, selected_person)
                        no_mask_dir = os.path.join(person_dir, "no_mask")
                        
                        if os.path.exists(no_mask_dir):
                            st.subheader("Ảnh gốc")
                            pose_images = [f for f in os.listdir(no_mask_dir) if f.endswith(('.jpg', '.png'))]
                            
                            if pose_images:
                                cols = st.columns(min(4, len(pose_images)))
                                for i, img_name in enumerate(pose_images):
                                    img_path = os.path.join(no_mask_dir, img_name)
                                    cols[i % 4].image(img_path, caption=img_name, width=150)
                            else:
                                st.write("Không tìm thấy ảnh gốc")
                else:
                    st.write("Không có dữ liệu khuôn mặt không khẩu trang")
                    
            with tab2:
                if person_data["with_mask"]:
                    st.write(f"Số lượng embeddings: {len(person_data['with_mask'])}")
                    
                    # Hiển thị các pose cụ thể
                    poses = list(person_data["with_mask"].keys())
                    st.json(poses)
                    
                    # Hiển thị ảnh có khẩu trang nếu có
                    if os.path.exists(DATASET_IMAGES_DIR):
                        person_dir = os.path.join(DATASET_IMAGES_DIR, selected_person)
                        with_mask_dir = os.path.join(person_dir, "with_mask")
                        
                        if os.path.exists(with_mask_dir):
                            st.subheader("Ảnh có khẩu trang")
                            mask_images = [f for f in os.listdir(with_mask_dir) if f.endswith(('.jpg', '.png'))]
                            
                            if mask_images:
                                mask_types = set()
                                for img_name in mask_images:
                                    if "_" in img_name:
                                        mask_type = img_name.split("_")[-1].split(".")[0]
                                        mask_types.add(mask_type)
                                
                                # Display images by mask type
                                for mask_type in mask_types:
                                    st.subheader(f"Loại khẩu trang: {mask_type}")
                                    type_images = [f for f in mask_images if f.endswith(f"_{mask_type}.jpg")]
                                    
                                    cols = st.columns(min(4, len(type_images)))
                                    for i, img_name in enumerate(type_images):
                                        img_path = os.path.join(with_mask_dir, img_name)
                                        cols[i % 4].image(img_path, caption=img_name.split("_")[0], width=150)
                            else:
                                st.write("Không tìm thấy ảnh có khẩu trang")
                else:
                    st.write("Không có dữ liệu khuôn mặt có khẩu trang")
                
    except Exception as e:
        st.error(f"Không thể đọc database: {str(e)}")
        st.warning("Database có thể bị hỏng hoặc không hợp lệ.")

if __name__ == "__main__":
    main()