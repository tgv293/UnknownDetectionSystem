import os
import cv2
import numpy as np
import time
import shutil
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DATASET_IMAGES_DIR, MODEL_DIR, DEVICE, FACE_DETECTION_THRESHOLD
from static.models.face_detection.scrfd import SCRFD
from utils.helpers import norm_crop_image

# ==== Configuration Constants ====

# Pose definitions with corresponding angles and instructions
POSE_ANGLES = {
    "front": (0, 0),
    "left": (-30, 0),
    "right": (30, 0),
    "down": (0, -20)
}

POSE_INSTRUCTIONS = {
    "front": "Look straight at the camera",
    "left": "Turn your face to the left",
    "right": "Turn your face to the right",
    "down": "Tilt your head down"
}

# Quality issue messages
QUALITY_WARNINGS = {
    "blur": "Image is blurry! Please maintain your position.",
    "dark": "Too dark! Please increase lighting.",
    "bright": "Too bright! Please reduce lighting.",
    "multiple_faces": "Multiple faces detected! Please keep only one face in the frame.",
    "no_face": "No face detected! Please position your face in the center of the frame.",
    "outside_roi": "Please place your face in the center capture area.",
    "wrong_angle": "Incorrect face angle! Please follow the instructions."
}

# Quality thresholds
BLUR_THRESHOLD = 100
DARK_THRESHOLD = 50
BRIGHT_THRESHOLD = 200
FACE_CONFIDENCE_THRESHOLD = FACE_DETECTION_THRESHOLD
POSITION_REQUIRED_TIME = 2.0
ANGLE_TOLERANCE = 15
VERIFICATION_TIME = 1.0
COUNTDOWN_TIME = 2.0

# ==== State Management ====
# Stores state for ongoing person additions
active_sessions = {}

# Global model instance for lazy loading
scrfd_detector = None


# ==== Data Models ====
class PersonAddRequest(BaseModel):
    name: str

class PoseValidationRequest(BaseModel):
    session_id: str
    pose: str
    
class ValidationResult(BaseModel):
    valid: bool
    issues: List[str] = []
    pose_data: dict = {}

class PoseResult(BaseModel):
    success: bool
    message: str
    pose: str = ""
    image_path: str = ""
    next_pose: str = ""
    session_id: str = ""
    completed: bool = False

class SessionInfo(BaseModel):
    session_id: str
    person_name: str
    current_pose: str
    completed_poses: List[str]
    remaining_poses: List[str]
    temp_dir: str
    backup_created: bool
    existing_person: bool


# ==== Face Detection Module ====
class FaceDetector:
    @staticmethod
    def get_detector():
        """Lazy load SCRFD model to save memory until needed"""
        global scrfd_detector
        if scrfd_detector is None:
            try:
                model_path = os.path.join(MODEL_DIR, "face_detection", "det_34g.onnx")
                scrfd_detector = SCRFD(
                    model_path=model_path,
                    input_size=(640, 640),
                    conf_thres=FACE_CONFIDENCE_THRESHOLD
                )
                print(f"SCRFD model loaded: {model_path}")
            except Exception as e:
                print(f"SCRFD loading error: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to load face detection model: {str(e)}")
        return scrfd_detector
    
    @staticmethod
    def detect_faces(frame):
        """Detect faces in an image and return face information"""
        detector = FaceDetector.get_detector()
        
        try:
            boxes, landmarks = detector.detect(frame)
            
            faces_info = []
            for i, box in enumerate(boxes):
                if box[4] > FACE_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2, score = box
                    
                    # Convert landmarks to dictionary format if available
                    lm_dict = None
                    if landmarks is not None and i < len(landmarks):
                        lm_dict = FaceDetector._convert_landmarks_to_dict(landmarks[i])
                        
                    faces_info.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(score),
                        'landmarks': landmarks[i] if landmarks is not None and i < len(landmarks) else None,
                        'landmarks_dict': lm_dict
                    })
                    
            return faces_info
        
        except Exception as e:
            print(f"Face detection error: {e}")
            raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")
    
    @staticmethod
    def _convert_landmarks_to_dict(landmarks):
        """Convert landmark array to named dictionary format"""
        if landmarks is None or landmarks.shape[0] != 5:
            return None
            
        return {
            'left_eye': landmarks[0],
            'right_eye': landmarks[1],
            'nose': landmarks[2],
            'mouth_left': landmarks[3],
            'mouth_right': landmarks[4]
        }


# ==== Face Analysis Module ====
class FaceAnalysis:
    @staticmethod
    def get_roi(frame):
        """Get region of interest in center of frame"""
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = min(w, h) // 3
        return (center_x - roi_size, center_y - roi_size, center_x + roi_size, center_y + roi_size)

    @staticmethod
    def check_face_angle(landmarks, target_angle, tolerance=ANGLE_TOLERANCE):
        """Check if face angle matches the target angle within tolerance"""
        try:
            # Extract facial landmarks
            left_eye = np.array(landmarks['left_eye'])
            right_eye = np.array(landmarks['right_eye'])
            nose = np.array(landmarks['nose'])
            mouth_left = np.array(landmarks['mouth_left'])
            mouth_right = np.array(landmarks['mouth_right'])

            # Calculate facial geometry
            eye_center = (left_eye + right_eye) / 2
            mouth_center = (mouth_left + mouth_right) / 2
            eye_width = np.linalg.norm(right_eye - left_eye)
            face_height = np.linalg.norm(eye_center - mouth_center)

            # Estimate angles based on facial landmarks
            eye_to_nose = nose - eye_center
            yaw = (eye_to_nose[0] / eye_width) * 60

            nose_to_mouth_vertical = (mouth_center[1] - nose[1]) / face_height
            pitch = (nose_to_mouth_vertical - 0.5) * 60

            # Compare with target angles using appropriate tolerances
            target_yaw, target_pitch = target_angle
            yaw_tol = tolerance * 0.8 if abs(target_yaw) <= 15 else tolerance
            pitch_tol = tolerance * 0.8 if abs(target_pitch) <= 10 else tolerance

            return (
                abs(yaw - target_yaw) <= yaw_tol and abs(pitch - target_pitch) <= pitch_tol
            ), (yaw, pitch)
        except Exception as e:
            print(f"Face angle calculation error: {e}")
            return False, (0, 0)

    @staticmethod
    def check_image_quality(image, face_info=None, current_pose=None):
        """Check image for quality issues like blur, brightness, and face positioning"""
        issues = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check basic image quality
        if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
            issues.append("blur")

        brightness = np.mean(gray)
        if brightness < DARK_THRESHOLD:
            issues.append("dark")
        elif brightness > BRIGHT_THRESHOLD:
            issues.append("bright")

        # Skip face checks if no face info provided
        if not face_info:
            return issues

        # Check face presence and positioning
        if len(face_info) == 0:
            issues.append("no_face")
            return issues
            
        # Get region of interest for face positioning
        roi = FaceAnalysis.get_roi(image)
        roi_x1, roi_y1, roi_x2, roi_y2 = roi

        # Find faces inside the region of interest
        faces_in_roi = []
        for idx, face in enumerate(face_info):
            x1, y1, x2, y2 = face['bbox']
            if (roi_x1 <= x1 and x2 <= roi_x2 and 
                roi_y1 <= y1 and y2 <= roi_y2):
                faces_in_roi.append(idx)

        # Check face positioning issues
        if len(faces_in_roi) == 0:
            issues.append("outside_roi")
        elif len(faces_in_roi) > 1:
            issues.append("multiple_faces")
        elif current_pose:  # Check face angle for the pose
            primary_face = face_info[faces_in_roi[0]]
            landmarks_dict = primary_face['landmarks_dict']
            valid_angle, _ = FaceAnalysis.check_face_angle(
                landmarks_dict, POSE_ANGLES[current_pose]
            )
            if not valid_angle:
                issues.append("wrong_angle")
        
        return issues


# ==== Image Processing Module ====
class ImageProcessor:
    @staticmethod
    def capture_image(frame, faces_info, current_pose, temp_dir):
        """Extract and align face using facial landmarks"""
        try:
            if not faces_info or len(faces_info) == 0:
                raise ValueError("No faces detected")
            
            # Get the first face for processing
            face_info = faces_info[0]
            
            # Check if landmarks exist
            if face_info['landmarks'] is None:
                raise ValueError("No facial landmarks detected")
            
            # Apply normalization crop for consistent face alignment
            face_img = norm_crop_image(
                frame, 
                face_info['landmarks'],
                image_size=112,
                mode='arcface'
            )
            
            # Save the extracted face
            save_path = os.path.join(temp_dir, f"{current_pose}.jpg")
            cv2.imwrite(save_path, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return save_path
                
        except Exception as e:
            print(f"Error in face extraction: {e}")
            raise ValueError(f"Failed to capture face: {str(e)}")
    
    @staticmethod
    def create_masked_dataset(person_name, no_mask_dir, mask_types=None):
        """Generate masked versions of face images using MaskTheFace"""
        try:
            mask_types = mask_types or ["surgical"]
            
            # Setup directories
            person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
            with_mask_dir = os.path.join(person_dir, "with_mask")
            os.makedirs(with_mask_dir, exist_ok=True)

            # Find MaskTheFace tool
            script_dir = os.path.dirname(os.path.abspath(__file__))
            masktheface_dir = os.path.join(script_dir, "MaskTheFace")
            original_dir = os.getcwd()
            
            generated_count = 0

            for mask_type in mask_types:
                try:
                    # Run MaskTheFace for this mask type
                    import subprocess
                    os.chdir(masktheface_dir)
                    cmd = [
                        "python", "mask_the_face.py",
                        "--path", no_mask_dir,
                        "--mask_type", mask_type,
                        "--verbose"
                    ]

                    # Use timeout to prevent hanging processes
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                    if result.returncode != 0:
                        print(f"Error running MaskTheFace: {result.stderr}")
                        continue

                    # Process generated masked images
                    masked_dir = f"{no_mask_dir}_masked"
                    if os.path.exists(masked_dir):
                        for pose_file in os.listdir(masked_dir):
                            if pose_file.endswith(('.jpg', '.png')):
                                base_name = os.path.splitext(pose_file)[0]
                                if "_" in base_name:
                                    base_name = base_name.split("_masked")[0]
                                
                                # Copy masked image to dataset
                                src = os.path.join(masked_dir, pose_file)
                                img = cv2.imread(src)
                                if img is not None:
                                    dst = os.path.join(with_mask_dir, f"{base_name}_{mask_type}.jpg")
                                    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                    generated_count += 1

                        # Clean up temporary masked directory
                        shutil.rmtree(masked_dir, ignore_errors=True)
                        
                except subprocess.TimeoutExpired:
                    print(f"Timeout while creating masks with {mask_type}")
                    # Try to kill any hanging processes
                    try:
                        import psutil
                        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                            if 'python' in proc.info['name'] and 'mask_the_face.py' in ' '.join(proc.info['cmdline']):
                                proc.kill()
                    except:
                        pass
                finally:
                    os.chdir(original_dir)
                    
            # Ensure we're back in the original directory
            if os.getcwd() != original_dir:
                os.chdir(original_dir)
                
            return True, generated_count
            
        except Exception as e:
            print(f"Error creating masked dataset: {e}")
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return False, 0


# ==== Face Capture Module ====
class FaceCapture:
    @staticmethod
    def get_ordered_poses():
        """Get standard order of poses for consistent capture process"""
        return ["front", "left", "right", "down"]
    
    @staticmethod
    def save_to_dataset(person_name, captured_images):
        """Save captured images to permanent dataset location"""
        try:
            # Use DATASET_IMAGES_DIR
            person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
            
            # Remove existing directory completely if it exists
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
            
            # Create fresh directories
            no_mask_dir = os.path.join(person_dir, "no_mask")
            os.makedirs(no_mask_dir, exist_ok=True)

            # Copy captured images
            for pose, src_path in captured_images.items():
                shutil.copy2(src_path, os.path.join(no_mask_dir, f"{pose}.jpg"))
            return True
        except Exception as e:
            print(f"Error saving to dataset: {e}")
            return False


# ==== Session Management ====
def create_session(person_name: str) -> str:
    """Create a new session for adding a person"""
    session_id = str(uuid.uuid4())
    
    # Check if person already exists
    person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
    existing_person = os.path.exists(person_dir)
    
    # Create temp directory
    temp_dir = os.path.join("temp_captures", person_name, "no_mask")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create backup directory for existing person
    backup_dir = None
    if existing_person:
        backup_dir = os.path.join("temp_captures", f"{person_name}_backup_{int(time.time())}")
    
    # Initialize session
    active_sessions[session_id] = {
        "person_name": person_name,
        "temp_dir": temp_dir,
        "backup_dir": backup_dir,
        "existing_person": existing_person,
        "backup_created": False,
        "completed_poses": {},
        "current_pose": "front",
        "poses": FaceCapture.get_ordered_poses(),
        "start_time": time.time(),
        "last_activity": time.time()
    }
    
    return session_id

def get_session(session_id: str) -> dict:
    """Get session data or raise 404 if not found"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Update last activity time
    active_sessions[session_id]["last_activity"] = time.time()
    return active_sessions[session_id]

def update_session(session_id: str, updates: dict) -> dict:
    """Update session data"""
    session = get_session(session_id)
    for key, value in updates.items():
        session[key] = value
    session["last_activity"] = time.time()
    return session

def cleanup_session(session_id: str, success: bool = False):
    """Clean up session resources based on outcome"""
    if session_id not in active_sessions:
        return
    
    session = active_sessions[session_id]
    person_name = session["person_name"]
    person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
    parent_temp_dir = os.path.dirname(session["temp_dir"]) if session["temp_dir"] else None
    
    try:
        # Handle successful completion
        if success:
            if session["backup_dir"] and os.path.exists(session["backup_dir"]):
                shutil.rmtree(session["backup_dir"])
        # Restore backup if failed
        elif session["existing_person"] and session["backup_created"]:
            if session["backup_dir"] and os.path.exists(session["backup_dir"]):
                if os.path.exists(person_dir):
                    shutil.rmtree(person_dir)
                shutil.copytree(session["backup_dir"], person_dir)
                shutil.rmtree(session["backup_dir"])
        
        # Clean up temp files
        if session["temp_dir"] and os.path.exists(session["temp_dir"]):
            shutil.rmtree(session["temp_dir"])
        
        # Clean up empty parent directory
        if parent_temp_dir and os.path.exists(parent_temp_dir) and not os.listdir(parent_temp_dir):
            shutil.rmtree(parent_temp_dir)
            
        # Clean up masked temp dir if it exists
        temp_masked_dir = f"{session['temp_dir']}_masked"
        if os.path.exists(temp_masked_dir):
            shutil.rmtree(temp_masked_dir)
            
    except Exception as e:
        print(f"Error during session cleanup: {e}")
        
    finally:
        # Always remove session from active_sessions
        active_sessions.pop(session_id, None)

def cleanup_stale_sessions(max_age: int = 3600):
    """Clean up sessions that have been inactive for over an hour"""
    current_time = time.time()
    for session_id in list(active_sessions.keys()):
        session = active_sessions.get(session_id)
        if session and (current_time - session["last_activity"]) > max_age:
            cleanup_session(session_id, False)


# ==== API Router Setup ====
router = APIRouter(
    prefix="/api/person",
    tags=["person_management"],
    responses={404: {"description": "Not found"}},
)

@router.post("/start", response_model=SessionInfo)
async def start_person_addition(data: PersonAddRequest):
    """Initialize a new person addition session"""
    cleanup_stale_sessions()  # Clean up old sessions
    
    person_name = data.name.strip()
    if not person_name:
        raise HTTPException(status_code=400, detail="Person name cannot be empty")
    
    # Create session
    session_id = create_session(person_name)
    session = get_session(session_id)
    
    return SessionInfo(
        session_id=session_id,
        person_name=session["person_name"],
        current_pose=session["current_pose"],
        completed_poses=list(session["completed_poses"].keys()),
        remaining_poses=[p for p in session["poses"] if p not in session["completed_poses"]],
        temp_dir=session["temp_dir"],
        backup_created=session["backup_created"],
        existing_person=session["existing_person"]
    )

@router.post("/validate_image", response_model=ValidationResult)
async def validate_image(file: UploadFile = File(...), session_id: str = Form(...), pose: str = Form(...)):
    """Validate an image frame for face quality, positioning and angle"""
    session = get_session(session_id)
    
    # Read and decode the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return ValidationResult(valid=False, issues=["invalid_image"])
    
    # Detect faces
    try:
        faces_info = FaceDetector.detect_faces(frame)
    except Exception as e:
        return ValidationResult(valid=False, issues=["detection_error"])
    
    # Check quality, ROI, and face angle
    issues = FaceAnalysis.check_image_quality(frame, faces_info, pose)
    
    # Additional pose data for UI feedback
    pose_data = {
        "pose": pose,
        "instructions": POSE_INSTRUCTIONS.get(pose, ""),
        "target_angle": POSE_ANGLES.get(pose, (0, 0))
    }
    
    # Add angle data for UI feedback when needed
    if faces_info and "wrong_angle" in issues:
        for face in faces_info:
            if 'landmarks_dict' in face and face['landmarks_dict']:
                landmarks = face['landmarks_dict']
                _, current_angle = FaceAnalysis.check_face_angle(landmarks, POSE_ANGLES[pose])
                pose_data["current_angle"] = current_angle
                break
    
    # Add face bounding boxes for UI display
    face_data = []
    if faces_info:
        for face in faces_info:
            face_data.append({
                "bbox": face.get("bbox", []),
                "confidence": face.get("confidence", 0),
                "landmarks": face.get("landmarks", []).tolist() if hasattr(face.get("landmarks", []), "tolist") else []
            })
    pose_data["faces"] = face_data
    
    return ValidationResult(
        valid=(len(issues) == 0),
        issues=issues,
        pose_data=pose_data
    )

@router.post("/capture", response_model=PoseResult)
async def capture_face(file: UploadFile = File(...), 
                      session_id: str = Form(...), 
                      pose: str = Form(...),
                      retry: bool = Form(False)):
    """Capture a face for a specific pose"""
    session = get_session(session_id)
    person_name = session["person_name"]
    temp_dir = session["temp_dir"]
    
    # Validate pose
    if pose not in session["poses"]:
        raise HTTPException(status_code=400, detail="Invalid pose")
        
    if retry and pose != session["current_pose"]:
        raise HTTPException(status_code=400, detail="Can only retry current pose")
    
    # Process uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Verify face quality
    faces_info = FaceDetector.detect_faces(frame)
    issues = FaceAnalysis.check_image_quality(frame, faces_info, pose)
    
    if issues:
        return PoseResult(
            success=False,
            message=f"Image has quality issues: {', '.join(issues)}",
            pose=pose,
            session_id=session_id
        )
    
    try:
        # Process and save face
        save_path = ImageProcessor.capture_image(frame, faces_info, pose, temp_dir)
        
        # Update session state
        session["completed_poses"][pose] = save_path
        
        # Create backup if needed
        if not session["backup_created"] and session["existing_person"]:
            person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
            backup_dir = session["backup_dir"]
            
            if os.path.exists(person_dir):
                shutil.copytree(person_dir, backup_dir)
                session["backup_created"] = True
        
        # Determine next pose or completion
        all_poses = session["poses"]
        current_pose_idx = all_poses.index(pose)
        completed = False
        next_pose = ""
        
        if current_pose_idx + 1 < len(all_poses):
            next_pose = all_poses[current_pose_idx + 1]
            session["current_pose"] = next_pose
        else:
            completed = True
        
        # Get display path for frontend
        rel_path = os.path.relpath(save_path, start=".")
        display_path = "/" + rel_path.replace("\\", "/")
        
        return PoseResult(
            success=True,
            message=f"Successfully captured {pose} pose",
            pose=pose,
            image_path=display_path,
            next_pose=next_pose,
            session_id=session_id,
            completed=completed
        )
        
    except Exception as e:
        return PoseResult(
            success=False,
            message=f"Error capturing face: {str(e)}",
            pose=pose,
            session_id=session_id
        )

@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    """Get current session information"""
    session = get_session(session_id)
    
    return SessionInfo(
        session_id=session_id,
        person_name=session["person_name"],
        current_pose=session["current_pose"],
        completed_poses=list(session["completed_poses"].keys()),
        remaining_poses=[p for p in session["poses"] if p not in session["completed_poses"]],
        temp_dir=session["temp_dir"],
        backup_created=session["backup_created"],
        existing_person=session["existing_person"]
    )

@router.post("/finalize/{session_id}")
async def finalize_person(session_id: str, background_tasks: BackgroundTasks):
    """Save the captured poses and create masked versions"""
    session = get_session(session_id)
    person_name = session["person_name"]
    
    # Verify all poses are completed
    required_poses = set(session["poses"])
    completed_poses = set(session["completed_poses"].keys())
    
    if not required_poses.issubset(completed_poses):
        missing_poses = required_poses - completed_poses
        raise HTTPException(
            status_code=400, 
            detail=f"Not all required poses have been captured. Missing: {', '.join(missing_poses)}"
        )
    
    try:
        # Save to permanent dataset
        if not FaceCapture.save_to_dataset(person_name, session["completed_poses"]):
            raise HTTPException(status_code=500, detail="Failed to save images to dataset")
        
        # Create masked versions
        no_mask_dir = os.path.join(DATASET_IMAGES_DIR, person_name, "no_mask")
        mask_types = ["surgical", "N95", "cloth"]
        success, with_mask_count = ImageProcessor.create_masked_dataset(person_name, no_mask_dir, mask_types)
        
        # Clean up in background
        background_tasks.add_task(cleanup_session, session_id, True)
        
        if success:
            return {
                "success": True, 
                "message": f"Successfully added {person_name} to dataset with {with_mask_count} masked versions"
            }
        else:
            return {
                "success": True,
                "warning": True,
                "message": f"Added {person_name} successfully but failed to generate masked versions"
            }
            
    except Exception as e:
        # Clean up but don't mark as success
        background_tasks.add_task(cleanup_session, session_id, False)
        raise HTTPException(status_code=500, detail=f"Error finalizing person: {str(e)}")

@router.delete("/cancel/{session_id}")
async def cancel_person_addition(session_id: str, background_tasks: BackgroundTasks):
    """Cancel adding a person and clean up"""
    try:
        session = get_session(session_id)
        background_tasks.add_task(cleanup_session, session_id, False)
        return {"success": True, "message": "Person addition cancelled and resources cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cancelling: {str(e)}")


# Legacy function for compatibility
def add_new_person():
    """Legacy function for command-line addition of a new person"""
    print("The add_new_person function is now web-based.")
    print("Please use the web interface at http://localhost:8000/add_person.html")
    return False


# Setup routes in main FastAPI app
def setup_person_routes(app):
    """Configure routes for person management in the main app"""
    app.include_router(router)
    
    # Mount static directories
    app.mount("/static", StaticFiles(directory="static"), name="static")
    app.mount("/temp_captures", StaticFiles(directory="temp_captures"), name="temp_captures")
    
    # Add person page route
    @app.get("/add_person", response_class=HTMLResponse)
    async def get_add_person_page(request: Request):
        from fastapi.templating import Jinja2Templates
        templates = Jinja2Templates(directory="static/templates")
        templates.env.charset = 'utf-8'
        templates.env.auto_reload = True
        return templates.TemplateResponse(
            "add_person.html", 
            {"request": request}, 
            media_type="text/html; charset=utf-8"
        )