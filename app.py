import os
import shutil
import json
import time
import logging
import asyncio
import cv2
import numpy as np
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import base64
import uuid
from datetime import datetime

# Import components from the security system
from recognize import RecognizerSystem, load_email_config
from notification_service import NotificationService
from config import EMAILCONFIG_DIR, logger
from add_person import add_new_person
import os


# Models for request/response
class RecognitionResult(BaseModel):
    success: bool
    message: str
    faces: Optional[List[Dict]] = None
    performance: Optional[Dict] = None

class SystemStatus(BaseModel):
    status: str
    camera_id: str
    features: Dict[str, bool]
    processing_fps: float
    version: str = "1.0.0"

class CleanupResult(BaseModel):
    success: bool
    deleted_count: int
    remaining_count: int
    message: str

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Security System",
    description="API for face recognition, unknown person detection, and security monitoring",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")

# Template engine
templates = Jinja2Templates(directory="static/templates")

# Global RecognizerSystem instance
recognizer = None
email_config = load_email_config(EMAILCONFIG_DIR)
active_websockets = []

# Background processing queue for WebRTC frames
frame_queue = asyncio.Queue(maxsize=5)
processing_lock = asyncio.Lock()
is_processing_active = False

# System initialization
@app.on_event("startup")
async def startup_event():
    global recognizer
    # Don't initialize the recognizer system at startup
    recognizer = None
    
    # Tạo tất cả các thư mục cần thiết
    os.makedirs("temp_captures", exist_ok=True)
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Đảm bảo các thư mục có quyền ghi
    try:
        # Tạo file test để kiểm tra quyền ghi
        test_file_path = os.path.join("temp_captures", "test_write.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test write access")
        # Xóa file test
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        logger.info("Write access verified for required directories")
    except Exception as e:
        logger.error(f"Write access error: {e}")
    
    logger.info("Application started - waiting for user to start recognition system")

# System shutdown
@app.on_event("shutdown")
async def shutdown_event():
    global recognizer
    if recognizer:
        # Clean up resources
        logger.info("Shutting down recognition system")
        recognizer = None


# Main routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# API endpoints
@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    global recognizer
    
    # Initialize if not already initialized
    if not recognizer:
        try:
            recognizer = RecognizerSystem(
                frame_skip=2,
                use_anti_spoofing=True,
                use_mask_detection=True,
                use_recognition=True,
                use_tracking=True,
                use_notification=True,
                email_config=email_config,
                verbose=False
            )
            recognizer.set_camera_id("web")
        except Exception as e:
            logger.error(f"Failed to initialize recognition system: {e}")
            raise HTTPException(status_code=500, detail=f"System initialization failed: {str(e)}")
    
    # Return system status
    return SystemStatus(
        status="active",
        camera_id=recognizer.camera_id,
        features={
            "anti_spoofing": recognizer.use_anti_spoofing,
            "mask_detection": recognizer.use_mask_detection,
            "face_recognition": recognizer.use_recognition,
            "face_tracking": recognizer.use_tracking,
            "notifications": recognizer.use_notification
        },
        processing_fps=recognizer.fps
    )


@app.post("/api/process", response_model=RecognitionResult)
async def process_image(file: UploadFile = File(...)):
    global recognizer
    
    # Initialize if not already initialized
    if not recognizer:
        try:
            recognizer = RecognizerSystem(
                frame_skip=2,
                use_anti_spoofing=True,
                use_mask_detection=True,
                use_recognition=True,
                use_tracking=True,
                use_notification=True,
                email_config=email_config,
                verbose=False
            )
            recognizer.set_camera_id("api")
        except Exception as e:
            logger.error(f"Failed to initialize recognition system: {e}")
            return RecognitionResult(
                success=False,
                message=f"System initialization failed: {str(e)}"
            )
    
    try:
        # Read and process the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return RecognitionResult(
                success=False,
                message="Invalid image format"
            )
        
        # Process the frame
        start_time = time.time()
        _, face_results = recognizer.process_frame(img)
        process_time = time.time() - start_time
        
        # Clean results for JSON serialization
        clean_results = []
        for face in face_results:
            # Convert numpy arrays to lists
            bbox = face.get("bbox", []).tolist() if hasattr(face.get("bbox", []), "tolist") else face.get("bbox", [])
            
            # Tạo kết quả sạch ban đầu không có face_image
            face_result = {k: v for k, v in face.items() if k != 'face_image'}
            face_result["bbox"] = bbox
            
            # Thêm mới: Chuyển đổi face_image sang base64 nếu có
            if "face_image" in face and face["face_image"] is not None:
                try:
                    # Encode ảnh khuôn mặt dạng JPEG và chuyển sang base64
                    _, buffer = cv2.imencode('.jpg', face["face_image"])
                    face_result["image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                except Exception as e:
                    logger.error(f"Error encoding face image: {e}")
                    face_result["image"] = None
                    
            clean_results.append(face_result)
        
        return RecognitionResult(
            success=True,
            message="Image processed successfully",
            faces=clean_results,
            performance={"process_time": process_time, "fps": recognizer.fps}
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return RecognitionResult(
            success=False,
            message=f"Error processing image: {str(e)}"
        )


@app.delete("/api/cleanup/temp_captures", response_model=CleanupResult)
async def cleanup_temp_captures(days: int = 0):
    """Delete all files and folders in temp_captures directory immediately.
    
    Args:
        days: Parameter kept for API compatibility but ignored - all content is deleted regardless.
    """
    try:
        temp_dir = Path("temp_captures")
        
        if not temp_dir.exists():
            return CleanupResult(
                success=True,
                deleted_count=0,
                remaining_count=0,
                message="No temp_captures directory found"
            )
        
        deleted_count = 0
        
        # Delete all files first
        for file_path in list(temp_dir.glob("**/*")):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete file {file_path}: {e}")
        
        # Then delete all directories (bottom-up to handle nested dirs)
        for dir_path in sorted([p for p in temp_dir.glob("**/*") if p.is_dir()], 
                              key=lambda p: len(str(p).split(os.sep)), reverse=True):
            if dir_path != temp_dir:
                try:
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete directory {dir_path}: {e}")
        
        # Remove dependency on recognizer system
        # Don't call recognizer.notification_service.cleanup_temp_captures()
        
        message = f"Successfully deleted all {deleted_count} items in temporary captures"
        logger.info(message)
        
        return CleanupResult(
            success=True,
            deleted_count=deleted_count,
            remaining_count=0,  # Should be 0 after full cleanup
            message=message
        )
    
    except Exception as e:
        logger.error(f"Error cleaning temp_captures: {e}")
        return CleanupResult(
            success=False,
            deleted_count=0,
            remaining_count=-1,
            message=f"Error cleaning temp_captures: {str(e)}"
        )
        
@app.post("/api/reset_caches")
async def reset_caches():
    """Reset all caches in the recognition system."""
    if not recognizer:
        raise HTTPException(status_code=404, detail="Recognition system not initialized")
    
    try:
        recognizer.reset_caches()
        return {"success": True, "message": "All caches reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting caches: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting caches: {str(e)}")


# WebSocket for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global recognizer, active_websockets, is_processing_active
    
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        # Initialize recognizer if needed
        if not recognizer:
            try:
                recognizer = RecognizerSystem(
                    frame_skip=1,  # Lower for real-time WebSocket
                    use_anti_spoofing=True,
                    use_mask_detection=True,
                    use_recognition=True,
                    use_tracking=True,
                    use_notification=True,
                    email_config=email_config,
                    verbose=False
                )
                recognizer.set_camera_id("websocket")
            except Exception as e:
                logger.error(f"Failed to initialize recognition system: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"System initialization failed: {str(e)}"
                })
                return
        
        # Start background processing if not already running
        if not is_processing_active:
            is_processing_active = True
            asyncio.create_task(process_frame_queue())
        
        # WebSocket communication loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "frame":
                # Process video frame
                image_data = message["data"].split(",")[1]  # Remove data:image/jpeg;base64,
                
                # Add to processing queue
                if frame_queue.qsize() < frame_queue.maxsize:
                    await frame_queue.put((websocket, image_data))
                # Otherwise skip (to avoid lag)
            
            elif message["type"] == "command":
                # Handle commands
                command = message["command"]
                
                if command == "reset_caches":
                    recognizer.reset_caches()
                    await websocket.send_json({
                        "type": "command_result",
                        "command": "reset_caches",
                        "success": True
                    })
                
                elif command == "embeddings_complete":
                    # This will be sent from the background task when complete
                    await websocket.send_json({
                        "type": "command_result",
                        "command": "embeddings_complete",
                        "success": True,
                        "message": "Embeddings have been regenerated successfully"
                    })
                   
                elif command == "cleanup_temp":
                    days = message.get("days", 0)
                    result = await cleanup_temp_captures(days)
                    await websocket.send_json({
                        "type": "command_result",
                        "command": "cleanup_temp",
                        "success": result.success,
                        "data": result.dict()
                    })
                    
    except WebSocketDisconnect:
        # Handle client disconnect
        logger.info(f"WebSocket disconnected - removing from active connections")
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        # Try to send error message
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
        # Clean up
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# Background task to process frames from the queue
async def process_frame_queue():
    global recognizer, is_processing_active, frame_queue
    
    try:
        while is_processing_active:
            if frame_queue.empty():
                await asyncio.sleep(0.01)
                continue
            
            websocket, image_data = await frame_queue.get()
            
            # Skip processing if websocket is closed or removed from active_websockets
            if websocket not in active_websockets:
                frame_queue.task_done()  # Make sure to mark task as done
                continue
            
            # Process frame
            try:
                # Decode base64 image
                nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    frame_queue.task_done()
                    continue
                
                # Process frame
                start_time = time.time()
                
                # Use async lock to prevent multiple frames being processed simultaneously
                async with processing_lock:
                    _, face_results = recognizer.process_frame(img)
                
                process_time = time.time() - start_time
                
                # Clean results for JSON serialization
                clean_results = []
                for face in face_results:
                    # Convert numpy arrays to lists
                    bbox = face.get("bbox", []).tolist() if hasattr(face.get("bbox", []), "tolist") else face.get("bbox", [])
                    
                    # Create result without face_image first
                    face_result = {k: v for k, v in face.items() if k != 'face_image'}
                    face_result["bbox"] = bbox
                    
                    # Now add face image as base64 if available
                    if "face_image" in face and face["face_image"] is not None:
                        try:
                            # Resize to smaller dimensions to reduce data size
                            face_img = face["face_image"]
                            if face_img.shape[0] > 96 or face_img.shape[1] > 96:
                                face_img = cv2.resize(face_img, (96, 96))
                                
                            # Encode with reduced quality
                            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                            _, buffer = cv2.imencode('.jpg', face_img, encode_param)
                            face_result["image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                        except Exception as e:
                            logger.error(f"Error encoding face image for WebSocket: {e}")
                            face_result["image"] = None
                    
                    clean_results.append(face_result)
                
                # Send results back to client - check again if websocket is still active
                if websocket in active_websockets:
                    try:
                        await websocket.send_json({
                            "type": "recognition_result",
                            "faces": clean_results,
                            "performance": {
                                "process_time": process_time,
                                "fps": recognizer.fps
                            }
                        })
                    except Exception as e:
                        logger.error(f"Failed to send result: {e}")
                        # If we can't send, the WebSocket is likely closed - remove it
                        if websocket in active_websockets:
                            active_websockets.remove(websocket)
                
            except Exception as e:
                logger.error(f"Error processing WebSocket frame: {e}")
            finally:
                # Always mark the task as done, regardless of success or failure
                frame_queue.task_done()
            
    except asyncio.CancelledError:
        # Task was cancelled
        logger.info("Frame processing task cancelled")
    except Exception as e:
        logger.error(f"Fatal error in frame processing task: {e}")
    finally:
        is_processing_active = False

@app.get("/api/dataset/users")
async def get_dataset_users():
    """Get all users in the dataset with their front-facing images."""
    dataset_dir = Path("dataset/images")
    users = []
    
    if dataset_dir.exists():
        for person_dir in dataset_dir.iterdir():
            if person_dir.is_dir():
                name = person_dir.name
                front_image_path = person_dir / "no_mask" / "front.jpg"
                
                if front_image_path.exists():
                    users.append({
                        "id": name,  # Using name as ID for simplicity
                        "name": name.capitalize(),  # Capitalize name for display
                        "image_path": f"/dataset/images/{name}/no_mask/front.jpg"
                    })
    
    return users

@app.delete("/api/dataset/users/{user_id}")
async def delete_dataset_user(user_id: str):
    """Delete a user from the dataset."""
    dataset_dir = Path("dataset/images")
    user_path = dataset_dir / user_id
    
    if not user_path.exists():
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in dataset")
    
    try:
        # Delete the user's directory recursively
        shutil.rmtree(user_path)
        
        # Reset recognizer caches if active
        if recognizer:
            recognizer.reset_caches()
            
        return {"success": True, "message": f"User '{user_id}' deleted from dataset"}
    except Exception as e:
        logger.error(f"Error deleting user from dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")

from add_person import setup_person_routes
setup_person_routes(app)

@app.post("/api/regenerate_embeddings")
async def regenerate_embeddings(background_tasks: BackgroundTasks):
    """Regenerate embeddings with data augmentation."""
    try:
        # Execute the create_embeddings.py script with augmentation flag in the background
        def create_embeddings_task():
            import subprocess
            import sys
            
            try:
                subprocess.run([
                    sys.executable, 
                    "create_embeddings.py", 
                    "--augment"
                ], check=True)
                
                # Reset recognizer caches after embeddings are generated
                if recognizer:
                    recognizer.reset_caches()
                    
                logger.info("Embeddings regenerated successfully with augmentation")
            except Exception as e:
                logger.error(f"Error regenerating embeddings: {e}")
                
        # Run the task in the background
        background_tasks.add_task(create_embeddings_task)
        
        return {"success": True, "message": "Regenerating embeddings in background"}
    except Exception as e:
        logger.error(f"Error starting embeddings regeneration: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting embeddings regeneration: {str(e)}")

class EmailSettings(BaseModel):
    email: str
    enable_notifications: bool

# Thêm endpoint để cập nhật email
@app.post("/api/settings/email")
async def update_email_settings(settings: EmailSettings):
    """Update email settings for notifications."""
    try:
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(EMAILCONFIG_DIR), exist_ok=True)
        
        # Tải cấu hình hiện tại
        config = load_email_config()
        
        # Cập nhật cấu hình mới
        config["receiver"] = settings.email
        config["enable_notifications"] = settings.enable_notifications
        
        # Lưu cấu hình
        with open(EMAILCONFIG_DIR, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Cập nhật notification service nếu đã khởi tạo
        if recognizer and recognizer.notification_service:
            recognizer.notification_service.update_email_config(config)
            
        return {"success": True, "message": "Email settings updated"}
    except Exception as e:
        logger.error(f"Error updating email settings: {e}")
        return {"success": False, "message": str(e)}

# Thêm endpoint để lấy cấu hình email hiện tại
@app.get("/api/settings/email")
async def get_email_settings():
    """Get current email settings."""
    global email_config
    
    try:
        use_notifications = False
        email = ""
        
        if email_config:
            email = email_config.get("receiver", "")
            # Lấy trạng thái notification từ recognizer
            if recognizer:
                use_notifications = recognizer.use_notification
        
        return {
            "email": email,
            "enable_notifications": use_notifications
        }
    except Exception as e:
        logger.error(f"Error getting email settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting email settings: {str(e)}")

# Run the app
if __name__ == "__main__":
    import uvicorn
    import os
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    
    uvicorn.run(app, host=host, port=port)