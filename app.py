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
from add_person import add_new_person, setup_person_routes


# === Models ===
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

class EmailSettings(BaseModel):
    email: str
    enable_notifications: bool


# === Application Setup ===
app = FastAPI(
    title="Face Recognition Security System",
    description="API for face recognition, unknown person detection, and security monitoring",
    version="1.0.0"
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/dataset", StaticFiles(directory="dataset"), name="dataset")
templates = Jinja2Templates(directory="static/templates")

# Global variables
recognizer = None
email_config = load_email_config(EMAILCONFIG_DIR)
active_websockets = []
frame_queue = asyncio.Queue(maxsize=5)
processing_lock = asyncio.Lock()
is_processing_active = False


# === Helper Functions ===
def initialize_recognizer(camera_id="default"):
    """Initialize the recognition system if not already initialized"""
    global recognizer
    
    if recognizer:
        return recognizer
        
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
        recognizer.set_camera_id(camera_id)
        return recognizer
    except Exception as e:
        logger.error(f"Failed to initialize recognition system: {e}")
        raise HTTPException(status_code=500, detail=f"System initialization failed: {str(e)}")


def process_image_to_array(image_data):
    """Convert image data to numpy array for processing"""
    if isinstance(image_data, bytes):
        # Direct binary data
        nparr = np.frombuffer(image_data, np.uint8)
    else:
        # Base64 encoded data
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
    
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def prepare_face_result(face):
    """Clean and prepare face detection result for JSON serialization"""
    # Convert bbox numpy array to list if needed
    bbox = face.get("bbox", []).tolist() if hasattr(face.get("bbox", []), "tolist") else face.get("bbox", [])
    
    # Create result without face_image first
    face_result = {k: v for k, v in face.items() if k != 'face_image'}
    face_result["bbox"] = bbox
    
    # Add face image as base64 if available
    if "face_image" in face and face["face_image"] is not None:
        try:
            # Encode with reduced quality for efficiency
            face_img = face["face_image"]
            if face_img.shape[0] > 96 or face_img.shape[1] > 96:
                face_img = cv2.resize(face_img, (96, 96))
                
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', face_img, encode_param)
            face_result["image"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        except Exception as e:
            logger.error(f"Error encoding face image: {e}")
            face_result["image"] = None
    
    return face_result


# === Application Lifecycle ===
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global recognizer
    recognizer = None
    
    # Create required directories
    for directory in ["temp_captures", "dataset", "config"]:
        os.makedirs(directory, exist_ok=True)
    
    # Verify write permissions
    try:
        test_file_path = os.path.join("temp_captures", "test_write.txt")
        with open(test_file_path, 'w') as f:
            f.write("Test write access")
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        logger.info("Write access verified for required directories")
    except Exception as e:
        logger.error(f"Write access error: {e}")
    
    logger.info("Application started - waiting for user to start recognition system")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global recognizer
    if recognizer:
        logger.info("Shutting down recognition system")
        recognizer = None


# === Web Routes ===
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Render main application page"""
    return templates.TemplateResponse("index.html", {"request": request})


# === API Endpoints ===
@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get the current system status"""
    rec = initialize_recognizer("web")
    
    return SystemStatus(
        status="active",
        camera_id=rec.camera_id,
        features={
            "anti_spoofing": rec.use_anti_spoofing,
            "mask_detection": rec.use_mask_detection,
            "face_recognition": rec.use_recognition,
            "face_tracking": rec.use_tracking,
            "notifications": rec.use_notification
        },
        processing_fps=rec.fps
    )


@app.post("/api/process", response_model=RecognitionResult)
async def process_image(file: UploadFile = File(...)):
    """Process an uploaded image for face recognition"""
    try:
        # Initialize recognizer if needed
        rec = initialize_recognizer("api")
        
        # Read and process the image
        contents = await file.read()
        img = process_image_to_array(contents)
        
        if img is None:
            return RecognitionResult(success=False, message="Invalid image format")
        
        # Process the frame
        start_time = time.time()
        _, face_results = rec.process_frame(img)
        process_time = time.time() - start_time
        
        # Clean results for JSON serialization
        clean_results = [prepare_face_result(face) for face in face_results]
        
        return RecognitionResult(
            success=True,
            message="Image processed successfully",
            faces=clean_results,
            performance={"process_time": process_time, "fps": rec.fps}
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return RecognitionResult(success=False, message=f"Error processing image: {str(e)}")


@app.delete("/api/cleanup/temp_captures", response_model=CleanupResult)
async def cleanup_temp_captures(days: int = 0):
    """Delete all files in the temp_captures directory"""
    try:
        temp_dir = Path("temp_captures")
        
        if not temp_dir.exists():
            return CleanupResult(success=True, deleted_count=0, remaining_count=0, 
                                message="No temp_captures directory found")
        
        deleted_count = 0
        
        # Delete all files first
        for file_path in list(temp_dir.glob("**/*")):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete file {file_path}: {e}")
        
        # Then delete empty directories (bottom-up)
        for dir_path in sorted([p for p in temp_dir.glob("**/*") if p.is_dir()], 
                              key=lambda p: len(str(p).split(os.sep)), reverse=True):
            if dir_path != temp_dir:
                try:
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete directory {dir_path}: {e}")
        
        message = f"Successfully deleted all {deleted_count} items in temporary captures"
        logger.info(message)
        
        return CleanupResult(
            success=True,
            deleted_count=deleted_count,
            remaining_count=0,
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
    """Reset all caches in the recognition system"""
    if not recognizer:
        raise HTTPException(status_code=404, detail="Recognition system not initialized")
    
    try:
        recognizer.reset_caches()
        return {"success": True, "message": "All caches reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting caches: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting caches: {str(e)}")


# === WebSocket Handlers ===
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time face recognition and admin operations"""
    global recognizer, active_websockets, is_processing_active
    
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        # Main WebSocket communication loop - no recognizer initialized yet
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "frame":
                # Only initialize recognizer when frames are received
                if not recognizer:
                    rec = initialize_recognizer("websocket")
                
                # Start background processing if not already running
                if not is_processing_active:
                    is_processing_active = True
                    asyncio.create_task(process_frame_queue())
                
                # Process video frame
                image_data = message["data"].split(",")[1]  # Remove data:image/jpeg;base64,
                
                # Add to processing queue (skip if queue is full to avoid lag)
                if frame_queue.qsize() < frame_queue.maxsize:
                    await frame_queue.put((websocket, image_data))
            
            elif message["type"] == "command":
                # Handle commands - some don't need recognizer
                await handle_websocket_command(websocket, message)
                   
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected - removing from active connections")
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass
        if websocket in active_websockets:
            active_websockets.remove(websocket)

async def handle_websocket_command(websocket: WebSocket, message: dict):
    """Handle WebSocket commands - with or without recognizer"""
    global recognizer
    command = message["command"]
    
    # Commands that require recognizer
    if command == "reset_caches":
        if not recognizer:
            await websocket.send_json({
                "type": "command_result",
                "command": "reset_caches",
                "success": False,
                "message": "Recognition system not initialized"
            })
            return
            
        recognizer.reset_caches()
        await websocket.send_json({
            "type": "command_result",
            "command": "reset_caches",
            "success": True
        })
    
    # Commands that don't require recognizer
    elif command == "regenerate_embeddings":
        # Start background task
        background_tasks = BackgroundTasks()
        background_tasks.add_task(create_embeddings_task)
        background_tasks.add_task(notify_embedding_complete)
        
        await websocket.send_json({
            "type": "command_result",
            "command": "regenerate_embeddings",
            "success": True,
            "message": "Regenerating embeddings in background"
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

async def process_frame_queue():
    """Process frames from the queue in the background"""
    global recognizer, is_processing_active, frame_queue
    
    try:
        while is_processing_active:
            if frame_queue.empty():
                await asyncio.sleep(0.01)
                continue
            
            websocket, image_data = await frame_queue.get()
            
            # Skip if websocket disconnected
            if websocket not in active_websockets:
                frame_queue.task_done()
                continue
            
            try:
                # Process frame
                img = process_image_to_array(image_data)
                if img is None:
                    frame_queue.task_done()
                    continue
                
                # Process frame with lock to prevent simultaneous processing
                start_time = time.time()
                async with processing_lock:
                    _, face_results = recognizer.process_frame(img)
                process_time = time.time() - start_time
                
                # Clean and send results
                clean_results = [prepare_face_result(face) for face in face_results]
                
                # Send results if websocket still active
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
                        if websocket in active_websockets:
                            active_websockets.remove(websocket)
                
            except Exception as e:
                logger.error(f"Error processing WebSocket frame: {e}")
            finally:
                frame_queue.task_done()
            
    except asyncio.CancelledError:
        logger.info("Frame processing task cancelled")
    except Exception as e:
        logger.error(f"Fatal error in frame processing task: {e}")
    finally:
        is_processing_active = False


# === Dataset Management ===
@app.get("/api/dataset/users")
async def get_dataset_users():
    """Get all users in the dataset with their front-facing images"""
    dataset_dir = Path("dataset/images")
    users = []
    
    if dataset_dir.exists():
        for person_dir in dataset_dir.iterdir():
            if person_dir.is_dir():
                name = person_dir.name
                front_image_path = person_dir / "no_mask" / "front.jpg"
                
                if front_image_path.exists():
                    users.append({
                        "id": name,
                        "name": name.capitalize(),
                        "image_path": f"/dataset/images/{name}/no_mask/front.jpg"
                    })
    
    return users


@app.delete("/api/dataset/users/{user_id}")
async def delete_dataset_user(user_id: str):
    """Delete a user from the dataset"""
    user_path = Path("dataset/images") / user_id
    
    if not user_path.exists():
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in dataset")
    
    try:
        shutil.rmtree(user_path)
        
        # Reset caches if recognizer is active
        if recognizer:
            recognizer.reset_caches()
            
        return {"success": True, "message": f"User '{user_id}' deleted from dataset"}
    except Exception as e:
        logger.error(f"Error deleting user from dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting user: {str(e)}")


# === Embeddings Management ===
embedding_generation_complete = asyncio.Event()
@app.post("/api/regenerate_embeddings")
async def regenerate_embeddings(background_tasks: BackgroundTasks):
    """Regenerate embeddings with data augmentation"""
    global embedding_generation_complete
    embedding_generation_complete.clear()  # Reset state
    
    try:
        def create_embeddings_task():
            import subprocess
            import sys
            
            try:
                subprocess.run([
                    sys.executable, 
                    "create_embeddings.py", 
                    "--augment"
                ], check=True)
                
                # Reset caches after embeddings are generated
                if recognizer:
                    recognizer.reset_caches()
                    
                logger.info("Embeddings regenerated successfully with augmentation")
                
                # Set event when complete
                embedding_generation_complete.set()
            except Exception as e:
                logger.error(f"Error regenerating embeddings: {e}")
                embedding_generation_complete.set()  # Set even on failure
                
        background_tasks.add_task(create_embeddings_task)
        # Start another task to notify when complete
        background_tasks.add_task(notify_embedding_complete)
        return {"success": True, "message": "Regenerating embeddings in background"}
    except Exception as e:
        logger.error(f"Error starting embeddings regeneration: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting embeddings regeneration: {str(e)}")

async def notify_embedding_complete():
    """Wait for embeddings to complete and notify clients"""
    await embedding_generation_complete.wait()
    
    # Notify all active websocket clients
    for ws in active_websockets:
        try:
            await ws.send_json({
                "type": "command_result",
                "command": "embeddings_complete",
                "success": True
            })
        except:
            pass  # Ignore errors


# === Email Settings ===
@app.post("/api/settings/email")
async def update_email_settings(settings: EmailSettings):
    """Update email settings for notifications"""
    try:
        os.makedirs(os.path.dirname(EMAILCONFIG_DIR), exist_ok=True)
        
        # Load and update config
        config = load_email_config()
        config["receiver"] = settings.email
        config["enable_notifications"] = settings.enable_notifications
        
        # Save config
        with open(EMAILCONFIG_DIR, 'w') as f:
            json.dump(config, f, indent=4)
            
        # Update notification service if initialized
        if recognizer and recognizer.notification_service:
            recognizer.notification_service.update_email_config(config)
            
        return {"success": True, "message": "Email settings updated"}
    except Exception as e:
        logger.error(f"Error updating email settings: {e}")
        return {"success": False, "message": str(e)}


@app.get("/api/settings/email")
async def get_email_settings():
    """Get current email settings"""
    try:
        use_notifications = False
        email = ""
        
        if email_config:
            email = email_config.get("receiver", "")
            if recognizer:
                use_notifications = recognizer.use_notification
        
        return {
            "email": email,
            "enable_notifications": use_notifications
        }
    except Exception as e:
        logger.error(f"Error getting email settings: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting email settings: {str(e)}")


# Setup routes from add_person module
setup_person_routes(app)

# Run the app
if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 7860))
    
    uvicorn.run(app, host=host, port=port)
