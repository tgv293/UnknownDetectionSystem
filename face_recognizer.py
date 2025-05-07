import os
import cv2
import numpy as np
import pickle
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import hnswlib

# Import config
from config import EMBEDDINGS_PATH, EMBEDDINGS_SOURCE_INFO_PATH, EMBEDDINGS_HNSW_PATH
from config import COSINE_SIMILARITY_THRESHOLD, MODEL_DIR, logger
from preprocessing import resize_for_face_recognition
from static.models.face_recognition.arcface import ArcFace

class FaceRecognizer:
    """Face recognition system using ArcFace model and HNSWlib."""

    def __init__(
        self, 
        model_path: str = None,
        embeddings_path: str = EMBEDDINGS_PATH, 
        threshold: float = COSINE_SIMILARITY_THRESHOLD,
        cache_size: int = 50,
        cache_ttl: float = 2.5,
        verbose: bool = True
    ):
        """Initialize face recognizer with ArcFace model."""
        # Model settings
        self.model_path = model_path or os.path.join(MODEL_DIR, "r100_arcface_glint.onnx")
        self.model = None
        
        # Recognition settings
        self.embeddings_path = embeddings_path
        self.threshold = threshold
        self.embedding_dim = 512  # ArcFace has 512 dimensions
        
        # Cache system
        self.track_cache = {}  # track_id -> result
        self.image_cache = {}  # image_hash -> result (fallback)
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self.cache_timestamps = {}
        
        # Core components
        self.embeddings_db = {}
        self.hnsw_index = None
        self.index_id_to_info = []
        self.embedding_source_info = {}
        
        # HNSWlib parameters
        self.ef_construction = 200
        self.M = 16
        self.ef_search = 50
        
        # Performance tracking
        self.process_times = deque(maxlen=30)
        self.embedding_times = deque(maxlen=30)
        
        # Status flags
        self.embeddings_loaded = False
        self.model_loaded = False
        
        # Load model and embeddings
        self._load_model()
        self.load_embeddings()
        
        if verbose:
            logger.info(f"FaceRecognizer initialized with threshold={threshold}")
            if self.model_loaded:
                logger.info(f"Using ArcFace model from {self.model_path}")
            if self.embeddings_loaded:
                logger.info(f"Embeddings loaded with {len(self.embeddings_db)} persons")

    def _load_model(self) -> bool:
        """Load ArcFace model."""
        try:
            logger.info(f"Loading ArcFace model from {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
                
            # Use ArcFace class
            self.model = ArcFace(model_path=self.model_path)
            
            # Set flag
            self.model_loaded = True
            logger.info("ArcFace model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {e}")
            self.model_loaded = False
            return False
        
    def load_embeddings(self) -> bool:
        """Load embeddings database and HNSW index."""
        if self.embeddings_loaded:
            return True
            
        try:
            if not os.path.exists(self.embeddings_path):
                logger.error(f"Embeddings database not found at {self.embeddings_path}")
                return False
                
            logger.info(f"Loading embeddings database from {self.embeddings_path}")
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            # Handle different formats
            if isinstance(data, dict) and "embeddings_db" in data:
                self.embeddings_db = data["embeddings_db"]
                self.index_id_to_info = data.get("index_id_to_info", [])
            else:
                # Legacy format
                self.embeddings_db = data
                
            # Load or rebuild HNSW index
            hnsw_path = EMBEDDINGS_HNSW_PATH
            if os.path.exists(hnsw_path):
                try:
                    self._initialize_hnsw()
                    self.hnsw_index.load_index(hnsw_path, max_elements=len(self.index_id_to_info))
                    logger.info(f"HNSW index loaded with {self.hnsw_index.get_current_count()} vectors")
                    
                    # Verify index is cosine type
                    if not self._is_cosine_index(self.hnsw_index):
                        logger.warning("HNSW index is not cosine type, rebuilding...")
                        self._rebuild_hnsw_index()
                except Exception as e:
                    logger.warning(f"Error loading HNSW index: {e}, rebuilding...")
                    self._rebuild_hnsw_index()
            else:
                logger.info("HNSW index not found, rebuilding...")
                self._rebuild_hnsw_index()
                
            # Load source info if available
            source_info_path = EMBEDDINGS_SOURCE_INFO_PATH
            if os.path.exists(source_info_path):
                try:
                    with open(source_info_path, 'rb') as f:
                        self.embedding_source_info = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load source info: {e}")
            
            # Database stats
            person_count = len(self.embeddings_db)
            no_mask_count = sum(len(person_data.get("no_mask", {})) 
                              for person_data in self.embeddings_db.values())
            with_mask_count = sum(len(person_data.get("with_mask", {})) 
                                for person_data in self.embeddings_db.values())
            
            logger.info(f"Embeddings loaded: {person_count} people, "
                       f"{no_mask_count} no_mask, {with_mask_count} with_mask")
            
            self.embeddings_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embeddings database: {e}")
            return False
    
    def _is_cosine_index(self, index) -> bool:
        """Check if HNSW index is cosine type."""
        try:
            return index.space == 'cosine'
        except:
            return False
            
    def _initialize_hnsw(self):
        """Initialize HNSW index for efficient similarity search"""
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.hnsw_index.init_index(
            max_elements=10000,
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.hnsw_index.set_ef(self.ef_search)
            
    def _rebuild_hnsw_index(self) -> None:
        """Rebuild HNSW index from embeddings database."""
        try:
            # Create new cosine index
            self._initialize_hnsw()
            self.index_id_to_info = []
            
            # Count total elements to set max_elements
            total_elements = sum(
                len(poses) for person_data in self.embeddings_db.values() 
                for poses in person_data.values()
            )
            
            # Resize index if needed
            if total_elements > self.hnsw_index.get_max_elements():
                self.hnsw_index.resize_index(total_elements)
            
            # Add all embeddings to the index
            index_id = 0
            for person_name, data in self.embeddings_db.items():
                for mask_status, poses in data.items():
                    for pose_name, embedding in poses.items():
                        if embedding is not None:
                            # Normalize embedding for cosine similarity
                            emb_copy = np.array(embedding, dtype=np.float32)
                            norm = np.linalg.norm(emb_copy)
                            if norm > 0:
                                emb_copy = emb_copy / norm
                            
                            # Add to index
                            self.index_id_to_info.append((person_name, mask_status, pose_name))
                            self.hnsw_index.add_items(emb_copy, index_id)
                            index_id += 1
            
            logger.info(f"Rebuilt HNSW index with {self.hnsw_index.get_current_count()} vectors")
            
            # Save rebuilt index
            self.hnsw_index.save_index(EMBEDDINGS_HNSW_PATH)
            logger.info(f"Saved rebuilt HNSW index to {EMBEDDINGS_HNSW_PATH}")
                
        except Exception as e:
            logger.error(f"Failed to rebuild HNSW index: {e}")
            self.hnsw_index = None

    def extract_embedding(self, face_img: np.ndarray, landmarks: np.ndarray = None) -> Optional[np.ndarray]:
        """Extract embedding vector from face image."""
        if face_img is None or not self.model_loaded:
            return None
        
        start_time = time.time()
            
        try:
            # Resize face for recognition
            resized_face = resize_for_face_recognition(face_img)
            if resized_face is None:
                logger.warning("Failed to resize face for recognition")
                return None
                
            # Extract embedding
            embedding = self.model.get_feat(resized_face)[0]
            
            # Check for valid embedding
            if np.isnan(embedding).any() or np.sum(embedding) == 0:
                logger.warning("Invalid embedding extracted (NaN or zeros)")
                return None
                
            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Track embedding time
            self.embedding_times.append(time.time() - start_time)
                    
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def _clean_cache(self):
        """Remove expired entries from caches."""
        current_time = time.time()
        
        # Find and process expired entries
        expired_keys = [key for key, timestamp in self.cache_timestamps.items() 
                       if current_time - timestamp > self.cache_ttl]
        
        for key in expired_keys:
            prefix, actual_key = key.split('_', 1)
            
            if prefix == "track":
                self.track_cache.pop(int(actual_key), None)
            elif prefix == "img":
                self.image_cache.pop(actual_key, None)
                
            self.cache_timestamps.pop(key, None)
        
        # If still too many entries, remove oldest ones
        total_entries = len(self.track_cache) + len(self.image_cache)
        if total_entries > self.cache_size:
            # Get oldest entries
            oldest_entries = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            to_remove = total_entries - self.cache_size
            
            # Remove them
            for key, _ in oldest_entries[:to_remove]:
                prefix, actual_key = key.split('_', 1)
                
                if prefix == "track":
                    self.track_cache.pop(int(actual_key), None)
                elif prefix == "img":
                    self.image_cache.pop(actual_key, None)
                    
                self.cache_timestamps.pop(key, None)
    
    def _compute_cache_key(self, face_img: np.ndarray) -> str:
        """Compute a cache key for a face image."""
        try:
            small_img = cv2.resize(face_img, (32, 32))
            means = np.mean(small_img, axis=(0, 1))
            return f"{means[0]:.2f}_{means[1]:.2f}_{means[2]:.2f}"
        except Exception as e:
            logger.error(f"Error generating face hash: {e}")
            return f"error_{time.time()}"

    def _map_mask_status(self, mask_status: str) -> str:
        """Map mask status to database format."""
        if mask_status == "without_mask":
            return "no_mask"
        elif mask_status in ["with_mask", "incorrect_mask"]:
            return "with_mask"
        else:
            return "no_mask"  # Default to no_mask

    def recognize(self, face_img: np.ndarray, track_id: int = -1, mask_status: str = "unknown", 
            landmarks: np.ndarray = None, force_refresh: bool = False) -> Dict[str, Any]:
        """Recognize a face image using HNSW similarity search."""
        start_time = time.time()
        
        # Handle invalid input
        if face_img is None or not self.model_loaded:
            return {"name": "unknown", "confidence": 0.0, 
                   "mask_status": mask_status, "similarity": 0.0}
        
        # Check for mask status change
        if track_id >= 0 and track_id in self.track_cache:
            cached_status = self.track_cache[track_id].get("mask_status", "unknown")
            if cached_status != mask_status:
                force_refresh = True
        
        # Check caches if not forcing refresh
        if not force_refresh:
            # Check track cache first
            if track_id >= 0 and track_id in self.track_cache:
                self.cache_timestamps[f"track_{track_id}"] = time.time()
                result = self.track_cache[track_id]
                result["mask_status"] = mask_status
                return result
                
            # Check image cache as fallback
            image_hash = self._compute_cache_key(face_img)
            if image_hash in self.image_cache:
                self.cache_timestamps[f"img_{image_hash}"] = time.time()
                result = self.image_cache[image_hash]
                result["mask_status"] = mask_status
                
                # Update track cache if we have a track_id
                if track_id >= 0:
                    self.track_cache[track_id] = result
                    self.cache_timestamps[f"track_{track_id}"] = time.time()
                    
                return result
        
        # Ensure embeddings are loaded
        if not self.embeddings_loaded and not self.load_embeddings():
            return {"name": "unknown", "confidence": 0.0, 
                   "mask_status": mask_status, "similarity": 0.0}
            
        # Map mask status to database format
        db_mask_status = self._map_mask_status(mask_status)
        
        # Extract embedding
        face_embedding = self.extract_embedding(face_img, landmarks)
        if face_embedding is None:
            return {"name": "unknown", "confidence": 0.0, 
                   "mask_status": mask_status, "similarity": 0.0}
        
        # Adjust threshold for masks
        threshold = self.threshold
        if mask_status in ["with_mask", "incorrect_mask"]:
            threshold *= 0.83  # 17% reduction for masks
            
        # Recognize with HNSW
        result = self._recognize_with_hnsw(face_embedding, db_mask_status, threshold)
        result["mask_status"] = mask_status
        
        # Update caches
        image_hash = self._compute_cache_key(face_img)
        if track_id >= 0:
            self.track_cache[track_id] = result
            self.cache_timestamps[f"track_{track_id}"] = time.time()
            
        self.image_cache[image_hash] = result
        self.cache_timestamps[f"img_{image_hash}"] = time.time()
        
        # Track processing time
        self.process_times.append(time.time() - start_time)
        return result

    def _recognize_with_hnsw(self, face_embedding: np.ndarray, db_mask_status: str = "no_mask", 
                           threshold: float = None) -> Dict[str, Any]:
        """Recognize face using HNSW index with cosine similarity."""
        # Check if HNSW index is available
        if self.hnsw_index is None or self.hnsw_index.get_current_count() == 0:
            return {"name": "unknown", "confidence": 0.0, "similarity": 0.0}
        
        # Use provided threshold or default
        threshold = threshold if threshold is not None else self.threshold
            
        # Search k nearest neighbors
        k = min(5, self.hnsw_index.get_current_count())
        labels, distances = self.hnsw_index.knn_query(face_embedding, k=k)
        
        # Convert distances to similarities (1 - distance for cosine)
        similarities = 1 - np.array(distances)[0]
        
        # Find best match above threshold
        best_similarity = 0.0
        best_name = "unknown"
        
        for similarity, idx in zip(similarities, labels[0]):
            if idx < 0 or idx >= len(self.index_id_to_info):
                continue
                
            person_name = self.index_id_to_info[idx][0]
            
            if similarity >= threshold and similarity > best_similarity:
                best_similarity = similarity
                best_name = person_name
        
        # Return recognition result
        return {
            "name": best_name,
            "confidence": float(best_similarity),
            "similarity": float(best_similarity)
        }
      
    def batch_recognize(self, faces: List[np.ndarray], mask_statuses: List[str], 
                    track_ids: List[int] = None, landmarks_list: List[np.ndarray] = None,
                    force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Recognize multiple faces at once."""
        if not faces:
            return []
        
        # Clean cache periodically
        self._clean_cache()
        
        # Ensure inputs have correct length
        track_ids = track_ids if track_ids and len(track_ids) == len(faces) else [-1] * len(faces)
        landmarks = landmarks_list if landmarks_list and len(landmarks_list) == len(faces) else [None] * len(faces)
        
        # Process each face
        results = []
        for i, face in enumerate(faces):
            mask_status = mask_statuses[i] if i < len(mask_statuses) else "unknown"
            result = self.recognize(face, track_ids[i], mask_status, landmarks[i], force_refresh)
            results.append(result)
            
        return results
    
    def reset_cache(self):
        """Clear all recognition caches."""
        self.track_cache.clear()
        self.image_cache.clear()
        self.cache_timestamps.clear()
        logger.info("Face recognition caches cleared")
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_process = np.mean(self.process_times) * 1000 if self.process_times else 0
        avg_embedding = np.mean(self.embedding_times) * 1000 if self.embedding_times else 0
        
        return {
            "avg_process_time_ms": avg_process,
            "avg_embedding_time_ms": avg_embedding,
            "track_cache_size": len(self.track_cache),
            "image_cache_size": len(self.image_cache),
            "model_loaded": self.model_loaded,
            "embeddings_loaded": self.embeddings_loaded,
            "database_size": len(self.embeddings_db),
            "index_size": self.hnsw_index.get_current_count() if self.hnsw_index else 0
        }