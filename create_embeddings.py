import os
import cv2
import numpy as np
import pickle
import hnswlib
import sys
import time
import logging
import argparse
from typing import Dict, Optional, Any, Tuple, List, Set
from pathlib import Path
from tqdm import tqdm

from config import DATASET_DIR, MODEL_DIR, EMBEDDINGS_PATH, EMBEDDINGS_SOURCE_INFO_PATH
from models.face_recognition.arcface import ArcFace
# Thay đổi import để sử dụng hàm resize chuyên dụng
from preprocessing import resize_for_face_recognition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('embedding_errors.log'), logging.StreamHandler()]
)

# Đường dẫn mô hình ArcFace
ARCFACE_MODEL_PATH = os.path.join(MODEL_DIR, "r100_arcface_glint.onnx")

class EmbeddingCreator:
    """Creates, manages and stores face embeddings for recognition."""
    
    def __init__(self, model_path: Optional[str] = None, use_augmentation: bool = False):
        self.model_path = model_path or ARCFACE_MODEL_PATH
        self.model = self._create_model()
        self.embeddings_db = {}
        self.error_records = {
            "failed_images": [], 
            "failed_people": set(),
            "null_embeddings": 0, 
            "error_count": 0
        }
        self.embedding_source_info = {}
        self.mask_status_mapping = {
            "no_mask": "no_mask",
            "with_mask": "with_mask"
        }
        
        # HNSW index configuration
        self.embedding_dimension = 512  # ArcFace thường có 512 chiều
        self.hnsw_index = None
        self.index_id_to_info = []
        self.ef_construction = 200  # Higher = more accurate but slower construction
        self.M = 16                 # Number of connections per element
        self.ef_search = 50         # Higher = more accurate but slower search
        
        # Quality settings
        self.normalize_embeddings = True
        
        # Augmentation settings
        self.use_augmentation = use_augmentation
        self.augmentation_types = ["flip", "rotate", "brightness"]
        self.rotation_angles = [-5, 5]
        self.brightness_factors = [0.9, 1.1]
        
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        
        status = "with" if self.use_augmentation else "without"
        print(f"Initialized EmbeddingCreator {status} augmentation")
    
    def _create_model(self) -> ArcFace:
        """Create and load ArcFace model."""
        print(f"Loading ArcFace model from {self.model_path}...")
        try:
            model = ArcFace(model_path=self.model_path)
            print("ArcFace model loaded successfully")
            return model
        except Exception as e:
            print(f"CRITICAL ERROR creating ArcFace model: {e}")
            sys.exit(1)

    def _create_augmented_images(self, face_img):
        """Create augmented versions of an input face image."""
        augmented_images = [("original", face_img)]
        
        if not self.use_augmentation:
            return augmented_images
            
        try:
            h, w = face_img.shape[:2]
            center = (w // 2, h // 2)
            
            # Horizontal flip
            if "flip" in self.augmentation_types:
                flipped = cv2.flip(face_img, 1)
                augmented_images.append(("flip", flipped))
            
            # Rotation augmentations
            if "rotate" in self.augmentation_types:
                for angle in self.rotation_angles:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(face_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    augmented_images.append((f"rotate_{angle}", rotated))
            
            # Brightness variations
            if "brightness" in self.augmentation_types:
                for factor in self.brightness_factors:
                    brightened = cv2.convertScaleAbs(face_img, alpha=factor, beta=0)
                    augmented_images.append((f"bright_{factor:.1f}", brightened))
            
            return augmented_images
        except Exception as e:
            logging.error(f"Error during image augmentation: {e}")
            return [("original", face_img)]

    def extract_embedding(self, face_img, landmarks=None):
        """Extract face embedding from an image."""
        try:
            if not self.use_augmentation:
                return self._extract_single_embedding(face_img, landmarks)
            else:
                return self._extract_augmented_embeddings(face_img, landmarks)
        except Exception as e:
            logging.error(f"Error extracting embedding: {e}")
            self.error_records["error_count"] += 1
            return None
            
    def _extract_single_embedding(self, face_img, landmarks=None):
        """Extract embedding from a single face image."""
        try:
            # ArcFace cần face và landmarks
            if landmarks is None:
                # Nếu không có landmarks, xử lý ảnh trực tiếp
                # Thay đổi: Sử dụng hàm resize_for_face_recognition từ preprocessing
                face_img = resize_for_face_recognition(face_img)
                if face_img is None:
                    return None
                embedding = self.model.get_feat(face_img)[0]
            else:
                # Sử dụng landmarks để căn chỉnh khuôn mặt
                embedding = self.model(face_img, landmarks)
                
            if embedding is None or np.isnan(embedding).any() or np.sum(embedding) == 0:
                self.error_records["null_embeddings"] += 1
                return None

            return embedding
        except Exception as e:
            logging.error(f"Error in _extract_single_embedding: {e}")
            return None
            
    def _extract_augmented_embeddings(self, face_img, landmarks=None):
        """Extract and combine embeddings from augmented versions of a face image."""
        augmented_images = self._create_augmented_images(face_img)
        if not augmented_images:
            return None
        
        valid_embeddings = []
        
        for aug_name, aug_img in augmented_images:
            try:
                if landmarks is None:
                    # Thay đổi: Sử dụng hàm resize_for_face_recognition từ preprocessing
                    aug_img = resize_for_face_recognition(aug_img)
                    if aug_img is None:
                        continue
                    curr_embedding = self.model.get_feat(aug_img)[0]
                else:
                    # Sử dụng landmarks để căn chỉnh
                    curr_embedding = self.model(aug_img, landmarks)
                    
                if curr_embedding is None or np.isnan(curr_embedding).any() or np.sum(curr_embedding) == 0:
                    continue
                    
                valid_embeddings.append(curr_embedding)
            except Exception as e:
                logging.error(f"Error processing augmented image {aug_name}: {e}")
                continue
        
        if not valid_embeddings:
            return None
            
        # Average the embeddings with weighting
        if len(valid_embeddings) == 1:
            final_embedding = valid_embeddings[0]
        else:
            # Original image gets double weight
            weights = np.ones(len(valid_embeddings))
            if augmented_images[0][0] == "original":
                weights[0] = 2.0
            weights = weights / np.sum(weights)
            
            final_embedding = np.zeros_like(valid_embeddings[0])
            for i, emb in enumerate(valid_embeddings):
                final_embedding += emb * weights[i]
                
            # Normalize the final averaged embedding
            norm = np.linalg.norm(final_embedding)
            if norm > 1e-10:
                final_embedding = final_embedding / norm
        
        return final_embedding

    def _initialize_hnsw(self):
        """Initialize HNSW index for efficient similarity search."""
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dimension)
        self.hnsw_index.init_index(
            max_elements=10000,
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.hnsw_index.set_ef(self.ef_search)
        self.index_id_to_info = []

    def _add_to_hnsw(self, person_name, mask_status, pose_name, embedding):
        """Add an embedding to the HNSW index."""
        if self.hnsw_index is None:
            self._initialize_hnsw()
            
        index_id = len(self.index_id_to_info)
        self.index_id_to_info.append((person_name, mask_status, pose_name))
        self.hnsw_index.add_items(embedding, index_id)

    def process_dataset(self):
        """Process all faces in the dataset and create embeddings."""
        start_time = time.time()
        success = True

        if not os.path.exists(DATASET_DIR):
            logging.error(f"Dataset directory {DATASET_DIR} does not exist")
            print(f"ERROR: Dataset directory {DATASET_DIR} does not exist")
            sys.exit(1)

        person_dirs = [d for d in os.listdir(DATASET_DIR) 
                      if os.path.isdir(os.path.join(DATASET_DIR, d))]
        if not person_dirs:
            logging.warning("No persons found in the dataset")
            print("ERROR: No persons found in the dataset directory")
            sys.exit(1)

        print(f"Found {len(person_dirs)} persons in the dataset")
        
        # Initialize data structures
        self.embeddings_db = {}
        self._initialize_hnsw()
        self.embedding_source_info = {}
        
        print("Creating embeddings for all persons")

        for person_name in tqdm(person_dirs, desc="Processing persons"):
            person_dir = os.path.join(DATASET_DIR, person_name)
            self.embeddings_db[person_name] = {"no_mask": {}, "with_mask": {}}
            self.embedding_source_info[person_name] = {"no_mask": {}, "with_mask": {}}
            person_success = True

            # Process no_mask directory (required)
            no_mask_dir = os.path.join(person_dir, "no_mask")
            if os.path.exists(no_mask_dir):
                mask_success = self._process_directory(no_mask_dir, person_name, "no_mask")
                if not mask_success:
                    person_success = False
                    logging.error(f"Failed to process no_mask images for {person_name}")
            else:
                logging.error(f"No 'no_mask' directory found for {person_name}")
                person_success = False

            # Process with_mask directory (optional)
            with_mask_dir = os.path.join(person_dir, "with_mask")
            if os.path.exists(with_mask_dir):
                self._process_directory(with_mask_dir, person_name, "with_mask")

            # Validate person data
            if not person_success or len(self.embeddings_db[person_name]["no_mask"]) == 0:
                logging.error(f"Removing {person_name} due to insufficient valid data")
                if person_name in self.embeddings_db:
                    del self.embeddings_db[person_name]
                self.error_records["failed_people"].add(person_name)
                success = False
                
            # Create combined embeddings for each mask status
            if person_name in self.embeddings_db:
                self._create_combined_embeddings(person_name)

        if len(self.embeddings_db) == 0:
            logging.error("No valid embeddings were created")
            print("ERROR: No valid embeddings were created")
            sys.exit(1)

        if success and self.error_records["null_embeddings"] == 0:
            self.save_embeddings()
            elapsed = time.time() - start_time
            print(f"Embedding creation completed in {elapsed:.2f} seconds")
            print(f"Created embeddings for {len(self.embeddings_db)} people")
            return True
        else:
            logging.error(f"Embeddings NOT saved due to errors: "
                        f"{len(self.error_records['failed_images'])} failed images, "
                        f"{len(self.error_records['failed_people'])} failed people, "
                        f"{self.error_records['null_embeddings']} null embeddings")
            print("ERROR: Failed to create valid embeddings")
            sys.exit(1)

    def _create_combined_embeddings(self, person_name):
        """Create combined embeddings by averaging all embeddings for each mask status."""
        for mask_status in ["no_mask", "with_mask"]:
            poses = self.embeddings_db[person_name][mask_status]
            if poses:
                embeddings = list(poses.values())
                if embeddings:
                    # Calculate average embedding
                    combined = np.mean(embeddings, axis=0)
                    # Normalize
                    norm = np.linalg.norm(combined)
                    if norm > 1e-10:
                        combined = combined / norm
                        # Add combined embedding
                        pose_name = "combined"
                        self.embeddings_db[person_name][mask_status][pose_name] = combined
                        self.embedding_source_info[person_name][mask_status][pose_name] = "combined"
                        self._add_to_hnsw(person_name, mask_status, pose_name, combined)

    def _process_directory(self, directory, person_name, status):
        """Process all images in a directory."""
        mask_status = self.mask_status_mapping.get(status, status)
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        success = len(image_files) > 0
        processed_count = 0

        for img_file in tqdm(image_files, desc=f"Processing {status} for {person_name}", leave=False):
            try:
                img_path = os.path.join(directory, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    self.error_records["failed_images"].append(img_path)
                    continue

                # Giả định ảnh đã được căn chỉnh trong quá trình thu thập
                embedding = self.extract_embedding(img)

                if embedding is not None:
                    pose_name = os.path.splitext(img_file)[0]
                    self.embeddings_db[person_name][mask_status][pose_name] = embedding
                    self.embedding_source_info[person_name][mask_status][pose_name] = status
                    self._add_to_hnsw(person_name, mask_status, pose_name, embedding)
                    processed_count += 1
                else:
                    self.error_records["failed_images"].append(img_path)
            except Exception as e:
                logging.error(f"Error processing {img_file}: {e}")
                self.error_records["failed_images"].append(os.path.join(directory, img_file))

        if processed_count < len(image_files) / 2 and len(image_files) > 0:
            logging.error(f"Insufficient valid images for {person_name}/{status}: "
                        f"only {processed_count}/{len(image_files)} successful")
            success = False

        return success

    def save_embeddings(self):
        """Save embeddings and HNSW index to disk."""
        try:
            # Save main embeddings data
            embedding_data = {
                "embeddings_db": self.embeddings_db,
                "index_id_to_info": self.index_id_to_info,
                "distance_metric": "cosine",
                "with_augmentation": self.use_augmentation,
                "model": "r100_arcface_glint.onnx"
            }
            
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(embedding_data, f)

            # Save HNSW index
            hnsw_path = f"{os.path.splitext(EMBEDDINGS_PATH)[0]}_hnsw.bin"
            self.hnsw_index.save_index(hnsw_path)

            # Save source info
            with open(EMBEDDINGS_SOURCE_INFO_PATH, 'wb') as f:
                pickle.dump(self.embedding_source_info, f)

            print(f"Embeddings saved to {EMBEDDINGS_PATH}")
            print(f"HNSW index saved to {hnsw_path} (using cosine similarity)")
            aug_status = "with" if self.use_augmentation else "without"
            print(f"Embeddings created {aug_status} augmentation")
            return True
        except Exception as e:
            logging.error(f"Error saving embeddings: {e}")
            print(f"ERROR: Failed to save embeddings: {e}")
            sys.exit(1)

    def load_embeddings(self):
        """Load embeddings and HNSW index from disk."""
        try:
            if os.path.exists(EMBEDDINGS_PATH):
                with open(EMBEDDINGS_PATH, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, dict) and "embeddings_db" in data:
                    self.embeddings_db = data["embeddings_db"]
                    self.index_id_to_info = data.get("index_id_to_info", [])
                    
                    distance_metric = data.get("distance_metric", "L2")
                    if distance_metric != "cosine":
                        print(f"Warning: Loading embeddings with {distance_metric} distance metric, converting to cosine")
                        
                    # Check if embeddings were created with augmentation
                    was_augmented = data.get("with_augmentation", False)
                    if was_augmented != self.use_augmentation:
                        print(f"Note: Loaded embeddings were created with{'out' if not was_augmented else ''} augmentation")
                        print(f"Current setting: {self.use_augmentation}")
                else:
                    self.embeddings_db = data
                    print("Warning: Loading embeddings in old format, converting to cosine similarity")
                
                # Load HNSW index
                hnsw_path = f"{os.path.splitext(EMBEDDINGS_PATH)[0]}_hnsw.bin"
                if os.path.exists(hnsw_path):
                    try:
                        self._initialize_hnsw()  # Create empty index first
                        self.hnsw_index.load_index(hnsw_path, max_elements=len(self.index_id_to_info))
                        print(f"HNSW index loaded with {self.hnsw_index.get_current_count()} vectors")
                        
                        # Verify correct space type
                        if not self.is_cosine_index(self.hnsw_index):
                            print("HNSW index found but is not cosine type, rebuilding...")
                            self._rebuild_hnsw_index()
                    except Exception as e:
                        print(f"Error loading HNSW index: {e}, rebuilding...")
                        self._rebuild_hnsw_index()
                else:
                    print("No HNSW index found, rebuilding with cosine similarity")
                    self._rebuild_hnsw_index()

                # Load embedding source info
                if os.path.exists(EMBEDDINGS_SOURCE_INFO_PATH):
                    try:
                        with open(EMBEDDINGS_SOURCE_INFO_PATH, 'rb') as f:
                            self.embedding_source_info = pickle.load(f)
                    except Exception:
                        pass

                print(f"Embeddings loaded from {EMBEDDINGS_PATH}")
                return True
            return False
        except Exception as e:
            logging.error(f"Error loading embeddings: {e}")
            print(f"ERROR: Failed to load embeddings: {e}")
            return False

    def is_cosine_index(self, index):
        """Check if HNSW index uses cosine similarity."""
        try:
            return index.space == 'cosine'
        except:
            return False

    def _rebuild_hnsw_index(self):
        """Rebuild HNSW index from embeddings_db."""
        self._initialize_hnsw()
        self.index_id_to_info = []
        index_id = 0
        
        max_elements = 0
        for person_name, data in self.embeddings_db.items():
            for mask_status, poses in data.items():
                max_elements += len(poses)
        
        # Resize index if needed
        if max_elements > self.hnsw_index.get_max_elements():
            self.hnsw_index.resize_index(max_elements)
        
        for person_name, data in self.embeddings_db.items():
            for mask_status, poses in data.items():
                for pose_name, embedding in poses.items():
                    if embedding is not None:
                        # Check if embedding is already normalized
                        norm = np.linalg.norm(embedding)
                        if not (0.99 <= norm <= 1.01):
                            # Only normalize if not already normalized
                            if norm > 1e-10:
                                embedding = embedding / norm
                            else:
                                continue
                        
                        self.index_id_to_info.append((person_name, mask_status, pose_name))
                        self.hnsw_index.add_items(embedding, index_id)
                        index_id += 1
        
        print(f"Rebuilt HNSW index with {self.hnsw_index.get_current_count()} vectors")

    def analyze_embedding_distribution(self):
        """Analyze distribution of similarities between embeddings."""
        same_person_similarities = []
        diff_person_similarities = []
        same_person_mask_to_nomask = []
        same_person_same_status = []

        if self.hnsw_index and self.hnsw_index.get_current_count() > 0:
            for person, data in tqdm(self.embeddings_db.items(), desc="Analyzing embedding distributions"):
                person_embeddings = []
                
                for status, poses in data.items():
                    for pose, embedding in poses.items():
                        if embedding is not None:
                            # Check if embedding is normalized
                            norm = np.linalg.norm(embedding)
                            if not (0.99 <= norm <= 1.01):
                                # Normalize if needed
                                if norm > 1e-10:
                                    embedding = embedding / norm
                                else:
                                    continue
                            
                            person_embeddings.append((embedding, status, pose))
                
                # Compare each embedding with all others
                for _, (emb1, status1, pose1) in enumerate(person_embeddings):
                    # For cosine similarity, we need to search k nearest neighbors
                    k = min(50, self.hnsw_index.get_current_count())
                    labels, distances = self.hnsw_index.knn_query(emb1, k=k)
                    
                    # Convert cosine distances to similarities (1 - distance)
                    similarities = 1 - np.array(distances)[0]
                    
                    for similarity, idx in zip(similarities, labels[0]):
                        if idx >= len(self.index_id_to_info):
                            continue
                            
                        other_person, other_status, other_pose = self.index_id_to_info[idx]
                        
                        if other_person == person and other_status == status1 and other_pose == pose1:
                            continue  # Skip comparing with itself
                            
                        if other_person == person:
                            same_person_similarities.append(float(similarity))
                            
                            # Additional analysis for same person
                            if status1 != other_status:
                                same_person_mask_to_nomask.append(float(similarity))
                            else:
                                same_person_same_status.append(float(similarity))
                        else:
                            diff_person_similarities.append(float(similarity))
        
        # Basic analysis
        if same_person_similarities and diff_person_similarities:
            same_mean = np.mean(same_person_similarities)
            diff_mean = np.mean(diff_person_similarities)

            print("\nEmbedding Similarity Analysis (using HNSW with cosine similarity):")
            print(f"Same person mean similarity: {same_mean:.4f} (higher is better)")
            print(f"Different person mean similarity: {diff_mean:.4f}")
            print(f"Separation margin: {same_mean - diff_mean:.4f} (higher is better)")

            # Additional detailed analysis
            if same_person_mask_to_nomask and same_person_same_status:
                mask_to_nomask_mean = np.mean(same_person_mask_to_nomask)
                same_status_mean = np.mean(same_person_same_status)
                
                print("\nDetailed Similarity Analysis:")
                print(f"Same person, same mask status: {same_status_mean:.4f}")
                print(f"Same person, different mask status: {mask_to_nomask_mean:.4f}")
                print(f"Mask impact on similarity: {(same_status_mean - mask_to_nomask_mean):.4f} lower")

            # Calculate threshold
            same_arr = np.array(same_person_similarities)
            diff_arr = np.array(diff_person_similarities)

            min_error = float('inf')
            best_threshold = 0.5

            for threshold in np.arange(0.1, 1.0, 0.01):
                false_reject = np.sum(same_arr < threshold) / len(same_arr) if len(same_arr) > 0 else 0
                false_accept = np.sum(diff_arr >= threshold) / len(diff_arr) if len(diff_arr) > 0 else 0
                error_rate = false_reject + false_accept

                if error_rate < min_error:
                    min_error = error_rate
                    best_threshold = threshold

            print(f"Recommended cosine similarity threshold: {best_threshold:.2f}")
            print(f"Error rate at this threshold: {min_error:.4f}")
            print(f"Note: With cosine similarity, HIGHER values mean more similarity")
            print(f"Augmentation: {'Enabled' if self.use_augmentation else 'Disabled'}")
            print(f"Model: r100_arcface_glint.onnx")

            return {
                "same_person_mean": same_mean,
                "diff_person_mean": diff_mean,
                "separation": same_mean - diff_mean,
                "recommended_threshold": best_threshold,
                "error_rate": min_error,
                "mask_impact": (same_status_mean - mask_to_nomask_mean) 
                               if same_person_mask_to_nomask and same_person_same_status else None
            }
        return None


def print_statistics(embeddings_db, embedding_source_info=None):
    """Print statistics about the embeddings database."""
    total_people = len(embeddings_db)
    
    # Count regular and combined embeddings separately
    total_no_mask_regular = 0
    total_with_mask_regular = 0
    total_no_mask_combined = 0
    total_with_mask_combined = 0
    
    for person_data in embeddings_db.values():
        for pose_name in person_data["no_mask"]:
            if pose_name == "combined":
                total_no_mask_combined += 1
            else:
                total_no_mask_regular += 1
                
        for pose_name in person_data["with_mask"]:
            if pose_name == "combined":
                total_with_mask_combined += 1
            else:
                total_with_mask_regular += 1

    print(f"\nGenerated embeddings for {total_people} people:")
    print(f"- Total no_mask: {total_no_mask_regular} regular embeddings, {total_no_mask_combined} combined")
    print(f"- Total with_mask: {total_with_mask_regular} regular embeddings, {total_with_mask_combined} combined")

    print("\nEmbeddings by person:")
    for person, data in embeddings_db.items():
        no_mask = sum(1 for p in data["no_mask"] if p != "combined")
        with_mask = sum(1 for p in data["with_mask"] if p != "combined")
        no_mask_combined = 1 if "combined" in data["no_mask"] else 0
        with_mask_combined = 1 if "combined" in data["with_mask"] else 0
        
        combined_info = f" (+{no_mask_combined + with_mask_combined} combined)" if no_mask_combined + with_mask_combined > 0 else ""
        print(f"- {person}: {no_mask+with_mask} embeddings ({no_mask} no_mask, {with_mask} with_mask){combined_info}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create or update face embeddings')
    parser.add_argument('--validate', action='store_true', help='Validate database only')
    parser.add_argument('--analyze', action='store_true', help='Analyze embedding distributions')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation for robust embeddings')
    args = parser.parse_args()

    print(f"Creating face embeddings from {DATASET_DIR} using ArcFace model: r100_arcface_glint.onnx")
    creator = EmbeddingCreator(use_augmentation=args.augment)

    print("Using pre-aligned images from add_person.py (160x160)")
    if args.augment:
        print("Data augmentation ENABLED: Will create additional embeddings from augmented images")

    if args.validate:
        creator.load_embeddings()
        print_statistics(creator.embeddings_db)
        print("All embeddings validated successfully!")
    elif args.analyze:
        creator.load_embeddings()
        creator.analyze_embedding_distribution()
    else:
        success = creator.process_dataset()
        if success:
            print_statistics(creator.embeddings_db)
            creator.analyze_embedding_distribution()
        else:
            print("Failed to create embeddings database. No data was saved.")
            sys.exit(1)