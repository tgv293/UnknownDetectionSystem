"""Face embedding creation and management for recognition system."""
import os
import cv2
import numpy as np
import pickle
import hnswlib
import time
import logging
from typing import Dict, Optional, Tuple, List, Any, Set
from pathlib import Path
from tqdm import tqdm

from config import DATASET_IMAGES_DIR, MODEL_DIR, EMBEDDINGS_PATH, EMBEDDINGS_SOURCE_INFO_PATH
from static.models.face_recognition.arcface import ArcFace
from preprocessing import resize_for_face_recognition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('embedding_errors.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
ARCFACE_MODEL_PATH = os.path.join(MODEL_DIR, "face_recognition", "r100_arcface_glint.onnx")


class EmbeddingCreator:
    """Creates and manages face embeddings for recognition."""
    
    def __init__(self, model_path: Optional[str] = None, use_augmentation: bool = False):
        """Initialize with model path and augmentation settings."""
        self.model_path = model_path or ARCFACE_MODEL_PATH
        self.use_augmentation = use_augmentation
        
        # Core components
        self.model = self._load_model()
        self.embeddings_db = {}
        self.embedding_source_info = {}
        self.error_records = {
            "failed_images": [], 
            "failed_people": set(),
            "null_embeddings": 0, 
            "error_count": 0
        }
        
        # HNSW configuration
        self.embedding_dimension = 512
        self.hnsw_index = None
        self.index_id_to_info = []
        self.ef_construction = 200
        self.M = 16
        self.ef_search = 50
        
        # Augmentation settings
        self.augmentation_types = ["flip", "rotate", "brightness"]
        self.rotation_angles = [-5, 5]
        self.brightness_factors = [0.9, 1.1]
        
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        logger.info(f"Initialized EmbeddingCreator {'with' if use_augmentation else 'without'} augmentation")
    
    def _load_model(self) -> ArcFace:
        """Load ArcFace model."""
        logger.info(f"Loading ArcFace model from {self.model_path}")
        try:
            return ArcFace(model_path=self.model_path)
        except Exception as e:
            logger.error(f"Failed to load ArcFace model: {e}")
            raise

    def extract_embedding(self, face_img: np.ndarray, landmarks=None) -> Optional[np.ndarray]:
        """Extract embedding from a face image."""
        try:
            if not self.use_augmentation:
                return self._extract_single_embedding(face_img, landmarks)
            return self._extract_augmented_embeddings(face_img, landmarks)
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            self.error_records["error_count"] += 1
            return None
            
    def _extract_single_embedding(self, face_img: np.ndarray, landmarks=None) -> Optional[np.ndarray]:
        """Extract embedding from a single image."""
        try:
            if landmarks is None:
                face_img = resize_for_face_recognition(face_img)
                if face_img is None:
                    return None
                embedding = self.model.get_feat(face_img)[0]
            else:
                embedding = self.model(face_img, landmarks)
                
            if embedding is None or np.isnan(embedding).any() or np.sum(embedding) == 0:
                self.error_records["null_embeddings"] += 1
                return None

            return embedding
        except Exception as e:
            logger.error(f"Error in single embedding extraction: {e}")
            return None
    
    def _create_augmented_images(self, face_img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Create augmented versions of an image."""
        augmented_images = [("original", face_img)]
        
        if not self.use_augmentation:
            return augmented_images
            
        try:
            h, w = face_img.shape[:2]
            center = (w // 2, h // 2)
            
            # Add horizontal flip
            if "flip" in self.augmentation_types:
                augmented_images.append(("flip", cv2.flip(face_img, 1)))
            
            # Add rotations
            if "rotate" in self.augmentation_types:
                for angle in self.rotation_angles:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    rotated = cv2.warpAffine(face_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    augmented_images.append((f"rotate_{angle}", rotated))
            
            # Add brightness variations
            if "brightness" in self.augmentation_types:
                for factor in self.brightness_factors:
                    brightened = cv2.convertScaleAbs(face_img, alpha=factor, beta=0)
                    augmented_images.append((f"bright_{factor:.1f}", brightened))
            
            return augmented_images
        except Exception as e:
            logger.error(f"Error during image augmentation: {e}")
            return [("original", face_img)]
            
    def _extract_augmented_embeddings(self, face_img: np.ndarray, landmarks=None) -> Optional[np.ndarray]:
        """Extract embeddings from augmented images and combine them."""
        valid_embeddings = []
        
        for aug_name, aug_img in self._create_augmented_images(face_img):
            try:
                if landmarks is None:
                    aug_img = resize_for_face_recognition(aug_img)
                    if aug_img is None:
                        continue
                    curr_embedding = self.model.get_feat(aug_img)[0]
                else:
                    curr_embedding = self.model(aug_img, landmarks)
                    
                if curr_embedding is None or np.isnan(curr_embedding).any() or np.sum(curr_embedding) == 0:
                    continue
                    
                valid_embeddings.append(curr_embedding)
            except Exception as e:
                logger.error(f"Error processing augmented image {aug_name}: {e}")
                continue
        
        if not valid_embeddings:
            return None
            
        # Use single embedding directly if only one is valid
        if len(valid_embeddings) == 1:
            return valid_embeddings[0]
            
        # Original image gets double weight
        weights = np.ones(len(valid_embeddings))
        if len(valid_embeddings) > 1:  # Only adjust weights if we have multiple embeddings
            weights[0] = 2.0
            weights = weights / np.sum(weights)
            
        # Calculate weighted average
        final_embedding = np.zeros_like(valid_embeddings[0])
        for i, emb in enumerate(valid_embeddings):
            final_embedding += emb * weights[i]
            
        # Normalize
        norm = np.linalg.norm(final_embedding)
        if norm > 1e-10:
            final_embedding = final_embedding / norm
    
        return final_embedding

    def _initialize_hnsw(self, max_elements: int = 10000) -> None:
        """Initialize HNSW index for similarity search."""
        self.hnsw_index = hnswlib.Index(space='cosine', dim=self.embedding_dimension)
        self.hnsw_index.init_index(
            max_elements=max_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )
        self.hnsw_index.set_ef(self.ef_search)
        self.index_id_to_info = []

    def _add_to_hnsw(self, person_name: str, mask_status: str, pose_name: str, embedding: np.ndarray) -> None:
        """Add an embedding to the HNSW index."""
        if self.hnsw_index is None:
            self._initialize_hnsw()
            
        index_id = len(self.index_id_to_info)
        self.index_id_to_info.append((person_name, mask_status, pose_name))
        self.hnsw_index.add_items(embedding, index_id)

    def process_dataset(self) -> bool:
        """Process all faces in the dataset and create embeddings."""
        start_time = time.time()
        
        # Validate dataset directory
        if not os.path.exists(DATASET_IMAGES_DIR):
            logger.error(f"Dataset directory {DATASET_IMAGES_DIR} does not exist")
            return False

        person_dirs = [d for d in os.listdir(DATASET_IMAGES_DIR) 
                      if os.path.isdir(os.path.join(DATASET_IMAGES_DIR, d))]
        
        if not person_dirs:
            logger.warning("No persons found in the dataset")
            return False

        logger.info(f"Found {len(person_dirs)} persons in the dataset")
        
        # Initialize data structures
        self.embeddings_db = {}
        self._initialize_hnsw()
        self.embedding_source_info = {}
        success = True
        
        # Process each person
        for person_name in tqdm(person_dirs, desc="Processing persons"):
            if not self._process_person(person_name):
                success = False
                
        # Check if we have valid embeddings
        if len(self.embeddings_db) == 0:
            logger.error("No valid embeddings were created")
            return False

        # Save results if successful
        if success and self.error_records["null_embeddings"] == 0:
            self.save_embeddings()
            elapsed = time.time() - start_time
            logger.info(f"Embedding creation completed in {elapsed:.2f} seconds")
            logger.info(f"Created embeddings for {len(self.embeddings_db)} people")
            return True
        else:
            logger.error(f"Embeddings NOT saved due to errors: "
                        f"{len(self.error_records['failed_images'])} failed images, "
                        f"{len(self.error_records['failed_people'])} failed people, "
                        f"{self.error_records['null_embeddings']} null embeddings")
            return False

    def _process_person(self, person_name: str) -> bool:
        """Process a person's directory to create embeddings."""
        person_dir = os.path.join(DATASET_IMAGES_DIR, person_name)
        self.embeddings_db[person_name] = {"no_mask": {}, "with_mask": {}}
        self.embedding_source_info[person_name] = {"no_mask": {}, "with_mask": {}}
        person_success = True

        # Process no_mask directory (required)
        no_mask_dir = os.path.join(person_dir, "no_mask")
        if os.path.exists(no_mask_dir):
            if not self._process_directory(no_mask_dir, person_name, "no_mask"):
                person_success = False
                logger.error(f"Failed to process no_mask images for {person_name}")
        else:
            logger.error(f"No 'no_mask' directory found for {person_name}")
            person_success = False

        # Process with_mask directory (optional)
        with_mask_dir = os.path.join(person_dir, "with_mask")
        if os.path.exists(with_mask_dir):
            self._process_directory(with_mask_dir, person_name, "with_mask")

        # Remove person if insufficient data
        if not person_success or not self.embeddings_db[person_name]["no_mask"]:
            logger.error(f"Removing {person_name} due to insufficient valid data")
            if person_name in self.embeddings_db:
                del self.embeddings_db[person_name]
            self.error_records["failed_people"].add(person_name)
            return False
        
        # Create combined embeddings
        self._create_combined_embeddings(person_name)
        return True

    def _create_combined_embeddings(self, person_name: str) -> None:
        """Create combined embeddings by averaging all embeddings for each mask status."""
        for mask_status in ["no_mask", "with_mask"]:
            poses = self.embeddings_db[person_name][mask_status]
            if not poses:
                continue
                
            # Calculate average embedding
            embeddings = list(poses.values())
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

    def _process_directory(self, directory: str, person_name: str, status: str) -> bool:
        """Process images in a directory."""
        mask_status = status  # Our mapping is already 1:1
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            logger.warning(f"No images found in {directory}")
            return False
            
        processed_count = 0

        for img_file in tqdm(image_files, desc=f"Processing {status} for {person_name}", leave=False):
            try:
                img_path = os.path.join(directory, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    self.error_records["failed_images"].append(img_path)
                    continue

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
                logger.error(f"Error processing {img_file}: {e}")
                self.error_records["failed_images"].append(os.path.join(directory, img_file))

        # Check if we processed sufficient images
        if processed_count < len(image_files) / 2:
            logger.error(f"Insufficient valid images for {person_name}/{status}: "
                        f"only {processed_count}/{len(image_files)} successful")
            return False

        return True

    def save_embeddings(self) -> bool:
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

            logger.info(f"Embeddings saved to {EMBEDDINGS_PATH}")
            logger.info(f"HNSW index saved to {hnsw_path} (using cosine similarity)")
            logger.info(f"Embeddings created {'with' if self.use_augmentation else 'without'} augmentation")
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False

    def load_embeddings(self) -> bool:
        """Load embeddings and HNSW index from disk."""
        try:
            if not os.path.exists(EMBEDDINGS_PATH):
                logger.warning(f"Embeddings file not found: {EMBEDDINGS_PATH}")
                return False
                
            with open(EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
            
            # Process loaded data
            if isinstance(data, dict) and "embeddings_db" in data:
                self.embeddings_db = data["embeddings_db"]
                self.index_id_to_info = data.get("index_id_to_info", [])
            else:
                # Handle old format
                self.embeddings_db = data
                logger.warning("Loading embeddings in old format")
            
            # Load HNSW index
            self._load_hnsw_index()

            # Load embedding source info
            self._load_embedding_source_info()

            logger.info(f"Embeddings loaded from {EMBEDDINGS_PATH}")
            return True
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False

    def _load_hnsw_index(self) -> None:
        """Load HNSW index from disk or rebuild if necessary."""
        hnsw_path = f"{os.path.splitext(EMBEDDINGS_PATH)[0]}_hnsw.bin"
        if os.path.exists(hnsw_path):
            try:
                self._initialize_hnsw()  # Create empty index first
                self.hnsw_index.load_index(hnsw_path, max_elements=len(self.index_id_to_info))
                logger.info(f"HNSW index loaded with {self.hnsw_index.get_current_count()} vectors")
                
                # Verify correct space type
                if not self._is_cosine_index():
                    logger.warning("HNSW index found but is not cosine type, rebuilding...")
                    self._rebuild_hnsw_index()
            except Exception as e:
                logger.error(f"Error loading HNSW index: {e}, rebuilding...")
                self._rebuild_hnsw_index()
        else:
            logger.warning("No HNSW index found, rebuilding with cosine similarity")
            self._rebuild_hnsw_index()

    def _load_embedding_source_info(self) -> None:
        """Load embedding source information."""
        if os.path.exists(EMBEDDINGS_SOURCE_INFO_PATH):
            try:
                with open(EMBEDDINGS_SOURCE_INFO_PATH, 'rb') as f:
                    self.embedding_source_info = pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load embedding source info: {e}")
                self.embedding_source_info = {}

    def _is_cosine_index(self) -> bool:
        """Check if HNSW index uses cosine similarity."""
        try:
            return self.hnsw_index.space == 'cosine'
        except:
            return False

    def _rebuild_hnsw_index(self) -> None:
        """Rebuild HNSW index from embeddings_db."""
        count = sum(len(poses) for person_data in self.embeddings_db.values() 
                  for poses in person_data.values())
        max_elements = max(10000, count)
        
        self._initialize_hnsw(max_elements=max_elements)
        self.index_id_to_info = []
        index_id = 0
        
        for person_name, data in self.embeddings_db.items():
            for mask_status, poses in data.items():
                for pose_name, embedding in poses.items():
                    if embedding is None:
                        continue
                        
                    # Normalize embedding if needed
                    norm = np.linalg.norm(embedding)
                    if not (0.99 <= norm <= 1.01) and norm > 1e-10:
                        embedding = embedding / norm
                    
                    self.index_id_to_info.append((person_name, mask_status, pose_name))
                    self.hnsw_index.add_items(embedding, index_id)
                    index_id += 1
        
        logger.info(f"Rebuilt HNSW index with {self.hnsw_index.get_current_count()} vectors")

    def analyze_embedding_distribution(self) -> Dict:
        """Analyze distribution of similarities between embeddings."""
        if not self.hnsw_index or self.hnsw_index.get_current_count() == 0:
            logger.warning("No embeddings available for analysis")
            return {}
            
        # Collect similarity data
        similarities = self._collect_similarity_data()
        if not similarities or not similarities.get("same_person"):
            logger.warning("Insufficient data for analysis")
            return {}
            
        return self._calculate_statistics(similarities)
        
    def _collect_similarity_data(self) -> Dict:
        """Collect similarity data between embeddings."""
        same_person_similarities = []
        diff_person_similarities = []
        same_person_mask_to_nomask = []
        same_person_same_status = []

        for person, data in tqdm(self.embeddings_db.items(), desc="Analyzing embeddings"):
            person_embeddings = []
            for status, poses in data.items():
                for pose, embedding in poses.items():
                    if embedding is None:
                        continue
                        
                    # Ensure embedding is normalized
                    norm = np.linalg.norm(embedding)
                    if not (0.99 <= norm <= 1.01) and norm > 1e-10:
                        embedding = embedding / norm
                    
                    person_embeddings.append((embedding, status, pose))
            
            # Compare with nearest neighbors
            for emb1, status1, pose1 in person_embeddings:
                k = min(50, self.hnsw_index.get_current_count())
                labels, distances = self.hnsw_index.knn_query(emb1, k=k)
                
                # Convert distances to similarities
                similarities = 1 - np.array(distances)[0]
                
                for similarity, idx in zip(similarities, labels[0]):
                    if idx >= len(self.index_id_to_info):
                        continue
                        
                    other_person, other_status, other_pose = self.index_id_to_info[idx]
                    
                    # Skip self-comparison
                    if other_person == person and other_status == status1 and other_pose == pose1:
                        continue
                        
                    if other_person == person:
                        same_person_similarities.append(float(similarity))
                        
                        if status1 != other_status:
                            same_person_mask_to_nomask.append(float(similarity))
                        else:
                            same_person_same_status.append(float(similarity))
                    else:
                        diff_person_similarities.append(float(similarity))
        
        return {
            "same_person": same_person_similarities,
            "diff_person": diff_person_similarities,
            "mask_to_nomask": same_person_mask_to_nomask,
            "same_status": same_person_same_status
        }
        
    def _calculate_statistics(self, similarity_data: Dict) -> Dict:
        """Calculate statistics from similarity data."""
        same_person = similarity_data["same_person"]
        diff_person = similarity_data["diff_person"]
        mask_to_nomask = similarity_data["mask_to_nomask"]
        same_status = similarity_data["same_status"]
        
        same_mean = np.mean(same_person)
        diff_mean = np.mean(diff_person)
        separation = same_mean - diff_mean

        # Calculate optimal threshold
        best_threshold, min_error = self._calculate_optimal_threshold(same_person, diff_person)
        
        # Calculate mask impact if data available
        mask_impact = None
        if mask_to_nomask and same_status:
            mask_to_nomask_mean = np.mean(mask_to_nomask)
            same_status_mean = np.mean(same_status)
            mask_impact = same_status_mean - mask_to_nomask_mean

        # Log the results
        logger.info(f"\nSame person similarity: {same_mean:.4f} | Different person: {diff_mean:.4f}")
        logger.info(f"Separation margin: {separation:.4f} | Recommended threshold: {best_threshold:.2f}")
        if mask_impact is not None:
            logger.info(f"Mask impact: {mask_impact:.4f} lower similarity")

        return {
            "same_person_mean": same_mean,
            "diff_person_mean": diff_mean,
            "separation": separation,
            "recommended_threshold": best_threshold,
            "error_rate": min_error,
            "mask_impact": mask_impact
        }
        
    def _calculate_optimal_threshold(self, same_arr: List[float], diff_arr: List[float]) -> Tuple[float, float]:
        """Calculate optimal threshold for classification."""
        same_arr = np.array(same_arr)
        diff_arr = np.array(diff_arr)
        
        min_error = float('inf')
        best_threshold = 0.5

        # Search for best threshold
        for threshold in np.arange(0.1, 1.0, 0.01):
            false_reject = np.sum(same_arr < threshold) / len(same_arr) if len(same_arr) > 0 else 0
            false_accept = np.sum(diff_arr >= threshold) / len(diff_arr) if len(diff_arr) > 0 else 0
            error_rate = false_reject + false_accept

            if error_rate < min_error:
                min_error = error_rate
                best_threshold = threshold
                
        return best_threshold, min_error


def print_statistics(embeddings_db: Dict) -> None:
    """Print statistics about the embeddings database."""
    total_people = len(embeddings_db)
    
    no_mask_regular = sum(len(data["no_mask"]) - (1 if "combined" in data["no_mask"] else 0) 
                          for data in embeddings_db.values())
    with_mask_regular = sum(len(data["with_mask"]) - (1 if "combined" in data["with_mask"] else 0) 
                           for data in embeddings_db.values())
    no_mask_combined = sum(1 if "combined" in data["no_mask"] else 0 for data in embeddings_db.values())
    with_mask_combined = sum(1 if "combined" in data["with_mask"] else 0 for data in embeddings_db.values())

    logger.info(f"\nGenerated embeddings for {total_people} people:")
    logger.info(f"- No mask: {no_mask_regular} regular, {no_mask_combined} combined")
    logger.info(f"- With mask: {with_mask_regular} regular, {with_mask_combined} combined")


def main() -> int:
    """Main function to run the embedding creation."""
    import argparse
    parser = argparse.ArgumentParser(description='Create or update face embeddings')
    parser.add_argument('--validate', action='store_true', help='Validate database only')
    parser.add_argument('--analyze', action='store_true', help='Analyze embedding distributions')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation for embeddings')
    args = parser.parse_args()

    logger.info(f"Creating face embeddings from {DATASET_IMAGES_DIR}")
    creator = EmbeddingCreator(use_augmentation=args.augment)

    try:
        if args.validate:
            if creator.load_embeddings():
                print_statistics(creator.embeddings_db)
                logger.info("All embeddings validated successfully!")
                return 0
            logger.error("Failed to load embeddings for validation")
            return 1
        elif args.analyze:
            if creator.load_embeddings():
                creator.analyze_embedding_distribution()
                return 0
            logger.error("Failed to load embeddings for analysis")
            return 1
        else:
            if creator.process_dataset():
                print_statistics(creator.embeddings_db)
                creator.analyze_embedding_distribution()
                return 0
            logger.error("Failed to create embeddings database")
            return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())