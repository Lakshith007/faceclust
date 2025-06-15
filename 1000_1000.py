import os
import pickle
import logging
import numpy as np
from deepface import DeepFace
import faiss
from tqdm import tqdm
import cv2
import sys
import shutil
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import uuid

# Set up logging
log_dir = "face_clustering_logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(log_dir, "app.log"))
    ]
)
logger = logging.getLogger(__name__)

class FaceClusterer:
    def __init__(self, input_folder, output_folder, model_name="ArcFace", detector_backend="retinaface"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.model_name = model_name
        self.detector_backend = detector_backend

        # Initialize model once to avoid repeated loading
        try:
            self.embedding_model = DeepFace.build_model(self.model_name)
            logger.info(f"Successfully loaded embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            self.embedding_model = None
            return

        # Configuration
        self.cache_file = os.path.join(output_folder, "face_clusters_cache.pkl")
        self.index_file = os.path.join(output_folder, "faiss_index.index")
        self.metadata_file = os.path.join(output_folder, "cluster_metadata.pkl")

        # Create directories
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.faces_folder = os.path.join(self.output_folder, "extracted_faces")
        self.clusters_folder = os.path.join(self.output_folder, "clusters")
        self.representative_folder = os.path.join(self.output_folder, "representative_faces")
        Path(self.faces_folder).mkdir(parents=True, exist_ok=True)
        Path(self.clusters_folder).mkdir(parents=True, exist_ok=True)
        Path(self.representative_folder).mkdir(parents=True, exist_ok=True)

        self.face_paths = []
        self.embeddings = None
        self.labels = None
        self.index = None
        self.cluster_metadata = {}
        self.face_to_original = {}
        self.representative_to_cluster = {}

        self.load_existing_data()

    def load_existing_data(self):
        if os.path.exists(self.cache_file) and os.path.exists(self.index_file):
            try:
                logger.info("Loading existing clustered data...")
                with open(self.cache_file, "rb") as f:
                    data = pickle.load(f)
                    self.face_paths = data["face_paths"]
                    self.embeddings = data["embeddings"]
                    self.labels = data["labels"]
                    self.face_to_original = data.get("face_to_original", {})
                self.index = faiss.read_index(self.index_file)
                if os.path.exists(self.metadata_file):
                    with open(self.metadata_file, "rb") as f:
                        self.cluster_metadata = pickle.load(f)
                logger.info(f"Loaded {len(self.face_paths)} faces in {len(set(self.labels)) - 1 if self.labels is not None else 0} clusters")
            except Exception as e:
                logger.error(f"Failed to load existing data: {str(e)}")

    def save_current_state(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump({
                    "face_paths": self.face_paths,
                    "embeddings": self.embeddings,
                    "labels": self.labels,
                    "face_to_original": self.face_to_original
                }, f)
            if self.index is not None:
                faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, "wb") as f:
                pickle.dump(self.cluster_metadata, f)
            logger.info("Saved current clustering state")
        except Exception as e:
            logger.error(f"Failed to save state: {str(e)}")

    def extract_faces_from_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                return []

            face_objs = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=False
            )

            extracted_faces = []
            for i, face_obj in enumerate(face_objs):
                facial_area = face_obj["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]

                face_img = img[y:y+h, x:x+w]
                face_filename = f"{Path(img_path).stem}_face_{i}.jpg"
                face_path = os.path.join(self.faces_folder, face_filename)
                cv2.imwrite(face_path, face_img)

                self.face_to_original[face_path] = {
                    'original_path': img_path,
                    'bbox': (x, y, w, h)
                }

                extracted_faces.append(face_path)
            return extracted_faces
        except Exception as e:
            logger.warning(f"Failed to process image {img_path}: {str(e)}")
            return []

    def extract_all_faces(self):
        logger.info("Extracting faces from images...")
        image_files = [
            os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

        all_faces = []
        cpu_count = os.cpu_count() if os.cpu_count() is not None else 8
        with ThreadPoolExecutor(max_workers=min(8, cpu_count)) as executor:
            futures = [executor.submit(self.extract_faces_from_image, img_path) for img_path in image_files]
            for future in tqdm(as_completed(futures), total=len(image_files), desc="Extracting faces"):
                faces = future.result()
                all_faces.extend(faces)
        logger.info(f"Extracted {len(all_faces)} faces. Saved to {self.faces_folder}")
        return all_faces

    def get_face_embedding(self, face_path):
        try:
            if self.embedding_model is None:
                logger.error("Embedding model is not loaded. Cannot generate embedding.")
                return None

            representation = DeepFace.represent(
                img_path=face_path,
                model_name=self.model_name,
                enforce_detection=False,
                align=True,
                detector_backend="skip",
            )
            if representation and len(representation) > 0:
                embedding = representation[0]["embedding"]
                embedding = np.array(embedding, dtype=np.float32)
                embedding /= np.linalg.norm(embedding)
                return face_path, embedding
            else:
                logger.warning(f"No embedding found for face: {face_path}")
                return None
        except Exception as e:
            logger.warning(f"Failed to get embedding for {face_path}: {str(e)}")
            return None

    def cluster_faces(self, eps=0.5, min_samples=3):
        logger.info("Clustering faces...")
        face_paths = [
            os.path.join(self.faces_folder, f) for f in os.listdir(self.faces_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        logger.info(f"Found {len(face_paths)} extracted face images for clustering.")

        batch_size = 32
        embeddings, paths = [], []

        def process_batch(batch_paths):
            batch_results = []
            for path in batch_paths:
                result = self.get_face_embedding(path)
                if result:
                    batch_results.append(result)
            return batch_results

        cpu_count = os.cpu_count() if os.cpu_count() is not None else 4
        with ThreadPoolExecutor(max_workers=min(4, cpu_count)) as executor:
            batches = [face_paths[i:i + batch_size] for i in range(0, len(face_paths), batch_size)]
            futures = [executor.submit(process_batch, batch) for batch in batches]

            for future in tqdm(as_completed(futures), total=len(batches), desc="Generating embeddings"):
                batch_results = future.result()
                for path, embedding in batch_results:
                    paths.append(path)
                    embeddings.append(embedding)

        if not embeddings:
            logger.error("No embeddings generated. Clustering cannot proceed.")
            return

        self.embeddings = np.vstack(embeddings)
        self.face_paths = paths
        logger.info(f"Generated {self.embeddings.shape[0]} embeddings.")

        similarity_matrix = np.dot(self.embeddings, self.embeddings.T)
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        distance_matrix = 1 - similarity_matrix

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit(distance_matrix)
        self.labels = db.labels_
        logger.info(f"Found {len(set(self.labels)) - (1 if -1 in self.labels else 0)} clusters.")

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

        self.organize_clusters()
        self.save_current_state()

    def organize_clusters(self):
        logger.info("Organizing clusters...")
        for f in os.listdir(self.clusters_folder):
            shutil.rmtree(os.path.join(self.clusters_folder, f), ignore_errors=True)

        clusters = {}
        for path, label in zip(self.face_paths, self.labels):
            clusters.setdefault(label, []).append(path)

        for label, face_paths in clusters.items():
            cluster_name = f"person_{label}" if label != -1 else "noise"
            cluster_path = os.path.join(self.clusters_folder, cluster_name)
            Path(cluster_path).mkdir(parents=True, exist_ok=True)

            for i, face_path in enumerate(face_paths):
                shutil.copy(face_path, os.path.join(cluster_path, f"face_{i}_{os.path.basename(face_path)}"))

                if face_path in self.face_to_original:
                    original_info = self.face_to_original[face_path]
                    original_img = cv2.imread(original_info['original_path'])
                    if original_img is not None:
                        x, y, w, h = original_info['bbox']
                        annotated_img = original_img.copy()
                        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        original_filename = f"original_{i}_{Path(original_info['original_path']).name}"
                        cv2.imwrite(os.path.join(cluster_path, original_filename), annotated_img)
                    else:
                        logger.warning(f"Could not read original image: {original_info['original_path']}")
                else:
                    logger.warning(f"No original info found for face: {face_path}")
        logger.info("Clusters organized.")

    def create_representative_folder(self):
        logger.info("Creating representative folder...")
        
        for f in os.listdir(self.representative_folder):
            file_path = os.path.join(self.representative_folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for cluster_name in os.listdir(self.clusters_folder):
            if cluster_name.startswith("person_") and cluster_name != "person_0":
                cluster_path = os.path.join(self.clusters_folder, cluster_name)
                face_images = [f for f in os.listdir(cluster_path) if f.startswith("face_") and f.lower().endswith((".jpg", ".jpeg", ".png"))]
                if face_images:
                    first_face = face_images[0]
                    source_path = os.path.join(cluster_path, first_face)
                    rep_filename = f"{cluster_name}_rep.jpg"
                    rep_path = os.path.join(self.representative_folder, rep_filename)
                    shutil.copy(source_path, rep_path)
                    self.representative_to_cluster[rep_path] = cluster_path
                    logger.info(f"Copied representative image {rep_filename} for {cluster_name}")
                else:
                    logger.warning(f"No face images found in cluster {cluster_name}")

        logger.info(f"Representative folder created with {len(self.representative_to_cluster)} images.")

    def traverse_and_match(self, query_img_path, threshold=0.8):
        logger.info("Traversing representative folder to match query image using ArcFace...")

        if not os.path.exists(query_img_path):
            logger.error(f"Query image not found at {query_img_path}")
            return []

        try:
            representation = DeepFace.represent(
                img_path=query_img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )
            if not representation or len(representation) == 0:
                logger.warning(f"No face detected in query image: {query_img_path}")
                return []
            query_embedding = representation[0]["embedding"]
            query_embedding = np.array(query_embedding, dtype=np.float32)
            query_embedding /= np.linalg.norm(query_embedding)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query image: {str(e)}")
            return []

        matching_clusters = []
        for rep_path in self.representative_to_cluster:
            rep_embedding_result = self.get_face_embedding(rep_path)
            if rep_embedding_result is None:
                continue
            _, rep_embedding = rep_embedding_result

            similarity = np.dot(query_embedding, rep_embedding)
            if similarity > threshold:
                cluster_path = self.representative_to_cluster[rep_path]
                cluster_name = os.path.basename(cluster_path)
                matching_clusters.append({
                    'cluster': cluster_name,
                    'representative_image': rep_path,
                    'similarity': float(similarity)
                })
                logger.info(f"Match found: {cluster_name} with similarity {similarity:.4f}")

        return sorted(matching_clusters, key=lambda x: x['similarity'], reverse=True)

    def find_similar_faces(self, query_img_path, threshold=0.8, top_k=5):
        try:
            if self.embedding_model is None or self.index is None or self.embeddings is None or not self.face_paths or not self.face_to_original:
                logger.error("Necessary data or model not loaded for similarity search.")
                return []

            representation = DeepFace.represent(
                img_path=query_img_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True,
            )
            if not representation or len(representation) == 0:
                logger.warning(f"No face detected or represented in the query image: {query_img_path}")
                return []

            query_embedding = representation[0]["embedding"]
            query_embedding = np.array(query_embedding, dtype=np.float32)
            query_embedding /= np.linalg.norm(query_embedding)

            distances, indices = self.index.search(query_embedding.reshape(1, -1), k=min(top_k, self.index.ntotal))

            matches = []
            for idx, distance in zip(indices[0], distances[0]):
                if distance > threshold:
                    face_path = self.face_paths[idx]
                    if face_path in self.face_to_original:
                        original_info = self.face_to_original[face_path]
                        matches.append({
                            'cropped_face': face_path,
                            'original_image': original_info['original_path'],
                            'bbox': original_info['bbox'],
                            'distance': float(distance)
                        })
                    else:
                        logger.warning(f"No original info found for matched face: {face_path}")
            return sorted(matches, key=lambda x: x['distance'], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Error in find_similar_faces: {str(e)}")
            return []

if __name__ == "__main__":
    INPUT_FOLDER = r"C:\Users\ADMIN\Documents\Imgcf"
    OUTPUT_FOLDER = r"C:\Users\ADMIN\Documents\face_cluster"
    QUERY_IMAGE = r"C:\Users\ADMIN\Documents\reference_image.jpg"

    clusterer = FaceClusterer(INPUT_FOLDER, OUTPUT_FOLDER)
    if clusterer.embedding_model is not None:
        # Step 1: Extract and cluster faces
        clusterer.extract_all_faces()
        clusterer.cluster_faces(eps=0.5, min_samples=3)

        # Step 2: Create representative folder (excluding person_0)
        clusterer.create_representative_folder()

        # Step 3: Traverse representative folder and match query image
        if os.path.exists(QUERY_IMAGE):
            matches = clusterer.traverse_and_match(QUERY_IMAGE, threshold=0.8)
            if matches:
                print("\nMatching Clusters Found:")
                for i, match in enumerate(matches, 1):
                    print(f"{i}. {match['cluster']}: Similarity {match['similarity']:.4f}")
                    print(f"   Representative Image: {match['representative_image']}\n")
            else:
                print("\nNo matching clusters found in representative folder.")
        else:
            print(f"\nERROR: Query image not found at {QUERY_IMAGE}")

        # Step 4: (Optional) Perform detailed similarity search
        if clusterer.index is not None and clusterer.embeddings is not None and clusterer.face_paths and clusterer.face_to_original and os.path.exists(QUERY_IMAGE):
            results = clusterer.find_similar_faces(QUERY_IMAGE)
            if results:
                print("\nDetailed Matching Faces (from FAISS index):")
                for r in results:
                    print(f"Original: {r['original_image']}")
                    print(f"Face Location: {r['bbox']}")
                    print(f"Similarity Score: {r['distance']:.4f}")
                    print(f"Cropped Face: {r['cropped_face']}\n")
            else:
                print("\nNo detailed matches found using FAISS index.")
        else:
            print("\nWarning: Detailed similarity search cannot be performed as clustering data is incomplete.")

    print("\nDone")