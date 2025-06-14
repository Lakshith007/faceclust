import time
import psutil
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from insightface.app import FaceAnalysis

# --- Configuration ---
gallery_path = r"D:\anishshreenidhi-photo-download-1of1\test"
reference_image_path = r"C:\Users\LAKSHITH.S\Pictures\rref.jpg"
output_folder = r"D:\anishshreenidhi-photo-download-1of1\matching_faces"
model_name = "ArcFace (InsightFace)"
similarity_threshold = 0.4  # Adjusted for ArcFace (typical range: 0.3–0.5)
detector_model = "retinaface_r50_v1"  # Face detection model
recognition_model = "arcface_r100_v1"  # ArcFace recognition model

# --- Create Output Folder ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Model Loading and Inference ---
def load_model():
    """Loads the ArcFace model with face detection."""
    print(f"Loading {model_name} with {detector_model} detector...")
    try:
        # Initialize FaceAnalysis with detection and recognition
        app = FaceAnalysis(name=detector_model, providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app, None  # No separate preprocess function needed
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def preprocess_image(image_path, app):
    """Loads an image and detects/crops faces."""
    try:
        img = np.array(Image.open(image_path).convert('RGB'))
        faces = app.get(img)
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return None
        # Use the first detected face
        embedding = faces[0].normed_embedding  # Already normalized
        return embedding
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def get_face_embedding(app, image_path):
    """Extracts the face embedding from the image."""
    if app is None:
        return None
    return preprocess_image(image_path, app)

def compare_faces(reference_embedding, gallery_embedding):
    """Compares two face embeddings using cosine similarity."""
    if reference_embedding is None or gallery_embedding is None:
        return 0.0
    try:
        similarity = cosine_similarity([reference_embedding], [gallery_embedding])[0][0]
        return similarity
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return 0.0

# --- Benchmarking and Matching Function ---
def benchmark_and_match_arcface(gallery_path, reference_image_path, output_folder):
    """Benchmarks the ArcFace model and saves matching faces to output folder."""
    app, _ = load_model()
    if app is None:
        print("Failed to load model. Exiting.")
        return

    # Process the reference image
    start_time = time.time()
    reference_embedding = get_face_embedding(app, reference_image_path)
    reference_embedding_time = time.time() - start_time

    if reference_embedding is None:
        print("Failed to get reference embedding. Exiting.")
        return

    # Process gallery images, measure speed, and find matches
    processing_times = []
    similarities = []
    matching_files = []
    total_embedding_time = reference_embedding_time  # Initialize with reference embedding time

    for filename in os.listdir(gallery_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(gallery_path, filename)
            start = time.time()
            embedding = get_face_embedding(app, image_path)
            end = time.time()
            embedding_time = end - start
            processing_times.append(embedding_time)
            total_embedding_time += embedding_time  # Accumulate embedding time
            if embedding is not None:
                similarity = compare_faces(reference_embedding, embedding)
                similarities.append(similarity)
                if similarity >= similarity_threshold:
                    matching_files.append((filename, similarity))
                    # Copy matching image to output folder
                    shutil.copy(image_path, os.path.join(output_folder, filename))

    # Calculate average speed and similarity
    average_speed = sum(processing_times) / len(processing_times) if processing_times else 0
    average_similarity = sum(similarities) / len(similarities) if similarities else 0

    # Measure RAM usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB

    # Print results
    print(f"--- Benchmark Results for {model_name} ---")
    print(f"Embedding Time (Reference): {reference_embedding_time:.4f} seconds")
    print(f"Total Embedding Time (Reference + Gallery): {total_embedding_time:.4f} seconds")
    print(f"Average Processing Speed (Gallery): {average_speed:.4f} seconds per image")
    print(f"Average Similarity (Gallery vs. Reference): {average_similarity:.4f}")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    print(f"Matching Images (Threshold: {similarity_threshold}):")
    for filename, sim in matching_files:
        print(f"  {filename}: Similarity = {sim:.4f}")
    print(f"Total Matching Images: {len(matching_files)}")
    print(f"Matching images saved to: {output_folder}")

# --- Run Benchmark and Matching ---
benchmark_and_match_arcface(gallery_path, reference_image_path, output_folder)