import time
import psutil
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from deepface import DeepFace

# --- Configuration ---
gallery_path = r"D:\anishshreenidhi-photo-download-1of1\test"
reference_image_path = r"C:\Users\LAKSHITH.S\Pictures\rref.jpg"
output_folder = r"D:\anishshreenidhi-photo-download-1of1\matching_faces"
model_name = "RetinaFace + ArcFace"
similarity_threshold = 0.6  # Adjust based on experimentation

# --- Create Output Folder ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Image Processing and Embedding Functions ---
def preprocess_image(image_path):
    """Loads an image for DeepFace processing."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def get_face_embedding(image_array, detector_backend='retinaface', model_name='ArcFace'):
    """Extracts face embedding using RetinaFace for detection and ArcFace for embedding."""
    try:
        # Extract embedding with RetinaFace for detection and ArcFace for recognition
        embedding = DeepFace.represent(
            img_path=image_array,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=True
        )[0]["embedding"]
        return np.array(embedding).reshape(1, -1)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def compare_faces(reference_embedding, gallery_embedding):
    """Compares two face embeddings using cosine similarity."""
    if reference_embedding is None or gallery_embedding is None:
        return 0.0
    try:
        similarity = cosine_similarity(reference_embedding, gallery_embedding)[0][0]
        return similarity
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return 0.0

# --- Benchmarking and Matching Function ---
def benchmark_and_match_retinaface(gallery_path, reference_image_path, output_folder):
    """Benchmarks RetinaFace + ArcFace and saves matching faces to output folder."""
    print(f"Using {model_name} for face detection and recognition...")

    # Load and preprocess the reference image
    reference_image = preprocess_image(reference_image_path)
    if reference_image is None:
        print("Failed to load reference image. Exiting.")
        return

    start_time = time.time()
    reference_embedding = get_face_embedding(reference_image)
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
            gallery_image = preprocess_image(image_path)
            if gallery_image is not None:
                start = time.time()
                embedding = get_face_embedding(gallery_image)
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
benchmark_and_match_retinaface(gallery_path, reference_image_path, output_folder)