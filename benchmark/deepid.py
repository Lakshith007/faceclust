import time
import psutil
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import shutil
from deepface import DeepFace
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
gallery_path = r"D:\anishshreenidhi-photo-download-1of1\test"
reference_image_path = r"C:\Users\LAKSHITH.S\Pictures\rref.jpg"
output_folder = r"D:\anishshreenidhi-photo-download-1of1\matching_faces"  # Output folder for similar images
model_name = "DeepID"
distance_threshold = 0.4  # Cosine distance threshold (lower = more similar)

# --- Create Output Folder ---
os.makedirs(output_folder, exist_ok=True)

# --- Embedding and Comparison Functions ---
def get_face_embedding(image_path):
    """Extracts face embedding using DeepID."""
    try:
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="DeepID",
            detector_backend="opencv",
            enforce_detection=True,
            align=True
        )
        return np.array(embedding[0]['embedding'])
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def compare_faces(reference_embedding, gallery_embedding):
    """Compares two face embeddings using cosine similarity."""
    if reference_embedding is None or gallery_embedding is None:
        return 1.0  # High distance (not similar) if embeddings are invalid
    try:
        similarity = cosine_similarity([reference_embedding], [gallery_embedding])[0][0]
        distance = 1 - similarity  # Convert to cosine distance
        return distance
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return 1.0

# --- Benchmarking and Matching Function ---
def benchmark_and_match_deepid(gallery_path, reference_image_path, output_folder):
    """Benchmarks DeepID and saves matching faces to output folder."""
    print(f"Generating embedding for reference image with {model_name}...")
    start_time = time.time()
    reference_embedding = get_face_embedding(reference_image_path)
    reference_embedding_time = time.time() - start_time

    if reference_embedding is None:
        print("Failed to get reference embedding. Exiting.")
        return

    # Process gallery images, measure speed, and find matches
    processing_times = []
    distances = []
    matching_files = []
    total_embedding_time = reference_embedding_time  # Initialize with reference embedding time

    for filename in tqdm(os.listdir(gallery_path), desc="Processing images"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(gallery_path, filename)
            start = time.time()
            embedding = get_face_embedding(image_path)
            end = time.time()
            embedding_time = end - start
            processing_times.append(embedding_time)
            total_embedding_time += embedding_time  # Accumulate embedding time
            if embedding is not None:
                distance = compare_faces(reference_embedding, embedding)
                distances.append(distance)
                if distance <= distance_threshold:
                    matching_files.append((filename, distance))
                    shutil.copy(image_path, os.path.join(output_folder, filename))

    # Calculate average speed and distance
    average_speed = sum(processing_times) / len(processing_times) if processing_times else 0
    average_distance = sum(distances) / len(distances) if distances else 0

    # Measure RAM usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB

    # Print results
    print(f"\n--- Benchmark Results for {model_name} ---")
    print(f"Embedding Time (Reference): {reference_embedding_time:.4f} seconds")
    print(f"Total Embedding Time (Reference + Gallery): {total_embedding_time:.4f} seconds")
    print(f"Average Processing Speed (Gallery): {average_speed:.4f} seconds per image")
    print(f"Average Cosine Distance (Gallery vs. Reference): {average_distance:.4f}")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    print(f"Matching Images (Threshold: {distance_threshold}):")
    for filename, dist in matching_files:
        print(f"  {filename}: Cosine Distance = {dist:.4f}")
    print(f"Total Matching Images: {len(matching_files)}")
    print(f"Matching images saved to: {output_folder}")

# --- Run Benchmark and Matching ---
benchmark_and_match_deepid(gallery_path, reference_image_path, output_folder)