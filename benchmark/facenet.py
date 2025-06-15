import time
import psutil
import os
import shutil
from deepface import DeepFace

# --- Configuration ---
gallery_path = r"C:\Users\ADMIN\Desktop\23dx43\Imgcf"
reference_image_path = r"C:\Users\ADMIN\Desktop\23dx43\reference_face.jpeg"
output_folder = r"C:\Users\ADMIN\Desktop\23dx43\face_cluster"
similarity_threshold = 0.6  # Cosine similarity threshold (0.6 is a good starting point)

# --- Create Output Folder ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Extract Face Embeddings ---
def get_face_embedding(image_path):
    """Extracts face embedding from an image using DeepFace."""
    try:
        # Use DeepFace.represent to get embeddings (Facenet model)
        embedding = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=True,  # Ensure a face is detected
            detector_backend="opencv"  # Use opencv for face detection
        )
        return embedding[0]["embedding"]  # Return the embedding vector
    except Exception as e:
        print(f"Error getting embedding for {image_path}: {e}")
        return None

# --- Compare Two Face Embeddings ---
def compare_faces(reference_embedding, gallery_embedding):
    """Compares two face embeddings using cosine similarity."""
    if reference_embedding is None or gallery_embedding is None:
        return 0.0
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        ref_array = np.array(reference_embedding).reshape(1, -1)
        gal_array = np.array(gallery_embedding).reshape(1, -1)
        similarity = cosine_similarity(ref_array, gal_array)[0][0]
        return similarity
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return 0.0

# --- Benchmark and Match Faces ---
def benchmark_and_match_facenet(gallery_path, reference_image_path, output_folder):
    """Benchmarks FaceNet and saves matching faces to output folder."""
    # Load and extract embedding for the reference image
    start_time = time.time()
    reference_embedding = get_face_embedding(reference_image_path)
    embedding_time = time.time() - start_time

    if reference_embedding is None:
        print("Failed to get reference embedding. Exiting.")
        return

    # Process gallery images, measure speed, and find matches
    processing_times = []
    similarities = []
    matching_files = []

    for filename in os.listdir(gallery_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(gallery_path, filename)
            start = time.time()
            embedding = get_face_embedding(image_path)
            end = time.time()
            processing_times.append(end - start)
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
    print("\n--- FaceNet Benchmark Results ---")
    print(f"Reference Embedding Time: {embedding_time:.4f} seconds")
    print(f"Average Processing Speed (Gallery): {average_speed:.4f} seconds per image")
    print(f"Average Similarity (Gallery vs. Reference): {average_similarity:.4f}")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    print(f"\nMatching Images (Threshold: {similarity_threshold}):")
    for filename, sim in matching_files:
        print(f"  {filename}: Similarity = {sim:.4f}")
    print(f"\nTotal Matching Images: {len(matching_files)}")
    print(f"Matching images saved to: {output_folder}")

# --- Run Benchmark and Matching ---
if _name_ == "_main_":
    benchmark_and_match_facenet(gallery_path, reference_image_path, output_folder)