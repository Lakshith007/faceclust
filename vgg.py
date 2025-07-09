!pip install faiss-cpu -q


import time
import psutil
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import faiss

from google.colab import drive
import os
import shutil

mountpoint = '/content/drive'
if os.path.exists(mountpoint) and os.path.isdir(mountpoint):
    # Check if the mountpoint is not empty and remove its content
    if os.listdir(mountpoint):
        print(f"Mountpoint {mountpoint} is not empty. Removing content...")
        try:
            for item in os.listdir(mountpoint):
                item_path = os.path.join(mountpoint, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            print(f"Content of {mountpoint} removed.")
        except Exception as e:
            print(f"Error removing content from {mountpoint}: {e}")

drive.mount(mountpoint)

gallery_path = "/content/drive/MyDrive/reception"  # CHANGE THIS to your gallery directory path
reference_image_path1 = "/content/drive/MyDrive/rref.jpg"  # CHANGE THIS to your reference image 1 path
reference_image_path2 = "/content/drive/MyDrive/ref2.jpg"  # CHANGE THIS to your reference image 2 path
output_folder = "/content/drive/MyDrive/matching_faces"
similarity_threshold = 0.6
  # Set your matching threshold (typical: 0.6-0.8)
model_name = "VGG16 + GlobalAvgPool (Face Embeddings)"

os.makedirs(output_folder, exist_ok=True)


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {[device.name for device in physical_devices]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("TensorFlow will use the GPU.")
else:
    print("No GPU found. Running on CPU.")


print("=== Starting Face Matching Process ===")
print(f"Gallery path: {gallery_path}")
print(f"Reference image 1: {reference_image_path1}")
print(f"Reference image 2: {reference_image_path2}")
print(f"Output folder: {output_folder}")

if not os.path.exists(output_folder):
    print(f"Creating output folder at {output_folder}")
    os.makedirs(output_folder)
else:
    print(f"Output folder already exists at {output_folder}")

# --- Model Loading and Inference ---
def load_model():
    """Loads a VGG16 model adapted for face recognition."""
    print(f"\nLoading {model_name}...")
    try:
        # Load VGG16 without top layers, add global average pooling
        print("Initializing VGG16 base model...")
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        print("Adding GlobalAveragePooling2D layer...")
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        model = Model(inputs=base_model.input, outputs=x)
        print("Model loaded successfully!")
        return model, preprocess_input
    except Exception as e:
        print(f"\nERROR loading model: {str(e)}")
        return None, None

def preprocess_image(image_path, preprocess_function):
    """Loads and preprocesses an image using TensorFlow ops for GPU acceleration."""
    try:
        print(f"\nProcessing image: {os.path.basename(image_path)}")
        # Read image as a tensor (decode_jpeg/png is fast and runs on GPU if available)
        image_bytes = tf.io.read_file(image_path)
        if image_path.lower().endswith(".png"):
            img = tf.image.decode_png(image_bytes, channels=3)
        else:
            img = tf.image.decode_jpeg(image_bytes, channels=3)
        # Use tf.image.resize (GPU-accelerated) and cast to float32
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32)
        # Add batch dimension
        img = tf.expand_dims(img, axis=0)
        # Use Keras preprocess_input as usual (CPU or GPU; it's just math ops)
        img = preprocess_function(img)
        print("Image preprocessed successfully (on GPU if available)")
        return img.numpy()  # Convert back to numpy array for compatibility
    except Exception as e:
        print(f"\nERROR processing image {image_path}: {str(e)}")
        return None

def get_face_embedding(model, preprocessed_image):
    """Extracts the face embedding from the preprocessed image."""
    if model is None or preprocessed_image is None:
        print("Cannot get embedding - model or image is None")
        return None
    try:
        print("Extracting face embedding...")
        embedding = model.predict(preprocessed_image, verbose=0)
        print("Embedding extracted successfully")
        return embedding
    except Exception as e:
        print(f"\nERROR getting embedding: {str(e)}")
        return None

def compare_faces(reference_embedding, gallery_embedding):
    """Compares two face embeddings using FAISS for cosine similarity."""
    if reference_embedding is None or gallery_embedding is None:
        print("Cannot compare - one or both embeddings are None")
        return 0.0
    try:
        # Normalize embeddings for cosine similarity
        ref = reference_embedding.astype('float32')
        gal = gallery_embedding.astype('float32')
        faiss.normalize_L2(ref)
        faiss.normalize_L2(gal)
        # Compute similarity (dot product after normalization = cosine similarity)
        sim = float((ref @ gal.T)[0][0])
        print(f"Similarity score (FAISS): {sim:.4f}")
        return sim
    except Exception as e:
        print(f"\nERROR comparing faces with FAISS: {str(e)}")
        return 0.0

def plot_embeddings(reference_embedding, gallery_embeddings, matching_files, filenames):
    """Visualizes embeddings in 2D space using PCA"""
    print("\nGenerating embeddings visualization...")
    try:
        # Combine all embeddings
        all_embeddings = [reference_embedding]
        all_embeddings.extend(gallery_embeddings)

        # Convert to numpy array
        embeddings_array = np.vstack(all_embeddings)

        # Reduce dimensionality to 2D using PCA
        print("Running PCA dimensionality reduction...")
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings_array)

        # Separate reference and gallery points
        ref_point = reduced_embeddings[0]
        gallery_points = reduced_embeddings[1:]

        # Create figure
        plt.figure(figsize=(12, 10))

        # Plot gallery points - color based on whether they matched
        match_indices = [i for i, (name, sim) in enumerate(matching_files)]
        non_match_indices = [i for i in range(len(gallery_points)) if i not in match_indices]

        plt.scatter(gallery_points[non_match_indices, 0],
                    gallery_points[non_match_indices, 1],
                    c='gray', alpha=0.5, label='Non-matches')
        plt.scatter(gallery_points[match_indices, 0],
                    gallery_points[match_indices, 1],
                    c='green', alpha=0.7, label='Matches')
        plt.scatter([ref_point[0]], [ref_point[1]],
                    c='red', marker='*', s=300, label='Reference')

        for i, (x, y) in enumerate(gallery_points):
            if i in match_indices:  # Only label matches to reduce clutter
                plt.annotate(f"{filenames[i][:10]}... (sim: {matching_files[match_indices.index(i)][1]:.2f})",
                            (x, y), fontsize=8, alpha=0.8)

        plt.title('Face Embeddings Visualization (PCA-reduced)', pad=20)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        plot_path = os.path.join(output_folder, 'embeddings_plot.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Embeddings plot saved to: {plot_path}")
        return True
    except Exception as e:
        print(f"\nERROR generating embeddings plot: {str(e)}")
        return False

def benchmark_and_match_vggface(gallery_path, reference_image_path1, reference_image_path2, output_folder, batch_size=32):
    """Benchmarks the adapted VGG16 model and saves matching faces to output folder with batch processing."""
    print("\n=== Starting Benchmark and Matching (Batch Enabled) ===")

    # Load model
    model, preprocess_function = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return False

    # Load and preprocess reference images
    reference_image1 = preprocess_image(reference_image_path1, preprocess_function)
    reference_image2 = preprocess_image(reference_image_path2, preprocess_function)
    if reference_image1 is None or reference_image2 is None:
        print("Failed to preprocess reference images. Exiting.")
        return False

    # Extract embeddings for reference images
    start_time = time.time()
    reference_embedding1 = get_face_embedding(model, reference_image1)
    reference_embedding2 = get_face_embedding(model, reference_image2)
    reference_embedding_time = time.time() - start_time
    if reference_embedding1 is None or reference_embedding2 is None:
        print("Failed to get one or both reference embeddings. Exiting.")
        return False

    print(f"\nProcessing gallery images from: {gallery_path}")
    if not os.path.exists(gallery_path):
        print(f"Gallery path does not exist: {gallery_path}")
        return False

    image_files = [f for f in os.listdir(gallery_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("No image files found in gallery.")
        return False

    print(f"Found {len(image_files)} images to process")

    # Normalize reference embeddings once
    ref1 = reference_embedding1.astype('float32')
    ref2 = reference_embedding2.astype('float32')
    faiss.normalize_L2(ref1)
    faiss.normalize_L2(ref2)

    processing_times = []
    similarities = []
    gallery_embeddings = []
    matching_files = []
    filenames = []
    total_embedding_time = reference_embedding_time
    processed_count = 0

    batch_images = []
    batch_filenames = []

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(gallery_path, filename)
        gallery_image = preprocess_image(image_path, preprocess_function)
        if gallery_image is None:
            continue

        batch_images.append(gallery_image[0])  # remove batch dim for stacking
        batch_filenames.append(filename)

        # If batch is full or last image, process it
        if len(batch_images) == batch_size or idx == len(image_files) - 1:
            batch_tensor = np.stack(batch_images, axis=0)  # shape: (B, 224, 224, 3)

            start = time.time()
            batch_embeddings = model.predict(batch_tensor, verbose=0)
            end = time.time()
            batch_time = end - start
            processing_times.append(batch_time / len(batch_embeddings))
            total_embedding_time += batch_time

            batch_embeddings = batch_embeddings.astype('float32')
            faiss.normalize_L2(batch_embeddings)

            sim_matrix1 = np.dot(batch_embeddings, ref1.T)
            sim_matrix2 = np.dot(batch_embeddings, ref2.T)

            for i in range(len(batch_filenames)):
                sim1 = float(sim_matrix1[i][0])
                sim2 = float(sim_matrix2[i][0])
                sim = max(sim1, sim2)
                similarities.append(sim)

                embedding = batch_embeddings[i].reshape(1, -1)
                gallery_embeddings.append(embedding)
                filenames.append(batch_filenames[i])

                if sim >= similarity_threshold:
                    matching_files.append((batch_filenames[i], sim))
                    try:
                        shutil.copy(os.path.join(gallery_path, batch_filenames[i]),
                                    os.path.join(output_folder, batch_filenames[i]))
                        print(f"Copied match: {batch_filenames[i]} (sim={sim:.4f})")
                    except Exception as e:
                        print(f"Error copying {batch_filenames[i]}: {str(e)}")

            batch_images = []
            batch_filenames = []
            processed_count += len(batch_embeddings)

    print("\n=== Gallery Processing Complete ===")
    print(f"Processed {processed_count} images")
    if not similarities:
        print("No valid similarity scores calculated.")
        return False

    # Benchmark Summary
    avg_speed = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # in MB

    print(f"\n--- Benchmark Results for {model_name} ---")
    print(f"Reference images: {os.path.basename(reference_image_path1)}, {os.path.basename(reference_image_path2)}")
    print(f"Embedding Time (References): {reference_embedding_time:.4f} sec")
    print(f"Total Embedding Time (All): {total_embedding_time:.4f} sec")
    print(f"Avg Processing Speed (Gallery): {avg_speed:.4f} sec/image")
    print(f"Avg Similarity (Gallery): {avg_similarity:.4f}")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    print(f"\nMatched Images (Threshold={similarity_threshold}):")
    if matching_files:
        for fname, sim in matching_files:
            print(f"  {fname}: Similarity = {sim:.4f}")
    else:
        print("  No matching images found.")
    print(f"Total Matches: {len(matching_files)}")
    print(f"Results saved to: {output_folder}")

    if gallery_embeddings and matching_files:
        success_plot = plot_embeddings(reference_embedding1, gallery_embeddings, matching_files, filenames)
        if not success_plot:
            print("Embedding visualization failed.")

    return True


if __name__ == "__main__":
    print("\n=== Face Matching System ===")
    success = benchmark_and_match_vggface(gallery_path, reference_image_path1, reference_image_path2, output_folder)
    if success:
        print("\nProcess completed successfully!")
    else:
        print("\nProcess completed with errors")
    print("\n=== End of Program ===")
