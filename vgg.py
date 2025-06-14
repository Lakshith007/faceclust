import time
import psutil
import os
import shutil
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input  # Example backbone preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import faiss

# Configure TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {[device.name for device in physical_devices]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("TensorFlow will use the GPU.")
else:
    print("No GPU found. Running on CPU.")

# --- Utility Functions ---
def normalize_embedding(embedding):
    """Normalize embeddings to unit length for cosine similarity."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm

def plot_embeddings(reference_embeddings, gallery_embeddings, matching_results, gallery_filenames):
    """Visualize embeddings using PCA."""
    try:
        print("\nPreparing to plot embeddings...")
        
        # Combine reference and gallery embeddings
        all_embeddings = []
        for ref_emb in reference_embeddings:
            all_embeddings.append(ref_emb.flatten())
        for gallery_emb in gallery_embeddings:
            all_embeddings.append(gallery_emb.flatten())
        
        # Convert to numpy array
        all_embeddings = np.array(all_embeddings)
        
        # Apply PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(all_embeddings)
        
        # Split back into reference and gallery points
        n_ref = len(reference_embeddings)
        ref_points = reduced_embeddings[:n_ref]
        gallery_points = reduced_embeddings[n_ref:]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot gallery points (gray)
        plt.scatter(gallery_points[:, 0], gallery_points[:, 1], 
                   c='gray', alpha=0.5, label='Gallery Images')
        
        # Plot reference points (red)
        for i, point in enumerate(ref_points):
            plt.scatter(point[0], point[1], c='red', marker='*', s=200, 
                       label=f'Reference {i+1}' if i == 0 else None)
            
            # Highlight matches for this reference
            matches = matching_results[i]
            if matches:
                match_indices = [idx for idx, _ in matches]
                match_points = gallery_points[match_indices]
                plt.scatter(match_points[:, 0], match_points[:, 1], 
                           c='blue', alpha=0.7, 
                           label=f'Matches for Ref {i+1}' if i == 0 else None)
        
        plt.title('Face Embeddings Visualization (PCA)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_path = os.path.join(output_folder, 'embeddings_plot.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Embeddings plot saved to {plot_path}")
        return True
    
    except Exception as e:
        print(f"Error generating embeddings plot: {str(e)}")
        return False

# --- Configuration ---
# Adjust these paths for your local file system
gallery_path = r"D:\anishshreenidhi-photo-download-1of1\reception"  # Folder containing images to search through
reference_image_path1 = r"C:\Users\LAKSHITH.S\Pictures\ref2.jpg"  # First reference image
reference_image_path2 = r"C:\Users\LAKSHITH.S\Pictures\rref.jpg"  # Second reference image
output_folder = r"D:\face"  # Where to save matching images
model_name = "ArcFace"
similarity_threshold = 0.6  # Adjust based on experimentation with ArcFace
distance_threshold = 1 - similarity_threshold  # FAISS uses distance, not similarity

# --- Debugging Setup ---
print("=== Starting Face Matching Process ===")
print(f"Gallery path: {gallery_path}")
print(f"Reference images: {reference_image_path1}, {reference_image_path2}")
print(f"Output folder: {output_folder}")
print(f"Similarity Threshold: {similarity_threshold}")
print(f"Corresponding FAISS Distance Threshold: {distance_threshold}")

# --- Create Output Folder ---
if not os.path.exists(output_folder):
    print(f"Creating output folder at {output_folder}")
    os.makedirs(output_folder)
else:
    print(f"Output folder already exists at {output_folder}")

# --- Model Loading and Inference Functions ---
def preprocess_image(image_path, preprocess_function):
    """Preprocess image for ArcFace (typically 112x112)."""
    try:
        img = Image.open(image_path).convert('RGB').resize((112, 112))  # ArcFace standard size
        img_array = np.array(img, dtype='float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_function(img_array)
        return img_array
    except Exception as e:
        print(f"\nERROR processing image {image_path}: {str(e)}")
        return None

def load_model():
    """Loads a pre-trained ArcFace model."""
    print(f"\nLoading ArcFace model...")
    try:
        # Placeholder: Replace with actual ArcFace model loading
        # Example using insightface (adjust based on your implementation)
        from insightface.app import FaceAnalysis  # Requires insightface library
        app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use GPU if available
        app.prepare(ctx_id=0, det_size=(112, 112))
        
        # Wrap the model for embedding extraction
        class ArcFaceWrapper(tf.keras.Model):
            def __init__(self, insightface_app):
                super(ArcFaceWrapper, self).__init__()
                self.app = insightface_app
            
            def call(self, inputs):
                # Extract embedding using insightface
                img = inputs[0].numpy().astype(np.uint8)
                faces = self.app.get(img)
                if faces:
                    return tf.convert_to_tensor(faces[0].embedding, dtype=tf.float32)
                return None

        model = ArcFaceWrapper(app)
        preprocess_function = preprocess_input  # Adjust if ArcFace requires different preprocessing
        return model, preprocess_function
    except Exception as e:
        print(f"\nERROR loading model: {str(e)}")
        return None, None

def get_face_embedding(model, preprocessed_image):
    """Extract embedding using the ArcFace model."""
    if model is None or preprocessed_image is None:
        print("Cannot get embedding - model or image is None")
        return None
    try:
        embedding = model(preprocessed_image)
        if embedding is not None:
            return embedding.numpy() if tf.is_tensor(embedding) else embedding
        return None
    except Exception as e:
        print(f"\nERROR getting embedding: {str(e)}")
        return None

# --- Main Execution Function with FAISS ---
def benchmark_and_match_arcface_faiss(gallery_path, reference_image_paths, output_folder, similarity_threshold):
    """
    Benchmarks the ArcFace model and finds matching faces in a gallery
    using FAISS for accelerated search.
    """
    print("\n=== Starting Benchmark and Matching with FAISS ===")

    model, preprocess_function = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return False, [], []  # Return success status, matched files list, gallery filenames

    # --- Process Reference Images ---
    reference_embeddings = []
    print("\nProcessing reference images...")
    start_time_ref_processing = time.time()
    for ref_path in reference_image_paths:
        print(f"  Processing {os.path.basename(ref_path)}")
        ref_image = preprocess_image(ref_path, preprocess_function)
        if ref_image is None:
            print(f"  Failed to load reference image: {ref_path}. Skipping.")
            continue  # Skip this reference image

        ref_embedding = get_face_embedding(model, ref_image)
        if ref_embedding is not None:
            # Normalize the embedding for FAISS Inner Product index (cosine similarity)
            normalized_ref_embedding = normalize_embedding(ref_embedding)
            if normalized_ref_embedding is not None:
                reference_embeddings.append(normalized_ref_embedding)
            else:
                print(f"  Normalization failed for {ref_path}. Skipping.")
        else:
            print(f"  Failed to get embedding for reference image: {ref_path}. Skipping.")

    reference_embedding_time = time.time() - start_time_ref_processing
    if not reference_embeddings:
        print("No valid reference embeddings were generated. Exiting.")
        return False, [], []

    print(f"Processed {len(reference_embeddings)} reference images.")
    print(f"Embedding Time (References): {reference_embedding_time:.4f} seconds")

    # --- Process Gallery Images and Build FAISS Index ---
    print(f"\nProcessing gallery images from: {gallery_path}")
    gallery_embeddings = []
    gallery_filenames = []
    processing_times = []

    if not os.path.exists(gallery_path):
        print(f"Gallery path does not exist: {gallery_path}")
        return False, [], []

    image_files = [f for f in os.listdir(gallery_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("No image files found in gallery directory")
        return False, [], []

    print(f"Found {len(image_files)} images to process")

    # Collect embeddings and filenames
    print("Extracting embeddings for gallery images...")
    total_images = len(image_files)
    for idx, filename in enumerate(image_files, 1):
        print(f"Processing image {idx} of {total_images}: {filename}")
        image_path = os.path.join(gallery_path, filename)
        gallery_image = preprocess_image(image_path, preprocess_function)
        if gallery_image is None:
            continue  # Skip problematic images

        start = time.time()
        embedding = get_face_embedding(model, gallery_image)
        end = time.time()
        embedding_time = end - start
        processing_times.append(embedding_time)

        if embedding is not None:
            # Normalize gallery embedding for FAISS Inner Product index
            normalized_embedding = normalize_embedding(embedding)
            if normalized_embedding is not None:  # Check if normalization was successful
                gallery_embeddings.append(normalized_embedding)
                gallery_filenames.append(filename)  # Keep track of original filenames
            else:
                print(f"Warning: Could not normalize embedding for {filename}. Skipping.")
        else:
            print(f"Warning: No embedding extracted for {filename}. Skipping.")

    if not gallery_embeddings:
        print("No valid gallery embeddings were generated. Exiting.")
        return False, [], []

    gallery_embeddings_array = np.vstack(gallery_embeddings).astype('float32')  # FAISS requires float32
    embedding_dimension = gallery_embeddings_array.shape[1]

    print(f"\nExtracted embeddings for {len(gallery_embeddings)} gallery images.")
    print(f"Embedding dimension: {embedding_dimension}")
    average_embedding_speed = sum(processing_times) / len(processing_times) if processing_times else 0
    print(f"Average Embedding Speed (Gallery): {average_embedding_speed:.4f} seconds per image")

    # --- Build FAISS Index ---
    print("\nBuilding FAISS index...")
    start_time_faiss_build = time.time()

    # Use IndexFlatIP for Inner Product (equivalent to cosine similarity for L2 normalized vectors)
    index = faiss.IndexFlatIP(embedding_dimension)

    # Attempt to use GPU if available
    try:
        res = faiss.StandardGpuResources()  # Use standard GPU resources
        # Transfer index to GPU
        index = faiss.index_cpu_to_gpu(res, 0, index)  # 0 is the GPU device ID
        print("FAISS index transferred to GPU.")
    except Exception as e:
        print(f"Could not transfer FAISS index to GPU: {e}. Using CPU.")
        index = faiss.IndexFlatIP(embedding_dimension)
        print("Using FAISS index on CPU.")

    index.add(gallery_embeddings_array)  # Add gallery embeddings to the index

    faiss_build_time = time.time() - start_time_faiss_build
    print(f"FAISS index built in {faiss_build_time:.4f} seconds.")
    print(f"Number of vectors in FAISS index: {index.ntotal}")

    # --- Search FAISS Index with Reference Embeddings ---
    print("\nSearching FAISS index with reference embeddings...")
    start_time_faiss_search = time.time()

    K = index.ntotal  # Search against all gallery images for each reference
    reference_embeddings_array = np.vstack(reference_embeddings).astype('float32')
    all_distances, all_indices = index.search(reference_embeddings_array, K)

    faiss_search_time = time.time() - start_time_faiss_search
    print(f"FAISS search completed in {faiss_search_time:.4f} seconds.")

    # --- Analyze Search Results and Identify Matches ---
    print("\nAnalyzing search results...")
    matching_files = []  # Store (filename, max_similarity) for matched files
    matching_results_for_plotting = []  # Store list of (gallery_index, similarity) per reference

    # Process search results for each reference embedding
    for i, ref_embedding in enumerate(reference_embeddings):
        print(f"  Analyzing results for Reference Image {i+1}:")
        distances = all_distances[i]  # These are cosine similarities
        indices = all_indices[i]

        ref_matches_for_plotting = []  # Matches found for this specific reference

        # Iterate through search results
        for j in range(K):
            gallery_index = indices[j]
            similarity = distances[j]  # Cosine similarity from FAISS IndexFlatIP

            # Check if the similarity meets the threshold
            if similarity >= similarity_threshold:
                filename = gallery_filenames[gallery_index]
                # Store max similarity found for this gallery file across all references
                found = False
                for k in range(len(matching_files)):
                    if matching_files[k][0] == filename:
                        # Update if current similarity is higher
                        if similarity > matching_files[k][1]:
                            matching_files[k] = (filename, similarity)
                        found = True
                        break
                if not found:
                    matching_files.append((filename, similarity))

                # Add to plotting results for this reference
                ref_matches_for_plotting.append((gallery_index, similarity))

        matching_results_for_plotting.append(ref_matches_for_plotting)

    print(f"\nFound {len(matching_files)} matching images (threshold: {similarity_threshold})")

    # --- Copy Matching Files to Output Folder ---
    if matching_files:
        print(f"\nCopying matching files to {output_folder}")
        copy_count = 0
        for filename, sim in matching_files:
            image_path = os.path.join(gallery_path, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                shutil.copy(image_path, output_path)
                copy_count += 1
            except Exception as e:
                print(f"  Error copying file {filename}: {str(e)}")
        print(f"Successfully copied {copy_count} files.")
    else:
        print("\nNo files to copy.")

    # --- Copy Unmatched Files to a Separate Folder ---
    unmatched_folder = os.path.join(output_folder, 'unmatched')
    if not os.path.exists(unmatched_folder):
        os.makedirs(unmatched_folder)
    matched_filenames = set([filename for filename, _ in matching_files])
    unmatched_files = [f for f in gallery_filenames if f not in matched_filenames]
    if unmatched_files:
        print(f"\nCopying unmatched files to {unmatched_folder}")
        unmatched_count = 0
        for filename in unmatched_files:
            image_path = os.path.join(gallery_path, filename)
            output_path = os.path.join(unmatched_folder, filename)
            try:
                shutil.copy(image_path, output_path)
                unmatched_count += 1
            except Exception as e:
                print(f"  Error copying unmatched file {filename}: {str(e)}")
        print(f"Successfully copied {unmatched_count} unmatched files.")
    else:
        print("\nNo unmatched files to copy.")

    # --- Benchmark Results ---
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB

    print(f"\n--- Benchmark Results for {model_name} with FAISS ---")
    print(f"Reference images: {[os.path.basename(p) for p in reference_image_paths]}")
    print(f"Embedding Time (References): {reference_embedding_time:.4f} seconds")
    print(f"Average Embedding Speed (Gallery): {average_embedding_speed:.4f} seconds per image")
    print(f"FAISS Index Build Time: {faiss_build_time:.4f} seconds")
    print(f"FAISS Search Time (Gallery Size {index.ntotal}): {faiss_search_time:.4f} seconds")
    print(f"Total RAM Usage: {ram_usage:.2f} MB")
    print(f"\nMatching Images (Threshold: {similarity_threshold}):")

    if matching_files:
        # Sort matches by similarity descending for better readability
        matching_files_sorted = sorted(matching_files, key=lambda item: item[1], reverse=True)
        for filename, sim in matching_files_sorted:
            print(f"  {filename}: Similarity = {sim:.4f}")
    else:
        print("  No matching images found")

    print(f"Total Matching Images: {len(matching_files)}")
    print(f"\nMatching images saved to: {output_folder}")

    # --- Plot Embeddings ---
    if gallery_embeddings and matching_files:
        plot_success = plot_embeddings(reference_embeddings, gallery_embeddings, 
                                     matching_results_for_plotting, gallery_filenames)
        if not plot_success:
            print("Failed to generate embeddings plot")

    return True, matching_files, gallery_filenames

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n=== Face Matching System ===")

    start_total_time = time.time()  # Record the start time of the entire process

    # Define the list of reference image paths
    reference_image_paths = [reference_image_path1, reference_image_path2]

    # Call the FAISS-based benchmark function
    success, matched_files_list, gallery_filenames = benchmark_and_match_arcface_faiss(
        gallery_path, reference_image_paths, output_folder, similarity_threshold
    )

    if success:
        print("\nFace matching process completed successfully!")
    else:
        print("\nFace matching process completed with errors or no results.")

    end_total_time = time.time()  # Record the end time of the entire process
    total_execution_time = end_total_time - start_total_time

    print(f"\nTotal Execution Time: {total_execution_time:.4f} seconds")
    print("\n=== End of Program ===")