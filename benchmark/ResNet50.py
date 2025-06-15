import os
import time
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import psutil
from PIL import Image
import shutil

# Set paths
reference_image_path = "D:\SRI INDIRA\ref.jpg" # Replace with your reference image path
gallery_folder = "D:\SRI INDIRA\Imgcf"  # Replace with your gallery folder path
output_folder = "D:\SRI INDIRA\output"  # Folder to save matching images
similarity_threshold = 0.7  # Similarity threshold for matching

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def get_image_embedding(img_path):
    """Generate embedding for an image using ResNet50."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    embedding = model.predict(img_array, verbose=0)
    return embedding.flatten()

def get_ram_usage():
    """Return current RAM usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

# Benchmarking
start_time = time.time()

# Get reference image embedding
initial_ram = get_ram_usage()
ref_start_time = time.time()
reference_embedding = get_image_embedding(reference_image_path)
ref_embedding_time = time.time() - ref_start_time
ref_ram = get_ram_usage() - initial_ram

# Process gallery images
gallery_embeddings = []
gallery_filenames = []
processing_times = []
similarities = []
matching_images = []

initial_ram = get_ram_usage()
for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(gallery_folder, filename)
        start_time = time.time()
        embedding = get_image_embedding(img_path)
        processing_time = time.time() - start_time
        similarity = cosine_similarity([reference_embedding], [embedding])[0][0]
        
        gallery_embeddings.append(embedding)
        gallery_filenames.append(filename)
        processing_times.append(processing_time)
        similarities.append(similarity)
        
        if similarity >= similarity_threshold:
            matching_images.append((filename, similarity))
            # Copy matching image to output folder
            shutil.copy(img_path, os.path.join(output_folder, filename))

total_time = time.time() - start_time
total_ram = get_ram_usage() - initial_ram

# Calculate metrics
avg_processing_speed = np.mean(processing_times) if processing_times else 0
avg_similarity = np.mean(similarities) if similarities else 0
total_matching_images = len(matching_images)

# Print results
print(f"Reference Embedding Time: {ref_embedding_time:.4f} seconds")
print(f"Average Processing Speed (Gallery): {avg_processing_speed:.4f} seconds/image")
print(f"Average Similarity (Gallery vs. Reference): {avg_similarity:.4f}")
print(f"RAM Usage: {total_ram:.2f} MB")
print("\nMatching Images:")
for filename, sim in matching_images:
    print(f"{filename}: Similarity = {sim:.4f}")
print(f"\nTotal Matching Images: {total_matching_images}")
print(f"Matching images saved to: {output_folder}")

# Save benchmark results to a file
with open(os.path.join(output_folder, "benchmark_results_resnet50.txt"), "w") as f:
    f.write(f"Reference Embedding Time: {ref_embedding_time:.4f} seconds\n")
    f.write(f"Average Processing Speed (Gallery): {avg_processing_speed:.4f} seconds/image\n")
    f.write(f"Average Similarity (Gallery vs. Reference): {avg_similarity:.4f}\n")
    f.write(f"RAM Usage: {total_ram:.2f} MB\n")
    f.write("\nMatching Images:\n")
    for filename, sim in matching_images:
        f.write(f"{filename}: Similarity = {sim:.4f}\n")
    f.write(f"\nTotal Matching Images: {total_matching_images}\n")
    f.write(f"Matching images saved to: {output_folder}\n")