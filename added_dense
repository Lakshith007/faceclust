!pip install faiss-cpu

import os
import time
import shutil
import psutil
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from PIL import Image
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, Flatten, Activation, Convolution2D, MaxPooling2D,
                                     ZeroPadding2D, GlobalAveragePooling2D)

import faiss

from google.colab import drive

# === Mount Google Drive ===
drive.mount('/content/drive')

# === Path Configuration ===
gallery_path = "/content/drive/MyDrive/reception"
reference_image_path1 = "/content/drive/MyDrive/rref.jpg"
reference_image_path2 = "/content/drive/MyDrive/ref2.jpg"
output_folder = "/content/drive/MyDrive/matching_faces"
os.makedirs(output_folder, exist_ok=True)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPUs detected: {[device.name for device in physical_devices]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("TensorFlow will use the GPU.")
else:
    print("No GPU found. Running on CPU.")

# === Lightweight CNN model (optional) ===
NEW_INPUT_SIZE = (128, 128)
def preprocess_light(img_path):
    img = image.load_img(img_path, target_size=NEW_INPUT_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def small_base_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(*NEW_INPUT_SIZE, 3)))
    model.add(Convolution2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Activation("relu"))
    return model

# === Embedding Extractor ===
class EmbeddingExtractor:
    def __init__(self, model):  # fix: use __init__ not _init_
        self.model = model

    def forward(self, img: np.ndarray):
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0)
        if img.ndim == 4:
            return self.model.predict_on_batch(img)
        raise ValueError(f"Input shape not supported: {img.shape}")

# === Load Model ===
def load_model():
    print(f"\nLoading model: VGG16 + GlobalAvgPool + Dense(256)...")
    try:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        model = Model(inputs=base_model.input, outputs=x)
        print("Model loaded successfully!")
        return EmbeddingExtractor(model), preprocess_input
    except Exception as e:
        print(f"\nERROR loading model: {str(e)}")
        return None, None

# === Image Preprocessing ===
def preprocess_image(image_path, preprocess_function):
    try:
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype='float32')
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_function(img_array)
    except Exception as e:
        print(f"\nERROR processing image {image_path}: {str(e)}")
        return None

# === Get Embedding ===
def get_face_embedding(extractor, preprocessed_image):
    try:
        embedding = extractor.forward(preprocessed_image)
        return np.array(embedding).reshape(1, -1)
    except Exception as e:
        print(f"\nERROR getting embedding: {str(e)}")
        return None

# === Compare Faces ===
def compare_faces(reference_embedding, gallery_embedding):
    try:
        ref = reference_embedding.astype('float32')
        gal = gallery_embedding.astype('float32')
        faiss.normalize_L2(ref)
        faiss.normalize_L2(gal)
        sim = float((ref @ gal.T)[0][0])
        return sim
    except Exception as e:
        print(f"\nERROR comparing faces with FAISS: {str(e)}")
        return 0.0

# === Plot PCA ===
def plot_embeddings(reference_embedding, gallery_embeddings, matching_files, filenames):
    try:
        all_embeddings = [reference_embedding]
        all_embeddings.extend(gallery_embeddings)
        embeddings_array = np.vstack(all_embeddings)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings_array)

        ref_point = reduced[0]
        gallery_points = reduced[1:]
        match_indices = [i for i, (name, _) in enumerate(matching_files)]
        non_match_indices = [i for i in range(len(gallery_points)) if i not in match_indices]

        plt.figure(figsize=(12, 10))
        plt.scatter(gallery_points[non_match_indices, 0], gallery_points[non_match_indices, 1], c='gray', label='Non-matches')
        plt.scatter(gallery_points[match_indices, 0], gallery_points[match_indices, 1], c='green', label='Matches')
        plt.scatter([ref_point[0]], [ref_point[1]], c='red', marker='*', s=300, label='Reference')
        for i, (x, y) in enumerate(gallery_points):
            if i in match_indices:
                label = f"{filenames[i][:10]}... ({matching_files[match_indices.index(i)][1]:.2f})"
                plt.annotate(label, (x, y), fontsize=8, alpha=0.8)

        plt.title('Face Embedding Visualization (PCA)')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(output_folder, 'embeddings_plot.png')
        plt.savefig(plot_path, dpi=150)
        print(f"PCA plot saved to: {plot_path}")
        return True
    except Exception as e:
        print(f"\nERROR generating PCA plot: {str(e)}")
        return False

# === Main Benchmark & Matching ===
def benchmark_and_match_vggface(gallery_path, reference_image_path1, reference_image_path2, output_folder):
    print("\n=== Starting Benchmark and Matching ===")
    total_execution_start = time.time()  # Start total execution timer
    total_preprocessing_time = 0
    total_embedding_time = 0
    similarity_threshold = 0.8  # Define similarity threshold
    model_name = "VGG16"  # Define model name

    extractor, preprocess_function = load_model()
    if extractor is None:
        return False

    # Preprocess reference images
    start = time.time()
    reference_image1 = preprocess_image(reference_image_path1, preprocess_function)
    reference_image2 = preprocess_image(reference_image_path2, preprocess_function)
    preprocessing_time = time.time() - start
    total_preprocessing_time += preprocessing_time

    if reference_image1 is None or reference_image2 is None:
        return False

    # Get reference embeddings
    start = time.time()
    reference_embedding1 = get_face_embedding(extractor, reference_image1)
    reference_embedding2 = get_face_embedding(extractor, reference_image2)
    embedding_time = time.time() - start
    total_embedding_time += embedding_time

    if reference_embedding1 is None or reference_embedding2 is None:
        return False

    processing_times = []
    similarities = []
    matching_files = []
    gallery_embeddings = []
    filenames = []

    image_files = [f for f in os.listdir(gallery_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No image files found in gallery directory")
        return False

    print(f"Found {len(image_files)} images to process")
    for filename in image_files:
        image_path = os.path.join(gallery_path, filename)

        # Preprocess gallery image
        start = time.time()
        gallery_image = preprocess_image(image_path, preprocess_function)
        preprocessing_time = time.time() - start
        total_preprocessing_time += preprocessing_time

        if gallery_image is None:
            continue

        # Get gallery embedding
        start = time.time()
        embedding = get_face_embedding(extractor, gallery_image)
        embedding_time = time.time() - start
        total_embedding_time += embedding_time
        processing_times.append(embedding_time)

        if embedding is not None:
            gallery_embeddings.append(embedding)
            filenames.append(filename)

            similarity1 = compare_faces(reference_embedding1, embedding)
            similarity2 = compare_faces(reference_embedding2, embedding)
            similarities.append(max(similarity1, similarity2))

            if similarity1 >= similarity_threshold or similarity2 >= similarity_threshold:
                matching_files.append((filename, max(similarity1, similarity2)))
                shutil.copy(image_path, os.path.join(output_folder, filename))

    # Print total number of matches
    print(f"Total Matching Images Found: {len(matching_files)}")

    average_speed = sum(processing_times) / len(processing_times) if processing_times else 0
    average_similarity = sum(similarities) / len(similarities) if similarities else 0
    ram_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    total_execution_time = time.time() - total_execution_start

    print(f"\n--- Benchmark Results for {model_name} ---")
    print(f"Total Preprocessing Time: {total_preprocessing_time:.4f} seconds")
    print(f"Total Embedding Creation Time: {total_embedding_time:.4f} seconds")
    print(f"Total Execution Time: {total_execution_time:.4f} seconds")
    print(f"Average Speed: {average_speed:.4f} s/image")
    print(f"Average Similarity: {average_similarity:.4f}")
    print(f"RAM Usage: {ram_usage:.2f} MB")
    print(f"Matching images saved to: {output_folder}")

    if gallery_embeddings and matching_files:
        plot_embeddings(reference_embedding1, gallery_embeddings, matching_files, filenames)

    return True

# === Run the Full System ===
print("\n=== Face Matching System ===")
success = benchmark_and_match_vggface(gallery_path, reference_image_path1, reference_image_path2, output_folder)
if success:
    print("\nProcess completed successfully!")
else:
    print("\nProcess completed with errors")
print("\n=== End of Program ===")
