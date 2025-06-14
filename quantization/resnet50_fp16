import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import psutil
from torchvision import models, transforms
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Set paths
reference_image_paths = ["/content/drive/MyDrive/test_gal/gg/sample/ref.jpeg",
                        "/content/drive/MyDrive/test_gal/gg/sample/ref2.jpeg"]  # Added second reference image
gallery_folder = "/content/drive/MyDrive/test_gal/gg/ritual"
output_folder = "/content/drive/MyDrive/test_gal/op"
non_matching_folder = "/content/drive/MyDrive/test_gal/nm_op"
similarity_threshold = 0.5

# Create output folders
os.makedirs(output_folder, exist_ok=True)
os.makedirs(non_matching_folder, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Device:", device)

# Define ArcMarginProduct
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        if label is None:
            return cosine * self.s
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        return output

# Define ArcFace + ResNet50 Model
class ArcFaceResNet50(nn.Module):
    def __init__(self, num_classes=5749, pretrained=True, feature_dim=2048, s=30.0, m=0.50):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = feature_dim
        self.arcface = ArcMarginProduct(feature_dim, num_classes, s=s, m=m)

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        if labels is not None:
            logits = self.arcface(x, labels)
            return logits
        return x

# Load FP16 model
model = ArcFaceResNet50(num_classes=5749).to(device).half()
resnet_state_dict = torch.load('/content/drive/MyDrive/arc/fp16_model.pth', map_location=device)
resnet_state_dict = {k.replace('module.', ''): v for k, v in resnet_state_dict.items()}
model.backbone.load_state_dict(resnet_state_dict, strict=False)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def get_image_embedding(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device).half()
    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy()
    return embedding.flatten()

def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

# Benchmarking
start_time = time.time()
initial_ram = get_ram_usage()

# Get reference image embeddings
reference_embeddings = []
ref_embedding_times = []
for ref_path in reference_image_paths:
    ref_start_time = time.time()
    embedding = get_image_embedding(ref_path)
    reference_embeddings.append(embedding)
    ref_embedding_times.append(time.time() - ref_start_time)
ref_embedding_time = sum(ref_embedding_times)
ref_ram = get_ram_usage() - initial_ram

# Process gallery images
gallery_embeddings = []
gallery_filenames = []
processing_times = []
embedding_times = []
similarities = []
matching_images = []

initial_ram = get_ram_usage()
for filename in os.listdir(gallery_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(gallery_folder, filename)
        start_time_embed = time.time()
        embedding = get_image_embedding(img_path)
        embedding_time = time.time() - start_time_embed
        # Calculate max similarity across all reference images
        max_similarity = max(cosine_similarity([embedding], reference_embeddings)[0])

        gallery_embeddings.append(embedding)
        gallery_filenames.append(filename)
        processing_times.append(time.time() - start_time_embed)
        embedding_times.append(embedding_time)
        similarities.append(max_similarity)

        if max_similarity >= similarity_threshold:
            matching_images.append((filename, max_similarity))
            shutil.copy(img_path, os.path.join(output_folder, filename))
        else:
            shutil.copy(img_path, os.path.join(non_matching_folder, filename))

total_time_overall = time.time() - start_time
total_ram = get_ram_usage() - initial_ram

# Calculate metrics
avg_processing_speed = np.mean(processing_times) if processing_times else 0
avg_embedding_time = np.sum(embedding_times) if embedding_times else 0
avg_similarity = np.mean(similarities) if similarities else 0
total_matching_images = len(matching_images)

# Print results
print(f"Reference Embedding Time: {ref_embedding_time:.4f} seconds")
print(f"Total Embedding Time (Gallery): {avg_embedding_time:.4f} seconds")
print(f"Average Processing Speed (Gallery): {avg_processing_speed:.4f} seconds/image")
print(f"Average Similarity (Gallery vs. References): {avg_similarity:.4f}")
print(f"RAM Usage: {total_ram:.2f} MB")
print(f"Total time for gallery processing: {sum(processing_times):.4f} seconds")
print(f"Overall script execution time: {total_time_overall:.4f} seconds")
print(f"\nMatching Images: {len(matching_images)} saved to: {output_folder}")
print(f"Non-matching Images: {len(gallery_filenames) - len(matching_images)} saved to: {non_matching_folder}")

# Save benchmark results
with open(os.path.join(output_folder, "benchmark_results_fp16.txt"), "w") as f:
    f.write(f"Reference Embedding Time: {ref_embedding_time:.4f} seconds\n")
    f.write(f"Total Embedding Time (Gallery): {avg_embedding_time:.4f} seconds\n")
    f.write(f"Average Processing Speed (Gallery): {avg_processing_speed:.4f} seconds/image\n")
    f.write(f"Average Similarity (Gallery vs. References): {avg_similarity:.4f}\n")
    f.write(f"RAM Usage: {total_ram:.2f} MB\n")
    f.write(f"Total time for gallery processing: {sum(processing_times):.4f} seconds\n")
    f.write(f"Overall script execution time: {total_time_overall:.4f} seconds\n")
    f.write("\nMatching Images:\n")
    for filename, sim in matching_images:
        f.write(f"{filename}: Similarity = {sim:.4f}\n")
    f.write(f"\nTotal Matching Images: {total_matching_images}\n")
    f.write(f"Matching images saved to: {output_folder}\n")
    f.write(f"Non-matching images saved to: {non_matching_folder}\n")

# Visualize embeddings using t-SNE
if gallery_embeddings:
    all_embeddings = np.vstack([ref.reshape(1, -1) for ref in reference_embeddings] + [gallery_embeddings])
    all_labels = ['Reference 1', 'Reference 2'] + gallery_filenames  # Updated labels
    perplexity_val = min(30, len(all_embeddings)-1)
    if perplexity_val < 2:
        print("Not enough samples for t-SNE visualization with perplexity >= 2.")
    else:
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)
            embeddings_2d = tsne.fit_transform(all_embeddings)
            ref_points = embeddings_2d[:2]  # First two are reference images
            gallery_points = embeddings_2d[2:]
            matching_indices = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
            non_matching_indices = [i for i, sim in enumerate(similarities) if sim < similarity_threshold]
            plt.figure(figsize=(10, 8))
            if non_matching_indices:
                plt.scatter(
                    gallery_points[non_matching_indices, 0],
                    gallery_points[non_matching_indices, 1],
                    c='blue', label='Non-matching Images', alpha=0.6
                )
            if matching_indices:
                plt.scatter(
                    gallery_points[matching_indices, 0],
                    gallery_points[matching_indices, 1],
                    c='green', label='Matching Images', alpha=0.6
                )
            plt.scatter(ref_points[:, 0], ref_points[:, 1], c='red', label='Reference Images', s=100, marker='*')
            if len(gallery_filenames) < 50:
                for i, filename in enumerate(gallery_filenames):
                    plt.annotate(filename, (gallery_points[i, 0], gallery_points[i, 1]), fontsize=8, alpha=0.75)
            plt.title('t-SNE Visualization of Image Embeddings')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_folder, 'embeddings_visualization.png'))
            plt.close()
            print(f"Embedding visualization saved to: {os.path.join(output_folder, 'embeddings_visualization.png')}")
        except ValueError as e:
            print(f"Could not perform t-SNE visualization: {e}")
else:
    print("No gallery images to visualize.")
