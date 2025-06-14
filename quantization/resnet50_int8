# 1. Install dependencies (if needed)
!pip install torch torchvision onnx onnxruntime tqdm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torch.quantization import prepare_qat, convert
from tqdm import tqdm


# 3. Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
from google.colab import drive
drive.mount('/content/drive')
DATA_ROOT = "/content/drive/MyDrive/lfw_home/lfw_funneled"  # Place your dataset here: /content/data/class_name/*.jpg
BATCH_SIZE = 64
IMAGE_SIZE = 112
NUM_WORKERS = 2
EPOCHS = 5   # For demo, increase for real training
LEARNING_RATE = 0.01
MOMENTUM = 0.9
ONNX_PATH = "/content/drive/MyDrive/arcface_resnet50_qat_int8.onnx"
MODEL_PATH = "/content/drive/MyDrive/arcface_resnet50_qat_int8.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = nn.functional.linear(nn.functional.normalize(input), nn.functional.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        logits = self.s * (one_hot * phi + (1.0 - one_hot) * cosine)
        return logits

class ArcFaceResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()
        self.arcface = ArcFace(2048, num_classes)
    def forward(self, x, labels=None):
        x = self.backbone(x)
        if labels is not None:
            x = self.arcface(x, labels)
        return x


model = ArcFaceResNet50(num_classes).to(device)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
prepare_qat(model, inplace=True)
model.train()

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
criterion = nn.CrossEntropyLoss()


for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader.dataset):.4f}")

# ...after QAT training loop, but before quantization...
model.eval()
model.cpu()

dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
dummy_label = torch.zeros(1, dtype=torch.long)
torch.onnx.export(
    model,  # NOT quantized_model
    (dummy_input, dummy_label),
    ONNX_PATH,
    input_names=["input", "labels"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True
)
print(f"ONNX model exported at: {ONNX_PATH}")

model.eval()
model.cpu()
quantized_model = convert(model, inplace=False)
torch.save(quantized_model.state_dict(), MODEL_PATH)
print(f"Quantized model saved at: {MODEL_PATH}")

dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
dummy_label = torch.zeros(1, dtype=torch.long)
torch.onnx.export(
    quantized_model,
    (dummy_input, dummy_label),
    ONNX_PATH,
    input_names=["input", "labels"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True
)
print(f"ONNX model exported at: {ONNX_PATH}")

import onnx
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")


import onnxruntime as ort
ort_session = ort.InferenceSession(ONNX_PATH)
outputs = ort_session.run(
    None,
    {"input": dummy_input.numpy(), "labels": dummy_label.numpy()}
)
print("ONNX inference output shape:", outputs[0].shape)

from google.colab import files
files.download(ONNX_PATH)

from google.colab import drive
drive.flush_and_unmount()
