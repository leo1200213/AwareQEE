import torch
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import timm
import os
import time
from tqdm import tqdm
import random

# ------------------------------
# 1. Define Transformations
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize CIFAR-100 images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# ------------------------------
# 2. Load CIFAR-100 Dataset
# ------------------------------
print("Loading CIFAR-100 test dataset...")
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=8)
print("CIFAR-100 test dataset loaded.")

# ------------------------------
# 3. Create Calibration Subset
# ------------------------------
# Use a subset of CIFAR-100 for calibration (e.g., 100 samples)
calibration_size = 100
indices = list(range(len(test_dataset)))
random.seed(42)
random.shuffle(indices)
calibration_indices = indices[:calibration_size]
calibration_subset = Subset(test_dataset, calibration_indices)
calibration_loader = DataLoader(calibration_subset, batch_size=32, shuffle=False, num_workers=8)
print("Calibration subset created.")

# ------------------------------
# 4. Load the Pretrained DeiT-Base Model
# ------------------------------
print("Loading the pretrained DeiT-Base model...")
model = timm.create_model('deit_base_patch16_224', pretrained=True)
print("Model loaded.")

# ------------------------------
# 5. Modify the Classification Head
# ------------------------------
print("Modifying the classification head for CIFAR-100...")
model.head = nn.Linear(model.head.in_features, 100)  # CIFAR-100 has 100 classes
print("Classification head modified.")

# ------------------------------
# 6. Define Loss Function and Optimizer
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ------------------------------
# 7. Load CIFAR-100 Training Dataset
# ------------------------------
print("Loading CIFAR-100 training dataset...")
train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
print("CIFAR-100 training dataset loaded.")

# ------------------------------
# 8. Training Loop
# ------------------------------
num_epochs = 1  # Adjust based on your requirements
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Training on device: {device}")

print("Starting fine-tuning...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
print("Fine-tuning completed.")

# ------------------------------
# 9. Save the Fine-Tuned Model
# ------------------------------
model_save_path = 'fine_tuned_deit_cifar100.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Fine-tuned model saved to {model_save_path}.")

# ------------------------------
# 10. Load the Fine-Tuned Model (for quantization)
# ------------------------------
print("Loading the fine-tuned model for quantization...")
model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
model.to('cpu')  # Quantization is typically done on CPU
model.eval()
print("Fine-tuned model loaded and moved to CPU.")

# ------------------------------
# 11. Define Quantization Configuration
# ------------------------------
print("Setting up quantization configuration...")

# Define the default qconfig for static quantization
qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Assign qconfig to the model
model.qconfig = qconfig

# ------------------------------
# 12. Prepare the Model for Quantization
# ------------------------------
print("Preparing the model for static quantization...")

# Define a helper function to apply quantization only to Linear and Conv2d layers
def prepare_model_for_quantization(model):
    """
    Prepares the model for static quantization by specifying that only
    nn.Linear and nn.Conv2d modules should be quantized.
    """
    # Iterate over all modules in the model
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.quantization.prepare(module, inplace=True)
    return model

model_prepared = prepare_model_for_quantization(model)
print("Model prepared for static quantization.")

# ------------------------------
# 13. Calibrate the Model
# ------------------------------
def calibrate(model, loader):
    """
    Runs calibration data through the model to collect activation statistics.
    """
    model.eval()
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Calibrating"):
            model(inputs)

print("Calibrating the model with calibration data...")
calibrate(model_prepared, calibration_loader)
print("Calibration completed.")

# ------------------------------
# 14. Convert to Quantized Model
# ------------------------------
print("Converting the calibrated model to a quantized model...")
quantized_model = torch.quantization.convert(model_prepared, inplace=False)
print("Model converted to quantized version.")

# ------------------------------
# 15. Define Evaluation Function
# ------------------------------
def evaluate(model, dataloader, device):
    """
    Evaluates the model on the given dataloader and returns accuracy.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# ------------------------------
# 16. Evaluate the Quantized Model
# ------------------------------
device = torch.device('cpu')  # Quantized models are optimized for CPU
print(f"Evaluating on device: {device}")

print("Evaluating the quantized model on the CIFAR-100 test set...")
quantized_accuracy = evaluate(quantized_model, test_loader, device)
print(f"Quantized Model Test Accuracy: {quantized_accuracy*100:.2f}%")  # Should be meaningful after proper fine-tuning

# ------------------------------
# 17. Compare Model Sizes
# ------------------------------
def get_size(model, path="temp.p"):
    """
    Saves the model's state_dict to a temporary file to measure its size.
    """
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6  # Size in MB
    os.remove(path)
    return size

original_size = get_size(model)
quantized_size = get_size(quantized_model)

print(f"Original Fine-Tuned Model Size: {original_size:.2f} MB")
print(f"Quantized Model Size: {quantized_size:.2f} MB")

# ------------------------------
# 18. Measure Inference Time
# ------------------------------
def measure_inference_time(model, dataloader, device, runs=5):
    """
    Measures the average inference time of the model over a number of runs.
    """
    model.to(device)
    model.eval()
    total_time = 0.0
    with torch.no_grad():
        for run in tqdm(range(runs), desc="Measuring Inference Time"):
            start_time = time.time()
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                _ = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
    avg_time = total_time / runs
    return avg_time

print("Measuring inference time for the original model...")
original_time = measure_inference_time(model, test_loader, device, runs=5)
print(f"Average Inference Time (Original Model): {original_time:.4f} seconds")

print("Measuring inference time for the quantized model...")
quantized_time = measure_inference_time(quantized_model, test_loader, device, runs=5)
print(f"Average Inference Time (Quantized Model): {quantized_time:.4f} seconds")
