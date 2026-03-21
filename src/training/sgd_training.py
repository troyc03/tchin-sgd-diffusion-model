from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# 1. Setup Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# 2. Iterate and Train the Pipeline
pipeline = Pipeline([
    ("Scaler", StandardScaler()),
    ("sgd_classifier", SGDClassifier(max_iter=1000, random_state=42, warm_start=True)) # warm_start=True allows incremental training
])

print("Starting training...")

# Need to get all possible class labels for the first call to partial_fit
# MNIST has 10 classes (0-9)
classes = np.arange(10)

for batch_idx, (images, labels) in enumerate(train_loader):
    # Convert torch tensors to numpy arrays
    # Use .detach().cpu().numpy() to handle tensors that might be on a GPU or require gradients
    images_np = images.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Flatten the images from [batch_size, channels, height, width] to [batch_size, n_features]
    # For MNIST, this is [64, 1, 28, 28] -> [64, 784]
    n_samples = images_np.shape[0]
    images_np = images_np.reshape(n_samples, -1)

    # Use partial_fit for incremental learning on each batch
    # The first call to partial_fit needs the list of all unique classes
    if batch_idx == 0:
        pipeline.named_steps['sgd_classifier'].partial_fit(images_np, labels_np, classes=classes)
    else:
        pipeline.named_steps['sgd_classifier'].partial_fit(images_np, labels_np)

    if batch_idx % 100 == 0:
        print(f"Trained on batch {batch_idx}")

print("Training finished.")
