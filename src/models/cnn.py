import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Simple CNN model for MNIST classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Calculate the size of the flattened layer: (Image size after pooling) * 32
        # MNIST images are 28x28. After two 2x2 max pools, they become 7x7.
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_parameter_vector(self):
        """
        Flatten the model parameters into a single vector.
        This function now operates on the instance it belongs to.
        """
        param_vector = []
        for param in self.parameters():
            # Use .detach().cpu().numpy() for safe operation with tensors
            param_vector.append(param.data.detach().cpu().numpy().flatten())
        return np.concatenate(param_vector)
    
    def set_parameter_vector(self, param_vector):
        """
        Set the model parameters from a flattened vector.
        This function now operates on the instance it belongs to.
        """
        pointer = 0
        for param in self.parameters():
            num_param = param.numel() # number of parameters in this layer
            # Copy the corresponding parameters from the vector to the model
            param.data.copy_(torch.from_numpy(param_vector[pointer:pointer + num_param].reshape(param.shape)))
            pointer += num_param

# Function to train the model
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Function to test the model
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')

# Main execution block
if __name__ == '__main__':
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 3

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST data
    # Use the official [PyTorch documentation](https://pytorch.org) for details on the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Data loaders
    # The [PyTorch DataLoader documentation](https://pytorch.org) provides comprehensive usage information
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, and train
    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE) # Use the [torch.optim documentation](https://pytorch.org) for other optimization algorithms

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    print("Training finished.")

    # Demonstration of parameter vector functions
    print("\n--- Demonstrating get/set parameter vectors ---")
    original_vector = model.get_parameter_vector()
    print(f"Original parameter vector shape: {original_vector.shape}")
    
    # Create a new, untrained model instance
    new_model = SimpleCNN().to(device)
    # The new model should have randomly initialized parameters, so its vector should be different
    initial_new_vector = new_model.get_parameter_vector()
    print(f"New model's initial parameter vector is different: {not np.array_equal(original_vector, initial_new_vector)}")

    # Set the new model's parameters to those of the trained model
    new_model.set_parameter_vector(original_vector)
    copied_new_vector = new_model.get_parameter_vector()
    print(f"After setting parameters, vectors are identical: {np.array_equal(original_vector, copied_new_vector)}")
    print("New model is now a clone of the trained model.")
