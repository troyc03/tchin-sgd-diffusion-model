import torch
from torch.nn import functional as F
import torch.nn as nn
import numpy as np

# Simple CNN model for MNIST classification

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_parameter_vector(self, model):
        """
        Flatten the model parameters into a single vector.
        """
        param_vector = []
        for param in model.parameters():
            param_vector.append(param.data.cpu().numpy().flatten())
        return np.concatenate(param_vector)
    
    def set_parameter_vector(self, model, param_vector):
        """
        Set the model parameters from a flattened vector.
        """
        pointer = 0
        for param in model.parameters():
            num_param = param.numel() # number of parameters in this layer
            # Copy the corresponding parameters from the vector to the model
            param.data.copy_(torch.from_numpy(param_vector[pointer:pointer + num_param].reshape(param.shape)))
            pointer += num_param
            