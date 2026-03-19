import os
import sys

# THIS MUST BE THE FIRST LINE EXECUTED
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.func import vmap, grad, functional_call
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class GradientNoiseAnalyzer:
    def __init__(self, model, data_loader, loss_fn):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn

    def get_per_sample_grads(self, images, labels):
        """Helper to get individual gradients for the batch."""
        params = dict(self.model.named_parameters())
        
        def compute_loss_stateless(params, x, y):
            out = functional_call(self.model, params, x.unsqueeze(0))
            return self.loss_fn(out, y.unsqueeze(0))

        return vmap(grad(compute_loss_stateless), in_dims=(None, 0, 0))(params, images, labels)

    def compute_anisotropy(self, batch):
        """
        Calculates how 'stretched' the noise is.
        Anisotropy = Max Eigenvalue of Covariance / Mean Eigenvalue.
        """
        images, labels = batch
        per_sample_grads = self.get_per_sample_grads(images, labels)
        
        anisotropy_stats = {}
        
        for name, grads in per_sample_grads.items():
            if grads.dim() < 2: continue # Skip scalars
            
            # Flatten parameters: [batch, features]
            flat_grads = grads.view(grads.shape[0], -1)
            
            # Center the gradients to get noise: (g_i - g_avg)
            noise = flat_grads - flat_grads.mean(dim=0)
            
            # Compute the top eigenvalue of the Noise Covariance B using Power Iteration
            # This represents the 'widest' direction of the noise cloud
            v = torch.randn(1, flat_grads.shape[1]).to(grads.device)
            for _ in range(5): # Quick power iteration
                v = torch.mm(v, noise.t())
                v = torch.mm(v, noise)
                v = v / torch.norm(v)
            
            top_eigenval = torch.norm(torch.mm(v, noise.t()))**2 / noise.shape[0]
            avg_eigenval = torch.var(flat_grads, dim=0).sum()
            
            anisotropy_stats[name] = {
                "top_eig": top_eigenval.item(),
                "avg_eig": avg_eigenval.item(),
                "ratio": (top_eigenval / (avg_eigenval + 1e-6)).item()
            }
            
        return anisotropy_stats
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.fc = nn.Linear(1440, 10)
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    # Simplified data trigger
    print("Initializing Model...")
    model = SimpleCNN().to(device)
    
    # Fake a batch so we don't trigger the complex MNIST loader yet
    # This checks if the logic works without the library conflict
    images = torch.randn(64, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (64,)).to(device)
    batch = (images, labels)

    analyzer = GradientNoiseAnalyzer(model, None, F.cross_entropy)
    
    print("Analyzing synthetic gradient noise...")
    stats = analyzer.compute_anisotropy(batch)

    print(f"{'Layer Name':<20} | {'Anisotropy Ratio':<20}")
    print("-" * 45)
    for name, stat in stats.items():
        print(f"{name:<20} | {stat['ratio']:.4f}")