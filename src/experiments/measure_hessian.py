import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

# 1. Hessian Spectrum Analysis Logic
class HessianSpectraAnalyzer:
    def __init__(self, model, data_loader, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device

    def compute_hessian(self):
        """Compute the full Hessian matrix for a single batch."""
        self.model.zero_grad()
        # Use a single batch to avoid OOM for full Hessian
        x, y = next(iter(self.data_loader))
        x, y = x.to(self.device), y.to(self.device)
        
        output = self.model(x)
        loss = self.loss_fn(output, y)
        
        # First-order gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        grad1 = torch.autograd.grad(loss, params, create_graph=True)
        grad1_vec = torch.cat([g.contiguous().view(-1) for g in grad1])
        
        num_params = grad1_vec.size(0)
        hessian = torch.zeros(num_params, num_params).to(self.device)
        
        # Second-order gradients row by row
        for i in range(num_params):
            grad2_i = torch.autograd.grad(grad1_vec[i], params, retain_graph=True)
            grad2_i_vec = torch.cat([g.contiguous().view(-1) for g in grad2_i])
            hessian[i] = grad2_i_vec
            
        return hessian.detach().cpu().numpy()

    def analyze_spectra(self, hessian):
        """Returns sorted eigenvalues and eigenvectors."""
        eigenvalues, eigenvectors = eigh(hessian)
        return eigenvalues, eigenvectors

# 2. Main Execution Steps
def main():
    # Model definition
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5) # Small size for tractable Hessian
            self.fc2 = nn.Linear(5, 1)
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    # Synthetic Data Preparation
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    data_loader = DataLoader(TensorDataset(X, y), batch_size=32)
    
    model = SimpleNN()
    loss_fn = nn.MSELoss()
    analyzer = HessianSpectraAnalyzer(model, data_loader, loss_fn)
    
    print("Computing Hessian...")
    hessian = analyzer.compute_hessian()
    
    print("Analyzing Spectrum...")
    eigenvalues, _ = analyzer.analyze_spectra(hessian)
    
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.semilogy(np.abs(eigenvalues), 'o-')
    plt.title('Hessian Eigenvalue Magnitude')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
