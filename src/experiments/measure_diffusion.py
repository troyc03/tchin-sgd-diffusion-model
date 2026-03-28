import os

# Environment fix for OpenMP collisions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class HessianSpectraAnalyzer:
    def __init__(self, model, data_loader, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device

    def compute_hessian(self):
        """Compute the full Hessian matrix of the loss w.r.t. model parameters."""
        self.model.zero_grad()
        # Get a single batch for the Hessian calculation
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
        
        for i in range(num_params):
            # Compute the gradient of the i-th gradient element w.r.t. parameters
            grad2_i = torch.autograd.grad(grad1_vec[i], params, retain_graph=True)
            grad2_i_vec = torch.cat([g.contiguous().view(-1) for g in grad2_i])
            hessian[i] = grad2_i_vec
            
        return hessian.detach().cpu().numpy()

    def analyze_spectra(self, hessian):
        eigenvalues, _ = eigh(hessian)
        return eigenvalues

    def plot_spectra(self, eigenvalues, title="Hessian Eigenvalues"):
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvalues, 'o-', markersize=4, color='blue')
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.axhline(0, color='red', ls='--', lw=1)
        plt.yscale('symlog', linthresh=1e-5)
        plt.grid(True, alpha=0.3)
        plt.show()

def measure_diffusion(model, data_loader, loss_fn, num_iterations=3):
    analyzer = HessianSpectraAnalyzer(model, data_loader, loss_fn)
    diffusion_measurements = []
    
    print(f"Starting diffusion measurement over {num_iterations} iterations...")
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        hessian = analyzer.compute_hessian()
        eigenvalues = analyzer.analyze_spectra(hessian)
        diffusion_measurements.append(eigenvalues)
    return diffusion_measurements

def main():
    # Simple model for fast computation
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(5, 4)
            self.fc2 = nn.Linear(4, 1)
        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    # Synthetic data
    X = torch.randn(64, 5)
    y = torch.randn(64, 1)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32)
    
    model = SimpleNN()
    loss_fn = nn.MSELoss()
    
    # Analyze single spectra
    analyzer = HessianSpectraAnalyzer(model, data_loader, loss_fn)
    hessian = analyzer.compute_hessian()
    eigenvalues = analyzer.analyze_spectra(hessian)
    analyzer.plot_spectra(eigenvalues)
    
    # Measure diffusion over steps
    # (Note: In a real scenario, you'd perform model.step() between these)
    diffusion = measure_diffusion(model, data_loader, loss_fn, num_iterations=2)
    print(f"Captured {len(diffusion)} spectral snapshots.")

if __name__ == '__main__':
    main()
