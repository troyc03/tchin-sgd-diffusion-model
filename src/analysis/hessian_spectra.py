import os

# Environment fix for OpenMP collisions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from numpy.linalg import eigh
import matplotlib.pyplot as plt

class HessianSpectraAnalyzer:
    # ... (rest of the __init__ method) ...
    def __init__(self, model, data_loader, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device

    def compute_hessian(self):
        """Compute the Hessian matrix of the loss w.r.t. model parameters."""
        self.model.zero_grad()
        x, y = next(iter(self.data_loader))
        x, y = x.to(self.device), y.to(self.device)
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward(create_graph=True)

        params = [p for p in self.model.parameters() if p.requires_grad]
        num_params = sum(p.numel() for p in params)
        hessian = torch.zeros(num_params, num_params).to(self.device)
        
        # Compute Hessian using autograd
        for i in range(num_params):
            # 1. Zero out previous gradients
            self.model.zero_grad() 
            
            # 2. Extract the i-th component of the first gradient
            #    We need to compute the full first gradient first.
            grad1 = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
            # Flatten all gradient tensors into a single vector
            grad1_vec = torch.cat([g.view(-1) for g in grad1 if g is not None])
            
            # Select the i-th element to compute its gradient
            if i < len(grad1_vec):
                grad_i = grad1_vec[i]
                
                # 3. Compute the gradient of the i-th component (the i-th row of the Hessian)
                grad2_i = torch.autograd.grad(grad_i, params, retain_graph=True, allow_unused=True)
                # Flatten the second gradient tensors into a single vector
                grad2_i_vec = torch.cat([g.view(-1) for g in grad2_i if g is not None])
                
                # 4. Assign the resulting vector to the i-th row of the Hessian matrix
                if len(grad2_i_vec) == num_params:
                    hessian[i] = grad2_i_vec
                else:
                    raise RuntimeError(f"Hessian row size mismatch: Expected {num_params}, got {len(grad2_i_vec)}")
            else:
                # Handle potential edge case if the grad1_vec is smaller than expected
                raise IndexError(f"Index {i} out of bounds for grad1_vec of size {len(grad1_vec)}")

        return hessian.cpu().numpy()

    def analyze_spectra(self, hessian):
        """Compute eigenvalues and eigenvectors of the Hessian."""
        eigenvalues, eigenvectors = eigh(hessian)
        return eigenvalues, eigenvectors

    def plot_spectra(self, eigenvalues):
        plt.figure(figsize=(10, 6))
        plt.plot(eigenvalues, 'o-', color='blue', label='Eigenvalues')
        plt.title('Hessian Eigenvalues')
        plt.xlabel('Index')
        plt.ylabel('Eigenvalue')
        plt.axvline(0, color='red', ls='--', lw=1, label='Zero Line')
        plt.yscale('symlog', linthresh=1e-5)  # Log scale for better visibility of small eigenvalues
        plt.grid(True)
        plt.legend()
        plt.show()

def main():
    # Example usage with a simple model and dataset
    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 1)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    # Create synthetic dataset
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)
    data_loader = DataLoader(dataset, batch_size=32)
    model = SimpleNN()
    loss_fn = nn.MSELoss()
    analyzer = HessianSpectraAnalyzer(model, data_loader, loss_fn)
    hessian = analyzer.compute_hessian()
    eigenvalues, _ = analyzer.analyze_spectra(hessian)
    analyzer.plot_spectra(eigenvalues)

if __name__ == '__main__':
    main()
