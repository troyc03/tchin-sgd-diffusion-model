import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =========================
# Model
# =========================
class MyModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# =========================
# Enhanced Hessian + Lanczos
# =========================
class HessianAnalyzer:
    def __init__(self, model, dataloader, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.dim = sum(p.numel() for p in self.params)

    def _hvp(self, v):
        """Hessian-Vector Product averaged over the dataset."""
        hv_accum = torch.zeros(self.dim, device=self.device)
        n_batches = len(self.dataloader)
        
        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            
            # First gradient
            grads = grad(loss, self.params, create_graph=True)
            flat_grad = torch.cat([g.view(-1) for g in grads])
            
            # Inner product for HVP
            grad_v = torch.dot(flat_grad, v)
            
            # Second gradient
            hv = grad(grad_v, self.params, retain_graph=False)
            hv_flat = torch.cat([h.contiguous().view(-1) for h in hv])
            
            hv_accum += hv_flat / n_batches
            
        return hv_accum

    def run_lanczos(self, max_iter=50, tol=1e-5):
        """Lanczos algorithm to find Ritz values (eigenvalue approximations)."""
        alpha, beta = [], []
        Vs = []
        
        # Initial random vector
        v = torch.randn(self.dim, device=self.device)
        v /= torch.norm(v)
        Vs.append(v)

        for j in tqdm(range(max_iter), desc="Lanczos Steps"):
            hv = self._hvp(Vs[j])
            
            a = torch.dot(Vs[j], hv)
            alpha.append(a)
            
            # Orthogonalize
            w = hv - a * Vs[j]
            if j > 0:
                w -= beta[-1] * Vs[j-1]
            
            b = torch.norm(w)
            if b < tol: break
            
            beta.append(b)
            Vs.append(w / b)

        # Construct Tridiagonal Matrix T
        k = len(alpha)
        T = torch.zeros((k, k), device=self.device)
        for i in range(k):
            T[i, i] = alpha[i]
            if i < k - 1:
                T[i, i+1] = beta[i]
                T[i+1, i] = beta[i]
        
        evals, evecs_T = torch.linalg.eigh(T)
        return evals.cpu().numpy(), Vs, evecs_T.cpu().numpy()

# =========================
# Spectral Density (ESD)
# =========================
def plot_spectral_density(evals, sigma=0.1):
    """Plots the Empirical Spectral Density using Gaussian smoothing."""
    x_min, x_max = evals.min() - 1, evals.max() + 1
    x = np.linspace(x_min, x_max, 1000)
    
    # Sum of Gaussians centered at each eigenvalue
    density = np.sum([np.exp(-(x - ev)**2 / (2 * sigma**2)) for ev in evals], axis=0)
    density /= (len(evals) * np.sqrt(2 * np.pi * sigma**2))

    plt.figure(figsize=(10, 4))
    plt.fill_between(x, density, alpha=0.3, color='blue')
    plt.plot(x, density, color='blue', lw=2)
    plt.title("Approximated Hessian Spectral Density (ESD)")
    plt.xlabel("Eigenvalue λ")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.show()

# =========================
# Execution
# =========================
if __name__ == "__main__":
    # 1. Setup Mock Data
    X = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))
    loader = DataLoader(TensorDataset(X, y), batch_size=32)

    # 2. Initialize Model
    model = MyModel()
    analyzer = HessianAnalyzer(model, loader)

    # 3. Compute Spectrum
    # top_k in your prompt implies iterations here
    evals, Vs, evecs_T = analyzer.run_lanczos(max_iter=30)

    # 4. Results & Plotting
    print(f"\nTop Eigenvalue (Spectral Radius): {evals.max():.4f}")
    print(f"Bottom Eigenvalue: {evals.min():.4f}")
    
    plt.figure(figsize=(12, 5))
    
    # Plot Eigenvalues
    plt.subplot(1, 2, 1)
    plt.stem(evals)
    plt.title("Recovered Ritz Values")
    plt.xlabel("Index")
    plt.ylabel("λ")
    
    # Plot Magnitude
    plt.subplot(1, 2, 2)
    plt.semilogy(np.abs(evals), 'ro')
    plt.title("Log-Scale Eigenvalues")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 5. Density Visualization
    plot_spectral_density(evals, sigma=0.05)
