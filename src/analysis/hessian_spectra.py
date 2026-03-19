import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

class Hessian:
    def __init__(self, model, dataloader, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.model.eval()
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.dim = sum(p.numel() for p in self.params)

    def _hessian_vector_product(self, v):
        try:
            inputs, targets = next(self.data_iter)
        except (AttributeError, StopIteration):
            self.data_iter = iter(self.dataloader)
            inputs, targets = next(self.data_iter)

        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        # First gradient
        grads = grad(loss, self.params, create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grads])
        
        # Second gradient (HVP)
        v = v.to(self.device)
        grad_v_prod = torch.sum(flat_grad * v)
        hv = grad(grad_v_prod, self.params, retain_graph=False)
        return torch.cat([g.contiguous().view(-1) for g in hv])

    def lanczos(self, k=50, max_iter=100, tol=1e-6):
        m = max_iter
        alpha = torch.zeros(m, device=self.device)
        beta = torch.zeros(m, device=self.device)
        Vs = []
        
        v = torch.randn(self.dim, device=self.device)
        v /= torch.norm(v)
        Vs.append(v.clone())
        
        pbar = tqdm(range(m), desc="Lanczos Iterations")
        for j in pbar:
            hv = self._hessian_vector_product(Vs[j])
            alpha[j] = torch.dot(Vs[j], hv)
            
            w = hv - alpha[j] * Vs[j]
            if j > 0:
                w -= beta[j-1] * Vs[j-1]
            
            bj = torch.norm(w)
            if bj < tol or j == m - 1:
                break
                
            beta[j] = bj
            Vs.append(w / bj)
            
        actual_iters = len(Vs)
        T = torch.zeros((actual_iters, actual_iters), device=self.device)
        for i in range(actual_iters):
            T[i, i] = alpha[i]
            if i < actual_iters - 1:
                T[i, i+1] = beta[i]
                T[i+1, i] = beta[i]
        
        evals, evecs_T = torch.linalg.eigh(T)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs_T = evecs_T[:, idx]
        
        V_matrix = torch.stack(Vs, dim=1)
        # Limit k to actual iterations performed
        k = min(k, actual_iters)
        ritz_vectors = V_matrix @ evecs_T[:, :k]
        
        return evals[:k].cpu().numpy(), ritz_vectors.cpu().numpy()

# Example Model: A simple 3-layer MLP
class MyModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(MyModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Flatten image inputs if necessary
        x = x.view(x.size(0), -1)
        return self.layers(x)

def compute_hessian_spectrum(model, dataloader, top_k=50, max_iter=100):
    hess = Hessian(model, dataloader)
    evals, evecs = hess.lanczos(k=top_k, max_iter=max_iter)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(evals) + 1), evals, 'o-', color='crimson')
    plt.yscale('log')
    plt.xlabel('Eigenvalue Rank')
    plt.ylabel('Eigenvalue Magnitude')
    plt.title(f'Hessian Spectrum (Top {len(evals)})')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    
    return evals, evecs

if __name__ == '__main__':
    # 1. Create dummy data (mimicking MNIST: 28x28 images, 10 classes)
    X = torch.randn(500, 784)
    y = torch.randint(0, 10, (500,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 2. Initialize Model
    model = MyModel()

    # 3. Run Analysis
    print("Starting Hessian Spectrum Calculation...")
    eigenvalues, eigenvectors = compute_hessian_spectrum(model, dataloader, top_k=20, max_iter=50)
    
    print("\nTop 5 Eigenvalues:")
    print(eigenvalues[:5])
