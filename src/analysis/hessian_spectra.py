import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
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
# Hessian + Lanczos
# =========================
class HessianSpectrum:
    def __init__(self, model, dataloader, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.model.eval()

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.dim = sum(p.numel() for p in self.params)

    def _hvp_full(self, v):
        """
        Full-dataset averaged Hessian-vector product
        """
        v = v.to(self.device)
        hv_accum = torch.zeros_like(v)

        for inputs, targets in self.dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)

            grads = grad(loss, self.params, create_graph=True)
            flat_grad = torch.cat([g.reshape(-1) for g in grads])

            grad_v = torch.dot(flat_grad, v)
            hv = grad(grad_v, self.params, retain_graph=False)

            hv_flat = torch.cat([h.reshape(-1) for h in hv])
            hv_accum += hv_flat

        return hv_accum / len(self.dataloader)

    def lanczos(self, max_iter=100, tol=1e-6):
        """
        Lanczos tridiagonalization
        """
        alpha = []
        beta = []
        Vs = []

        v = torch.randn(self.dim, device=self.device)
        v /= torch.norm(v)
        Vs.append(v)

        prev_eval = None

        for j in tqdm(range(max_iter), desc="Lanczos"):
            hv = self._hvp_full(Vs[j])

            a = torch.dot(Vs[j], hv)
            alpha.append(a)

            w = hv - a * Vs[j]
            if j > 0:
                w -= beta[-1] * Vs[j - 1]

            b = torch.norm(w)

            if b < tol:
                print(f"Converged at iteration {j}")
                break

            beta.append(b)
            Vs.append(w / b)

            # Convergence check (top eigenvalue stability)
            if j > 5:
                if prev_eval is not None and abs(a.item() - prev_eval) < tol:
                    print(f"Early convergence at iteration {j}")
                    break
                prev_eval = a.item()

        k = len(alpha)

        # Build tridiagonal matrix
        T = torch.zeros((k, k), device=self.device)
        for i in range(k):
            T[i, i] = alpha[i]
            if i < k - 1:
                T[i, i + 1] = beta[i]
                T[i + 1, i] = beta[i]

        # Eigen decomposition
        evals, evecs_T = torch.linalg.eigh(T)

        # Sort (descending)
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs_T = evecs_T[:, idx]

        # Ritz vectors
        V = torch.stack(Vs[:k], dim=1)
        ritz_vecs = V @ evecs_T

        return evals.cpu().numpy(), ritz_vecs.cpu().numpy()


# =========================
# Training
# =========================
def train_model(model, dataloader, epochs=5, lr=0.1, device='cpu'):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

    model.eval()


# =========================
# Plotting
# =========================
def plot_spectrum(evals):
    evals = torch.tensor(evals)

    plt.figure()
    plt.plot(evals.numpy(), 'o-')
    plt.title("Hessian Eigenvalues (Raw)")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)

    plt.figure()
    plt.plot(torch.abs(evals).numpy(), 'o-')
    plt.yscale('log')
    plt.title("Log Magnitude Spectrum")
    plt.xlabel("Index")
    plt.ylabel("|Eigenvalue|")
    plt.grid(True)

    print("Max eigenvalue:", evals.max().item())
    print("Min eigenvalue:", evals.min().item())


# =========================
# Main
# =========================
if __name__ == "__main__":
    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dummy dataset (like MNIST)
    X = torch.randn(500, 784)
    y = torch.randint(0, 10, (500,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Model
    model = MyModel().to(device)

    # Train first (IMPORTANT)
    print("Training model...")
    train_model(model, dataloader, epochs=5, lr=0.1, device=device)

    # Compute Hessian spectrum
    print("\nComputing Hessian spectrum...")
    hessian = HessianSpectrum(model, dataloader, device=device)
    evals, evecs = hessian.lanczos(max_iter=50)

    # Plot
    plot_spectrum(evals)

    print("\nTop 5 eigenvalues:")
    print(evals[:5])