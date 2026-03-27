import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Environment fix for OpenMP collisions
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- 1. Langevin Dynamics Engine ---
class LangevinDynamics:
    def __init__(self, model, data_loader, loss_fn, gamma=0.1, sigma=0.01, device="cpu"):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.sigma = sigma
        self.device = device
        self.data_iter = iter(self.data_loader)

    def get_parameters(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def set_parameters(self, theta):
        pointer = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(theta[pointer:pointer + numel].view_as(p))
            pointer += numel

    def compute_gradient(self):
        try:
            x, y = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            x, y = next(self.data_iter)

        x, y = x.to(self.device), y.to(self.device)
        self.model.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(output, y)
        loss.backward()

        return torch.cat([p.grad.view(-1) for p in self.model.parameters()]), loss.item()

    def step(self, dt=0.1):
        theta = self.get_parameters()
        grad, loss_val = self.compute_gradient()
        
        # Wiener Process Increment: sqrt(2 * gamma * sigma^2 * dt) * dW
        noise = torch.randn_like(theta)
        diffusion_scale = np.sqrt(2 * self.gamma * self.sigma**2 * dt)
        
        # SDE Update (Euler-Maruyama)
        theta_new = theta - self.gamma * grad * dt + diffusion_scale * noise
        
        self.set_parameters(theta_new)
        return theta_new, grad, loss_val

    def simulate(self, num_steps=1000, dt=0.1, log_every=10):
        trajectory, grad_norms, losses = [], [], []
        
        print(f"Starting SDE Simulation for {num_steps} steps...")
        for step in range(num_steps):
            theta, grad, loss_val = self.step(dt)
            if step % log_every == 0:
                trajectory.append(theta.detach().cpu().numpy())
                grad_norms.append(torch.norm(grad).item())
                losses.append(loss_val)
        
        return {
            "trajectory": np.array(trajectory),
            "grad_norms": grad_norms,
            "losses": losses,
            "log_every": log_every
        }

# --- 2. Analysis & Visualization ---
def plot_simulation_results(results):
    traj = results["trajectory"]
    steps = np.arange(len(results["losses"])) * results["log_every"]
    
    fig = plt.figure(figsize=(18, 5))

    # Plot A: Loss Convergence (Energy Minimization)
    ax1 = fig.add_subplot(131)
    ax1.plot(steps, results["losses"], color='forestgreen', lw=1.5)
    ax1.set_title("Loss Evolution (Potential Energy)")
    ax1.set_xlabel("SDE Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Plot B: Mean Squared Displacement (Diffusion Rate)
    ax2 = fig.add_subplot(132)
    msd = np.linalg.norm(traj - traj[0], axis=1)**2
    ax2.plot(steps, msd, color='royalblue', lw=1.5)
    ax2.set_title("Mean Squared Displacement (MSD)")
    ax2.set_xlabel("SDE Step")
    ax2.set_ylabel(r"$\mathbb{E}[\|\theta_t - \theta_0\|^2]$")
    ax2.grid(True, alpha=0.3)

    # Plot C: PCA Projection (The "Random Walk" Path)
    ax3 = fig.add_subplot(133)
    pca = PCA(n_components=2)
    path_2d = pca.fit_transform(traj)
    
    ax3.plot(path_2d[:, 0], path_2d[:, 1], color='black', alpha=0.2, ls='--')
    sc = ax3.scatter(path_2d[:, 0], path_2d[:, 1], c=steps, cmap='magma', s=15)
    ax3.set_title("PCA Trajectory (Parameter Space)")
    ax3.set_xlabel("Principal Component 1")
    ax3.set_ylabel("Principal Component 2")
    plt.colorbar(sc, label='Simulation Step')

    plt.tight_layout()
    plt.show()

def plot_simulation_with_landscape(results, model, loader, loss_fn):
    traj = results["trajectory"]
    steps = np.arange(len(results["losses"])) * results["log_every"]
    
    # 1. Fit PCA to get the 2D subspace
    pca = PCA(n_components=2)
    path_2d = pca.fit_transform(traj)
    components = pca.components_
    mean_param = pca.mean_
    
    # 2. Create a grid over the PCA space to sample the loss
    x_min, x_max = path_2d[:, 0].min() - 1, path_2d[:, 0].max() + 1
    y_min, y_max = path_2d[:, 1].min() - 1, path_2d[:, 1].max() + 1
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    zz = np.zeros((nx, ny))

    # 3. Sample the loss landscape
    print("Sampling Loss Landscape (Hessian Curvature)...")
    model.eval()
    images, labels = next(iter(loader)) # Use a representative batch
    
    with torch.no_grad():
        for i in range(nx):
            for j in range(ny):
                # Reconstruct high-dim parameters: theta = mean + c1*pc1 + c2*pc2
                point_2d = np.array([xx[i, j], yy[i, j]])
                point_nd = mean_param + point_2d @ components
                
                # Load into model
                pointer = 0
                for p in model.parameters():
                    numel = p.numel()
                    p.data.copy_(torch.from_numpy(point_nd[pointer:pointer+numel]).view_as(p))
                    pointer += numel
                
                # Compute loss
                zz[i, j] = loss_fn(model(images), labels).item()

    # 4. Final Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the "Valleys" and "Peaks"
    contour = ax.contourf(xx, yy, zz, levels=20, cmap='RdGy', alpha=0.5)
    plt.colorbar(contour, label='Loss (Potential Energy)')
    
    # Draw the SDE Trajectory (The Random Walk)
    ax.plot(path_2d[:, 0], path_2d[:, 1], color='blue', lw=1, alpha=0.6, label='SDE Path')
    ax.scatter(path_2d[:, 0], path_2d[:, 1], c=steps, cmap='plasma', s=20, edgecolors='white', zorder=5)
    
    ax.set_title("Langevin Diffusion on the Projected Loss Landscape")
    ax.set_xlabel("Principal Component 1 (Flatness)")
    ax.set_ylabel("Principal Component 2 (Curvature)")
    plt.legend()
    plt.show()

# --- 3. Main Execution ---
def main():
    # Synthetic Data for Simulation (Binary Classification)
    x = torch.randn(500, 20)
    y = (torch.sum(x, dim=1) > 0).long()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Simple Model: 20 -> 10 -> 2
    model = nn.Sequential(
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 2)
    )

    # Initialize SDE with "Moderate Noise"
    # gamma = friction/learning rate, sigma = noise volatility
    langevin = LangevinDynamics(
        model, loader, nn.CrossEntropyLoss(), 
        gamma=0.2, sigma=0.1
    )

    # Run Simulation
    results = langevin.simulate(num_steps=2000, dt=0.05, log_every=10)

    # Visualize
    plot_simulation_results(results)
    plot_simulation_with_landscape(results, model, loader, nn.CrossEntropyLoss())

if __name__ == "__main__":
    main()