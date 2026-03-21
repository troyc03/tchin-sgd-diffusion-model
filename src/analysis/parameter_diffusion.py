import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np

class ParameterDiffusionAnalyzer:
    """
    Simulates the Stochastic Differential Equation (SDE) dynamics using Euler-Maruyama integration to model the Wiener Process (Brownian Motion).
    """
    def __init__(self, model, learning_rate=0.01, noise_scale=0.001):
        self.model = model
        self.lr = learning_rate
        self.gamma = noise_scale
        self.trajectory = []

    def _get_flat_params(self):
        """Helper to extract and flatten all model parameters."""
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def _load_flat_params(self, flat_params):
        """Helper to reload flattened parameters back into the model layers."""
        pointer = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[pointer:pointer + numel].view_as(p))
            pointer += numel

    def compute_drift(self, batch, loss_fn):
        """
        Calculates the deterministic (Drift) component: -η * ∇L
        This represents the standard gradient descent force.
        """
        images, labels = batch
        self.model.zero_grad()
        output = self.model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        
        # Flattened gradient vector
        grads = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
        return -self.lr * grads

    def compute_diffusion(self, params_flat, noise_cov):
        """
        Calculates the stochastic (Diffusion) component: sqrt(2ηγ) * sqrt(B) * dW
        Shapes the Wiener process using the Noise Covariance matrix B.
        """
        # Standard Wiener Increment dW ~ N(0, I)
        dw = torch.randn_like(params_flat)
        
        # Scalar scaling factor for the noise
        diffusion_scale = np.sqrt(2 * self.lr * self.gamma)
        
        # Extract the diagonal of the Diffusion Tensor B for stable computation
        # Mathematically: sqrt(B) * dW
        noise_diag = torch.from_numpy(np.sqrt(np.diag(noise_cov))).to(params_flat.device)
        
        return diffusion_scale * noise_diag * dw

    def step(self, batch, loss_fn, noise_cov):
        """
        Performs a single Euler-Maruyama step of the SDE.

        """
        current_theta = self._get_flat_params()
        
        # SDE Components
        drift_term = self.compute_drift(batch, loss_fn)
        diffusion_term = self.compute_diffusion(current_theta, noise_cov)
        
        # Update
        new_theta = current_theta + drift_term + diffusion_term
        
        # Apply update to model and log results
        self._load_flat_params(new_theta)
        self.trajectory.append(new_theta.detach().cpu().numpy())
        
        return new_theta

def main():
    import torch.nn as nn
    
    # 1. Setup minimal environment for testing the module
    device = torch.device("cpu")
    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    loss_fn = nn.Cross_entropy = nn.MSELoss() # Placeholder for math test
    
    # 2. Initialize Analyzer
    # Higher gamma = more "Brownian" noise in the parameters
    analyzer = ParameterDiffusionAnalyzer(model, learning_rate=0.01, noise_scale=0.05)
    
    # 3. Dummy data and dummy Noise Covariance (Identity for testing)
    batch = (torch.randn(8, 10), torch.randn(8, 2))
    num_params = sum(p.numel() for p in model.parameters())
    noise_cov = np.eye(num_params) * 0.1 
    
    # 4. Run SDE Step
    print(f"Simulating Diffusion in {num_params}-dimensional parameter space...")
    updated_theta = analyzer.step(batch, loss_fn, noise_cov)
    
    print(f"Update successful. L2 Norm of parameters: {torch.norm(updated_theta).item():.4f}")

if __name__ == '__main__':
    main()