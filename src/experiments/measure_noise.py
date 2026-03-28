import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

class GradientNoiseAnalyzer:
    def __init__(self, model, loss_fn=F.cross_entropy):
        self.model = model
        self.loss_fn = loss_fn

    def compute_per_sample_gradients(self, batch):
        inputs, targets = batch
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())

        def compute_loss(params, buffers, x, y):
            # Use unsqueeze(0) to simulate a batch of size 1 for functional_call
            outputs = functional_call(self.model, (params, buffers), x.unsqueeze(0))
            return self.loss_fn(outputs, y.unsqueeze(0))

        # Vectorize the gradient computation over the batch dimension (dim 0)
        ft_compute_grad = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))
        return ft_compute_grad(params, buffers, inputs, targets)

    def get_noise_statistics(self, batch):
        """Helper to get stats for a single batch."""
        ps_grads = self.compute_per_sample_gradients(batch)
        stats = {}
        for name, g in ps_grads.items():
            mean_grad = g.mean(dim=0)
            # Variance = E[g^2] - (E[g])^2
            variance = (g**2).mean(dim=0) - mean_grad**2
            stats[name] = {
                "drift": mean_grad,
                "diffusion_diag": variance
            }
        return stats

def measure_noise(model, data_loader, loss_fn, num_iterations=10):
    # Pass only the model and loss_fn to the analyzer
    noise_analyzer = GradientNoiseAnalyzer(model, loss_fn)
    noise_measurements = []
    
    # Iterate through the data_loader
    for i, batch in enumerate(data_loader):
        if i >= num_iterations:
            break
        # Get statistics for the current batch
        stats = noise_analyzer.get_noise_statistics(batch)
        noise_measurements.append(stats)

    return noise_measurements

# Example model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.fc = nn.Linear(8 * 30 * 30, 10)
    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.fc(x.view(x.size(0), -1))

if __name__ == '__main__':
    model = SimpleCNN()
    
    # Create dummy data: 10 batches of 64 images
    dummy_loader = [
        (torch.randn(64, 3, 32, 32), torch.randint(0, 10, (64,))) 
        for _ in range(10)
    ]

    loss_fn = nn.CrossEntropyLoss()

    # Measure noise over 3 iterations
    noise_results = measure_noise(model, dummy_loader, loss_fn, num_iterations=3)
    
    print(f"Measured noise for {len(noise_results)} batches.")
    # Example: print variance of the first layer's weights from the first batch
    first_layer_var = noise_results[0]['conv.weight']['diffusion_diag']
    print(f"Mean variance in first layer: {first_layer_var.mean().item():.8f}")
