import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.func import vmap, grad, functional_call
import torch.nn as nn
import torch.nn.functional as F

class GradientNoiseAnalyzer:
    def __init__(self, model, loss_fn=F.cross_entropy):
        self.model = model
        self.loss_fn = loss_fn

    def compute_per_sample_gradients(self, batch):
        """
        Uses torch.func (vmap + grad) to compute gradients for each 
        individual sample in a batch simultaneously.
        This is the raw data for the Diffusion Tensor B.
        """
        inputs, targets = batch
        
        # 1. Separate parameters from the model structure
        params = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())

        # 2. Define a functional loss for a single sample
        def compute_loss(params, buffers, x, y):
            # functional_call makes the model act like a pure function
            outputs = functional_call(self.model, (params, buffers), x.unsqueeze(0))
            return self.loss_fn(outputs, y.unsqueeze(0))

        # 3. Use vmap to vectorize the 'grad' function over the batch dimension
        # grad(compute_loss) gets grad w.r.t params
        # in_dims=(None, None, 0, 0) means don't map over params/buffers, only x and y
        ft_compute_grad = vmap(grad(compute_loss), in_dims=(None, None, 0, 0))
        
        # 4. Execute: returns a dict where each value has shape [batch_size, ...]
        per_sample_grads = ft_compute_grad(params, buffers, inputs, targets)
        return per_sample_grads

    def get_noise_statistics(self, per_sample_grads):
        """
        Calculates the mean gradient (Drift) and the variance (Diffusion Scale).
        """
        stats = {}
        for name, g in per_sample_grads.items():
            mean_grad = g.mean(dim=0)
            # Variance per parameter: E[g^2] - (E[g])^2
            variance = (g**2).mean(dim=0) - mean_grad**2
            stats[name] = {
                "drift": mean_grad,
                "diffusion_diag": variance
            }
        return stats

def main():
    # Example setup for MNIST
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 10)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    model = SimpleNet()
    analyzer = GradientNoiseAnalyzer(model)

    # Dummy Batch (Batch Size 16)
    x = torch.randn(16, 1, 28, 28)
    y = torch.randint(0, 10, (16,))
    
    print("Computing Per-Sample Gradients via vmap...")
    ps_grads = analyzer.compute_per_sample_gradients((x, y))
    
    # Check shape of the weight gradients
    weight_grad_shape = ps_grads['fc.weight'].shape
    print(f"Gradient shape for fc.weight: {weight_grad_shape}") 
    # Output should be [16, 10, 784] -> [Batch, Out_Features, In_Features]

if __name__ == '__main__':
    main()