from analysis.gradient_noise import GradientNoiseAnalyzer
from models.cnn import SimpleCNN
import torch

def measure_noise(model, data_loader, loss_fn, num_iterations=100):
    """
    Measure the noise in the gradients during training by computing the variance of the gradients across mini-batches.
    """
    noise_analyzer = GradientNoiseAnalyzer(model, data_loader, loss_fn)
    noise_values = []
    for _ in range(num_iterations):
        noise = noise_analyzer.compute_gradient_noise()
        noise_values.append(noise)
    noise_analyzer.plot_gradient_noise(noise_values)
    