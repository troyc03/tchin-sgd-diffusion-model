from analysis.gradient_noise import GradientNoiseAnalyzer
from models.cnn import SimpleCNN
import torch

def measure_noise(model, data_loader, loss_fn, num_iterations=100):
    pass