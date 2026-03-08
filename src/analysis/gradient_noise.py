import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import torch

class GradientNoiseAnalyzer:
    def __init__(self, model, data_loader, loss_fn):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn

    def compute_gradient_noise(self):
        """
        Compute the variance of the gradients across mini-batches to analyze the noise in the optimization process.
        """
        gradient_variances = []
        for inputs, targets in self.data_loader:
            inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            # Compute the variance of the gradients for each parameter
            for param in self.model.parameters():
                if param.grad is not None:
                    gradient_variances.append(param.grad.var().item())
        return np.mean(gradient_variances)

    def plot_gradient_noise(self, noise_values):
        """
        Plot the gradient noise over training iterations to visualize how it evolves during optimization.
        """
        plt.plot(noise_values)
        plt.title('Gradient Noise Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Noise (Variance)')
        plt.show()
    
