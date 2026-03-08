import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

class Hessian:
    def __init__(self, model, loss_fn, data_loader):
        self.model = model 
        self.loss_fn = loss_fn
        self.data_loader = data_loader
    
    def compute_hessian(self):
        """
        Symbolically compute the Hessian matrix of the loss function with respect to model parameters.
        """
        # Get model parameters
        params = list(self.model.parameters())
        # Create symbolic variables for parameters
        param_symbols = [sp.symbols(f'p{i}') for i in range(len(params))]
        # Compute symbolic loss
        symbolic_loss = 0
        for inputs, targets in self.data_loader:
            inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            outputs = self.model(inputs)
            symbolic_loss += self.loss_fn(outputs, targets)
        # Compute Hessian matrix as an array of second derivatives
        hessian = sp.hessian(symbolic_loss, param_symbols)
        return hessian

    def compute_spectra(self, hessian):
        """
        Compute the eigenvalues of the Hessian matrix to analyze the curvature of the loss landscape.
        """
        eigenvalues = np.linalg.eigvals(hessian)
        eigenvectors = np.linalg.eig(hessian)[1]
        return eigenvalues, eigenvectors

    def plot_spectra(self, eigenvalues):
        """
        Plot the distribution of eigenvalues to visualize the curvature of the loss landscape.
        """
        sns.heatmap(eigenvalues.reshape(1, -1), cmap='viridis', cbar=True)
        plt.figure(figsize=(10, 2))
        plt.title('Hessian Eigenvalue Spectrum')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Magnitude')
        plt.show()

    