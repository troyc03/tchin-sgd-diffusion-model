from analysis.hessian_spectra import Hessian
from models.cnn import SimpleCNN
import torch

def measure_hessian(model, data_loader, loss_fn, num_iterations=100):
    """
    Measure the Hessian matrix of the loss function with respect to model parameters to analyze the curvature of the loss landscape.
    """
    hessian_analyzer = Hessian(model, loss_fn, data_loader)
    hessian_matrix = hessian_analyzer.compute_hessian()
    eigenvalues = hessian_analyzer.compute_spectra(hessian_matrix)
    hessian_analyzer.plot_spectra(eigenvalues)
