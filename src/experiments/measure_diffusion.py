from analysis.hessian_spectra import Hessian
from models.cnn import SimpleCNN
import torch

def measure_diffusion(model, data_loader, num_iterations=100):
    """
    Measure the diffusion properties of the model's parameter space by analyzing the Hessian matrix.
    """
    hessian_analyzer = Hessian(model, loss_fn, data_loader)
    hessian_matrix = hessian_analyzer.compute_hessian()
    eigenvalues = hessian_analyzer.compute_spectra(hessian_matrix)
    hessian_analyzer.plot_spectra(eigenvalues)

if __name__ == "__main__":
    # Example usage
    model = SimpleCNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    data_loader = ...  # Define your data loader here
    measure_diffusion(model, data_loader)
