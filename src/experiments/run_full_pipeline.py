# Run full pipeline here.
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from analysis.covariance_estimation import CovarianceEstimator
from analysis.hessian_spectra import Hessian
from analysis.gradient_noise import GradientNoiseAnalyzer
from models.cnn import SimpleCNN
import torch

def pipeline(data, loss_fn, data_loader, model):

    pipeline = Pipeline([
        ('covariance_estimation', CovarianceEstimator(data)),
        ('hessian_spectra', Hessian(model, loss_fn, data_loader)),
        ('gradient_noise', GradientNoiseAnalyzer(model, data_loader, loss_fn))
    ])

    return pipeline

if __name__ == "__main__":
    # Example usage
    model = SimpleCNN()
    loss_fn = torch.nn.CrossEntropyLoss()
    data_loader = ...  # Define your data loader here
    data = ...  # Load your dataset here
    full_pipeline = pipeline(data, loss_fn, data_loader, model) 
    full_pipeline.fit(data)

