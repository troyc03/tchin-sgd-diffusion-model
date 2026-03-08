import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

class CovarianceEstimator:
    
    def __init__(self, data, shrinkage=0.1):
        self.data = data
        self.shrinkage = shrinkage
    
    def compute_covariance(self):
        """
        Compute the covariance matrix of the data with optional shrinkage for regularization.
        """
        n_samples, n_features = self.data.shape
        empirical_cov = np.cov(self.data, rowvar=False)
        shrinkage_target = np.eye(n_features) * np.trace(empirical_cov) / n_features
        covariance_matrix = (1 - self.shrinkage) * empirical_cov + self.shrinkage * shrinkage_target
        return covariance_matrix
    
    def plot_covariance(self, covariance_matrix):
        """
        Plot the covariance matrix as a heatmap to visualize the relationships between features.
        """
        sns.heatmap(covariance_matrix, cmap='viridis', cbar=True)
        plt.title('Covariance Matrix Heatmap')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Index')
        plt.show()
    