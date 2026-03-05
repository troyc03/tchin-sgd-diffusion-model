import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class CovarianceEstimator:
    def __init__(self, data):
        self.data = data

    def estimate_covariance(self):
        # Estimate the covariance matrix from the data
        self.cov_matrix = np.cov(self.data, rowvar=False)
        return self.cov_matrix

    def plot_covariance(self):
        if not hasattr(self, 'cov_matrix'):
            raise ValueError("Covariance matrix not estimated yet. Call estimate_covariance() first.")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.cov_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Estimated Covariance Matrix')
        plt.show()