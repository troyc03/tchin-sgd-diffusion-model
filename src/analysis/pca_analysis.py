from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import SGDRegressor

class PCAAnalysis:
    def __init__(self, data, n_components=3):
        self.data = data
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
    
    def perform_pca(self):
        """
        Perform Principal Component Analysis.
        """
        self.pca_result = self.pca.fit_transform(self.data)
        return self.pca_result
    
    def plot_pca(self):
        """
        Plot the PCA results in 3D space.
        """
        if self.n_components < 3:
            raise ValueError("n_components must be at least 3 for 3D plotting.")
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pca_result[:, 0], self.pca_result[:, 1], self.pca_result[:, 2])
        ax.set_title('PCA of Model Parameters')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()  