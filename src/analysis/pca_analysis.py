from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import SGDRegressor

class PCAAnalysis:
    def __init__(self, data, n_components=3):
        self.data = data
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit_transform(self):
        self.transformed_data = self.pca.fit_transform(self.data)
        return self.transformed_data

    def plot_3d(self):
        if self.n_components < 3:
            raise ValueError("n_components must be at least 3 for 3D plotting.")
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.transformed_data[:, 0], self.transformed_data[:, 1], self.transformed_data[:, 2], c='b', marker='o')
        ax.set_title('3D PCA Plot')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()

    def explained_variance(self):
        return self.pca.explained_variance_ratio_
    
    def analyze(self):
        self.fit_transform()
        self.plot_3d()
        return self.explained_variance()