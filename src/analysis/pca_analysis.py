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
       pass