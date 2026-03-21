import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D # For older matplotlib compatibility


class PCAAnalysis:
    """
    Analyzes and visualizes the noise structure (B) of the SGD process
    via Principal Component Analysis and draws the 2D plane of best fit.
    """
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.transformed_data = None
        self.var_ratios = None
        self.mean_ = None

    def fit_transform(self, gradient_data):
        """Fits PCA to the noise data [Batch, Params] and returns the projection."""
        self.transformed_data = self.pca.fit_transform(gradient_data)
        self.var_ratios = self.pca.explained_variance_ratio_
        # We need the mean and components to reconstruct the plane in original space
        self.mean_ = self.pca.mean_
        self.components_ = self.pca.components_
        
        print(f"Top {self.n_components} PCA Variance Ratios: {self.var_ratios}")
        return self.transformed_data

    def plot_3d_with_plane(self):
        """Visualizes the diffusion geometry and draws the PC1-PC2 plane of best fit."""
        if self.transformed_data is None:
            raise ValueError("Run fit_transform first.")
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. Plot the 3D Scatter Points (Color mapping by depth/PC3)
        ax.scatter(
            self.transformed_data[:, 0], 
            self.transformed_data[:, 1], 
            self.transformed_data[:, 2], 
            alpha=0.7, c=self.transformed_data[:, 2], cmap='viridis', edgecolors='k', label='SGD Noise Cloud'
        )
        
        # 2. Draw the Plane of Best Fit (PC1-PC2 Plane)
        # Create a meshgrid covering the span of PC1 and PC2
        pc1_min, pc1_max = self.transformed_data[:, 0].min(), self.transformed_data[:, 1].max()
        pc2_min, pc2_max = self.transformed_data[:, 1].min(), self.transformed_data[:, 1].max()
        
        # Generate points covering this 2D plane
        # 10 points per axis for visibility
        x_surf, y_surf = np.meshgrid(np.linspace(pc1_min, pc1_max, 10), np.linspace(pc2_min, pc2_max, 10))
        
        # The equation of this plane in our reduced coordinate system is simply Z = 0
        # (Since we are plotting relative to the components)
        z_surf = np.zeros_like(x_surf) 
        
        # Plot the semi-transparent surface
        ax.plot_surface(x_surf, y_surf, z_surf, alpha=0.3, color='cyan', label='PC1-PC2 Optimal Projection Plane')
        
        ax.set_title("3D Diffusion Geometry with Best-Fit Plane Projection")
        ax.set_xlabel(f"PC1 ({self.var_ratios[0]:.2%})")
        ax.set_ylabel(f"PC2 ({self.var_ratios[1]:.2%})")
        ax.set_zlabel(f"PC3 ({self.var_ratios[2]:.2%})")
        
        # For legend support with surface/scatter
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Gradient Samples', markerfacecolor='g', markersize=10),
            Line2D([0], [0], color='cyan', lw=4, alpha=0.3, label='PC1-PC2 optimal plane')
        ]
        ax.legend(handles=legend_elements)
        
        plt.show()

def main():
    # Setup standard model and data
    device = torch.device("cpu")
    model = nn.Sequential(nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 26 * 26, 10)).to(device)
    
    from gradient_noise import GradientNoiseAnalyzer
    analyzer = GradientNoiseAnalyzer(model)

    # Batch data [Batch Size, Channels, H, W]
    batch_size = 64
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))

    # Capture per-sample gradients
    print(f"Capturing gradient noise cloud for batch size {batch_size}...")
    ps_grads_dict = analyzer.compute_per_sample_gradients((images, labels))
    
    # Flatten and Concatenate all parameters into a [64, Num_Params] matrix
    flattened_grads = []
    for name in ps_grads_dict:
        grad_tensor = ps_grads_dict[name]
        flattened_grads.append(grad_tensor.reshape(batch_size, -1))
    
    all_grads = torch.cat(flattened_grads, dim=1).detach().numpy()
    
    # Run PCA Analysis
    print(f"Projecting {all_grads.shape[1]} parameters into 3D subspace...")
    analysis = PCAAnalysis(n_components=3)
    analysis.fit_transform(all_grads)
    
    # Visualize 3D with best-fit plane
    analysis.plot_3d_with_plane()

if __name__ == '__main__':
    main()