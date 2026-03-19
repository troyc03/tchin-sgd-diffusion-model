import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import torch
from gradient_noise import GradientNoiseAnalyzer

class PCAAnalysis:
    def __init__(self, data, n_components=3):
        # 'data' here should be the per-sample gradients 
        # Shape: [batch_size, num_parameters]
        self.data = data
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
    
    def perform_pca(self):
        # 1. Fit PCA to the noise data
        self.transformed_data = self.pca.fit_transform(self.data)
        
        # 2. Print how much 'noise' is captured in the top directions
        # This tells you the 'Diffusion Scale' along the principal axes
        var_ratio = self.pca.explained_variance_ratio_
        print(f"Top 3 PCA Variance Ratios: {var_ratio}")
        return self.transformed_data

    def plot_2d(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.transformed_data[:, 0], self.transformed_data[:, 1], alpha=0.5)
        plt.title("SGD Noise Cloud (First 2 Principal Components)")
        plt.xlabel(f"PC1 (Var: {self.pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"PC2 (Var: {self.pca.explained_variance_ratio_[1]:.2f})")
        plt.grid(True)
        plt.show()

    def plot_3d(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter the first 3 components
        ax.scatter(self.transformed_data[:, 0], 
                   self.transformed_data[:, 1], 
                   self.transformed_data[:, 2], 
                   alpha=0.6, c=self.transformed_data[:, 2], cmap='viridis')
        
        ax.set_title("3D SGD Noise Diffusion Geometry")
        ax.set_xlabel(f"PC1 ({self.pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"PC2 ({self.pca.explained_variance_ratio_[1]:.2f})")
        ax.set_zlabel(f"PC3 ({self.pca.explained_variance_ratio_[2]:.2f})")
        plt.show()

if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # The OpenMP fix
    
    import torch.nn as nn
    import torch.nn.functional as F
    
    # 1. Setup a dummy model and data to test the pipeline
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.fc = nn.Linear(1440, 10)
        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    
    # Create a synthetic batch (64 images of 28x28)
    images = torch.randn(64, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (64,)).to(device)
    batch = (images, labels)
    analyzer = GradientNoiseAnalyzer(model, None, F.cross_entropy)

    # 3. Get the raw per-sample gradients
    print("Extracting per-sample gradients...")
    per_sample_grads = analyzer.get_per_sample_grads(batch[0], batch[1])

    # 4. Flatten and concatenate into one matrix [Batch, Params]
    flat_grad_list = []
    for name, grads in per_sample_grads.items():
        flat_grad_list.append(grads.view(grads.shape[0], -1))
    
    all_grads = torch.cat(flat_grad_list, dim=1).detach().cpu().numpy()

    # 5. Run PCA
    print(f"Running PCA on gradient matrix of shape {all_grads.shape}...")
    analysis = PCAAnalysis(data=all_grads, n_components=3)
    analysis.perform_pca()
    
    # 6. Visualize
    analysis.plot_2d()
    analysis.plot_3d()