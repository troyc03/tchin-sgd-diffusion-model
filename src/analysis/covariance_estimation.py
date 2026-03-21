import numpy as np
import matplotlib.pyplot as plt
import torch

class CovarianceEstimator:
    def __init__(self, model, train_loader, shrinkage=0.1):
        self.shrinkage = shrinkage
        self.model = model
        self.train_loader = train_loader
    
    def estimate_noise_covariance(self, num_samples=100):
        """
        Computes the empirical covariance of gradients across batches.
        B(theta) = Var(gradients)
        """
        grad_samples = []
        
        # Collect gradient samples
        self.model.train()
        for i, (images, labels) in enumerate(self.train_loader):
            if i >= num_samples: break
            
            self.model.zero_grad()
            output = self.model(images)
            loss = torch.nn.functional.cross_entropy(output, labels)
            loss.backward()
            
            # Flatten gradients into a single vector
            grads = torch.cat([p.grad.view(-1) for p in self.model.parameters()])
            grad_samples.append(grads.detach().cpu().numpy())
            
        # Stack samples: [num_samples, num_params]
        grad_matrix = np.array(grad_samples)
        
        # Empirical Covariance
        cov = np.cov(grad_matrix, rowvar=False)
        
        # Shrinkage to handle high-dimensionality (Ledoit-Wolf approach)
        identity = np.eye(cov.shape[0])
        shrunk_cov = (1 - self.shrinkage) * cov + self.shrinkage * np.mean(np.diag(cov)) * identity
        
        return shrunk_cov

def main(model, train_loader):
    # Initialize the estimator with the model and data
    estimator = CovarianceEstimator(model, train_loader)
    
    # Calculate the Diffusion Tensor B(theta)
    print("Estimating Noise Covariance (Diffusion Tensor B)...")
    noise_cov = estimator.estimate_noise_covariance(num_samples=10) # Start small to test
    
    print(f"Covariance Matrix Shape: {noise_cov.shape}")
    return noise_cov

if __name__ == '__main__':
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import torch.nn as nn

    # 1. Define a tiny model for the math test
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(784, 10) # 28x28 MNIST pixels to 10 classes
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    # 2. Setup Data
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyModel().to(device)

    # 4. RUN (Passing the required arguments)
    main(model, train_loader)