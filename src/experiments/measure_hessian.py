import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.cnn import SimpleCNN
from datasets.data_loader import load_mnist
from training.sgd_training import train_with_logging
from analysis.hessian_spectra import compute_hessian_spectrum

def measure_hessian():
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist(batch_size=128)
    
    print("Initializing and training model...")
    model, _ = train_with_logging(
        train_loader, 
        test_loader,
        epochs=10,  # Fewer epochs for demo
        learning_rate=0.01
    )
    
    print("Computing Hessian spectrum...")
    eigenvalues, eigenvectors = compute_hessian_spectrum(
        model, 
        test_loader,
        top_k=50
    )
    
    # Save results
    np.save('hessian_eigenvalues.npy', eigenvalues)
    print(f"Top eigenvalue: {eigenvalues[0]:.2e}, Trace approx: {np.sum(eigenvalues):.2e}")
    
    # Additional plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(eigenvalues)
    plt.xlabel('Eigenvalue index')
    plt.ylabel('Eigenvalue')
    plt.title('Hessian Top Eigenvalues')
    
    plt.subplot(1, 2, 2)
    plt.loglog(range(1, len(eigenvalues)+1), eigenvalues, 'o-')
    plt.xlabel('Rank')
    plt.ylabel('Eigenvalue')
    plt.title('Hessian Spectrum (log-log)')
    plt.savefig('hessian_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return eigenvalues, eigenvectors

if __name__ == '__main__':
    measure_hessian()

