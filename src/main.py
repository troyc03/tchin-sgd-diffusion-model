# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 14:01:06 2026

@author: Troy
"""

# Main pipeline for running experiments

from training.sgd_training import train_with_logging
from datasets.data_loader import load_mnist
from analysis.hessian_spectra import compute_hessian_spectrum
from analysis.gradient_noise import estimate_noise_covariance

train_loader, test_loader = load_mnist(batch_size=128)
model, trajectory = train_with_logging(
    train_loader, 
    test_loader,
    epochs=20,
    learning_rate=0.01
)

eigenvalues, eigenvectors = compute_hessian_spectrum(
    model, 
    test_loader,
    top_k=50  # Top 50 eigenvalues
)

noise_cov = estimate_noise_covariance(
    model,
    train_loader,
    num_samples=100
)

