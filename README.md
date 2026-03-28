# On the Diffusion Geometry of Stochastic Gradient Descent: A Spectral Analysis on Supervised Image Classification Algorithms

This project is centered on modeling SGD as an SDE and visualize its noise structure using Hessian and PDE-informed spectral analysis.

## Overview

Stochastic Gradient Descent (SGD) is one of the most widely-used optimization algorithms in machine learning. This project provides a rigorous mathematical framework for understanding SGD's behavior through the lens of **diffusion geometry** and **spectral analysis**.

The key insight is to model SGD as a continuous-time Stochastic Differential Equation (SDE), allowing us to:
- Analyze the **noise structure** of the optimization trajectory
- Examine the **Hessian spectrum** of the loss landscape along the optimization path
- Apply **PDE-informed techniques** to understand the long-term dynamics
- Characterize how parameter diffusion relates to generalization and implicit regularization

## Motivation

Traditional convergence analysis of SGD focuses on expected loss values, but this project goes deeper by analyzing the *geometry* of the trajectory itself. By studying:
- **Parameter covariance** structures during training
- **Gradient noise covariance** and its spectral properties
- **Hessian eigenvalue distributions** at different stages of training
- **Fokker-Planck equations** governing the probability distribution evolution

we gain new insights into why SGD works well in practice and how its implicit noise acts as a regularizer.

## Key Features

- **SDE Modeling**: Represent SGD as a continuous stochastic process (Langevin-BFGS dynamics)
- **Spectral Analysis**: Compute and visualize Hessian eigenvalues and their evolution
- **Noise Characterization**: Estimate gradient noise covariance and perform PCA analysis
- **Theoretical Framework**: Connect to Fokker-Planck equations and OU processes
- **Empirical Validation**: Run experiments on MNIST with reproducible notebooks

## Project Structure

```
src/
├── analysis/                    # Core analytical components
│   ├── covariance_estimation.py # Parameter and gradient covariance analysis
│   ├── gradient_noise.py         # Gradient noise characterization
│   ├── hessian_spectra.py        # Hessian eigenvalue computation and tracking
│   ├── parameter_diffusion.py    # Diffusion tensor analysis
│   └── pca_analysis.py           # Principal component analysis of dynamics
│
├── datasets/                     # Data loading utilities
│   └── data_loader.py            # MNIST and dataset utilities
│
├── models/                       # Mathematical models and simulators
│   ├── cnn.py                    # CNN architecture for MNIST
│   ├── sde_simulator.py          # SDE numerical integration
│   ├── langevin_dynamics.py      # Langevin dynamics implementation
│   ├── ou_process.py             # Ornstein-Uhlenbeck process
│   └── fokker_planck.py          # Fokker-Planck equation solver
│
├── experiments/                  # Reproducible experimental pipelines
│   ├── train_model.py            # Model training with logging
│   ├── measure_noise.py          # Compute gradient noise statistics
│   ├── measure_hessian.py        # Hessian spectrum measurement
│   ├── measure_diffusion.py      # Diffusion tensor estimation
│   ├── simulate_sde.py           # SDE simulation and comparison
│   └── run_full_pipeline         # End-to-end experimental pipeline
│
├── training/                     # Training utilities
│   ├── sgd_training.py           # SGD with trajectory logging
│   ├── trajectory_logger.py      # Logging parameter trajectories
│   └── evaluation.py             # Model evaluation metrics
│
└── notebooks/                    # Interactive analysis notebooks
    ├── dataset_exploration.ipynb
    ├── gradient_noise_analysis.ipynb
    ├── hessian_spectrum.ipynb
    ├── diffusion_geometry.ipynb
    ├── sde_vs_sgd.ipynb
    └── training_diagnostics.ipynb
```

## Installation

Clone the repository and set up dependencies:

```bash
git clone <repository-url>
cd tchin-sgd-diffusion-model
pip install -r requirements.txt
```

### Dependencies
- PyTorch
- NumPy, SciPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter

## Theoretical Background

### SGD as an SDE

Under appropriate scaling limits, SGD can be approximated by:

$$d\theta_t = -\nabla L(\theta_t) dt + \sqrt{2\gamma B(\theta_t)} dW_t$$

where:
- $L(\theta)$ is the loss function
- $B(\theta)$ is the noise covariance tensor
- $\gamma$ is a noise scale parameter
- $W_t$ is a standard Wiener process

### Fokker-Planck Dynamics

The probability density of the SDE evolves according to:

$$\frac{\partial \rho}{\partial t} = \nabla \cdot (\rho \nabla L) + \gamma \nabla \cdot (\nabla \cdot (B\rho))$$

### Spectral Analysis

We track the Hessian eigenvalue spectrum $\lambda_1(t) \geq \lambda_2(t) \geq ... \geq \lambda_d(t)$ throughout training to understand the curvature landscape's evolution.

## Results

The project demonstrates:
1. **Gradient noise is anisotropic**: Noise structure correlates with loss landscape curvature
2. **Hessian spectrum evolves sharply**: Eigenvalues show characteristic phases (acceleration, plateau, decay)
3. **Implicit bias through diffusion**: The noise-induced diffusion term provides implicit regularization similar to explicit penalties
4. **SDE approximation validity**: Continuous-time SDE models accurately capture SGD behavior over appropriate time scales

## Contributing

Contributions are welcome. Please ensure code follows the project structure and includes appropriate analysis/visualization components.

## License

See [LICENSE](https://github.com/troyc03/tchin-sgd-diffusion-model/blob/main/LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{tchin_sgd_diffusion,
  title={On the Diffusion Geometry of Stochastic Gradient Descent: A Spectral Analysis on Supervised Image Classification Algorithms},
  author={Tchin, [Author Name]},
  year={2024},
  howpublished={GitHub Repository}
}
```

## References

- [Mandt, Stephan, Matthew D. Hoffman, and David M. Blei. "Stochastic gradient descent as approximate bayesian inference." Journal of Machine Learning Research 18.134 (2017): 1-35.](https://www.jmlr.org/papers/volume18/17-214/17-214.pdf))
- [Yang, Ning, Chao Tang, and Yuhai Tu. "Stochastic gradient descent introduces an effective landscape-dependent regularization favoring flat solutions." Physical Review Letters 130.23 (2023): 237101.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.130.237101)
- [Yang, Ning, et al. "Transient learning dynamics drive escape from sharp valleys in Stochastic Gradient Descent." arXiv preprint arXiv:2601.10962 (2026).](https://arxiv.org/abs/2601.10962)
- [Wei, Stanley, Alex Damian, and Jason D. Lee. "Improved high-dimensional estimation with Langevin dynamics and stochastic weight averaging." arXiv preprint arXiv:2603.06028 (2026).](https://arxiv.org/pdf/2603.06028)
- [Granziol, Diego, and Khurshid Juarev. "Hessian Spectral Analysis at Foundation Model Scale." arXiv preprint arXiv:2602.00816 (2026).](https://arxiv.org/abs/2602.00816)
- [Granziol, Diego, and Khurshid Juarev. "Hessian Spectral Analysis at Foundation Model Scale." arXiv preprint arXiv:2602.00816 (2026).](https://arxiv.org/pdf/2602.05600)
