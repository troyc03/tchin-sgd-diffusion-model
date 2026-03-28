import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

class OUProcess:
    """
    Ornstein-Uhlenbeck Process Simulator for SGD trajectory modeling.
    SDE: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    """
    def __init__(self, theta=2.0, mu=0.0, sigma=1.0, paths=10, points=1000, T=1.0, X0=None):
        self.theta = theta  # Reversion speed
        self.mu = mu  # Long-term mean
        self.sigma = sigma  # Volatility
        self.paths = paths
        self.points = points
        self.T = T
        self.X0 = X0 if X0 is not None else mu
        self.dt = T / (points - 1)
        self.t_axis = np.linspace(0, T, points)

    def drift(self, x, t):
        """Drift term: theta*(mu - x)"""
        return self.theta * (self.mu - x)

    def diffusion(self, x, t):
        """Constant diffusion sigma."""
        return self.sigma

    def simulate(self):
        """Simulate paths using Euler-Maruyama (numpy)."""
        X = np.full((self.paths, self.points), self.X0)
        dW = np.random.normal(0, np.sqrt(self.dt), (self.paths, self.points-1))
        for i in range(self.points - 1):
            X[:, i+1] = (X[:, i] + 
                         self.drift(X[:, i], self.t_axis[i]) * self.dt + 
                         self.diffusion(X[:, i], self.t_axis[i]) * dW[:, i])
        return X

    def plot_paths(self, X=None):
        if X is None:
            X = self.simulate()
        plt.figure(figsize=(12, 6))
        plt.plot(self.t_axis, X.T, alpha=0.4, color='blue')
        plt.axhline(self.mu, color='red', ls='--', lw=2, label='mu = {:.2f}'.format(self.mu))
        plt.title('OU Process: theta={:.2f}, mu={:.2f}, sigma={:.2f}'.format(self.theta, self.mu, self.sigma))
        plt.xlabel('Time t')
        plt.ylabel('X_t')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def fit_to_trajectory(self, trajectory, t=None):
        """
        Fit params to observed trajectory (e.g., SGD norm).
        Returns [theta, mu, sigma]
        """
        n = len(trajectory)
        if t is None:
            t = np.linspace(0, 1, n)
        dt_arr = np.diff(t)
        dX = np.diff(trajectory)
        X_mid = 0.5 * (trajectory[:-1] + trajectory[1:])

        def neg_log_likelihood(params):
            theta, mu, sigma = params
            drift_pred = theta * (mu - X_mid)
            residuals = (dX - drift_pred * dt_arr) / (sigma * np.sqrt(dt_arr))
            return 0.5 * np.sum(residuals**2 + np.log(2 * np.pi * sigma**2 * dt_arr))

        init_guess = [1.0, trajectory.mean(), np.std(dX) / np.sqrt(np.mean(dt_arr))]
        bounds = [(0.01, None), (None, None), (0.01, None)]
        res = minimize(neg_log_likelihood, init_guess, bounds=bounds)
        self.theta, self.mu, self.sigma = res.x
        print('Fitted: theta={:.3f}, mu={:.3f}, sigma={:.3f}'.format(self.theta, self.mu, self.sigma))
        return res.x

    def simulate_torch(self, device='cpu'):
        """Torch/GPU version for ML integration."""
        X0_t = torch.tensor(self.X0, device=device).repeat(self.paths)
        dt_t = torch.tensor(self.dt, device=device)
        dW = torch.randn(self.paths, self.points-1, device=device) * torch.sqrt(dt_t)
        X = torch.zeros(self.paths, self.points, device=device)
        X[:, 0] = X0_t
        for i in range(self.points - 1):
            drift_t = self.theta * (self.mu - X[:, i])
            X[:, i+1] = X[:, i] + drift_t * dt_t + self.sigma * dW[:, i]
        return X.cpu().numpy()

# Test / Example
if __name__ == '__main__':
    # 1. Basic simulation
    ou = OUProcess(theta=2.0, mu=0.0, sigma=0.5, paths=5)
    paths = ou.simulate()
    ou.plot_paths(paths)

    # 2. Fit to synthetic SGD-like trajectory
    t_synth = np.linspace(0, 1, 500)
    synth_traj = OUProcess(theta=1.5, mu=1.0, sigma=0.3, paths=1, points=len(t_synth)).simulate()[0]  # Single path
    ou_fit = OUProcess()
    ou_fit.fit_to_trajectory(synth_traj, t_synth)
    ou_fit.plot_paths()

    print('OUProcess expanded and tested successfully!')

