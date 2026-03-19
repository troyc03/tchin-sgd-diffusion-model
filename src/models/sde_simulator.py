import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class SDESimulator:
    def __init__(self, drift_func, diffusion_func, paths, points, T=1.0, X0=0.0):
        """
        Initializes the SDE simulator (dX = drift*dt + diffusion*dW).
        """
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.paths = paths
        self.points = points
        self.T = T
        self.X0 = X0
        self.dt = T / (points - 1)
        self.t_axis = np.linspace(0, T, points)

    def simulate(self):
        """Simulates paths using Euler-Maruyama method."""
        X = np.zeros((self.paths, self.points))
        X[:, 0] = self.X0
        
        # Standard normal random variables for Weiner process
        dW = np.random.normal(0, np.sqrt(self.dt), (self.paths, self.points - 1))
        
        for i in range(self.points - 1):
            X[:, i+1] = X[:, i] + \
                        self.drift_func(X[:, i], self.t_axis[i]) * self.dt + \
                        self.diffusion_func(X[:, i], self.t_axis[i]) * dW[:, i]
        return X

    def plot_paths(self, X):
        """Plots simulation paths."""
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        plt.plot(self.t_axis, X.T, alpha=0.5)
        plt.title(f"SDE Simulation: {self.paths} Paths")
        plt.xlabel("Time (t)")
        plt.ylabel("X(t)")
        plt.show()

# --- Test Case ---
if __name__ == '__main__':
    # Example: Geometric Brownian Motion (drift_func=mu*x, diff_func=sigma*x)
    mu, sigma = 0.5, 0.2
    
    # Define SDE functions
    drift = lambda x, t: mu * x
    diffusion = lambda x, t: sigma * x
    
    # Run simulation
    sim = SDESimulator(drift, diffusion, paths=10, points=1000, T=1.0, X0=1.0)
    X = sim.simulate()
    
    # Visualize
    sim.plot_paths(X)
