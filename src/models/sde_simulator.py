import numpy as np
import torch
import matplotlib.pyplot as plt

class SDESimulator:
    def __init__(self, drift_func, diffusion_func, dt=0.01):
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.dt = dt

    def euler_maruyama(self, x0, num_steps):
        """
        Simulate the SDE using the Euler-Maruyama method.
        x0: initial condition
        """
        x = np.zeros(num_steps)
        x[0] = x0
        for i in range(1, num_steps):
            drift = self.drift_func(x[i-1])
            diffusion = self.diffusion_func(x[i-1])
            x[i] = x[i-1] + drift * self.dt + diffusion * np.sqrt(self.dt) * np.random.normal()
        return x
    
    def simulate(self, x0, num_steps):
        return self.euler_maruyama(x0, num_steps)

    def plot_trajectory(self, x0, num_steps):
        trajectory = self.simulate(x0, num_steps)
        plt.plot(trajectory)
        plt.title('SDE Trajectory')
        plt.xlabel('Time Steps')
        plt.ylabel('State')
        plt.show()

    
    
    