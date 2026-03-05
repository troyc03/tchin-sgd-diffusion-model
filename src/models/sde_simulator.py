import numpy as np
import torch
import matplotlib.pyplot as plt

class SDESimulator:
    def __init__(self, drift_func, diffusion_func, dt=0.01):
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.dt = dt

    def simulate(self, x0, T):
        num_steps = int(T / self.dt)
        x = torch.zeros(num_steps + 1, *x0.shape)
        x[0] = x0
        for t in range(1, num_steps + 1):
            drift = self.drift_func(x[t-1])
            diffusion = self.diffusion_func(x[t-1])
            noise = torch.randn_like(x[t-1]) * np.sqrt(self.dt)
            x[t] = x[t-1] + drift * self.dt + diffusion * noise
        return x
    
# Example usage
if __name__ == "__main__":
    # Define drift and diffusion functions
    def drift(x):
        return -0.5 * x

    def diffusion(x):
        return 0.1 * torch.ones_like(x)

    # Create simulator
    simulator = SDESimulator(drift, diffusion, dt=0.01)

    # Simulate SDE
    x0 = torch.tensor([1.0])  # Initial condition
    T = 10.0  # Total time
    trajectory = simulator.simulate(x0, T)

    # Plot trajectory
    plt.plot(np.arange(trajectory.shape[0]) * simulator.dt, trajectory.numpy())
    plt.xlabel('Time')
    plt.ylabel('X(t)')
    plt.title('SDE Simulation')
    plt.grid()
    plt.show()