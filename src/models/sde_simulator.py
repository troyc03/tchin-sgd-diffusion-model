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
    
    