import numpy as np
import torch

class OUProcess:
    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=1e-2):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def sample(self, x_prev):
        noise = torch.randn_like(x_prev) * np.sqrt(self.dt)
        x_next = x_prev + self.theta * (self.mu - x_prev) * self.dt + self.sigma * noise
        return x_next
    
    def simulate(self, x0, T):
        num_steps = int(T / self.dt)
        x = torch.zeros(num_steps + 1, *x0.shape)
        x[0] = x0
        for t in range(1, num_steps + 1):
            x[t] = self.sample(x[t-1])
        return x
    
    