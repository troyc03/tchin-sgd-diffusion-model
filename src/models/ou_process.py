import numpy as np
import torch

class OUProcess:

    def __init__(self, theta=0.15, mu=0.0, sigma=0.2, dt=0.01):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

    def sample(self, x0, num_steps):
        """
        Sample from the Ornstein-Uhlenbeck process. 
        """
        x = np.zeros(num_steps)
        x[0] = x0
        for i in range(1, num_steps):
            dx = self.theta * (self.mu - x[i-1]) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
            x[i] = x[i-1] + dx
        return x
    
    def simulate(self, x0, num_steps):
        return self.sample(x0, num_steps)
    
    