import numpy as np
import torch
import matplotlib.pyplot as plt

class LangevinDynamics:

    def __init__(self, model, data_loader, loss_fn, gamma=0.1, sigma=0.01):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.gamma = gamma
        self.sigma = sigma

    def step(self):
        # Compute the gradient of the loss with respect to model parameters
        self.model.train()
        for inputs, targets in self.data_loader:
            inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            # Update parameters using Langevin dynamics
            with torch.no_grad():
                for param in self.model.parameters():
                    noise = torch.randn_like(param) * self.sigma
                    param.add_(-self.gamma * param.grad + noise)

    def simulate(self, num_steps):
        for _ in range(num_steps):
            self.step()
       


