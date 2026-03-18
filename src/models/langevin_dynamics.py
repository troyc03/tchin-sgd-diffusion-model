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

    pass