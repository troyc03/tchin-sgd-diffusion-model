import numpy as np
import torch
import matplotlib.pyplot as plt

class SDESimulator:
    def __init__(self, drift_func, diffusion_func, dt=0.01):
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.dt = dt

    pass