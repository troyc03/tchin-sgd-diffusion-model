import numpy as np
import matplotlib.pyplot as plt

class FokkerPlanckSolver:

    def __init__(self, rho, drift, diffusion, x_grid, dt=0.01):
        """
        rho: initial probability density function
        drift: function representing the drift term in the Fokker-Planck equation
        diffusion: function representing the diffusion term in the Fokker-Planck equation
        dt: time step for numerical integration
        """
        self.rho = rho
        self.drift = drift
        self.diffusion = diffusion
        self.dx = x_grid[1]-x_grid[0]
        self.dt = dt

    pass