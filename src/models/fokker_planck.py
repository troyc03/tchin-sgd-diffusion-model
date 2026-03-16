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

    def solve_step(self, p_n, drift, diffusion):
        """
        Solve Fokker-Planck equation using Crank-Nicolson method for one time step.
        """
        n = len(p_n) # number of spatial points
        p_next = np.zeros_like(p_n)
        for i in range(n):
            # Compute the drift and diffusion terms
            drift_term = -drift[i] * (p_n[i+1] - p_n[i-1]) / (2 * self.dx)
            diffusion_term = diffusion[i] * (p_n[i+1] - 2*p_n[i] + p_n[i-1]) / (self.dx**2)
            # Update the probability density
            p_next[i] = p_n[i] + self.dt * (drift_term + diffusion_term)
        return p_next
    
    def simulate(self, num_steps):
        """
        Simulate the evolution of the probability density function over time.
        """
        p_n = self.rho
        for _ in range(num_steps):
            p_n = self.solve_step(p_n, self.drift, self.diffusion)
        return p_n

    def plot_trajectory(self, num_steps):
        """
        Plot the evolution of the probability density function over time.
        """
        p_n = self.rho
        plt.plot(p_n, label='t=0')
        for t in range(1, num_steps+1):
            p_n = self.solve_step(p_n, self.drift, self.diffusion)
            if t % (num_steps // 5) == 0: # plot at 5 time points
                plt.plot(p_n, label=f't={t*self.dt:.2f}')
        plt.title('Fokker-Planck Evolution')
        plt.xlabel('State')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()
