import numpy as np
import matplotlib.pyplot as plt

class FokkerPlanckSolver:
    def __init__(self, drift_func, diffusion_func, x_range=(-10, 10), dx=0.1):
        self.drift_func = drift_func
        self.diffusion_func = diffusion_func
        self.x_range = x_range
        self.dx = dx
        self.x_grid = np.arange(x_range[0], x_range[1] + dx, dx)

    def step(self, p):
        drift = self.drift_func(self.x_grid)
        diffusion = self.diffusion_func(self.x_grid)
        
        dp_dt = -np.gradient(drift * p, self.dx) + 0.5 * np.gradient(diffusion * np.gradient(p, self.dx), self.dx)
        return dp_dt

    def solve(self, p0, T, dt):
        num_steps = int(T / dt)
        p = np.zeros((num_steps + 1, len(self.x_grid)))
        p[0] = p0
        
        for t in range(1, num_steps + 1):
            dp_dt = self.step(p[t-1])
            p[t] = p[t-1] + dp_dt * dt
            
            # Ensure non-negativity and normalization
            p[t] = np.maximum(p[t], 0)
            p[t] /= np.sum(p[t]) * self.dx
        
        return p
    
# Example usage
if __name__ == "__main__":
    # Define drift and diffusion functions
    def drift(x):
        return -0.5 * x

    def diffusion(x):
        return 0.1 * np.ones_like(x)

    # Create solver
    solver = FokkerPlanckSolver(drift, diffusion, x_range=(-10, 10), dx=0.1)

    # Initial condition: Gaussian centered at 0
    x_grid = solver.x_grid
    p0 = np.exp(-x_grid**2 / (2 * 0.5**2)) / (0.5 * np.sqrt(2 * np.pi))

    # Solve Fokker-Planck equation
    T = 10.0  # Total time
    dt = 0.01  # Time step
    p = solver.solve(p0, T, dt)

    # Plot results
    plt.figure(figsize=(10, 6))
    for t in range(0, p.shape[0], int(p.shape[0] / 5)):
        plt.plot(x_grid, p[t], label=f'Time={t*dt:.2f}')
    plt.xlabel('x')
    plt.ylabel('p(x, t)')
    plt.title('Fokker-Planck Equation Solution')
    plt.legend()
    plt.grid()
    plt.show()