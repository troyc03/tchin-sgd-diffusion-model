import numpy as np
import matplotlib.pyplot as plt

class FokkerPlanckSolver:
    def __init__(self, fpe_config, trap_strength, init_pos, init_var):
        """
        Initializes the Harmonic Fokker-Planck Solver from scratch.
        Fokker-Planck Eq: dP/dt = -d/dx [drift(x) * P] + D * d^2P/dx^2
        """
        self.D = fpe_config['D']
        self.dx = fpe_config['dx']
        self.dt = fpe_config['dt']
        self.trap_strength = trap_strength
        
        # Spatial grid
        self.x = np.arange(fpe_config['x_min'], fpe_config['x_max'] + self.dx, self.dx)
        self.N = len(self.x)
        
        # Initialize probability density P(x, t=0) - Gaussian
        self.rho = np.exp(-0.5 * (self.x - init_pos)**2 / init_var)
        self.rho /= np.sum(self.rho) * self.dx  # Normalize
        
        self.rho_init = self.rho.copy()
        
        # Precompute force (drift) at each point: F(x) = -dU/dx = -k*x
        self.force = -self.trap_strength * self.x

    def step(self):
        """Performs one time step using Forward Euler."""
        rho_new = np.zeros_like(self.rho)
        
        # Flux calculation: J = force*P - D*dP/dx
        # Use centered difference for spatial derivatives (interior points)
        
        # Drift term: -d/dx (force * P)
        drift_flux = self.force * self.rho
        
        # Diffusion term: D * d^2P/dx^2 (centered)
        diff_flux = self.D * (np.roll(self.rho, -1) - 2*self.rho + np.roll(self.rho, 1)) / (self.dx**2)
        
        # Finite difference update
        # dP/dt = -d/dx(J) -> we use vectorized approximation
        
        # Simplified vectorized FPE step (Forward Euler)
        # 1. Flux J = F*P - D*dP/dx
        # 2. dP/dt = -d/dx(J)
        
        # Gradient of the probability density
        drho_dx = (np.roll(self.rho, -1) - np.roll(self.rho, 1)) / (2 * self.dx)
        drho_dx[0] = (self.rho[1] - self.rho[0]) / self.dx  # Forward diff at boundary
        drho_dx[-1] = (self.rho[-1] - self.rho[-2]) / self.dx # Backward diff at boundary
        
        # Gradient of the drift force * density
        d_force_rho_dx = (np.roll(drift_flux, -1) - np.roll(drift_flux, 1)) / (2 * self.dx)
        d_force_rho_dx[0] = (drift_flux[1] - drift_flux[0]) / self.dx
        d_force_rho_dx[-1] = (drift_flux[-1] - drift_flux[-2]) / self.dx
        
        # dP/dt = -d(force*P)/dx + D * d^2P/dx^2
        # Use second derivative for diffusion
        d2rho_dx2 = (np.roll(self.rho, -1) - 2*self.rho + np.roll(self.rho, 1)) / (self.dx**2)
        
        # Final update
        self.rho = self.rho + self.dt * (-d_force_rho_dx + self.D * d2rho_dx2)
        
        # Enforce boundary conditions (zero probability at boundaries)
        self.rho[0] = 0
        self.rho[-1] = 0
        
        # Normalize to ensure sum is 1
        self.rho /= np.sum(self.rho) * self.dx

    def run_simulation(self, total_time):
        """Runs the simulation for a given total time."""
        num_steps = int(total_time / self.dt)
        print(f"Running {num_steps} steps...")
        for _ in range(num_steps):
            self.step()
        self.rho_final = self.rho.copy()
        return self

    def plot_results(self):
        """Plots the initial and final probability densities."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.rho_init, label='Initial Density', linestyle='--', color='blue')
        plt.plot(self.x, self.rho_final, label='Final Density', color='red', linewidth=2)
        
        # Theoretical Equilibrium: P(x) ~ exp(-U(x)/D)
        # U(x) = 0.5 * k * x^2
        theoretical = np.exp(-0.5 * self.trap_strength * self.x**2 / self.D)
        theoretical /= np.sum(theoretical) * self.dx # Normalize
        
        plt.plot(self.x, theoretical, label='Theoretical Equilibrium', linestyle=':', color='black')
        plt.title('Fokker-Planck Simulation')
        plt.xlabel('x')
        plt.ylabel('Density $\\rho(x, t)$')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    # 1. Configuration
    fpe_config = {
        'D': 1.0,          # Diffusion coefficient
        'dx': 0.01,        # Discretization in x
        'dt': 0.00001,     # Time step (must be small for stability)
        'x_min': -2,
        'x_max': 2
    }
    
    # 2. Physical parameters
    trap_strength = 16.0
    init_pos = -1.0
    init_var = 1.0 / 32.0 
    total_time = 0.05
    
    # 3. Run
    solver = FokkerPlanckSolver(fpe_config, trap_strength, init_pos, init_var)
    solver.run_simulation(total_time)
    
    # 4. Visualize
    solver.plot_results()
