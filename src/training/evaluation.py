import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

class OUProcess:
    """
    Ornstein-Uhlenbeck Process Simulator for SGD trajectory modeling.
    SDE: dX_t = theta * (mu - X_t) * dt + sigma * dW_t
    """
    def __init__(self, theta=2.0, mu=0.0, sigma=0.5, paths=1, points=1000, T=1.0, X0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.paths = paths
        self.points = points
        self.T = T
        self.X0 = X0 if X0 is not None else mu
        self.dt = T / (points - 1)
        self.t_axis = np.linspace(0, T, points)

    def simulate(self):
        """Vectorized simulation using Euler-Maruyama."""
        X = np.zeros((self.paths, self.points))
        X[:, 0] = self.X0
        # Pre-generate noise for efficiency
        dW = np.random.normal(0, np.sqrt(self.dt), (self.paths, self.points - 1))
        
        for i in range(self.points - 1):
            # Ito discretization: X_{t+1} = X_t + theta*(mu - X_t)*dt + sigma*dW_t
            X[:, i+1] = X[:, i] + self.theta * (self.mu - X[:, i]) * self.dt + self.sigma * dW[:, i]
        return X

    def fit_to_trajectory(self, trajectory, t=None):
        """
        Fits OU parameters using Maximum Likelihood Estimation (MLE).
        MLE is more robust for stochastic data than direct MSE minimization.
        """
        if t is None: t = np.linspace(0, self.T, len(trajectory))
        dt_arr = np.diff(t)
        dX = np.diff(trajectory)
        X_prev = trajectory[:-1]

        def neg_log_likelihood(params):
            theta, mu, sigma = params
            # Numerical stability: prevent non-positive parameters
            if sigma <= 1e-6 or theta <= 1e-6: return 1e10
            
            # Theoretical change in X and variance for a Gaussian transition
            expected_dX = theta * (mu - X_prev) * dt_arr
            variance = (sigma**2) * dt_arr
            
            # Log-likelihood for a normal distribution transition
            # Sum of log(PDF of normal distribution)
            nll = 0.5 * np.sum(np.log(2 * np.pi * variance) + (dX - expected_dX)**2 / variance)
            return nll

        # Smart initial guess: [unit theta, data mean, scaled volatility]
        init_guess = [1.0, np.mean(trajectory), np.std(dX) / np.sqrt(np.mean(dt_arr))]
        res = minimize(neg_log_likelihood, init_guess, 
                       bounds=[(1e-3, 50), (None, None), (1e-3, 20)])
        
        self.theta, self.mu, self.sigma = res.x
        return res.x

def evaluate_sgd_vs_ou(sgd_norms, t=None):
    """
    Evaluates fit by comparing SGD data against the OU analytical mean.
    Analytical Mean: E[X_t] = X_0 * e^(-theta*t) + mu * (1 - e^(-theta*t))
    """
    if t is None: t = np.linspace(0, 1, len(sgd_norms))
    
    # Fit the model to the specific data provided
    model = OUProcess(X0=sgd_norms[0], T=t[-1], points=len(t))
    theta, mu, sigma = model.fit_to_trajectory(sgd_norms, t)
    
    # The expected deterministic path (the "backbone" of the process)
    expected_path = sgd_norms[0] * np.exp(-theta * t) + mu * (1 - np.exp(-theta * t))
    
    mse = mean_squared_error(sgd_norms, expected_path)
    return [theta, mu, sigma], mse, expected_path

if __name__ == '__main__':
    # 1. Create Synthetic "SGD Gradient Norm" data 
    # (Starting high and decaying toward a noisy equilibrium)
    true_params = {'theta': 4.0, 'mu': 0.1, 'sigma': 0.08}
    t_synth = np.linspace(0, 2, 1000)
    simulator = OUProcess(**true_params, X0=0.7, T=2, points=1000)
    sgd_simulated_data = simulator.simulate()[0]

    # 2. Evaluate and Fit
    fitted_params, mse, mean_path = evaluate_sgd_vs_ou(sgd_simulated_data, t_synth)
    
    print("--- Results ---")
    print(f"True Params:   theta={true_params['theta']}, mu={true_params['mu']}, sigma={true_params['sigma']}")
    print(f"Fitted Params: theta={fitted_params[0]:.3f}, mu={fitted_params[1]:.3f}, sigma={fitted_params[2]:.3f}")
    print(f"MSE (Data vs Analytical Mean): {mse:.6f}")

    # 3. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(t_synth, sgd_simulated_data, label='Noisy SGD Norms (Simulated)', color='gray', alpha=0.4)
    plt.plot(t_synth, mean_path, label='OU Analytical Mean', color='red', lw=2)
    plt.axhline(fitted_params[1], color='blue', ls='--', label=f'Fitted Mu: {fitted_params[1]:.3f}')
    plt.title('SGD Gradient Norm Evaluation via OU Process')
    plt.xlabel('Time (normalized)')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
