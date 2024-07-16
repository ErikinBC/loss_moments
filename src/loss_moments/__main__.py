"""
Script to test that package compiled properly:

python3 -m loss_moments
"""

def main(n_train: int=30, n_oos: int=10000, p: int = 10):
    """
    Call the main methods
    """
    # Packages and modules
    import numpy as np
    from scipy.stats import norm, multivariate_normal
    from .methods import MonteCarloIntegrator, NumericalIntegrator

    # Create a data generating process
    class dgp_yx():
        def __init__(self, p: int = 10, seed: int = 1) -> None:
            self.p = p
            Sigma = norm().rvs(size=(p, p), random_state=seed)
            Sigma = Sigma.T.dot(Sigma)
            self.dist_x = multivariate_normal(cov=Sigma)
            self.beta = norm(scale=1/np.sqrt(p)).rvs(size=p, random_state=seed)
            self.dist_u = norm()
        
        def gen_yX(
                    self, n: int, seed: int = 1,
                    beta_hat: np.ndarray | None = None,
                    ):
            # Generate the data
            u = self.dist_u.rvs(size=n, random_state=seed)    
            x = self.dist_x.rvs(size=n, random_state=seed)
            eta = x.dot(self.beta)
            y = eta + u
            if beta_hat is None:
                return y, x
            else:
                eta_hat = x.dot(beta_hat)
                return y, eta_hat

    # Use the MSE as the loss function
    def MSE(y, x):
        return np.mean((y - x)**2)

    # Generate some training data
    generator = dgp_yx(p=p)
    y, x = generator.gen_yX(n=n_train)
    # Learn the coefficients
    theta_hat = np.dot((np.linalg.inv(np.dot(x.T,x))), np.dot(x.T, y))
    gamma_hat = np.random.permutation(theta_hat)

    # Calculate the OOS MSE
    y_oos, eta_oracle = generator.gen_yX(n_oos, beta_hat=generator.beta)
    mse_oracle = MSE(y_oos, eta_oracle)
    _, eta_theta = generator.gen_yX(n_oos, beta_hat=theta_hat)
    mse_theta = MSE(y_oos, eta_theta)
    np.testing.assert_equal(y_oos, _)
    _, eta_gamma = generator.gen_yX(n_oos, beta_hat=gamma_hat)
    mse_gamma = MSE(y_oos, eta_gamma)
    
    # emp_gamma = MSE(y_oos, eta_gamma)
    print(f'oracle: Empirical={mse_oracle:.2f}, theory={0}, integral={0}')
    print(f'theta: Empirical={mse_theta:.2f}, theory={0}, integral={0}')
    print(f'gamma: Empirical={mse_gamma:.2f}, theory={0}, integral={0}')


if __name__ == '__main__':
    main()