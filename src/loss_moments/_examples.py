"""
Contains examples of stlyzed distributions for unit testing and demonstrations
"""

import numpy as np
from typing import Tuple
from scipy.stats import norm, multivariate_normal
   
# Use the MSE as the loss function
def empirical_MSE_risk_lossvar(y, x) -> Tuple[float, float]:
    risk = np.mean((y - x)**2)
    lossvar = np.var((y - x)**2, ddof=1)
    return risk, lossvar


# Create a data generating process
class dgp_yx():
    def __init__(self, 
                    p: int = 10, 
                    sigma2_u: float = 1,
                    seed: int = 1,
                ) -> None:
        """
        Construct a simple data generating process (DGP):
        X ~ MVN(0, Sigma)
        u ~ N(0, sigma^2_u)
        beta ~ N(0, 1/sqrt(p))
        y = X'beta + u
        y | beta ~ N(0, beta'Sigma beta + sigma^2_u)
        """
        self.p = p
        self.sigma2_u = sigma2_u
        self.dist_u = norm(loc=0, scale=sigma2_u**0.5)
        self.Sigma = norm().rvs(size=(p, p), random_state=seed)
        self.Sigma = self.Sigma.T.dot(self.Sigma)
        self.dist_x = multivariate_normal(cov=self.Sigma)
        self.beta = norm(scale=1/np.sqrt(p)).rvs(size=p, random_state=seed)
    
    def gen_yX(
                self, 
                n: int, 
                beta_hat: np.ndarray | None = None,
                seed: int = 1,
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate either new labels and features, or new labels and predicted values for a given beta_hat
        """
        u = self.dist_u.rvs(size=n, random_state=seed)    
        x = self.dist_x.rvs(size=n, random_state=seed)
        eta = x.dot(self.beta)
        y = eta + u
        if beta_hat is None:
            return y, x
        else:
            eta_hat = x.dot(beta_hat)
            return y, eta_hat
        
    def get_risk_lossvar(self, beta_hat: np.ndarray):
        """
        For a given beta_hat, calculate the oracle values of the risk and loss variance
        """
        beta_delta = self.beta - beta_hat
        covar_error = beta_delta.dot(self.Sigma).dot(beta_delta)
        risk = covar_error + self.sigma2_u
        lossvar = 2 * (risk ** 2)
        return risk, lossvar


def simulation(
        n_train: int=30, 
        n_oos: int=10000, 
        p: int = 10, 
        verbose: bool = True,
         ):
    """
    Call the main methods
    """
    # Packages and modules
    import numpy as np
    from ._examples import dgp_yx, empirical_MSE_risk_lossvar
    from .methods import MonteCarloIntegrator, NumericalIntegrator

    # (i) Generate some training data
    generator = dgp_yx(p=p)
    y, x = generator.gen_yX(n=n_train)
    # "Learn" the coefficients (oracle, OLS, and random)
    beta_oracle = generator.beta
    theta_hat = np.dot((np.linalg.inv(np.dot(x.T,x))), np.dot(x.T, y))
    gamma_hat = np.random.permutation(theta_hat)

    # (ii) Calculate the theoretical (oracle) value of a given coeffient
    theory_risk_oracle, theory_lossvar_oracle = generator.get_risk_lossvar(beta_oracle)
    theory_risk_theta, theory_lossvar_theta = generator.get_risk_lossvar(theta_hat)
    theory_risk_gamma, theory_lossvar_gamma = generator.get_risk_lossvar(gamma_hat)

    # (iii) Calculate empirical risk and loss variance for the MSE loss function
    y_oos, eta_oracle = generator.gen_yX(n_oos, beta_oracle)
    emp_risk_oracle, emp_lossvar_oracle = empirical_MSE_risk_lossvar(y_oos, eta_oracle)
    _, eta_theta = generator.gen_yX(n_oos, theta_hat)
    emp_risk_theta, emp_lossvar_theta = empirical_MSE_risk_lossvar(y_oos, eta_theta)
    np.testing.assert_equal(y_oos, _)
    _, eta_gamma = generator.gen_yX(n_oos, gamma_hat)
    emp_risk_gamma, emp_lossvar_gamma = empirical_MSE_risk_lossvar(y_oos, eta_gamma)
    
    # (iv) Calculate the 

    # (v) Print results
    if verbose:
        print('\n--- Risk ---')
        print(f'oracle: Empirical={emp_risk_oracle:.2f}, theory={theory_risk_oracle:.2f}, integral={0:.2f}')
        print(f'theta: Empirical={emp_risk_theta:.2f}, theory={theory_risk_theta:.2f}, integral={0:.2f}')
        print(f'gamma: Empirical={emp_risk_gamma:.2f}, theory={theory_risk_gamma:.2f}, integral={0:.2f}')

        print('\n--- Loss variance ---')
        print(f'oracle: Empirical={emp_lossvar_oracle:.2f}, theory={theory_lossvar_oracle:.2f}, integral={0:.2f}')
        print(f'theta: Empirical={emp_lossvar_theta:.2f}, theory={theory_lossvar_theta:.2f}, integral={0:.2f}')
        print(f'gamma: Empirical={emp_lossvar_gamma:.2f}, theory={theory_lossvar_gamma:.2f}, integral={0:.2f}')

    # Return values (for unit testing)
    di_ret = {
                'oracle':
                    {
                        'risk':[emp_risk_oracle, theory_risk_oracle, ],
                        'lossvar': [emp_lossvar_oracle, theory_lossvar_oracle, ], 
                    },
                'theta':
                    {
                        'risk':[emp_risk_theta, theory_risk_theta, ],
                        'lossvar': [emp_lossvar_theta, theory_lossvar_theta, ], 
                    },
                'gamma':
                    {
                        'risk':[emp_risk_gamma, theory_risk_gamma, ],
                        'lossvar': [emp_lossvar_gamma, theory_lossvar_gamma, ], 
                    },
             }
    return di_ret