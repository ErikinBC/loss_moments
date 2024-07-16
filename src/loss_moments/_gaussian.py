"""
Contains classes for when the data distribution is Gaussian
"""

import numpy as np
from typing import Tuple
from scipy.stats import norm, multivariate_normal

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
        sigma_u = sigma2_u**0.5
        self.dist_u = norm(loc=0, scale=sigma_u)
        self.Sigma = norm().rvs(size=(p, p), random_state=seed)
        self.Sigma = self.Sigma.T.dot(self.Sigma)
        self.dist_x = multivariate_normal(cov=self.Sigma)
        self.beta = norm(scale=1/np.sqrt(p)).rvs(size=p, random_state=seed)
        # define the conditional distribution: y | x
        self.dist_ycond = lambda x: norm(loc=x.dot(self.beta), scale=sigma_u)
        
        
    
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
        """For a given beta_hat, calculate the oracle values of the risk and loss variance"""
        beta_delta = self.beta - beta_hat
        covar_error = beta_delta.dot(self.Sigma).dot(beta_delta)
        risk = covar_error + self.sigma2_u
        lossvar = 2 * (risk ** 2)
        return risk, lossvar