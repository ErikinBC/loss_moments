"""
Contains classes for when the data distribution is Gaussian
"""

import numpy as np
from typing import Tuple
from scipy.stats import norm, multivariate_normal


class linear_predictor():
    """Convenience wrapper for a trained model to return a predicted value"""
    def __init__(self, weights: np.ndarray) -> None:
        self.weights = np.atleast_1d(weights)
        self.p = len(self.weights)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        l_x_shape = len(x.shape)
        assert l_x_shape in [1, 2], f'x should have 1 or two dimensions, not {l_x_shape}'
        if l_x_shape == 1:
            assert self.p == 1, f'If x is a 1-d array, weights needs to be a scalar'
            return x * self.weights
        else:
            return x.dot(self.weights)


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
        # (i) Set up the parameters
        self.p = p
        self.sigma2_u = sigma2_u
        self.sigma_u = sigma2_u**0.5
        self.dist_u = norm(loc=0, scale=self.sigma_u)
        self.Sigma = norm().rvs(size=(p, p), random_state=seed)
        self.Sigma = self.Sigma.T.dot(self.Sigma)
        self.Sigma /= np.max(np.abs(self.Sigma))
        self.mu_p = np.zeros(p)
        self.mu_p1 = np.zeros(p + 1)
        self.dist_x = multivariate_normal(mean=self.mu_p, cov=self.Sigma)
        self.beta = norm(scale=1/np.sqrt(p)).rvs(size=p, random_state=seed)
        # (ii) Define the distributions w.r.t. y, x
        self.dist_ycondx = lambda x: norm(loc=x.dot(self.beta), scale=self.sigma_u)
        self.cov_eta = self.beta.dot(self.Sigma).dot(self.beta)
        self.cov_y = self.cov_eta + sigma2_u
        cov_yx = self.beta.dot(self.Sigma)
        cov_joint = np.vstack([
            np.hstack((self.cov_y, cov_yx)),
            np.hstack((np.atleast_2d(cov_yx).T, self.Sigma)),
        ])
        self.dist_yx = multivariate_normal(mean=self.mu_p1, cov=cov_joint)
        
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
    

    def gen_f_theta_dists(self, theta: np.ndarray):
        """
        For a given set of weights (coefficients), calculate the distributions of:

        (y, X^beta_hat) ~ BVN(...)
        y | X^beta_hat ~ N(loc=X^beta_hat, sigma^2_u)
        X^beta_hat ~ N(0, \Sigma_{\hat\beta})
        """
        # (i) Calculate BVN
        Sigma_theta = theta.dot(self.Sigma).dot(theta)
        Sigma_beta = self.cov_eta
        Sigma_theta_beta = theta.dot(self.Sigma).dot(self.beta)
        Sigma_yeta = np.array([[self.cov_y, Sigma_theta_beta], 
                               [Sigma_theta_beta, Sigma_theta]])
        mu2 = np.zeros(2)
        dist_y_eta = multivariate_normal(mean=mu2, cov=Sigma_yeta)
        # (ii) Calculate unconditional distribution of f_theta
        dist_eta = norm(loc=0, scale=np.sqrt(Sigma_theta))
        # (iii) Calculate conditional distribution of y | f_theta
        dist_ycond_eta = lambda x: norm(loc=x.dot(theta) * (Sigma_theta_beta / Sigma_theta), scale=np.sqrt(Sigma_beta + self.sigma2_u - (Sigma_theta_beta**2/Sigma_theta) ))
        # Return
        return dist_y_eta, dist_eta, dist_ycond_eta
