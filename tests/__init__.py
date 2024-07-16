"""Unit test package for loss_moments."""

import numpy as np
from scipy.stats import multivariate_normal, norm

# Parameters
n_samp = 100000  # How many samples to draw?
atol = 0.2  # How close should results be to each other
seed = 1234  # Ensure same results


def loss_fun_works(y, x):
    """Made up loss function"""
    return np.abs(y) * np.log(x **2)

def loss_fun_fails(y, z):
    """Loss function should have named argument x rather than z"""
    return np.abs(y) * np.log(z **2)

# Set up a bivariate normal distribution
mu_Y = -1.1
mu_X = 2.24
rho = 0.5
sigma2_Y = 0.8
sigma2_X = 1.1
off_diag = rho * np.sqrt(sigma2_Y * sigma2_X)
dist_BVN = multivariate_normal(mean=[mu_Y, mu_X], cov=[[sigma2_Y, off_diag],[off_diag, sigma2_X]])
dist_X_uni = norm(loc=mu_X, scale=np.sqrt(sigma2_X))
dist_Ycond = lambda x: norm(loc=mu_Y + rho*(sigma2_Y/sigma2_X)**0.5*(x - mu_X), scale=np.sqrt(sigma2_Y * (1-rho**2)))
