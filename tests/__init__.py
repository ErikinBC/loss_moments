"""
Unit test package for loss_moments.

This initialization script contains the parameters used in the test_{filename}.py scripts
"""

# External
import numpy as np
from scipy.stats import multivariate_normal, norm
# Internal
from src.loss_moments.methods import MonteCarloIntegrator, NumericalIntegrator

# Parameters
n_samp = 100000  # How many samples to draw?
atol = 0.2  # How close should results be to each other
seed = 1234  # Ensure same results

# Test loss functions
def loss_fun_works(y, x):
    """Made up loss function"""
    return np.abs(y) * np.log(x **2)

def loss_fun_fails(y, z):
    """Loss function should have named argument x rather than z"""
    return np.abs(y) * np.log(z **2)


# Integrate method function arguments for the different methods
kwargs_base_mci = {'num_samples':n_samp, 'seed':seed}
kwargs_base_loop = {'method':'trapz_loop', 'k_sd':4, 'n_Y':105, 'n_X': 107}
kwargs_base_grid = {**kwargs_base_loop, **{'method':'trapz_grid'}}
di_kwargs_integrate_method = {
    'MonteCarloIntegrator': kwargs_base_mci,
    'NumericalIntegrator_loop': kwargs_base_loop,
    'NumericalIntegrator_grid': kwargs_base_grid,
}
# Add on calculate variance and not
di_kwargs_integrate_method = {
        k: {'mu':{**v, **{'calc_variance':False}}, 
            'var':{**v, **{'calc_variance':True}}} 
                for k, v in di_kwargs_integrate_method.items()}
# What methods match what integrate calls?
di_methods = {
    'MonteCarloIntegrator': MonteCarloIntegrator,
    'NumericalIntegrator_loop': NumericalIntegrator,
    'NumericalIntegrator_grid': NumericalIntegrator,
}
assert di_kwargs_integrate_method.keys() == di_methods.keys(), \
    'keys need to align b/w di_methods and di_kwargs_integrate_method'


# Set up a bivariate normal distribution and its conditional
mu_Y = -1.1
mu_X = 2.24
rho = 0.5
sigma2_Y = 0.8
sigma2_X = 1.1
off_diag = rho * np.sqrt(sigma2_Y * sigma2_X)
dist_BVN = multivariate_normal(mean=[mu_Y, mu_X], cov=[[sigma2_Y, off_diag],[off_diag, sigma2_X]])
dist_X_uni = norm(loc=mu_X, scale=np.sqrt(sigma2_X))
dist_Ycond = lambda x: norm(loc=mu_Y + rho*(sigma2_Y/sigma2_X)**0.5*(x - mu_X), scale=np.sqrt(sigma2_Y * (1-rho**2)))

# Set up a multivariate joint distribution in the Gaussian case

