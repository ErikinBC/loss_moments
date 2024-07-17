"""
Unit test package for loss_moments.

This initialization script contains the parameters used in the test_{filename}.py scripts
"""

# External
import numpy as np
from scipy.stats import multivariate_normal, norm, expon, rv_continuous
from scipy.stats._distn_infrastructure import rv_continuous_frozen
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

# Test f_theta's



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

# Set up a multivariate joint distribution in the non-Gaussian case
class correlated_expon_yX_gen(rv_continuous):
    def __init__(self, p: int = 5, rate: float = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.dist_y = expon(loc=rate, scale=rate+1)
        self.dist_x = correlated_expon_X_gen(p=p, rate=rate)

    def _rvs(self, size=None, random_state=None):
        x = self.dist_x.rvs(size, random_state)
        y = self.dist_y.rvs(size, random_state) + x.mean(axis=1)
        yx = np.hstack((y, x))
        return yx
    
    def _pdf(self, x):
        # This is not technically correct, just making sure it works computationally
        f_y = self.dist_y.pdf(x[:,0])
        f_x = self.dist_x.pdf(x[:,1:]).mean(axis=1)
        return f_y + f_x


class correlated_expon_X_gen(rv_continuous):
    def __init__(self, p: int = 5, rate: float = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.dist_corr = expon(scale=rate)
        self.dist_indep = expon(scale=np.repeat(rate, p))

    def _rvs(self, size=None, random_state=None):
        u = self.dist_corr.rvs(size=(size, self.p), random_state=random_state)
        x = self.dist_indep.rvs(size=(size, self.p), random_state=random_state)
        z = x + np.atleast_2d(u).T
        return z
    
    def _pdf(self, x):
        # This is not technically correct, just making sure it works computationally
        f = self.dist_indep.pdf(x).mean(axis=1)
        return f

class y_expon_X_gen():
    def __init__(self, p: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        self.p = p
        self.theta = expon().rvs(p, seed)
    
    def __call__(self, x: np.ndarray):
        eta = x.dot(self.theta)
        return expon(loc=np.abs(eta), scale=eta**2)


# # I THINK THIS SHUOLD FAIL B/C X IS MULTIDIMENSIONAL!
# def dist_yx_expon(x):
#     return expon(loc=np.abs(x), scale=x**2)
expon_x = correlated_expon_X_gen()
expon_yx = correlated_expon_yX_gen()
dist_expon_x = expon_x()
dist_expon_y_x = y_expon_X_gen(p=expon_x.p)
dist_expon_yx = expon_yx()
