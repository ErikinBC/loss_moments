"""
Unit test package for loss_moments.

This initialization script contains the parameters used in the test_{filename}.py scripts

python3 tests/__init__.py
"""

# External
import numpy as np
from scipy.stats import multivariate_normal, norm, expon
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
def f_theta_mean(x):
    z = np.atleast_1d(x)
    if len(z.shape) == 1:
        return z
    else:
        return z.mean(axis=1)

def f_theta_coef(x):
    z = np.atleast_1d(x)
    if len(z.shape) == 1:
        return x
    else:
        theta = norm().rvs(x.shape[1], x.shape[1])
        return x.dot(theta)

f_theta_identity = lambda x: x

f_theta_tuple = lambda x: np.array([x.mean(), x.var()])


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
        k: {'mu':{**v, **{'calc_variance':False, 'seed': seed}}, 
            'var':{**v, **{'calc_variance':True, 'seed': seed}}} 
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

class expon_X:
    """
    A class which generates correlated exponential data: X_j = expon_j(rate) + expon(rate)

    The .rvs() produces real correlated data, the pdf is made up
    """
    def __init__(self, p: int, seed: int | None = None) -> None:
        self.p = p
        # Generate some scales
        self.dist_z = expon(scale=expon().rvs(p, seed))
        self.dist_u = expon(scale = 1)
    
    def rvs(self, size=None, random_state=None) -> np.ndarray:
        """Returns a (size, p) array"""
        u = self.dist_u.rvs(size=size, random_state=random_state)
        u = np.atleast_2d(u).T  # Make into a column vector
        z = self.dist_z.rvs(size=(size, self.p), random_state=random_state)
        x = z + u
        x = np.squeeze(x)
        return x
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Maps the R^p vector to a scalar"""
        f_u = self.dist_u.pdf(x)
        f_z = self.dist_z.pdf(x)
        if self.p > 1:
            dens = f_u.mean(axis=1) + f_z.mean(axis = 1)
        else:
            dens = f_u + f_z
        return dens

    def mean(self):
        return self.dist_z.mean()
    
    def std(self):
        return self.dist_z.std()


class expon_yX:
    """
    A class where we pretend we know the joint distribution of (y, x) where y and x_j are all exponentials
    """
    def __init__(self, p: int, seed: int | None = None) -> None:
        self.p = p
        self.dist_y = expon(scale=expon().rvs(1, seed))
        self.dist_x = expon_X(p=p, seed=seed)
        self.mean = np.squeeze(expon().rvs(p+1, seed)) + 5
        self.cov = np.squeeze(expon().rvs((p+1, p+1), seed))
        self.cov = self.cov.T.dot(self.cov)

    def rvs(self, size=None, random_state=None):
        """
        When we generate data, we use correlated_expon_X_gen to generate correlate X data, and then have y be a mix of an exponential and an average of correlated exponentials
        """
        x = self.dist_x.rvs(size, random_state)
        y = self.dist_y.rvs(size, random_state)
        if self.p > 1:
            eta = x.mean(axis=1)
        else:
            eta = x
        y += eta
        y = np.atleast_2d(y).T  # Make a column vector
        yx = np.hstack((y, x))
        return yx
    
    def pdf(self, x):
        # This is not technically correct, just making sure it works computationally
        assert x.shape[-1] == 1 + self.p, f'expected x to have the last dimenions be of size {1 + self.p}, not {x.shape[-1]}'
        yy = x[..., 0]
        xx = np.squeeze(x[..., 1:])
        f_y = self.dist_y.pdf(yy)
        f_x = self.dist_x.pdf(xx)
        if xx.ndim > yy.ndim:
            f_x = np.mean(f_x, axis=-1)
        dens = f_y + f_x
        return dens

class expon_y_cond_X:
    """
    A class where we pretend, conditional on x, the distribution of y is exponential
    """
    def __init__(self, p: int, seed: int | None = None) -> None:
        self.p = p
        self.theta = expon().rvs(p, seed)
    
    def __call__(self, x: np.ndarray):
        eta = x
        if self.p > 1:
            eta = eta.dot(self.theta)
        cond_dist = expon(loc=np.abs(eta), scale=eta**2)
        return cond_dist

# Create the distributions to be used in the unit testing (p > 1 can only be for monte carlo)
p = 5
di_dists_expon = {
    'MonteCarloIntegrator': 
        {'dist_joint': expon_yX(p=5, seed=seed), 
         'dist_X_uncond': expon_X(p=5, seed=seed),
         'dist_Y_condX': expon_y_cond_X(p=5, seed=seed),
         },
    'NumericalIntegrator_loop': 
        {'dist_joint': expon_yX(p=1, seed=seed), 
         'dist_X_uncond': expon_X(p=1, seed=seed),
         'dist_Y_condX': expon_y_cond_X(p=1, seed=seed),
         },
}
# Distributions are the same for the grip method
di_dists_expon['NumericalIntegrator_grid'] = di_dists_expon['NumericalIntegrator_loop'].copy()

# which f_theta's will work with the different methods?
di_f_theta = {
    'MonteCarloIntegrator': {'works': [f_theta_mean, f_theta_coef,],
                             'fails': [f_theta_identity, f_theta_tuple,]}, 
    'NumericalIntegrator_loop': {'works': [f_theta_identity, f_theta_mean, f_theta_coef,],
                                 'fails': [f_theta_tuple, ]},
    'NumericalIntegrator_grid': {'works': [f_theta_identity,],
                                 'fails': [f_theta_tuple, f_theta_coef, f_theta_mean, ]},
}

