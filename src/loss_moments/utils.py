"""
Utility files
"""

# External modules
import numpy as np
from scipy import stats
from typing import Callable
from inspect import signature, Parameter

def is_int(x):
    """Checks that x is an integer"""
    return int(x) == x

def find_mandatory_kwargs(func) -> list:
    """Find the mandatory named key word arguments of any function"""
    assert isinstance(func, Callable), 'func needs to be callable'
    # Get the signature of the function
    sig = signature(func)
    # Get the parameters of the function
    params = sig.parameters
    # List to hold the names of mandatory arguments
    mandatory_args = []
    # Iterate through the parameters
    for name, param in params.items():
        # Check if the parameter is positional-only or keyword-only and does not have a default value
        if param.default == Parameter.empty:
            mandatory_args.append(name)
    return mandatory_args


class dist_Ycond_BVN:
    def __init__(self, mu_Y, sigma_Y, sigma_X, rho, mu_X):
        """
        Function to calculate the a univariate gaussian of: Y | X, when:

        (Y, X) ~ BVN([mu_Y, mu_X], 
                     [[sigma_Y^2, sigma_Y*sigma_X*rho],
                     [sigma_Y*sigma_X*rho, sigma_X^2]])

        
        Remember that Y | X = x ~ N(mu_Y + sigma_Y/sigma_X*rho*(x-mu_X), sigma2_Y*(1-rho^2))
        """
        self.mu_Y = mu_Y
        self.sigma_Y = sigma_Y
        self.sigma_X = sigma_X
        self.rho = rho
        self.mu_X = mu_X
        self.sigma2_Y = sigma_Y**2

    def __call__(self, x):
        loc = self.mu_Y + (self.sigma_Y / self.sigma_X) * self.rho * (np.array(x) - self.mu_X)
        scale = np.sqrt(self.sigma2_Y * (1 - self.rho ** 2))
        return stats.norm(loc=loc, scale=scale)

