"""
Unit tests for `loss_moments` package.

python3 -m pytest tests/ -W ignore
"""

# External
import pytest
import numpy as np
from scipy.stats import multivariate_normal, norm
# Internal
from src.loss_moments._base import BaseIntegrator
from src.loss_moments.methods import MonteCarloIntegrator, NumericalIntegrator

def loss_fun_works(y, x):
    """Made up loss function"""
    return np.abs(y) * np.log(x **2)

def loss_fun_fails(y, z):
    """Loss function should have named argument x rather than z"""
    return np.abs(y) * np.log(z **2)

dist_BVN = multivariate_normal(mean=[1,2], cov=[[0.5, 2.1],[0.2, 0.9]])
dist_X_uni = norm(loc=1.1, scale=0.4)
dist_Ycond = lambda x: norm(loc=x**2, scale=0.1)

def test_MCI():
    # (i) Failing to pass in distribution will case it to fail
    with pytest.raises(AssertionError):
        MonteCarloIntegrator(loss=loss_fun_works)
    # (ii) Only passing in a joint distribution should have the others None
    mci = MonteCarloIntegrator(loss=loss_fun_works, dist_joint=dist_BVN)
    assert mci.dist_X_uncond is None
    assert mci.dist_Y_condX is None
    # (iii) vice-versa
    mci = MonteCarloIntegrator(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
    assert mci.dist_joint is None
    

