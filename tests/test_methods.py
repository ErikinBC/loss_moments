"""
Unit tests for `loss_moments` package.

python3 -m pytest tests/ -W ignore
"""

# External
import pytest
import numpy as np
# Internal
from . import n_samp, atol, seed
from . import loss_fun_works, loss_fun_fails
from . import dist_BVN, dist_X_uni, dist_Ycond
from src.loss_moments._base import BaseIntegrator
from src.loss_moments.methods import MonteCarloIntegrator, NumericalIntegrator


def test_MCI():
    # (i) Failing to pass in distribution will case it to fail
    with pytest.raises(AssertionError):
        MonteCarloIntegrator(loss=loss_fun_works)
    # (ii) Only passing in a joint distribution should have the others None
    mci1 = MonteCarloIntegrator(loss=loss_fun_works, dist_joint=dist_BVN)
    assert mci1.dist_X_uncond is None
    assert mci1.dist_Y_condX is None
    # (iii) Passing unconditional and conditional should result in joint_dist being None
    mci2 = MonteCarloIntegrator(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
    assert mci2.dist_joint is None
    # (iv) Passing in a vanilla dist should cause Y_condX to fial
    with pytest.raises(AssertionError):
        MonteCarloIntegrator(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_X_uni)
    # (v) Calling the integrate method shuold work
    kwargs_mu_seed = {'num_samples':n_samp, 'seed':seed, 'calc_variance':False}
    kwargs_var_seed = {**kwargs_mu_seed, **{'calc_variance':True}}
    kwargs_var_seed['calc_variance'] = True
    mu1a = mci1.integrate(**kwargs_mu_seed)
    mu1b, var1b = mci1.integrate(**kwargs_var_seed)
    assert isinstance(mu1a, float)
    assert mu1a == mu1b
    # (vi) Repeat for conditional
    mu2a = mci2.integrate(**kwargs_mu_seed)
    mu2b, var2b = mci2.integrate(**kwargs_var_seed)
    assert isinstance(mu2a, float)
    assert mu2a == mu2b
    np.testing.assert_allclose(mu1b, mu2b, atol=atol)
    np.testing.assert_allclose(var1b, var2b, atol=atol)
        
