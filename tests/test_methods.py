"""
Unit tests for `loss_moments` package.

python3 -m pytest tests/test_methods.py -W ignore -s
"""

# External
import pytest
import numpy as np
# Internal
from . import atol
from . import loss_fun_works, loss_fun_fails
from . import dist_BVN, dist_X_uni, dist_Ycond
from . import di_kwargs_integrate_method, di_methods


class TestIntegrators:
    @pytest.mark.parametrize("method_name", list(di_methods.keys()))
    def test_method(self, method_name: str):
        """
        Call all internal methods, and check that the joint vs cond/uncond are "roughly" equal with {n_samp} samples
        """
        # Class the method and set up integrate keywords
        self.method_class = di_methods[method_name]
        self.di_kwargs = di_kwargs_integrate_method[method_name]
        self.kwargs_mu_seed = self.di_kwargs['mu']
        self.kwargs_var_seed = self.di_kwargs['var']
        assert hasattr(self.method_class, 'integrate')
        # Test the setups
        mci1 = self._test_only_joint_distribution()
        mci2 = self._test_unconditional_and_conditional_distribution()
        # Test integrals
        mu1b, var1b = self._test_integration_results(mci1)
        mu2b, var2b = self._test_integration_results(mci2)
        # Test equivalence between conditional/unconditional vs joint
        np.testing.assert_allclose(mu1b, mu2b, atol=atol)
        np.testing.assert_allclose(var1b, var2b, atol=atol)
        print(f'\nPassed all tests for {method_name}\n')

    def _test_missing_distribution(self):
        """Method construction needs at least one distribution"""
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_works)


    def _test_only_joint_distribution(self):
        """When the joint is provided the conditional/unconditional should be None"""
        mci1 = self.method_class(loss=loss_fun_works, dist_joint=dist_BVN)
        assert mci1.dist_X_uncond is None
        assert mci1.dist_Y_condX is None
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_fails, dist_joint=dist_BVN)
        return mci1

    def _test_unconditional_and_conditional_distribution(self):
        """When the conditional/unconditional is provided the joint should be None"""
        mci2 = self.method_class(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
        assert mci2.dist_joint is None
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_fails, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
        return mci2

    def _test_invalid_conditional_distribution(self):
        """You should not be able to assign Y_condX that isn't a callable"""
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_X_uni)

    def _test_integration_results(self, integrator):
        """
        Call the integrate method, and the variance flag should return a tuple, and the risk (mena) should be the same
        """
        mu1a = integrator.integrate(**self.kwargs_mu_seed)
        mu1b, var1b = integrator.integrate(**self.kwargs_var_seed)
        assert isinstance(mu1a, float)
        assert mu1a == mu1b
        return mu1a, var1b