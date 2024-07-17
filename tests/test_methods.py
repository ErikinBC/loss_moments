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
from . import dist_expon_x, dist_expon_y_x, dist_expon_yx
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
        intergator_joint_gauss = self._test_only_joint_distribution()
        intergator_cond_gauss = self._test_unconditional_and_conditional_distribution()
        # Test integrals
        mu1b, var1b = self._test_integration_results(intergator_joint_gauss)
        mu2b, var2b = self._test_integration_results(intergator_cond_gauss)
        # Test equivalence between conditional/unconditional vs joint
        np.testing.assert_allclose(mu1b, mu2b, atol=atol)
        np.testing.assert_allclose(var1b, var2b, atol=atol)
        print(f'\nPassed all tests for {method_name}\n')
        # Check the different f-theta's
        self._test_f_theta()


    def _test_missing_distribution(self):
        """Method construction needs at least one distribution"""
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_works)

    def _test_f_theta(self):
        """Checks that f_theta is valid"""
        integrator_joint_expon = self.method_class(loss=loss_fun_works, dist_joint=dist_expon_yx)
        integrator_cond_expon = self.method_class(loss=loss_fun_works, 
                        dist_X_uncond=dist_expon_x, dist_Y_condX=dist_expon_y_x)


    def _test_only_joint_distribution(self):
        """When the joint is provided the conditional/unconditional should be None"""
        intergator_joint_gauss = self.method_class(loss=loss_fun_works, dist_joint=dist_BVN)
        assert intergator_joint_gauss.dist_X_uncond is None
        assert intergator_joint_gauss.dist_Y_condX is None
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_fails, dist_joint=dist_BVN)
        return intergator_joint_gauss

    def _test_unconditional_and_conditional_distribution(self):
        """When the conditional/unconditional is provided the joint should be None"""
        intergator_cond_gauss = self.method_class(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
        assert intergator_cond_gauss.dist_joint is None
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_fails, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
        return intergator_cond_gauss

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