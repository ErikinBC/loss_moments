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
from . import f_theta_works_v1, f_theta_works_v2, f_theta_fails_v1, f_theta_fails_v2

class TestIntegrators:
    @pytest.mark.parametrize("method_name", list(di_methods.keys()))
    def test_method(self, method_name: str):
        """
        Call all internal methods, and check that the joint vs cond/uncond are "roughly" equal with {n_samp} samples
        """
        # (i) Class the method and set up integrate keywords
        self.method_class = di_methods[method_name]
        self.di_kwargs = di_kwargs_integrate_method[method_name]
        self.kwargs_mu_seed = self.di_kwargs['mu']
        self.kwargs_var_seed = self.di_kwargs['var']
        assert hasattr(self.method_class, 'integrate')
        # # (ii) Test Gaussian (univariate) results
        # intergator_joint_gauss = self._test_joint_construct_gauss()
        # intergator_cond_gauss = self._test_cond_construct_gauss()
        # # Test integrals
        # mu1b, var1b = self._test_integration_gauss(intergator_joint_gauss)
        # mu2b, var2b = self._test_integration_gauss(intergator_cond_gauss)
        # # Test equivalence between conditional/unconditional vs joint
        # np.testing.assert_allclose(mu1b, mu2b, atol=atol)
        # np.testing.assert_allclose(var1b, var2b, atol=atol)

        # (iii) Test Non-Gaussian (multivariate_ results)
        #       These methods also need an f_theta function to map the vector
        self.kwargs_construct_expon_joint = {'loss':loss_fun_works, 'dist_joint': dist_expon_yx}
        self.kwargs_construct_expon_cond = {'loss':loss_fun_works, 'dist_X_uncond': dist_expon_x, 'dist_Y_condX': dist_expon_y_x}
        # Construct the different methods
        self._test_joint_construct_expon()
        self._test_cond_construct_expon()
        # Check the integration call
        self._test_joint_integrate_expon()

        print(f'\nPassed all tests for {method_name}\n')

    def _test_joint_construct_expon(self):
        """Should be able to return the constructed classes"""
        self.intergator_joint_expon_w1 = self.method_class(**{**self.kwargs_construct_expon_joint, 'f_theta': f_theta_works_v1})
        self.intergator_joint_expon_w2 = self.method_class(**{**self.kwargs_construct_expon_joint, 'f_theta': f_theta_works_v2})
        self.intergator_joint_expon_f1 = self.method_class(**{**self.kwargs_construct_expon_joint, 'f_theta': f_theta_fails_v1})
        self.intergator_joint_expon_f2 = self.method_class(**{**self.kwargs_construct_expon_joint, 'f_theta': f_theta_fails_v2})

    def _test_cond_construct_expon(self):
        """Should be able to return the constructed class"""
        self.intergator_cond_expon_w1 = self.method_class(**{**self.kwargs_construct_expon_cond, 'f_theta': f_theta_works_v1})
        self.intergator_cond_expon_w2 = self.method_class(**{**self.kwargs_construct_expon_cond, 'f_theta': f_theta_works_v2})
        self.intergator_cond_expon_f1 = self.method_class(**{**self.kwargs_construct_expon_cond, 'f_theta': f_theta_fails_v1})
        self.intergator_cond_expon_f2 = self.method_class(**{**self.kwargs_construct_expon_cond, 'f_theta': f_theta_fails_v2})

    def _test_joint_integrate_expon(self):
        """Make sure we can call the integrate method for the correct f_theta's, but not but the wrong f_theta's"""
        risk_w1 = self.intergator_joint_expon_w1.integrate(**self.kwargs_mu_seed)
        _, lossvar_w1 = self.intergator_joint_expon_w1.integrate(**self.kwargs_var_seed)
        np.testing.assert_equal(risk_w1, _)
        risk_w2 = self.intergator_joint_expon_w2.integrate(**self.kwargs_mu_seed)
        _, lossvar_w2 = self.intergator_joint_expon_w2.integrate(**self.kwargs_var_seed)
        np.testing.assert_equal(risk_w2, _)


    def _test_missing_distribution(self):
        """Method construction needs at least one distribution"""
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_works)

    def _test_joint_construct_gauss(self):
        """When the joint is provided the conditional/unconditional should be None"""
        intergator_joint_gauss = self.method_class(loss=loss_fun_works, dist_joint=dist_BVN)
        assert intergator_joint_gauss.dist_X_uncond is None
        assert intergator_joint_gauss.dist_Y_condX is None
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_fails, dist_joint=dist_BVN)
        return intergator_joint_gauss

    def _test_cond_construct_gauss(self):
        """When the conditional/unconditional is provided the joint should be None"""
        intergator_cond_gauss = self.method_class(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
        assert intergator_cond_gauss.dist_joint is None
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_fails, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_Ycond)
        return intergator_cond_gauss

    def _test_invalid_cond_dist_gauss(self):
        """You should not be able to assign Y_condX that isn't a callable"""
        with pytest.raises(AssertionError):
            self.method_class(loss=loss_fun_works, dist_X_uncond=dist_X_uni, dist_Y_condX=dist_X_uni)

    def _test_integration_gauss(self, integrator):
        """
        Call the integrate method, and the variance flag should return a tuple, and the risk (mena) should be the same
        """
        mu1a = integrator.integrate(**self.kwargs_mu_seed)
        mu1b, var1b = integrator.integrate(**self.kwargs_var_seed)
        assert isinstance(mu1a, float)
        assert mu1a == mu1b
        return mu1a, var1b