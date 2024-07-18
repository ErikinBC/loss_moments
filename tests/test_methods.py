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
from . import di_dists_expon, di_f_theta
from . import di_kwargs_integrate_method, di_methods

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
        
        # (ii) Test Gaussian (univariate) results
        intergator_joint_gauss = self._test_joint_construct_gauss()
        intergator_cond_gauss = self._test_cond_construct_gauss()
        # Test integrals
        mu1b, var1b = self._test_integration_gauss(intergator_joint_gauss)
        mu2b, var2b = self._test_integration_gauss(intergator_cond_gauss)
        # Test equivalence between conditional/unconditional vs joint
        np.testing.assert_allclose(mu1b, mu2b, atol=atol)
        np.testing.assert_allclose(var1b, var2b, atol=atol)

        # (iii) Test Non-Gaussian (multivariate_ results)
        # Determine the joint and conditional distributions
        dists_expon = di_dists_expon[method_name]
        self.kwargs_construct_expon_joint = {'loss':loss_fun_works, 'dist_joint': dists_expon['dist_joint']}
        self.kwargs_construct_expon_cond = {'loss':loss_fun_works, 'dist_X_uncond': dists_expon['dist_X_uncond'], 'dist_Y_condX': dists_expon['dist_Y_condX']}
        # Determine the working and failing f_theta functions
        working_f_theta = di_f_theta[method_name]['works']
        failing_f_theta = di_f_theta[method_name]['fails']
        lst_f_theta = working_f_theta + failing_f_theta
        # Loop over the working f_theta's
        for f_theta in lst_f_theta:
            # Set up to the integrators
            integrator_joint = self._test_joint_construct_expon(f_theta)
            integrator_cond = self._test_cond_construct_expon(f_theta)
            if f_theta in working_f_theta:  # Should execute
                self._test_integrate_expon(integrator_joint)
                self._test_integrate_expon(integrator_cond)
            else:  # Should fail
                with pytest.raises(ValueError):
                        self._test_integrate_expon(integrator_joint)
        # Check the integration call
        print(f'\nPassed all tests for {method_name}\n')

    # --- GAUSSIAN TESTERS --- #
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

    # --- NON-GAUSSIAN TESTERS --- #    
    def _test_joint_construct_expon(self, f_theta):
        """Should be able to return the constructed classes"""
        return self.method_class(**{**self.kwargs_construct_expon_joint, 'f_theta': f_theta})

    def _test_cond_construct_expon(self, f_theta):
        """Should be able to return the constructed class"""
        return self.method_class(**{**self.kwargs_construct_expon_cond, 'f_theta': f_theta})

    def _test_integrate_expon(self, integrator):
        """Make sure we can call the integrate method for the correct f_theta's, but not but the wrong f_theta's"""
        risk = integrator.integrate(**self.kwargs_mu_seed)
        _, lossvar = integrator.integrate(**self.kwargs_var_seed)
        assert isinstance(lossvar, float)
        np.testing.assert_equal(risk, _)