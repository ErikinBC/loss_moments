"""
Contains the two main classes of this package: 
1. MonteCarloIntegrator
2. NumericalIntegrator
"""

# External modules
import numpy as np
from scipy.integrate import dblquad
from scipy.stats import multivariate_normal
from typing import Tuple, Union, Callable, Any
from scipy.stats._distn_infrastructure import rv_continuous_frozen
from scipy.stats._multivariate import multivariate_normal_frozen
# Intenral modules
from ._base import BaseIntegrator

class MonteCarloIntegrator(BaseIntegrator):
    def __init__(self, 
                 loss: Callable[..., Any], 
                 dist_joint: multivariate_normal_frozen | None = None, 
                 dist_X_uncond: rv_continuous_frozen | None = None, 
                 dist_Y_condX: Callable[..., Any] | None = None,
                 f_theta: Union[Callable, None] = None, 
                 ):
        """Initialize the Monte Carlo Integration class. See _base.BaseIntegrator for constructor argument details."""
        super().__init__(loss=loss, dist_joint=dist_joint, 
                         dist_X_uncond=dist_X_uncond, dist_Y_condX=dist_Y_condX, 
                         f_theta=f_theta)
        
    def _draw_samples(self, num_samples: int, seed: int | None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal method for generating labels and features

        If has_joint:
            (y_i, x_i) ~ F_{Y, X}
        If ~has_joint:
            X_i \sim F_X
            Y_i | X_i \sim F_{Y | X}
        """
        if self.has_joint:
            yx_samples = self.dist_joint.rvs(num_samples, random_state=seed).T
            y_samples = yx_samples[0]
            x_samples = np.squeeze(yx_samples[1:].T)
        else:
            x_samples = self.dist_X_uncond.rvs(num_samples, random_state=seed)
            y_samples = self.dist_Y_condX(x_samples).rvs(num_samples, random_state=seed+1)
        return y_samples, x_samples

    def _integrate(self, num_samples, calc_variance, seed):
        """
        Internal method for calculating self.integrate() for an arbitrary number of chunks
        """
        # Draw samples with the pre-defines method
        y_samples, x_samples = self._draw_samples(num_samples, seed)
        # Calculate the losses
        losses = self._loss_f_theta(y=y_samples, x=x_samples)
        res = np.mean(losses)
        if calc_variance:
            # Add on variance if requested
            var = np.var(losses, ddof=1)
            res = (res, var)
        return res

    @staticmethod
    def _seed_increment(seed: int | None, increment: int):
        """Try to increment a seed by a fixed amount or return None"""
        if seed is None:
            return seed
        else:
            return seed + increment


    def integrate(self, 
                  num_samples : int, 
                  calc_variance : bool = False, 
                  n_chunks : int = 1,
                  seed : int | None = None
                  ) -> Union[float, Tuple[float, float], np.ndarray]:
        """
        Compute the integral using Monte Carlo integration.

        Parameters
        ----------
        num_samples : int
            How many samples to draw: dist_joint.rvs(num_samples, ...)?
        calc_variance : bool, optional
            Should the variance be returned (in addition to the mean)?
        n_chunks : int, optional
            Should the Monte Carlo sampling be repeated n_chunks times?
        seed: int | None, optional
            Reproducibility seed
        
            
        Returns
        -------
        float or tuple or np.ndarray
            If ~calc_variance returns \hat{R}
            If calc_variance, then returns (\hat{R}, \hat{V})
            If n_chunks > 0, returns either a 
            \hat{R} = mean([loss(y_1, x_1), ..., loss(y_{num_samples}, x_{num_samples})])
            \hat{V} = var([loss(y_1, x_1), ..., loss(y_{num_samples}, x_{num_samples})])
        """
        # Input checks
        assert n_chunks >= 1, f'n_chunks needs to be >=1, not {n_chunks}'
        # Run calculations
        holder = []
        for chunk in range(1,n_chunks+1):
            seed_chunk = self._seed_increment(seed,chunk-1)
            tmp_res = self._integrate(num_samples, calc_variance, seed=seed_chunk)
            holder.append(tmp_res)
        res = np.vstack(holder).mean(axis=0)
        res = self._return_tuple_or_float(res)
        return res



class NumericalIntegrator(BaseIntegrator):
    def __init__(self, 
                loss: Callable[..., Any], 
                dist_joint: multivariate_normal_frozen | None = None, 
                dist_X_uncond: rv_continuous_frozen | None = None, 
                dist_Y_condX: Callable[..., Any] | None = None,
                f_theta: Union[Callable, None] = None,
                ):
        """Initialize the Numerical Integration class. See utils.BaseIntegrator for constructor arguments."""
        super().__init__(loss=loss, dist_joint=dist_joint, 
                         dist_X_uncond=dist_X_uncond, dist_Y_condX=dist_Y_condX, 
                         f_theta=f_theta)
        # Assign the dictionary/methods that will get called later
        self.di_methods = {
                    'trapz_loop': 
                        {'method':self._trapz_integrate, 
                         'kwargs': {'use_grid': False}},
                    'trapz_grid': 
                        {'method': self._trapz_integrate,
                         'kwargs': {'use_grid': True}},
                    'quadrature': 
                        {'method': self._quad_integrate,
                         'kwargs': {}},
                     }
        self.valid_methods = list(self.di_methods)
        self.di_grid_methods = {
                        True: self._trapz_integrate_grid, 
                        False: self._trapz_integrate_loop
                        }
        

    def _gen_bvn_bounds(self,
                        k_sd : int,
                        ) -> Tuple[float, float, float, float]:
        """
        Generates BVN bounds of integration for BVN([mu1, mu2], [[var1, rho*sd1*sd2], [rho*sd1*sd2, var2]]).
        
        Specifically the y_bounds are from mu1 +/- sd1*k_sd, and the x_bounds are from mu2 +/- sd2*k_sd

        Returns
        -------
        tuple
            (y_min, y_max, x_min, x_max)
        """
        if self.has_joint:
            # Should already be a MVN, easy to extract
            yx_bounds = np.atleast_2d(self.dist_joint.mean) + np.tile([-1, 1], [2, 1]).T * k_sd * np.sqrt(np.diag(self.dist_joint.cov))
            y_min, y_max = yx_bounds.T[0]
            x_min, x_max = yx_bounds.T[1]
        else:
            # Need to use the formula of a conditional distribution to extract the joint
            # We can use simple algebra to solve (mu_Y, sigma_Y) from Y | X = x ~ norm(loc=mu_Y + rho*(sigma2_Y/sigma2_X)**0.5*(x - mu_X), scale=np.sqrt(sigma2_Y * (1-rho**2)))
            mu_X = self.dist_X_uncond.mean()
            sigma_X = self.dist_X_uncond.std()
            mu_Y = self.dist_Y_condX(mu_X).mean()
            sigmaY_rho = sigma_X*(self.dist_Y_condX(1 + mu_X).mean() - mu_Y)
            c = self.dist_Y_condX(mu_X).var() / sigmaY_rho**2
            rho = np.sqrt(1 / (c + 1))
            sigma_Y = sigmaY_rho / rho
            mu = np.array([mu_Y, mu_X])
            off_diag = rho * sigma_X * sigma_Y
            cov = np.array([[sigma_Y ** 2, off_diag], [off_diag, sigma_X ** 2]])
            dist_joint = multivariate_normal(mean=mu, cov=cov)
            yx_bounds = np.atleast_2d(dist_joint.mean) + np.tile([-1, 1], [2, 1]).T * k_sd * np.sqrt(np.diag(dist_joint.cov))
        y_min, y_max = yx_bounds.T[0]
        x_min, x_max = yx_bounds.T[1]
        return y_min, y_max, x_min, x_max
        
    def _integrand_for_quad(self, y, x) -> float | np.ndarray:
        """Compute the integrand for joint integration."""
        loss_values = self._loss_f_theta(y=y, x=x)
        density = self.dist_joint.pdf([y, x])
        return loss_values * density

    def _integrand2_for_quad(self, y, x) -> float | np.ndarray:
        """Compute the integrand for joint integration (variance calculation)."""
        loss_values = self._loss_f_theta(y=y, x=x) ** 2
        density = self.dist_joint.pdf([y, x])
        return loss_values * density

    def _quad_integrate(self, 
                        x_min: float, x_max: float, 
                        y_min: float, y_max: float, 
                        calc_variance: bool,
                        sol_tol: float,
                        ) -> float | Tuple[float, float]:
        """
        Internal wrapper for used the double quadrature method to estimate the risk (and variance)
        """
        # Calculate the risk
        risk_var, _ = dblquad(self._integrand_for_quad, x_min, x_max, y_min, y_max,
                            epsabs=sol_tol, epsrel=sol_tol)
        if calc_variance:
            risk2_var, _ = dblquad(self._integrand2_for_quad, x_min, x_max, y_min, y_max,
                                epsabs=sol_tol, epsrel=sol_tol)
            loss_var = risk2_var - risk_var**2
            risk_var = (risk_var, loss_var)
        res = self._return_tuple_or_float(risk_var)
        return res

    def _trapz_integrate(self,
                         yvals: np.ndarray, xvals: np.ndarray, 
                         calc_variance: bool, 
                         use_grid: bool,
                         ) -> float | Tuple[float, float]:
        """
        Internal wrapper to calculate either grid or non-grid based integration.
        """
        # Let the named argument {use_grid} determine the internal method
        # Store function arguments
        di_args = {'yvals': yvals, 'xvals': xvals}
        # Calculate the risk
        risk_var = self.di_grid_methods[use_grid](**di_args, power=1)
        if calc_variance:
            # Add on the loss variance if it's requested
            risk2_var = self.di_grid_methods[use_grid](**di_args, power=2)
            loss_var = risk2_var - risk_var**2
            risk_var = (risk_var, loss_var)
        res = self._return_tuple_or_float(risk_var)
        return res
        
    def _trapz_integrate_grid(self, 
                              yvals: np.ndarray, xvals: np.ndarray, 
                              power: int = 1
                              ) -> float:
        """Perform trapezoidal integration using the meshgrid. Needs to store several (n_X, n_Y) arrays"""
        # Get the grid of values
        Yvals, Xvals = np.meshgrid(yvals, xvals)
        # Get the n_Y by n_X loss
        loss_values = np.power(self._loss_f_theta(Yvals, Xvals), power)
        if self.has_joint:
            # For the joint distribution, the inner integral is the same thing as the outer intergrand
            density_inner = self.dist_joint.pdf(np.dstack((Yvals, Xvals)))
            inner_integrand = loss_values * density_inner
            outer_integrand = np.trapz(inner_integrand, yvals, axis=1)
        else:
            # For the conditional distribution, the outer integrand needs to weighted by the unconditional X dist
            density_inner = self.dist_Y_condX(Xvals).pdf(Yvals)
            inner_integrand = loss_values * density_inner
            inner_integral = np.trapz(inner_integrand, yvals, axis=1)
            outer_integrand = inner_integral * self.dist_X_uncond.pdf(xvals)
        # Get the final integral (float)
        outer_integral = np.trapz(outer_integrand, xvals)
        return outer_integral
        
    def _trapz_integrate_loop(self, 
                              yvals: np.ndarray, xvals: np.ndarray, 
                              power: int = 1
                              ) -> float:
            """Perform trapezoidal integration for a given value at a time. Only needs to store an evaluate arrays of size (n_X,) or (n_Y,)"""
            # Holder for the n_X inner integral values
            inner_integral = np.zeros(xvals.shape[0])
            if self.has_joint:
                # For joint distribution, the inner integral is weighted by the joint PDF, and this is equivalent to the outer integrand
                for i, x_i in enumerate(xvals):
                    inner_integrand_i = np.power(self._loss_f_theta(yvals, x_i), power)
                    points_i = np.c_[yvals, np.broadcast_to(x_i, yvals.shape)]
                    inner_integrand_i *= self.dist_joint.pdf(points_i)
                    inner_integral[i] = np.trapz(inner_integrand_i, yvals)
                # The outer integral is simply the integral over the feature space
                outer_integral = np.trapz(inner_integral, xvals)
            else:
                # For the conditional distribution, the inner integral is weighted by the conditional distribution
                for i, x_i in enumerate(xvals):
                    inner_integrand_i = np.power(self._loss_f_theta(yvals, x_i), power)
                    inner_integrand_i *= self.dist_Y_condX(x_i).pdf(yvals)
                    inner_integral[i] = np.trapz(inner_integrand_i, yvals)
                # The outer integrand is the inner integral weighted by the feature space
                outer_integrand = inner_integral * self.dist_X_uncond.pdf(xvals)
                outer_integral = np.trapz(outer_integrand, xvals)
            return outer_integral


    def integrate(self,
                method: str = 'trapz_loop', 
                calc_variance: bool = False,
                k_sd: int = 4,
                n_Y: Union[int, np.ndarray] = 100,
                n_X: Union[int, np.ndarray] = 101,
                sol_tol: float = 1e-3,
                ) -> Union[float, Tuple[float, float]]:
        """
        Calculates the integral of a l(y,x) assuming (y,x) ~ BVN, using a double integral approach and the trapezoidal rule. 

        Parameters
        ----------
        method : str, optional
            Which method to use? Can be either ['trapz_loop', 'trapz_grid', 'quadratrure']. Defaults to the first.
        calc_variance : bool, optional
            Should the variance be calculated?
        k_sd : int, optional
            How many standard deviations away from the mean to perform the grid search over?
        n_{Y,X} : int | np.ndarray, optional
            The number of points along the Y, X direction to generate (equal spacing). If an array is provided, will assume these are the points to use for integration
        sol_tol : float, optional
            The values of `epsabs` & `epsrel` to pass into `scipy.integrate.dblquad`

    
        Description
        -----------
        For the joint method:
        1. We are solving the I_{YX} = \int_Y \int_X l(y,x) f_{YX}(y,x) dx dy. 
        2. For each x_i in {x_1, ..., x_{n_X}}, calculate the inner integral: I_{inner}(x_i) = \int_Y l(y,X=x_i) f_{YX}(y,X=x_i) dy. This gets back an (n_X,) length arrary. 
        3. Calculate the outer integral: I_{outer} = \int_X I_{inner}(x) dx. This is equivalent to the risk.

        For he conditional method
        1.  We are solving the I_{YX} = \int_Y \int_X l(y,x) f_{YX}(y,x) dx dy = \int_X [\int_Y l(y, X=x) f_{Y|X}(y,X=x) dy] f_X(x) dx.
        2. For each x_i in {x_1, ..., x_{n_X}}, calculate the inner integral: I_{inner}(x_i) = \int_Y l(y,X=x_i) f_{Y|X}(y,X=x_i) dy. This gets back an (n_X,) length arrary. 
        3. Calculate the outer integral: I_{outer} = \int_X I_{inner}(x) f_X(x) dx.  This is equivalent to the risk.

        To calculate the variance, we simply square the loss function, and then do: I^2_{YX} - [I_{YX}]^2

        Returns
        -------
        float or Tuple[float, float]
            Integrated value, optionally with variance.
        """
        # Input checks
        assert method in self.valid_methods, f'method must be one {self.valid_methods}, not {method}'
        # Calculate the limits of integration
        y_min, y_max, x_min, x_max = self._gen_bvn_bounds(k_sd=k_sd)
        yvals = np.linspace(y_min, y_max, n_Y)
        xvals = np.linspace(x_min, x_max, n_X)
        # Prepare all possible sharable arguments
        di_args = {
                'calc_variance':calc_variance,
                'x_min':x_min, 'x_max':x_max,
                'y_min':y_min, 'y_max':y_max,
                'xvals': xvals, 'yvals': yvals,
                'sol_tol':sol_tol,
                }
        # Select the integration method
        func_method = self.di_methods[method]['method']
        di_args = self._subset_args(di_args, func_method)
        # Add on any special method kwargs
        di_args = {**di_args, **self.di_methods[method]['kwargs']}
        # Call method and return
        res = func_method(**di_args)
        return res
