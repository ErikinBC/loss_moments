"""
Contains examples of stlyzed distributions for unit testing and demonstrations
"""

import numpy as np
from typing import Tuple
   
# Use the MSE as the loss function
def squared_error(y, x):
    return np.power(y - x, 2)

def empirical_MSE_risk_lossvar(y, x) -> Tuple[float, float]:
    errors = squared_error(y, x)
    risk = np.mean(errors)
    lossvar = np.var(errors, ddof=1)
    return risk, lossvar


def simulation(
        n_train: int=30, 
        n_oos: int=10000, 
        p: int = 10, 
        verbose: bool = True,
        seed: int | None = 1
         ):
    """
    Call the main methods
    """
    # Packages and modules
    import numpy as np
    from scipy.stats import norm
    from ._gaussian import dgp_yx, linear_predictor
    from .methods import MonteCarloIntegrator, NumericalIntegrator

    # (i) Generate some training data
    generator = dgp_yx(p=p)
    y, x = generator.gen_yX(n=n_train)
    # "Learn" the coefficients (oracle, OLS, and random)
    beta_oracle = generator.beta
    theta_hat = np.dot((np.linalg.inv(np.dot(x.T,x))), np.dot(x.T, y))
    gamma_hat = theta_hat[np.argsort(norm().rvs(p, random_state=seed))]
    # Set up the predictors
    mdl_oracle = linear_predictor(beta_oracle)
    mdl_theta = linear_predictor(theta_hat)
    mdl_gamma = linear_predictor(gamma_hat)

    # (ii) Calculate the theoretical (oracle) value of a given coeffient
    theory_risk_oracle, theory_lossvar_oracle = generator.get_risk_lossvar(beta_oracle)
    theory_risk_theta, theory_lossvar_theta = generator.get_risk_lossvar(theta_hat)
    theory_risk_gamma, theory_lossvar_gamma = generator.get_risk_lossvar(gamma_hat)

    # (iii) Calculate empirical risk and loss variance for the MSE loss function
    y_oos, eta_oracle = generator.gen_yX(n_oos, beta_oracle)
    emp_risk_oracle, emp_lossvar_oracle = empirical_MSE_risk_lossvar(y_oos, eta_oracle)
    _, eta_theta = generator.gen_yX(n_oos, theta_hat)
    emp_risk_theta, emp_lossvar_theta = empirical_MSE_risk_lossvar(y_oos, eta_theta)
    np.testing.assert_equal(y_oos, _)
    _, eta_gamma = generator.gen_yX(n_oos, gamma_hat)
    emp_risk_gamma, emp_lossvar_gamma = empirical_MSE_risk_lossvar(y_oos, eta_gamma)
    
    # (iv) Set the MCI integral functions for:
    #   a) Integration approaches (joint vs cond/uncond)
    #   b) Algorithms (oracle, theta, gamma)
    kwargs_joint = {'loss':squared_error, 'dist_joint':generator.dist_yx}
    kwargs_cond = {'loss':squared_error, 'dist_X_uncond':generator.dist_x, 'dist_Y_condX':generator.dist_ycond}
    mci_joint_oracle = MonteCarloIntegrator(**{**kwargs_joint, **{'f_theta':mdl_oracle}})
    mci_cond_oracle = MonteCarloIntegrator(**{**kwargs_cond, **{'f_theta':mdl_oracle}})
    mci_joint_theta = MonteCarloIntegrator(**{**kwargs_joint, **{'f_theta':mdl_theta}})
    mci_cond_theta = MonteCarloIntegrator(**{**kwargs_cond, **{'f_theta':mdl_theta}})
    mci_joint_gamma = MonteCarloIntegrator(**{**kwargs_joint, **{'f_theta':mdl_gamma}})
    mci_cond_gamma = MonteCarloIntegrator(**{**kwargs_cond, **{'f_theta':mdl_gamma}})
    # Call the monte carlo integration
    kwargs_method_integrate = {'num_samples':10000, 'calc_variance':True, 'seed':seed, 'n_chunks':10}
    mci_joint_risk_oracle, mci_joint_lossvar_oracle = mci_joint_oracle.integrate(**kwargs_method_integrate)
    mci_cond_risk_oracle, mci_cond_lossvar_oracle = mci_cond_oracle.integrate(**kwargs_method_integrate)
    mci_joint_risk_theta, mci_joint_lossvar_theta = mci_joint_theta.integrate(**kwargs_method_integrate)
    mci_cond_risk_theta, mci_cond_lossvar_theta = mci_cond_theta.integrate(**kwargs_method_integrate)
    mci_joint_risk_gamma, mci_joint_lossvar_gamma = mci_joint_gamma.integrate(**kwargs_method_integrate)
    mci_cond_risk_gamma, mci_cond_lossvar_gamma = mci_cond_gamma.integrate(**kwargs_method_integrate)
    # print(f'Risk ~ Oracle: Empirical={emp_risk_oracle:.2f}, theory={theory_risk_oracle:.2f}, joint={mci_joint_risk_oracle:.2f}, conditional={mci_cond_risk_oracle:.2f}')
    # print(f'Loss Variance ~ Oracle: Empirical={emp_lossvar_oracle:.2f}, theory={theory_lossvar_oracle:.2f}, joint={mci_joint_lossvar_oracle:.2f}, conditonal={mci_cond_lossvar_oracle:.2f}')

    # (v) Print results
    if verbose:
        print('\n--- Risk ---')
        print(f'oracle: Empirical={emp_risk_oracle:.2f}, theory={theory_risk_oracle:.2f}')
        print(f'theta: Empirical={emp_risk_theta:.2f}, theory={theory_risk_theta:.2f}, integral={0:.2f}')
        print(f'gamma: Empirical={emp_risk_gamma:.2f}, theory={theory_risk_gamma:.2f}, integral={0:.2f}')

        print('\n--- Loss variance ---')
        print(f'oracle: Empirical={emp_lossvar_oracle:.2f}, theory={theory_lossvar_oracle:.2f}')
        print(f'theta: Empirical={emp_lossvar_theta:.2f}, theory={theory_lossvar_theta:.2f}, integral={0:.2f}')
        print(f'gamma: Empirical={emp_lossvar_gamma:.2f}, theory={theory_lossvar_gamma:.2f}, integral={0:.2f}')

    # Return values (for unit testing)
    di_ret = {
                'oracle':
                    {
                        'risk':[emp_risk_oracle, theory_risk_oracle, 
                                mci_joint_risk_oracle, mci_cond_risk_oracle, ],   
                        'lossvar': [emp_lossvar_oracle, theory_lossvar_oracle, 
                                    mci_joint_lossvar_oracle, mci_cond_lossvar_oracle],
                    },
                'theta':
                    {
                        'risk':[emp_risk_theta, theory_risk_theta, 
                                mci_joint_risk_theta, mci_cond_risk_theta,],
                        'lossvar': [emp_lossvar_theta, theory_lossvar_theta, 
                                    mci_joint_lossvar_theta, mci_cond_lossvar_theta, ], 
                    },
                'gamma':
                    {
                        'risk':[emp_risk_gamma, theory_risk_gamma, 
                                mci_joint_risk_gamma, mci_cond_risk_gamma, ],
                        'lossvar': [emp_lossvar_gamma, theory_lossvar_gamma, 
                                    mci_joint_lossvar_gamma, mci_cond_lossvar_gamma, ], 
                    },
             }
    return di_ret