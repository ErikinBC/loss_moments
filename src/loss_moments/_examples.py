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
    di_mdl = {'oracle': {'coef': beta_oracle, 'mdl': mdl_oracle},
              'theta': {'coef': theta_hat, 'mdl': mdl_theta}, 
              'gamma': {'coef': gamma_hat, 'mdl':mdl_gamma}}
    
    # (ii) Prepare to loop over all models (and methods for the integrators)
    kwargs_joint_mci = {'loss':squared_error, 'dist_joint':generator.dist_yx}
    kwargs_cond_mci = {'loss':squared_error, 'dist_X_uncond':generator.dist_x, 'dist_Y_condX':generator.dist_ycondx}
    di_kwargs = {'joint': kwargs_joint_mci, 
                 'cond': kwargs_cond_mci}
    # Argumetns for integrate method
    kwargs_method_integrate = {'num_samples':10000, 'calc_variance':True, 'seed':seed, 'n_chunks':10}
    # Set up storage
    di_ret = {k: {'risk':[], 'lossvar':[]} for k in di_mdl.keys()}

    # (iii) Loop over methods
    for model, predictor in di_mdl.items():
        print(f'~~~ model={model} ~~~')
        coef_mdl = predictor['coef']
        # (a) Calculate the theoretical (oracle) value of a given coeffient
        theory_risk, theory_lossvar = generator.get_risk_lossvar(coef_mdl)
        # (b) Calculate the out-of-sample (OOS) loss
        y_oos, eta_oos = generator.gen_yX(n_oos, coef_mdl)
        emp_risk, emp_lossvar = empirical_MSE_risk_lossvar(y_oos, eta_oos)
        del y_oos, eta_oos
        # Append results
        di_ret[model]['risk'] += [theory_risk, emp_risk, ]
        di_ret[model]['lossvar'] += [theory_lossvar, emp_lossvar, ]
        for approach, kwargs in di_kwargs.items():    
            print(f'-- approach = {approach} --')
            # (c) Monte carlo integration
            approach_mdl = predictor['mdl']
            mci_integrator = MonteCarloIntegrator(**{**kwargs, **{'f_theta':approach_mdl}})
            mci_risk,  mci_lossvar = mci_integrator.integrate(**kwargs_method_integrate)
            # (d) Numerical integration

            # Append results
            di_ret[model]['risk'] += [mci_risk]
            di_ret[model]['lossvar'] += [mci_lossvar]

    # # (v) For numerical integration, x has to be uni-dimensional, which in this case means we need to know the distribution of (y, f_theta), f_theta, and y | f_theta. I'll denote f_theta as eta for simplicity.
    # dist_y_eta, dist_eta, dist_ycond_eta = generator.gen_f_theta_dists(theta=beta_oracle)
    # kwargs_joint_numint = {'loss':squared_error, 'dist_joint':...}
    # kwargs_cond_numint = {'loss':squared_error, 'dist_X_uncond':..., 'dist_Y_condX':...}
    # numint_joint_oracle = NumericalIntegrator()
    

    # (vi) Print results (optional)
    if verbose:
        for method, res in di_ret.items():
            print(f'--- Coefficient = {method} ---')
            print([f'{r:.2f}' for r in res['risk']])

    return di_ret