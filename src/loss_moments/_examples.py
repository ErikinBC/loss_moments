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
    di_mdl = {'oracle': {'coef': beta_oracle, 
                         'mdl': {'mci':mdl_oracle, 
                                 'numint': None}},
              'theta': {'coef': theta_hat, 
                        'mdl': {'mci': mdl_theta,
                                'numint': None}}, 
              'gamma': {'coef': gamma_hat, 
                        'mdl':{'mci': mdl_gamma, 
                               'numint': None}}}
    
    # (ii) Prepare to loop over all models (and methods for the integrators)
    di_methods =  {
                    'mci': MonteCarloIntegrator,
                    # 'numint': NumericalIntegrator,
                  }
    kwargs_joint_mci = {'loss':squared_error, 'dist_joint':generator.dist_yx}
    kwargs_cond_mci = {'loss':squared_error, 'dist_X_uncond':generator.dist_x, 'dist_Y_condX':generator.dist_ycondx}
    dist_y_eta, dist_eta, dist_ycond_eta = generator.gen_f_theta_dists(theta=gamma_hat)
    # For numerical integration, x has to be uni-dimensional, which in this case means we need to know the distribution of (y, f_theta), f_theta, and y | f_theta
    kwargs_joint_numint = {'loss':squared_error, 'dist_joint':dist_y_eta}
    kwargs_cond_numint = {'loss':squared_error, 'dist_X_uncond':dist_eta, 'dist_Y_condX':dist_ycond_eta}
    di_kwargs_construct = {
                            'mci': {'joint': kwargs_joint_mci, 'cond': kwargs_cond_mci},
                            'numint': {'joint': kwargs_joint_numint, 'cond': kwargs_cond_numint}
                          }
    # Argumetns for integrate method
    kwargs_integrate_mci = {'num_samples':10000, 'calc_variance':True, 'seed':seed, 'n_chunks':10}
    kwargs_integrate_numint = {'method': 'trapz_loop', 'k_sd':4.5, 'n_Y':200, 'n_X':201, 'calc_variance': True}
    di_kwargs_integrate = {'mci': kwargs_integrate_mci, 
                           'numint': kwargs_integrate_numint}

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
        # Loop over our integration methods (e.g. MonteCarloIntegrator)
        for method, integrator in di_methods.items():
            print(f'Integration method: {method}')
            # Extract the kwargs
            kwargs_construct = di_kwargs_construct[method]
            kwargs_integrate = di_kwargs_integrate[method]
            for approach in kwargs_construct.keys():
                print(f'Approach = {approach}')
                kwargs_construct_approach = kwargs_construct[approach]
                # Add on model predictor as f_theta
                kwargs_construct_approach = {**kwargs_construct_approach, **{'f_theta':predictor['mdl'][method]}}
                # Set up integrator
                method_integrator = integrator(**kwargs_construct_approach)
                # Run the integration
                integrator_risk,  integrator_lossvar = method_integrator.integrate(**kwargs_integrate)
                # Append results
                di_ret[model]['risk'].append(integrator_risk)
                di_ret[model]['lossvar'].append(integrator_lossvar)
    
    # (vi) Print results (optional)
    if verbose:
        for method, res in di_ret.items():
            print(f'--- Coefficient = {method} ---')
            print([f'{r:.2f}' for r in res['risk']])
    return di_ret