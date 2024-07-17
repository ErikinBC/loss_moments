"""
Contains examples of stlyzed distributions for unit testing and demonstrations
"""

import numpy as np
from typing import Tuple, Callable
   
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
        seed: int | None = 1, 
        loss_fun: Callable = squared_error,
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
    di_coef = {'oracle': beta_oracle, 
               'theta': theta_hat, 
               'gamma': gamma_hat}
    # (ii) Set up the predictors
    mdl_oracle = linear_predictor(beta_oracle)
    mdl_theta = linear_predictor(theta_hat)
    mdl_gamma = linear_predictor(gamma_hat)
    di_mdl = {'oracle': mdl_oracle, 
              'theta': mdl_theta, 
              'gamma': mdl_gamma}
    keys_mdl = list(di_mdl.keys())
    # (iii) Set up the different methods
    di_methods =  {
                'mci': MonteCarloIntegrator,
                'numint': NumericalIntegrator,
                }
    keys_method = list(di_methods.keys())    
    # (iv) When constructing the Integrators, MonteCarlo needs a different f_theta for each model, whereas NumInt needs a different RV distriution (i.e. (Y, Eta) ~ P) 
    kwargs_construct_mci = {'loss':loss_fun}
    kwargs_construct_joint_mci = {**kwargs_construct_mci, **{'dist_joint':generator.dist_yx}}
    kwargs_construct_cond_mci = {**kwargs_construct_mci, **{'dist_X_uncond':generator.dist_x, 'dist_Y_condX':generator.dist_ycondx}}
    di_kwargs_construct_mci = {'joint': kwargs_construct_joint_mci, 'cond': kwargs_construct_cond_mci}
    keys_approach = list(di_kwargs_construct_mci.keys())
    # fix if "MCI'"
    di_kwargs_construct = {k1: {k2: di_kwargs_construct_mci if k1 == 'mci' else dict.fromkeys(di_kwargs_construct_mci) for k2 in di_coef.keys()} for k1 in di_methods.keys()}
    # loop over each model
    for mdl in keys_mdl:
        # for monte carlo, set the f_theta
        di_kwargs_construct['mci'][mdl] = {k: {**{'f_theta': di_mdl[mdl]}, **v} for k, v in di_kwargs_construct['mci'][mdl].items()}
        # for numerical methods, set the distribution
        dist_y_eta, dist_eta, dist_ycond_eta = generator.gen_f_theta_dists(theta=di_coef[mdl])
        di_kwargs_construct['numint'][mdl]['joint'] = {'loss':loss_fun, 'dist_joint':dist_y_eta}
        di_kwargs_construct['numint'][mdl]['cond'] = {'loss':loss_fun, 'dist_X_uncond':dist_eta, 'dist_Y_condX':dist_ycond_eta}
    # (vi) Set up the different integrate arguments (only varies by method)
    kwargs_integrate_mci = {'num_samples':10000, 'calc_variance':True, 'seed':seed, 'n_chunks':10}
    kwargs_integrate_numint = {'method': 'trapz_loop', 'k_sd':4.5, 'n_Y':200, 'n_X':201, 'calc_variance': True}
    di_kwargs_integrate = {'mci': kwargs_integrate_mci, 'numint': kwargs_integrate_numint}

    # Set up storage
    di_ret = {k: {'risk':[], 'lossvar':[]} for k in di_coef.keys()}
    # (iii) Loop over methods
    for model in keys_mdl:
        print(f'~~~ model={model} ~~~')
        coef_mdl = di_coef[model]
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
        for method in keys_method:
            print(f'Integration method: {method}')
            kwargs_integrate = di_kwargs_integrate[method]
            for approach in keys_approach:
                print(f'Approach = {approach}')
                # Extract the kwargs
                kwargs_construct = di_kwargs_construct[method][model][approach]
                # Set up integrator
                integrator = di_methods[method](**kwargs_construct)
                # Run the integration
                integrator_risk, integrator_lossvar = integrator.integrate(**kwargs_integrate)
                # Append results
                di_ret[model]['risk'].append(integrator_risk)
                di_ret[model]['lossvar'].append(integrator_lossvar)
    
    # (vi) Print results (optional)
    if verbose:
        for method, res in di_ret.items():
            print(f'--- Coefficient = {method} ---')
            print([f'{r:.2f}' for r in res['risk']])
    return di_ret
