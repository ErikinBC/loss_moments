"""
Makes sure that the theory used in the main module is correct

python3 -m pytest tests/test_dists.py -W ignore -s
"""

import pytest
import numpy as np
from src.loss_moments._examples import simulation

def test_simulation(n_train: int=30, 
                    n_oos: int=250000, 
                    p: int = 10, 
                    rtol: float = 3e-2):
    """
    Checks that our stylized simulation result works as expected
    """
    # Generate the results
    di_results = simulation(n_train=n_train, n_oos=n_oos, p=p, verbose = False)
    # Loop over each coefficient
    for coef, result in di_results.items():
        print(f'Checking dist for coefficient: {coef}')
        # Loop over each metric
        for metric, values in result.items():
            n_val = len(values)
            for i in range(n_val-1):
                for j in range(i+1, n_val):
                    # The i/j'th approach should be roughly similar
                    np.testing.assert_allclose(values[i], values[j], rtol=rtol, 
                                err_msg=f'Tolerance failed for {metric}')
