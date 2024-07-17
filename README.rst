=====================
Loss Function Moments
=====================

.. role:: raw-math(raw)
    :format: latex html

.. image:: https://img.shields.io/pypi/v/loss_moments.svg
        :target: https://pypi.python.org/pypi/loss_moments

.. image:: https://img.shields.io/travis/ErikinBC/loss_moments.svg
        :target: https://travis-ci.com/ErikinBC/loss_moments

.. image:: https://readthedocs.org/projects/loss-function/badge/?version=latest
        :target: https://loss-function.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Package to help calculate the mean and variance (first and second moments) of a loss function :raw-math:`$\ell(\cdot)$` for a supervised ML model:

:raw-math:`$$
\begin{align*}
R(\theta; f) &= E_{(y,x) \sim P_{h(\phi)}}[\ell(y, f_\theta(x))] \\
V(\theta; f) &= E_{(y,x) \sim P_{h(\phi)}}[ (\ell(y, f_\theta(x)) - R(\theta))^2 ]
\end{align*}
$$`

For more information about notation and theory, see this blog post: `SOME TITLE <http://www.erikdrysdale.com/.../>`_.

* Free software: GNU General Public License v3
* Documentation: https://loss-function.readthedocs.io.

Features
--------

* Monte Carlo Integration
        * Joint
        * Conditional
* Numerical Integration 
        * Joint (Trapezoidal, Quadratric)
        * Conditional (Trapezoidal)


Development
-----------

After making changes, try running (from the main directory)

.. code:: bash

        source src/loss_moments.sh

This will call both the unit tests and the main module.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
