Identifiability
===============

``kipe sensitivity`` analyzes how well the measurements of a study determine its parameters,
before an estimation or alongside it. It evaluates the Fisher information of the measurements at
the nominal parameters, i.e., at the truth of a twin experiment, in the reparameterized
:math:`\theta` that the filter estimates (:mod:`kipe.sensitivity` gives the theory). The study
needs nothing new: the forward solver, the estimated parameters with their priors, and the
measurements with their times and noise are all the analysis uses.

Running it
----------

.. code-block:: bash

    kipe sensitivity study.yaml

The analysis consists of two forward runs per estimated parameter, at
:math:`\theta_\mathrm{true} \pm h_\theta` along it. The step ``sensitivity.step``
(:ref:`study-sensitivity`) is a relative change of the parameter, by default 0.01, i.e., 1 %; for
``additive`` parameters, it is a fraction of their prior ``stddev``, which gives it its unit.
:math:`h_\theta` is that step in :math:`\theta`. The runs do not depend on each other: for an
expensive model, ``--run I`` computes one of them, e.g., as a task of a job array
(:doc:`parallel`), and ``--assemble`` combines them once all are done. The results go to
``<output.path>/sensitivity/``: the predicted data of each run in ``runs/``, the result in
``sensitivity.npz``, and a summary in the log.

Reading the result
------------------

For the FitzHugh–Nagumo example, with its three parameters in ``log`` reparameterization and a
prior of 10 %, i.e., :math:`\sigma_\theta = 0.1`, the log shows:

.. code-block:: text

    sensitivity at the nominal parameters (step 0.01)
      parameter  nominal  prior σ_θ    CR σ_θ  CR 1σ range       with prior σ_θ
      a              0.2        0.1   0.00366  [0.1993, 0.2007]         0.00365
      b              0.2        0.1    0.0188  [0.1963, 0.2038]          0.0185
      c                3        0.1  0.000452  [2.999, 3.001]          0.000451

      Cramér-Rao correlations
               a      b      c
        a   1.00  -0.42  -0.52
        b  -0.42   1.00  -0.43
        c  -0.52  -0.43   1.00

      singular values of the weighted sensitivities (header), and the magnitudes
      of their right singular vectors: the parameters of each combination
           4.78e+03   300  52.9
        a      0.11  0.99  0.08
        b      0.02  0.08  1.00
        c      0.99  0.11  0.01
      condition number  90.4

- **CR σ_θ** is the Cramér–Rao bound of the estimate's standard deviation from the data alone,
  and **CR 1σ range** the corresponding physical range. **With prior σ_θ** is the Bayesian bound,
  which includes the prior, as the filter does. Here the two are almost equal: the data dominate
  the prior. Where the bound with the prior is clearly smaller, the prior contributes to the
  estimate.
- **The correlations** are those of the Cramér–Rao bound. Strongly correlated parameters are
  only determined together, not each on its own.
- **The singular values and vectors**: each column is a combination of the parameters, from the
  best determined, with the largest singular value, to the worst. The magnitudes in a column
  show which parameters make up the combination. Here each combination is almost a single
  parameter: c is determined best, then a, then b. The condition number is the ratio of the best
  to the worst determined combination.

``sensitivity.npz`` holds these quantities, the sensitivities and the Fisher information of each
measurement (:class:`~kipe.sensitivity.Sensitivity`). The Fisher information of several
measurements adds up, so the information of a combination of measurements is the sum of theirs.

Comparison with the estimation
------------------------------

After an estimation, the filter's covariance can be compared with the Bayesian bound. For
measurements linear in :math:`\theta`, they are equal. For FitzHugh–Nagumo, the filter's
standard deviations after its iterations are within 4 % of the bound:

.. code-block:: python

    import numpy as np

    history = np.load("results/estimation/history.npz")
    bound = np.load("results/sensitivity/sensitivity.npz")
    np.sqrt(np.diag(history["P_theta"][-1]))  # the filter: [0.00378, 0.0183, 0.000447]
    bound["posterior_stddev"]                  # the bound:  [0.00365, 0.0185, 0.000451]

Larger differences point to a model that is nonlinear on the scale of the prior. A filter
standard deviation far below the bound means that the filter is overconfident.

Caveats
-------

- The nominal parameters must be the truth. With synthetic data from the checkpoints of an
  earlier run, the forward solver's input must hold the parameters of that run.
- For ``multiplicative`` parameters, the filter's :math:`\theta` is relative to the prior
  means, which outer iterations move to the estimate; the comparison then needs a rescaling
  (:mod:`kipe.sensitivity`). ``log`` and ``additive`` need none.
- For ``additive`` parameters, :math:`\theta` has the parameters' units, and the singular values
  and vectors depend on them.
