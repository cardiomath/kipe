Analyzing the results
=====================

``kipe plot-estimation`` and ``kipe plot-sensitivity`` are quick checks of a single run. For
the statistics of the estimates over the noise, e.g., how far they scatter compared with the
filter's uncertainty and with the bounds of the sensitivity analysis (:doc:`identifiability`),
kipe provides a reader for the results of a study's noise realizations
(:doc:`synthetic_data`), and a template to analyze them.

Reading the results
-------------------

:func:`~kipe.results.read_realizations` reads the final estimate of each realization, the
filter's covariance of it, and the estimate at the end of each outer iteration:

.. code-block:: python

    from kipe.options import load_study
    from kipe.results import read_realizations

    result = read_realizations(load_study("study_realizations.yaml"))
    result.parameters   # the final estimates, one row per realization
    result.covariance   # the filter's covariance of each, in physical units
    result.iterations   # the estimate at the end of each outer iteration

A realization whose estimation has not completed all of its outer iterations is skipped, with
a warning, so that the results of a job array can be read while it runs (:doc:`parallel`).

The covariances are in physical units, which are comparable across realizations. The filter's
:math:`\theta` is not: outer iterations anchor the last pass at the estimate of the pass before,
which differs from realization to realization, so that the realizations' :math:`\theta`, and for
``multiplicative`` parameters even its scale, refer to different anchors.

The template
------------

``examples/fitzhugh_nagumo/analysis.py`` is a template to copy and adapt, for a study with
noise realizations, such as ``examples/fitzhugh_nagumo/study_realizations.yaml``:

.. code-block:: bash

    kipe synthesis study_realizations.yaml
    kipe estimation study_realizations.yaml
    kipe sensitivity study_realizations.yaml   # optional: the bounds
    python analysis.py study_realizations.yaml  # or --save DIR for the figures

It runs as a script, or cell by cell as a notebook (its cells are marked with ``# %%``, which
VS Code and, with jupytext, Jupyter understand). It prints a summary per parameter: the bias of
the mean estimate, and the standard deviations of the estimates over the realizations
(empirical), of the filter (mean over the realizations), and of the Cramér–Rao and Bayesian
bounds. Its figures show:

- **the estimates over the realizations**, with their mean, spread and the truth;
- **the consistency**: the empirical standard deviation against the filter's and the bounds.
  A filter standard deviation clearly below the empirical one means an overconfident filter;
- **the correlations**: empirical, of the filter and of the bounds. A filter can have the
  right spreads but miss correlations;
- **the pairs**: for each pair of parameters, the estimates and the 2σ ellipses of the
  covariances, which show size and orientation together.

For the example's 20 realizations, the filter's standard deviation of ``b`` is about 30 %
below the empirical one, which agrees with the Cramér–Rao bound, the bound of the data alone.
The filter's standard deviation agrees with the Bayesian bound instead, which includes the
prior. The outer iterations largely remove the prior's influence on the estimate, by
anchoring each pass at the estimate of the one before, so the estimates scatter as the data
alone allow. The filter's covariance, however, still contains the prior of its last pass, and
understates that scatter where the prior is not small against the information of the data,
as for ``b``. The correlations show the same: the filter's are those of the Bayesian bound,
the estimates' those of the Cramér–Rao bound, and so do the ellipses of the pairs.
