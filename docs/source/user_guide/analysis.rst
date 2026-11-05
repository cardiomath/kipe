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

A template notebook
-------------------

``examples/fitzhugh_nagumo/analysis.py`` is a (fully documented) template to copy and adapt, for a study with noise realizations, such as ``examples/fitzhugh_nagumo/study_realizations.yaml``. The template contains also an explanation on how to interpret the results, and how the results should look like when changing the configuration.
