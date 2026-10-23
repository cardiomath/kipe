Estimation
==========

Configuration
-------------

The ``parameters`` section of the study file lists the estimated parameters with their priors:
the ``mean``, where the estimation starts, and the standard deviation, a ``relative_stddev`` for
``log`` and ``multiplicative`` reparameterizations, an absolute ``stddev`` for ``additive``. In
the example:

.. literalinclude:: ../../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml
   :start-at: parameters:
   :end-before: measurements:


Parameters without a prior remain fixed at their nominal values. In a twin experiment, the prior
means differ from the truth: the estimation then tests whether the measurements recover it.
See :doc:`../reference/study_file` for the available options.

Run ``kipe estimation study.yaml`` to estimate parameters from the data specified in the study's
``measurements`` section. The ``estimation`` section configures the setup of the estimation process, as exemplified in the following snippet:

.. literalinclude:: ../../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml
   :start-at: estimation:
   :end-before: end-of-study-file

With multiple noise realizations, estimation processes each realization in
turn and writes its results to a separate directory. To run a single realization — for example, as a
task in a cluster job array — use ``kipe estimation study.yaml --realization R`` (:doc:`parallel`).

Results
-------

Each command writes to its own subdirectory of ``output.path``: ``synthesis/`` or
``estimation/``. Generated measurement data are written to their configured ``data.path`` locations.

Estimation writes ``history.csv`` and ``history.npz`` under ``estimation/``; with multiple
realizations, each has its own subdirectory, e.g., ``estimation/r07/``. The CSV has one row per
assimilation step, including each parameter's estimate and 1σ range. The NPZ contains the same
history and the full covariance of :math:`\theta`:

.. code-block:: python

    import numpy as np

    history = np.load("results/estimation/history.npz")
    history["names"]       # parameter names, in study-file order
    history["parameters"]  # estimates, one row per step
    history["P_theta"]     # covariance of theta, one matrix per step

Both run directories also contain ``kipe.log`` and provenance files: ``run.yaml``,
``study.yaml``, ``environment.yaml``, ``packages.txt``, and patches for uncommitted package
changes.

``kipe plot`` draws one subplot per parameter over the assimilation steps. The title shows the
latest estimate and its 1σ range. The colored line is the estimate after each step; the two gray
lines around it are the ends of its 1σ range (lower and upper), hidden behind the estimate wherever
the range is narrower than one dot of the terminal plot; vertical lines mark the starts of the outer
iterations (green with ``--terminal`` or ``--watch``, gray in the figure); and, with ``--truth``, a
differently colored horizontal line is the true value. With ``--terminal`` or ``--watch``, a line
below the plot explains these.

The next run of the same command overwrites its files. To keep previous results, change
``output.path`` before rerunning. Synthesis also overwrites files at the measurements'
``data.path`` values; changing ``output.path`` does not relocate them. Change those paths or
archive the existing measurement files before rerunning synthesis.
