Estimation
==========

Configuration
-------------

Select parameters and their initial estimates in the ``parameters`` section of the study file.
Set ``initial`` to choose where estimation starts. Use ``relative_stddev`` for ``log`` and
``multiplicative`` reparameterizations, or absolute ``stddev`` for ``additive`` reparameterizations. In the example:

.. literalinclude:: ../../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml
   :start-at: parameters:
   :end-before: measurements:


If ``initial`` is omitted, estimation starts at the solver's nominal value. Parameters not
selected remain fixed at their nominal values. In a twin experiment, choose an initial value
different from the truth to test whether the measurements can recover it. See :doc:`../reference/study_file`
for the available options.

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

The next run of the same command overwrites its files. To keep previous results, change
``output.path`` before rerunning. Synthesis also overwrites files at the measurements'
``data.path`` values; changing ``output.path`` does not relocate them. Change those paths or
archive the existing measurement files before rerunning synthesis.
