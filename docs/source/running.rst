Running a study
===============

A study is one YAML file: the forward solver, the parameters to estimate, the measurements,
and the settings of synthesis and estimation. The same file serves all commands:

.. code-block:: bash

    kipe synthesis study.yaml     # synthetic data from a forward run (for a twin experiment)
    kipe estimation study.yaml    # estimate the parameters from the data
    kipe plot study.yaml          # the estimates with their 1σ range

With measured data, there is nothing to synthesize: ``kipe estimation`` reads the data files the
measurements name.

The study file
--------------

The study file of the FitzHugh–Nagumo example (:mod:`kipe.examples.fitzhugh_nagumo`) has all
sections:

.. literalinclude:: ../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml

- ``output``: where the results go, and how much the console shows;
- ``forward_solver``: the model, built by a factory with its arguments. The solver's settings,
  e.g., its time step, belong there: kipe never changes them;
- ``parameters``: the parameters to estimate, with their initial estimate and uncertainty, and
  how the filter represents them (``reparameterization``). All other parameters keep their
  nominal values, those of the solver's input. ``kipe list-parameters study.yaml`` shows the
  solver's parameters and the initial 1σ range of the selected ones;
- ``measurements``: what is measured, when, and where the data are (:doc:`measurements`);
- ``synthesis``: how ``kipe synthesis`` generates data (:ref:`synthetic-data`);
- ``estimation``: the filter's sigma points and the number of passes over the data.

:doc:`study_file` lists every key, with its type, its default and what it does. Relative paths,
in the study file and in the solver's ``arguments``, are relative to the directory kipe runs in,
not to the study file.

Commands
--------

==================================== =========================================================
``kipe synthesis study.yaml``        generate the measurement data (also ``kipe synthesize``)
``kipe estimation study.yaml``       estimate the parameters (also ``kipe estimate``)
``kipe plot study.yaml``             plot the estimation history, also while it runs
``kipe list-parameters study.yaml``  show the solver's parameters and the selected ones
==================================== =========================================================

``synthesis`` and ``estimation`` take ``--log-level``, which overrides ``output.log_level`` on
the console. ``plot`` takes several study files, drawn into one figure, and:

- ``--truth``: the nominal parameters of the first study's solver as reference lines;
- ``--save FILE``: save the figure instead of showing it;
- ``--terminal``: draw it as text, without a display; ``--watch``: redraw it as the estimation
  runs.

``kipe <command> --help`` shows all options, ``kipe --version`` the version.

Results
-------

Each command writes to its subdirectory of ``output.path``, ``synthesis/`` or ``estimation/``.
The data that ``synthesis`` generates go to the measurements' ``data`` paths.

``estimation/`` holds the estimation history:

- ``history.csv``: one row per assimilation step, written as the filter runs: the iteration,
  the time, and per parameter, e.g., ``c``: its estimate ``c``, its 1σ range ``lower_c``,
  ``upper_c``, and the filter's ``theta_c`` and ``stddev_theta_c``. The first row of each
  iteration is the initial estimate, at the solver's start time.
- ``history.npz``: the same, and the full covariance of :math:`\theta`:

.. code-block:: python

    import numpy as np

    history = np.load("results/estimation/history.npz")
    history["names"]       # the parameters, in the order of the study file
    history["parameters"]  # the estimates, one row per step
    history["P_theta"]     # the covariance of theta, one matrix per step
    final = history["parameters"][-1]

Both subdirectories also hold:

- ``kipe.log``: all messages, from the debug level on, with where they were logged;
- the provenance of the run: ``run.yaml`` (status, start and end time), ``study.yaml`` (the
  options as they ran, with all defaults), ``environment.yaml`` (host, versions, git state),
  ``packages.txt``, and ``<package>.patch`` for uncommitted changes of a git checkout.

The next run of the same command overwrites them.

Running in parallel
-------------------

Both commands run on several MPI ranks, if the forward solver does:

.. code-block:: bash

    mpirun -n 4 kipe estimation study.yaml

The solver distributes its fields; kipe exchanges only the rank-local parts. The measurement
data do not depend on the number of ranks, so ``synthesis`` and ``estimation`` can use different
numbers, unless the solver's own files (e.g., checkpoints) tie it to one. On the console, rank 0
reports; the other ranks only report warnings and errors.
