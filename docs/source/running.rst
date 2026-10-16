Running a study
===============

Commands
--------

A study is configured through a YAML file: the forward solver, the parameters to estimate, the measurements, and the settings of synthesis and estimation. The same file serves all commands:

.. DAVID: Maybe just send to the API documentation instead of repeating all the details here? What if the "commands" functions are public and then listed in the API reference, which we can link here. Or a way that avoids writing things twice.

==================================== =========================================================
``kipe synthesis study.yaml``        generate the measurement data (also ``kipe synthesize``)
``kipe estimation study.yaml``       estimate the parameters (also ``kipe estimate``)
``kipe plot study.yaml``             plot the estimation history, also while it runs
``kipe list-parameters study.yaml``  show the solver's parameters and the selected ones
``kipe list-fields study.yaml``      show the solver's state fields, which measurements observe
``kipe schema``                      print the JSON schema of the study file, for editors
==================================== =========================================================

``synthesis`` and ``estimation`` take ``--log-level``, which overrides ``output.log_level`` on
the console; ``estimation`` and ``plot`` take ``--realization`` (`Noise realizations`_).
``plot`` takes several study files, drawn into one figure, and:

- ``--truth``: the nominal parameters of the first study's solver as reference lines;
- ``--save FILE``: save the figure instead of showing it;
- ``--terminal``: draw it as text, without a display; ``--watch``: redraw it as the estimation
  runs.

``kipe <command> --help`` shows all options, ``kipe --version`` the version.

With measured data, there is nothing to synthesize: ``kipe estimation`` reads the data files the
measurements name.

The study file
--------------

The study file of the FitzHugh–Nagumo example (:mod:`kipe.examples.fitzhugh_nagumo`) has all
sections:

.. DAVID: Maybe comment the FHN study file instead of writing the details here?


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
- ``synthesis``: how ``kipe synthesis`` generates data (see `Synthetic data: twin
  experiments`_);
- ``estimation``: the filter's sigma points and the number of passes over the data.

:doc:`study_file` lists every key, with its type, its default and what it does. Relative paths,
in the study file and in the solver's ``arguments``, are relative to the directory kipe runs in,
not to the study file.


.. _synthetic-data:

Synthetic data: twin experiments
--------------------------------

In a twin experiment, the data come from the model itself, with known parameters. The estimate
can then be compared with the truth: you learn whether the measurements identify the
parameters, and how precisely, before you use measured data. ``kipe synthesis`` generates such
data, and ``kipe estimation`` reads them as it would read measured data.

..  the wording point number 3 is not clear. Also, what would be the reason to have the noise definition in both the measurement and the synthesis sections? And what about the seeds? Also, how seeds are used for a number of noise realizations is not clear to me.

.. Moreover, I feel this has duplication with the Options ... is it maybe possible to point here to the class (options) in the API documentation instead of giving all the details again? Namely to improve the class documentation using this text, but having everything at one place.

For each measurement, ``kipe synthesis``

1. takes the true state at the measurement times (``synthesis.truth``): from a run of the
   forward solver with its nominal parameters, the values of its input (``type: run``, the
   default), or from the checkpoints of an earlier run (``type: checkpoints``), which may differ
   from the study's model, e.g., in its time step;
2. predicts the data from it, with the measurement's observation operator, the same as in the
   estimation (:doc:`measurements`);
3. adds noise: with the measurement's ``noise.stddev`` and ``noise.seed``, or the noise
   ``synthesis.noise`` sets for it, e.g., none;
4. writes the data to the measurement's ``data.path``, from where ``kipe estimation`` reads
   them.

Give the estimated parameters an ``initial`` value other than their nominal one: without it, the
estimation starts from the nominal value, i.e., from the truth, and tests nothing. An example:
noise-free data from an earlier run of the solver, while the estimation still assumes the
measurement's noise:

.. code-block:: yaml

    synthesis:
      truth: {type: checkpoints, path: results/reference}  # the earlier run's output
      noise:
        u: {stddev: 0}

Noise realizations
^^^^^^^^^^^^^^^^^^

To see how much a parameter estimate depends on the noise, it is useful to estimate the parameters from several noise realizations. With
``synthesis.realizations``, ``kipe synthesis`` computes the true state once and writes the data
of each realization to its own files. For this, the measurements' ``data.path`` must contain
``{realization}``, which kipe replaces by the realization number, starting from 0:

.. code-block:: yaml

    measurements:
      u:
        # ...
        data: {type: numpy, path: "data/r{realization:02d}/u.npz"}  # quoted: braces
        noise: {stddev: 0.01, seed: 1}

    synthesis:
      realizations: 30

The noise of realization :math:`r` at the :math:`k`-th time of a measurement is drawn from
``numpy.random.default_rng([seed, k, r])``. Realization 0 has the noise of the same study
without realizations.

``kipe estimation study.yaml`` then estimates from each realization in turn, into
``<output.path>/estimation/r00/`` to ``r29/``. ``--realization r`` estimates from one only,
e.g., as one task of a SLURM job array:

.. code-block:: bash

    #SBATCH --array=0-29
    srun kipe estimation study.yaml --realization $SLURM_ARRAY_TASK_ID

``kipe plot study.yaml --realization r`` plots the estimation of one realization.

Results
-------

Each command writes to its subdirectory of ``output.path``, ``synthesis/`` or ``estimation/``.
The data that ``synthesis`` generates go to the measurements' ``data`` paths.

``estimation/`` holds the estimation history; with noise realizations, each realization's
subdirectory, e.g., ``estimation/r07/``, holds the history of its estimation:

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

MPI splits one computation over several processes, called *ranks*, that work at the same time.
``mpirun -n 4`` starts 4 ranks. This makes large models faster, but it only helps if the forward
solver supports MPI. With ``-n 1``, or without ``mpirun``, kipe runs on a single rank.

Each rank holds only a part of the solver's fields (the mesh is divided among the ranks). kipe
works with each rank's part and exchanges only what is needed between them. You do not have to
do anything for this in the study file.

The measurement data do not depend on the number of ranks. So you can run ``synthesis`` on, say,
2 ranks and ``estimation`` on 8. The exception: files written by the solver itself (e.g.,
checkpoints) may be tied to a particular number of ranks. Then use the same number.

On the console, only rank 0 reports progress. The other ranks only show warnings and errors, so
the output is not repeated once per rank.
