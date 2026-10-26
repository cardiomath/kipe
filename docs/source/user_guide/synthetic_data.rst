Synthetic data
==============

In a twin experiment, the true parameters are known. Comparing the estimates with the true
values shows whether, and how precisely, the measurements identify the parameters before
measured data are used.

For each measurement, ``kipe synthesis``:

1. obtains the true state at the measurement times, either by running the forward solver with its
   nominal parameters (the default) or by reading checkpoints from an earlier run;
2. predicts the measurement using the observation operator described in :doc:`measurements`;
3. adds noise and writes the data to the measurement's ``data.path``.

The checkpoint run may use a different model or time step. The ``synthesis.truth`` option selects
the source of the true state; see :doc:`../reference/study_file` for its options.

Then, ``kipe estimation`` reads the synthetic data just as it would measured data and estimates
the parameters from it.


Noise
-----

The measurement's ``noise.stddev`` is the noise level assumed during estimation. Synthesis adds
noise using that standard deviation and ``noise.seed`` by default. ``synthesis.noise`` can override
the noise for generated data, for example to generate noise-free data while estimation still
assumes the measurement's noise:

.. code-block:: yaml

    synthesis:
      truth: {type: checkpoints, path: results/reference}
      noise:
        u: {stddev: 0}

Multiple noise realizations
---------------------------

To see how much an estimate depends on noise, set ``synthesis.realizations`` to the number of
realizations. Synthesis computes the true state once and writes each realization to a separate
file. Include ``{realization}`` in each measurement's ``data.path``; kipe replaces it with the
realization number, starting at 0:

.. code-block:: yaml

    measurements:
      u:
        # ...
        data: {type: numpy, path: "data/r{realization:02d}/u.npz"}
        noise: {stddev: 0.01, seed: 1}

    synthesis:
      realizations: 30

The noise for realization :math:`r` at the :math:`k`-th measurement time is generated using
``numpy.random.default_rng([seed, k, r])``. Realization 0 has the same noise as a study without
multiple realizations.

Checking the data files
-----------------------

Synthesis writes ``data.sha256`` to ``<output.path>/synthesis/``: the SHA-256 hash and the path
of each data file it wrote, in the format of ``sha256sum``. Run from the directory kipe ran in,
since the paths are relative to it,

.. code-block:: bash

    sha256sum -c results/synthesis/data.sha256

checks that the data files are the ones synthesis wrote, e.g., after copying them to a cluster.
Estimation records the hashes of the data files it reads in its ``run.yaml``, and compares them
when it resumes (:ref:`resuming-an-estimation`).
