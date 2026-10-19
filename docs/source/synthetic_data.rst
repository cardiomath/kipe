Synthetic data
==============

In a twin experiment, the true parameters are known. Comparing estimated parameters with the
truth shows whether, and how precisely, the measurements identify them before you use measured
data. ``kipe synthesis`` generates data that ``kipe estimation`` reads just like measured data.

For each measurement, synthesis:

1. obtains the true state at the measurement times, either by running the forward solver with its
   nominal parameters (the default) or by reading checkpoints from an earlier run;
2. predicts the measurement using the observation operator described in :doc:`measurements`;
3. adds noise and writes the data to the measurement's ``data.path``.

The checkpoint run may use a different model or time step. The ``synthesis.truth`` option selects
the source of the true state; see :doc:`study_file` for its options.

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

The noise for realization :math:`r` at the :math:`k`-th time of a measurement is drawn from
``numpy.random.default_rng([seed, k, r])``. Realization 0 has the same noise as a study without
multiple realizations.

**TODO:** Revisit whether noise seeds and realization-specific data paths belong under
``measurements`` or ``synthesis``. The current study-file format keeps them under ``measurements``.

**TODO:** Consider whether future versions should add controlled noise to externally measured
data for robustness tests.
