Anatomy of a study
==================

In kipe, a *study* is a parameter estimation problem, described by a study file. The study file defines the forward model, the parameters to estimate with their priors, the measurements and where their data are, and the settings of the Kalman filter. The data are either measured or synthesized from the model with ``kipe synthesis``. For synthetic data, a study can define several independent noise realizations, to study the statistics of the estimates under noise.

A single study file also defines a *twin experiment*: ``kipe synthesis`` computes a ground truth with known parameters and generates noisy data from it, and ``kipe estimation`` then estimates the parameters from these data, starting from an inaccurate prior.
That is, the following commands generate the synthetic data, estimate the parameters, and plot the estimates against the true parameters:

.. code-block:: bash

    kipe synthesis study.yaml
    kipe estimation study.yaml
    kipe plot study.yaml --truth

.. mermaid::

   flowchart LR
       study[/"study file"/] --> solver["forward solver"]
       solver --> truth["true model state"]
       truth --> operator["measurement operator"]
       operator --> data[/"synthetic data"/]
       measured[/"measured data"/] --> data
       data --> estimate["parameter estimation"]
       solver --> estimate
       estimate --> history[/"estimation results"/]

:doc:`measurements` explains how kipe compares the model with measured data and :doc:`../extending/architecture` describes the data flow and the code structure in detail, for those who extend kipe.

The study file
--------------

The FitzHugh–Nagumo example (:mod:`kipe.examples.fitzhugh_nagumo`) has all the sections of a
study file:

.. literalinclude:: ../../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml

:doc:`../reference/study_file` lists every option and its default.

Run a study
-----------

For a twin experiment, run synthesis, estimation, then plotting:

.. code-block:: bash

    kipe synthesis study.yaml
    kipe estimation study.yaml
    kipe plot study.yaml --truth

With measured data, skip synthesis and configure their file paths under ``measurements``. The
commands write logs and run provenance under subdirectories of ``output.path``; estimation also
writes its history there. Synthesized measurement data go to the ``data.path`` values in the
study file. See :doc:`../reference/command_line` for command options, :doc:`synthetic_data` for data generation
and :doc:`estimation` for parameter estimates and results.
