Anatomy of a study
==================

A *study* connects a forward solver, measurements, data and parameter estimation. Its study file
configures the solver and the measurements, and tells kipe how to synthesize data and estimate
parameters. The same file is used throughout a twin experiment:

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

The measurements' observation operators turn model states into predicted data. Synthesis uses
them to generate data; estimation uses the same operators to compare predictions with data.
:doc:`measurements` explains how measurements and observation operators work.

The study file
--------------

The FitzHugh–Nagumo example (:mod:`kipe.examples.fitzhugh_nagumo`) has all the sections of a
study file:

.. literalinclude:: ../../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml

Broadly, the sections configure:

- ``output``: where kipe writes results and how much it reports;
- ``forward_solver``: the model and its arguments;
- ``parameters``: which parameters to estimate and their initial estimates;
- ``measurements``: what is observed, when, and where the data are;
- ``synthesis``: how to generate synthetic data;
- ``estimation``: the filter's sigma-point stencil and number of iterations.

:doc:`../reference/study_file` lists every option and its default. Parameters selected for estimation start
from their configured initial values; unselected parameters stay at the solver's nominal values.
Synthesis uses the solver's nominal parameters. Relative paths are resolved from the directory
where kipe is run, not from the study file's directory.

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
