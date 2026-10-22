Anatomy of a study
==================

In kipe, a *study* is a parameter estimation problem defined by a study file. It specifies the forward model, parameters and priors, measurement settings and data paths, and Kalman filter settings. Data can be measured or synthesized, and synthesized data can include multiple independent noise realizations.

For instance, using a single study file, *twin experiments* can be conducted:

.. code-block:: bash

    kipe synthesis study.yaml     # generates noisy data from known ground-truth parameters
    kipe estimation study.yaml    # recovers them from an inaccurate prior using ROUKF
    kipe plot study.yaml --truth  # compares estimates with the true parameters

``synthesis`` writes data to the measurements' ``data.path``; ``estimation`` writes results to
``output.path``.

.. mermaid::

    flowchart TB
        study[/"study.yaml"/] --> synthesis["kipe synthesis"]
        synthesis --> data[/"data files (data.path)"/]
        measured[/"measured data"/] -.-> data
        study --> estimation["kipe estimation"]
        data --> estimation
        estimation --> results[/"estimates & uncertainties (output.path)"/]

:doc:`measurements` explains how kipe compares the model with measured data, while :doc:`../extending/architecture` describes the data flow and code structure for readers who want to extend kipe.

The study file
--------------

The FitzHugh–Nagumo example (:mod:`kipe.examples.fitzhugh_nagumo`) has all the sections of a
study file:

.. literalinclude:: ../../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml

:doc:`../reference/study_file` lists every option and its default.
