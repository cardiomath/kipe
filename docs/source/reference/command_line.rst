Command line interface
======================

The commands take a study file unless noted otherwise. The sections below the table list the
options of every command, as ``kipe <command> --help`` shows them.

========================================= ============================================================
Command                                   Purpose
========================================= ============================================================
``kipe synthesis study.yaml``             Generate synthetic measurement data
``kipe synthesize study.yaml``            Alias for ``kipe synthesis``
``kipe estimation study.yaml``            Estimate parameters from measurement data
``kipe estimate study.yaml``              Alias for ``kipe estimation``
``kipe plot study.yaml``                  Plot the estimation history
``kipe list-parameters study.yaml``       List the solver's parameters and nominal values
``kipe list-fields study.yaml``           List the solver's state fields and their sizes
``kipe sensitivity study.yaml``           Analyze the identifiability: the Fisher information
``kipe plot-sensitivity study.yaml``      Plot the result of the sensitivity analysis
``kipe schema``                           Print the study-file JSON schema
========================================= ============================================================

.. sphinx_argparse_cli::
   :module: kipe.cli
   :func: main
   :hook:
   :prog: kipe
   :title:
   :description: Generated from kipe's argument parser.
