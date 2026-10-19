Command line interface
======================

The commands take a study file unless noted otherwise. Run ``kipe <command> --help`` for all
options.

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
``kipe schema``                           Print the study-file JSON schema
========================================= ============================================================

``kipe plot`` accepts several study files to plot in one figure. Its options include ``--truth``
to show nominal parameter values, ``--save FILE`` to save a figure, ``--terminal`` to draw in
the terminal, and ``--watch`` to redraw while estimation runs. ``kipe estimation`` and
``kipe plot`` accept ``--realization R`` to select one noise realization. ``kipe synthesis`` and
``kipe estimation`` accept ``--log-level`` to override ``output.log_level`` on the console.

Use ``kipe --version`` to print the installed version.
