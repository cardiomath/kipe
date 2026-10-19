# kipe

```{include} ../../README.md
:start-after: "<!-- docs: start of the introduction (index.md) -->"
:end-before: "<!-- docs: end of the introduction -->"
```

## User guide

- {doc}`user_guide/getting_started`: install kipe and run a first twin experiment.
- {doc}`user_guide/anatomy`: how a study connects the solver, measurements, data and estimation.
- {doc}`user_guide/measurements`: what kipe compares with data, and how.
- {doc}`user_guide/synthetic_data`: generate data for a twin experiment.
- {doc}`user_guide/estimation`: configure estimation and read its results.
- {doc}`user_guide/parallel`: run with MPI or across cluster tasks.

## Theory

- {doc}`theory/theory`: the estimation problem and the reduced-order unscented Kalman filter.

## Reference

- {doc}`reference/study_file`: every study-file key, its type, default and purpose.
- {doc}`reference/command_line`: the available commands and their options.
- {doc}`reference/api`: modules, classes and functions, from their docstrings.

## Extending kipe

- {doc}`extending/architecture`: how kipe is put together and where extensions fit.
- {doc}`extending/write_a_sampler`: add a spatial sampler.
- {doc}`extending/development`: work on kipe.

```{toctree}
:hidden:
:includehidden:

user_guide/index
theory/index
reference/index
extending/index
```
