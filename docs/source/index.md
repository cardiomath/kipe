# kipe

```{include} ../../README.md
:start-after: "<!-- docs: start of the introduction (index.md) -->"
:end-before: "<!-- docs: end of the introduction -->"
```

## User guide

- {doc}`getting_started`: install kipe and run a first twin experiment.
- {doc}`anatomy`: how a study connects the solver, measurements, data and estimation.
- {doc}`measurements`: what kipe compares with data, and how.
- {doc}`synthetic_data`: generate data for a twin experiment.
- {doc}`estimation`: configure estimation and read its results.
- {doc}`parallel`: run with MPI or across cluster tasks.

## Reference

- {doc}`study_file`: every study-file key, its type, default and purpose.
- {doc}`command_line`: the available commands and their options.
- {doc}`api`: modules, classes and functions, from their docstrings.

## Theory

- {doc}`theory`: the estimation problem and the reduced-order unscented Kalman filter.

## Extending kipe

- {doc}`architecture`: how kipe is put together and where extensions fit.
- {doc}`write_a_sampler`: add a spatial sampler.
- {doc}`development`: work on kipe.

```{toctree}
:hidden:

getting_started
anatomy
measurements
synthetic_data
estimation
parallel
study_file
command_line
theory
architecture
write_a_sampler
development
api
```
