# kipe

```{include} ../../README.md
:start-after: "<!-- docs: start of the introduction (index.md) -->"
:end-before: "<!-- docs: end of the introduction -->"
```

## Where to start

- {doc}`getting_started`: install kipe, run a first twin experiment, and connect your own model.
- {doc}`running`: the study file, the commands and their results, synthetic data.
- {doc}`measurements`: what kipe compares with the data, and how; {doc}`study_file`: every key
  of a study file.
- {doc}`theory`: the estimation problem and the reduced-order unscented Kalman filter.
- To add a sampler or a measurement model: {doc}`architecture`, then {doc}`write_a_sampler`;
  to work on kipe: {doc}`development`.
- {doc}`api`: the modules, classes and functions, from their docstrings.

```{toctree}
:hidden:

getting_started
running
measurements
study_file
theory
architecture
write_a_sampler
development
api
```
