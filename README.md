# kipe

[![CI](https://github.com/cardiomath/kipe/actions/workflows/ci.yml/badge.svg)](https://github.com/cardiomath/kipe/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-cardiomath.github.io%2Fkipe-blue)](https://cardiomath.github.io/kipe/)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<!-- docs: start of the introduction (index.md) -->

**K**alman filtering and **i**dentifiability analysis for
**p**arameter **e**stimation of distributed dynamical systems

kipe estimates the parameters of distributed dynamical models, e.g., time-dependent PDE models,
from time series of measurements,  with the reduced-order unscented Kalman filter (ROUKF).

The filter needs only forward runs of your model, no derivatives, through a lean interface. Around
the estimation, kipe generates synthetic measurement data for twin experiments, from a run of the
model or from the checkpoints of an earlier run, and will analyze the identifiability of the
parameters with the Fisher information (planned).

## Key features

+ **Solver-agnostic**: Connect your model through a small forward-solver interface
+ **Derivative-free**: Optimizer is based on the reduced-order
unscented Kalman filter (ROUKF) and hence it needs only forward runs.
+ **Modular observation operators**: New observation operators can be implemented easily by
  extending the library.
+ **Twin experiments**: Generate synthetic measurements from a model run.
+ **Reproducible runs**: Keep results, logs, and run provenance together.

> **Status: early development.** The core works (ROUKF estimation, synthetic data, plotting, run
> provenance), as does a first FEniCSx backend for measurements on finite element meshes, but the
> study file format and the Python API may still change. Feedback is very welcome: please open an
> [issue](https://github.com/cardiomath/kipe/issues).

<!-- docs: end of the introduction -->

## Installation

`kipe` needs Python ≥ 3.12 and an MPI library (for `mpi4py`):

```bash
pip install kipe
pip install mpich  # only if there is no MPI library on your system
```

For FEniCSx models, install kipe into your FEniCSx (dolfinx 0.9 to 0.11) environment.

## Quick start

A twin experiment with the FitzHugh–Nagumo model: synthetic data from the true parameters, then
the estimation from a wrong initial guess.

```bash
git clone https://github.com/cardiomath/kipe
cd kipe/examples/fitzhugh_nagumo
kipe synthesis study.yaml
kipe estimation study.yaml
kipe plot-estimation study.yaml --truth  # --terminal without a display
```

## Documentation

**[cardiomath.github.io/kipe](https://cardiomath.github.io/kipe/)**: getting started, running a
study, measurements, the study file reference, the theory, and how to extend kipe.

## Development

```bash
git clone https://github.com/cardiomath/kipe
cd kipe
pip install -e ".[dev]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
pytest
```

See [Development](https://cardiomath.github.io/kipe/development.html) in the documentation.

## License

MIT, see [LICENSE](https://github.com/cardiomath/kipe/blob/main/LICENSE). If you use kipe, please
cite it as described in
[CITATION.cff](https://github.com/cardiomath/kipe/blob/main/CITATION.cff).
