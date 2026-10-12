# kipe

***K**alman-based **i**dentifiability and **p**arameter **e**stimation*

kipe estimates parameters of dynamical models, e.g., PDE models, from time series of
measurements, with the reduced-order unscented Kalman filter (ROUKF). The filter needs only
forward runs of your model, no derivatives and no changes to the solver. kipe also generates
synthetic measurement data from a model run, for twin experiments.

> **Status: early development.** The core works (ROUKF estimation, synthetic data, plotting,
> run provenance), as does a first FEniCSx backend for measurements on finite element meshes,
> but the study file format and the Python API may still change. Feedback is very welcome:
> please open an [issue](https://github.com/cardiomath/kipe/issues).

## Installation

kipe needs Python ≥ 3.12 and an MPI library (for `mpi4py`). Use your system's MPI, or install one
with pip:

```bash
pip install "kipe[plot]"
pip install mpich  # only if there is no MPI library on your system
```

The `plot` extra adds matplotlib and plotext for `kipe plot`. The latest development version
installs from GitHub:

```bash
pip install "kipe[plot] @ git+https://github.com/cardiomath/kipe"
```

For FEniCSx models, kipe also needs dolfinx (0.9), which pip does not install: use your FEniCSx
installation (e.g., Spack, conda or the `dolfinx/dolfinx` Docker image) and install kipe into its
Python environment. Its MPI is then the one kipe uses.

## First run: the FitzHugh–Nagumo example

The example estimates the three parameters of the FitzHugh–Nagumo model from noisy measurements
that kipe generated itself with the true parameters (a twin experiment):

```bash
git clone https://github.com/cardiomath/kipe
cd kipe/examples/fitzhugh_nagumo
kipe synthesis study.yaml     # generate the measurement data
kipe estimation study.yaml    # estimate a, b, c from a wrong initial guess
kipe plot study.yaml --truth  # estimates with their 1σ range; --terminal without a display
```

Everything is configured in the study file,
[`study.yaml`](https://github.com/cardiomath/kipe/blob/main/examples/fitzhugh_nagumo/study.yaml):
the forward solver, the estimated parameters with their initial guess and uncertainty, the
measurements and the filter. Results, a log and the provenance of the run (versions, git state,
environment) go to the `output.path` it names.

Other subcommands: `kipe list-parameters study.yaml` shows the parameters a solver accepts;
`kipe <subcommand> --help` shows the options.

## Your own model

kipe calls your model through a small interface, the `ForwardSolver` protocol in
[`kipe/forward_solver.py`](https://github.com/cardiomath/kipe/blob/main/src/kipe/forward_solver.py):
the initial state, the nominal parameters, and a `propagate` that advances a state from one time
to the next with given parameters. The study file names a factory that builds your solver:

```yaml
forward_solver:
  factory: "my_package.my_module:MySolver"   # module:name, imported by kipe
  arguments: {dt: 0.01}                      # arguments passed to MySolver
```

The `arguments` are those passed to the factory, i.e., `MySolver(dt=0.01)`.

The FitzHugh–Nagumo solver in
[`kipe/examples/fitzhugh_nagumo.py`](https://github.com/cardiomath/kipe/blob/main/src/kipe/examples/fitzhugh_nagumo.py)
is a complete example in about 150 lines, docstrings included.

## FEniCSx models

A FEniCSx solver exchanges the owned degrees of freedom of its fields as arrays, like any other
state, and additionally provides their function spaces, `function_space(field)` (see
[`kipe/fenicsx/forward_solver.py`](https://github.com/cardiomath/kipe/blob/main/src/kipe/fenicsx/forward_solver.py)).
A measurement can then interpolate a field onto a measurement mesh, which need not match the
solver's:

```yaml
measurements:
  u:
    fields: [u]
    spatial_sampling: {type: fenicsx.mesh, mesh: data/measurement_mesh.xdmf, element: P1}
    ...
```

A vector field is measured as a whole, as its component along a direction
(`projection: [0, 0, 1]`), or normal to a surface mesh (`projection: surface_normal`, with
`element: DG0`).

The measurement data are in the order of the measurement mesh file (one value per vertex for
`P1`, per cell for `DG0`), so synthesis and estimation can run on different numbers of MPI ranks;
they must use the same mesh file. The heat equation example shows a complete setup:

```bash
cd kipe/examples/heat
python measurement_mesh.py         # the measurement mesh
kipe synthesis study.yaml
mpirun -n 2 kipe estimation study.yaml
```

Its solver,
[`kipe/fenicsx/examples/heat.py`](https://github.com/cardiomath/kipe/blob/main/src/kipe/fenicsx/examples/heat.py),
is a template for your own.

Existing data in PETSc binary files, one per time and rank, are read with
`data: {type: fenicsx.petscbinary, path: "data/{i}/u_np{mpi_size}-{mpi_rank}.dat"}` and a
`fenicsx.mesh` sampling that matches the one the data were written with. They hold each rank's
part of the data, so estimation must run on the same number of ranks.

## Development

```bash
git clone https://github.com/cardiomath/kipe
cd kipe
pip install -e ".[test,plot,check]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
pytest
```

The pre-commit hooks run ruff, mypy and a check of the commit message, which follows
[Conventional Commits](https://www.conventionalcommits.org), e.g., `fix(roukf): ...`; the
allowed scopes are listed in `.pre-commit-config.yaml`.

## License

MIT, see [LICENSE](https://github.com/cardiomath/kipe/blob/main/LICENSE). If you use kipe, please
cite it as described in
[CITATION.cff](https://github.com/cardiomath/kipe/blob/main/CITATION.cff).
