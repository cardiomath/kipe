# kipe: design overview

**Goal:** one open-source package (GitHub, MIT) for Kalman-based parameter estimation (ROUKF),
Fisher-information sensitivity analysis and synthetic measurements. It should work with *any*
forward solver, not only fsix. kipe is the successor of roukf: a fresh repo with ported code,
while roukf stays in use until kipe reaches parity.

## Principles

1. **Solver-agnostic core.** The core is pure numpy, with no FEniCS dependency. FEniCSx is an
   optional backend (`pip install kipe[fenicsx]`).
2. **kipe never changes solver settings.** The solver owns `dt`, `T`, output paths. kipe only
   passes state, parameters and time. This removes today's conflicting settings, such as the
   timestep defined in both input files.
3. **Configuration lives in input files, not command-line arguments.** One archived YAML file
   fully reproduces an HPC run.
4. **Small, explicit extension points.** Adding a new solver, data format, spatial sampler or
   measurement type means writing one small class.

## Forward solver interface

- **Stateless calls:** `initial_state()`, `advance_to(t0, t1, state, θ)`,
  `timestep(t0, state, θ)`.
- **State:** named numpy fields, local to each MPI rank. Measurements refer to fields by name
  (`velocity`), not by index.
- **Parameters:** the solver lists what can be estimated, with nominal values
  (e.g. `fluid.bc4.R_d`). The kipe config selects parameters and sets the prior (std next to
  the reparameterization; optionally the initial guess). Today the prior is split across two
  files.
- **Construction:** the input file names a `factory` (a class, or a function returning the
  solver) plus its arguments. Using kipe directly from Python works just as well.
- **Time:** `advance_to` guarantees that all particles reach the same time. Solvers with a
  fixed step are checked against the measurement times at startup, before anything is
  computed.

## Measurements, sampling, innovations

- **Measurement:** data (read by a pluggable Reader) + times + noise σ. One generic class.
- **SpatialSampler:** samples the solver's fields at the measurement locations; depends on
  the solver: interpolation, voxelization, point evaluation, projections (component, surface
  normal). ("Spatial" as opposed to the temporal sampling given by the measurement times.)
- **MeasurementModel:** the acquisition physics. `predict` is the signal model (e.g. MRI
  phase, FFT, mask); `innovation` is the comparison formula (plain difference, phase-based,
  complex).
- **Observation operator** H = M ∘ S: measurement model after spatial sampling,
  h(x) = `predict(sample(x))`. It's shared by estimation, synthetic data and sensitivity
  analysis, so the same model that generates the data also inverts it. ("Observation" is
  reserved for this composite, as in the data-assimilation literature.)
- **Extension cost:** a new data format is one reader; a new sampling method is one
  spatial sampler; a new innovation rule is one model. Costs add up (N + M), instead of one class
  per combination (N × M).

## Parallelism

- **Particles in parallel:** within a step, particles are independent; they only synchronize at
  the correction. Ranks are split into groups, each running its own solver instance on n
  ranks; the filter combines particles across groups with elementwise MPI reductions (the
  state is rank-local numpy). Up to q·n tasks for q particles in one job; serial solvers
  (numpy/scipy) are the case n = 1 (`mpirun -n q`).
- **After 0.1, prepared for now:** 0.1 runs particles sequentially. The core takes its MPI
  communicator as an argument from day one (no hardcoded global communicator), and solvers
  receive theirs at construction, so particle groups can be added later without re-plumbing:
  first serial solvers (n = 1), then groups of several ranks (partition check, fsix
  sub-communicator support).
- **Sensitivity analysis (FIM):** the 2n+1 finite-difference runs are independent, so they
  map onto a job array: `kipe sensitivity plan / run --index i / assemble`, or a local loop.

## Problems fixed along the way

- **Measurement data at other MPI rank counts:** PETScBinaryIO only reads back with the same
  number of ranks. io4dolfinx reads with any number.
- **MPI-safe innovations:** each entry is owned by exactly one rank.
- **Generic resume** of interrupted runs.
- **Parameter names instead of `theta_0..n`** in the outputs.

## Input file and CLI

**One CLI with subcommands** (`kipe estimation / synthesis / sensitivity`, with verb aliases
`kipe estimate` / `kipe synthesize`) and **one input file per study**, with optional sections:
each subcommand uses its subset. A twin experiment is `kipe synthesis study.yaml` followed by
`kipe estimation study.yaml` on the same file.

```yaml
output:
  path: results/aorta_freq

forward_solver:
  factory: "fsix.fluid.inverse:init"
  arguments: {input_file: input/forward_aorta.yaml}

parameters:
  reparameterization: exponential
  select:
    fluid.bc4.R_d: {stddev: 1.0, initial: 8000}

measurements:
  mri_velocity:
    fields: [velocity]
    times: {start: 0.0, stop: 0.4, step: 0.02}
    data: {type: numpy, path: data/mri_velocity}
    spatial_sampling: {type: fenicsx.voxel, resolution: [0.2, 0.2, 0.2]}
    model: {type: pcmri_frequency, venc: 351.05}
    noise: {stddev: 10.0}

estimation:
  particles: simplex
  iterations: 1
  write: [sensitivities, gramian]      # optional filter products

synthesis:
  truth: {type: run, cache: results/aorta/truth}
  noise_seeds: [0, 1, 2]

# future
sensitivity:
  method: finite_differences
  step: 1.0e-3
  write: [sensitivities, fim, eigen]
```

- **Shared sections:** `forward_solver`, `parameters` and `measurements` are used by all
  subcommands. The prior and its reparameterization live together in `parameters`.
- **One measurement section serves both directions:** `synthesis` writes what `estimation`
  reads, so a twin experiment can't mismatch.
- **Per measurement,** not global: data format, spatial sampling and measurement model. Different
  kinds of measurements can be combined in one study.
- **Named entries** (measurements, parameters) appear in logs, outputs and error messages.
- **Explicit backend in type names** where one is involved (`fenicsx.voxel`,
  `fenicsx.io4dolfinx`); backend-free components have plain names (`numpy`, `array`,
  `pcmri_frequency`).
- **Times:** the data says which times exist; `times` selects from them. `synthesis`
  requires `times` and writes them into the generated data.
- **No `time:` section:** the timestep belongs to the solver; the end time follows from the
  last measurement.
- **Truth for `synthesis`:** kipe runs the solver with its nominal (true) parameters and
  generates measurements with the same sampling and model as estimation — works for any
  solver. Optionally it caches the true states, so measurements with other noise or
  sampling cost no further forward solve. Existing solver checkpoints (as used today) can
  be read instead.

## Validation

- **FitzHugh–Nagumo** (no FEniCS at all; a standard ODE parameter-estimation benchmark with an
  attracting limit cycle) proves the interface is really solver-agnostic. Later: a 1D heat
  equation in plain numpy with an MPI-distributed field, to test distributed fields without
  FEniCS.
- **Parallel equivalence** (after 0.1): FitzHugh–Nagumo with particles in parallel
  (`mpirun -n 4`) must give results identical to the serial run.
- **Parity:** results must match roukf on the fsix fluid example, serial and on 2 ranks.
- **Design check:** two CMR feature-tracking scenarios (tracked points; contours via signed
  surface distance) must be expressible without changing the interfaces.

## 4-week plan

| Week | Focus |
|---|---|
| 1 | Scaffold; interfaces; numpy core; FitzHugh–Nagumo twin experiment (synthesis → estimation), without FEniCS |
| 2 | FEniCSx backend; fsix adapters (fluid, solid, FSI); parity with roukf |
| 3 | MRI measurement models; `kipe synthesis` (replaces synthmeas); resume; validation |
| 4 | Docs, CI, release 0.1.0; buffer |

**If time runs short, cut in this order:** sensitivity analysis (FIM), the adaptive FitzHugh–Nagumo
variant, mid-iteration resume, the output hook, `pcmri_frequency`. Particle parallelism is
postponed to after 0.1. **Kept regardless:** FitzHugh–Nagumo, fsix parity, all three
adapters, `synthesis`, docs, the release.
