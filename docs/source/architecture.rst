Architecture
============

This page is for those who want to extend kipe: use it with a new kind of measurement (e.g.,
MRI phase images instead of velocities on a mesh), compare model and data differently, read
another data format, or connect their own forward solver. It explains how kipe is put together
and where each of these extensions goes. The API pages document each part in detail.

Principles
----------

**The core is solver-agnostic.** The core (``kipe``) uses numpy and MPI only. Everything that
needs a discretization library lives in a backend subpackage (``kipe.fenicsx``), imported only
when a study uses it. The FitzHugh–Nagumo example (:mod:`kipe.examples.fitzhugh_nagumo`)
runs without any backend and keeps it that way.

**The solver owns its settings.** Time step, end time, output paths and solver tolerances
belong to the forward solver. kipe passes only state, parameters and time, so no setting is
defined twice.

**A study is one file.** The study file (:mod:`kipe.options`) holds everything a run needs. An
archived study file reproduces an HPC run; command-line options only change what is printed.
The same file serves synthesis and estimation, so a twin experiment cannot mismatch.

**Extensions are small and explicit.** A new component is a class, its options and one
``case`` in a build function, all in kipe. There are no plugin registries or entry points:
reading the build function shows every choice there is.

Overview
--------

The study file is read into options dataclasses. Build functions turn the options into the
objects that do the work: the forward solver, the parameterization and, for each measurement,
its observation operator. Synthesis runs the solver with its nominal parameters, or reads the
states from the solver's checkpoints, and writes the measurement data. The data can also come
from an acquisition, converted to kipe's format outside kipe. Estimation reads the data and
runs the filter.

.. mermaid::

   flowchart TD
       study[/"study file"/] --> options["options<br/>kipe.options"]
       options --> solver["forward solver<br/>(external)"]
       options --> operators["observation operators<br/>H = M ∘ S<br/>kipe.measurements"]
       options --> parameterization["parameterization<br/>kipe.parameters"]
       solver -- "run, or read checkpoints" --> synthesis["kipe synthesis<br/>kipe.synthesis"]
       operators --> synthesis
       synthesis --> data[/"measurement data"/]
       acquired[/"acquired data<br/>(e.g., MRI)"/] -. "converted" .-> data
       solver --> estimation["kipe estimation<br/>kipe.estimation"]
       operators --> estimation
       parameterization --> estimation
       data --> estimation
       estimation --> history[/"estimation history"/]

Source-derived dependencies
---------------------------

The conceptual diagram above shows how a study flows through kipe. These generated diagrams
instead show imports between Python modules; they are derived from the source code by
``scripts/generate_diagrams.py``.

.. mermaid:: _static/diagrams/dependencies-core.mmd
   :alt: Dependencies between modules in the core kipe package.
   :caption: Core package module dependencies.

The optional FEniCSx backend has a separate graph to keep the core view readable.

.. mermaid:: _static/diagrams/dependencies-fenicsx.mmd
   :alt: Dependencies between modules in the optional kipe.fenicsx backend.
   :caption: FEniCSx backend module dependencies.

Regenerate these diagrams locally with ``python scripts/generate_diagrams.py`` before building
the documentation. The docs extra provides Pyreverse; Sphinx renders the generated Mermaid
source using the existing Mermaid extension.

The forward solver
------------------

kipe sees the forward model through :class:`kipe.forward_solver.ForwardSolver`, a protocol:
a solver conforms structurally and does not import kipe. It is constructed by the
``factory`` named in the study file, with the ``arguments`` given there; from Python, the
arguments can be any objects, e.g., the solver's own options.

- **Stateless calls.** :meth:`~kipe.forward_solver.ForwardSolver.initial_state` and
  :meth:`~kipe.forward_solver.ForwardSolver.propagate` take and return the state; a call's
  result must not depend on earlier calls. The filter propagates every sigma point, each from
  its own state, one after another, with the same solver instance.
- **Named state fields.** The state is a mapping from field names to numpy arrays, local to
  the MPI rank. A field is ``distributed`` (each entry owned by one rank, e.g., the owned
  degrees of freedom of a finite element function) or ``replicated`` (every rank holds all
  of it, e.g., the state of a lumped-parameter model). Measurements refer to fields by name.
- **Named parameters.** The solver lists what can be estimated, with nominal values. The study
  selects parameters by name and sets their prior; parameters it does not select keep their
  nominal values.
- **Backend capabilities.** A backend may need more from the solver: FEniCSx samplers need the
  function space of a field (:class:`kipe.fenicsx.forward_solver.SupportsFunctionSpaces`).
  Such a capability is a separate protocol, so a solver without it still works with the core
  samplers.

Measurements
------------

A :class:`~kipe.measurements.Measurement` is data at measurement times, with a noise level,
and the observation operator that predicts those data from a model state. The observation
operator :math:`\mathcal{H} = M \circ S` has two parts:

- the **spatial sampler** :math:`S` (:mod:`kipe.sampling`) evaluates the observed fields at
  the measurement locations: interpolation onto a measurement mesh, a projection onto a
  direction or the surface normal, all entries of a field;
- the **measurement model** :math:`M` (:mod:`kipe.measurements`) predicts the data from the
  sampled values and compares the prediction with the measured data, giving the innovation.

The diagram shows one observation operator of the overview, with its two parts, and how its
result enters the filter.

.. mermaid::

   flowchart LR
       state["model state χ"] --> sampler["SpatialSampler S<br/>.sample"]
       sampler -- "sampled values y" --> predict["MeasurementModel M<br/>.predict"]
       predict -- "predicted data ẑ" --> innovation["MeasurementModel M<br/>.innovation"]
       data[/"measured data z"/] --> innovation
       innovation -- "innovation Γ(z, ẑ)" --> filter["ROUKF correction<br/>kipe.roukf"]

Synthesis and estimation use the same operator, so the model that generates synthetic data
is the one that inverts them; a sensitivity analysis would use it too.

**Why two parts.** Samplers depend on the discretization, models on the acquisition physics
(e.g., the phase of an MRI signal, a Fourier transform, a mask). Models act on arrays only,
so they work with every backend. Split this way, N samplers and M models give N × M
combinations from N + M classes.

**Why "observation".** In the data-assimilation literature, the observation operator is the
map from the model state to predicted data. kipe keeps the word for that composition and
calls its parts by what they do.

Parallel layout
---------------

Everything in kipe that communicates takes the MPI communicator as an argument (``comm``,
``MPI.COMM_WORLD`` by default); only the command line uses ``MPI.COMM_WORLD`` directly. The
same code runs serially and in parallel, and sigma points can later run in parallel in groups
of ranks, each group with its own communicator.

**Each entry on one rank.** A sampler knows the size of the sampled vector over all ranks
and, on each rank, the ``global_indices`` of the entries that rank returns. Every entry is
returned by exactly one rank. The filter's reductions (sums over entries) are then plain
``allreduce`` operations, with nothing counted twice.

**Data independent of the rank count.** Measurement data are stored in the order of the
global indices, which the sampler defines independently of the partition (e.g.,
:class:`~kipe.fenicsx.sampling.MeshSampler` uses the row of each vertex or cell in the mesh
file). Synthetic noise depends only on an entry's position. So data synthesized on 1 rank are
the same as on 8, and either can be used with any number of ranks.

Extending kipe
--------------

Each kind of component follows the same pattern:

1. **The class**, in the module of its kind: a subclass of the abstract base class.
2. **Its options**, a frozen dataclass in :mod:`kipe.options` with a literal ``type`` field,
   added to the union of its kind. The ``type`` selects it in the study file. Backend
   components carry the backend in the name (``fenicsx.mesh``); core components have plain
   names (``array``, ``difference``).
3. **One case** in the build function of its kind, which constructs the class from its
   options. A backend case imports the backend inside the case, so the core never imports it.
4. **Tests**, with the FitzHugh–Nagumo example for core components, with the heat example
   (:mod:`kipe.fenicsx.examples.heat`) for FEniCSx components.

========================= ================================================ =====================================================
component                 base class                                        build function
========================= ================================================ =====================================================
spatial sampler           :class:`kipe.sampling.SpatialSampler`            :func:`kipe.sampling.build_sampler`
measurement model         :class:`kipe.measurements.MeasurementModel`      :func:`kipe.measurements.build_model`
========================= ================================================ =====================================================

**A spatial sampler**: :doc:`write_a_sampler` walks through a complete one. It sets ``size``
and ``global_indices`` and implements
:meth:`~kipe.sampling.SpatialSampler.sample`, which takes the observed fields (rank-local
arrays) and returns this rank's sampled entries. It may take what it needs from the solver at
construction, e.g., function spaces, through a backend capability.

**A measurement model** implements :meth:`~kipe.measurements.MeasurementModel.predict` and
:meth:`~kipe.measurements.MeasurementModel.innovation`. It sees this rank's entries as arrays.
A model that needs all entries at once (e.g., a Fourier transform) will also need the
communicator; none does so far.

**A data format** is read where the data are read
(:func:`kipe.estimation.estimate` and :func:`kipe.synthesis.synthesize`, through
:func:`~kipe.measurements.read_numpy` and :func:`~kipe.measurements.write_numpy`). kipe has
one format so far, ``numpy``; the second one will add a build function for readers, following
the pattern above.

**A forward solver** lives outside kipe. It implements
:class:`~kipe.forward_solver.ForwardSolver`, plus the capabilities of the backend it uses.
:mod:`kipe.examples.fitzhugh_nagumo` is the reference for a solver without a backend,
:mod:`kipe.fenicsx.examples.heat` for a FEniCSx solver; both are tested against the protocol.
