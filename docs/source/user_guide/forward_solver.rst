Forward solver requirements
===========================

kipe runs the model through a small interface, the :class:`~kipe.forward_solver.ForwardSolver`
protocol. A solver conforms structurally: it has the members below, and neither imports nor
subclasses anything from kipe. This page lists what kipe expects from a solver; the protocol's
API documentation gives the exact signatures. The FitzHugh–Nagumo solver
(:mod:`kipe.examples.fitzhugh_nagumo`) is a small, complete example.

Construction
------------

The ``forward_solver`` section of the study file names a factory, a class or a function, and
the keyword arguments kipe calls it with. The factory returns the solver: an object with the
members listed under :ref:`forward-solver-interface`.

.. code-block:: yaml

    forward_solver:
      factory: "my_package.my_module:MySolver"  # module:name, imported by kipe
      arguments: {dt: 0.01}                     # MySolver(dt=0.01)

An existing code with its own input file is typically built by a small factory function, which
reads the file:

.. code-block:: python

    def build(config: str) -> MySolver:
        """Build the solver from its own input file."""
        return MySolver(read_my_config(config))

.. code-block:: yaml

    forward_solver:
      factory: "my_package.kipe_interface:build"
      arguments: {config: input/model.yaml}  # build(config="input/model.yaml")

kipe passes the arguments as keywords, as they are in the study file, and does not interpret
them: a path stays a string. kipe does not change the working directory either, so relative
paths are relative to the directory kipe runs in, like every path in the study file. That
includes the paths in the solver's own input file, e.g., to a mesh, unless the solver resolves
them otherwise, e.g., relative to its input file.

Before anything expensive runs, kipe checks the arguments against the factory's signature, and
the object the factory returns against the protocol, so that a misspelled argument or a missing
method is reported right away. Each command builds its own solver, on every MPI rank of the run.

.. _forward-solver-interface:

The interface
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - member
     - what kipe uses it for
   * - :attr:`~kipe.forward_solver.ForwardSolver.state_spec`
     - the names of the state fields and their layout on this rank
   * - :meth:`~kipe.forward_solver.ForwardSolver.initial_state`
     - the start time and the initial state
   * - :meth:`~kipe.forward_solver.ForwardSolver.nominal_parameters`
     - the parameters a study can estimate, with their nominal values
   * - :meth:`~kipe.forward_solver.ForwardSolver.propagate`
     - the state at time ``t1`` from the state at ``t0``, with the given parameters

What kipe relies on
-------------------

- **propagate is stateless.** Its result depends only on its arguments. The filter calls it for
  every sigma point at every measurement time, each time from another state and with other
  parameters, so nothing the solver keeps between calls (time levels, the states of boundary
  models, parameters updated in place) may change the result.
- **The state is complete.** Everything a step depends on is a field of the state: earlier time
  levels of a multistep scheme, the states of lumped-parameter boundary models, and so on. The
  filter corrects the states, so ``propagate`` continues from states the solver has not
  computed itself.
- **propagate lands exactly on t1.** How many internal steps it takes is up to the solver. A
  solver on a fixed time grid raises a ``ValueError`` if ``t1`` is not on the grid; an adaptive
  one builds its integrator from the state on every call.
- **Parameters are physical values, applied per call.** ``propagate`` receives the estimated
  parameters as physical values (the reparameterization is kipe's business), and the
  parameters it does not receive keep their nominal values. Synthesis, for instance, runs the
  solver without parameters, i.e., with the nominal ones.
- **initial_state returns the same state every time.** Every outer iteration of an estimation,
  and every run of synthesis and of the sensitivity analysis, starts from it. Its time is
  before the first measurement time.
- **Arrays are handed over.** ``propagate`` may modify or reuse the arrays of the state it
  receives, and the arrays it returns may share memory with the solver's own data. They only
  need to stay valid until the next call: kipe copies what it keeps.
- **Fields are distributed or replicated.** On several MPI ranks, a ``distributed`` field
  holds the entries this rank owns, each entry on exactly one rank (e.g., the owned degrees of
  freedom of a finite element function); a ``replicated`` field holds the full, identical
  array on every rank (e.g., the state of a small lumped-parameter model). The local sizes stay
  the same from call to call.

If ``propagate`` raises an exception, the estimation stops, and the log shows the step, the
times and the parameters of the sigma point that failed.

Optional capabilities
---------------------

Some features need more from the solver, again as structural protocols:

- :class:`~kipe.forward_solver.SupportsCheckpoints`: ``read_checkpoint(path, t)`` reads the
  state at time ``t`` from the checkpoints of an earlier run. Synthesis then generates data
  from them instead of running the solver (:doc:`synthetic_data`).
- :class:`~kipe.fenicsx.forward_solver.SupportsFunctionSpaces`: ``function_space(field)``
  returns the function space of a distributed field. The FEniCSx samplers need it to
  interpolate a field onto a measurement mesh or to evaluate it at points
  (:doc:`measurements`).

Checking a solver
-----------------

- ``kipe list-parameters study.yaml`` builds the solver, checks it against the protocol, and
  lists its parameters with their nominal values; ``kipe list-fields study.yaml`` lists its
  state fields.
- **Statelessness** is easy to test directly: two calls of ``propagate`` from the same state
  (passed as copies, since ``propagate`` may modify it) and with the same parameters, with
  another call in between, return the same state.
- A **twin experiment** (:doc:`synthetic_data`, :doc:`estimation`) tests the whole setup: with
  little noise, the estimates should approach the parameters the data were generated with.

The heat equation example (:mod:`kipe.fenicsx.examples.heat`) is a FEniCSx solver with function
spaces, which runs on several MPI ranks.
