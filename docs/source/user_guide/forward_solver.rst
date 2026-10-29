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

- The filter calls :meth:`~kipe.forward_solver.ForwardSolver.propagate` once for each sigma
  point (a candidate state and parameter set) at each measurement time. Calls may therefore
  arrive with different states and parameters; each result must depend only on that call's
  arguments. Put everything needed to continue from a state, such as multistep history or
  boundary-model state, in the state itself. See the method's API documentation for details.
- ``propagate`` receives physical parameter values and must return the state exactly at ``t1``.
  Parameters it does not receive retain their nominal values.
- ``initial_state`` must return the same state each time, at a time before the first measurement.
- On each rank, state fields must match the names, layout and local sizes in ``state_spec``
  (:class:`~kipe.forward_solver.FieldSpec`).

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
