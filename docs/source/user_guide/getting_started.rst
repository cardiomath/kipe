Getting started
===============

Installation
------------

kipe needs Python ≥ 3.12 and an MPI library, for ``mpi4py``. Use your system's MPI, or install
one with pip:

.. code-block:: bash

    pip install kipe
    pip install mpich  # only if there is no MPI library on your system

The latest development version installs from GitHub:

.. code-block:: bash

    pip install "kipe @ git+https://github.com/cardiomath/kipe"

**For FEniCSx models,** kipe also needs dolfinx (0.9), which pip does not install. Install kipe
into the Python environment of your FEniCSx installation (e.g., Spack, conda or the
``dolfinx/dolfinx`` Docker image): its MPI is then the one kipe uses.

A first twin experiment
-----------------------

In a twin experiment, the true parameters are known: kipe generates the measurement data
itself, with the model's nominal parameters, and then estimates the parameters from them,
starting from a wrong initial guess. The FitzHugh–Nagumo example
(:mod:`kipe.examples.fitzhugh_nagumo`) estimates the three parameters of a small ODE model:

.. code-block:: bash

    git clone https://github.com/cardiomath/kipe
    cd kipe/examples/fitzhugh_nagumo
    kipe synthesis study.yaml     # the measurement data, from the true parameters
    kipe estimation study.yaml    # the estimate of a, b, c, from a wrong initial guess
    kipe plot study.yaml --truth  # the estimates with their 1σ range; --terminal without a display

Everything is configured in the study file, ``study.yaml``: the forward solver, the estimated
parameters with their initial guess and uncertainty, the measurements and the filter. The
results, a log and the provenance of the runs go to the directory ``output.path`` names, here
``results/``. :doc:`anatomy` introduces the study file and :doc:`estimation` explains the results.

Your own model
--------------

kipe calls your model through a small interface, the
:class:`~kipe.forward_solver.ForwardSolver` protocol: its state fields
(:attr:`~kipe.forward_solver.ForwardSolver.state_spec`), its parameters with their nominal
values (:meth:`~kipe.forward_solver.ForwardSolver.nominal_parameters`), the initial state
(:meth:`~kipe.forward_solver.ForwardSolver.initial_state`), and a stateless step from one time
to the next (:meth:`~kipe.forward_solver.ForwardSolver.propagate`). Your solver conforms
structurally: it does not import kipe.

The study file names a factory that builds your solver, a class or a function, and the
arguments kipe passes to it:

.. code-block:: yaml

    forward_solver:
      factory: "my_package.my_module:MySolver"  # module:name, imported by kipe
      arguments: {dt: 0.01}                     # MySolver(dt=0.01)

``kipe list-parameters study.yaml`` then shows the parameters your solver offers. The
FitzHugh–Nagumo solver, :mod:`kipe.examples.fitzhugh_nagumo`, is a complete example, docstrings included.

FEniCSx models
--------------

A FEniCSx solver exchanges the owned degrees of freedom of its fields as arrays, like any other
state, and also provides their function spaces
(:class:`~kipe.fenicsx.forward_solver.SupportsFunctionSpaces`). Measurements can then
interpolate a field onto a measurement mesh, which need not match the solver's, or evaluate it
at points (:doc:`measurements`).

The heat equation example (:mod:`kipe.fenicsx.examples.heat`) shows a complete setup, also on
several MPI ranks:

.. code-block:: bash

    cd kipe/examples/heat
    python measurement_mesh.py  # the measurement mesh
    kipe synthesis study.yaml
    mpirun -n 2 kipe estimation study.yaml

Its solver is a template for your own.
