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
    kipe synthesis study.yaml                # the measurement data, from the true parameters
    kipe estimation study.yaml               # the estimates of a, b, c, from a wrong guess
    kipe plot-estimation study.yaml --truth  # the estimates with their 1σ range
    # without a display, e.g., on a cluster: the same with --terminal

Everything is configured in the study file, ``study.yaml``: the forward solver, the estimated
parameters with their priors (initial guess and uncertainty), the measurements, and the
filter. The results, a log, and run provenance go to the directory named by ``output.path``;
here, that is ``results/``. :doc:`anatomy` introduces the study file, and :doc:`estimation`
explains the results.

Your own model
--------------

kipe calls your model through a small interface, the
:class:`~kipe.forward_solver.ForwardSolver` protocol, which your solver implements without
importing kipe. :doc:`forward_solver` lists what kipe expects from a solver; the
FitzHugh–Nagumo solver, :mod:`kipe.examples.fitzhugh_nagumo`, is a complete example.

FEniCSx models
--------------

A FEniCSx solver exchanges the owned degrees of freedom of its fields as arrays, like any other
state, and also provides their function spaces
(:class:`~kipe.fenicsx.forward_solver.SupportsFunctionSpaces`). Measurements can then
interpolate a field onto a measurement mesh, which need not match the solver's, or evaluate it
at points (:doc:`measurements`).

The heat equation example (:mod:`kipe.fenicsx.examples.heat`) shows a complete setup, including
execution across several MPI ranks:

.. code-block:: bash

    cd kipe/examples/heat
    python measurement_mesh.py  # the measurement mesh
    kipe synthesis study.yaml
    mpirun -n 2 kipe estimation study.yaml

Its solver is a template for your own.
