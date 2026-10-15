Measurements
============

A measurement is a series of measured data, and the description of how to predict these data
from the state of the model. kipe compares the prediction with the data at each measurement
time: in an estimation, to correct the parameters; in a synthesis, it writes the prediction,
with noise, as synthetic data.

Each measurement is an entry under ``measurements`` in the study file, by name. The
FitzHugh–Nagumo example (:mod:`kipe.examples.fitzhugh_nagumo`) measures both fields of its
model:

.. literalinclude:: ../../examples/fitzhugh_nagumo/study.yaml
   :language: yaml
   :start-at: measurements:
   :end-before: synthesis:

A measurement names the state ``fields`` it observes, its ``times``, where its ``data`` are, its
``noise``, and how the fields are brought to the measurement locations (``spatial_sampling``).
``kipe list-fields study.yaml`` lists the solver's fields, with their kind, size and, for a
FEniCSx solver, their function space. :doc:`study_file` lists all keys of
``measurements.<name>``; this page explains how they work together.

From the model state to the data
--------------------------------

kipe predicts a measurement's data from the model state in two steps:

1. The **spatial sampler** evaluates the observed fields where the measurement is taken, e.g.,
   interpolates a finite element field onto the cells of a measurement mesh, giving the
   sampled values :math:`y`.
2. The **measurement model** predicts the data from them, :math:`\hat{z} = M(y)`, and compares
   them with the measured data :math:`z`, giving the innovation, which drives the filter's
   correction. The ``difference`` model takes the sampled values as the data,
   :math:`\hat{z} = y`, and the innovation is :math:`z - \hat{z}`.

Together, they are the measurement's observation operator, the same in synthesis and
estimation: the data kipe generates are the data it expects. :doc:`architecture` explains the
split and how to add a sampler or a model.

Times
-----

``times`` is a list, e.g., ``[0.1, 0.2, 0.5]``, or a range, ``{start: 0.05, stop: 0.5, step:
0.05}``. For now, all measurements of a study must have the same times. They must lie after the
forward solver's start time, on its time grid if it has a fixed time step.

Data
----

Measurement data are ``.npz`` files with two arrays (:class:`~kipe.options.NumpyDataOptions`):

- ``times``, shape ``(n,)``: the measurement times;
- ``values``, shape ``(n, m)``: the m values of the measurement at each time.

The order of the m values is the sampler's (see below), and it does not depend on the number of
MPI ranks: data written on one rank can be read on eight. To use data from elsewhere, e.g.,
from an image, write them in this order:

.. code-block:: python

    from kipe.measurements import write_numpy

    write_numpy("data/u.npz", times, values)  # = numpy.savez(path, times=..., values=...)

Noise
-----

``noise.stddev`` is the standard deviation of the measurement noise: ``estimation`` assumes
it, ``synthesis`` adds it, drawn with ``noise.seed``. ``synthesis.noise`` can add another noise,
e.g., none for noise-free data, while estimation still assumes ``noise.stddev``:

.. code-block:: yaml

    synthesis:
      noise:
        u: {stddev: 0}

Spatial sampling
----------------

``array``
^^^^^^^^^

All entries of the observed fields, one after another in the order of ``fields``
(:class:`~kipe.options.ArraySamplingOptions`). The fields must be replicated, i.e., held in full
by every MPI rank, as the states of the FitzHugh–Nagumo model or of a lumped-parameter model.
Fields that are distributed over the ranks, as finite element fields, need a sampler of their
backend.

``fenicsx.mesh``
^^^^^^^^^^^^^^^^

A finite element field, interpolated onto a measurement mesh, which need not match the
solver's: it can be coarser, or a surface of it. A vector field can be measured as a whole, or
its component along a direction or the surface normal (``projection``). The heat example
(:mod:`kipe.fenicsx.examples.heat`) measures the temperature on a coarser mesh:

.. literalinclude:: ../../examples/heat/study.yaml
   :language: yaml
   :start-at: measurements:
   :end-before: synthesis:

The values are in the order of the mesh file: by vertex for ``P1``, by cell for ``DG0``.
Meshes written by meshio need ``mesh_name: Grid``.

**Points outside the solver's mesh.** A surface mesh, or a coarser mesh, can cut slightly
across the boundary of the solver's mesh, so that some of its points lie outside. kipe stops
with an error then. ``padding`` lets points within a distance outside take the value of the
nearest cell:

.. code-block:: yaml

    spatial_sampling:
      type: fenicsx.mesh
      mesh: data/endocardium.xdmf
      element: DG0
      projection: surface_normal
      padding: 6.3e-4  # in the unit of the mesh

``fenicsx.points``
^^^^^^^^^^^^^^^^^^

A finite element field, evaluated at given points, in their order:

.. code-block:: yaml

    spatial_sampling:
      type: fenicsx.points
      points: [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]

Another sampler? :doc:`write_a_sampler` walks through this one, step by step.

Synthetic data
--------------

``kipe synthesis`` generates the data of all measurements at their ``times``: with their
observation operator, from a true state, with the noise above. :ref:`synthetic-data` explains
how, and what for.
