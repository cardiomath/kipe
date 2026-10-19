Write a spatial sampler
=======================

A spatial sampler brings the model's fields to where a measurement is taken: it interpolates
them onto a measurement mesh, evaluates them at points, integrates them over a surface, and so
on. This page walks through a small, complete one, ``fenicsx.points``, which evaluates a field
at given points (:mod:`kipe.fenicsx.points`). A new sampler follows the same steps. Read
:doc:`../user_guide/measurements` and :doc:`architecture` first: they explain what a sampler is
for and how kipe uses it.

Start without kipe
------------------

Write the observation first as a plain function, outside kipe: a ``dolfinx.fem.Function`` in, a
numpy array out. Try it on the solver's mesh, with a field whose values you know (e.g., a
linear function, which P1 represents exactly), on one rank and on several. Turning it into a
sampler is then a matter of the steps below.

.. code-block:: python

    def observe(u: dolfinx.fem.Function) -> numpy.ndarray:
        """The values of u where the measurement is taken."""

What a sampler must do
----------------------

A sampler is a subclass of :class:`kipe.sampling.SpatialSampler`:

- **The sampled vector** has ``size`` entries over all ranks. Each rank returns some of them,
  at the positions ``global_indices``, and **each entry is returned by exactly one rank**. The
  filter sums over the entries of all ranks, so an entry returned twice would count twice.
- **The positions do not depend on the number of ranks.** The data are stored in this order, so
  data written on one rank must be readable on eight. ``fenicsx.mesh`` uses the rows of the
  measurement mesh file, ``fenicsx.points`` the order of the points in the study file.
- :meth:`~kipe.sampling.SpatialSampler.sample` gets the observed fields as numpy arrays: for
  a finite element field, this rank's owned degrees of freedom. It returns this rank's entries,
  in the order of ``global_indices``, as a new array.

Everything that does not change between calls (searching the points, building interpolation
data) belongs in the constructor: ``sample`` runs for every sigma point at every measurement
time.

1. The options
--------------

The options say what a study file can set for the sampler: a frozen dataclass in
:mod:`kipe.options`, with a literal ``type``. The name carries the backend, ``fenicsx.``. The
docstrings are the documentation of the study file, here and in the API pages:

.. literalinclude:: ../../../src/kipe/options.py
   :pyobject: PointSamplingOptions

It joins the choices of ``spatial_sampling``, the union that pydantic selects from by
``type``:

.. literalinclude:: ../../../src/kipe/options.py
   :start-at: type SpatialSamplingOptions
   :end-at: ]

2. The sampler
--------------

The class gets everything it needs as arguments, not as options, so that it can be tested
without a study file:

.. literalinclude:: ../../../src/kipe/fenicsx/points.py
   :pyobject: PointSampler

The constructor:

- **Which rank evaluates which point.** A rank searches its *owned* cells for the points;
  searching its ghost cells too would find a cell on two ranks. A point on the boundary
  between the cells of two ranks is still found by both, so the lowest of them takes it: the
  ``MIN`` reduction over all ranks. A point that no rank found lies outside the mesh, which is
  an error. Without that check, it would get no value, or a wrong one, silently.
- **The positions.** The values of point i are at position i, or ``i * value_size + c`` for
  component c of a vector field. They depend only on the order of the points in the study file.

``sample``:

- **The field's values** come as this rank's owned degrees of freedom. A cell at the partition
  boundary also uses degrees of freedom owned by another rank, the ghosts:
  ``scatter_forward`` fetches their values before the evaluation.

3. Building it from the options
-------------------------------

A build function turns the options into the sampler. It takes what it needs from the solver:
here, the observed field and its function space, from
:func:`~kipe.fenicsx.sampling.observed_function_space`, which every FEniCSx sampler can use:

.. literalinclude:: ../../../src/kipe/fenicsx/points.py
   :pyobject: build_point_sampler

kipe calls it from :func:`kipe.sampling.build_sampler`, which has one ``case`` per kind of
sampler. A backend's sampler is imported inside its case, so that kipe works without dolfinx
as long as no study uses it:

.. literalinclude:: ../../../src/kipe/sampling.py
   :pyobject: build_sampler

4. Tests
--------

The tests use a field whose values are known exactly, and run on one rank and on several:

.. literalinclude:: ../../../tests/fenicsx/test_point_sampler.py
   :pyobject: test_values_at_the_points

.. literalinclude:: ../../../tests/fenicsx/test_point_sampler.py
   :pyobject: test_each_point_on_exactly_one_rank

``_gathered`` collects the entries of all ranks by their ``global_indices``, so the result is
the same on any number of ranks. The test file also checks a vector field, the path from the
study's options through ``build_sampler``, and the errors
(``tests/fenicsx/test_point_sampler.py``):

.. code-block:: bash

    python -m pytest tests/fenicsx/test_point_sampler.py
    mpirun -n 2 python -m pytest tests/fenicsx/test_point_sampler.py
    mpirun -n 3 python -m pytest tests/fenicsx/test_point_sampler.py

With more ranks, some ranks hold few or none of the points: a case worth testing too.

5. Use it
---------

In a study file:

.. code-block:: yaml

    measurements:
      u:
        fields: [u]
        times: {start: 0.05, stop: 0.5, step: 0.05}
        data: {type: numpy, path: data/u.npz}
        noise: {stddev: 0.001, seed: 0}
        spatial_sampling:
          type: fenicsx.points
          points: [[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]

Before a pull request:

- ``ruff check``, ``ruff format --check`` and ``mypy`` pass (``pre-commit run --all-files``);
- the tests pass on one rank and on several;
- the new options appear in :doc:`../user_guide/measurements`, with an example.
