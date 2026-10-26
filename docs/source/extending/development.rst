Development
===========

To add a sampler or a measurement model, read :doc:`architecture`, then
:doc:`write_a_sampler`.

Setup
-----

.. code-block:: bash

    git clone https://github.com/cardiomath/kipe
    cd kipe
    pip install -e ".[dev]"
    pre-commit install --hook-type pre-commit --hook-type commit-msg

For the FEniCSx backend, install kipe into a FEniCSx (dolfinx 0.9) environment instead.

Tests
-----

.. code-block:: bash

    pytest                                     # the core, without dolfinx
    pytest tests/fenicsx                       # the FEniCSx backend, needs dolfinx
    mpirun -n 2 python -m pytest tests/fenicsx # the same on two MPI ranks

The FEniCSx tests are skipped where dolfinx is missing. CI runs all three, the FEniCSx tests in
the ``dolfinx/dolfinx`` Docker image.

Checks and commits
------------------

Pre-commit and CI run yamlfix, ruff (lint and format), mypy, markdownlint and yamllint. Commit
messages follow
`Conventional Commits <https://www.conventionalcommits.org>`_, e.g., ``fix(roukf): ...``. The
scope is optional: the module a commit is about, or the area of other changes;
``.pre-commit-config.yaml`` lists the allowed ones. To run the hooks on all files:

.. code-block:: bash

    pre-commit run --all-files
    pre-commit run yamllint --all-files

Documentation
-------------

The documentation is built with Sphinx, from ``docs/source``: reStructuredText and Markdown
pages, the API pages generated from the docstrings, and the :doc:`../reference/study_file` generated from
the options. Warnings are errors, as in CI:

.. code-block:: bash

    pip install -e ".[docs]"
    python -m sphinx -W --keep-going -c docs -b html docs/source docs/build/html
    # or, the same as a pre-commit hook that also opens the result:
    pre-commit run sphinx-build --all-files --hook-stage manual

The documentation is published from ``main`` to https://cardiomath.github.io/kipe/.

Before a pull request
---------------------

- ``pre-commit run --all-files`` passes;
- the tests pass, those of the FEniCSx backend also on several ranks;
- the documentation builds without warnings; new options are documented in their docstrings,
  from which the :doc:`../reference/study_file` is generated.
