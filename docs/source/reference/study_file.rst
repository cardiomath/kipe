Study file reference
====================

All keys of a study file: first as an overview, a commented study file, then section by
section, each key with its type, its default (or whether it is required) and what it does.
``<name>`` stands for a name you choose, e.g., of a measurement. :doc:`../user_guide/anatomy` and
:doc:`../user_guide/measurements` explain how the keys work together.

This page is generated from the study file's options (:mod:`kipe.options`), so it describes
exactly what kipe accepts. Unknown keys are an error.

.. include:: ../_generated/study_file.rst

Editor support
--------------

``kipe schema`` prints the JSON schema of the study file. With it, an editor completes the keys,
shows their documentation and marks invalid options while you type:

.. code-block:: bash

    kipe schema > study.schema.json

and as the first line of the study file:

.. code-block:: yaml

    # yaml-language-server: $schema=study.schema.json

This works in editors that use the YAML language server, e.g., VS Code with Red Hat's YAML
extension, or Neovim and Helix with ``yaml-language-server``. The path is relative to the study
file. The schema belongs to the installed kipe version: generate it again after an update.
The example study files have this line already, with the schema in ``examples/``.
