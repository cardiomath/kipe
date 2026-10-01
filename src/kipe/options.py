"""Options of the kipe study file.

A study file is a YAML file with one section per concern. Each subcommand reads the sections
it needs. Relative paths are relative to the working directory kipe is run from.

Example:

.. code-block:: yaml

    output:
      path: results/fhn

    forward_solver:
      factory: "kipe.examples.fitzhugh_nagumo:Solver"
      arguments: {dt: 0.05}
"""

from dataclasses import field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic.dataclasses import dataclass
from ruamel.yaml import YAML, YAMLError

_CONFIG = ConfigDict(extra="forbid", use_attribute_docstrings=True)


class StudyFileError(Exception):
    """The study file cannot be read, or is not a YAML mapping of sections."""


@dataclass(frozen=True, config=_CONFIG)
class OutputOptions:
    """Where and how kipe writes its results."""

    path: str
    """Directory for all results. Subcommands write into subdirectories."""

    log_level: Literal["debug", "info", "warning", "error"] = "info"
    """Minimum level of the log messages kipe writes."""


@dataclass(frozen=True, config=_CONFIG)
class ForwardSolverOptions:
    """How to construct the forward solver.

    kipe imports ``factory`` and calls it with ``arguments`` as keyword arguments. The factory
    can be a class or a function; it must return an object that conforms to
    :class:`kipe.protocols.ForwardSolver`.
    """

    factory: Annotated[str, Field(pattern=r"^[\w.]+:\w+$")]
    """Import path of the factory, ``"module:name"``, e.g.,
    ``"kipe.examples.fitzhugh_nagumo:Solver"``."""

    arguments: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the factory, as given (no path resolution)."""


@dataclass(frozen=True, config=_CONFIG)
class StudyOptions:
    """All sections of a study file."""

    output: OutputOptions
    """Output location and logging."""

    forward_solver: ForwardSolverOptions
    """Construction of the forward solver."""


def load_study(path: str | Path) -> StudyOptions:
    """Read and validate a study file.

    Args:
        path: path to the YAML study file

    Returns:
        validated study options

    Raises:
        StudyFileError: if the file cannot be read, is not valid YAML or is not a mapping
        pydantic.ValidationError: if the options are invalid
    """
    try:
        with open(path) as f:
            data = YAML(typ="safe").load(f)
    except OSError as err:
        raise StudyFileError(f"cannot read study file {path}: {err.strerror}") from err
    except YAMLError as err:
        raise StudyFileError(f"study file {path} is not valid YAML:\n{err}") from err

    if not isinstance(data, dict):
        raise StudyFileError(f"study file {path} must be a YAML mapping of sections")

    return TypeAdapter(StudyOptions).validate_python(data)
