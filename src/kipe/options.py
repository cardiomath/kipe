"""Options of the kipe study file.

A study file is a YAML file with one section per concern. Each subcommand reads the sections
it needs. Relative paths are relative to the working directory kipe is run from.

================== ==================================================== =====================
section            content                                              read by
================== ==================================================== =====================
``output``         result directory, logging                            all
``forward_solver`` how to construct the forward solver                  all
``parameters``     estimated parameters, initial estimate, uncertainty  estimation
``measurements``   observed fields, data, sampling, model, noise        estimation, synthesis
``synthesis``      source of the true state                             synthesis
================== ==================================================== =====================

See ``examples/fitzhugh_nagumo/study.yaml`` for a complete study file.
"""

from dataclasses import field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, PositiveFloat, TypeAdapter
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
    :class:`kipe.forward_solver.ForwardSolver`.
    """

    factory: Annotated[str, Field(pattern=r"^[\w.]+:\w+$")]
    """Import path of the factory, ``"module:name"``, e.g.,
    ``"kipe.examples.fitzhugh_nagumo:Solver"``."""

    arguments: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the factory, as given (no path resolution)."""


type Reparameterization = Literal["log", "multiplicative", "additive"]


@dataclass(frozen=True, config=_CONFIG)
class ParameterPrior:
    r"""Initial estimate of one estimated parameter and its uncertainty.

    The two define the prior (Gaussian in the estimated parameter :math:`\theta`) and the
    starting state of the filter: the standard deviation sets the initial size of the
    sigma-point stencil used for the derivative-free linearization, and the initial weight
    against the measurement noise.

    ``log`` and ``multiplicative`` take a ``relative_stddev``, ``additive`` an absolute
    ``stddev``.
    """

    initial: float | None = None
    """Initial estimate (prior mean), as physical value. The solver's nominal value if not
    given."""

    relative_stddev: PositiveFloat | None = None
    r"""Standard deviation relative to ``initial``, e.g., ``0.3`` for 30%. For ``log`` and
    ``multiplicative``.

    Converted to the standard deviation of :math:`\theta`, to first order around ``initial``:
    :math:`\sigma_\theta` = ``relative_stddev`` / ln 2 for ``log``, :math:`\sigma_\theta` =
    ``relative_stddev`` for ``multiplicative``. So both give the same prior as long as
    ``relative_stddev`` is small. For large values they differ: ``log`` keeps the parameter
    positive, at the price of an asymmetric distribution. ``kipe list-parameters`` shows the
    resulting 1σ range."""

    stddev: PositiveFloat | None = None
    """Absolute standard deviation, in the parameter's physical unit. For ``additive``."""

    reparameterization: Reparameterization | None = None
    """Overrides the section's ``reparameterization`` for this parameter."""


@dataclass(frozen=True, config=_CONFIG)
class ParametersOptions:
    """Selection, initial estimate and reparameterization of the estimated parameters.

    ``kipe list-parameters`` shows the names the forward solver accepts.
    """

    reparameterization: Reparameterization
    r"""How the estimated parameter :math:`\theta` relates to the physical value :math:`\phi`,
    relative to the initial estimate :math:`\phi_0`. Default for all parameters in
    ``select``, each can override it.

    Available reparameterizations:

    - ``log``: :math:`\phi = \phi_0 2^\theta`, starting at :math:`\theta = 0`, with a
      ``relative_stddev``. Keeps :math:`\phi` positive; the initial estimate must be positive.
    - ``multiplicative``: :math:`\phi = \phi_0 \theta`, starting at :math:`\theta = 1`, with
      a ``relative_stddev``. The initial estimate must not be zero; :math:`\phi` can change
      sign.
    - ``additive``: :math:`\phi = \phi_0 + \theta`, starting at :math:`\theta = 0`, with an
      absolute ``stddev``. :math:`\theta` is the deviation from the initial estimate, in the
      parameter's physical unit. Accepts any initial estimate, including zero."""

    select: Annotated[dict[str, ParameterPrior], Field(min_length=1)]
    """Estimated parameters: parameter name -> initial estimate and uncertainty. All other
    parameters keep their nominal values."""

    def __post_init__(self) -> None:
        """Check that each parameter has the standard deviation its reparameterization needs.

        Raises:
            ValueError: if a parameter gives the wrong kind of standard deviation, or none
        """
        for name, prior in self.select.items():
            reparameterization = prior.reparameterization or self.reparameterization
            if reparameterization == "additive":
                if prior.stddev is None or prior.relative_stddev is not None:
                    raise ValueError(
                        f"select.{name}: 'additive' needs an absolute 'stddev' "
                        "(and no 'relative_stddev')"
                    )
            elif prior.relative_stddev is None or prior.stddev is not None:
                raise ValueError(
                    f"select.{name}: '{reparameterization}' needs a 'relative_stddev' "
                    "(and no 'stddev')"
                )


@dataclass(frozen=True, config=_CONFIG)
class TimeRange:
    """Equidistant times from ``start`` to ``stop``, both included."""

    start: float
    """First time."""

    stop: float
    """Last time, included (up to round-off)."""

    step: PositiveFloat
    """Distance between successive times."""

    def __post_init__(self) -> None:
        """Check that the range is not empty.

        Raises:
            ValueError: if ``stop`` is before ``start``
        """
        if self.stop < self.start:
            raise ValueError(f"stop = {self.stop} is before start = {self.start}")


@dataclass(frozen=True, config=_CONFIG)
class NumpyDataOptions:
    """Measurement data in a numpy ``.npz`` file.

    The file holds the arrays ``times``, shape ``(n,)``, and ``values``, shape ``(n, m)``: m
    values at each of the n times.
    """

    type: Literal["numpy"]
    """Data format."""

    path: str
    """Path to the ``.npz`` file."""


@dataclass(frozen=True, config=_CONFIG)
class ArraySamplingOptions:
    """Sample all entries of the observed fields, concatenated in the order of ``fields``."""

    type: Literal["array"] = "array"
    """Kind of spatial sampling."""


@dataclass(frozen=True, config=_CONFIG)
class DifferenceModelOptions:
    """Measured data are the sampled fields; innovation = measured - predicted."""

    type: Literal["difference"] = "difference"
    """Kind of measurement model."""


@dataclass(frozen=True, config=_CONFIG)
class NoiseOptions:
    """Measurement noise: assumed by ``estimation``, added by ``synthesis``."""

    stddev: PositiveFloat
    """Standard deviation of the noise, in the unit of the data."""

    seed: Annotated[int, Field(ge=0)] | None = None
    """Seed of the noise added by ``synthesis`` (required there). The noise of the k-th time
    of the measurement is drawn from ``numpy.random.default_rng([seed, k])``, so it does not
    depend on the other times. Measurements must have different seeds."""


@dataclass(frozen=True, config=_CONFIG)
class MeasurementOptions:
    """One measurement: which fields it observes, its data, and how to compare them."""

    fields: Annotated[list[str], Field(min_length=1)]
    """State fields the measurement observes."""

    data: NumpyDataOptions
    """Where the measured data are: read by ``estimation``, written by ``synthesis``."""

    noise: NoiseOptions
    """Measurement noise."""

    times: list[float] | TimeRange | None = None
    """Measurement times: a list or a range. Required by ``synthesis`` (the times to
    generate); for ``estimation`` a selection from the times in the data, all if not
    given."""

    spatial_sampling: ArraySamplingOptions = field(default_factory=ArraySamplingOptions)
    """How the observed fields are brought to the measurement locations."""

    model: DifferenceModelOptions = field(default_factory=DifferenceModelOptions)
    """How predicted data are computed from the sampled fields and compared with the data."""


@dataclass(frozen=True, config=_CONFIG)
class RunTruthOptions:
    """Generate the data by running the forward solver with its nominal parameters."""

    type: Literal["run"] = "run"
    """Source of the true state."""


@dataclass(frozen=True, config=_CONFIG)
class SynthesisOptions:
    """Generation of synthetic measurement data."""

    truth: RunTruthOptions = field(default_factory=RunTruthOptions)
    """Source of the true state the data are generated from."""


type Particles = Literal["simplex", "canonical", "star", "unique"]


@dataclass(frozen=True, config=_CONFIG)
class EstimationOptions:
    """Parameter estimation with the reduced-order unscented Kalman filter (ROUKF)."""

    particles: Particles = "simplex"
    r"""Sigma-point stencil, the pattern along which the estimates are perturbed:

    - ``simplex``: :math:`p + 1` sigma points, the cheapest.
    - ``canonical``: :math:`2p` sigma points, :math:`\pm` along each parameter.
    - ``star``: the canonical points and the center, :math:`2p + 1` sigma points.
    - ``unique``: one sigma point at the estimate, no spread: no correction. A sanity check
      of the whole pipeline (with the true parameters, the innovations are pure noise).
    """

    iterations: Annotated[int, Field(ge=1)] = 1
    """Number of passes over the measurements. Each pass after the first starts from the
    previous estimate, with the initial uncertainty."""


@dataclass(frozen=True, config=_CONFIG)
class StudyOptions:
    """All sections of a study file."""

    output: OutputOptions
    """Output location and logging."""

    forward_solver: ForwardSolverOptions
    """Construction of the forward solver."""

    parameters: ParametersOptions | None = None
    """Estimated parameters."""

    measurements: Annotated[dict[str, MeasurementOptions], Field(min_length=1)] | None = None
    """Measurements: name -> measurement. The names appear in logs, output and errors."""

    synthesis: SynthesisOptions = field(default_factory=SynthesisOptions)
    """Generation of synthetic measurement data."""

    estimation: EstimationOptions = field(default_factory=EstimationOptions)
    """Parameter estimation."""

    def __post_init__(self) -> None:
        """Check that measurements do not share a noise seed.

        Raises:
            ValueError: if two measurements have the same noise seed
        """
        seeds: dict[int, str] = {}
        for name, measurement in (self.measurements or {}).items():
            seed = measurement.noise.seed
            if seed is None:
                continue
            if seed in seeds:
                raise ValueError(
                    f"measurements {seeds[seed]} and {name} share the noise seed {seed}, "
                    "their noise would be identical"
                )
            seeds[seed] = name


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
