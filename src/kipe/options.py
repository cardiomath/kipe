"""Options of the kipe study file.

A study file is a YAML file with one section per concern: ``output``, ``forward_solver``,
``parameters``, ``measurements``, ``synthesis`` and ``estimation`` (:class:`StudyOptions`).
Each command reads the sections it needs. Relative paths are relative to the working directory
kipe is run from.

The dataclasses of this module are the only definition of the study file: pydantic validates
study files against them, unknown keys are an error, and the documentation's study file
reference is generated from them, with their attribute docstrings as the descriptions.

See ``examples/fitzhugh_nagumo/study.yaml`` for a complete study file.
"""

import string
from dataclasses import field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt, TypeAdapter
from pydantic.dataclasses import dataclass
from ruamel.yaml import YAML, YAMLError

from kipe._yaml import write_yaml

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
    r"""Prior of one estimated parameter: its mean and standard deviation.

    The prior is Gaussian in the estimated parameter :math:`\theta`, and the filter starts
    from it: at the mean, with the standard deviation as the initial size of the sigma-point
    stencil used for the derivative-free linearization, and as the initial weight against the
    measurement noise.

    ``log`` and ``multiplicative`` take a ``relative_stddev``, ``additive`` an absolute
    ``stddev``.
    """

    mean: float
    """Prior mean, as physical value: where the estimation starts."""

    relative_stddev: PositiveFloat | None = None
    r"""Standard deviation relative to the ``mean``, e.g., ``0.3`` for 30%. For ``log`` and
    ``multiplicative``.

    The standard deviation of :math:`\theta` is :math:`\sigma_\theta` = ``relative_stddev``,
    i.e., that of :math:`\ln \phi` for ``log`` and that of :math:`\phi / \phi_0` for
    ``multiplicative``. To first order, both are the standard deviation of :math:`\phi`
    relative to the ``mean``, so both give about the same prior as long as ``relative_stddev``
    is small. For large values they differ: ``log`` keeps the sign of the parameter, at the
    price of an asymmetric distribution. ``kipe list-parameters`` shows the resulting 1σ
    range."""

    stddev: PositiveFloat | None = None
    """Absolute standard deviation, in the parameter's physical unit. For ``additive``."""

    reparameterization: Reparameterization | None = None
    r"""The distribution of this prior: Gaussian in :math:`\theta`, i.e., log-normal in the
    parameter for ``log``, normal for ``multiplicative`` and ``additive``. Overrides the
    section's ``reparameterization`` for this parameter."""


@dataclass(frozen=True, config=_CONFIG)
class ParametersOptions:
    """Priors and reparameterization of the estimated parameters.

    ``kipe list-parameters`` shows the names the forward solver accepts.
    """

    reparameterization: Reparameterization
    r"""How the estimated parameter :math:`\theta` relates to the physical value :math:`\phi`,
    relative to the prior mean :math:`\phi_0`.
    Default for all parameters in ``priors``, each can override it.

    Available reparameterizations:

    - ``log``: :math:`\phi = \phi_0 e^\theta`, starting at :math:`\theta = 0`, with a
      ``relative_stddev``. Keeps the sign of :math:`\phi_0`; the prior mean may be
      positive or negative, but not zero.
    - ``multiplicative``: :math:`\phi = \phi_0 \theta`, starting at :math:`\theta = 1`, with
      a ``relative_stddev``. The prior mean must not be zero; :math:`\phi` can change
      sign.
    - ``additive``: :math:`\phi = \phi_0 + \theta`, starting at :math:`\theta = 0`, with an
      absolute ``stddev``. :math:`\theta` is the deviation from the prior mean, in the
      parameter's physical unit. Accepts any prior mean, including zero."""

    priors: Annotated[
        dict[str, ParameterPrior],
        Field(min_length=1, json_schema_extra={"keys": "a parameter of the solver"}),
    ]
    """Estimated parameters: parameter name -> prior (mean and standard deviation). All other
    parameters keep their nominal values."""

    def __post_init__(self) -> None:
        """Check that each parameter has the standard deviation its reparameterization needs.

        Raises:
            ValueError: if a parameter gives the wrong kind of standard deviation, or none
        """
        for name, prior in self.priors.items():
            reparameterization = prior.reparameterization or self.reparameterization
            if reparameterization == "additive":
                if prior.stddev is None or prior.relative_stddev is not None:
                    raise ValueError(
                        f"priors.{name}: 'additive' needs an absolute 'stddev' "
                        "(and no 'relative_stddev')"
                    )
            elif prior.relative_stddev is None or prior.stddev is not None:
                raise ValueError(
                    f"priors.{name}: '{reparameterization}' needs a 'relative_stddev' "
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
    """Path to the ``.npz`` file. With several noise realizations
    (``synthesis.realizations``), it contains the placeholder ``{realization}``, e.g.,
    ``"data/r{realization}/u.npz"`` or ``"data/u_{realization:02d}.npz"``: each realization has
    its own file. In quotes, since YAML reads braces in ``{type: ..., path: ...}`` as a
    mapping."""


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
    """Measurement noise: assumed by ``estimation``, added by ``synthesis``.

    ``synthesis.noise`` can set another noise for synthesis to add, e.g., none.
    """

    stddev: PositiveFloat
    """Standard deviation of the noise, in the unit of the data."""

    seed: Annotated[int, Field(ge=0)] | None = None
    """Seed of the noise added by ``synthesis`` (required there, unless it adds none, see
    ``synthesis.noise``). The noise of the k-th time of the measurement is drawn from
    ``numpy.random.default_rng([seed, k])``, so it does not depend on the other times.
    Measurements must have different seeds."""


# FEniCSx backend (kipe.fenicsx): its options are plain options like all others, so that the
# study file's schema is complete in this module; nothing here imports dolfinx.


@dataclass(frozen=True, config=_CONFIG)
class MeshSamplingOptions:
    """Interpolate the observed field onto a measurement mesh (FEniCSx backend).

    The field must be a distributed field whose function space the forward solver provides
    (:class:`kipe.fenicsx.forward_solver.SupportsFunctionSpaces`). The measurement mesh need
    not match the solver's mesh. The data are in the order of the mesh file: by vertex for
    ``P1``, by cell for ``DG0``, independent of the number of ranks; for a vector field without
    projection, the components of each vertex or cell one after another.
    """

    type: Literal["fenicsx.mesh"]
    """Kind of spatial sampling."""

    mesh: str
    """Path to the measurement mesh, an XDMF file."""

    mesh_name: str = "mesh"
    """Name of the mesh in the XDMF file (its grid): ``mesh`` as dolfinx writes it; meshio
    writes ``Grid``."""

    element: Literal["P1", "DG0"] = "P1"
    """Finite element on the measurement mesh: vertex values (``P1``) or cell values
    (``DG0``)."""

    projection: (
        Annotated[list[float], Field(min_length=1)] | Literal["surface_normal"] | None
    ) = None
    """What is measured of a vector field; all components if not given. A direction, e.g.,
    ``[0, 0, 1]`` for the z component (normalized by kipe), or ``surface_normal``: the
    component along the cell normal of a surface mesh (needs ``element: DG0``, one normal per
    cell). The normal's orientation follows the order of each cell's vertices in the mesh file,
    as ``ufl.CellNormal``: the right-hand rule for triangles in 3D, the tangent rotated by +90°
    for lines in 2D; it is outward only if the mesh's cells are oriented that way."""

    padding: Annotated[float, Field(ge=0)] = 0.0
    """Distance, in mesh units, by which a measurement point may lie outside the solver's mesh,
    e.g., where a surface mesh cuts slightly across the solver's boundary. Such a point takes
    the value of the field in the nearest cell, extrapolated; a point farther out is an
    error."""

    def __post_init__(self) -> None:
        """Check the projection.

        Raises:
            ValueError: if the direction is zero, or ``surface_normal`` is combined with ``P1``
        """
        if isinstance(self.projection, list) and not any(self.projection):
            raise ValueError("projection: the direction must not be zero")
        if self.projection == "surface_normal" and self.element != "DG0":
            raise ValueError("projection surface_normal needs element DG0 (one normal per cell)")


@dataclass(frozen=True, config=_CONFIG)
class PointSamplingOptions:
    """Evaluate the observed field at given points (FEniCSx backend).

    The field must be a distributed field whose function space the forward solver provides
    (:class:`kipe.fenicsx.forward_solver.SupportsFunctionSpaces`). The data are in the order of
    ``points``; for a vector field, the components of each point one after another.
    """

    type: Literal["fenicsx.points"]
    """Kind of spatial sampling."""

    points: Annotated[list[list[float]], Field(min_length=1)]
    """Coordinates of the measurement points, one list per point, e.g., ``[[0.5, 0.5], [0.2,
    0.8]]`` in 2D. Each point must lie in the solver's mesh."""


# the choices of a measurement, from the core and the backends

type SpatialSamplingOptions = Annotated[
    ArraySamplingOptions | MeshSamplingOptions | PointSamplingOptions,
    Field(discriminator="type"),
]


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

    spatial_sampling: SpatialSamplingOptions = field(default_factory=ArraySamplingOptions)
    """How the observed fields are brought to the measurement locations; the kind is selected
    by ``type``, ``array`` by default."""

    model: DifferenceModelOptions = field(default_factory=DifferenceModelOptions)
    """How predicted data are computed from the sampled fields and compared with the data."""


@dataclass(frozen=True, config=_CONFIG)
class RunTruthOptions:
    """Generate the data by running the forward solver with its nominal parameters."""

    type: Literal["run"] = "run"
    """Source of the true state."""


@dataclass(frozen=True, config=_CONFIG)
class CheckpointsTruthOptions:
    """Generate the data from the checkpoints of an earlier run of the forward solver.

    The solver reads the true states at the measurement times from the checkpoints at
    ``path``; it needs :class:`kipe.forward_solver.SupportsCheckpoints`. The run may differ
    from the model of the ``forward_solver`` section (e.g., in its time step or parameters), as
    long as the solver can read its checkpoints into the model's state.
    """

    type: Literal["checkpoints"]
    """Source of the true state."""

    path: str
    """Location of the checkpoints, as the solver understands it (e.g., the output directory of
    the run)."""


@dataclass(frozen=True, config=_CONFIG)
class DataTruthOptions:
    """Generate the data from given true data, read from files, with noise added.

    E.g., the noise-free data of an earlier synthesis (in ``<output.path>/synthesis/
    noise_free/``), for other noise levels, seeds or realizations without running the solver
    again; or measured data, to test the estimation with more noise. A measurement's ``times``
    select from the file's times; without ``times``, all of them are used. The forward solver is
    built, for the layout of the measurements, but not run.
    """

    type: Literal["data"]
    """Source of the true state."""

    path: str
    """Path to the true data of each measurement, an ``.npz`` file in the format of the data
    files, with the placeholder ``{measurement}`` for the measurement's name, e.g.,
    ``"results/synthesis/noise_free/{measurement}.npz"``. In quotes, since YAML reads braces as
    a mapping. Without the placeholder, all measurements read the same file, so a study with
    several measurements needs it."""


type TruthOptions = Annotated[
    RunTruthOptions | CheckpointsTruthOptions | DataTruthOptions, Field(discriminator="type")
]


@dataclass(frozen=True, config=_CONFIG)
class SynthesisNoiseOptions:
    """Noise that synthesis adds to the data of a measurement."""

    stddev: Annotated[float, Field(ge=0)]
    """Standard deviation of the noise, in the unit of the data; 0 for noise-free data."""


@dataclass(frozen=True, config=_CONFIG)
class SynthesisOptions:
    """Generation of synthetic measurement data."""

    truth: TruthOptions = field(default_factory=RunTruthOptions)
    """Source of the true state the data are generated from: a run of the forward solver, its
    checkpoints, or given data. From a run or checkpoints, synthesis also writes the noise-free
    data, to ``<output.path>/synthesis/noise_free/<measurement>.npz``."""

    noise: Annotated[
        dict[str, SynthesisNoiseOptions],
        Field(json_schema_extra={"keys": "one of the measurements"}),
    ] = field(default_factory=dict)
    """Noise synthesis adds to the data of the listed measurements (measurement name ->
    noise), instead of their ``noise.stddev``, which estimation assumes: for noise-free data
    (``stddev: 0``) or studies with misspecified noise. Measurements not listed get their
    ``noise.stddev``."""

    realizations: PositiveInt = 1
    r"""Number of noise realizations, e.g., for the statistics of the estimates over the noise.
    Synthesis computes the true state once and writes the data of realizations
    :math:`r = 0, \dots, n - 1`, each to the ``data.path`` of the measurements with
    ``{realization}`` replaced by :math:`r`; ``kipe estimation --realization r`` estimates from
    one of them. Realization 0 has the noise of a study without realizations."""


type SigmaPoints = Literal["simplex", "canonical", "unique"]


@dataclass(frozen=True, config=_CONFIG)
class EstimationOptions:
    """Parameter estimation with the reduced-order unscented Kalman filter (ROUKF)."""

    sigma_points: SigmaPoints = "simplex"
    r"""Sigma-point stencil, the pattern along which the estimates are perturbed. Each sigma
    point costs one forward run per measurement time:

    - ``simplex``: :math:`p + 1` sigma points, the cheapest.
    - ``canonical``: :math:`2p` sigma points, :math:`\pm` along each parameter.
    - ``unique``: one sigma point at the estimate, no spread: no correction. A sanity check
      of the whole pipeline (with the true parameters, the innovations are pure noise).
    """

    iterations: Annotated[int, Field(ge=1)] = 1
    """Number of passes over the measurements. Each pass after the first starts from the
    previous estimate, with the initial uncertainty."""


@dataclass(frozen=True, config=_CONFIG)
class SensitivityOptions:
    r"""Sensitivity analysis with the Fisher information (``kipe sensitivity``).

    It computes the sensitivities of the measurements to the estimated parameters, and from
    them the Fisher information the measurements carry about the parameters. The sensitivities
    are evaluated at the nominal parameters, i.e., the truth of a twin experiment, in
    :math:`\theta` of the ``parameters`` section, by centered finite differences with two
    forward runs per parameter. The measurements are those of the study, their predicted data
    noise-free, and weighted with the measurements' ``noise.stddev``, the noise that estimation
    assumes.
    """

    step: PositiveFloat = 0.01
    r"""Step of the finite differences: the relative change of each parameter, e.g., 0.01 for
    1 % (``log``, ``multiplicative``); for ``additive`` parameters, a fraction of their prior
    ``stddev``, which gives the step its unit."""


@dataclass(frozen=True, config=_CONFIG)
class StudyOptions:
    """All sections of a study file."""

    output: OutputOptions
    """Output location and logging."""

    forward_solver: ForwardSolverOptions
    """Construction of the forward solver."""

    parameters: ParametersOptions | None = None
    """Estimated parameters."""

    measurements: (
        Annotated[
            dict[str, MeasurementOptions],
            Field(min_length=1, json_schema_extra={"keys": "a name you choose"}),
        ]
        | None
    ) = None
    """Measurements: name -> measurement. The names appear in logs, output and errors."""

    synthesis: SynthesisOptions = field(default_factory=SynthesisOptions)
    """Generation of synthetic measurement data."""

    estimation: EstimationOptions = field(default_factory=EstimationOptions)
    """Parameter estimation."""

    sensitivity: SensitivityOptions = field(default_factory=SensitivityOptions)
    """Sensitivity analysis with the Fisher information (``kipe sensitivity``)."""

    def __post_init__(self) -> None:
        """Check the synthesis noise's measurement names, that measurements do not share a
        noise seed, and the placeholders of the data paths and of the true data's path.

        Raises:
            ValueError: if the synthesis noise names a measurement the study does not have, two
                measurements have the same noise seed, or a path's placeholders do not fit
        """
        unknown = sorted(self.synthesis.noise.keys() - (self.measurements or {}).keys())
        if unknown:
            raise ValueError(f"synthesis.noise: no measurement {', '.join(unknown)}")

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

        for name, measurement in (self.measurements or {}).items():
            _check_data_path(name, measurement.data.path, self.synthesis.realizations)

        if isinstance(self.synthesis.truth, DataTruthOptions):
            _check_truth_path(self.synthesis.truth.path, len(self.measurements or {}))


def _check_data_path(name: str, path: str, realizations: int) -> None:
    """Check the placeholder of a measurement's data path.

    Args:
        name: name of the measurement
        path: its data path
        realizations: number of noise realizations

    Raises:
        ValueError: if the path has a placeholder other than ``{realization}``, or lacks it
            although there are several realizations
    """
    prefix = f"measurements.{name}.data.path: {path!r}"
    try:
        placeholders = {p for _, p, _, _ in string.Formatter().parse(path) if p is not None}
    except ValueError as err:  # e.g., a single brace
        raise ValueError(f"{prefix}: {err}") from err

    if placeholders - {"realization"}:
        raise ValueError(f"{prefix}: the only placeholder is {{realization}}")
    if realizations > 1 and not placeholders:
        raise ValueError(
            f"{prefix} needs the placeholder {{realization}}: with {realizations} "
            "realizations, each needs its own file"
        )
    try:
        path.format(realization=0)
    except ValueError as err:  # e.g., {realization:x2}
        raise ValueError(f"{prefix}: {err}") from err


def _check_truth_path(path: str, measurements: int) -> None:
    """Check the placeholder of the path to the true data, ``synthesis.truth: {type: data}``.

    Args:
        path: the path
        measurements: number of measurements of the study

    Raises:
        ValueError: if the path has a placeholder other than ``{measurement}``, or lacks it
            although the study has several measurements
    """
    prefix = f"synthesis.truth.path: {path!r}"
    try:
        placeholders = {p for _, p, _, _ in string.Formatter().parse(path) if p is not None}
    except ValueError as err:  # e.g., a single brace
        raise ValueError(f"{prefix}: {err}") from err

    if placeholders - {"measurement"}:
        raise ValueError(f"{prefix}: the only placeholder is {{measurement}}")
    if measurements > 1 and not placeholders:
        raise ValueError(
            f"{prefix} needs the placeholder {{measurement}}: with {measurements} "
            "measurements, each needs its own file"
        )


def truth_path(truth: DataTruthOptions, measurement: str) -> str:
    """Return the path to the true data of a measurement, ``synthesis.truth: {type: data}``.

    Args:
        truth: the truth options
        measurement: name of the measurement

    Returns:
        ``truth.path``, with ``{measurement}`` replaced
    """
    return truth.path.format(measurement=measurement)


def data_path(measurement: MeasurementOptions, realization: int = 0) -> str:
    """Return the path to a measurement's data of one noise realization.

    Args:
        measurement: the measurement options
        realization: number of the noise realization

    Returns:
        ``data.path``, with ``{realization}`` replaced
    """
    return measurement.data.path.format(realization=realization)


def estimation_directory(study: StudyOptions, realization: int = 0) -> Path:
    """Return the output directory of an estimation.

    Args:
        study: the study options
        realization: number of the noise realization the estimation uses

    Returns:
        ``<output.path>/estimation``; with several realizations, its subdirectory of the
        realization, e.g., ``r07``, numbered with as many digits as the largest number

    Raises:
        StudyFileError: if the study has no such realization
    """
    n = study.synthesis.realizations
    if not 0 <= realization < n:
        raise StudyFileError(
            f"realization {realization}: the study has realizations 0 to {n - 1} "
            "(synthesis.realizations)"
        )

    directory = Path(study.output.path) / "estimation"
    if n == 1:
        return directory
    return directory / f"r{realization:0{len(str(n - 1))}d}"


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


def study_schema() -> dict[str, Any]:
    """Return the JSON schema of the study file, e.g., for editors (``kipe schema``).

    It is pydantic's schema of :class:`StudyOptions`, in JSON Schema draft 2020-12, with the
    options' docstrings as descriptions.

    Returns:
        the JSON schema, as a mapping that :func:`json.dumps` writes
    """
    schema = TypeAdapter(StudyOptions).json_schema()
    del schema["title"]  # "StudyOptions"
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "kipe study file",
        **schema,
    }


def dump_study(study: StudyOptions, path: str | Path) -> None:
    """Write study options as a study file, which :func:`load_study` reads back.

    All options are written, defaults included, ``null`` for those not set. Useful to generate
    study files from Python, e.g., for a sweep over noise seeds.

    Args:
        study: the study options
        path: where to write the study file
    """
    write_yaml(path, TypeAdapter(StudyOptions).dump_python(study, mode="json"))
