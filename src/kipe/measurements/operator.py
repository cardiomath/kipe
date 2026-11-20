r"""The observation operator of a measurement, :math:`\mathcal{H} = M \circ S`.

It combines the spatial sampler :math:`S` (:mod:`kipe.sampling`) and the measurement model
:math:`M` (:mod:`kipe.measurements.models`): it takes a model state to predicted data.
"""

from dataclasses import dataclass, field

from mpi4py import MPI

import numpy as np

from kipe._types import NDArray_f64, NDArray_i64
from kipe.forward_solver import ForwardSolver, State
from kipe.measurements.models import MeasurementModel, build_model
from kipe.measurements.plane import Plane, PlaneModel, orient_normal
from kipe.options import MeasurementOptions, PlaneModelOptions, StudyFileError
from kipe.sampling import SpatialSampler, build_sampler


@dataclass
class _PlaneOrientation:
    """Retain the first normal and latest oriented normal for one observation operator."""

    toward: tuple[float, float, float] | None = None
    reference: NDArray_f64 | None = field(default=None, init=False)
    latest: NDArray_f64 | None = field(default=None, init=False)

    def record(self, plane: Plane) -> None:
        """Observe a fitted plane without changing its measurement coefficients.

        Args:
            plane: fitted geometry, before orientation relative to this operator's history

        Raises:
            ValueError: if the reference point or normal does not determine an orientation
        """
        self.latest = orient_normal(
            plane.normal,
            plane.centroid,
            normal_toward=self.toward,
            reference_normal=self.reference,
        )
        if self.reference is None:
            self.reference = self.latest.copy()


@dataclass(frozen=True)
class ObservationOperator:
    r"""The observation operator :math:`\mathcal{H} = M \circ S` of a measurement.

    Takes a model state :math:`\chi` to predicted data :math:`\hat{z} = \mathcal{H}(\chi)`:
    the spatial sampler :math:`S` evaluates the observed fields at the measurement locations,
    the measurement model :math:`M` predicts the data from them.
    """

    fields: list[str]
    """Names of the observed state fields."""

    sampler: SpatialSampler
    """The spatial sampler :math:`S`."""

    model: MeasurementModel
    """The measurement model :math:`M`."""

    comm: MPI.Comm = field(default_factory=lambda: MPI.COMM_WORLD, repr=False, compare=False)
    """Communicator used to assemble global inputs when the model requests them."""

    _orientation: _PlaneOrientation | None = field(default=None, repr=False, compare=False)
    _input_indices: list[NDArray_i64] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Cache the sampler's validated positions for models needing complete input."""
        if not self.model.gather_input:
            return
        indices = self.comm.gather(self.sampler.global_indices.copy(), root=0)
        object.__setattr__(self, "_input_indices", indices)

    @property
    def normal(self) -> NDArray_f64 | None:
        """Copy of the latest plane normal on root, or None before prediction or after failure.

        The first normal is retained separately across failed predictions. Non-plane
        operators and non-root ranks return None. A new operator establishes a new reference.
        """
        if self.comm.rank != 0:
            return None
        normal = None if self._orientation is None else self._orientation.latest
        return None if normal is None else normal.copy()

    @property
    def size(self) -> int:
        """Number of entries of the predicted data, over all ranks, as the model lays them out
        (:meth:`~kipe.measurements.models.MeasurementModel.layout`).
        """
        return self.model.layout(self.sampler)[0]

    @property
    def global_indices(self) -> NDArray_i64:
        """Positions of this rank's entries in the measurement's data, as the model lays them
        out (:meth:`~kipe.measurements.models.MeasurementModel.layout`).
        """
        if self.model.gather_input and self.comm.rank != 0:
            return np.empty(0, dtype=np.int64)
        return self.model.layout(self.sampler)[1]

    def __call__(self, state: State) -> NDArray_f64:
        r"""Return the predicted data :math:`\hat{z} = \mathcal{H}(\chi)` of a model state.

        Args:
            state: the model state :math:`\chi`

        Returns:
            this rank's entries of the predicted data, at the positions :attr:`global_indices`
        """
        y = self.sampler.sample({name: state[name] for name in self.fields})

        return self.predict(y)

    def predict(self, y: NDArray_f64) -> NDArray_f64:
        """Apply the model, assembling global input and distributing failures if requested.

        Args:
            y: this rank's sampled entries in the sampler's local global-index order

        Returns:
            this rank's predicted entries; gathered models produce data only on root

        Raises:
            ValueError: on every rank for an invalid gathered input, model prediction,
                orientation reference or output shape
        """
        if not self.model.gather_input:
            return self.model.predict(y)
        if self._orientation is not None:
            self._orientation.latest = None
        parts = self.comm.gather(y, root=0)
        predicted = np.empty(0, dtype=np.float64)
        error = None
        if self.comm.rank == 0:
            assert parts is not None and self._input_indices is not None
            try:
                coordinates = np.empty(self.sampler.size, dtype=np.float64)
                for indices, values in zip(self._input_indices, parts, strict=True):
                    if values.ndim != 1 or len(values) != len(indices):
                        raise ValueError("sampled entries do not match the sampler layout")
                    coordinates[indices] = values
                predicted = self.model.predict(coordinates)
                if predicted.shape != (self.size,):
                    raise ValueError("predicted entries do not match the model output layout")
            except ValueError as err:
                error = str(err)
                if self._orientation is not None:
                    self._orientation.latest = None
        error = self.comm.bcast(error, root=0)
        if error is not None:
            raise ValueError(error)
        return predicted


def build_observation_operator(
    name: str,
    options: MeasurementOptions,
    solver: ForwardSolver,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> ObservationOperator:
    """Construct the observation operator of a measurement.

    Args:
        name: name of the measurement, for error messages
        options: the measurement options
        solver: the forward solver
        comm: communicator of the run

    Returns:
        the observation operator

    Raises:
        StudyFileError: if the measurement observes a field the solver does not have, or one
            its sampler cannot sample
    """
    state_spec = solver.state_spec
    unknown = [f for f in options.fields if f not in state_spec]
    if unknown:
        raise StudyFileError(
            f"measurements.{name}: the forward solver has no field {', '.join(unknown)}. "
            f"Available: {', '.join(state_spec)}"
        )

    try:
        sampler = build_sampler(options.spatial_sampling, options.fields, solver, comm)
        model = build_model(options.model)
        orientation = None
        if isinstance(options.model, PlaneModelOptions):
            assert isinstance(model, PlaneModel)
            orientation = _PlaneOrientation(options.model.normal_toward)
            model.on_fit = orientation.record
        return ObservationOperator(
            fields=options.fields,
            sampler=sampler,
            model=model,
            comm=comm,
            _orientation=orientation,
        )
    except StudyFileError as err:
        raise StudyFileError(f"measurements.{name}: {err}") from err
