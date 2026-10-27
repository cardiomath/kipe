r"""The observation operator of a measurement, :math:`\mathcal{H} = M \circ S`.

It combines the spatial sampler :math:`S` (:mod:`kipe.sampling`) and the measurement model
:math:`M` (:mod:`kipe.measurements.models`): it takes a model state to predicted data.
"""

from dataclasses import dataclass

from mpi4py import MPI

from kipe._types import NDArray_f64, NDArray_i64
from kipe.forward_solver import ForwardSolver, State
from kipe.measurements.models import MeasurementModel, build_model
from kipe.options import MeasurementOptions, StudyFileError
from kipe.sampling import SpatialSampler, build_sampler


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
        return self.model.layout(self.sampler)[1]

    def __call__(self, state: State) -> NDArray_f64:
        r"""Return the predicted data :math:`\hat{z} = \mathcal{H}(\chi)` of a model state.

        Args:
            state: the model state :math:`\chi`

        Returns:
            this rank's entries of the predicted data, at the positions :attr:`global_indices`
        """
        y = self.sampler.sample({name: state[name] for name in self.fields})

        return self.model.predict(y)


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
    except StudyFileError as err:
        raise StudyFileError(f"measurements.{name}: {err}") from err

    return ObservationOperator(
        fields=options.fields, sampler=sampler, model=build_model(options.model)
    )
