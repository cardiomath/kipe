"""Protocols that external components implement to work with kipe.

:class:`ForwardSolver` is the contract between kipe and a forward model. Solvers conform
structurally (no inheritance from, or import of, kipe required).
See :mod:`kipe.examples.fitzhugh_nagumo` for a reference implementation.
"""

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

type NDArray64 = npt.NDArray[np.float64]
type State = dict[str, NDArray64]
"""Model state as named fields: field name -> rank-local array of the field state"""
type Parameters = dict[str, float]
"""Parameter name -> physical value."""


class FieldSpec(Protocol):
    """Layout of one state field on the current MPI rank.

    Any object with these two attributes conforms, e.g. a solver's own frozen dataclass.
    """

    @property
    def local_size(self) -> int:
        """Number of entries of the field held by this rank: the owned entries for a
        distributed field, the full size for a replicated one.
        """
        ...

    @property
    def kind(self) -> Literal["distributed", "replicated"]:
        """``distributed``: each entry is owned by exactly one rank (e.g. the owned dofs of a
        finite element function). ``replicated``: every rank holds the full, identical field
        (e.g. the state of a small lumped-parameter model).
        """
        ...


@runtime_checkable
class ForwardSolver(Protocol):
    """Forward model as seen by kipe.

    From kipe's point of view the solver is stateless: state, parameters and time are passed
    in on every call of :meth:`~.timestep`/:meth:`~.propagate` and results are returned.
    Whatever the solver keeps internally (meshes, assembled operators, parameter objects it
    updates in place) must not make a call's result depend on previous calls.

    States are exchanged as named numpy fields, local to each MPI rank (see
    :class:`FieldSpec`). Both directions are lenient for solvers: a state passed in is handed
    over (the solver may modify or reuse its arrays), and returned arrays may share memory with
    the solver's internal data. They only need to stay valid until the next call, kipe copies
    whatever it keeps. kipe passes parameters as physical values, any reparameterization is
    kipe's business.

    The state must be complete: everything a time step depends on (e.g. previous time levels
    of a multistep scheme, states of lumped-parameter boundary models) is a field of the state,
    so that a step can be restarted from any given state.
    """

    def initial_state(self) -> tuple[float, State]:
        """Return the start time and the initial state.

        Returns:
            - start time ``t0``
            - initial state, valid until the next call
        """
        ...

    @property
    def state_spec(self) -> Mapping[str, FieldSpec]:
        """Names and layout of the state fields.

        Every state returned or accepted by the solver has exactly these fields, with these
        local sizes.
        """
        ...

    def nominal_parameters(self) -> Parameters:
        """Return the estimable parameters and their nominal values.

        The nominal values are those the solver was configured with (e.g. by its input file
        or constructor). kipe uses the names to validate the parameters selected in a study,
        and the values as default initial guesses.

        Returns:
            parameter name -> nominal physical value dict (a new dict that the caller may modify)
        """
        ...

    def timestep(self, t0: float, state: State, parameters: Parameters) -> tuple[float, State]:
        """Advance the state by one step of the solver's own choosing.

        Args:
            t0: time of ``state``
            state: state at ``t0`` (handed over, may be modified or reused)
            parameters: physical values of the selected parameters, applied for this call.
                Parameters not in ``parameters`` keep their nominal values.

        Returns:
            - time ``t1`` reached by the step
            - state at ``t1``, valid until the next call
        """
        ...

    def propagate(self, t0: float, t1: float, state: State, parameters: Parameters) -> State:
        """Advance the state from ``t0`` to exactly ``t1``.

        The solver may take any number of internal steps, but must land exactly on ``t1``.
        Adaptive solvers rebuild their integrator from ``state`` on every call rather than
        keeping step-size history across calls.

        Args:
            t0: time of ``state``
            t1: target time, ``t1 > t0``
            state: state at ``t0`` (handed over, may be modified or reused)
            parameters: physical values of the selected parameters, applied for this call.
                Parameters not in ``parameters`` keep their nominal values.

        Returns:
            state at ``t1``, valid until the next call

        Raises:
            ValueError: if ``t1`` cannot be reached exactly (e.g. not on a fixed time grid)
        """
        ...
