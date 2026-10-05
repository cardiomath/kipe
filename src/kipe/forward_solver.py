"""The forward solver: the interface it implements, and its construction from a study file.

Forward solvers are external. They conform to :class:`ForwardSolver` structurally: no
inheritance from, or import of, kipe required. See :mod:`kipe.examples.fitzhugh_nagumo` for a
reference implementation.

:func:`build_forward_solver` constructs the solver from the ``forward_solver`` section of a
study file and checks that it conforms.
"""

import importlib
import inspect
from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from kipe._types import NDArray_f64
from kipe.options import ForwardSolverOptions

# --- interface implemented by forward solvers --------------------------------------------------

type State = dict[str, NDArray_f64]
"""Model state as named fields: field name -> rank-local array of the field state"""
type Parameters = dict[str, float]
"""Parameter name -> physical value."""


class FieldSpec(Protocol):
    """Layout of one state field on the current MPI rank.

    Any object with these two attributes conforms, e.g., a solver's own frozen dataclass.
    """

    @property
    def local_size(self) -> int:
        """Number of entries of the field held by this rank: the owned entries for a
        distributed field, the full size for a replicated one.
        """
        ...

    @property
    def kind(self) -> Literal["distributed", "replicated"]:
        """``distributed``: each entry is owned by exactly one rank (e.g., the owned dofs of a
        finite element function). ``replicated``: every rank holds the full, identical field
        (e.g., the state of a small lumped-parameter model).
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

    The state must be complete: everything a time step depends on (e.g., previous time levels
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

        The nominal values are those the solver was configured with (e.g., by its input file
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
            ValueError: if ``t1`` cannot be reached exactly (e.g., not on a fixed time grid)
        """
        ...


# --- construction ------------------------------------------------------------------------------


class ForwardSolverError(Exception):
    """The forward solver cannot be constructed from the given options."""


def build_forward_solver(options: ForwardSolverOptions) -> ForwardSolver:
    """Import the factory, check the arguments against its signature and call it.

    Args:
        options: ``forward_solver`` section of the study file

    Returns:
        the forward solver

    Raises:
        ForwardSolverError: if the factory cannot be imported or does not accept the arguments
    """
    factory = _import_factory(options.factory)
    _check_arguments(factory, options.factory, options.arguments)

    solver = factory(**options.arguments)
    _check_conformance(solver, options.factory)

    return solver


def _import_factory(path: str) -> Callable[..., Any]:
    """Import the factory given as ``"module:name"``.

    Args:
        path: import path of the factory

    Returns:
        the factory

    Raises:
        ForwardSolverError: if the module or the name within it cannot be imported
    """
    module_name, name = path.split(":")

    try:
        module = importlib.import_module(module_name)
    except ImportError as err:
        raise ForwardSolverError(
            f"could not import forward solver factory {path!r}: {err}"
        ) from err

    try:
        factory: Callable[..., Any] = getattr(module, name)
    except AttributeError as err:
        raise ForwardSolverError(
            f"could not import forward solver factory {path!r}: "
            f"module {module_name!r} has no attribute {name!r}"
        ) from err

    return factory


def _check_arguments(
    factory: Callable[..., Any], factory_path: str, arguments: dict[str, Any]
) -> None:
    """Check that the factory accepts the arguments, before anything expensive runs.

    Args:
        factory: the forward solver factory
        factory_path: import path of the factory, for the error message
        arguments: keyword arguments from the study file

    Raises:
        ForwardSolverError: if the factory does not accept the arguments
    """
    try:
        signature = inspect.signature(factory)
    except ValueError:  # no signature available (e.g., some builtins): let the call decide
        return

    try:
        signature.bind(**arguments)
    except TypeError as err:
        raise ForwardSolverError(
            f"invalid arguments for forward solver factory {factory_path!r}: {err}. "
            f"Expected: {signature}"
        ) from err


def _check_conformance(solver: Any, factory_path: str) -> None:
    """Check that the solver conforms to the interface specs, defined by ``ForwardSolver``.

    Args:
        solver: forward solver object created by the factory
        factory_path: factory option defined in forward solver options

    Raises:
        ForwardSolverError: if the solver object does not conform to the forward solver protocol
    """
    if not isinstance(solver, ForwardSolver):
        required = [name for name in dir(ForwardSolver) if not name.startswith("_")]
        missing = [name for name in required if not hasattr(solver, name)]
        raise ForwardSolverError(
            f"forward solver factory {factory_path!r} returned a {type(solver).__name__}, "
            f"which lacks the forward solver members {missing}"
        )
