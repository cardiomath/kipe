"""Construction of the forward solver from the ``forward_solver`` section of a study file."""

import importlib
import inspect
from collections.abc import Callable
from typing import Any

from kipe.options import ForwardSolverOptions
from kipe.protocols import ForwardSolver


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
