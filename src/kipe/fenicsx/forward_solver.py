"""What a FEniCSx forward solver provides on top of :class:`kipe.forward_solver.ForwardSolver`.

kipe exchanges states with the solver as numpy arrays. Samplers on meshes (``fenicsx.*``) turn
the array of a field back into a ``dolfinx.fem.Function``, for which they need the field's
function space: the solver provides it with :meth:`SupportsFunctionSpaces.function_space`.
A solver without it can still be used with ``array`` measurements.
"""

from typing import Protocol, runtime_checkable

from dolfinx import fem


@runtime_checkable
class SupportsFunctionSpaces(Protocol):
    """A forward solver that provides the function spaces of its distributed state fields."""

    def function_space(self, field: str) -> fem.FunctionSpace:
        """Return the function space of a state field.

        The field's array (see :attr:`kipe.forward_solver.ForwardSolver.state_spec`) holds the
        values of the space's degrees of freedom owned by this rank, in their local order.

        Args:
            field: name of the state field

        Returns:
            the function space
        """
        ...
