"""What a FEniCSx forward solver provides on top of :class:`kipe.forward_solver.ForwardSolver`.

kipe exchanges states with the solver as numpy arrays. Samplers on meshes (``fenicsx.*``) turn
the array of a field back into a ``dolfinx.fem.Function``, for which they need the field's
function space: the solver provides it with :meth:`SupportsFunctionSpaces.function_space`.
A solver without it can still be used with ``array`` measurements.
:func:`describe_function_space` describes such a space briefly, for ``kipe list-fields``.
"""

from typing import Protocol, runtime_checkable

import basix.ufl
from dolfinx import fem
from dolfinx.mesh import MeshTags


@runtime_checkable
class SupportsMeshTags(Protocol):
    """A solver that provides tags for selecting vertices of a state field's mesh."""

    def mesh_tags(self, field: str) -> MeshTags:
        """Return tags on the mesh of ``function_space(field)``.

        Args:
            field: name of the state field

        Returns:
            tags of mesh entities, e.g., facet tags defining boundary surfaces
        """
        ...


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


def describe_function_space(space: fem.FunctionSpace) -> str:
    """Describe the element of a function space in a few characters, for ``kipe list-fields``.

    Examples: ``P1``, ``DG0``, ``P2, vector 3``, ``P1, tensor 3x3``, ``N1E1, vector 3``,
    ``mixed: P2, vector 3 + P1``.

    Args:
        space: the function space

    Returns:
        the element family and degree, ``DG`` for discontinuous Lagrange, and the value shape
        of a non-scalar element; the sub-elements of a mixed element
    """
    return _describe_element(space.ufl_element())


def _describe_element(element: basix.ufl._ElementBase) -> str:
    """Describe an element, see :func:`describe_function_space`.

    Args:
        element: the element

    Returns:
        the description
    """
    if element.is_mixed:
        return "mixed: " + " + ".join(map(_describe_element, element.sub_elements))

    family = element.family_name
    if family == "P" and element.discontinuous:
        family = "DG"
    description = f"{family}{element.degree}"

    shape = element.reference_value_shape
    if not shape:
        return description
    kind = "vector" if len(shape) == 1 else "tensor"
    return f"{description}, {kind} {'x'.join(map(str, shape))}"
