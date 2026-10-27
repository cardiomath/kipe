"""Common functionality for the FEniCSx spatial samplers."""

from dolfinx import fem

from kipe.fenicsx.forward_solver import SupportsFunctionSpaces
from kipe.forward_solver import ForwardSolver
from kipe.options import StudyFileError


def observed_function_space(
    kind: str, fields: list[str], solver: ForwardSolver
) -> tuple[str, fem.FunctionSpace]:
    """Return the field a FEniCSx sampler observes, and its function space.

    Args:
        kind: the sampler's ``type``, for error messages, e.g., ``"fenicsx.mesh"``
        fields: names of the fields the measurement observes; exactly one
        solver: the forward solver, which provides the field's function space

    Returns:
        - the field's name
        - its function space, on the solver's mesh

    Raises:
        StudyFileError: if the measurement does not observe exactly one distributed field whose
            function space the solver provides
    """
    if len(fields) != 1:
        raise StudyFileError(f"{kind} samples one field, got {', '.join(fields)}")
    (field,) = fields

    if solver.state_spec[field].kind != "distributed":
        raise StudyFileError(f"{kind} needs a distributed field, {field} is replicated")
    if not isinstance(solver, SupportsFunctionSpaces):
        raise StudyFileError(
            f"{kind} needs the function space of {field}, but the forward solver has no "
            "function_space method (see kipe.fenicsx.forward_solver)"
        )

    return field, solver.function_space(field)
