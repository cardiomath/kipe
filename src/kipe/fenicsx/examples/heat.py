r"""Heat equation in dolfinx, as a minimal kipe forward solver for FEniCSx.

.. math::

    \partial_t u - \kappa \Delta u = a \sin(\pi x) \sin(\pi y) \quad \text{in } (0, 1)^2,
    \qquad u = 0 \text{ on the boundary}, \qquad u(0) = 0,

with P1 elements and backward Euler. The exact solution is

.. math::

    u(t) = \frac{a}{2\pi^2\kappa} \left(1 - e^{-2\pi^2\kappa t}\right) \sin(\pi x) \sin(\pi y),

so the diffusivity :math:`\kappa` and the source amplitude :math:`a` can both be identified
from a time series of :math:`u`: early on, :math:`u` grows like :math:`a t`; the approach to
the steady state :math:`a / (2\pi^2\kappa)` reveals :math:`\kappa`.

The state is one field, ``u``, distributed over the MPI ranks: each rank holds the values of
its owned degrees of freedom. The public members of :class:`Solver`, together with
:class:`FieldSpec`, are the complete kipe interface, including the FEniCSx capability
:meth:`Solver.function_space`; the private methods are the model. This module is fully
independent of kipe.
"""

from dataclasses import dataclass
from typing import Literal

from mpi4py import MPI

from petsc4py import PETSc

import numpy as np
import numpy.typing as npt
import ufl
from dolfinx import fem, mesh
from dolfinx.fem import petsc

type NDArray_f64 = npt.NDArray[np.float64]
type State = dict[str, NDArray_f64]
type Parameters = dict[str, float]


@dataclass(frozen=True)
class FieldSpec:
    """Layout of one state field on the current MPI rank."""

    local_size: int
    """Number of entries of the field held by this rank: its owned degrees of freedom."""

    kind: Literal["distributed", "replicated"] = "distributed"
    """``distributed``: each entry is owned by exactly one rank."""


class Solver:
    r"""Heat equation on the unit square with P1 elements and backward Euler.

    Args:
        n: number of cells per side of the unit square (``2 n^2`` triangles)
        dt: time step size
        kappa: nominal value of the diffusivity :math:`\kappa`
        a: nominal value of the source amplitude :math:`a`
        comm: communicator over which the mesh is distributed
    """

    def __init__(
        self,
        n: int = 16,
        dt: float = 0.01,
        kappa: float = 0.1,
        a: float = 1.0,
        comm: MPI.Comm = MPI.COMM_WORLD,
    ) -> None:
        self._dt = dt
        self._nominal = {"kappa": kappa, "a": a}

        domain = mesh.create_unit_square(comm, n, n)
        self._V = fem.functionspace(domain, ("Lagrange", 1))
        self._n_owned = self._V.dofmap.index_map.size_local

        tdim = domain.topology.dim
        domain.topology.create_connectivity(tdim - 1, tdim)
        boundary = fem.locate_dofs_topological(
            self._V, tdim - 1, mesh.exterior_facet_indices(domain.topology)
        )
        self._bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary, self._V)

        self._u = fem.Function(self._V)  # u at the current time step
        self._kappa = fem.Constant(domain, PETSc.ScalarType(kappa))
        self._a = fem.Constant(domain, PETSc.ScalarType(a))

        u, v = ufl.TrialFunction(self._V), ufl.TestFunction(self._V)
        x = ufl.SpatialCoordinate(domain)
        f = self._a * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        self._lhs = fem.form(
            (u * v + dt * self._kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
        )
        self._rhs = fem.form((self._u + dt * f) * v * ufl.dx)

        self._A = petsc.create_matrix(self._lhs)
        self._b = petsc.create_vector(self._rhs)
        self._solver = PETSc.KSP().create(comm)
        self._solver.setOperators(self._A)
        self._solver.setType(PETSc.KSP.Type.PREONLY)
        self._solver.getPC().setType(PETSc.PC.Type.LU)
        self._solver.getPC().setFactorSolverType("mumps")

    # kipe interface

    @property
    def state_spec(self) -> dict[str, FieldSpec]:
        """Names and layout of the state fields: one distributed field, ``u``."""
        return {"u": FieldSpec(local_size=self._n_owned)}

    def nominal_parameters(self) -> Parameters:
        """Return the estimable parameters and their nominal values.

        Returns:
            parameter name -> nominal value dict (a new dict)
        """
        return dict(self._nominal)

    def initial_state(self) -> tuple[float, State]:
        """Return the start time and the initial state.

        Returns:
            - start time ``t0 = 0``
            - initial state :math:`u = 0`
        """
        return 0.0, {"u": np.zeros(self._n_owned)}

    def propagate(self, t0: float, t1: float, state: State, parameters: Parameters) -> State:
        """Propagate the state from ``t0`` to ``t1`` with backward Euler steps.

        Args:
            t0: time of ``state``
            t1: target time
            state: state at ``t0``
            parameters: parameter values for this call, overriding the nominal values

        Returns:
            state at ``t1``

        Raises:
            ValueError: if ``t1 - t0`` is not a positive multiple of ``dt``
        """
        n = round((t1 - t0) / self._dt)
        if n < 1 or not np.isclose(t0 + n * self._dt, t1, rtol=0, atol=1e-12 * max(1, abs(t1))):
            raise ValueError(f"cannot reach t1 = {t1} from t0 = {t0} with dt = {self._dt}")

        merged = self._nominal | parameters
        self._kappa.value = merged["kappa"]
        self._a.value = merged["a"]
        self._assemble_matrix()  # depends on kappa

        self._u.x.array[: self._n_owned] = state["u"]
        self._u.x.scatter_forward()
        for _ in range(n):
            self._step()

        return {"u": self._u.x.array[: self._n_owned].copy()}

    # FEniCSx capability

    def function_space(self, field: str) -> fem.FunctionSpace:
        """Return the function space of a state field, for samplers on meshes.

        Args:
            field: name of the state field

        Returns:
            the function space whose owned degrees of freedom hold the field's entries

        Raises:
            KeyError: if there is no such field
        """
        if field != "u":
            raise KeyError(field)

        return self._V

    # model

    def _assemble_matrix(self) -> None:
        """Assemble the system matrix for the current diffusivity."""
        self._A.zeroEntries()
        petsc.assemble_matrix(self._A, self._lhs, bcs=[self._bc])
        self._A.assemble()
        self._solver.setOperators(self._A)  # refactorize

    def _step(self) -> None:
        """Advance ``u`` by one backward Euler step."""
        with self._b.localForm() as b_local:
            b_local.set(0.0)
        petsc.assemble_vector(self._b, self._rhs)
        petsc.apply_lifting(self._b, [self._lhs], bcs=[[self._bc]])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(self._b, [self._bc])

        self._solver.solve(self._b, self._u.x.petsc_vec)
        self._u.x.scatter_forward()
