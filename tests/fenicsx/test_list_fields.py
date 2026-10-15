"""``kipe list-fields`` with a FEniCSx solver, and the description of function spaces."""

import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

import basix.ufl
from dolfinx import fem, mesh

from kipe.cli import main
from kipe.fenicsx.forward_solver import describe_function_space
from kipe.options import ForwardSolverOptions, OutputOptions, StudyOptions, dump_study


def test_list_fields_shows_the_function_space(shared_tmp_path, capsys):
    comm = MPI.COMM_WORLD
    study = StudyOptions(
        output=OutputOptions(path=str(shared_tmp_path / "results")),
        forward_solver=ForwardSolverOptions("kipe.fenicsx.examples.heat:Solver", {"n": 16}),
    )
    if comm.rank == 0:
        dump_study(study, shared_tmp_path / "study.yaml")
    comm.barrier()

    assert main(["list-fields", str(shared_tmp_path / "study.yaml")]) == 0
    lines = capsys.readouterr().out.splitlines()
    if comm.rank == 0:  # the size is global: 17 x 17 vertices on any number of ranks
        assert [line.split() for line in lines] == [
            ["field", "kind", "size", "space"],
            ["u", "distributed", "289", "P1"],
        ]
    else:
        assert lines == []


@pytest.mark.parametrize(
    ("element", "description"),
    [
        (("Lagrange", 1), "P1"),
        (("Lagrange", 2, (3,)), "P2, vector 3"),
        (("Lagrange", 1, (3, 3)), "P1, tensor 3x3"),
        (("DG", 0), "DG0"),
        (("N1curl", 1), "N1E1, vector 3"),
    ],
)
def test_describe_function_space(element, description):
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    assert describe_function_space(fem.functionspace(domain, element)) == description


def test_describe_mixed_function_space():
    domain = mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
    taylor_hood = basix.ufl.mixed_element([
        basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,)),
        basix.ufl.element("Lagrange", "tetrahedron", 1),
    ])
    space = fem.functionspace(domain, taylor_hood)
    assert describe_function_space(space) == "mixed: P2, vector 3 + P1"
