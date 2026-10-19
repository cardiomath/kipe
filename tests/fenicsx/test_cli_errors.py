"""An error of all ranks is printed once, by rank 0.

Not about FEniCSx: here because the FEniCSx tests are the ones that run on several ranks.
"""

from mpi4py import MPI

from kipe.cli import main
from kipe.options import ForwardSolverOptions, OutputOptions, StudyOptions, dump_study


def test_error_printed_once(shared_tmp_path, capsys):
    comm = MPI.COMM_WORLD
    study = StudyOptions(
        output=OutputOptions(path=str(shared_tmp_path / "results")),
        forward_solver=ForwardSolverOptions("kipe.examples.fitzhugh_nagumo:Missing"),
    )
    if comm.rank == 0:
        dump_study(study, shared_tmp_path / "study.yaml")
    comm.barrier()

    assert main(["list-parameters", str(shared_tmp_path / "study.yaml")]) == 1
    err = capsys.readouterr().err
    if comm.rank == 0:
        assert err.startswith("kipe: error: ") and "Missing" in err
    else:
        assert err == ""
