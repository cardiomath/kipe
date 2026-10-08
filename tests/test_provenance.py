"""Provenance of a run: run.yaml, study.yaml, environment.yaml, packages.txt, patches."""

import subprocess
from pathlib import Path

import pytest
from ruamel.yaml import YAML

import kipe
from kipe.cli import main
from kipe.estimation import estimate
from kipe.examples.fitzhugh_nagumo import Solver
from kipe.options import load_study
from kipe.provenance import _container, git_state


def _study(tmp_path: Path) -> Path:
    """Write a small FitzHugh-Nagumo study file; return its path."""
    study = {
        "output": {"path": str(tmp_path / "results")},
        "forward_solver": {"factory": "kipe.examples.fitzhugh_nagumo:Solver"},
        "parameters": {
            "reparameterization": "log",
            "select": {"a": {"initial": 0.25, "relative_stddev": 0.1}},
        },
        "measurements": {
            "v": {
                "fields": ["v"],
                "times": [0.5, 1.0],
                "data": {"type": "numpy", "path": str(tmp_path / "v.npz")},
                "noise": {"stddev": 0.05, "seed": 0},
            }
        },
        "estimation": {"iterations": 2},
    }
    path = tmp_path / "study.yaml"
    YAML().dump(study, path)
    return path


def _read(path: Path):
    return YAML(typ="safe").load(path)


def test_estimation_writes_provenance(tmp_path):
    study = _study(tmp_path)
    assert main(["synthesis", str(study)]) == 0
    assert main(["estimation", str(study)]) == 0
    output = tmp_path / "results" / "estimation"

    run = _read(output / "run.yaml")
    assert run["status"] == "finished"
    assert run["finished"] >= run["started"]
    assert run["assimilation_times"] == [0.5, 1.0]

    written = _read(output / "study.yaml")
    assert written["estimation"]["sigma_points"] == "simplex"  # defaults filled in
    assert load_study(output / "study.yaml") == load_study(study)

    environment = _read(output / "environment.yaml")
    assert environment["packages"]["kipe"]["version"] == kipe.__version__
    assert environment["mpi"]["size"] == 1
    assert "numpy" in environment["libraries"]

    packages = (output / "packages.txt").read_text().splitlines()
    assert any(line.startswith("numpy==") for line in packages)

    synthesis = _read(tmp_path / "results" / "synthesis" / "run.yaml")
    assert synthesis["status"] == "finished"
    assert "assimilation_times" not in synthesis


def test_failed_run_is_marked(tmp_path, monkeypatch):
    study = _study(tmp_path)
    assert main(["synthesis", str(study)]) == 0

    def failing_propagate(self, t0, t1, state, parameters):
        raise RuntimeError("Newton did not converge")

    monkeypatch.setattr(Solver, "propagate", failing_propagate)
    with pytest.raises(RuntimeError, match="Newton"):
        main(["estimation", str(study)])

    run = _read(tmp_path / "results" / "estimation" / "run.yaml")
    assert run["status"] == "failed"
    assert run["finished"] is not None


def test_python_api_writes_provenance(tmp_path):
    """estimate() writes the provenance too, not only the CLI."""
    study = load_study(_study(tmp_path))
    assert main(["synthesis", str(tmp_path / "study.yaml")]) == 0
    estimate(study)

    output = tmp_path / "results" / "estimation"
    assert _read(output / "run.yaml")["status"] == "finished"
    assert load_study(output / "study.yaml") == study


def _git(directory: Path, *args: str) -> None:
    """Run git in a directory, independent of the user's configuration (e.g., signing)."""
    config = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    config += ["-c", "commit.gpgsign=false", "-c", "init.defaultBranch=main"]
    subprocess.run(
        ["git", "-C", str(directory), *config, *args], check=True, capture_output=True
    )


def test_git_state_of_a_modified_checkout(tmp_path, monkeypatch):
    module = tmp_path / "kipe_test_checkout.py"
    module.write_text("VALUE = 1\n")
    _git(tmp_path, "init")
    _git(tmp_path, "add", module.name)
    _git(tmp_path, "commit", "-m", "initial")
    module.write_text("VALUE = 2\n")  # uncommitted change
    monkeypatch.syspath_prepend(tmp_path)

    patch = tmp_path / "checkout.patch"
    state = git_state("kipe_test_checkout", patch)

    assert state is not None
    assert state["repository"] == str(tmp_path.resolve())
    assert len(state["commit"]) == 40
    assert state["describe"].endswith("-dirty")
    assert state["dirty"]
    assert "+VALUE = 2" in patch.read_text()


def test_no_git_state_outside_a_checkout(tmp_path, monkeypatch):
    (tmp_path / "kipe_test_not_tracked.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(tmp_path)
    assert git_state("kipe_test_not_tracked") is None


@pytest.mark.parametrize("kind", ["apptainer", "singularity"])
def test_apptainer_container_and_image(monkeypatch, kind):
    """Apptainer and Singularity tell the image; it is recorded with the container."""
    monkeypatch.delenv("APPTAINER_CONTAINER", raising=False)
    monkeypatch.delenv("SINGULARITY_CONTAINER", raising=False)
    monkeypatch.setenv(f"{kind.upper()}_CONTAINER", "/images/fenicsx.sif")
    assert _container() == {"type": kind, "image": "/images/fenicsx.sif"}
