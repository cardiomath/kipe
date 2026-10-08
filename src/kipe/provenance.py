"""Provenance of a run: what ran, with which code, where; written next to the results.

:func:`recorded_run` writes, to the output directory of a subcommand (e.g.,
``<output.path>/estimation/``), next to the log and the results:

- ``run.yaml``: what the study does not hold: the status, ``running`` while the run lasts,
  then ``finished`` or ``failed``, with start and end time; for estimation, the assimilation
  times (which may come from the data files).
- ``study.yaml``: the validated study options, with all defaults filled in: what actually ran.
  The comments of the study file are not kept; the CLI logs the study file's path.
- ``environment.yaml``: host, platform, Python, MPI, thread settings, and the versions of kipe,
  of the forward solver's package and of the main libraries. For kipe and the solver's package,
  also their git state if they are run from a git checkout (e.g., an editable install).
- ``<package>.patch``: the uncommitted changes of such a checkout (``git diff HEAD``), so that
  a run from a modified checkout can be reproduced. Untracked files are not included.
- ``packages.txt``: all installed Python distributions, ``name==version``.
- ``spack.yaml`` and ``spack.lock``, or ``conda_environment.yml``: the environment, if the run
  happens in a Spack or conda environment.

Only rank 0 writes. The files are overwritten by the next run in the same directory.
"""

import functools
import importlib.util
import logging
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from mpi4py import MPI

from kipe._yaml import write_yaml
from kipe.options import StudyOptions, dump_study

logger = logging.getLogger(__name__)

_LIBRARIES = ["numpy", "mpi4py", "petsc4py", "fenics-dolfinx"]  # distribution names
_THREADS = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]


@contextmanager
def recorded_run(
    output: Path,
    study: StudyOptions,
    extra: Mapping[str, Any] | None = None,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> Iterator[None]:
    """Write the provenance of the run in the ``with`` block, and its status at the end.

    Args:
        output: output directory of the subcommand
        study: the study options
        extra: further entries of ``run.yaml`` that the study file does not hold, e.g., the
            assimilation times
        comm: communicator of the run; only its rank 0 writes

    Yields:
        nothing; the run happens in the ``with`` block
    """
    run: dict[str, Any] = {
        "status": "running",
        "started": _now(),
        "finished": None,
    } | dict(extra or {})

    if comm.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_yaml(output / "run.yaml", run)
        dump_study(study, output / "study.yaml")
        environment = _environment(output, study, comm)
        write_yaml(output / "environment.yaml", environment)
        _write_packages(output / "packages.txt")
        logger.info("%s", _summary(environment))

    try:
        yield
    except BaseException:
        if comm.rank == 0:
            write_yaml(output / "run.yaml", run | {"status": "failed", "finished": _now()})
        raise

    if comm.rank == 0:
        write_yaml(output / "run.yaml", run | {"status": "finished", "finished": _now()})
        logger.info("%-11s%s", "run info", output / "run.yaml")


def git_state(module: str, patch: Path | None = None) -> dict[str, Any] | None:
    """Return the git state of the checkout a module is run from, if any.

    Args:
        module: import name of the module or package, e.g., ``"kipe"``
        patch: file to write the uncommitted changes to (``git diff HEAD``), if there are any

    Returns:
        ``repository``, ``commit``, ``describe`` (e.g., ``v0.1-3-gabc1234-dirty``) and
        ``dirty``; None if the module's file is not tracked in a git checkout, e.g., for an
        installed package, or if git is not available
    """
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return None
    origin = Path(spec.origin)
    directory = origin.parent

    # tracked, not just inside a repository: an installed package may sit in a venv within one
    if _git(directory, "ls-files", "--error-unmatch", origin.name) is None:
        return None

    status = _git(directory, "status", "--porcelain", "--untracked-files=no")
    state = {
        "repository": _git(directory, "rev-parse", "--show-toplevel"),
        "commit": _git(directory, "rev-parse", "HEAD"),
        "describe": _git(directory, "describe", "--tags", "--always", "--dirty"),
        "dirty": bool(status),
    }
    if state["dirty"] and patch is not None:
        patch.write_text((_git(directory, "diff", "HEAD") or "") + "\n")

    return state


def _environment(output: Path, study: StudyOptions, comm: MPI.Comm) -> dict[str, Any]:
    """Return where and with what the run happens; write patches and environment copies.

    Args:
        output: output directory, for the patch files and the Spack or conda environment
        study: the study options, for the forward solver's package
        comm: communicator of the run

    Returns:
        the content of ``environment.yaml``
    """
    packages = {"kipe": _package("kipe", output)}
    solver_package = study.forward_solver.factory.split(":")[0].split(".")[0]
    if solver_package != "kipe":  # e.g., kipe.examples
        packages[solver_package] = _package(solver_package, output)

    environment: dict[str, Any] = {
        "timestamp": _now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "os": _os(),
        "container": _container(),
        "python": platform.python_version(),
        "prefix": sys.prefix,  # the Python environment, e.g., a venv
        "mpi": {"size": comm.size, "library": _mpi_library()},
        "threads": {name: os.environ[name] for name in _THREADS if name in os.environ},
        "packages": packages,
        "libraries": {name: _version(name) for name in _LIBRARIES if _version(name)},
    }

    if "SPACK_ENV" in os.environ:
        spack = Path(os.environ["SPACK_ENV"])
        environment["spack"] = str(spack)
        for name in ["spack.yaml", "spack.lock"]:
            if (spack / name).exists():
                shutil.copyfile(spack / name, output / name)

    if "CONDA_DEFAULT_ENV" in os.environ:
        environment["conda"] = os.environ["CONDA_DEFAULT_ENV"]
        export = _run([os.environ.get("MAMBA_EXE", "conda"), "env", "export"])
        if export is not None:
            (output / "conda_environment.yml").write_text(export + "\n")

    return environment


def _package(module: str, output: Path) -> dict[str, Any]:
    """Return the version and git state of a package; write its uncommitted changes.

    Args:
        module: import name of the package
        output: output directory, for ``<module>.patch``

    Returns:
        ``version`` (None if not installed as a distribution) and ``git`` (None if not run
        from a git checkout)
    """
    # usually, the distribution has the name of the package; packages_distributions() finds
    # the others (e.g., dolfinx in fenics-dolfinx) but is slow and misses editable installs
    version = _version(module)
    if version is None:
        distributions = metadata.packages_distributions().get(module)
        version = _version(distributions[0]) if distributions else None

    return {
        "version": version,
        "git": git_state(module, output / f"{module}.patch"),
    }


def _version(distribution: str) -> str | None:
    """Return the installed version of a distribution, without importing it.

    Args:
        distribution: distribution name, e.g., ``"fenics-dolfinx"``

    Returns:
        the version; None if not installed
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _os() -> str | None:
    """Return the name of the operating system, e.g., ``Ubuntu 24.04.1 LTS``.

    Inside a container, this is the base system of the image.

    Returns:
        the name from ``/etc/os-release``; None where there is none, e.g., on macOS
    """
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        return None


def _container() -> dict[str, str | None] | None:
    """Return the container the run happens in, and its image if the container tells.

    Apptainer (formerly Singularity) passes the path of the image; Podman usually writes the
    image name to ``/run/.containerenv``; Docker does not tell the image.

    Returns:
        ``type`` (``apptainer``, ``singularity``, ``podman`` or ``docker``) and ``image``;
        None outside a container
    """
    for kind in ["apptainer", "singularity"]:
        variable = f"{kind.upper()}_CONTAINER"
        if variable in os.environ:
            return {"type": kind, "image": os.environ[variable]}

    containerenv = Path("/run/.containerenv")
    if containerenv.exists():
        lines = containerenv.read_text().splitlines()
        entries = dict(line.split("=", 1) for line in lines if "=" in line)
        return {"type": "podman", "image": entries.get("image", "").strip('"') or None}

    if Path("/.dockerenv").exists():
        return {"type": "docker", "image": None}

    return None


def _mpi_library() -> str:
    """Return the name and version of the MPI library, e.g., ``Open MPI v5.0.3``.

    The first line of the library's version report, up to the first comma: Open MPI adds its
    build details there.
    """
    first_line = MPI.Get_library_version().strip("\0 \n").splitlines()[0]
    return first_line.split(",")[0].strip()


def _summary(environment: Mapping[str, Any]) -> str:
    """Return a one-line summary for the log: versions, git state, host and ranks.

    Args:
        environment: the content of ``environment.yaml``

    Returns:
        e.g., ``kipe 0.1.0 (v0.1.0-dirty), fsix 2.0.1, host, 4 ranks``
    """
    parts = []
    for name, package in environment["packages"].items():
        git = package["git"]
        parts.append(f"{name} {package['version']}" + (f" ({git['describe']})" if git else ""))
    size = environment["mpi"]["size"]
    parts += [environment["host"], f"{size} rank" + ("s" if size > 1 else "")]
    return ", ".join(parts)


def _write_packages(path: Path) -> None:
    """Write all installed distributions, one ``name==version`` per line.

    Args:
        path: where to write
    """
    path.write_text(_packages())


@functools.cache  # takes about 0.1 s; the installed packages do not change during a process
def _packages() -> str:
    """Return all installed distributions, one ``name==version`` per line, sorted by name."""
    packages = {f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()}
    return "\n".join(sorted(packages, key=str.lower)) + "\n"


def _git(directory: Path, *args: str) -> str | None:
    """Run a git command in a directory.

    Args:
        directory: working directory
        args: the git command and its arguments, e.g., ``"rev-parse", "HEAD"``

    Returns:
        the output, without trailing whitespace; None if git fails or is not available
    """
    return _run(["git", "-C", str(directory), *args])


def _run(command: list[str]) -> str | None:
    """Run a command and return its output.

    Args:
        command: the command and its arguments

    Returns:
        the output, without trailing whitespace; None if the command fails, is not
        available or takes longer than 10 s
    """
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None

    return result.stdout.rstrip()


def _now() -> str:
    """Return the current local time, ISO format, to the second."""
    return datetime.now().astimezone().isoformat(timespec="seconds")
