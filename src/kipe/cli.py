"""Command line interface: ``kipe <subcommand> study.yaml``."""

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from mpi4py import MPI

from pydantic import ValidationError

import kipe
from kipe import sensitivity
from kipe._formatting import format_table
from kipe.estimation import estimate
from kipe.forward_solver import ForwardSolverError, build_forward_solver
from kipe.options import (
    StudyFileError,
    StudyOptions,
    estimation_directory,
    load_study,
    study_schema,
)
from kipe.parameters import build_parameterization
from kipe.plot import (
    PlotError,
    plot_histories,
    read_history,
    render_histories,
    watch_histories,
)
from kipe.synthesis import synthesize


def main(argv: Sequence[str] | None = None) -> int:
    """Run the kipe command line interface.

    Args:
        argv: command line arguments without the program name; ``sys.argv[1:]`` if not given

    Returns:
        exit code
    """
    parser = argparse.ArgumentParser(prog="kipe", description=__doc__)
    parser.add_argument("--version", action="version", version=f"kipe {kipe.__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parameters = subparsers.add_parser(
        "list-parameters",
        help="list the estimable parameters of the forward solver and their nominal values",
    )
    list_parameters.add_argument("study", type=Path, help="study file (YAML)")
    list_parameters.set_defaults(func=_list_parameters)

    list_fields = subparsers.add_parser(
        "list-fields",
        help="list the state fields of the forward solver, which measurements can observe",
    )
    list_fields.add_argument("study", type=Path, help="study file (YAML)")
    list_fields.set_defaults(func=_list_fields)

    synthesis = subparsers.add_parser(
        "synthesis",
        aliases=["synthesize"],
        help="generate synthetic measurement data from a forward run",
    )
    synthesis.add_argument("study", type=Path, help="study file (YAML)")
    _add_log_level_argument(synthesis)
    synthesis.set_defaults(func=_synthesis)

    estimation = subparsers.add_parser(
        "estimation",
        aliases=["estimate"],
        help="estimate the parameters from the measurement data",
    )
    estimation.add_argument("study", type=Path, help="study file (YAML)")
    estimation.add_argument(
        "--realization",
        type=int,
        metavar="R",
        help="estimate from the data of noise realization R only (synthesis.realizations); "
        "default: all, one after the other",
    )
    estimation.add_argument(
        "--resume",
        action="store_true",
        help="continue an interrupted estimation after its completed iterations; starts from "
        "the beginning if there are none, does nothing if it is complete",
    )
    _add_log_level_argument(estimation)
    estimation.set_defaults(func=_estimation)

    plot = subparsers.add_parser(
        "plot",
        help="plot the estimation history: the estimates with their 1σ range",
    )
    plot.add_argument(
        "studies",
        type=Path,
        nargs="+",
        metavar="study",
        help="study file(s) (YAML); several are plotted into the same figure",
    )
    plot.add_argument(
        "--truth",
        action="store_true",
        help="draw the nominal parameters of the first study's forward solver as reference",
    )
    plot.add_argument(
        "--realization",
        type=int,
        metavar="R",
        help="plot the estimation of noise realization R, for studies with several",
    )
    plot.add_argument("--save", type=Path, metavar="FILE", help="save the figure, don't show it")
    plot.add_argument(
        "--terminal", action="store_true", help="draw the plot as text in the terminal"
    )
    plot.add_argument(
        "--watch",
        action="store_true",
        help="redraw in the terminal as the estimation runs, until Ctrl+C; implies --terminal",
    )
    plot.add_argument(
        "--interval",
        type=float,
        default=2.0,
        metavar="SECONDS",
        help="time between redraws with --watch (default: %(default)s)",
    )
    plot.set_defaults(func=_plot)

    analysis = subparsers.add_parser(
        "sensitivity",
        help="sensitivity analysis: the Fisher information of the measurements at the nominal "
        "parameters, from two forward runs per parameter",
    )
    analysis.add_argument("study", type=Path, help="study file (YAML)")
    which = analysis.add_mutually_exclusive_group()
    which.add_argument(
        "--run",
        type=int,
        metavar="I",
        help="compute forward run I only, e.g., as a task of a job array; default: all runs, "
        "then their assembly",
    )
    which.add_argument(
        "--assemble", action="store_true", help="only combine the runs, once all are done"
    )
    _add_log_level_argument(analysis)
    analysis.set_defaults(func=_sensitivity)

    schema = subparsers.add_parser(
        "schema",
        help="print the JSON schema of the study file, e.g., for an editor",
    )
    schema.set_defaults(func=_schema)

    args = parser.parse_args(argv)
    if args.command == "plot" and args.save and (args.terminal or args.watch):
        plot.error("--save writes a matplotlib figure, not with --terminal or --watch")

    errors = MPI.COMM_WORLD.Dup()  # for reporting errors, apart from the run's communication
    try:
        args.func(args)
    except (StudyFileError, ForwardSolverError, PlotError) as err:  # user errors: no traceback
        _report_error(err, errors)
        return 1
    finally:
        errors.Free()

    return 0


_ERROR_WAIT = 10.0  # seconds the ranks wait for each other after an error


def _report_error(err: Exception, comm: MPI.Comm, wait: float = _ERROR_WAIT) -> None:
    """Print an error once, or end the run if only some of the ranks have it.

    Usually all ranks raise the same error, since they read and check the same study and data:
    rank 0 prints it. An error of single ranks, e.g., a file system problem on one node, would
    leave the other ranks waiting in their next collective operation until the job's time
    limit: the failing ranks print it, with their rank, and abort the run.

    Args:
        err: the error
        comm: a communicator of all ranks that no other operation uses: the failing ranks meet
            in a barrier on it
        wait: seconds to wait for the other ranks to fail too
    """
    if comm.size == 1:
        print(f"kipe: error: {err}", file=sys.stderr)
        return

    request = comm.Ibarrier()
    deadline = time.monotonic() + wait
    while not request.Test():
        if time.monotonic() > deadline:  # the other ranks go on: an error of this rank only
            print(f"kipe[{comm.rank}]: error: {err}", file=sys.stderr, flush=True)
            comm.Abort(1)
        time.sleep(0.01)

    if comm.rank == 0:
        print(f"kipe: error: {err}", file=sys.stderr)


def _load_study(path: Path) -> StudyOptions:
    """Read and validate a study file, reporting invalid options as a user error.

    Args:
        path: path to the study file

    Returns:
        validated study options

    Raises:
        StudyFileError: if the file cannot be read, or its options are invalid; the message
            lists each invalid option
    """
    try:
        return load_study(path)
    except ValidationError as err:
        lines = [f"invalid study file {path}"]
        for error in err.errors(include_url=False):
            location = ".".join(map(str, error["loc"]))
            lines.append(f"  {location}: {error['msg']}")
        raise StudyFileError("\n".join(lines)) from err


def _list_parameters(args: argparse.Namespace) -> None:
    """Print the solver's estimable parameters and their nominal values.

    If the study file selects parameters, also print their initial estimate, standard
    deviation and the resulting physical 1σ range.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = _load_study(args.study)
    solver = build_forward_solver(study.forward_solver)
    nominal = solver.nominal_parameters()

    selected: dict[str, tuple[str, str, str, str]] = {}

    if study.parameters is not None:
        parameterization = build_parameterization(study.parameters, nominal)
        ranges = parameterization.one_sigma_range()

        for parameter, (lower, upper) in zip(parameterization.parameters, ranges, strict=True):
            prior = study.parameters.select[parameter.name]
            if prior.relative_stddev is not None:
                stddev = f"{prior.relative_stddev:.3g} (relative)"
            else:
                stddev = f"{prior.stddev:g}"

            selected[parameter.name] = (
                parameter.reparameterization,
                f"{parameter.initial:g}",
                stddev,
                f"[{lower:.3g}, {upper:.3g}]",
            )

    rows: list[tuple[str, ...]] = [
        ("parameter", "nominal", "reparameterization", "initial", "stddev", "1σ range")
    ]

    for name, value in nominal.items():
        rows.append((name, f"{value:g}", *selected.get(name, ("", "", "", ""))))

    if not selected:
        rows = [row[:2] for row in rows]

    for line in format_table(rows):
        print(line)


def _list_fields(args: argparse.Namespace) -> None:
    """Print the solver's state fields: their kind, global size and, for a FEniCSx solver, the
    element of their function space.

    The sizes are global, the same on any number of ranks; rank 0 prints.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = _load_study(args.study)
    solver = build_forward_solver(study.forward_solver)
    comm = MPI.COMM_WORLD

    # Only a FEniCSx solver has function spaces: import the backend for it only
    spaces: dict[str, str] = {}
    if callable(getattr(solver, "function_space", None)):
        from kipe.fenicsx.forward_solver import SupportsFunctionSpaces, describe_function_space

        if isinstance(solver, SupportsFunctionSpaces):
            spaces = {
                name: describe_function_space(solver.function_space(name))
                for name, spec in solver.state_spec.items()
                if spec.kind == "distributed"
            }

    rows: list[tuple[str, ...]] = [("field", "kind", "size", "space")]
    for name, spec in solver.state_spec.items():
        if spec.kind == "distributed":
            size = comm.allreduce(spec.local_size)
        else:
            size = spec.local_size
        rows.append((name, spec.kind, str(size), spaces.get(name, "")))

    if not spaces:
        rows = [row[:3] for row in rows]

    if comm.rank == 0:
        for line in format_table(rows, align="<<><"[: len(rows[0])]):
            print(line)


def _synthesis(args: argparse.Namespace) -> None:
    """Generate the measurement data of the study.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = _load_study(args.study)
    log_file = Path(study.output.path) / "synthesis" / "kipe.log"
    _setup_logging(args.log_level or study.output.log_level, log_file)
    logging.getLogger("kipe").info("%-11s%s", "study", args.study.resolve())
    synthesize(study)
    logging.getLogger("kipe").info("%-11s%s", "log", log_file)


def _estimation(args: argparse.Namespace) -> None:
    """Estimate the parameters of the study, from one or each of its noise realizations.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = _load_study(args.study)
    if args.realization is None:
        realizations = range(study.synthesis.realizations)
    else:
        realizations = range(args.realization, args.realization + 1)

    for r in realizations:
        log_file = estimation_directory(study, r) / "kipe.log"
        _setup_logging(args.log_level or study.output.log_level, log_file, append=args.resume)
        logging.getLogger("kipe").info("%-11s%s", "study", args.study.resolve())
        if study.synthesis.realizations > 1:
            logging.getLogger("kipe").info("%-11s%d", "realization", r)
        estimate(study, realization=r, resume=args.resume)
        logging.getLogger("kipe").info("%-11s%s", "log", log_file)


def _sensitivity(args: argparse.Namespace) -> None:
    """Run the sensitivity analysis of the study: its forward runs, and their assembly.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = _load_study(args.study)
    level = args.log_level or study.output.log_level
    if args.assemble:
        runs: Sequence[int] = []
    elif args.run is not None:
        runs = [args.run]
    else:
        runs = range(sensitivity.run_count(study))

    for i in runs:
        log_file = sensitivity.run_directory(study, i) / "kipe.log"
        _setup_logging(level, log_file)
        logging.getLogger("kipe").info("%-11s%s", "study", args.study.resolve())
        sensitivity.run(study, i)
        logging.getLogger("kipe").info("%-11s%s", "log", log_file)

    if args.run is None:
        log_file = Path(study.output.path) / "sensitivity" / "kipe.log"
        _setup_logging(level, log_file)
        logging.getLogger("kipe").info("%-11s%s", "study", args.study.resolve())
        sensitivity.assemble(study)
        logging.getLogger("kipe").info("%-11s%s", "log", log_file)


def _plot(args: argparse.Namespace) -> None:
    """Plot the estimation history of one or several studies.

    Args:
        args: parsed command line arguments: the study files and the plot options

    """
    studies = [_load_study(path) for path in args.studies]
    for path, study in zip(args.studies, studies, strict=True):
        if args.realization is None and study.synthesis.realizations > 1:
            raise StudyFileError(
                f"{path} has {study.synthesis.realizations} noise realizations: choose one "
                "with --realization"
            )
    paths = {
        str(path): estimation_directory(study, args.realization or 0) / "history.csv"
        for path, study in zip(args.studies, studies, strict=True)
    }

    truth = None
    if args.truth:
        truth = build_forward_solver(studies[0].forward_solver).nominal_parameters()

    if args.watch:
        watch_histories(paths, truth, args.interval)
        return

    histories = {label: read_history(path) for label, path in paths.items()}

    if args.terminal:
        print(render_histories(histories, truth, color=sys.stdout.isatty()))
        return

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figure = Figure(layout="constrained") if args.save else plt.figure(layout="constrained")
    figure.set_size_inches(4 * len(next(iter(histories.values())).names), 3.5)
    plot_histories(figure, histories, truth)
    if args.save:
        figure.savefig(args.save, dpi=150)
    else:
        plt.show()


def _schema(args: argparse.Namespace) -> None:
    """Print the JSON schema of the study file.

    Args:
        args: parsed command line arguments, none used
    """
    print(json.dumps(study_schema(), indent=2))


def _add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add the ``--log-level`` option, which overrides ``output.log_level`` of the study file.

    Args:
        parser: parser of a subcommand
    """
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        help="minimum level of the log messages; overrides output.log_level of the study file",
    )


def _setup_logging(level: str, log_file: Path, append: bool = False) -> None:
    """Send kipe's log messages to stderr and, in full detail, to a log file.

    On stderr, MPI rank 0 reports from the given level on; the other ranks only report warnings
    and errors, marked with their rank. At the debug level, each message shows where it was
    logged. Rank 0 also writes all messages, from the debug level on and with their location, to
    ``log_file``: overwritten, or appended to, e.g., when resuming an estimation.

    The messages don't propagate to the root logger: an imported library may have given it a
    handler on stderr (UFL does, with ``logging.debug``), which would show them at all levels.

    Args:
        level: minimum level of the messages on stderr of rank 0, e.g., ``"info"``
        log_file: path of the log file
        append: append to the log file instead of overwriting it
    """
    rank = MPI.COMM_WORLD.rank
    prefix = "kipe" if rank == 0 else f"kipe[{rank}]"
    location = "%(module)s:%(funcName)s:%(lineno)d  "

    logger = logging.getLogger("kipe")
    for handler in list(logger.handlers):  # main() may run several times in one process
        if handler.name in ("kipe.console", "kipe.file"):
            logger.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler()
    console.name = "kipe.console"
    console.setFormatter(
        logging.Formatter(f"{prefix}: {location if level == 'debug' else ''}%(message)s")
    )
    console.setLevel(level.upper() if rank == 0 else logging.WARNING)
    logger.addHandler(console)

    if rank == 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_file, mode="a" if append else "w", encoding="utf-8")
        file.name = "kipe.file"
        file.setFormatter(
            logging.Formatter(f"%(asctime)s %(levelname)-7s {location}%(message)s")
        )
        logger.addHandler(file)

    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    logger.propagate = False


if __name__ == "__main__":
    sys.exit(main())
