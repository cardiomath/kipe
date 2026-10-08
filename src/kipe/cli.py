"""Command line interface: ``kipe <subcommand> study.yaml``."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mpi4py import MPI

from pydantic import ValidationError

import kipe
from kipe._formatting import format_table
from kipe.estimation import estimate
from kipe.forward_solver import ForwardSolverError, build_forward_solver
from kipe.options import StudyFileError, StudyOptions, load_study
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

    args = parser.parse_args(argv)
    if args.command == "plot" and args.save and (args.terminal or args.watch):
        plot.error("--save writes a matplotlib figure, not with --terminal or --watch")

    try:
        args.func(args)
    except (StudyFileError, ForwardSolverError, PlotError) as err:  # user errors: no traceback
        print(f"kipe: error: {err}", file=sys.stderr)
        return 1

    return 0


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
    """Estimate the parameters of the study.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = _load_study(args.study)
    log_file = Path(study.output.path) / "estimation" / "kipe.log"
    _setup_logging(args.log_level or study.output.log_level, log_file)
    logging.getLogger("kipe").info("%-11s%s", "study", args.study.resolve())
    estimate(study)
    logging.getLogger("kipe").info("%-11s%s", "log", log_file)


def _plot(args: argparse.Namespace) -> None:
    """Plot the estimation history of one or several studies.

    Args:
        args: parsed command line arguments: the study files and the plot options

    Raises:
        PlotError: if matplotlib or plotext is not installed, or a history cannot be read
    """
    studies = [_load_study(path) for path in args.studies]
    paths = {
        str(path): Path(study.output.path) / "estimation" / "history.csv"
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

    try:
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
    except ImportError as err:
        raise PlotError("kipe plot needs matplotlib: pip install 'kipe[plot]'") from err

    figure = Figure(layout="constrained") if args.save else plt.figure(layout="constrained")
    figure.set_size_inches(4 * len(next(iter(histories.values())).names), 3.5)
    plot_histories(figure, histories, truth)
    if args.save:
        figure.savefig(args.save, dpi=150)
    else:
        plt.show()


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


def _setup_logging(level: str, log_file: Path) -> None:
    """Send kipe's log messages to stderr and, in full detail, to a log file.

    On stderr, MPI rank 0 reports from the given level on; the other ranks only report warnings
    and errors, marked with their rank. At the debug level, each message shows where it was
    logged. Rank 0 also writes all messages, from the debug level on and with their location, to
    ``log_file`` (overwritten).

    Args:
        level: minimum level of the messages on stderr of rank 0, e.g., ``"info"``
        log_file: path of the log file
    """
    rank = MPI.COMM_WORLD.rank
    prefix = "kipe" if rank == 0 else f"kipe[{rank}]"
    location = "%(module)s:%(funcName)s:%(lineno)d  "

    logger = logging.getLogger("kipe")
    for handler in list(logger.handlers):  # main() may run several times in one process
        logger.removeHandler(handler)
        handler.close()

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter(f"{prefix}: {location if level == 'debug' else ''}%(message)s")
    )
    console.setLevel(level.upper() if rank == 0 else logging.WARNING)
    logger.addHandler(console)

    if rank == 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file.setFormatter(
            logging.Formatter(f"%(asctime)s %(levelname)-7s {location}%(message)s")
        )
        logger.addHandler(file)

    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)


if __name__ == "__main__":
    sys.exit(main())
