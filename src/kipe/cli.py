"""Command line interface: ``kipe <subcommand> study.yaml``."""

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from kipe.forward_solver import ForwardSolverError, build_forward_solver
from kipe.options import StudyFileError, load_study
from kipe.parameters import build_parameterization
from kipe.synthesis import synthesize


def main(argv: Sequence[str] | None = None) -> int:
    """Run the kipe command line interface.

    Args:
        argv: command line arguments without the program name; ``sys.argv[1:]`` if not given

    Returns:
        exit code
    """
    parser = argparse.ArgumentParser(prog="kipe", description=__doc__)
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
    synthesis.set_defaults(func=_synthesis)

    args = parser.parse_args(argv)

    try:
        args.func(args)
    except ValidationError as err:
        print(f"kipe: error: invalid study file {args.study}", file=sys.stderr)
        for error in err.errors(include_url=False):
            location = ".".join(map(str, error["loc"]))
            print(f"  {location}: {error['msg']}", file=sys.stderr)
        return 1
    except (StudyFileError, ForwardSolverError) as err:  # user errors: no traceback
        print(f"kipe: error: {err}", file=sys.stderr)
        return 1

    return 0


def _list_parameters(args: argparse.Namespace) -> None:
    """Print the solver's estimable parameters and their nominal values.

    If the study file selects parameters, also print their initial estimate, standard
    deviation and the resulting physical 1σ range.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = load_study(args.study)
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

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths, strict=True)]
        print("  ".join(cells).rstrip())


def _synthesis(args: argparse.Namespace) -> None:
    """Generate the measurement data of the study.

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = load_study(args.study)
    _setup_logging(study.output.log_level)
    synthesize(study)


def _setup_logging(level: str) -> None:
    """Send kipe's log messages to stderr.

    Args:
        level: minimum level of the messages, e.g., ``"info"``
    """
    logger = logging.getLogger("kipe")
    if not logger.handlers:  # main() may run several times in one process, e.g., in tests
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("kipe: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level.upper())


if __name__ == "__main__":
    sys.exit(main())
