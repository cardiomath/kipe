"""Command line interface: ``kipe <subcommand> study.yaml``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from kipe.forward_solver import ForwardSolverError, build_forward_solver
from kipe.options import StudyFileError, load_study


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

    Args:
        args: parsed command line arguments, with the path to the study file
    """
    study = load_study(args.study)
    solver = build_forward_solver(study.forward_solver)
    nominal = solver.nominal_parameters()
    width = max(map(len, nominal), default=0)
    for name, value in nominal.items():
        print(f"{name:<{width}}  {value:g}")


if __name__ == "__main__":
    sys.exit(main())
