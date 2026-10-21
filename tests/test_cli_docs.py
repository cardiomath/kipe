"""The command line reference lists every command of the CLI, aliases included, and no other.

If this fails after adding a command, add a row ``kipe <command> study.yaml`` with its purpose
to the table in docs/source/reference/command_line.rst; after removing or renaming one, remove
or rename its row.
"""

import re
from pathlib import Path

import pytest

from kipe.cli import main


def test_command_line_reference_matches_cli(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0

    help_text = capsys.readouterr().out
    match = re.search(r"\{([^}]+)\}", help_text)
    assert match is not None
    commands = set(match.group(1).split(","))

    reference = Path(__file__).parents[1] / "docs" / "source" / "reference" / "command_line.rst"
    documented = set(re.findall(r"``kipe ([a-z][a-z0-9-]*)", reference.read_text()))
    missing, unknown = sorted(commands - documented), sorted(documented - commands)
    assert not missing, f"{reference}: add a row to the table for: {', '.join(missing)}"
    assert not unknown, f"{reference}: no such command, remove its row: {', '.join(unknown)}"
