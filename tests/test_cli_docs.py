"""The command-line reference lists every CLI command and alias."""

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

    reference = Path(__file__).parents[1] / "docs" / "source" / "command_line.rst"
    documented = set(re.findall(r"``kipe ([a-z][a-z0-9-]*)", reference.read_text()))
    assert documented == commands
