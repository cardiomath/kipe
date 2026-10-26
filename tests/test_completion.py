"""``kipe completion``: the shell completion scripts, generated from the parser."""

import shutil
import subprocess

import pytest

from kipe.cli import main


@pytest.mark.parametrize("shell", ["bash", "zsh", "tcsh", "fish", "powershell"])
def test_completion_script(shell, capsys):
    """Each shell's script knows the commands; all but tcsh's also their options."""
    assert main(["completion", shell]) == 0
    script = capsys.readouterr().out
    assert "estimation" in script
    if shell != "tcsh":
        assert "resume" in script


def test_bash_completes_commands_options_and_files(tmp_path, capsys):
    """The bash script completes commands, options, choices and the study file."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")

    assert main(["completion", "bash"]) == 0
    script = tmp_path / "kipe.bash"
    script.write_text(capsys.readouterr().out)
    (tmp_path / "study.yaml").touch()

    # what bash does on Tab after the words: the completion function fills COMPREPLY
    complete = (
        f"source {script}\n"
        "words=($LINE)\n"
        '[[ "$LINE" == *" " ]] && words+=("")\n'
        'COMP_WORDS=("${words[@]}"); COMP_CWORD=$(( ${#words[@]} - 1 ))\n'
        'COMP_LINE="$LINE"; COMP_POINT=${#LINE}\n'
        '_shtab_kipe kipe "${words[-1]}" "${words[-2]}"\n'
        'echo "${COMPREPLY[*]}"\n'
    )

    def completions(line: str) -> list[str]:
        result = subprocess.run(
            [bash, "-c", complete],
            env={"LINE": line, "PATH": ""},
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.split()

    assert completions("kipe est") == ["estimation", "estimate"]
    assert completions("kipe estimation --re") == ["--realization", "--resume"]
    assert completions("kipe estimation st") == ["study.yaml"]
    assert completions("kipe synthesis --log-level ") == ["debug", "info", "warning", "error"]
