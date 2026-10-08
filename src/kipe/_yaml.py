"""Writing YAML files: the study file and the provenance of a run."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a mapping as YAML: block style, one line per value, ``null`` for None.

    Args:
        path: where to write
        data: the mapping
    """
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096  # one line per value, no wrapping
    yaml.representer.add_representer(  # write "null", not an empty value
        type(None), lambda r, _: r.represent_scalar("tag:yaml.org,2002:null", "null")
    )
    with open(path, "w") as f:
        yaml.dump(dict(data), f)
