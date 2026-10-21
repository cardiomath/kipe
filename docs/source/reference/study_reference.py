"""Generate the study file reference from the options' JSON schema.

pydantic derives the schema from the option dataclasses in :mod:`kipe.options`: types and
choices from their annotations, defaults from their defaults, and descriptions from their
attribute docstrings (``use_attribute_docstrings``). This module turns it into a commented
YAML overview of a study file, and one table of keys per section. The options stay the only
source: the reference follows every change of them.
"""

import json
import re
import textwrap
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from kipe.options import StudyOptions

type Node = dict[str, Any]

_PLURALS = {"number": "numbers", "integer": "integers", "string": "strings"}
_COMMENT_COLUMN = 40  # of the comments in the YAML overview


def write_reference(path: Path) -> None:
    """Write the study file reference, as reStructuredText.

    Args:
        path: the file to write
    """
    schema = TypeAdapter(StudyOptions).json_schema()
    reference = _Reference(schema["$defs"])

    lines = [".. code-block:: yaml", ""]
    lines += [f"   {line}" if line else "" for line in reference.skeleton(schema)]
    lines.append("")
    lines += reference.tables(schema)
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


class _Reference:
    """The parts of the study file reference, from the schema's definitions.

    Args:
        definitions: the schema's ``$defs``
    """

    def __init__(self, definitions: dict[str, Node]) -> None:
        self._definitions = definitions

    # --- the YAML overview ---------------------------------------------------------------------

    def skeleton(self, section: Node, depth: int = 0, commented: bool = False) -> list[str]:
        """Return a section as commented YAML: its keys with a placeholder or default.

        Args:
            section: the section's schema, resolved
            depth: the section's nesting depth
            commented: whether the keys are commented out, as those of an alternative kind

        Returns:
            the lines of YAML
        """
        lines = []
        pad = "  " * depth + ("# " if commented else "")
        required = section.get("required", [])
        for key, node in section["properties"].items():
            if depth == 0 and lines:  # a blank line between the sections of the study file
                lines.append("")
            target = self._without_null(node)
            presence = "required" if key in required else ""
            if "oneOf" in target:  # a choice by type: the default kind, the others commented out
                lines.append(_line(f"{pad}{key}:", "one of, by type"))
                lines += self._kinds_skeleton(target, depth + 1, commented)
            elif "properties" in target:  # a section
                lines.append(_line(f"{pad}{key}:", presence))
                lines += self.skeleton(target, depth + 1)
            elif isinstance(target.get("additionalProperties"), dict):  # name -> section
                lines.append(_line(f"{pad}{key}:", presence))
                lines.append(_line(f"{pad}  <name>:", _keys(target)))
                lines += self.skeleton(self._resolve(target["additionalProperties"]), depth + 2)
            else:  # a value: its default, its fixed value, or its type
                default = node.get("default", self._resolve(node).get("const"))
                if default is not None:
                    value = _yaml(default)
                    comment = _plain(self._type(node)) if self._choices(node) else ""
                else:
                    value = f"<{_plain(self._type(node))}>"
                    comment = presence or "optional"
                lines.append(_line(f"{pad}{key}: {value}", comment))
        return lines

    def _kinds_skeleton(self, union: Node, depth: int, commented: bool) -> list[str]:
        """Return the kinds of a choice by ``type`` as YAML: the default, then the others
        commented out.

        Args:
            union: the choice's schema, with ``oneOf``
            depth: the nesting depth of the kinds' keys
            commented: whether the whole choice is commented out

        Returns:
            the lines of YAML
        """
        default = next((kind for kind, is_default in self._kinds(union) if is_default), None)
        variants = sorted(union["oneOf"], key=lambda v: self._kind(v) != default)
        lines = []
        for i, variant in enumerate(variants):
            section = self._resolve(variant)
            keys = {key: node for key, node in section["properties"].items() if key != "type"}
            alternative = commented or i > 0
            pad = "  " * depth + ("# " if alternative else "")
            if i > 0:
                lines.append(f"{pad}or:")
            lines.append(
                _line(f"{pad}type: {self._kind(variant)}", "the default" if i == 0 else "")
            )
            if keys:
                lines += self.skeleton({**section, "properties": keys}, depth, alternative)
        return lines

    # --- the tables ----------------------------------------------------------------------------

    def tables(self, schema: Node) -> list[str]:
        """Return one heading and table per section of the study file.

        Args:
            schema: the schema of the study file

        Returns:
            the lines of reStructuredText
        """
        lines = []
        for key, node in schema["properties"].items():
            lines += _heading(key, f"``{key}``", "-")
            lines += self._content(key, node)
        return lines

    def _content(self, path: str, node: Node) -> list[str]:
        """Return what is under a key's heading: its description, table and subsections.

        Args:
            path: the key's path in the study file
            node: the key's schema

        Returns:
            the lines of reStructuredText
        """
        target = self._without_null(node)
        lines = []
        description = node.get("description") or target.get("description")
        if description:
            lines += [description, ""]

        if "oneOf" in target:  # a choice by type: one subsection per kind
            for variant in target["oneOf"]:
                lines += self._variant(path, variant)
        elif "properties" in target:  # a section
            lines += self._section(path, target, self._class(node))
        elif isinstance(target.get("additionalProperties"), dict):  # name -> section
            value = target["additionalProperties"]
            lines += [f"Each ``{path}.<name>``, where ``<name>`` is {_keys(target)}:", ""]
            lines += self._section(f"{path}.<name>", self._resolve(value), self._class(value))
        else:  # alternatives: the keys of those that are sections
            for alternative in self._sections_among(target):
                section = self._resolve(alternative)
                lines += self._section(path, section, self._class(alternative))
        return lines

    def _section(self, path: str, section: Node, owner: str | None) -> list[str]:
        """Return the table of a section's keys, then the subsections of those that have keys.

        Args:
            path: the section's path
            section: the section's schema, resolved
            owner: the section's options class

        Returns:
            the lines of reStructuredText
        """
        assert owner is not None, f"{path}: a section without its options class"
        lines = self._table(path, section, owner)

        for key, node in section["properties"].items():
            if self._has_keys(node):
                lines += _heading(f"{path}.{key}", f"``{path}.{key}``", "^")
                lines += self._content(f"{path}.{key}", node)
        return lines

    def _variant(self, path: str, variant: Node) -> list[str]:
        """Return the subsection of one kind of a choice by ``type``.

        Args:
            path: the path of the key with the choice
            variant: the kind's schema, a reference to its options class

        Returns:
            the lines of reStructuredText
        """
        section = self._resolve(variant)
        kind = section["properties"]["type"]
        default = " (the default)" if "default" in kind else ""
        title = f"``type: {kind['const']}``{default}"
        lines = _heading(f"{path}.type.{kind['const']}", title, '"')
        if section.get("description"):
            lines += [section["description"], ""]

        others = {key: node for key, node in section["properties"].items() if key != "type"}
        if others:
            lines += self._section(path, {**section, "properties": others}, self._class(variant))
        return lines

    def _table(self, path: str, section: Node, owner: str) -> list[str]:
        """Return the table of a section's keys: key, type, default, description.

        Args:
            path: the section's path
            section: the section's schema, resolved
            owner: the section's options class, for the links to the API

        Returns:
            the lines of reStructuredText
        """
        lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 16 22 12 50",
            "",
            "   * - key",
            "     - type",
            "     - default",
            "     - description",
        ]
        required = section.get("required", [])
        for key, node in section["properties"].items():
            kind = self._type(node)
            if self._has_keys(node):
                kind += f", see :ref:`below <{_label(f'{path}.{key}')}>`"
            description = node.get("description") or self._resolve(node).get("description", "")
            lines += [
                f"   * - :attr:`{key} <kipe.options.{owner}.{key}>`",
                f"     - {kind}",
                f"     - {self._default(node, key in required)}",
                "     - " + textwrap.indent(description, " " * 7).lstrip(),
            ]
        return [*lines, ""]

    # --- types, defaults, classes --------------------------------------------------------------

    def _type(self, node: Node) -> str:
        """Return the type of a key, in words.

        Args:
            node: the key's schema

        Returns:
            e.g., ``number ≥ 0``, ``list of strings``, ``P1 | DG0``
        """
        node = self._resolve(node)
        if "anyOf" in node:
            alternatives = [self._resolve(n) for n in node["anyOf"] if n.get("type") != "null"]
            return " or ".join(
                "{" + ", ".join(n["properties"]) + "}" if "properties" in n else self._type(n)
                for n in alternatives
            )
        if "oneOf" in node:
            kinds = [kind for kind, _ in self._kinds(node)]
            return "one of, by ``type``: " + ", ".join(f"``{kind}``" for kind in kinds)
        if "const" in node:
            return f"``{node['const']}``"
        if "enum" in node:
            return " | ".join(f"``{value}``" for value in node["enum"])

        kind = node.get("type")
        if kind == "array":
            items = self._type(node["items"])
            items = _PLURALS.get(items, items.replace("list of", "lists of", 1))
            return f"list of {items}" + self._bounds(node)
        if kind == "object":
            if "properties" in node:
                return "section"
            if isinstance(node.get("additionalProperties"), dict):
                return "mapping: name → section"
            return "mapping"
        return str(kind) + self._bounds(node)

    def _choices(self, node: Node) -> bool:
        """Return whether a value is one of fixed choices.

        Args:
            node: the key's schema

        Returns:
            True for an enum, e.g., ``P1 | DG0``
        """
        return "enum" in self._resolve(node)

    def _kinds(self, union: Node) -> list[tuple[str, bool]]:
        """Return the kinds of a choice by ``type``, and which is the default.

        Args:
            union: the choice's schema, with ``oneOf``

        Returns:
            (kind, whether it is the default), in the order of the schema
        """
        kinds = []
        for variant in union["oneOf"]:
            kind = self._resolve(variant)["properties"]["type"]
            kinds.append((kind["const"], "default" in kind))
        return kinds

    def _kind(self, variant: Node) -> str:
        """Return the ``type`` of one kind of a choice.

        Args:
            variant: the kind's schema

        Returns:
            e.g., ``fenicsx.mesh``
        """
        return str(self._resolve(variant)["properties"]["type"]["const"])

    def _has_keys(self, node: Node) -> bool:
        """Return whether a key's value has keys of its own, so it gets a subsection.

        Args:
            node: the key's schema

        Returns:
            True for a section, a mapping of sections, or a choice by ``type``
        """
        target = self._without_null(node)
        return (
            "oneOf" in target
            or "properties" in target
            or isinstance(target.get("additionalProperties"), dict)
            or bool(self._sections_among(target))
        )

    def _sections_among(self, node: Node) -> list[Node]:
        """Return the alternatives of a value that are sections, e.g., the range of ``times``.

        Args:
            node: the value's schema, resolved

        Returns:
            the alternatives, as references to their options classes
        """
        return [n for n in node.get("anyOf", []) if "properties" in self._resolve(n)]

    @staticmethod
    def _bounds(node: Node) -> str:
        """Return the bounds of a number or a list, in words.

        Args:
            node: the schema of the number or list

        Returns:
            e.g., `` > 0``, `` ≥ 1``, `` (at least 1)``, or nothing
        """
        if "exclusiveMinimum" in node:
            return f" > {node['exclusiveMinimum']}"
        if "minimum" in node:
            return f" ≥ {node['minimum']}"
        if "minItems" in node:
            return f" (at least {node['minItems']})"
        return ""

    @staticmethod
    def _default(node: Node, required: bool) -> str:
        """Return the default of a key, or that it is required.

        Args:
            node: the key's schema
            required: whether the key is required

        Returns:
            e.g., ``**required**``, ````0.0````, ``optional`` (not set by default)
        """
        if required:
            return "**required**"
        default = node.get("default")
        return f"``{_yaml(default)}``" if default is not None else "optional"

    def _class(self, node: Node) -> str | None:
        """Return the options class a key's value is, or holds: the name of its definition.

        Args:
            node: the key's schema

        Returns:
            the class name, for a section, an optional section, or a mapping of sections;
            None for other values
        """
        if "$ref" in node:
            return str(node["$ref"].rsplit("/", 1)[-1])
        alternatives = [n for n in node.get("anyOf", []) if n.get("type") != "null"]
        if len(alternatives) == 1:
            return self._class(alternatives[0])
        if isinstance(node.get("additionalProperties"), dict):
            return self._class(node["additionalProperties"])
        return None

    def _resolve(self, node: Node) -> Node:
        """Return the schema a reference points to, or the schema itself.

        Args:
            node: a schema, possibly ``{"$ref": "#/$defs/<name>"}``

        Returns:
            the schema
        """
        if "$ref" in node:
            return self._definitions[node["$ref"].rsplit("/", 1)[-1]]
        return node

    def _without_null(self, node: Node) -> Node:
        """Return the schema of an optional key without its ``null`` alternative.

        Args:
            node: the key's schema

        Returns:
            the schema of the value, resolved
        """
        node = self._resolve(node)
        alternatives = [n for n in node.get("anyOf", []) if n.get("type") != "null"]
        if len(alternatives) == 1:
            return self._resolve(alternatives[0])
        return node


def _heading(path: str, title: str, underline: str) -> list[str]:
    """Return a heading with a label, for links to it.

    Args:
        path: the path the heading is about, for the label
        title: the heading's text
        underline: the underline character, for the heading's level

    Returns:
        the lines of reStructuredText
    """
    return [f".. _{_label(path)}:", "", title, underline * len(title), ""]


def _label(path: str) -> str:
    """Return the label of a heading.

    Args:
        path: e.g., ``measurements.<name>.noise``

    Returns:
        e.g., ``study-measurements-name-noise``
    """
    return "study-" + re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


def _keys(mapping: Node) -> str:
    """Return what the keys of a mapping are, as the options say (``json_schema_extra``).

    Args:
        mapping: the mapping's schema, resolved

    Returns:
        e.g., ``a parameter of the solver``; ``a name`` if the options do not say
    """
    return str(mapping.get("keys", "a name"))


def _line(text: str, comment: str) -> str:
    """Return a line of the YAML overview, with its comment aligned.

    Args:
        text: the YAML
        comment: the comment, if any

    Returns:
        the line
    """
    if not comment:
        return text
    return f"{text.ljust(_COMMENT_COLUMN - 1)} # {comment}"


def _plain(text: str) -> str:
    """Return reStructuredText without its literal markup, for the YAML overview.

    Args:
        text: e.g., ````P1`` | ``DG0````

    Returns:
        e.g., ``P1 | DG0``
    """
    return text.replace("``", "")


def _yaml(value: Any) -> str:
    """Return a default value as it is written in YAML.

    Args:
        value: the value

    Returns:
        e.g., ``info``, ``0.0``, ``[]``
    """
    return value if isinstance(value, str) else json.dumps(value)
