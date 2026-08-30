#!/usr/bin/env python3
"""Validate demoniC package manifests.

All output is machine-greppable: file:line:col: kind: message
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "demoni.package.v0"
NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
MODULE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
# PACKAGES.md §4: recognized per-project lint dials. The set is closed so a
# misspelled dial fails `make check` instead of silently doing nothing.
LINT_DIALS = {"max_file_lines"}

ERRORS: list[str] = []


def err(file: str, kind: str, msg: str) -> None:
    ERRORS.append(f"{file}:1:1: {kind}: {msg}")


def is_safe_relative(path: str) -> bool:
    p = Path(path)
    return not p.is_absolute() and ".." not in p.parts


def expect_string(rel: str, data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if not isinstance(value, str):
        err(rel, "manifest-field", f"`{field}` must be a string")
        return None
    return value


def check_dmc_path(rel: str, root: Path, label: str, value: str) -> None:
    if not is_safe_relative(value):
        err(rel, "manifest-path", f"`{label}` must be a relative path without `..`")
        return
    if Path(value).suffix != ".dmc":
        err(rel, "manifest-path", f"`{label}` must point at a .dmc file")
        return
    if not (root / value).is_file():
        err(rel, "manifest-path", f"`{label}` path does not exist: {value}")


def validate_manifest(path: Path) -> None:
    rel = str(path.relative_to(ROOT))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        err(rel, "json", f"invalid json: {e.msg}")
        return

    if not isinstance(data, dict):
        err(rel, "manifest", "manifest root must be an object")
        return

    schema = expect_string(rel, data, "schema")
    name = expect_string(rel, data, "name")
    version = expect_string(rel, data, "version")
    root_s = expect_string(rel, data, "root")
    entry = expect_string(rel, data, "entry")
    modules = data.get("modules")

    if schema is not None and schema != SCHEMA:
        err(rel, "manifest-schema", f"`schema` must be {SCHEMA!r}")
    if name is not None and not NAME_RE.match(name):
        err(rel, "manifest-name", "`name` must match [a-z][a-z0-9_-]*")
    if version is not None and not VERSION_RE.match(version):
        err(rel, "manifest-version", "`version` must match MAJOR.MINOR.PATCH")

    root = path.parent
    if root_s is not None:
        if not is_safe_relative(root_s):
            err(rel, "manifest-path", "`root` must be a relative path without `..`")
        else:
            root = path.parent / root_s
            if not root.is_dir():
                err(rel, "manifest-path", f"`root` path does not exist: {root_s}")

    if entry is not None:
        check_dmc_path(rel, root, "entry", entry)

    module_names: set[str] = set()
    if not isinstance(modules, dict) or not modules:
        err(rel, "manifest-field", "`modules` must be a non-empty object")
    else:
        for alias, module_path in modules.items():
            if not isinstance(alias, str) or not MODULE_RE.match(alias):
                err(rel, "manifest-module", f"invalid module alias: {alias!r}")
                continue
            module_names.add(alias)
            if not isinstance(module_path, str):
                err(rel, "manifest-module", f"`modules.{alias}` must be a string path")
                continue
            check_dmc_path(rel, root, f"modules.{alias}", module_path)

    lints = data.get("lints")
    if lints is not None:
        if not isinstance(lints, dict):
            err(rel, "manifest-field", "`lints` must be an object")
        else:
            for dial, value in lints.items():
                if dial not in LINT_DIALS:
                    err(rel, "manifest-lint", f"unknown lint dial: {dial!r}")
                elif dial == "max_file_lines" and (
                    not isinstance(value, int) or isinstance(value, bool) or value < 1
                ):
                    err(rel, "manifest-lint", "`lints.max_file_lines` must be a positive integer")

    exports = data.get("exports", [])
    if not isinstance(exports, list) or not all(isinstance(x, str) for x in exports):
        err(rel, "manifest-field", "`exports` must be a list of strings")
    else:
        for name in exports:
            if name not in module_names:
                err(rel, "manifest-export", f"`exports` names missing module: {name}")


def main() -> int:
    manifests = sorted(ROOT.glob("**/demoni.json"))
    # `publish/` is the GENERATED public tree (tools/build_public_tree.py). Its
    # manifest names `examples/*.dmc` paths that resolve in the published repo,
    # which ships those files; this repo mirrors only the docs into `publish/`,
    # so every path in it reads as missing here. Validating it against the
    # private tree is the wrong question, and it failed `make check` (and so
    # `make ci`) unconditionally. The authored manifest at the repo root is
    # still checked, and it is the one the generated copy is derived from.
    skip = {".git", "publish"}
    manifests = [p for p in manifests if not skip & set(p.parts)]
    if not manifests:
        return 0
    for path in manifests:
        validate_manifest(path)
    for e in ERRORS:
        print(e)
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
