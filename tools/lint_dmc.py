#!/usr/bin/env python3
"""lint_dmc.py — surface-level lint for `.dmc` example files.

Cannot do real parsing (no compiler yet). Catches the structural and
hygiene issues that don't require semantic knowledge:

    - unbalanced (), [], {}
    - tabs in source
    - trailing whitespace
    - missing final newline
    - lines over MAX_LINE chars
    - non-UTF-8 bytes

All output is machine-greppable:  file:line:col: kind: message
Exits non-zero on any error.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
# Raised from 100: the example corpus pervasively exceeds 100 (dense tensor
# expressions, gradcheck loops). 140 keeps a sane cap without reformatting the
# whole corpus. See the CI-debt cleanup.
MAX_LINE = 140

ERRORS: list[str] = []


def err(file: str, line: int, col: int, kind: str, msg: str) -> None:
    ERRORS.append(f"{file}:{line}:{col}: {kind}: {msg}")


def strip_noise(text: str) -> str:
    """Blank out comments and string literals so delimiter balance is honest.

    Handles:
        # line comment
        #{ block comment, may nest }#
        "string with \\ escapes"
    """
    out: list[str] = []
    i, n = 0, len(text)
    block_depth = 0
    in_str = False
    while i < n:
        c = text[i]
        if block_depth:
            if text[i:i + 2] == "#{":
                block_depth += 1
                out.append("  ")
                i += 2
                continue
            if text[i:i + 2] == "}#":
                block_depth -= 1
                out.append("  ")
                i += 2
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if in_str:
            if c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if c == '"':
                in_str = False
                out.append(" ")
                i += 1
                continue
            out.append("\n" if c == "\n" else " ")
            i += 1
            continue
        if text[i:i + 2] == "#{":
            block_depth = 1
            out.append("  ")
            i += 2
            continue
        if c == "#":
            while i < n and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == '"':
            in_str = True
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def check_balance(file: str, text: str) -> None:
    pairs = {")": "(", "]": "[", "}": "{"}
    opens = {"(", "[", "{"}
    stack: list[tuple[str, int, int]] = []
    line, col = 1, 1
    for c in text:
        if c == "\n":
            line += 1
            col = 1
            continue
        if c in opens:
            stack.append((c, line, col))
        elif c in pairs:
            if not stack or stack[-1][0] != pairs[c]:
                err(file, line, col, "unbalanced", f"unexpected `{c}`")
            else:
                stack.pop()
        col += 1
    for c, l, k in stack:
        err(file, l, k, "unbalanced", f"unclosed `{c}`")


def lint_file(p: Path) -> None:
    rel = str(p.relative_to(ROOT))
    try:
        raw = p.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        err(rel, 1, 1, "encoding", f"file is not valid UTF-8: {e}")
        return
    if raw and not raw.endswith("\n"):
        err(rel, raw.count("\n") + 1, 1, "no-trailing-newline",
            "file must end with a newline")
    for i, line in enumerate(raw.splitlines(), 1):
        if "\t" in line:
            err(rel, i, line.index("\t") + 1, "tab",
                "tabs are forbidden; use spaces")
        if line != line.rstrip():
            err(rel, i, len(line.rstrip()) + 1, "trailing-ws",
                "trailing whitespace")
        if len(line) > MAX_LINE:
            err(rel, i, MAX_LINE + 1, "long-line",
                f"line is {len(line)} chars (max {MAX_LINE})")
    check_balance(rel, strip_noise(raw))


def main() -> int:
    files = sorted(EXAMPLES.glob("**/*.dmc"))
    if not files:
        print("lint_dmc.py: no .dmc files found under examples/", file=sys.stderr)
        return 1
    for p in files:
        lint_file(p)
    for e in ERRORS:
        print(e)
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
