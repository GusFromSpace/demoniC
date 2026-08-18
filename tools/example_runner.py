#!/usr/bin/env python3
"""example_runner.py — per-example backend-conformance gate.

Supersedes an earlier two-integer ratchet. Instead of one aggregate
`interp_passed` / `jit_ran` count — which is blind to *which* example moved and
trips on every legitimate addition — this gates each example file individually
and can record genuinely-new, genuinely-correct examples by itself.

WHAT "CORRECT" MEANS HERE
-------------------------
`dmc --check` is type-only: it passes type-clean-but-numerically-wrong code. So
"the example compiles" is NOT evidence it is correct. The only correctness signal
this gate trusts is executable ground truth:

    * the example carries `fn test_*() -> bool` assertions, and
    * they PASS under the interpreter (`dmc test` — the numeric gate), and
    * interp and JIT AGREE wherever the example is JIT-runnable (divergence = wrong).

An example with no passing `test_*` is *unverified* — it is reported, never
counted as conformant. No tool can fully decide arbitrary-program correctness;
the above are strong necessary conditions, and `--accept-new` additionally
refuses to silently absorb a vacuous test (one whose body is a bare `true` / has
no call or comparison).

BASELINE  (tools/example_baseline.json)
---------------------------------------
A per-file manifest, repo-relative path → recorded counts:

    { "examples/xor_net.dmc": { "pass": 1, "jit_ran": 0 }, ... }

Only files with >=1 passing test appear. Gate semantics:

    * any interp failure                      -> hard fail
    * any interp/JIT divergence (parity fail) -> hard fail
    * baselined file: pass or jit_ran DROPPED -> hard fail (regression)
    * baselined file: counts INCREASED        -> hard fail, "re-run with --accept-new"
    * NEW correct file not in baseline        -> hard fail in gate mode;
                                                 auto-recorded under --accept-new

MODES
    python3 tools/example_runner.py                 # CI gate (read-only)
    python3 tools/example_runner.py --accept-new    # add new correct examples, keep regressions fatal
    python3 tools/example_runner.py --update        # full rewrite to measured (manual override)

All diagnostics are machine-greppable: `example-runner: kind: message`.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"
DEFAULT_EXAMPLES = REPO / "examples"
BASELINE = Path(__file__).resolve().parent / "example_baseline.json"

RESULT_RE = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed")
JIT_RE = re.compile(r"jit parity: (\d+) ran, (\d+) skipped")
TEST_FN_RE = re.compile(r"\bfn\s+(test_\w+)\s*\(")


def strip_noncode(src: str) -> str:
    """Drop `#` line comments and string contents so brace/identifier scans
    don't trip over text. Heuristic, but adequate for the vacuity check."""
    out = []
    for line in src.splitlines():
        line = re.sub(r"#.*$", "", line)
        line = re.sub(r'"(?:\\.|[^"\\])*"', '""', line)
        out.append(line)
    return "\n".join(out)


def test_bodies(src: str) -> dict[str, str]:
    """Map each `test_*` fn name to its brace-matched body text."""
    code = strip_noncode(src)
    bodies: dict[str, str] = {}
    for m in TEST_FN_RE.finditer(code):
        name = m.group(1)
        brace = code.find("{", m.end())
        if brace < 0:
            continue
        depth, i = 0, brace
        while i < len(code):
            if code[i] == "{":
                depth += 1
            elif code[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        bodies[name] = code[brace + 1 : i]
    return bodies


def is_vacuous(body: str) -> bool:
    """A test body is vacuous if it asserts nothing computational: no function or
    builtin call AND no comparison operator. `-> bool { true }` or a literal-only
    body fails this; anything that calls the example's logic or a builtin
    (`mlp[..](..)`, `isclose(..)`, `abs(..) < tol`, `sum(..)`) passes. Semantic
    tautologies that still call functions are out of scope here — that is the
    adversarial verifier's job, not a regex's."""
    has_call = bool(re.search(r"[A-Za-z_]\w*\s*\(", body))
    has_compare = bool(re.search(r"==|!=|<=|>=|<|>|&&|\|\||\.>|\.<", body))
    return not (has_call or has_compare)


def run_file(dmc: Path, path: Path, jit: bool) -> tuple[int, int, int]:
    """Return (passed, failed, jit_ran) for one file. jit_ran is 0 in interp mode."""
    cmd = [str(dmc), "test"] + (["--jit"] if jit else []) + [str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    if "0 tests" in out and not RESULT_RE.search(out):
        return 0, 0, 0
    m = RESULT_RE.search(out)
    if not m:
        print(f"example-runner: error: could not parse test summary for {path}",
              file=sys.stderr)
        print(out[-1500:], file=sys.stderr)
        sys.exit(2)
    passed, failed = int(m.group(1)), int(m.group(2))
    jit_ran = 0
    if jit:
        j = JIT_RE.search(out)
        jit_ran = int(j.group(1)) if j else 0
    return passed, failed, jit_ran


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def measure(dmc: Path, examples: Path):
    """Walk the corpus once. Returns (recorded, unverified, failures, vacuous)."""
    recorded: dict[str, dict[str, int]] = {}
    unverified: list[str] = []
    failures: list[str] = []
    vacuous: list[str] = []
    for path in sorted(examples.rglob("*.dmc")):
        src = path.read_text(errors="replace")
        bodies = test_bodies(src)
        if not bodies:
            unverified.append(rel(path))
            continue
        passed, failed, _ = run_file(dmc, path, jit=False)
        _, jit_failed, jit_ran = run_file(dmc, path, jit=True)
        if failed or jit_failed:
            failures.append(f"{rel(path)} (interp_failed={failed}, jit_failed={jit_failed})")
            continue
        if passed == 0:
            unverified.append(rel(path))
            continue
        if all(is_vacuous(b) for b in bodies.values()):
            vacuous.append(rel(path))
        recorded[rel(path)] = {"pass": passed, "jit_ran": jit_ran}
    return recorded, unverified, failures, vacuous


def main() -> int:
    ap = argparse.ArgumentParser(description="per-example backend-conformance gate")
    ap.add_argument("--dmc", type=Path, default=DEFAULT_DMC)
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--accept-new", action="store_true",
                    help="record genuinely-new correct examples; regressions stay fatal")
    ap.add_argument("--update", action="store_true",
                    help="rewrite the whole baseline to measured (manual override)")
    args = ap.parse_args()

    if not args.dmc.exists():
        print(f"example-runner: error: dmc binary not found at {args.dmc} "
              f"(build with `cargo build --release`)", file=sys.stderr)
        return 2

    recorded, unverified, failures, vacuous = measure(args.dmc, args.examples)
    print(f"example-runner: measured {len(recorded)} verified example(s), "
          f"{len(unverified)} unverified (no passing test_*)")
    if unverified:
        print(f"example-runner: note: {len(unverified)} unverified example(s) "
              f"(compile-only, not correctness-gated)")

    # Real failures are fatal in every mode — never record a failing corpus.
    if failures:
        for f in failures:
            print(f"example-runner: error: example FAILED — {f}", file=sys.stderr)
        return 1

    if args.update:
        BASELINE.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
        print(f"example-runner: baseline rewritten → {BASELINE.name} "
              f"({len(recorded)} examples)")
        return 0

    base: dict[str, dict[str, int]] = {}
    if BASELINE.exists():
        base = json.loads(BASELINE.read_text())
    elif not args.accept_new:
        print(f"example-runner: error: no baseline at {BASELINE.name}; "
              f"create it with `--update`", file=sys.stderr)
        return 2

    failed = False

    # 1. Regressions: a baselined file that lost tests or dropped a count.
    for path, want in sorted(base.items()):
        got = recorded.get(path)
        if got is None:
            print(f"example-runner: error: example DROPPED OUT — {path} was "
                  f"baselined ({want}) but now has no passing test", file=sys.stderr)
            failed = True
            continue
        for key in ("pass", "jit_ran"):
            if got[key] < want.get(key, 0):
                print(f"example-runner: error: {path} {key} REGRESSED "
                      f"{want.get(key, 0)} → {got[key]}", file=sys.stderr)
                failed = True
            elif got[key] > want.get(key, 0) and not args.accept_new:
                print(f"example-runner: error: {path} {key} improved "
                      f"{want.get(key, 0)} → {got[key]} — run `example_runner.py --accept-new`",
                      file=sys.stderr)
                failed = True

    # 2. New examples not yet in the baseline.
    new = {p: c for p, c in recorded.items() if p not in base}
    if new and args.accept_new:
        blocked = [p for p in new if p in vacuous]
        for p in blocked:
            print(f"example-runner: error: refusing to auto-accept {p}: its "
                  f"test_* are vacuous (no comparison or local call) — add a real "
                  f"assertion", file=sys.stderr)
            failed = True
        for p in sorted(p for p in new if p not in vacuous):
            print(f"example-runner: accept: recording new example {p} = {new[p]}")
    elif new:
        for p in sorted(new):
            tag = " [VACUOUS]" if p in vacuous else ""
            print(f"example-runner: error: NEW correct example {p} = {recorded[p]}"
                  f"{tag} — run `example_runner.py --accept-new` to record it", file=sys.stderr)
        failed = True

    if args.accept_new and not failed:
        merged = dict(base)
        merged.update(recorded)  # widen + bump verified entries; regressions already caught
        BASELINE.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(f"example-runner: baseline updated → {BASELINE.name} "
              f"({len(merged)} examples)")
        return 0

    if failed:
        return 1
    print(f"example-runner: ok — {len(recorded)} verified examples at baseline, "
          f"0 failures, 0 divergences")
    return 0


if __name__ == "__main__":
    sys.exit(main())
