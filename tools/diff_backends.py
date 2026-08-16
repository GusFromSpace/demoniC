#!/usr/bin/env python3
"""diff_backends.py — interpreter vs JIT differential test.

For every example the JIT can actually run, assert that `dmc run` and
`dmc jit` produce the same output. The two backends should be observationally
equivalent on the subset of the language the JIT supports; any divergence is a
bug (a miscompile, a missing guard, or a semantics mismatch).

Two backend-presentation differences are normalized away before comparing —
they are cosmetic, not semantic:
    * the interpreter's trailing `=> <value>` REPL echo of main's return value
    (the JIT does not print it), and
    * the tensor print prefix `Tensor[<shape>] ` (interp) vs bare `[...]` (jit).

Output is machine-greppable (`file: kind: message`); exits non-zero on any
non-allowlisted divergence. Run locally or in CI:

    python3 tools/diff_backends.py [--dmc PATH] [--timeout SECS]

Known divergences are listed in ALLOWLIST with the issue that tracks them, so
the gate stays green while they're open but new divergences fail loudly.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"

# Examples whose run/jit outputs are known to diverge, with the tracking issue.
# Keep this list short and cited — it should shrink as bugs are fixed.
ALLOWLIST = {
    # #241 closed the f32 element-precision gap (stores and dotted ops are now
    # bit-exact across backends). What remains is the documented accumulation
    # residual: the interpreter accumulates matmul/reductions in f64 and rounds
    # once, the JIT accumulates in f32 per step — visible only at f32-ulp scale
    # in iterative/reduction-heavy programs.
    "examples/pagerank.dmc": "#241 residual — power iteration `M @ v`: matmul accumulation width (f64 interp vs f32 JIT)",
    # Both became JIT-eligible only once `to_str` was lowered (#469); the
    # divergence is pre-existing and the same #241 class as pagerank above —
    # `total()` hand-accumulates 360 f32 adds, which the interpreter carries in
    # f64 and the JIT in f32. Agreement holds to ~7 significant digits.
    "examples/sim/lotka_volterra.dmc": "#241 residual — `total()` f32 accumulation loop (f64 interp vs f32 JIT)",
    "examples/sim/lotka_volterra_meso.dmc": "#241 residual — `total()` f32 accumulation loop (f64 interp vs f32 JIT)",
}

_TENSOR_PREFIX = re.compile(r"Tensor\[[^\]]*\]\s*")
_RETURN_ECHO = re.compile(r"\n?=> .*\n?\Z")


def normalize(out: str) -> str:
    """Strip cosmetic backend-presentation differences (see module docstring)."""
    out = _RETURN_ECHO.sub("", out)          # drop the interp's `=> <ret>` echo
    out = _TENSOR_PREFIX.sub("", out)        # `Tensor[..] [1,2]` -> `[1,2]`
    return out.rstrip("\n")


def run(dmc: str, mode: str, path: Path, timeout: int):
    try:
        p = subprocess.run(
            [dmc, mode, str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return None, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmc", default=str(DEFAULT_DMC))
    ap.add_argument("--timeout", type=int, default=25)
    args = ap.parse_args()

    if not os.path.exists(args.dmc):
        print(f"diff_backends: error: dmc binary not found at {args.dmc} "
                f"(build it: cd compiler && cargo build --release)", file=sys.stderr)
        return 2

    files = sorted(EXAMPLES.rglob("*.dmc"))
    matched = jit_unsupported = run_skipped = 0
    divergences = []
    stale_allow = set(ALLOWLIST)

    for f in files:
        rel = str(f.relative_to(REPO))
        rc_run, out_run = run(args.dmc, "run", f, args.timeout)
        if rc_run != 0:                       # needs args / too slow / errors under run
            run_skipped += 1
            continue
        rc_jit, out_jit = run(args.dmc, "jit", f, args.timeout)
        if rc_jit != 0:                       # JIT doesn't support this program
            jit_unsupported += 1
            continue
        if normalize(out_run) == normalize(out_jit):
            matched += 1
        elif rel in ALLOWLIST:
            stale_allow.discard(rel)
            print(f"{rel}: known: divergence allowlisted ({ALLOWLIST[rel]})")
        else:
            divergences.append(rel)
            print(f"{rel}: error: run/jit output diverges")

    print(f"\ndiff_backends: {matched} matched, {jit_unsupported} jit-unsupported, "
            f"{run_skipped} run-skipped, {len(divergences)} unexpected divergence(s)")

    # An allowlisted entry that no longer diverges (or no longer runs) is stale —
    # flag it so the list doesn't rot, but don't fail the build on it.
    for rel in sorted(stale_allow):
        print(f"{rel}: warning: allowlisted but did not diverge — remove from ALLOWLIST")

    return 1 if divergences else 0


if __name__ == "__main__":
    sys.exit(main())
