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
#
# IT IS EMPTY, and that is the point: as of #481 every example the JIT can run
# agrees with the interpreter byte for byte. The last two entries went away in
# this order, and the order is the interesting part:
#
#   * both lotka_volterra files (#473) — filed as a "#241 residual" but actually
#     SCALAR f32 accumulation, fixed by giving the interpreter a real f32 scalar;
#   * pagerank (#481) — filed as matmul accumulation WIDTH, but the real cause
#     was CONTRACTION: the JIT's three matmul kernels disagreed with each other
#     (`n % 4` decided whether you got an FMA kernel), and the interpreter used
#     ndarray's f64 dot. f32 matmul now contracts with FMA everywhere (SPEC §7.5).
#
# Both were mis-attributed to #241 for months. If something lands here again,
# re-derive the cause from a measurement before citing an issue for it.
ALLOWLIST: dict[str, str] = {}

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
