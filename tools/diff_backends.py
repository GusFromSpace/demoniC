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
non-allowlisted divergence AND on any timeout. Run locally or in CI:

    python3 tools/diff_backends.py [--dmc PATH] [--timeout SECS]

Known divergences are listed in ALLOWLIST with the issue that tracks them, so
the gate stays green while they're open but new divergences fail loudly.

Every file lands in exactly one bucket of the summary line:
    matched          both backends ran and agree (the verdict this gate exists for)
    jit-unsupported  `dmc run` succeeded, `dmc jit` refused or failed
    run-skipped      `dmc run` exited non-zero (needs a data runtime, args, ...)
    excluded         named in EXCLUDED with an issue, never started (exceeds any
                     budget on every machine, so a timeout could not classify it)
    timed-out        exceeded --timeout under either backend — a FAILURE, never a
                     skip: it means machine load or a hang, not a parity verdict.
The first four are properties of the corpus and the binary and reproduce on
every machine; only the fifth depends on the machine, which is why it is kept
apart and fails the run. tools/golden pins the summary line byte-for-byte.
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

# Examples this gate does not run at all, with the reason AND the issue that
# tracks it — the same discipline ALLOWLIST is held to. An entry earns its place
# only by exceeding the budget on EVERY machine on EVERY run, so a timeout is
# never the thing that classifies it; it would burn the budget and tell us
# nothing. Naming them here is what keeps a timeout on anything ELSE what it
# should be: a loud failure (see `--timeout`).
#
# This list is a claim about measured wall-clock, so it is stated as one. Both
# entries were re-measured 2026-09-05 on the macOS capture platform against a
# 320s cap, and both exceeded it. Keep it that way: an entry added because a
# file "looks like a game loop" is a guess, and the last one was wrong —
# games/connect_four.dmc was excluded as "game loop, never terminates" and is
# in fact depth-4 minimax SELF-play with a `move_count >= 42` draw exit. It
# terminates in ~49s under `dmc run` and ~0.4s under `dmc jit`, byte-identical,
# and it is back in the gate as a matched file.
EXCLUDED: dict[str, str] = {
    "examples/autograd.dmc":
        "exceeds --timeout on every machine: 1000 training steps of 768x3072 "
        "matmuls under the interpreter (>320s measured) (#653)",
    "examples/console/snake.dmc":
        "exceeds --timeout on every machine: console VM loop driven by an "
        "input opcode, no exit with stdin at EOF (>320s measured) (#653)",
}

_TENSOR_PREFIX = re.compile(r"Tensor\[[^\]]*\]\s*")
_RETURN_ECHO = re.compile(r"\n?=> .*\n?\Z")


def normalize(out: str) -> str:
    """Strip cosmetic backend-presentation differences (see module docstring)."""
    out = _RETURN_ECHO.sub("", out)          # drop the interp's `=> <ret>` echo
    out = _TENSOR_PREFIX.sub("", out)        # `Tensor[..] [1,2]` -> `[1,2]`
    return out.rstrip("\n")


def run(dmc: str, mode: str, path: Path, timeout: int):
    """Run one backend over one file. Returns (returncode, stdout); returncode is
    None if the child exceeded `timeout` — callers MUST treat that as a distinct
    outcome, never as "exited non-zero". stdin is /dev/null so an example that
    reads input gets EOF at once instead of inheriting (and blocking on) the
    terminal of whoever ran the gate — the verdict must not depend on that."""
    try:
        p = subprocess.run(
            [dmc, mode, str(path)],
            capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return None, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmc", default=str(DEFAULT_DMC))
    # Per-backend, per-file wall-clock budget. Sized so that no GATED example
    # comes near it on a loaded machine: the slowest are console/dv2.dmc
    # (~86s), games/connect_four.dmc (~49s), reel_to_reel.dmc (~34s) and
    # games/tictactoe.dmc (~23s) under `dmc run` on an M-series box. The old
    # default of 25s sat right on top of that last one, so `matched` flickered
    # 52/51/50 with machine load. A budget this size is paid only by a genuine
    # hang — the two examples that exceed it unconditionally are EXCLUDED and
    # never start. On a slower box (a 2-core hosted runner) pass a larger one;
    # the public workflow does exactly that rather than relying on this default.
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    if not os.path.exists(args.dmc):
        print(f"diff_backends: error: dmc binary not found at {args.dmc} "
                f"(build it: cd compiler && cargo build --release)", file=sys.stderr)
        return 2

    files = sorted(EXAMPLES.rglob("*.dmc"))
    matched = jit_unsupported = run_skipped = 0
    divergences = []
    timed_out = []
    stale_allow = set(ALLOWLIST)
    stale_excluded = set(EXCLUDED)

    for f in files:
        rel = str(f.relative_to(REPO))
        if rel in EXCLUDED:
            stale_excluded.discard(rel)
            print(f"{rel}: excluded: {EXCLUDED[rel]}")
            continue
        rc_run, out_run = run(args.dmc, "run", f, args.timeout)
        if rc_run is None:
            timed_out.append(rel)
            print(f"{rel}: error: dmc run exceeded --timeout {args.timeout}s "
                  f"(machine load or a hang; not a parity verdict)")
            continue
        if rc_run != 0:                       # needs args / errors under run
            run_skipped += 1
            continue
        rc_jit, out_jit = run(args.dmc, "jit", f, args.timeout)
        if rc_jit is None:
            timed_out.append(rel)
            print(f"{rel}: error: dmc jit exceeded --timeout {args.timeout}s "
                  f"(machine load or a hang; not a parity verdict)")
            continue
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

    excluded = len(EXCLUDED) - len(stale_excluded)
    print(f"\ndiff_backends: {matched} matched, {jit_unsupported} jit-unsupported, "
            f"{run_skipped} run-skipped, {excluded} excluded, "
            f"{len(timed_out)} timed-out, "
            f"{len(divergences)} unexpected divergence(s)")

    # An allowlisted entry that no longer diverges (or no longer runs) is stale —
    # flag it so the list doesn't rot, but don't fail the build on it. Same for
    # an EXCLUDED entry that names a file the corpus no longer has.
    for rel in sorted(stale_allow):
        print(f"{rel}: warning: allowlisted but did not diverge — remove from ALLOWLIST")
    for rel in sorted(stale_excluded):
        print(f"{rel}: warning: excluded but not in the corpus — remove from EXCLUDED")

    # A timeout is a failure, not a skip. Folding it into `run-skipped` made a
    # busy machine and a broken example indistinguishable: an example that
    # matches on a quiet box silently left `matched` under load, and the golden
    # master (tools/golden) reported DRIFT for a corpus that had not changed.
    return 1 if divergences or timed_out else 0


if __name__ == "__main__":
    sys.exit(main())
