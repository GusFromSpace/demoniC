#!/usr/bin/env python3
"""rss_gate.py — pins the peak resident set size (RSS) of a fixed, tiny set of
`dmc` runs so a regression like #822's first version cannot pass every gate
again.

#822 slice 3c's page-aligned `alloc_zeroed` default chunk touched all 512 MiB
of it eagerly (calloc's lazy zero pages only apply at align <= 16), so ANY
unflagged `dmc jit` run peaked at ~547 MB RSS instead of ~10 MB for a 64-byte
allocation; `examples/attn_grad_jit.dmc` went 525 MB vs 13 MB. Every existing
gate passed: budget minimums pin bytes CHARGED, not bytes TOUCHED, and
`make golden`/`parity`/`conformance` compare OUTPUTS, never memory. This is
the first gate that looks at RSS at all.

Portability: `/usr/bin/time -l` (macOS) and `/usr/bin/time -v` (Linux) report
peak RSS in incompatible formats, so this uses `resource.getrusage` instead.
A single long-lived process's `RUSAGE_CHILDREN.ru_maxrss` is a RUNNING MAXIMUM
across every child that has ever terminated under it, not a per-child value —
measuring three programs in one process would let an early large child mask a
later smaller one. So each measurement below spawns a dedicated one-shot
wrapper subprocess whose only child is the `dmc` run being measured; that
wrapper's own `RUSAGE_CHILDREN` starts at zero and holds exactly that one
child's peak. `ru_maxrss` is bytes on macOS/BSD and KiB on Linux — `sys.platform`
picks which.

Probes (kept deliberately few and cheap — this runs in CI):
    tiny64_jit    tools/rss_probes/tiny64.dmc, `dmc jit`  — one 64-byte,
                  no-flags Forge allocation; almost pure fixed overhead, so the
                  most sensitive of the three to a chunk that stops being lazy.
    tiny64_run    the same program, `dmc run` (interpreter) — a baseline for
                  what the JIT overhead itself is, and its own regression gate.
    attn_grad_jit examples/attn_grad_jit.dmc, `dmc jit` — a real fused-attention
                  + autodiff workload, not a synthetic minimum.

Ceilings (`_CEILINGS` below) are 4x the measured value on the capture host,
rounded up to a round MiB figure — generous headroom for normal variance
across machines and Cranelift/rustc versions, nowhere near enough to hide a
regression on the scale #822's was (52x, 547 MB vs ~10 MB).

Measured (this harness, macOS/arm64 (Apple Silicon), 2026-09-10, against the
main-line binary carrying the JIT budget table, `dmc` built `cargo build
--release`, three runs each, values stable within ~1%):
    tiny64_jit     ~10.6-10.7 MB  (10.1-10.2 MiB)  ceiling 48 MiB
    tiny64_run     ~8.6 MB        (8.2 MiB)         ceiling 40 MiB
    attn_grad_jit  ~13.6-13.7 MB  (13.0-13.1 MiB)   ceiling 64 MiB

    python3 tools/rss_gate.py [--dmc PATH] [--timeout SECS] [--ceiling-scale F]

`--ceiling-scale` multiplies every ceiling (default 1.0) — it exists so the
gate's ability to fail can be demonstrated without editing the file (a scaled-
down ceiling is exactly what a real regression looks like to this gate).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"
TINY64 = REPO / "tools" / "rss_probes" / "tiny64.dmc"
ATTN_GRAD_JIT = REPO / "examples" / "attn_grad_jit.dmc"

MIB = 1024 * 1024

# name -> (mode, program path, ceiling in bytes). Order is report order.
_CEILINGS = {
    "tiny64_jit": ("jit", TINY64, 48 * MIB),
    "tiny64_run": ("run", TINY64, 40 * MIB),
    "attn_grad_jit": ("jit", ATTN_GRAD_JIT, 64 * MIB),
}

# The wrapper subprocess: runs the measured command as its only child, then
# reports that child's exit code and this wrapper's own RUSAGE_CHILDREN peak
# (which is exactly the measured child's peak, since it is the wrapper's
# first and only child). Kept to stdlib only, matching the rest of tools/.
_WRAPPER = (
    "import json, resource, subprocess, sys\n"
    "r = subprocess.run(sys.argv[1:], capture_output=True)\n"
    "ru = resource.getrusage(resource.RUSAGE_CHILDREN)\n"
    "print(json.dumps({'returncode': r.returncode, 'maxrss': ru.ru_maxrss}))\n"
)


def measure_peak_rss(cmd: list[str], timeout: int) -> tuple[int, int]:
    """Run `cmd`, return (returncode, peak RSS in bytes) of that one process."""
    result = subprocess.run([sys.executable, "-c", _WRAPPER, *cmd],
                             capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"measurement wrapper failed for {cmd!r}: "
                            f"rc={result.returncode} stderr={result.stderr[:200]!r}")
    data = json.loads(result.stdout)
    maxrss = data["maxrss"]
    # macOS/BSD report ru_maxrss in bytes; Linux reports KiB.
    if sys.platform != "darwin" and not sys.platform.startswith(("freebsd", "openbsd")):
        maxrss *= 1024
    return data["returncode"], maxrss


def fmt_mib(n: int) -> str:
    return f"{n / MIB:.2f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmc", default=str(DEFAULT_DMC))
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--ceiling-scale", type=float, default=1.0,
                     help="multiply every ceiling by this factor "
                          "(for demonstrating the gate can fail; default 1.0)")
    args = ap.parse_args()

    dmc = Path(args.dmc)
    if not dmc.exists():
        print(f"rss_gate: error: dmc binary not found at {dmc} "
              f"(build it: cd compiler && cargo build --release)", file=sys.stderr)
        return 2
    if not TINY64.exists():
        print(f"rss_gate: error: fixture not found at {TINY64}", file=sys.stderr)
        return 2
    if not ATTN_GRAD_JIT.exists():
        print(f"rss_gate: error: fixture not found at {ATTN_GRAD_JIT}", file=sys.stderr)
        return 2

    failures = []
    for name, (mode, program, ceiling) in _CEILINGS.items():
        scaled_ceiling = int(ceiling * args.ceiling_scale)
        try:
            rc, rss = measure_peak_rss([str(dmc), mode, str(program)], args.timeout)
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as exc:
            failures.append(name)
            print(f"  {name:16} ERROR measuring peak RSS: {exc}")
            continue
        if rc != 0:
            failures.append(name)
            print(f"  {name:16} ERROR: `dmc {mode}` exited {rc}")
            continue
        verdict = "FAIL" if rss > scaled_ceiling else "ok"
        if verdict == "FAIL":
            failures.append(name)
        print(f"  {name:16} {verdict:4} peak {fmt_mib(rss):>10}  "
              f"ceiling {fmt_mib(scaled_ceiling):>10}  (dmc {mode})")

    print(f"\nrss_gate: {len(_CEILINGS) - len(failures)}/{len(_CEILINGS)} within ceiling"
          + (f", {len(failures)} over: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
