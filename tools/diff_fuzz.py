#!/usr/bin/env python3
"""diff_fuzz.py — structure-aware differential fuzzer, interp vs JIT (#302).

Our interp-vs-JIT differential testing is hand-written (diff_backends.py over
whole examples, jit_probes.py over curated edge cases). A manual differential
sweep found ~11 real parity bugs — including a JIT hang on continue-in-for
and a SIGILL on a non-exhaustive match. This automates that sweep.

This is the model Cranelift uses for its own backend (`cranelift-fuzzgen`
generates random CLIF and diffs the interpreter against compiled host code).
cargo-fuzz/libFuzzer needs a nightly toolchain (not available here), so instead
of a `fuzz/` libFuzzer target this is a deterministic, seeded **source**
generator: it builds well-typed scalar demoniC programs (arithmetic, casts,
control flow — the "start small" set from the issue), renders them to source,
runs each through `dmc run` and `dmc jit`, and diffs.

Triage mirrors the manual sweep / jit_probes.py:
    * both backends agree                         → OK
    * both agree it's an error                     → both-fail (fine)
    * run ok, jit emits a clean "not lowered"      → jit-gap (SKIP)
    * run ok, jit ok, values DIFFER                → DIVERGENCE (FAIL)
    * one side crashes/hangs/errs, the other is ok → DIVERGENCE (FAIL)

Generation stays in the *bit-exact* scalar regime (small integers, nonzero
divisors, no INT_MIN/-1, additive loop accumulators) so interp and JIT must
agree exactly (#241) — any divergence is a real, new bug, not a known residual.

Determinism: every program is seeded from --seed + its index, and a failing
program prints its exact seed and source for one-command repro. REGRESSION_SEEDS
pins the minimal historical repros so fixed bugs stay covered.

    python3 tools/diff_fuzz.py [--iters N] [--seed S] [--dmc PATH] [--verbose]
    python3 tools/diff_fuzz.py --repro SEED      # re-emit + run one program
    python3 tools/diff_fuzz.py --meta-test       # prove the differ has teeth

Exit non-zero iff any program DIVERGES.
"""
from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"

# A jit "gap" is a clean error for an unlowered feature (not a miscompile).
# #480: the JIT states whether it refused or failed — match that prefix rather
# than grepping the message for English fragments. See jit_probes.py.
_UNSUPPORTED = "jit unsupported"
_RETURN_RE = re.compile(r"=>\s*(.+?)\s*$", re.MULTILINE)

I64, F32, BOOL = "i64", "f32", "bool"


class Gen:
    """A small type-directed generator of well-typed scalar demoniC programs."""

    def __init__(self, rng: random.Random, floats: bool = False):
        self.rng = rng
        self.floats = floats
        self.vars: list[tuple[str, str]] = []  # (name, type)
        self.n = 0

    def types(self) -> tuple[str, ...]:
        return (I64, F32, BOOL) if self.floats else (I64, BOOL)

    def fresh(self) -> str:
        self.n += 1
        return f"v{self.n}"

    def lit(self, typ: str) -> str:
        if typ == I64:
            return str(self.rng.randint(-50, 50))
        if typ == F32:
            return f"{self.rng.uniform(-10, 10):.3f}f32"
        return self.rng.choice(("true", "false"))

    def var_of(self, typ: str) -> str | None:
        cand = [n for n, t in self.vars if t == typ]
        return self.rng.choice(cand) if cand else None

    def expr(self, typ: str, depth: int) -> str:
        # Leaf: literal or in-scope variable.
        if depth <= 0 or self.rng.random() < 0.3:
            v = self.var_of(typ)
            if v and self.rng.random() < 0.5:
                return v
            return self.lit(typ)
        if typ == I64:
            return self._i64(depth)
        if typ == F32:
            return self._f32(depth)
        return self._bool(depth)

    def _i64(self, depth: int) -> str:
        k = self.rng.random()
        if k < 0.4:
            op = self.rng.choice(("+", "-", "*"))
            return f"({self.expr(I64, depth-1)} {op} {self.expr(I64, depth-1)})"
        if k < 0.55:
            # divide / modulo by a nonzero positive constant (avoids /0 and the
            # INT_MIN/-1 SIGFPE, both tracked separately).
            op = self.rng.choice(("/", "%"))
            return f"({self.expr(I64, depth-1)} {op} {self.rng.randint(1, 9)})"
        if k < 0.7 and self.floats:
            return f"(({self.expr(F32, depth-1)}) as i64)"
        if k < 0.8:
            return f"(if {self.expr(BOOL, depth-1)} {{ 1 }} else {{ 0 }})"
        return f"(if {self.expr(BOOL, depth-1)} {{ {self.expr(I64, depth-1)} }} else {{ {self.expr(I64, depth-1)} }})"

    def _f32(self, depth: int) -> str:
        k = self.rng.random()
        if k < 0.45:
            op = self.rng.choice(("+", "-", "*"))
            return f"({self.expr(F32, depth-1)} {op} {self.expr(F32, depth-1)})"
        if k < 0.6:
            return f"({self.expr(F32, depth-1)} / {self.rng.uniform(1.0, 9.0):.3f}f32)"
        if k < 0.75:
            return f"(({self.expr(I64, depth-1)}) as f32)"
        return f"(if {self.expr(BOOL, depth-1)} {{ {self.expr(F32, depth-1)} }} else {{ {self.expr(F32, depth-1)} }})"

    def _bool(self, depth: int) -> str:
        k = self.rng.random()
        if k < 0.45:
            t = self.rng.choice((I64, F32)) if self.floats else I64
            # f32 uses only ordering comparisons; i64 uses all six.
            ops = ("<", "<=", ">", ">=") if t == F32 else ("<", "<=", ">", ">=", "==", "!=")
            return f"({self.expr(t, depth-1)} {self.rng.choice(ops)} {self.expr(t, depth-1)})"
        if k < 0.7:
            op = self.rng.choice(("&&", "||"))
            return f"({self.expr(BOOL, depth-1)} {op} {self.expr(BOOL, depth-1)})"
        return f"(!{self.expr(BOOL, depth-1)})"

    def program(self) -> str:
        ret = self.rng.choice(self.types())
        lines: list[str] = []
        # A few immutable let-bindings to give expressions some variables.
        for _ in range(self.rng.randint(0, 4)):
            t = self.rng.choice(self.types())
            name = self.fresh()
            lines.append(f"    let {name} = {self.expr(t, 2)}")
            self.vars.append((name, t))
        # Sometimes a bounded for-loop with an additive accumulator (control flow
        # + mutation, the class the continue-in-for hang lived in). i64/f32 only.
        if ret in (I64, F32) and self.rng.random() < 0.5:
            acc = self.fresh()
            init = "0" if ret == I64 else "0.0f32"
            n = self.rng.randint(1, 12)
            body_term = self.expr(ret, 1)
            idx_term = "i" if ret == I64 else "(i as f32)"
            lines.append(f"    let !{acc} = {init}")
            lines.append(f"    for i in 0..{n} {{ {acc} = {acc} + {idx_term} + {body_term} }}")
            self.vars.append((acc, ret))
            tail = acc
        else:
            tail = self.expr(ret, 3)
        body = "\n".join(lines)
        sep = "\n" if body else ""
        return f"fn main() -> {ret} {{\n{body}{sep}    {tail}\n}}\n"


# Minimal historical repros — kept so a regression that reintroduces the bug is
# caught immediately. All are now FIXED (interp == jit); see the cited issues.
REGRESSION_SEEDS = {
    "continue_in_for (#302 sweep — JIT hang)":
        "fn main() -> i64 {\n    let !s = 0\n    for i in 0..10 { if i == 3 { continue } s = s + i }\n    s\n}\n",
    "non_exhaustive_match (#302 sweep — JIT SIGILL)":
        "fn classify(x: i64) -> i64 {\n    match x { 0 => 10, 1 => 20, _ => 99 }\n}\n"
        "fn main() -> i64 { classify(1) + classify(7) }\n",
    "int_div_round_toward_zero (#296/#297 family)":
        "fn main() -> i64 { (0 - 7) / 2 }\n",
}


def run(dmc: Path, mode: str, source: str, timeout: float) -> tuple[str, str]:
    """Return (status, value_or_error). status in {ok, error, timeout}."""
    with tempfile.NamedTemporaryFile("w", suffix=".dmc", delete=True) as f:
        f.write(source)
        f.flush()
        try:
            p = subprocess.run([str(dmc), mode, f.name],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return ("timeout", "")
    out = p.stdout + p.stderr
    if p.returncode != 0:
        return ("error", out.strip()[:200])
    m = list(_RETURN_RE.finditer(p.stdout))
    return ("ok", m[-1].group(1) if m else p.stdout.strip())


def values_agree(a: str, b: str) -> bool:
    """Exact agreement, with NaN treated as agreeing with NaN.

    #473 made this exact. It used to tolerate `|a-b| <= 1e-4 + 1e-3*|b|`, on the
    reasoning that a scalar f32 was computed at different widths on the two
    backends and could only agree to ~7 digits. Scalar f32 is now f32 on both
    sides, and the interpreter's round-through-f32 is bit-exact with the JIT's
    native f32 for `+ - * /` (#241). The old window was three orders of
    magnitude wider than the #473 divergence, so it hid it — the same reason
    `dmc selftest`'s in-process compare was tightened alongside this one.
    """
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except ValueError:
        return False
    return fa != fa and fb != fb  # both NaN


def _values_agree_meta() -> bool:
    """The compare must flag what it is there to flag (see --meta-test)."""
    cases = [("1", "1", True), ("1", "2", False), ("nan", "nan", True),
             ("nan", "1", False), ("1.0", "1.00005", False),
             # a one-ulp f32 gap — the scale #473 lived at.
             ("0.30000001192092896", "0.30000000447034836", False)]
    return all(values_agree(a, b) is exp for a, b, exp in cases)


def classify(dmc: Path, source: str, timeout: float) -> tuple[str, str]:
    """Return (verdict, detail). verdict in {ok, gap, both-fail, DIVERGE}."""
    rs, rv = run(dmc, "run", source, timeout)
    js, jv = run(dmc, "jit", source, timeout)
    if rs == "ok" and js == "ok":
        return ("ok", rv) if values_agree(rv, jv) else ("DIVERGE", f"run={rv!r} jit={jv!r}")
    if rs == "ok" and js == "error" and _UNSUPPORTED in jv:
        return ("gap", jv[:80])
    if rs == "error" and js == "error":
        return ("both-fail", "")
    if rs == "timeout" or js == "timeout":
        return ("DIVERGE", f"timeout: run={rs} jit={js}")
    return ("DIVERGE", f"one-sided: run={rs}:{rv!r} jit={js}:{jv!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="interp-vs-jit differential fuzzer")
    ap.add_argument("--dmc", type=Path, default=DEFAULT_DMC)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--verbose", action="store_true")
    # #473: f32 generation is ON by default. It was opt-in while scalar f32
    # diverged between the backends and could only be compared tolerantly; both
    # of those are fixed, so the default sweep gates it. `--no-floats` restores
    # the integer-only regime.
    ap.add_argument("--floats", action="store_true", default=True,
                    help="generate f32 as well as i64/bool (default: on)")
    ap.add_argument("--no-floats", action="store_false", dest="floats",
                    help="integer/bool-only generation")
    ap.add_argument("--repro", type=int, help="re-emit and run a single seed")
    ap.add_argument("--meta-test", action="store_true")
    args = ap.parse_args()

    if not args.dmc.exists():
        print(f"diff_fuzz: error: dmc not found at {args.dmc} (cargo build --release)",
              file=sys.stderr)
        return 2

    if args.repro is not None:
        src = Gen(random.Random(args.repro), floats=args.floats).program()
        print(src)
        verdict, detail = classify(args.dmc, src, args.timeout)
        print(f"diff_fuzz: seed {args.repro}: {verdict} {detail}")
        return 1 if verdict == "DIVERGE" else 0

    if args.meta_test:
        # Prove the differ has teeth: two trivially different outputs must be
        # flagged as a divergence by values_agree. Runs alongside the fuzzing.
        if not _values_agree_meta():
            print("diff_fuzz: meta-test: error: the value comparison does not "
                  "flag/accept the cases it must (see _values_agree_meta)")
            return 2
        if values_agree("1", "2"):
            print("diff_fuzz: meta-test: error: differ failed to flag 1 vs 2", file=sys.stderr)
            return 1
        print("diff_fuzz: meta-test: ok — divergent values correctly flagged")

    counts = {"ok": 0, "gap": 0, "both-fail": 0, "DIVERGE": 0}
    diverged: list[tuple[str, str, str]] = []

    # Always run the pinned historical repros first.
    for label, src in REGRESSION_SEEDS.items():
        verdict, detail = classify(args.dmc, src, args.timeout)
        if verdict == "DIVERGE":
            diverged.append((f"regression:{label}", src, detail))
        elif args.verbose:
            print(f"diff_fuzz: regression {label}: {verdict} {detail}")

    for i in range(args.iters):
        seed = args.seed * 1_000_003 + i
        src = Gen(random.Random(seed), floats=args.floats).program()
        verdict, detail = classify(args.dmc, src, args.timeout)
        counts[verdict] += 1
        if verdict == "DIVERGE":
            diverged.append((f"seed {seed}", src, detail))
        elif args.verbose:
            print(f"diff_fuzz: seed {seed}: {verdict} {detail}")

    print(f"diff_fuzz: {counts['ok']} ok, {counts['gap']} jit-gap, "
          f"{counts['both-fail']} both-fail, {len(diverged)} divergence(s) "
          f"over {args.iters} iters (seed base {args.seed})")
    for tag, src, detail in diverged:
        print(f"\ndiff_fuzz: DIVERGENCE [{tag}]: {detail}\n--- repro "
              f"(python3 tools/diff_fuzz.py --repro <seed>) ---\n{src}", file=sys.stderr)
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
