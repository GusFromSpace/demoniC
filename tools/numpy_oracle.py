#!/usr/bin/env python3
"""numpy_oracle.py — external reference oracle for tensor-op correctness (#303).

Our 227 interp-vs-JIT differential assertions verify *internal* consistency
(`interp == jit`). A bug present in *both* backends — a wrong softmax, matmul, or
reduction that both compute identically — passes that silently. This harness adds
an *external* source of truth: it runs demoniC tensor programs through `dmc run`
and compares the results against an independent **NumPy** reimplementation of the
same op (the model tinygrad/TVM use: every op checked against numpy/torch with
explicit rtol/atol).

Mechanics: inputs are generated in NumPy (fixed seed), emitted as demoniC tensor
literals into a per-op source template, run through `dmc run`, and the printed
`Tensor[..] [[...]]` parsed back to an array. The NumPy reference is written from
the op's definition (the spec math) — deliberately *not* from demoniC's Rust — so
a shared interp+JIT bug shows up as a mismatch.

    python3 tools/numpy_oracle.py [--dmc PATH] [--verbose] [--meta-test]

`--meta-test` confirms the harness has teeth: it checks a correct demoniC result
against a deliberately-wrong reference and asserts the mismatch is caught.

All diagnostics are machine-greppable: `oracle: name: kind: message`.
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"

# f32 tolerances. demoniC computes elementwise ops in f32 and accumulates
# matmul/reductions in f64 then rounds — looser than f64 but well within these.
RTOL = 1e-3
ATOL = 1e-4


def dmc_literal(a: np.ndarray) -> str:
    """Render a NumPy array as a demoniC f32 tensor literal."""
    if a.ndim == 0:
        return f"{float(a):.5f}f32"
    if a.ndim == 1:
        return "[" + ", ".join(f"{float(x):.5f}f32" for x in a) + "]"
    return "[" + ", ".join(dmc_literal(row) for row in a) + "]"


def run_dmc(dmc: Path, source: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".dmc", delete=True) as f:
        f.write(source)
        f.flush()
        proc = subprocess.run([str(dmc), "run", f.name],
                              capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"dmc run failed: {(proc.stderr or proc.stdout).strip()[:400]}")
    return proc.stdout


def parse_tensor(out: str) -> np.ndarray:
    """Parse `dmc run` stdout holding one printed tensor/scalar into an array.

    Interp prints `Tensor[<shape>] [[..],[..]]` (possibly across lines) for a
    tensor, or a bare number for a scalar. We strip the `Tensor[..] ` prefix and
    literal_eval the bracketed remainder.
    """
    s = out.strip()
    if s.startswith("Tensor"):
        # drop the `Tensor[<shape>]` prefix; the value starts at the next `[`.
        s = s[s.index("]") + 1:].strip()
    return np.array(ast.literal_eval(s), dtype=np.float64)


# ── reference op implementations (from the spec math, independent of demoniC) ──

def ref_gelu(x):  # tanh approximation, matching docs/STDLIB + activation_f64
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))


def ref_softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def ref_rms_norm(x, g, eps=1e-6):
    r = np.sqrt((x**2).mean(axis=-1, keepdims=True) + eps)
    return x / r * g


def ref_layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


class Case:
    def __init__(self, name, source, expected):
        self.name = name
        self.source = source
        self.expected = np.asarray(expected, dtype=np.float64)


def build_cases(rng: np.random.Generator) -> list[Case]:
    cases: list[Case] = []

    def prog(body_expr: str, *, lets: str = "") -> str:
        return f"fn main() -> nil {{\n{lets}    print({body_expr})\n    nil\n}}\n"

    # matmul (2D and "batched" via 3D not supported in literal easily → 2D + a
    # second matmul shape).
    A = rng.uniform(-2, 2, (3, 4)); B = rng.uniform(-2, 2, (4, 2))
    cases.append(Case("matmul",
        prog("a @ b", lets=f"    let a = {dmc_literal(A)}\n    let b = {dmc_literal(B)}\n"),
        A @ B))

    M = rng.uniform(-2, 2, (2, 5))
    cases.append(Case("transpose",
        prog("m'", lets=f"    let m = {dmc_literal(M)}\n"), M.T))

    # broadcasting elementwise: [B,H] (op) [H]
    X = rng.uniform(-2, 2, (3, 4)); v = rng.uniform(0.5, 2, (4,))
    cases.append(Case("broadcast_add",
        prog("x .+ b", lets=f"    let x = {dmc_literal(X)}\n    let b = {dmc_literal(v)}\n"), X + v))
    cases.append(Case("broadcast_sub",
        prog("x .- b", lets=f"    let x = {dmc_literal(X)}\n    let b = {dmc_literal(v)}\n"), X - v))
    cases.append(Case("broadcast_mul",
        prog("x .* b", lets=f"    let x = {dmc_literal(X)}\n    let b = {dmc_literal(v)}\n"), X * v))
    cases.append(Case("broadcast_div",
        prog("x ./ b", lets=f"    let x = {dmc_literal(X)}\n    let b = {dmc_literal(v)}\n"), X / v))

    # reductions
    R = rng.uniform(-2, 2, (3, 4))
    cases.append(Case("sum", prog("sum(r)", lets=f"    let r = {dmc_literal(R)}\n"), R.sum()))
    cases.append(Case("mean", prog("mean(r)", lets=f"    let r = {dmc_literal(R)}\n"), R.mean()))
    cases.append(Case("variance", prog("variance(r)", lets=f"    let r = {dmc_literal(R)}\n"), R.var()))
    cases.append(Case("sum_along_1",
        prog("sum_along(r, 1)", lets=f"    let r = {dmc_literal(R)}\n"), R.sum(axis=1)))
    cases.append(Case("mean_along_1",
        prog("mean_along(r, 1)", lets=f"    let r = {dmc_literal(R)}\n"), R.mean(axis=1)))

    # argmax/argmin (along last axis)
    cases.append(Case("argmax",
        prog("argmax(r)", lets=f"    let r = {dmc_literal(R)}\n"), R.argmax(axis=-1)))
    cases.append(Case("argmin",
        prog("argmin(r)", lets=f"    let r = {dmc_literal(R)}\n"), R.argmin(axis=-1)))

    # activations (elementwise, f32 tensor)
    act = rng.uniform(-3, 3, (8,))
    cases.append(Case("relu", prog("relu(a)", lets=f"    let a = {dmc_literal(act)}\n"), np.maximum(0, act)))
    cases.append(Case("sigmoid", prog("sigmoid(a)", lets=f"    let a = {dmc_literal(act)}\n"), 1/(1+np.exp(-act))))
    cases.append(Case("tanh", prog("tanh(a)", lets=f"    let a = {dmc_literal(act)}\n"), np.tanh(act)))
    cases.append(Case("gelu", prog("gelu(a)", lets=f"    let a = {dmc_literal(act)}\n"), ref_gelu(act)))
    cases.append(Case("silu", prog("silu(a)", lets=f"    let a = {dmc_literal(act)}\n"), act/(1+np.exp(-act))))

    # fused ML ops
    sm = rng.uniform(-3, 3, (6,))
    cases.append(Case("softmax", prog("softmax(a, -1)", lets=f"    let a = {dmc_literal(sm)}\n"), ref_softmax(sm)))
    rn = rng.uniform(-2, 2, (3, 4)); gn = rng.uniform(0.5, 1.5, (4,))
    cases.append(Case("rms_norm",
        prog("rms_norm(x, g, 0.00001)",
             lets=f"    let x = {dmc_literal(rn)}\n    let g = {dmc_literal(gn)}\n"),
        ref_rms_norm(rn, gn, 1e-5)))
    bs = rng.uniform(-1, 1, (4,))
    cases.append(Case("layer_norm",
        prog("layer_norm(x, g, b, 0.00001)",
             lets=f"    let x = {dmc_literal(rn)}\n    let g = {dmc_literal(gn)}\n    let b = {dmc_literal(bs)}\n"),
        ref_layer_norm(rn, gn, bs, 1e-5)))

    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="numpy reference oracle for tensor ops")
    ap.add_argument("--dmc", type=Path, default=DEFAULT_DMC)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--meta-test", action="store_true",
                    help="assert the harness catches a deliberately-wrong reference")
    args = ap.parse_args()

    if not args.dmc.exists():
        print(f"oracle: error: dmc binary not found at {args.dmc} "
              f"(build with `cargo build --release`)", file=sys.stderr)
        return 2

    rng = np.random.default_rng(0xDEC0DE)
    cases = build_cases(rng)

    failures = []
    for c in cases:
        try:
            got = parse_tensor(run_dmc(args.dmc, c.source))
        except Exception as e:  # noqa: BLE001 — report any harness/run error per-case
            print(f"oracle: {c.name}: error: {e}", file=sys.stderr)
            failures.append(c.name)
            continue
        if got.shape != c.expected.shape:
            print(f"oracle: {c.name}: error: shape {got.shape} != numpy {c.expected.shape}",
                  file=sys.stderr)
            failures.append(c.name)
            continue
        if not np.allclose(got, c.expected, rtol=RTOL, atol=ATOL):
            err = float(np.max(np.abs(got - c.expected)))
            print(f"oracle: {c.name}: error: max|Δ|={err:.2e} exceeds rtol={RTOL} atol={ATOL}",
                  file=sys.stderr)
            failures.append(c.name)
        elif args.verbose:
            print(f"oracle: {c.name}: ok")

    if args.meta_test:
        # Feed a correct demoniC matmul result a deliberately-wrong reference
        # (B @ A-ish via transpose); the comparator must flag it.
        c = next(x for x in cases if x.name == "matmul")
        got = parse_tensor(run_dmc(args.dmc, c.source))
        wrong = c.expected.T  # wrong shape/values
        caught = (got.shape != wrong.shape) or (not np.allclose(got, wrong, rtol=RTOL, atol=ATOL))
        if not caught:
            print("oracle: meta-test: error: harness did NOT catch a wrong reference", file=sys.stderr)
            return 1
        print("oracle: meta-test: ok — wrong reference correctly flagged")

    print(f"oracle: {len(cases) - len(failures)}/{len(cases)} ops match numpy "
          f"(rtol={RTOL}, atol={ATOL})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
