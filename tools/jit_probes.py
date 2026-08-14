#!/usr/bin/env python3
"""jit_probes.py — curated run/jit differential edge-case battery.

Complements `diff_backends.py` (which diffs whole example files): this is a hand-
written set of *tiny synthetic probes* hitting the edge cases real examples rarely
exercise — NaN/inf propagation, integer div/mod by zero, saturating casts,
negative and out-of-bounds indexing, degenerate/nonfinite reductions, matmul edge
shapes, bitwise/shift/pow corners. Each probe returns a scalar so the two
backends' outputs compare directly.

Classification per probe:
    * OK            — both backends succeed and agree
    * DIVERGE       — both succeed but DISAGREE  → a miscompile/semantics bug (FAIL)
    * jit-gap       — run ok, jit emits a clean "not lowered" error (informational;
                    allowlist the tracked ones below so only NEW gaps surface)
    * both-fail     — both reject (e.g. a deliberate trap probe); fine

Exits non-zero iff any probe DIVERGES (a silent run/jit mismatch). This battery
found #270 (negative index), #271 (zero-return print), #272 (max/argmax) — keep
it green and add a probe whenever a new edge case is hardened.

    python3 tools/jit_probes.py [--dmc PATH] [--timeout SECS] [--verbose]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"

# A jit "gap" is a clean error for a feature the JIT doesn't lower yet (not a
# silent miscompile). These substrings identify one; tracked gaps are allowlisted.
_GAP_MARKERS = ("not yet", "not support", "use dmc run", "slice 1", "unknown function")
GAP_ALLOWLIST = {
    "pow_int": "#215 — integer `**` not lowered in the JIT",
}

# name -> (source, note on what it probes). Every program returns a scalar.
PROBES = {
    # integer edges
    "int_div_floor_neg":  ("fn main()->i64{ (0-7)/2 }", "truncated division, negative"),
    "int_mod_neg":        ("fn main()->i64{ (0-7)%3 }", "truncated modulo sign follows dividend"),
    "int_div_by_zero":    ("fn main()->i64{ let !z=0  10/z }", "div-by-zero guard → 0, not SIGFPE"),
    "int_mod_by_zero":    ("fn main()->i64{ let !z=0  10%z }", "mod-by-zero guard"),
    "int_shl_63":         ("fn main()->i64{ let !n=63  1<<n }", "max in-range shift"),
    "int_big_mul_wrap":   ("fn main()->i64{ 1000000000 * 1000000000 }", "i64 multiply wrap"),
    "int_add_overflow":   ("fn main()->i64{ let a:i64=9223372036854775807  a+1 }", "#300 add overflow wraps (both backends)"),
    "int_sub_overflow":   ("fn main()->i64{ let a:i64=-9223372036854775807-1  a-1 }", "#300 sub overflow wraps to MAX"),
    # float -> int casts
    "cast_big_float":     ("fn main()->i64{ 1.0e18 as i64 }", "large float cast"),
    "cast_inf":           ("fn main()->i64{ (1.0/0.0) as i64 }", "inf cast saturates"),
    "cast_neg_float":     ("fn main()->i64{ (0.0-3.9) as i64 }", "negative float truncates toward zero"),
    "cast_nan":           ("fn main()->i64{ (0.0/0.0) as i64 }", "NaN cast saturates to 0"),
    "cast_scalar_f32":    ("fn main()->f64{ (0.1 as f32) as f64 }", "#300 scalar as-f32 rounds (both backends)"),
    # map_get on a missing key reads as nil (#300.4)
    "map_miss_is_nil":    ("fn main()->i64{ let !m=map_new()  if map_get(m,\"k\")==nil {1} else {0} }", "#300 map miss == nil (both backends)"),
    "map_hit_value":      ("fn main()->i64{ let !m=map_new()  map_set(m,\"k\",7)  map_get(m,\"k\") }", "#300 map hit returns the value"),
    "map_hit_arith":      ("fn main()->i64{ let !m=map_new()  map_set(m,\"k\",7)  map_get(m,\"k\") + 1 }", "#300 map hit usable in arithmetic"),
    # global max/min NaN propagation (#300)
    "max_nan_propagate":  ("fn main()->f64{ let a=[0.0f32,2.0f32]  let z=[0.0f32,1.0f32]  max(a ./ z) }", "#300 max propagates NaN"),
    "min_nan_propagate":  ("fn main()->f64{ let a=[0.0f32,2.0f32]  let z=[0.0f32,1.0f32]  min(a ./ z) }", "#300 min propagates NaN"),
    # float compares with non-finite
    "float_inf_gt":       ("fn main()->i64{ if (1.0/0.0) > 1.0e300 {1} else {0} }", "inf ordering"),
    "float_nan_eq":       ("fn main()->i64{ let n=0.0/0.0  if n==n {1} else {0} }", "NaN != NaN"),
    "float_nan_lt":       ("fn main()->i64{ let n=0.0/0.0  if n<1.0 {1} else {0} }", "NaN compares false"),
    # zero/false return printing (#271)
    "ret_zero":           ("fn main()->i64{ 0 }", "#271 zero i64 prints"),
    "ret_false":          ("fn main()->bool{ false }", "#271 false bool prints"),
    # indexing (#270 negative; OOB traps both)
    "idx_neg_last":       ("fn main()->i64{ let !t=forge.zeros[f32,[4]] t[3]=9.0 t[-1] as i64 }", "#270 negative index"),
    "idx_neg_first":      ("fn main()->i64{ let !t=forge.zeros[f32,[4]] t[0]=7.0 t[-4] as i64 }", "#270 -dim index"),
    "idx_neg_write":      ("fn main()->i64{ let !t=forge.zeros[f32,[4]] t[-1]=5.0 t[3] as i64 }", "#270 negative write"),
    # reductions (#272) + degenerate/nonfinite
    "sum_size1":          ("fn main()->i64{ let !t=forge.zeros[f32,[1]] t[0]=5.0 sum(t) as i64 }", "size-1 sum"),
    "max_reduction":      ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=2.0 t[1]=9.0 t[2]=4.0 max(t) as i64 }", "#272 max"),
    "min_reduction":      ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=2.0 t[1]=9.0 t[2]=4.0 min(t) as i64 }", "#272 min"),
    "argmax_ties":        ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=2.0 t[1]=2.0 t[2]=1.0 argmax(t,0) }", "#272 argmax first-wins"),
    "argmin":             ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=5.0 t[1]=1.0 t[2]=4.0 argmin(t,0) }", "#272 argmin"),
    "max_with_inf":       ("fn main()->i64{ let !t=forge.zeros[f32,[2]] t[0]=1.0/0.0 t[1]=3.0 max(t) as i64 }", "#272 max with inf"),
    # matmul / broadcast / transpose
    "matmul_1xN_Nx1":     ("fn main()->i64{ let a=forge.ones[f32,[1,4]] let b=forge.ones[f32,[4,1]] (a@b)[0,0] as i64 }", "1xN @ Nx1"),
    "bcast_row":          ("fn main()->i64{ let !a=forge.zeros[f32,[2,3]] let !b=forge.zeros[f32,[3]] b[0]=1.0 sum(a .+ b) as i64 }", "row broadcast"),
    "transpose_sum":      ("fn main()->i64{ let !g=forge.zeros[f32,[2,2]] g[0,1]=5.0 sum(g') as i64 }", "transpose"),
    # control flow / recursion / pow / bitwise
    "recursion_fib":      ("fn fib(n:i64)->i64{ if n<2 {n} else {fib(n-1)+fib(n-2)} } fn main()->i64{ fib(15) }", "recursion"),
    "match_dispatch":     ("fn main()->i64{ let !s=0  for k in 0..4 { s=s+match k {0=>10,1=>20,2=>30,_=>40} }  s }", "match jump table"),
    "bit_ops":            ("fn main()->i64{ (12 & 10) + (12 | 10) + (12 ^ 10) }", "bitwise and/or/xor"),
    "pow_float":          ("fn main()->i64{ (2.0 ** 10.0) as i64 }", "#261 float power"),
    "pow_int":            ("fn main()->i64{ 2 ** 10 }", "integer power (jit gap, #215)"),
    "argmax_then_index":  ("fn main()->i64{\n  let !t=forge.zeros[f32,[3]]\n  t[0]=5.0\n  t[2]=9.0\n  let i = argmax(t, 0)\n  t[i] as i64\n}", "#272 argmax result usable as index"),
    # integer-element tensors (#274) — token ids / index data
    "i64_tensor_rw":      ("fn main()->i64{ let !t=forge.zeros[i64,[3]] t[1]=42 t[1] }", "#274 i64 zeros+write+read"),
    "i32_tensor_rw":      ("fn main()->i64{ let !t=forge.uninit[i32,[2]] t[0]=7 t[0] as i64 }", "#274 i32 uninit+rw"),
    "i64_literal_index":  ("fn main()->i64{ let ids=[[785,6722,315]] ids[0,1] }", "#274 i64 literal index"),
    "i64_param_read":     ("fn first[B,S](x:Tensor[i64,[B,S]])->i64{ x[0,0] } fn main()->i64{ first([[5,6]]) }", "#274 i64 tensor param"),
    "embed_gather":       ("fn main()->i64{ let ids=[10,20,30] let !e=forge.zeros[f32,[40]] e[20]=5.0 let id=ids[1] e[id] as i64 }", "#274 id-driven gather"),
    # cached GQA attention (#277) — k/v from KV caches, runtime history length
    "attn_gqa_cached_kv": ("fn main()->i64{\n"
                            "  let !kc=forge.kv[f32,[1,1,~,2]](capacity=4)\n"
                            "  let !vc=forge.kv[f32,[1,1,~,2]](capacity=4)\n"
                            "  let !k=forge.uninit[f32,[1,1,2,2]]\n"
                            "  let !v=forge.uninit[f32,[1,1,2,2]]\n"
                            "  for s in 0..2 { for d in 0..2 { k[0,0,s,d]=((s*2+d) as f32)*0.1  v[0,0,s,d]=((s*2+d) as f32)*0.2 } }\n"
                            "  kc <- k[..,..,0..2,..]\n"
                            "  vc <- v[..,..,0..2,..]\n"
                            "  let !q=forge.uninit[f32,[1,2,2,2]]\n"
                            "  for h in 0..2 { for s in 0..2 { for d in 0..2 { q[0,h,s,d]=((h*4+s*2+d) as f32)*0.1-0.2 } } }\n"
                            "  let o=attn_gqa(q,kc,vc)\n"
                            "  ((o[0,0,0,0]+o[0,1,1,1]*3.0)*1000.0) as i64 }",
                            "#277 cached GQA attn: KV cache, GQA grouping (H_q=2,H_kv=1), len==S"),
    # elementwise activations (relu/sigmoid/tanh/gelu/silu) — scalar + tensor.
    # Inputs/scales chosen mid-integer to avoid f32/f64 rounding flips. Statements
    # are newline-separated so a literal is never immediately followed by `(` (the
    # `3.0  (expr)` → `3.0(...)` indirect-call parse trap).
    "relu_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=relu(x)*1000.0\n  r as i64 }", "relu scalar -> 3000"),
    "sigmoid_scalar": ("fn main()->i64{\n  let x:f32=3.0\n  let r=sigmoid(x)*1000.0\n  r as i64 }", "sigmoid scalar ~952"),
    "tanh_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=tanh(x)*1000.0\n  r as i64 }", "tanh scalar ~995"),
    "gelu_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=gelu(x)*1000.0\n  r as i64 }", "gelu (tanh approx) scalar ~2996"),
    "silu_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=silu(x)*1000.0\n  r as i64 }", "silu scalar ~2857"),
    "silu_neg_scalar":("fn main()->i64{\n  let x:f32=-2.0\n  let r=silu(x)*1000.0\n  r as i64 }", "silu negative ~-238"),
    "relu_tensor":    ("fn main()->i64{\n  let t=[3.0,1.5,2.0]\n  let r=sum(relu(t))*1000.0\n  r as i64 }", "relu tensor sum -> 6500"),
    "gelu_tensor":    ("fn main()->i64{\n  let t=[3.0,1.5,2.0]\n  let r=sum(gelu(t))*1000.0\n  r as i64 }", "gelu tensor sum ~6350"),
    "silu_tensor":    ("fn main()->i64{\n  let t=[3.0,1.5,2.0]\n  let r=sum(silu(t))*1000.0\n  r as i64 }", "silu tensor sum ~5845"),
}

_VAL = re.compile(r"=>\s*(-?[\d.]+(?:e[-+]?\d+)?|true|false|NaN|inf)")


def run(dmc, mode, src, timeout):
    with tempfile.NamedTemporaryFile("w", suffix=".dmc", delete=False) as f:
        f.write(src); p = f.name
    try:
        r = subprocess.run([dmc, mode, p], capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
        m = _VAL.search(out)
        if m:
            return ("ok", m.group(1))
        if "type error" in out or "refusing to run" in out:
            return ("checkerr", out.splitlines()[0][:90] if out else "")
        if r.returncode != 0:
            return ("err", out.splitlines()[-1][:90] if out else "")
        # Exit 0, no `=> value`: a nil-returning probe (none here) — treat as ok/empty.
        return ("ok", "")
    except subprocess.TimeoutExpired:
        return ("timeout", "")
    finally:
        os.unlink(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmc", default=str(DEFAULT_DMC))
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--verbose", action="store_true", help="print every probe, not just problems")
    args = ap.parse_args()

    if not os.path.exists(args.dmc):
        print(f"jit_probes: error: dmc binary not found at {args.dmc} "
                f"(build it: cd compiler && cargo build --release)", file=sys.stderr)
        return 2

    diverged, new_gaps, ok, gaps = [], [], 0, 0
    for name, (src, _note) in PROBES.items():
        rs, rv = run(args.dmc, "run", src, args.timeout)
        js, jv = run(args.dmc, "jit", src, args.timeout)
        verdict = None
        if rs == "ok" and js == "ok" and rv == jv:
            ok += 1
            verdict = f"OK ({rv})"
        elif rs == "ok" and js == "ok":
            diverged.append(name)
            verdict = f"DIVERGE: run={rv!r} jit={jv!r}"
        elif rs == "ok" and any(k in jv for k in _GAP_MARKERS):
            gaps += 1
            verdict = f"jit-gap: {jv[:60]}"
            if name not in GAP_ALLOWLIST:
                new_gaps.append((name, jv[:80]))
        elif rs != "ok" and js != "ok":
            verdict = f"both-fail (run={rs} jit={js})"
        else:
            # run-only-ok with a non-gap jit error, or jit-only-ok — suspicious.
            diverged.append(name)
            verdict = f"DIVERGE: run={rs}:{rv!r} jit={js}:{jv!r}"
        if args.verbose or "DIVERGE" in verdict or (verdict.startswith("jit-gap") and name not in GAP_ALLOWLIST):
            print(f"  {name:22} {verdict}")

    print(f"\njit_probes: {ok} ok, {gaps} jit-gap, {len(diverged)} divergence(s), "
            f"{len(new_gaps)} untracked gap(s)")
    for name, msg in new_gaps:
        print(f"  {name}: warning: untracked jit-gap — fix or add to GAP_ALLOWLIST: {msg}")
    for name in diverged:
        print(f"  {name}: error: run/jit divergence")
    return 1 if diverged else 0


if __name__ == "__main__":
    sys.exit(main())
