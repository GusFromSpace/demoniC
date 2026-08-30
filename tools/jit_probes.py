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

# A jit "gap" is a clean REFUSAL to lower a feature (not a silent miscompile).
# #480: the JIT states which it is — `jit unsupported at L:C:` vs `jit error at
# L:C:` — so this matches one structural prefix instead of grepping the message
# for five English fragments. The old guess was wrong in both directions: a
# clean refusal worded differently ("`sum` is f32-only") was scored as a
# DIVERGENCE and could not be allowlisted, and a genuine miscompile whose text
# happened to contain "slice 1" would have been downgraded to a gap.
_UNSUPPORTED = "jit unsupported"
GAP_ALLOWLIST = {
    "pow_int": "#215 — integer `**` not lowered in the JIT",
    "f64r_sum_keeps_width": "#481 follow-up — the JIT's `sum` is f32-only",
    "copy_i64_mut_arg": "#552 — `!` params lower only inside a `@grad fn`",
    "copy_f64_mut_arg": "#552 — `!` params lower only inside a `@grad fn`",
    "copy_bool_mut_arg": "#552 — `!` params lower only inside a `@grad fn`",
}

# name -> (source, note on what it probes). Every program returns a scalar.
PROBES = {
    # integer edges
    "int_div_floor_neg":  ("fn main()->i64{ (0-7)/2 }", "truncated division, negative"),
    "int_mod_neg":        ("fn main()->i64{ (0-7)%3 }", "truncated modulo sign follows dividend"),
    "int_div_by_zero":    ("fn main()->i64{ let !z=0  10/z }", "div-by-zero guard → 0, not SIGFPE"),
    "int_mod_by_zero":    ("fn main()->i64{ let !z=0  10%z }", "mod-by-zero guard"),
    "int_shl_63":         ("fn main()->i64{ let !n=63  1<<n }", "max in-range shift"),
    # i32 operands: the count is checked at the operand's width, not always 64
    # (`OPERATORS.md §8a`). Newlines are load-bearing — `= 2  (a<<n)` parses as
    # a call on the literal. Only counts whose result fits in i32 are probed:
    # a 32-bit shift that loses bits has no shared answer with the interpreter,
    # which carries every integer as i64.
    "int_shl_i32":        ("fn main()->i64{\n let a:i32=256\n let n:i32=2\n (a<<n) as i64\n}", "i32 `<<`, count checked at 32 bits"),
    "int_shr_i32":        ("fn main()->i64{\n let a:i32=-8\n let n:i32=1\n (a>>n) as i64\n}", "i32 `>>`, arithmetic and 32-bit"),
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
    # ── scalar f32 (#473) ────────────────────────────────────────────────────
    # The corpus reaches no multi-op f32 scalar chain, so nothing here was
    # gated before: `diff_backends.py` was green while `dmc run` and `dmc jit`
    # disagreed on `0.1f32 + 0.2f32`. Each probe returns the f64 widening of an
    # f32 result, so the comparison is on all 17 digits — a one-ulp f32
    # disagreement (~1e-8 relative) shows, which is exactly the scale at which
    # "accumulate in f64, round once" parts company with true f32.
    # A SINGLE `+ - * /` is NOT enough: #241's double-rounding property makes
    # f64-then-round bit-exact with native f32 for one op. Chains are the test.
    "f32_add_two":        ("fn main()->f64{ (0.1f32 + 0.2f32) as f64 }", "#473 the issue's repro: one add"),
    "f32_chain_three":    ("fn main()->f64{ let a=0.1f32\n let b=0.2f32\n let c=0.3f32\n (a+b+c) as f64 }", "#473 three-term chain — diverges under f64 accumulation"),
    "f32_chain_mixed_ops":("fn main()->f64{ let a=0.1f32\n let b=0.3f32\n let c=0.7f32\n ((a+b)*c-a/b) as f64 }", "#473 + - * / in one f32 expression"),
    "f32_horner":         ("fn main()->f64{ let a=1.1f32\n let b=1.3f32\n (((a*b+a)*b+a)*b+a) as f64 }", "#473 nested multiply-add, 6 ops"),
    "f32_accum_loop":     ("fn main()->f64{ let !s=0.0f32\n for i in 0..100 { s = s + 0.01f32 }\n s as f64 }", "#473 100-step f32 accumulation (the `total()` shape)"),
    "f32_accum_mul":      ("fn main()->f64{ let !s=1.0f32\n for i in 0..20 { s = s * 1.1f32 }\n s as f64 }", "#473 multiplicative accumulation"),
    "f32_accum_elems":    ("fn main()->f64{ let !t=forge.zeros[f32,[8]]\n for i in 0..8 { t[i]=0.1f32 }\n let !s=0.0f32\n for i in 0..8 { s = s + t[i] }\n s as f64 }", "#473 summing f32 tensor elements in a scalar loop"),
    "f32_mixed_f64":      ("fn main()->f64{ let a=0.1f32\n let b=0.2\n a + b }", "#473 f32 meets f64 -> promotes, never narrows"),
    "f32_mixed_rev":      ("fn main()->f64{ let a=0.2\n let b=0.1f32\n a + b }", "#473 f64 meets f32, other order"),
    "f32_mixed_tail":     ("fn main()->f64{ let a=0.1f32\n let b=0.2f32\n let c=0.3\n a + b + c }", "#473 f32 chain then an f64 tail"),
    "f32_call_ret":       ("fn g(x:f32)->f32{ x*x+x }\nfn main()->f64{ g(0.1f32) as f64 }", "#473 f32 through a parameter and a return"),
    "f32_call_nested":    ("fn g(x:f32)->f32{ x+0.1f32 }\nfn main()->f64{ g(g(g(0.1f32))) as f64 }", "#473 f32 across three nested calls"),
    "f32_ret_unsuffixed": ("fn g()->f32{ 0.1 }\nfn main()->f64{ g() as f64 }", "#473 a declared f32 RETURN binds an unsuffixed literal's width"),
    "f32_tensor_read":    ("fn main()->f64{ let !t=forge.zeros[f32,[3]]\n t[0]=0.1f32\n t[1]=0.2f32\n t[2]=0.3f32\n (t[0]+t[1]+t[2]) as f64 }", "#473 reads out of an f32 tensor stay f32"),
    "f32_tensor_mixed":   ("fn main()->f64{ let !a=forge.zeros[f32,[1]]\n let !b=forge.uninit[f64,[1]]\n a[0]=0.1f32\n b[0]=0.2\n a[0]+b[0] }", "#473 f32 element + f64 element promotes"),
    "f32_cast_chain":     ("fn main()->f64{ let a=0.1 as f32\n let b=0.2 as f32\n (a+b) as f64 }", "#473 `as f32` produces a true f32"),
    "f32_let_annotated":  ("fn main()->f64{ let a: f32 = 0.1\n let b: f32 = 0.2\n let c: f32 = 0.3\n (a+b+c) as f64 }", "#473 a typed `let` binds width (#209: the literal itself stays f64)"),
    "f32_model_field":    ("model M { x: f32 }\nfn main()->f64{ let m = M { x: 0.1 }\n (m.x + 0.2f32) as f64 }", "#473 a declared f32 FIELD binds width"),
    "f32_unary_neg":      ("fn main()->f64{ let a=0.1f32\n let b=0.0f32-a\n (b+a+a) as f64 }", "#473 negation keeps f32"),
    # ── f32 tensor reductions (#481) ────────────────────────────────────────
    # `sum(t)` used to disagree with the loop it is documented to mean, WITHIN
    # `dmc run`. Each returns the f64 widening so all 17 digits are compared.
    "f32r_sum":           ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n sum(t) as f64 }", "#481 sum accumulates at f32 width"),
    "f32r_sum_vs_loop":   ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n let !s=0.0f32\n for i in 0..12 { s=s+t[i] }\n (sum(t)-s) as f64 }", "#481 sum(t) - hand loop == 0 on both backends"),
    "f32r_mean":          ("fn main()->f64{ let !t=forge.zeros[f32,[100]]\n for i in 0..100 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n mean(t) as f64 }", "#481 mean divides by an f32 count in f32"),
    "f32r_variance":      ("fn main()->f64{ let !t=forge.zeros[f32,[100]]\n for i in 0..100 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n variance(t) as f64 }", "#481 variance is two passes at f32 width"),
    "f32r_sum_then_add":  ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n (sum(t)+0.1f32) as f64 }", "#481 the reduction RESULT is f32-wide"),
    "f32r_max_then_add":  ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n (max(t)+0.1f32) as f64 }", "#481 max selects an f32 — `max(t)` alone agreed even before"),
    "f32r_sum_along":     ("fn main()->f64{ let !m=forge.zeros[f32,[3,4]]\n for i in 0..3 { for j in 0..4 { m[i,j]=((i*4+j) as f32)*0.3f32+0.1f32 } }\n sum(sum_along(m,1)) as f64 }", "#481 per-axis lanes reduce at f32 width"),
    "f32r_mean_along":    ("fn main()->f64{ let !m=forge.zeros[f32,[3,4]]\n for i in 0..3 { for j in 0..4 { m[i,j]=((i*4+j) as f32)*0.3f32+0.1f32 } }\n sum(mean_along(m,0)) as f64 }", "#481 mean_along keeps the source dtype"),
    "f32r_variance_along":("fn main()->f64{ let !m=forge.zeros[f32,[3,4]]\n for i in 0..3 { for j in 0..4 { m[i,j]=((i*4+j) as f32)*0.3f32+0.1f32 } }\n sum(variance_along(m,1)) as f64 }", "#481 variance_along is two-pass, matching the JIT's formula"),
    "f32r_softmax_sum":   ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n sum(softmax(t)) as f64 }", "#481 reducing a derived f32 tensor"),
    # Restored by #480: the JIT's `sum` is f32-only, and that refusal is now a
    # classified `jit unsupported` rather than a message this battery had to
    # score as a DIVERGENCE. Before #480 this probe could not live here at all.
    "f64r_sum_keeps_width":("fn main()->f64{ let !t=forge.uninit[f64,[3]]\n t[0]=0.1\n t[1]=0.2\n t[2]=0.3\n sum(t) }", "#481 must NOT narrow an f64 tensor's reduction (jit gap: sum is f32-only)"),
    # ── f32 matmul contraction (#481) ───────────────────────────────────────
    # C[0,0] depends only on A's row 0 and B's column 0, which are identical in
    # both probes below. Before #481 the JIT answered differently depending on
    # B's WIDTH, because `n % 4` picked the kernel and the kernel picked the
    # rounding. These two must now agree with each other AND across backends.
    "f32mm_width3":       ("fn main()->f64{ let !a=forge.zeros[f32,[8,16]]\n let !b=forge.zeros[f32,[16,3]]\n for i in 0..8 { for j in 0..16 { a[i,j]=(((i*16+j) as f32)*0.7f32+0.1f32)/3.0f32 } }\n for i in 0..16 { for j in 0..3 { b[i,j]=(((i*97+j) as f32)*0.3f32+0.2f32)/7.0f32 } }\n (a@b)[0,0] as f64 }", "#481 matmul, B width 3 (scalar kernel)"),
    "f32mm_width4":       ("fn main()->f64{ let !a=forge.zeros[f32,[8,16]]\n let !b=forge.zeros[f32,[16,4]]\n for i in 0..8 { for j in 0..16 { a[i,j]=(((i*16+j) as f32)*0.7f32+0.1f32)/3.0f32 } }\n for i in 0..16 { for j in 0..4 { b[i,j]=(((i*97+j) as f32)*0.3f32+0.2f32)/7.0f32 } }\n (a@b)[0,0] as f64 }", "#481 matmul, B width 4 (vector kernel) — same C[0,0] as width 3"),
    "f32mm_batched":      ("fn main()->f64{ let !a=forge.zeros[f32,[2,4,8]]\n let !b=forge.zeros[f32,[2,8,4]]\n for n in 0..2 { for i in 0..4 { for j in 0..8 { a[n,i,j]=(((n*32+i*8+j) as f32)*0.7f32+0.1f32)/3.0f32 } } }\n for n in 0..2 { for i in 0..8 { for j in 0..4 { b[n,i,j]=(((n*32+i*4+j) as f32)*0.3f32+0.2f32)/7.0f32 } } }\n (a@b)[1,2,3] as f64 }", "#481 batched matmul contracts too"),
    # #478: an unsuffixed literal beside an f32 branch adopts f32 on BOTH
    # backends (it was a tracked jit gap until the literal started being
    # lowered at the type the checker inferred). The literal is `0.1`, not
    # `0.0`: only a value that differs between the two widths can catch a
    # backend that reverted to f64. Both branch-taken directions are probed —
    # the else-taken one returns the literal's own value and is where `dmc run`
    # used to answer 0.3000000029802322 instead of 0.30000001192092896.
    "f32_branch_lit_then":  ("fn main()->f64{ let a=0.1f32\n let z=if a>0.0f32 {a} else {0.1}\n (z+0.2f32) as f64 }", "#478 unsuffixed literal adopts the f32 branch (then taken)"),
    "f32_branch_lit_else":  ("fn main()->f64{ let a=0.1f32\n let z=if a>1.0f32 {a} else {0.1}\n (z+0.2f32) as f64 }", "#478 unsuffixed literal adopts the f32 branch (else taken)"),
    "f32_match_lit_arm":    ("fn main()->f64{ let a=0.1f32\n let n=2\n let z=match n { 1 => a, _ => 0.1 }\n (z+0.2f32) as f64 }", "#478 same at a match join (literal arm taken)"),
    "f32_branch_lit_f64":   ("fn main()->f64{ let a=0.5\n let z=if a>1.0 {a} else {0.1}\n z }", "#478 must NOT narrow: an all-f64 join stays f64 (#209)"),
    "f32_branch_join":    ("fn main()->f64{ let a=0.1f32\n let z=if a>0.0f32 {a+0.2f32} else {0.0f32}\n (z+0.3f32) as f64 }", "#473 both if-branches f32"),
    "f32_match_join":     ("fn main()->f64{ let n=1\n let a=0.1f32\n let z=match n { 1 => a+0.2f32, _ => 0.0f32 }\n (z+0.3f32) as f64 }", "#473 both match arms f32"),
    "f32_pow":            ("fn main()->f64{ let a=1.1f32\n let b=3.0f32\n (a**b) as f64 }", "#473 `**` widens to the f64 libm call and rounds back"),
    "f32_mod":            ("fn main()->f64{ let a=5.3f32\n let b=2.1f32\n (a%b) as f64 }", "#473 `%` widens to the f64 fmod and rounds back"),
    "f32_sqrt_chain":     ("fn main()->f64{ let a=2.0f32\n (sqrt(a)+0.1f32) as f64 }", "#473 transcendentals compute in f64 on BOTH backends (#209)"),
    "f32_from_bits":      ("fn main()->f64{ let a=f32_from_bits(1036831949)\n (a+0.2f32) as f64 }", "#473 f32_from_bits yields an f32, not an f64"),
    "f32_bf16_suffix":    ("fn main()->f64{ let a=0.1bf16\n let b=0.2bf16\n (a+b) as f64 }", "#473/#179 bf16 is f32-backed"),
    "f32_f16_suffix":     ("fn main()->f64{ let a=0.1f16\n let b=0.2f16\n (a+b) as f64 }", "#473/#179 f16 is f32-backed"),
    "f32_deep_chain":     ("fn main()->f64{ let !s=1.0f32\n for i in 0..50 { s = s * 1.01f32 + 0.001f32 }\n s as f64 }", "#473 50 fused multiply-adds — drift is unmissable here"),
    "f32_loop_predicate": ("fn main()->i64{ let !s=0.0f32\n let !n=0\n while s < 1.0f32 { s = s + 0.1f32\n n = n + 1 }\n n }", "#473 f32 rounding decides the trip count"),
    "f64_unsuffixed_keeps_width": ("fn main()->f64{ let a=0.1\n let b=0.2\n let c=0.3\n a+b+c }", "#209 must NOT regress: unsuffixed literals stay f64"),
    # ── `if` joins: which side reaches the join, and carrying what ─────────
    # The join's phi param is decided before the else side is lowered, so a
    # wrong guess is a Cranelift `mismatched argument count` — a whole-function
    # compile failure, which scores as a divergence here rather than a gap.
    # These are the shapes that used to produce one; none of them is exotic,
    # and the first is how PORTS.md §6's guarded idiom is written.
    "if_else_trailing_bare_if":  ("fn main()->i64{ let n=5\n if n>3 { 0 } else { if n != 1 { print(\"x\") } }\n n }", "an else block ending in an `if` with no `else` yields nothing"),
    "if_else_trailing_bare_if_valued": ("fn main()->i64{ let n=5\n if n>3 { 0 } else { if n != 1 { 7 } }\n n }", "…still nothing when that bare `if` has a value to give"),
    "if_then_trailing_bare_if":  ("fn main()->i64{ let n=5\n if n>3 { if n != 1 { print(\"x\") } } else { 0 }\n n }", "the mirror: the THEN side is the one arriving empty-handed"),
    "if_elseif_chain_no_final_else": ("fn main()->i64{ let n=5\n if n>3 { 0 } else if n != 1 { print(\"x\") }\n n }", "an `else if` chain with no final `else` yields nothing"),
    "if_nested_both_sides_still_value": ("fn main()->i64{ let n=5\n let x=if n>3 { 1 } else { if n != 1 { 2 } else { 3 } }\n x }", "…but a trailing `if` with both sides still carries its value"),
    "if_diverging_else_keeps_then_value": ("fn main()->i64{ let n=5\n let x=if n>3 { 5 } else { return 0 }\n x }", "`else { return }` never reaches the join, so the then value is the answer"),
    "if_diverging_else_taken": ("fn main()->i64{ let n=1\n let x=if n>3 { 5 } else { return 42 }\n x }", "…and the diverging side still decides when the guard fails"),

    # ── process ports (PORTS.md §2/§6/§7.1) ─────────────────────────────────
    # Both backends now drive one `crate::ports::PortRegistry`, so these probe
    # the marshalling on each side of it: the tag texts, the `(_, Err)` shape,
    # and the nil-vs-empty-str distinction the JIT encodes as a null pointer.
    # Every python-dependent probe is guarded by its own `port_open` check, so
    # a machine without python3 answers `false` on BOTH backends (agreeing)
    # instead of crashing one of them.
    "port_open_unsupported_tag": ('fn main()->bool{ let (p,e)=port_open("lua")\n e.starts_with("port-open") }', "PORTS §6 port-open: a runtime that is not wired up"),
    "port_open_failed_handle_nil": ('fn main()->bool{ let (p,e)=port_open("lua")\n p == nil }', "a failed open hands back a nil handle, not an empty str"),
    "port_open_failed_handle_is_not_live": ('fn main()->bool{ let (p,e)=port_open("lua")\n p != nil }', "…so the `!= nil` liveness test says false, not true"),
    "port_call_error_result_is_empty_not_nil": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"math.sqrt","[\\"x\\"]")\n let (_,_)=port_close(p)\n o == "" && o != nil } }', "PORTS §6 the result half of a failed call is the EMPTY str, not nil"),
    "port_call_positional_spread": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (g,e2)=port_call(p,"math.gcd","[462, 1071]")\n let (_,_)=port_close(p)\n g == "21" && e2 == nil } }', "PORTS §2 a JSON array spreads positionally"),
    "port_call_niladic_nil_payload": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"list",nil)\n let (_,_)=port_close(p)\n o == "[]" && e2 == nil } }', "PORTS §2 a nil payload calls with no arguments"),
    "port_call_kwargs_envelope": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"round","{\\"args\\": [3.14159], \\"kwargs\\": {\\"ndigits\\": 2}}")\n let (_,_)=port_close(p)\n o == "3.14" && e2 == nil } }', "PORTS §2 the {args, kwargs} envelope"),
    "port_call_foreign_exception_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"math.sqrt","[\\"sixteen\\"]")\n let (_,_)=port_close(p)\n e2.starts_with("port-call") && o == "" } }', "PORTS §6 port-call, and the result half is the empty str"),
    "port_call_bare_scalar_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"math.sqrt","16")\n let (_,_)=port_close(p)\n e2.starts_with("port-protocol") } }', "PORTS §2/§6 a bare scalar is not an argument vector"),
    "port_call_after_close_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,_)=port_close(p)\n let (o,e2)=port_call(p,"len","[[1, 2, 3]]")\n e2.starts_with("port-closed") } }', "PORTS §6 port-closed: a call through a closed handle"),
    "port_double_close_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,first)=port_close(p)\n let (_,second)=port_close(p)\n first == nil && second.starts_with("port-closed") } }', "PORTS §6 port-closed on the second close, clean on the first"),
    "port_ids_are_not_reused": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,_)=port_close(p)\n let (q,_)=port_open("python")\n let (o,e2)=port_call(p,"len","[[1, 2, 3]]")\n let (o2,e3)=port_call(q,"len","[[1, 2, 3]]")\n let (_,_)=port_close(q)\n e2.starts_with("port-closed") && o2 == "3" && e3 == nil } }', "a stale handle cannot reach the port opened after it"),
    "port_survives_an_error": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,_)=port_call(p,"math.sqrt","[\\"sixteen\\"]")\n let (g,e2)=port_call(p,"math.gcd","[462, 1071]")\n let (_,_)=port_close(p)\n g == "21" && e2 == nil } }', "a foreign exception does not desync or kill the port"),
    "port_lang_annotation_is_checked": ('fn ask(q: Port[lua]) -> str { let (o,_)=port_call(q,"list",nil)\n o }\nfn main()->bool{ let (p,e)=port_open("python")\n if e != nil { true } else { let r=ask(p)\n let (_,_)=port_close(p)\n false } }', "SPEC §3.11 `Port[L]` refuses a handle opened for another runtime (both backends die)"),
    "port_lang_annotation_accepts_its_own": ('fn ask(q: Port[python]) -> str { let (o,_)=port_call(q,"list",nil)\n o }\nfn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let r=ask(p)\n let (_,_)=port_close(p)\n r == "[]" } }', "…and still binds the matching one"),
    "port_payload_escaping": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (n,e2)=port_call(p,"len","[\\"he said \\\\\\"hi\\\\\\"\\\\n\\"]")\n let (a,e3)=port_call(p,"len","[[1, 2, 3]]")\n let (_,_)=port_close(p)\n n == "13" && a == "3" && e2 == nil && e3 == nil } }', "a quote+newline payload crosses intact and leaves the frame in sync"),
    # ── tensor copy mode (PORTS.md §3.2) ────────────────────────────────────
    # The envelope is the ABI, so these probes compare it to its literal text:
    # both backends marshal into one `crate::ports` writer, and a probe that
    # returned `true` on one side and `false` on the other would be exactly the
    # drift the shared registry exists to prevent. Only the last one needs
    # python3 — the format itself is demoniC's, and is checked without a port.
    "port_tensor_envelope_i64": ('fn main()->bool{ let !g=forge.zeros[i64,[2,3]]\n g[0,0]=1 g[0,1]=2 g[0,2]=3 g[1,0]=-4 g[1,1]=5 g[1,2]=6\n port_tensor_encode(g) == "{\\"data\\":\\"AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA/P////////8FAAAAAAAAAAYAAAAAAAAA\\",\\"dmc_tensor\\":1,\\"dtype\\":\\"i64\\",\\"layout\\":\\"row_major\\",\\"shape\\":[2,3]}" }', "PORTS §3.2 the copy-mode envelope, byte for byte"),
    "port_tensor_envelope_bool_width": ('fn main()->bool{ let !b=forge.uninit[bool,[3]]\n b[0]=true b[1]=false b[2]=true\n port_tensor_encode(b) == "{\\"data\\":\\"AQAB\\",\\"dmc_tensor\\":1,\\"dtype\\":\\"bool\\",\\"layout\\":\\"row_major\\",\\"shape\\":[3]}" }', "PORTS §3.2 the wire dtype states the payload's real width — 1 byte per bool (#292)"),
    "port_tensor_round_trip_i64": ('fn main()->bool{ let !g=forge.zeros[i64,[2,2]]\n g[0,0]=10 g[0,1]=-20 g[1,0]=30 g[1,1]=-40\n let (t,e)=port_tensor_decode(port_tensor_encode(g), forge.zeros[i64,[2,2]])\n e == nil && t[0,1] == -20 && t[1,1] == -40 }', "PORTS §3.2 an integer tensor survives the round trip exactly"),
    "port_tensor_round_trip_bool": ('fn main()->bool{ let !b=forge.uninit[bool,[3]]\n b[0]=true b[1]=false b[2]=true\n let (t,e)=port_tensor_decode(port_tensor_encode(b), forge.uninit[bool,[3]])\n e == nil && t[0] && !t[1] && t[2] }', "PORTS §3.2 a bool tensor comes back as bools, not as numbers"),
    "port_tensor_decode_shape_tag": ('fn main()->bool{ let !g=forge.zeros[i64,[4]]\n let (t,e)=port_tensor_decode(port_tensor_encode(g), forge.zeros[i64,[2]])\n e == "decode-type: expected tensor shape [2], got [4]" && t[0] == 0 }', "PORTS §3.1/§3.2 an undeclared shape is decode-type, and the zero rides the error path"),
    "port_tensor_decode_kind_tag": ('fn main()->bool{ let !f=forge.zeros[f32,[2]]\n let (t,e)=port_tensor_decode(port_tensor_encode(f), forge.zeros[i64,[2]])\n e == "decode-type: expected a `i64` tensor, got `f32`" }', "PORTS §3.1 a typed tensor decode never coerces across a kind"),
    "port_tensor_decode_parse_tag": ('fn main()->bool{ let (t,e)=port_tensor_decode("{oops", forge.zeros[i64,[2]])\n e.starts_with("decode-parse") && t[0] == 0 && t[1] == 0 }', "PORTS §6 text that is not JSON is decode-parse, not decode-type"),
    "port_tensor_copy_through_the_port": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let !g=forge.zeros[i64,[2,2]]\n g[0,0]=1 g[0,1]=2 g[1,0]=3 g[1,1]=4\n let (o,e2)=port_call(p,"dmc.echo","[" + port_tensor_encode(g) + "]")\n let (s,e3)=port_call(p,"dmc.shape","[" + port_tensor_encode(g) + "]")\n let (_,_)=port_close(p)\n let (t,e4)=port_tensor_decode(o, forge.zeros[i64,[2,2]])\n e2 == nil && e4 == nil && s == "[2,2]" && t[1,1] == 4 } }', "PORTS §3.2 a tensor crosses a live port by copy and comes back"),
    # ── @grad reverse-tape widths and shapes (#422 wall 2) ──────────────────
    # `f$fwd_bwd` is a second function the JIT emits from the same body, and
    # for a long time nothing in this battery compiled one. These four cover
    # the shapes that used to make Cranelift reject it outright — a compile
    # failure, so they would have scored as a divergence, not a gap.
    "grad_unsuffixed_scalar_mul": (
        "@grad fn f(!w:Tensor[f32,[3]])->f32{ let s=sum(w)\n s*2.0 }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[3]]\n w[0]=1.0 w[1]=2.0 w[2]=3.0\n"
        " let (l,g)=f.fwd_bwd(w)\n (l + g.w[0]) as f64 }",
        "#422 wall 2: unsuffixed literal (f64) times a traced scalar — 12+2"),
    "grad_unsuffixed_scalar_div": (
        "@grad fn f(!w:Tensor[f32,[3]])->f32{ let s=sum(w)\n s/4.0 }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[3]]\n w[0]=1.0 w[1]=2.0 w[2]=3.0\n"
        " let (l,g)=f.fwd_bwd(w)\n (l + g.w[0]) as f64 }",
        "#422 wall 2: the other operand-reading rule — 1.5+0.25"),
    "grad_loss_declared_width": (
        "@grad fn f(!w:Tensor[f32,[3]])->f32{ let s=sum(w)\n s+0.1 }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[3]]\n w[0]=1.0 w[1]=2.0 w[2]=3.0\n"
        " let (l,_g)=f.fwd_bwd(w)\n l as f64 }",
        "#473: `.fwd_bwd`'s loss narrows to the declared `-> f32`, as a plain call does"),
    "grad_matmul_rank3": (
        "@grad fn f(!a:Tensor[f32,[1,1,2]], b:Tensor[f32,[1,2,2]])->f32{ let y=a@b\n sum(y .* y) }\n"
        "fn main()->f64{ let !a=forge.zeros[f32,[1,1,2]]\n a[0,0,0]=1.0 a[0,0,1]=2.0\n"
        " let !b=forge.zeros[f32,[1,2,2]]\n b[0,0,0]=3.0 b[0,0,1]=4.0 b[0,1,0]=5.0 b[0,1,1]=6.0\n"
        " let (l,g)=f.fwd_bwd(a,b)\n (l + g.a[0,0,0]) as f64 }",
        "#422: rank-3 batched matmul VJP — the backward transposes, and rank 3 was missing"),
    # ── model-field bindings (#524, SPEC §3.4) ──────────────────────────────
    # `let !y = m.w` on a TENSOR field is a LIVE ALIAS on both backends: every
    # write through the binding reaches the field. The JIT used to copy here
    # (its #249 value-copy rule swept up the field case too), so each of the
    # first three probes was a silent wrong-answer divergence: `dmc run` saw
    # the field move, `dmc jit` saw it stand still. The negatives that follow
    # pin the other half of the rule — `mut`, a plain `let`, a scalar field,
    # and a plain local all still snapshot, which is what #249 fixed.
    "field_alias_elem_write": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let !y=m.w\n y[0]=42.0\n m.w[0] as i64 }",
        "#524 element write through a tensor-field alias reaches the field"),
    "field_alias_whole_assign": (
        "model M { !w: Tensor[f32,[4]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[4]] }\n"
        " let !y=m.w\n y = forge.ones[f32,[4]] .* 7.0\n (m.w[0]+m.w[3]) as i64 }",
        "#524 whole-binding `=` through the alias writes the field, not the local"),
    "field_alias_compound_assign": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let !y=m.w\n y += forge.ones[f32,[2]]\n y += forge.ones[f32,[2]]\n m.w[1] as i64 }",
        "#524 compound `+=` reads the field and writes the result back to it"),
    "field_alias_mut_spelling_copies": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let mut y=m.w\n y[0]=42.0\n (m.w[0]+y[0]*100.0) as i64 }",
        "#524 `let mut` never aliases — this is where the two spellings differ"),
    "field_alias_immutable_bind_copies": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let y=m.w\n m.w[0]=42.0\n (y[0]+m.w[0]*100.0) as i64 }",
        "#524 a plain `let` from a field is a snapshot"),
    "field_alias_scalar_field_copies": (
        "model M { !n: i64 }\n"
        "fn main()->i64{ let !m=M { n: 7 }\n let !s=m.n\n s=99\n m.n*100+s }",
        "#524/#444 a SCALAR field binds its current value, not an alias"),
    "field_alias_plain_local_copies": (
        "fn main()->i64{ let !x=forge.zeros[f32,[3]]\n x[0]=1.0\n let !y=x\n y[0]=9.0\n"
        " (x[0]+y[0]*100.0) as i64 }",
        "#249 must NOT regress: a plain local still value-copies"),
    "field_alias_constructor_copies": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !t=forge.zeros[f32,[2]]\n let !m=M { w: t }\n t[0]=9.0\n"
        " m.w[0] as i64 }",
        "#249 must NOT regress: `M { w: t }` snapshots the constructor argument"),
    "field_alias_shadowed": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let !y=m.w\n y[0]=5.0\n let !y=forge.zeros[f32,[2]]\n y[0]=9.0\n"
        " (m.w[0]+y[0]*100.0) as i64 }",
        "#524 a shadowing `let` drops the alias — the later write stays local"),
    # ── non-f32 tensor value copies (#552) ──────────────────────────────────
    # #249's snapshot (`let !y = x`, `M { w: t }`, a tuple element, and the
    # `!`-argument copy/writeback) lowered to a loop that loaded EVERY element
    # as f32 while striding by the tensor's real element width. For an i64/f64
    # tensor that copied only the low 4 bytes of each element and left the high
    # 4 bytes as whatever the destination arena already held. A fresh forge
    # allocation is zeroed, so small positive values came out intact — which is
    # why the whole corpus stayed green. Every probe here therefore carries
    # bits ABOVE 2^32 (i64) or in the exponent (f64), where the two backends
    # part company: `dmc jit` answered 705032704 for 5000000000, and turned an
    # ordinary f64 into a denormal near zero.
    "copy_i64_let": (
        "fn main()->i64{ let !x=forge.uninit[i64,[2]]\n x[0]=5 x[1]=5000000000\n"
        " let !y=x\n y[0]=9\n y[1] }",
        "#552 `let !y = x` copies an i64 tensor at full width"),
    "copy_i64_ctor": (
        "model M { !w: Tensor[i64,[2]] }\n"
        "fn main()->i64{ let !t=forge.uninit[i64,[2]]\n t[0]=5000000000 t[1]=1\n"
        " let !m=M { w: t }\n t[0]=1\n m.w[0] }",
        "#552 `M { w: t }` snapshots an i64 tensor at full width"),
    "copy_i64_tuple": (
        "fn main()->i64{ let !x=forge.uninit[i64,[2]]\n x[0]=5000000000 x[1]=1\n"
        " let p=(x,7)\n x[0]=1\n let (a,b)=p\n a[0] }",
        "#552 a tuple element snapshots an i64 tensor at full width"),
    "copy_f64_let": (
        "fn main()->f64{ let !x=forge.uninit[f64,[2]]\n x[0]=0.5 x[1]=0.1\n"
        " let !y=x\n y[0]=9.0\n y[1] }",
        "#552 `let !y = x` copies an f64 tensor at full width (a 4-byte read loses the exponent)"),
    "copy_f64_ctor": (
        "model M { !w: Tensor[f64,[2]] }\n"
        "fn main()->f64{ let !t=forge.uninit[f64,[2]]\n t[0]=0.1 t[1]=0.25\n"
        " let !m=M { w: t }\n t[0]=9.0\n m.w[0]+m.w[1] }",
        "#552 `M { w: t }` snapshots an f64 tensor at full width"),
    # A bool element is 1 byte, so the old f32 read returned the RIGHT value
    # (the 4-byte store just rewrote its neighbours with their own bytes) while
    # running 3 bytes past the end of both buffers. These two agreed before and
    # must keep agreeing now that the copy is exactly `n` bytes wide.
    "copy_bool_let": (
        "fn main()->i64{ let !x=forge.uninit[bool,[2]]\n x[0]=false x[1]=true\n"
        " let !y=x\n y[0]=true\n if y[1] {1} else {0} }",
        "#552 `let !y = x` copies a bool tensor without over-reading past it"),
    "copy_bool_ctor": (
        "model M { !w: Tensor[bool,[2]] }\n"
        "fn main()->i64{ let !t=forge.uninit[bool,[2]]\n t[0]=true t[1]=false\n"
        " let !m=M { w: t }\n t[0]=false\n if m.w[0] {1} else {0} }",
        "#552 `M { w: t }` snapshots a bool tensor without over-reading past it"),
    # The `!`-ARGUMENT copy/writeback uses the same loop, but the JIT lowers a
    # `!` parameter only inside a `@grad fn` (and `@grad` is f32-only), so the
    # non-f32 writeback is not reachable today — these three are tracked gaps,
    # not silent agreement. They go live on their own the day `!` parameters
    # lower outside `@grad`; the f32 probe below covers the path that IS live.
    "copy_i64_mut_arg": (
        "fn bump(!t: Tensor[i64,[2]])->nil{ t[0]=t[0]+1  nil }\n"
        "fn main()->i64{ let !x=forge.uninit[i64,[2]]\n x[0]=5000000000 x[1]=0\n"
        " bump(x)\n x[0] }",
        "#552 `!` i64 tensor argument round-trips at full width (jit gap: `!` params are @grad-only)"),
    "copy_f64_mut_arg": (
        "fn bump(!t: Tensor[f64,[2]])->nil{ t[0]=t[0]+0.25  nil }\n"
        "fn main()->f64{ let !x=forge.uninit[f64,[2]]\n x[0]=0.5 x[1]=0.1\n"
        " bump(x)\n x[0]+x[1] }",
        "#552 `!` f64 tensor argument round-trips at full width (jit gap: `!` params are @grad-only)"),
    "copy_bool_mut_arg": (
        "fn flip(!t: Tensor[bool,[2]])->nil{ t[0]=true  nil }\n"
        "fn main()->i64{ let !x=forge.uninit[bool,[2]]\n x[0]=false x[1]=true\n"
        " flip(x)\n if x[0] {1} else {0} }",
        "#552 `!` bool tensor argument round-trips (jit gap: `!` params are @grad-only)"),
    "copy_f32_grad_arg": (
        "@grad fn loss(!w: Tensor[f32,[2]])->f32{ sum(w .* w) }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[2]]\n w[0]=1.5 w[1]=2.0\n"
        " let (l,g)=loss.fwd_bwd(w)\n (l+g.w[0]) as f64 }",
        "#552 the one `!` tensor-argument copy the JIT does lower stays correct"),
    # ── `!` parameter copy-in/copy-out (#249, #553) ─────────────────────────
    # A `!` tensor parameter is copy-in/copy-out: the caller hands the callee a
    # private copy and copies that buffer back on return. #553: a whole-binding
    # assignment in the callee rebinds the local to a FRESH forge buffer, so the
    # buffer the caller wrote back was the stale one it passed in — `dmc run`
    # saw the new values, `dmc jit` the old. Element writes hid it, since they
    # mutate the copy in place. Note the two shapes a `!` parameter can take
    # here: a `@grad fn` (where `declare_fn` requires the directive) and a
    # template (where `declare_template_signature` accepts one without it).
    "mut_param_reassign_writes_back": (
        "@grad fn f(!w:Tensor[f32,[2]])->f32{ w = w .* 2.0\n sum(w .* w) }\n"
        "fn main()->i64{ let !w=forge.zeros[f32,[2]]\n w[0]=1.0 w[1]=2.0\n"
        " let s=f(w)\n (s + w[0]*100.0) as i64 }",
        "#553 a reassigned `!` param writes back the buffer it now names — 220, not 120"),
    "mut_param_reassign_writes_back_template": (
        "fn scale[N](!t:Tensor[f32,[N]])->f32{ t = t .* 2.0\n t[0] }\n"
        "fn main()->i64{ let !x=forge.zeros[f32,[3]]\n x[0]=1.0\n"
        " let d=scale(x)\n (d + x[0]*100.0) as i64 }",
        "#553 the same on a template `!` param — no `@grad` needed to reach one"),
    "mut_param_reassign_leaves_others_alone": (
        "@grad fn f(!a:Tensor[f32,[2]], !b:Tensor[f32,[2]])->f32{ a = a .* 3.0\n sum(a .* b) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[2]]\n let !b=forge.zeros[f32,[2]]\n"
        " a[0]=1.0 a[1]=2.0 b[0]=5.0 b[1]=7.0\n let s=f(a,b)\n"
        " (s + a[0]*100.0 + b[0]*10000.0) as i64 }",
        "#553 only the reassigned param moves — `b` comes back as it went in"),
    "mut_param_reassign_isolated_from_aliased_arg": (
        "@grad fn f(!a:Tensor[f32,[2]], b:Tensor[f32,[2]])->f32{ a = a .* 2.0\n sum(a) + sum(b) }\n"
        "fn main()->i64{ let !t=forge.zeros[f32,[2]]\n t[0]=1.0 t[1]=2.0\n"
        " let s=f(t,t)\n (s + t[0]*100.0) as i64 }",
        "#249 must NOT regress: `f(t,t)` keeps the reassignment out of the plain `b`"),
    "mut_param_no_reassign_writes_back": (
        "fn poke[N](!t:Tensor[f32,[N]])->nil{ t[0]=77.0 nil }\n"
        "fn main()->i64{ let !x=forge.zeros[f32,[3]]\n poke(x)\n x[0] as i64 }",
        "#249 must NOT regress: an element write still reaches the caller"),
    "mut_param_fwd_bwd_does_not_write_back": (
        "@grad fn f(!w:Tensor[f32,[2]])->f32{ w = w .* 2.0\n sum(w .* w) }\n"
        "fn main()->i64{ let !w=forge.zeros[f32,[2]]\n w[0]=1.0 w[1]=2.0\n"
        " let (l,_g)=f.fwd_bwd(w)\n (l + w[0]*100.0 + w[1]*10000.0) as i64 }",
        "#553 `.fwd_bwd` passes the caller's own tensor and writes back nothing, on both backends"),
    # ── slab bounds built from a SHAPE PARAMETER ────────────────────────────
    # Every slice this battery and the corpus reached had LITERAL bounds. A
    # bound that mentions `S` cannot be resolved until the generic is
    # monomorphized, and the two backends reach it by different routes: the
    # interpreter substitutes the binding at call time, the JIT folds a
    # constant into the body it emits per instantiation. Nothing gated that.
    # Each probe therefore calls one generic at TWO shapes and packs both
    # answers into a single scalar, so a backend that folded once and reused
    # the result returns a plausible number that is arithmetically wrong.
    "slab_end_is_shape_param": (
        "fn tail[S](x:Tensor[f32,[S]])->f32{ sum(x[S-2 .. S]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[6]]\n for i in 0..6 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[3]]\n for i in 0..3 { b[i]=(i+10) as f32 }\n"
        " (tail(a)*1000.0+tail(b)) as i64 }",
        "a fixed-length window anchored at `S` — 11 on [6], 23 on [3]"),
    "slab_len_is_shape_param": (
        "fn half[S](x:Tensor[f32,[S]])->f32{ sum(x[0 .. S/2]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[5]]\n for i in 0..5 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[6]]\n for i in 0..6 { b[i]=(i+1) as f32 }\n"
        " (half(a)*1000.0+half(b)) as i64 }",
        "the slab's own LENGTH moves, and `S/2` truncates on the odd shape — 3 then 6"),
    "slab_both_ends_derived": (
        "fn mid[S](x:Tensor[f32,[S]])->f32{ sum(x[1 .. S-1]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[6]]\n for i in 0..6 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[3]]\n for i in 0..3 { b[i]=(i+10) as f32 }\n"
        " (mid(a)*1000.0+mid(b)) as i64 }",
        "start and end both derived — 14 on [6], 11 on [3]"),
    "slab_full_range_via_param": (
        "fn whole[S](x:Tensor[f32,[S]])->f32{ sum(x[0 .. S]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[6]]\n for i in 0..6 { a[i]=(i+1) as f32 }\n"
        " whole(a) as i64 }",
        "`x[0 .. S]` is the whole tensor — an off-by-one in the derived end shows only here"),
    "slab_2d_row_from_param": (
        "fn lastrow[R,C](m:Tensor[f32,[R,C]])->f32{ sum(m[R-1 .. R, ..]) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " lastrow(m) as i64 }",
        "row slab from `R` on a grid of distinct elements — 9+10+11"),
    "slab_2d_col_from_param": (
        "fn righthalf[R,C](m:Tensor[f32,[R,C]])->f32{ sum(m[.., C/2 .. C]) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " righthalf(m) as i64 }",
        "column slab from `C`, applied down every row — 48"),
    "slab_is_a_matmul_operand": (
        "fn slabmm[R,C](m:Tensor[f32,[R,C]], w:Tensor[f32,[C,C]])->f32{ sum(m[0 .. R/2, ..] @ w) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " let !w=forge.zeros[f32,[3,3]]\n for i in 0..3 { w[i,i]=1.0 }\n"
        " slabmm(m,w) as i64 }",
        "the derived extent has to survive into `@`, not just into a reduction — 15"),
    "slab_feeds_axis_reduction": (
        "fn slaba[R,C](m:Tensor[f32,[R,C]])->f32{ sum(sum_along(m[0 .. R-1, ..], 0)) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " slaba(m) as i64 }",
        "a derived slab as the argument of a per-axis reduction — 36"),
    # ── the shape parameter as a VALUE, and tensor-valued joins ─────────────
    # `N` is also an ordinary integer in the body: a `forge` extent, a loop
    # bound, an `N as f32` divisor. And the `if_*` family above joins only
    # i64 — a join whose arms are TENSORS carries a buffer through the phi,
    # which outside `@grad` nothing reached.
    "shape_param_as_divisor": (
        "fn avg[N](x:Tensor[f32,[N]])->f32{ let !acc=forge.zeros[f32,[N]]\n"
        " for i in 0..N { acc[i]=x[i]/(N as f32) }\n sum(acc) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[8]]\n for i in 0..8 { b[i]=(i+1) as f32 }\n"
        " ((avg(a)*1000.0+avg(b))*10.0) as i64 }",
        "`N` is the local's extent, the loop bound AND the divisor — 2.5 then 4.5"),
    "shape_param_bounds_a_while": (
        "fn total[N](x:Tensor[f32,[N]])->f32{ let !i=0\n let !acc=0.0f32\n"
        " while i < N { acc = acc + x[i]\n i = i + 1 }\n acc }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[8]]\n for i in 0..8 { b[i]=(i+1) as f32 }\n"
        " (total(a)*1000.0+total(b)) as i64 }",
        "the parameter as a `while` predicate — corpus `while` bounds are module constants"),
    "explicit_shape_arg_sizes_result": (
        "fn ramp[N]()->Tensor[f32,[N]]{ let !t=forge.zeros[f32,[N]]\n"
        " for i in 0..N { t[i]=(i*2) as f32 }\n t }\n"
        "fn main()->i64{ let a=ramp[5]()\n let b=ramp[3]()\n"
        " (sum(a)*1000.0+sum(b)) as i64 }",
        "no argument mentions `N`, so only the explicit shape argument can size the result"),
    "tensor_valued_if_join": (
        "fn branch[N](x:Tensor[f32,[N]], d:bool)->f32{ let y=if d { x .* 2.0 } else { x .+ 1.0 }\n sum(y) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " (branch(a,true)*1000.0+branch(a,false)) as i64 }",
        "both arms are TENSORS, so the phi carries a buffer — the `if_*` family above is all i64"),
    "tensor_valued_match_join": (
        "fn pick[N](x:Tensor[f32,[N]], k:i64)->f32{ let y=match k { 0 => x .* 2.0, 1 => x .+ 1.0, _ => x }\n sum(y) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " (pick(a,0)*10000.0+pick(a,1)*100.0+pick(a,9)) as i64 }",
        "…same at a match join, and the default arm yields a buffer the callee never allocated"),
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
        elif rs == "ok" and _UNSUPPORTED in jv:
            gaps += 1
            verdict = f"jit-gap: {jv[:60]}"
            if name not in GAP_ALLOWLIST:
                new_gaps.append((name, jv[:80]))
        elif rs == "ok" and js != "ok" and name in GAP_ALLOWLIST:
            # A jit FAILURE (not a refusal) on a program `dmc run` accepts is a
            # defect — but an explicitly tracked one is allowlistable, so a known
            # issue does not force the probe out of the battery (#480).
            gaps += 1
            verdict = f"jit-gap (tracked failure): {jv[:60]}"
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
