# demoniC — Numerics Contract

**Companion to:** `docs/SPEC.md §7.2b` (Float accumulation contract), `§7.3`
(Determinism contract (`@deterministic`)); `docs/PORTS.md §3.2`;
`docs/DIRECTIVES.md §1`; `docs/STABILITY.md` (§1's promises fall under that
document's change procedure).

This document is normative.

`docs/SPEC.md §7.2b` and `§7.3` each state one accumulation or reproducibility
rule. This document collects what those rules add up to across the whole
language and states each as a promise or a non-promise.

The distinction that matters throughout: a *promise* is append-only — it can
be extended, and it is not silently narrowed or reworded once shipped. A
*non-promise* is not a bug; it is scope the language has not taken on, named
so nobody depends on it by accident.

---

## 1. What is promised

### 1.1 Per-op accumulation, not blanket bit-exactness

`docs/SPEC.md §7.2b` fixes the accumulation of specific ops, not float
arithmetic in general:

- f32 **reductions** (`sum`, `mean`, `variance`, `*_along`) accumulate at f32
  width, in index order, rounding after every add.
- f32 **matmul** (`@`) accumulates ascending over the inner dimension and
  *contracts* — one fused multiply-add per step, one rounding — and does so
  identically whether or not the host has an FMA unit (a software fallback is
  correctly rounded rather than splitting the operation).

These are two different, individually-fixed primitives. Matmul is not
required to agree with a hand-written reduction loop over the same values,
because it is not shorthand for that loop (§7.2b says so directly). The
promise is *per op*: each one named above has one answer, stable across hosts
and across the two backends, and a future release may add a rule for a new op
without touching these two.

### 1.2 `dmc run` / `dmc jit` parity

Outside `@deterministic`: bit-identical for integer and bool programs within
f64's exact-integer range (2^53). The interpreter stores every tensor element
as f64, so an integer past that range is not the value it was given long
before either backend's numerics could differ.

Float programs are tolerance-matched, not bit-exact. `tools/numpy_oracle.py`'s
`rtol`/`atol` are the calibrated bound both backends are held to today;
closing the remaining gap is ongoing work, not a promise this document makes.

Inside `@deterministic` (`docs/SPEC.md §7.3`): bit-exact, given identical
inputs and `Rng` state, on the same host with the same rank count and mesh
configuration. This is the strongest reproducibility promise the language
makes; reach for it when an exact repeat matters more than which kernel runs
fastest.

### 1.3 Integer wrap-at-width

`i8`/`i16`/`u8`/`u16`/`u32`/`u64` arithmetic wraps at its declared width, on
both backends, scalar and elementwise-tensor alike. This is ordinary
two's-complement wraparound, not a moving target: it is the defined semantics
of these types now and stays so.

### 1.4 Port wire numerics (`docs/PORTS.md §3.2`)

The tensor envelope's `dtype` always reflects the value's storage width, never
the type annotation that produced it; narrowing on the wire is truncation,
never round-to-nearest; and an untouched round trip through a port is
byte-identical for every bit pattern a dtype can hold, signaling NaNs
included. `docs/PORTS.md §3.1`'s adjacent whole-value int/float wire decision
is ratified alongside this document — see that section for the dated entry.

---

## 2. What is NOT promised

### 2.1 Matching another implementation's summation order

§1.1's accumulation order is fixed *within demoniC*, across its own two
backends. It is not a claim that demoniC's float output matches any other
implementation's. NumPy and BLAS use pairwise or blocked summation; demoniC's
f32 reductions are sequential and its matmul contracts via fused multiply-add.
Both are valid f32 arithmetic that round differently.

The practical consequence, and it is a real one: a single pass over a deep
chain of f32 reductions can sit well inside `tools/numpy_oracle.py`'s
tolerance against a NumPy reference while a long *iterated* computation —
each step consuming the previous step's rounded output — drifts away from
that reference purely from compounded summation-order differences. Neither
implementation is wrong. Treat the oracle's tolerance as the calibrated
bound, not zero-diff, whenever comparing demoniC's float output to an
external reference.

### 2.2 The `--blas` GEMM offload

**The rule, in full:**

> The BLAS GEMM fast path is **off by default**. It is enabled for one run by
> `dmc jit --blas`. It is **never** selected inside a `@deterministic` block,
> whatever the flag says. With the flag off, every matmul lowers exactly as it
> did before the flag existed, bit for bit.

With `--blas`, an f32 matmul of at least 2^17 (131,072) MACs is routed to the
host's `cblas_sgemm` — Accelerate's, on macOS, which is how the platform's
matrix coprocessor gets reached at all. **That path accumulates in BLAS's
blocked order, not §1.1's ascending-k contracted FMA.** Its output is
tolerance-equal to the default kernel's, not bit-equal, and this document
promises nothing tighter about it. Below the threshold, without the flag, or
inside `@deterministic`, §1.1 holds unchanged and unqualified.

**Nothing moved from §2 to §1.** That is the point of the default. §1.1
promises the contracted-FMA order on both backends and every kernel, and the
project has already paid once for an answer that depended on which kernel a
shape selected: a vectorized matmul that contracted while its scalar fallback
did not, so widening an operand by one column changed output elements that did
not depend on it. Defaulting the fast path on would have made §1.1 silently
conditional on whether the host process happens to contain a BLAS — a
difference with no cause visible in the source. Opt-in puts the trade on the
command line, where a tool can read it, and leaves §1.1 a promise rather than
a probability.

The corollary for anyone comparing against an external reference: `--blas`
does not narrow §2.1 and is not a step toward narrowing it. It replaces
demoniC's summation order with *a* blocked one, not with NumPy's; two blocked
orders disagree as readily as a blocked and a sequential one.

`@host`'s contract (`docs/SPEC.md §7.2` (Hardware dispatch (`@host`))) is that
the compiled-in arm is authoritative and different arms may have different
numerics by design. The detection this path uses is exposed there as the host
feature `accelerate`, so `@host match { .accelerate => … }` asks the same
question the JIT's kernel selection asks. `@host`'s own JIT lowering remains
pending — the feature name works in the interpreter's dispatch today, and the
JIT's fast path reads the same detection directly rather than through `@host`
syntax.

### 2.3 Reproducing a *specific* external reference

There is no directive today for "I need this exact reduction reproducible
against an external reference". Accumulation-width and algorithm control — an
opt-in directive that could, for instance, match NumPy's own pairwise
summation order — is a design, not addable syntax: it is not in the directive
catalog (`docs/DIRECTIVES.md §1`) because it does not exist. Until it does,
`@deterministic` (§1.2) is the only strict reproducibility knob available, and
it fixes demoniC's own order — it does not close the §2.1 gap against an
external reference.

### 2.4 A directive `docs/DIRECTIVES.md` marks not-yet-implemented

A directive whose catalog entry (`docs/DIRECTIVES.md §1`) reads "not
implemented" or "parse-accepted no-op" makes no numeric promise at all,
whatever its name suggests. `@recompute` computes nothing differently from its
absence today, and `@host`'s JIT lowering is pending (interpreter-only
dispatch works). Attachment-legality enforcement — where a directive is legal
to write — is not the same as the effect it promises being implemented; §3 of
that document is explicit about the difference, and this document adds nothing
beyond pointing at it.

---

## 3. Versioning discipline

Section 1 is append-only: a new op gets its own stated accumulation rule,
here or directly in `docs/SPEC.md §7.2b`, and an existing rule is not
silently redefined or narrowed once shipped. Section 2 shrinks over time as
its items land: a non-promise becoming a promise moves into Section 1 with a
dated note and a `CHANGELOG.md` entry, never as a silent change in behavior
between releases. `docs/STABILITY.md §3` is the procedure.
