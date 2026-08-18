# demoniC — Standard Library

**Companion to:** `SPEC.md §1` (Design constraints), `OPERATORS.md §3`.

The demoniC standard library is **small and opinionated**. Every
function listed here is part of the language's stability surface — its
type signature and the JIT-emitted machine-code shape may not change
across minor revisions without a deprecation cycle.

This document is normative.

---

## 1. Inclusion criteria

A function earns a slot in the standard library when **all three** of
the following hold:

1. It is universal across demoniC's primary use cases — algorithms that
   appear in AI/ML training and inference, numerical simulation, or
   high-performance systems code — and is not domain-specific to a
   narrower field (computer vision, audio, RL).
2. Its hand-rolled form would be re-implemented in nearly every
   demoniC program, badly, with subtle performance bugs.
3. The JIT can emit a substantially better kernel for it than user
   code could express through composition.

These criteria are intentionally strict. The stdlib does not grow on
convenience.

---

## 2. The fusion contract

Every function in this document compiles to **one kernel pass** over
its inputs. No Forge materialization of intermediates is permitted. A
JIT release that violates this contract is **wrong**, not slow.

The mechanism is the same one user code accesses via `@fuse`
(`SPEC.md §7.4`, Forced fusion (`@fuse`)). Stdlib functions get it
implicitly because their implementations sit inside an `@fuse` scope in
the JIT IR.

**Element type under `@cast`.** These builtins are generic over the element
type `T`, but wrapping a call in `@cast(bf16)` — or any exotic-float cast
(`f16`, `tf32`, `fp8`) — does **not** make the builtin compute in that type.
Exotic-float casts are f32-backed no-ops, so a builtin under `@cast(bf16)`
accumulates in **f32** regardless; bf16 is a nominal tag, not a compute
width. `@cast(bf16) { rms_norm(x) }` equals `rms_norm(x)`. See
`DIRECTIVES.md §4.1` for the normative contract and
`examples/cast_bf16_boundary.dmc` for the interp + JIT regression test.

---

## 3. Catalog

### 3.1 `attn` — fused attention

```
fn attn[B, H, S, D, T](
    q:    Tensor[T, [B, H, S, D]],
    k:    Tensor[T, [B, H, S, D]],
    v:    Tensor[T, [B, H, S, D]],
    mask: Tensor[T, [S, S]] | nil   # 0/1 mask; no dedicated bool-tensor form yet
) -> Tensor[T, [B, H, S, D]]
```

Computes `softmax((q @ k') / sqrt(D)) @ v` as a single FlashAttention-
style tiled streaming kernel. The `[B, H, S, S]` attention-score matrix
**never lives in Forge** — it exists only in registers, tile by tile.

The optional `mask` is a numeric 0/1 tensor or `nil`: `nil` (or the
3-arg form) skips masking entirely — no instructions emitted. A
non-`nil` mask is applied per tile; positions where the mask is `0`
are excluded before the softmax. Passing anything else is a runtime
error. (Bool-typed tensors cannot currently be constructed in-language,
so 0/1 numeric masks are the shipped form.)

**Differentiable:** a `@grad fn` body containing `attn` backprops to
all three of `q`, `k`, and `v` in both the interpreter and the JIT (first
order; static dense tensors — KV-cache operands stay forward-only). The
`@grad` forward saves each head's post-softmax weights and the backward
replays them; a fully fused no-materialization backward remains future work.
VJP rules and limits: `AUTODIFF.md §4`; parity example:
`examples/attn_grad_jit.dmc`.

#### Autoregressive variant

For single-token decode against a KV cache:

```
fn attn[B, H, D, T](
    q:    Tensor[T, [B, H, 1, D]],
    k:    KV[T, [B, H, ~, D]],
    v:    KV[T, [B, H, ~, D]],
    mask: nil
) -> Tensor[T, [B, H, 1, D]]
```

The KV operands use the streaming-axis type from `SPEC.md §3.6` (Streaming
types (KV caches)). The kernel reads the live extent of `~` from the cache
header at entry; no shape recomputation per call.

### 3.1b `attn_gqa` — grouped query attention

```
fn attn_gqa[B, H_q, H_kv, S, D, T](
    q:    Tensor[T, [B, H_q,  S, D]],
    k:    Tensor[T, [B, H_kv, S, D]],
    v:    Tensor[T, [B, H_kv, S, D]],
    mask: Tensor[T, [S, S]] | nil   # 0/1 mask; no dedicated bool-tensor form yet
) -> Tensor[T, [B, H_q, S, D]]
```

Constraint: `H_q % H_kv == 0` (enforced at compile time). Group size
`G = H_q / H_kv`. For each Q head `h`, attends against KV head `h / G`.

Equivalent semantics to:

```dmc
# Reference: correct but expands K/V in memory
let k_exp = k[.., repeat(G), .., ..]
let v_exp = v[.., repeat(G), .., ..]
attn(q, k_exp, v_exp, mask)
```

The fused kernel **never materialises the expanded K/V tensors**: it
indexes into KV head `h / G` directly at the score accumulation site.
This removes the `G × H_kv` memory overhead of the expansion path.

The default attention primitive for every major model family since 2023:
Qwen2.5, Llama-3, Mistral-v0.3, Gemma-2, Phi-3.

**Differentiable:** backprops to `q`, `k`, and `v` in both backends
(first order, dense operands), with `dK`/`dV` accumulated across each KV
head's query group. See `AUTODIFF.md §4` and `examples/attn_grad_jit.dmc`.

### 3.2 `softmax`

```
fn softmax[..., N, T](
    x: Tensor[T, [..., N]],
    axis: i64 = -1
) -> Tensor[T, [..., N]]
```

Numerically stable form: subtract the per-row max before exponentiating.
One streaming pass: `max`, `exp-and-sum`, `normalize`. Default
`axis = -1` if omitted.

**Differentiable.** Backward fused as `y * (g - sum(y * g))`.

### 3.3 `rms_norm`

```
fn rms_norm[..., D, T](
    x:   Tensor[T, [..., D]],
    g:   Tensor[T, [D]],
    eps: T
) -> Tensor[T, [..., D]]
```

Root-mean-square normalization: one pass over `x` computing
`mean(x^2)`, then `x * g / sqrt(mean(x^2) + eps)`. No mean subtraction (the
defining feature vs `layer_norm`).

**Differentiable.**

### 3.4 `layer_norm`

```
fn layer_norm[..., D, T](
    x:   Tensor[T, [..., D]],
    g:   Tensor[T, [D]],
    b:   Tensor[T, [D]],
    eps: T
) -> Tensor[T, [..., D]]
```

Mean and variance over the last axis, normalize, then affine with `g`
and `b`. One pass.

**Differentiable.**

### 3.5 `rope` — rotary position embedding

```
fn rope[..., S, D, T](
    x:   Tensor[T, [..., S, D]],
    cos: Tensor[T, [S, D/2]],
    sin: Tensor[T, [S, D/2]]
) -> Tensor[T, [..., S, D]]
```

Applies rotary position embedding (RoPE) via paired-coordinate
rotation on the last axis. `D` must be even; pairs are
`(x[..., 2i], x[..., 2i+1])`. For each pair `i` at each position `s`:

```
new[..., 2i]   = x[..., 2i] * cos[s, i] - x[..., 2i+1] * sin[s, i]
new[..., 2i+1] = x[..., 2i] * sin[s, i] + x[..., 2i+1] * cos[s, i]
```

Lowered to a single bandwidth-bound SIMD pass over `x` with `vfmadd`
on interleaved pairs. **Never** materialized as a `D × D` rotation
matrix — that form costs ~`D` extra FLOPs and ~3× the memory traffic.

`cos` and `sin` are typically computed once at startup and reside in
the Vault; they may also be `View`s into a longer precomputed table
sliced by the live sequence length.

**Differentiable.** Backward is the same kernel with `sin` negated.

### 3.6 `embed` — embedding lookup

```
fn embed[V, D, T, ...B](
    vocab: Tensor[T, [V, D]],
    ids:   Tensor[i64, [...B]]
) -> Tensor[T, [...B, D]]
```

Canonical 2-arg embedding lookup. `vocab` is the embedding weight matrix
`[V, D]`; `ids` is a tensor of token indices of any batch shape. Output
shape is `ids.shape ++ [D]`.

Each index is clamped to `[0, V-1]` at runtime; out-of-range indices do
not error but produce implementation-defined rows (clamped access).

**Not differentiable in 0.0.x** (integer indexing on a differentiable
axis is excluded per `AUTODIFF.md §5`). Gradient accumulation for embedding
tables is a reserved feature.

### 3.7 Utility builtins

The following are registered builtins (not stdlib-catalog functions) and
do not carry the §2 fusion guarantee.

| Function | Signature | Description |
| -------- | --------- | ----------- |
| `median(x)` | `Tensor[T, S] -> T` | Returns the median scalar value of a tensor. |
| `sort(x)` | `Tensor[T, S] -> Tensor[T, S]` | Returns a same-shape tensor with elements sorted in ascending order. |
| `gcd(a, b)` | `(i64, i64) -> i64` | Greatest common divisor of two integers. |
| `to_str(x)` | `str` | Converts a value to its string representation. |
| `to_string(x)` | alias of `to_str` | Same conversion under the longer name. |
| `to_bin(x)` | `str` | Converts an integer to its binary-digit string. |
| `to_binary(x)` | alias of `to_bin` | Same conversion under the longer name. |

### 3.8 Reductions and index-reductions

Registered builtins (not §2-fused stdlib functions; no fusion guarantee).
All have interpreter + JIT parity.

**Index reductions** return the *index* of the extreme value along an axis.
The `axis` argument is optional and defaults to `-1` (last axis). The reduced
axis is dropped; when the result collapses to rank-0 a scalar `i64` is returned
— a usable index / token id.

| Function | Signature | Description |
| -------- | --------- | ----------- |
| `argmax(x, axis=-1)` | `(Tensor[T, [...]], i64) -> Tensor[i64, ...]` \| `i64` | Index of the maximum along `axis`. |
| `argmin(x, axis=-1)` | `(Tensor[T, [...]], i64) -> Tensor[i64, ...]` \| `i64` | Index of the minimum along `axis`. |

**Per-axis value reductions** reduce one axis, which is **dropped** from the
result shape. The `axis` argument is **required** (there is no default, unlike
the index reductions above).

| Function | Signature | Description |
| -------- | --------- | ----------- |
| `sum_along(x, axis)` | `(Tensor[T, [...]], i64) -> Tensor[T, ...]` | Sum along `axis`. |
| `mean_along(x, axis)` | `(Tensor[T, [...]], i64) -> Tensor[T, ...]` | Mean along `axis` (divides by the axis length). |
| `max_along(x, axis)` | `(Tensor[T, [...]], i64) -> Tensor[T, ...]` | Maximum along `axis`; errors on an empty axis. |
| `min_along(x, axis)` | `(Tensor[T, [...]], i64) -> Tensor[T, ...]` | Minimum along `axis`; errors on an empty axis. The `min` counterpart to `max_along`. |
| `variance_along(x, axis)` | `(Tensor[T, [...]], i64) -> Tensor[T, ...]` | Population variance (`mean(x²) − mean(x)²`) along `axis`. |

`pull_to_mean_along(x, axis, alpha)` is a **shape-preserving** variance-
minimizing pass (`out = x + alpha · (mean_along(x, axis) − x)`), not a
reduction — the axis is retained. See `examples/sim/`.

> ⚠️ Whole-tensor reductions (`sum`, `mean`, `max`, `argmax`, …) fold over the
> tensor's entire allocated capacity — see the capacity-vs-data callout in §7.

---

## 4. Elementwise activations

`gelu`, `silu`, `tanh` and friends exist in the stdlib but do **not**
carry the normative fusion guarantee of §2. They are fused into
surrounding `\|>` pipelines by the standard fusion pass
(`OPERATORS.md §4`). If you need a hard guarantee, wrap them in
`@fuse`.

`relu` is the unary `\>` operator and is also callable as `relu(x)` —
both are valid. It always fuses.

| Function | Signature | Formula | Differentiable |
| -------- | --------- | ------- | -------------- |
| `gelu` | `fn gelu(x: Tensor[f32, S]) -> Tensor[f32, S]` | `x * Φ(x)` where Φ is standard normal CDF | Yes |
| `silu` | `fn silu(x: Tensor[f32, S]) -> Tensor[f32, S]` | `x * σ(x)` where σ is sigmoid | Yes |
| `tanh` | `fn tanh(x: Tensor[f32, S]) -> Tensor[f32, S]` | `(e^x - e^-x) / (e^x + e^-x)` | Yes |
| `elu` | `fn elu(x: Tensor[f32, S]) -> Tensor[f32, S]` | `x if x > 0 else exp(x) - 1` (α=1) | Yes (interp + JIT, 1st order) |
| `mish` | `fn mish(x: Tensor[f32, S]) -> Tensor[f32, S]` | `x * tanh(softplus(x))` | Yes (interp + JIT, 1st order) |

**Note on `swish`:** `swish` is not a separate function — it is an alias for `silu` (which exists). Use `silu(x)` directly.

---

## 5. What is not in the standard library

The following are explicitly **not** in the standard library and never
will be:

- Optimizers (SGD, Adam, LARS). User code; trivially expressible; the
  stdlib is not a framework.
- Data loaders, samplers, batching utilities. Not the language's job.
- Tokenizers. Not the language's job.
- Conv2d / Conv3d / pooling / image-processing primitives. Not in 0.0.x
  (might be in 0.1 if a credible demand exists).
- Anything domain-specific (vision, audio, RL, graph).

If a model needs one of these, the model includes its own
implementation. demoniC compiles it. That's the deal.

> **Doc-coverage note.** This document specifies the tensor/ML core.
> The reference implementation additionally ships general-purpose
> builtin families this document does not yet cover: lists
> (`list_push`, `list_map`, `list_filter`, `list_reduce`, …), regex,
> HTTP (`http_get`/`http_post`), gzip/zlib, filesystem and path ops,
> date/time, string formatting (`format`), hashing (`hash_fnv`,
> `hash_crc32`), CLI helpers, process execution (`exec_cmd`), linear
> algebra (`solve`, `inv`, `lstsq`), trit-tensor ops, and the seeded
> global RNG helpers (`SPEC.md §3.7`, Random as a value). They are
> implemented and exercised by the example corpus; specifying them here
> is tracked work. The "never will be" list above still governs what the
> *core* promises.

---

## 6. Versioning

Stdlib signatures are part of the spec's stability surface. Removing a
function or changing its signature requires a major version bump.

Adding a new stdlib function follows the same procedure as adding a
directive (`DIRECTIVES.md §5`):

1. SPEC amendment that justifies the slot against §1's three criteria.
2. Tokenizer-safe naming (`TOKENIZER.md §6`).
3. A worked example under `examples/`.
4. Implementation must be a single fused kernel, not a composition of
   existing ops.

The catalog stays small on purpose. Each function in §3 earned its
slot. Future additions must earn theirs.

---

## 7. Python / Rust / NumPy idiom translation

Writers from Python, Rust, and NumPy backgrounds repeatedly reach for
constructs that don't exist in demoniC. This table maps common idioms
to their demoniC equivalents or explains why they're absent.

| Python/Rust/NumPy writes | demoniC equivalent | Notes |
| ------------------------ | ------------------ | ----- |
| `range(0, n)` | `0..n` | Half-open range, usable in `for x in 0..n` |
| `range(n)` | `0..n` | Same; there's no `range()` function |
| `range(a, b, step)` | `for x in 0..(b-a) { let i = a + x * step }` | No step in range literal |
| `[0] * n` or `Vec::new()` | `forge.zeros[T, [N]]` | Fixed-size, Forge-allocated |
| `list.append(x)` / `Vec::push(x)` | `cache <- x` | KV stream append, streaming axis only |
| `np.zeros((M, N))` | `forge.zeros[f32, [M, N]]` | |
| `np.ones((M, N))` | `forge.ones[f32, [M, N]]` | |
| `x * y` (elementwise tensors) | `x .* y` | `*` is scalar multiply — errors on tensors |
| `np.dot(a, b)` | `a @ b` | Matmul for rank ≥ 2 |
| `np.sum(x)` | `sum(x)` | Full reduction — **over capacity, not data; see ⚠️ below** |
| `np.max(x, axis)` | `max_along(x, axis)` | Per-axis max reduction (the reduced axis is dropped), f32, rank ≥ 2 — the `*_along` family alongside `sum_along` / `mean_along` / `variance_along`. Interp + JIT parity. Full-tensor max is `max(x)`. See §3.8. |
| `np.min(x, axis)` | `min_along(x, axis)` | Per-axis min reduction; the `min` counterpart to `max_along`. Interp + JIT parity. Full-tensor min is `min(x)`. |
| `np.argmax(x, axis)` | `argmax(x, axis)` | Index of the max along `axis` (axis defaults to `-1`, then dropped); `argmin` for the min. Collapses to a scalar `i64` when the result is rank-0. See §3.8. |
| `m.T` / `np.transpose(m)` | `m'` | Postfix transpose operator (swaps last two axes) — there is no `transpose()` function |
| `np.diag(m)` (extract) | `diag(m)` | Diagonal of a square 2-D tensor: `[N,N] → [N]`; `trace(m)` sums it |
| `np.trace(m)` | `trace(m)` | Sum of the diagonal |
| `f.view(np.int32)` (bit-reinterpret) | `f32_to_bits(x)` / `f32_from_bits(n)` | IEEE-754 bit pattern ↔ i64 (no arithmetic conversion); raw 32 bits zero-extended into i64 |
| `open(path,'rb').read()` | `read_bytes(path)` | `(Tensor[i64,[N]], err)` — one i64 per byte (0–255), lossless on non-UTF-8 (weights). Feeds `@cast(Model){bytes}` and `f32_from_bits`. Use `read_file` for UTF-8 text. `dmc run` only (not JIT). |
| `json.dumps([1.5, 2.5])` | `json_encode([1.5, 2.5])` | Lists/tensors → JSON arrays, maps → objects |
| `sigmoid(x)` | `sigmoid(x)` | Registered differentiable builtin; equivalent to `1/(1+exp(-x))` |
| `print(x)` | `print(x)` or `print_i64(x)` | `print_i64` / `print_f64` exist for numeric output |
| `x.reshape(M, N)` | `x.reshape[[M, N]]` | Zero-copy if contiguous. The shipped form is the `.reshape[[...]]` method (note the double brackets); `View[T, S]` is a non-owning *type*, not a reshape expression. |
| `x.T` | `x'` | Postfix transpose (swaps last two axes) |
| `x[:, 0]` | `x[.., 0]` | Column slice |

### `sigmoid` — registered differentiable builtin

`sigmoid(x)` is a registered differentiable builtin equivalent to
`1/(1+exp(-x))`. It is callable directly — no user-defined implementation
needed.

### ⚠️ Reductions span the full allocation, not a logical length

`sum`, `mean`, `variance`, `max`, `min`, `softmax`, … reduce over a tensor's
**entire allocated capacity**, including slots you never wrote. There is no
hidden fill-watermark: a tensor sized to a *capacity* (max sequence length, max
batch, ring-buffer size) and only partially filled will fold the zero/garbage
padding into the result — silently. `mean`/`variance` are the dangerous ones
because the **divisor** is the capacity, not your data count:

```dmc
let !t = forge.zeros[f32, [8]]
t[0] = 3.0  t[1] = 3.0  t[2] = 3.0
mean(t)   # 1.125  (9.0 / 8), NOT 3.0 — divides by 8 slots, not 3 values
```

This type-checks, runs, and JITs with **zero diagnostics** — it is the one
footgun that produces a plausible-but-wrong *number* rather than an error.
**Rule: size a tensor to its real data**, or carry the logical
length yourself and divide by it explicitly (`sum(t) / (n as f32)`).

### Syntax reminders

- Tensor literals use **angle brackets for types**: `Tensor[f32, [3, 4]]` — NOT `tensor(...)`.
- The `..` range syntax is **half-open**: `0..5` yields 0, 1, 2, 3, 4.
- `print_i64` and `print_f64` are builtins, but `print(x)` also works for any value.
- **Reductions count the whole allocation** — see the ⚠️ callout above; size tensors to their data.

### `let !` — mutable loop variables (common friction point)

**Every variable that is reassigned inside a loop must be declared `let !`:**

```dmc
# WRONG — x is immutable by default
let x = 0
while x < 10 { x = x + 1 }   # ERROR: cannot assign to immutable binding `x`

# CORRECT — let ! marks x as mutable
let !x = 0
while x < 10 { x = x + 1 }   # OK
```

`let mut x` also works but `let !` is idiomatic demoniC. The compiler now suggests `let !` first in the error message.
