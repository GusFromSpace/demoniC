# demoniC — Autodiff

**Companion to:** `docs/SPEC.md §6.2` (`@grad` — autodiff).

This document defines the semantics of `@grad`. It is normative.

---

## 1. Surface

```
@grad fn f(!W: Tensor[f32, [D, H]], x: Tensor[f32, [B, D]]) -> f32 {
    let h    = \>(x @ W)
    sum(h .* h)
}
```

`@grad fn f` produces five callable forms:

| Call                   | Returns               | Behavior                                        |
| ---------------------- | --------------------- | ----------------------------------------------- |
| `f(W, x)`              | `f32`                 | forward only; no backward graph retained        |
| `f.fwd(W, x)`          | `f32`                 | alias of the forward call                       |
| `f.grad(W, x)`         | `Grads`               | backward only; gradients without the loss       |
| `f.fwd_bwd(W, x)`      | `(f32, Grads)`        | forward + backward; `Grads` holds all gradients |
| `f.fwd_bwd_bwd(W, x)`  | `(f32, Grads)`        | second order — loss + second-order grads; requires a `@grad @grad fn` (see §7) |

`fwd_bwd` returns a 2-tuple: the loss scalar first, then a `Grads`
struct holding one gradient field per `!` parameter **in declaration
order**:

```
let (loss, g) = f.fwd_bwd(W, x)
```

Access gradients by their parameter names through the struct:

```
@grad fn step(!W1: Tensor[f32, [D, H]], !W2: Tensor[f32, [H, K]], x: ...) -> f32 { ... }

let (loss, g) = step.fwd_bwd(W1, W2, x)
// g.W1 and g.W2 are the gradient tensors
```

### The `Grads` struct

`Grads` is a **compiler-generated struct** produced for each `@grad fn`
declaration. Its fields are:

- One field per `!`-prefixed parameter, **in declaration order**.
- Field names match the parameter names exactly: a `!W` parameter
  yields a `g.W` field; a `!b` parameter yields a `g.b` field.
- Example: `@grad fn f(!W: Tensor[f32, [D, H]], !b: Tensor[f32, [H]], x: ...) -> f32`
  produces a `Grads` with fields `g.W` and `g.b`.

`Grads` is **Forge-allocated** — its backing memory lives in the Forge
arena and remains valid until the next `forge.reset()` call. Copy any
fields you need into a longer-lived arena (typically the Vault) before
calling `forge.reset()`.

---

## 2. Which inputs get gradients

| Source                       | Has gradient?                                    |
| ----------------------------- | ------------------------------------------------ |
| `mut` parameter (`!W: ...`)  | yes                                              |
| immut parameter              | no                                               |
| captured `mut` binding       | specified, not yet implemented: `Grads` comes back empty |
| captured immut binding       | no                                               |
| `Rng` value                  | no (stochastic ops have no gradient)             |

A `@grad fn` that captures **no** mut bindings and has **no** mut
parameters is a compile-time error — there is nothing for the
backward to produce.

---

## 3. Activation lifetime

The backward needs the forward's activations. demoniC handles this
without a retained tape, and reclaims them **automatically**:

1. A `@grad fn` call takes a Forge snapshot at entry.
2. The forward writes activations into the Forge as usual.
3. The backward consumes them in reverse.
4. On return, the gradients are **compacted down to the entry
   snapshot** and everything else the call allocated — every
   activation — is reclaimed in one cursor rewind. The `Grads`
   fields are the sole survivors: they occupy valid arena memory
   just above the entry watermark, so they are never dangling.

Gradient fields remain valid until the **caller** rewinds the arena
past them (its own `forge.restore(mark)` to an earlier mark). The
usual training-step idiom needs no ceremony:

```
let (loss, g) = f.fwd_bwd(W, x)
W = W .- g.W .* lr        # g.W is valid arena memory; read it freely
```

Per step, the arena retains only the gradients and the updated
weights (parameter-sized), not the activations (typically orders of
magnitude larger). Residual per-step growth is therefore bounded by
parameter bytes; true zero-growth steps would need an in-place weight
update or a copy-below-watermark primitive, neither of which exists
yet.

The forward-only entry (`f(...)` / `f.fwd(...)`) returns a scalar in
a register, so it reclaims **everything** it allocated.

> **Implementation status.** Automatic per-call lifetime shipped for
> the JIT: entry snapshot + gradient compaction on return, for
> `fwd_bwd`, `fwd_bwd_bwd`, and the forward-only entry — including
> nested (FOMAML-style) gradient calls, whose inner gradients survive
> as intermediates of the outer call. The interpreter satisfies the
> same observable contract by heap semantics. The earlier-designed
> `AUTODIFF-LIFETIME` diagnostic is unnecessary under compaction —
> gradients can no longer dangle. Gated by `examples/grad_lifetime.dmc`
> (both backends) and the `jit_grad_*` lifetime unit tests
> (arena-growth bounds).

### 3.1 Checkpointing (`@recompute`)

```
@grad fn f(...) -> f32 {
    @recompute(budget=2Gi) {          # 2 GiB (or budget=2147483648)
        let h = expensive(x)
    }
    use(h)
}
```

Inside `@recompute(budget=N)`, the JIT chooses which intermediates to
materialize for the backward and which to recompute, subject to the
budget `N` of activation bytes. The choice is comptime; no runtime
heuristic. The budget is a byte count of activation memory to retain,
given as a plain integer literal (`budget=4096`) or with a binary size
suffix `K`/`M`/`G` (= 1024^1/2/3; `Ki`/`Mi`/`Gi` aliases), e.g. `budget=4G`.

If the unrecomputable activations alone exceed `N`, compile-time error.

> **Implementation status.** `@recompute` is **not yet implemented** —
> the parser accepts the directive but it is currently a no-op (no
> materialize/recompute choice is made).

---

## 4. stdlib VJPs

The standard library operations that appear most frequently inside `@grad fn`
have VJP rules defined here (normative). They do not correspond to separate
backward calls.

> **Backend status.** The VJPs below are implemented and
> gradient-checked in the **interpreter** (`dmc run`) for `softmax`, `rms_norm`,
> `layer_norm`, `rope`, `attn`/`attn_gqa`, `variance`, and `max`/`min` — see
> `examples/gradcheck.dmc`. The **JIT** autodiff subset covers matmul,
> elementwise ops, and the elementwise activations `relu` / `tanh` / `gelu` /
> `sigmoid` / `silu` / `elu` / `mish` (first order; `silu`/`elu`/`mish` second
> order falls back to the interpreter), plus the fused ops **`softmax`**
> (last axis, the attention case), **`rms_norm`** (last axis, VJP to both `x`
> and `gain`), **`layer_norm`** (last axis, VJP to all three of `x`, `gain`,
> `bias`), **`rope`** (last two axes, VJP to `x`; `cos`/`sin` are read-only
> tables), and **`attn`/`attn_gqa`** (VJP to all three of `q`, `k`, `v`; GQA
> accumulates `dK`/`dV` across each KV head's query group) on rank ≥ 2 static
> tensors — first order. These run at interp/JIT parity
> (`examples/attn_grad_jit.dmc`, `rope_grad_jit.dmc`, `softmax_grad_jit.dmc`, …).
> KV-cache (`KV[...]`) attention operands and non-bool masks stay outside the
> JIT `@grad` subset (use `dmc run`); second order for every fused op falls
> back to the interpreter.

| Op          | Forward `y = f(x)`                             | VJP `∂L/∂x = ?`                                     |
| ----------- | ----------------------------------------------- | ---------------------------------------------------- |
| `softmax`   | `y = exp(x - max(x)) / sum(exp(x - max(x)))`   | `y * (g - sum(y * g))` along the same axis          |
| `attn`      | `softmax(q @ k' / sqrt(D)) @ v`                | Per (batch, query head), with `P` the saved softmax weights: `dV += Pᵀ·g`, `dP = g·Vᵀ`, `dS = P ∘ (dP − rowsum(dP ∘ P))/√D`, `dQ = dS·K`, `dK += dSᵀ·Q`. `attn_gqa` sums `dK`/`dV` over each KV head's query group; masked positions have `P = 0`, so the mask gets no gradient |
| `rms_norm`  | `x / rms(x) * g`                               | Fused kernel matching `docs/STDLIB.md §3.3` (`rms_norm`)                |
| `layer_norm`| `(x - μ) / σ * g + b`                          | Standard mean+var backward; grads for `x`, gain, bias |
| `rope`      | rotate `(x[2i], x[2i+1])` by `(cos, sin)`      | Same kernel with `sin` negated (`docs/STDLIB.md §3.5`, `rope` — rotary position embedding)   |
| `variance`  | `(1/N) Σ (x - μ)²`                             | `g · (2/N)(x - μ)`                                   |
| `max`/`min` | global reduction to the extreme element        | subgradient: `g` to the extreme element, `0` elsewhere (ties → first) |

For `attn`/`attn_gqa`, the current backward **materializes the per-head
softmax weights**: the `@grad` forward saves each head's `[S, S]` `P` matrix
(a `[B, H_q, S, S]` buffer) and the backward replays it. That violates the
FUSION contract's no-materialization aspiration but not the semantic one — a
fully fused FlashAttention-style backward (recomputing `P` tile by tile)
remains future work under the `@recompute` umbrella.

For all other ops not listed here, the VJP is built by composing primitive
rules. If an op is non-differentiable, it errors at compile time (see §5
below).

---

## 5. Non-differentiable operations

The following are non-differentiable inside a `@grad fn`. They are
evaluated concretely (define-by-run) and contribute no gradient — the
result is treated as a constant:

- Comparisons (`<`, `<=`, `>`, `>=`, `==`, `!=`) — this is what makes
  §6.1's data-dependent branches work.
- `argmax`, `argmin`.
- Integer indexing where the index depends on a differentiable value.
- **Most scalar-math builtins now trace**: `sqrt`, `exp`, `log`,
  `sin`, `cos`, and `tan` on a traced scalar stay on the tape with their
  elementary derivatives — `sqrt(sum(w .* w))` (the Euclidean norm every
  SDF primitive ends in) differentiates to `w / ‖w‖`. Interpreter,
  first order only: `@grad @grad` through them errors. The *other*
  scalar builtins (`abs`, `atan`, `atan2`, `hypot`, `floor`, …) still
  leave the graph, as do scalar `!` parameters and indexed reads
  `x[i]`.
- Any op the backend cannot differentiate through raises an
  unsupported-in-`@grad` diagnostic. (A general `nondiff` IR marker is
  planned but not yet a distinct flag; enforcement is best-effort.)

`\>` (ReLU) is differentiable; its derivative is `(x > 0) as T`,
emitted as a fused kernel.

Pattern matching on shapes is fine — shapes are comptime constants
inside a single monomorphization.

---

## 6. Composition rules

- `@grad` on `@grad fn`: the second-order form (§7). Stacking beyond
  two (`@grad @grad @grad`) is not implemented.
- `@grad fn` called from a `@grad fn`: legal; both backwards compose.
- `@grad fn` called from a non-`@grad fn`: legal; forward only.
- `@grad fn` inside `@cast(t) { ... }`: legal; the backward also runs
  in `t`, including its accumulators (this is the standard mixed-
  precision recipe). Note: `.fwd_bwd` returns gradients in the
  original parameter's **declared type**, not the cast type. An
  explicit cast is required if you want to accumulate gradients in
  the lower-precision type.
- `@grad fn` inside `@deterministic { ... }`: legal; the backward
  inherits the determinism contract.
- `@grad fn` containing `@host match { ... }`: each arm has its own
  backward — see the status note below.

---

### 6.1 Control flow in a `@grad fn` body

Control flow inside a differentiated body traces by **define-by-run**: the
condition (or `match` scrutinee, or loop trip count) is **non-differentiable**
— it selects *which* computation happens — and the gradient flows through the
path that actually executed. This is the same model as eager frameworks.

**Interpreter (reference).** Full support: `if` / `else` / `else if` and
`match` as value expressions, and `while` / `for` loops with accumulator
reassignment (`acc = …`, `acc += -= *= /=`). The forward pass runs concretely,
so a loop is effectively unrolled onto the tape and a branch records only the
taken side. `break` / `continue` / `return` and the unbounded `loop` are not
wired (they bail with a diagnostic rather than miscomputing a gradient).

**JIT.** The backward is emitted **ahead of time** as a static tape, so only
control flow the compiler can resolve at compile time is supported:

| Construct | Interp | JIT |
| --- | --- | --- |
| `if` / `match` on a **runtime** value | ✅ | ✖ (falls back to interp) |
| `for k in LO..HI`, `LO`/`HI` compile-time constant (literal / shape param) | ✅ | ✅ unrolled |
| `while`, runtime loop bound | ✅ | ✖ (falls back to interp) |
| accumulator reassignment (`acc += …`) | ✅ | ✅ |

Runtime-data-dependent control flow is **not lowerable** in an AOT reverse-mode
tape without conditional adjoint nodes and runtime activation stacking — a tape
redesign, not a missing case. When the JIT can't lower a `@grad fn`, it reports
`unsupported` and the caller runs the interpreter (the reference), so results
never diverge.

Worked, gradient-checked examples: `examples/grad_control_flow.dmc`
(interpreter, runtime conditions) and `examples/grad_control_flow_jit.dmc`
(compile-time control flow, interp + JIT parity).

Composition *between* functions — a `@grad fn` calling another — works
independently of the above.

---

## 7. What this is not

- Not a tape **for first order** — the backward is emitted statically per
  shape specialization. Second order (`@grad @grad`) builds a tape of the
  first backward in order to differentiate through it.
- Not source-to-source AD in the user's view. The backward is in IR.
- Higher-order is supported **to second order**: stack `@grad @grad fn`
  and call `f.fwd_bwd_bwd(...)` (interpreter and JIT). Third order and
  beyond are not implemented.
- Not optional once written. `@grad fn` always emits both forms. Pay
  the codegen cost or remove the directive.
