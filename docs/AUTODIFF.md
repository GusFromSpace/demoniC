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
- Each field has its parameter's shape: a tensor parameter yields a
  gradient tensor, a scalar `f32` parameter a scalar. A parameter the
  loss does not depend on gets zeros — that *is* the answer, not a
  placeholder.
- Then one field per captured `mut` binding the body differentiates
  (§2.1), named for the binding, in the order the fn's **own module**
  declares them — after the parameter fields.

`Grads` is **Forge-allocated** — its backing memory lives in the Forge
arena and remains valid until the next `forge.reset()` call. Copy any
fields you need into a longer-lived arena (typically the Vault) before
calling `forge.reset()`.

---

## 2. Which inputs get gradients

| Source                       | Has gradient?                                    |
| ----------------------------- | ------------------------------------------------ |
| `mut` tensor parameter (`!W: Tensor[...]`) | yes                                |
| `mut` scalar parameter (`!a: f32`) | yes, interpreter only; the JIT `@grad` subset requires tensors |
| immut parameter              | no                                               |
| captured `mut` binding       | yes, when the body reads it directly — §2.1      |
| captured immut binding       | no                                               |
| `Rng` value                  | no (stochastic ops have no gradient)             |

A `@grad fn` that captures **no** mut bindings and has **no** mut
parameters is a compile-time error — there is nothing for the
backward to produce.

### 2.1 Captured `mut` bindings

A module-level `let !x` (or `let mut x`) read inside a `@grad fn` body
is a **captured mut binding**, and it is differentiated exactly like a
`!` parameter: it enters the tape as an input and its adjoint comes
back in `Grads` under the binding's own name.

```
let !bias = [0.5, 1.0, 2.0]           # captured
let !gain = 2.0                       # captured scalar

@grad fn loss(!W: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
    let y = W .* x .+ bias
    sum(y .* y) * gain
}

let (l, g) = loss.fwd_bwd(W, x)
// g.W, g.bias and g.gain — parameter first, then the captures
```

The rules:

- **What is differentiated** is the binding's value **at call entry**.
  A body that reassigns the capture (`bias = bias .+ d`) still gets
  `∂L/∂bias` at the value the call started from — the reassignment is
  part of the traced computation, not a new input. (The forward really
  runs, so the reassignment is also a real side effect on the binding.)
- **Float only.** A captured `mut` of float scalar or float-tensor type
  gets a gradient; a captured `mut` integer, bool or string is a
  compile-time error rather than a silently absent field. A captured
  `mut` **scalar** traces, exactly as a scalar `!` parameter now does —
  the two are on the same footing, no asymmetry to remember.
- **Direct reads only.** The tape records what the differentiated body
  itself evaluates. Three ways a capture's path to the loss leaves the
  tape — the first two are compile-time errors, the third is not:
  - From inside a **closure literal**: the tape does not enter closure
    bodies. Compile-time error.
  - From inside a **fn or model method the body calls**: a call runs
    concretely and contributes no nodes, so the capture's gradient
    would omit that path. Compile-time error, naming the callee. The
    call graph is followed transitively, and a method is followed like
    any other callee.
  - **Passing the capture as an argument** into a call the tape does
    not trace (`sum(w .* w) + scale(cap)`). This one is **not** caught
    and **not** an error: the argument's contribution is simply absent
    from the backward, so `∂L/∂cap` comes back as zeros of the right
    shape. Nothing in the type system distinguishes that from an
    honest `∂L/∂cap = 0`, so read the value it returns with that in
    mind. **This is not capture-specific** — a `!` *parameter* routed
    through the same untraced call loses the same term
    (`sum(w) + scale(w)` gives `g.w = [1,1,1]`, not `1 + 2w`); it is
    the general "the tape does not trace calls" limitation, and
    closing it is fused-backward work (roadmap §3), not a capture fix.

  In all three cases the remedy is the same: read the capture in the
  differentiated body, inlining what the callee did with it, or pass it
  as a `!` parameter of a `@grad fn` that *is* traced.
- **Shadowing.** A `!` parameter (or a body-local `let`) of the same
  name shadows the module binding; the read is then not a capture, and
  no `Grads` field is produced for the shadowed binding. The local also
  keeps its **own** type: a body-local `let counter = 2.0f32` is fine
  even where the module's unread `counter` is an `i64`.
- **One module.** A capture is resolved in the `@grad fn`'s **own**
  module, and to the module binding itself — never to a same-named local
  in whatever frame happened to call the `@grad fn`. An imported
  `pub let !x` is not a capture (it is not in scope under its bare name);
  reach it through its module alias, or take it as a `!` parameter.
- **Define-by-run** (§6.1) applies as it does to parameters: a capture
  read only on the branch that did *not* execute contributes nothing,
  so its gradient is a zero of its own shape — the correct `∂L/∂x = 0`,
  not a missing field.
- A **captured immut** binding stays a constant on the tape and never
  becomes a `Grads` field.
- **Second order** (`@grad @grad` + `fwd_bwd_bwd`, §7) covers captures
  too: they get second-order fields alongside the `!` params. The
  second-order reduction is seeded from the first `!` parameter, or —
  when the fn has none — from the first captured binding.

> **Implementation status.** Implemented in the **interpreter**
> (`dmc run`), first and second order (`fwd_bwd`, `grad`,
> `fwd_bwd_bwd`), replacing the earlier "specified, not yet
> implemented — `Grads` comes back empty" state. The JIT is unaffected:
> a module-level non-const `let` is already unsupported at lowering, so
> a program with a captured mut runs on the interpreter and the two
> backends cannot disagree. Gated by
> `examples/gradcheck.dmc::test_gradcheck_captured_{tensor,scalar}`
> (central finite differences taken by perturbing the captured binding
> itself, plus a wrong-gradient meta-test). The escapes that made a
> capture's gradient wrong without saying so — a model method reading it,
> a same-named binding in another module or in a caller's frame, a
> body-local shadow — each carry a test of their own.

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
> `layer_norm`, `rope`, `attn`/`attn_gqa`, `variance`, `max`/`min`,
> `max_along`/`min_along`, the scalar-math builtins, element reads, and
> comparison masks — see `examples/gradcheck.dmc` and the compiler's
> `grad_*_gradchecks` unit tests, which check every rule against a central
> finite difference of the same program's forward pass. The **JIT** autodiff
> subset covers matmul,
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
> back to the interpreter. The last four rows of the table below —
> `max_along`/`min_along`, comparison masks, element reads, and the scalar-math
> builtins — are **interpreter-only** and **first order only**; `@grad @grad`
> through any of them errors cleanly rather than returning a wrong number, and
> the JIT reports `unsupported` and falls back to `dmc run`.

| Op          | Forward `y = f(x)`                             | VJP `∂L/∂x = ?`                                     |
| ----------- | ----------------------------------------------- | ---------------------------------------------------- |
| `softmax`   | `y = exp(x - max(x)) / sum(exp(x - max(x)))`   | `y * (g - sum(y * g))` along the same axis          |
| `attn`      | `softmax(q @ k' / sqrt(D)) @ v`                | Per (batch, query head), with `P` the saved softmax weights: `dV += Pᵀ·g`, `dP = g·Vᵀ`, `dS = P ∘ (dP − rowsum(dP ∘ P))/√D`, `dQ = dS·K`, `dK += dSᵀ·Q`. `attn_gqa` sums `dK`/`dV` over each KV head's query group; masked positions have `P = 0`, so the mask gets no gradient |
| `rms_norm`  | `x / rms(x) * g`                               | Fused kernel matching `docs/STDLIB.md §3.3` (`rms_norm`)                |
| `layer_norm`| `(x - μ) / σ * g + b`                          | Standard mean+var backward; grads for `x`, gain, bias |
| `rope`      | rotate `(x[2i], x[2i+1])` by `(cos, sin)`      | Same kernel with `sin` negated (`docs/STDLIB.md §3.5`, `rope` — rotary position embedding)   |
| `variance`  | `(1/N) Σ (x - μ)²`                             | `g · (2/N)(x - μ)`                                   |
| `max`/`min` | global reduction to the extreme element        | subgradient: `g` to the extreme element, `0` elsewhere (ties → first) |
| `max_along`/`min_along` | per-axis reduction to each lane's extreme element (axis dropped) | subgradient, per lane: that lane's `g` to its extreme element, `0` elsewhere (ties → first) — the axis form of `max`/`min` |
| comparison mask (`.<` `.<=` `.>` `.>=`, and the scalar comparisons) | `0`/`1` indicator | **stop-gradient** — the mask is locally constant, so `0` flows into both operands. This is what makes a masked select `(a .< b) .* a .+ (a .>= b) .* b` differentiate: the mask multiplies in as a constant and the product rule routes the whole cotangent to whichever operand it selected |
| element read `x[i]` / `x[i, j]` | one element of a traced tensor (index fixed at trace time) | scatter: `g` into slot `[i, j]` of a zero tensor shaped like `x`, `0` elsewhere |
| `sqrt` `exp` `log` `sin` `cos` `tan` | elementary function of a **traced scalar** | `g · f′(x)`, IEEE at the domain edges (`sqrt′(0) = log′(0) = +∞`) |

For `attn`/`attn_gqa`, the current backward **materializes the per-head
softmax weights**: the `@grad` forward saves each head's `[S, S]` `P` matrix
(a `[B, H_q, S, S]` buffer) and the backward replays it. That violates the
FUSION contract's no-materialization aspiration but not the semantic one — a
fully fused FlashAttention-style backward (recomputing `P` tile by tile)
remains future work under the `@recompute` umbrella.

**Row-wise softmax has no separate spelling.** `softmax(x, axis)` *is* the
axis-wise form — the `axis` argument is part of the signature
(`docs/STDLIB.md §3.2`, `softmax`) and its VJP applies along whichever axis
was given, so a per-item soft assignment over the `K` axis of an `[N, K]`
matrix is `softmax(d, 1)`. There is no `softmax_along`. The differentiable
**soft-min** over an axis follows from it:
`sum_along(softmax(0.0 .- d, 1) .* d, 1)` — the relaxation of
`min_along(d, 1)`, and the differentiable choice when the hard reduction's
subgradient (all the mass on one element) is too coarse.

For all other ops not listed here, the VJP is built by composing primitive
rules. If an op is non-differentiable, it errors at compile time (see §5
below).

---

## 5. Non-differentiable operations

The following are non-differentiable inside a `@grad fn`. They are
evaluated concretely (define-by-run) and contribute no gradient — the
result is treated as a constant:

- Comparisons — scalar (`<`, `<=`, `>`, `>=`, `==`, `!=`) and dotted
  (`.<`, `.<=`, `.>`, `.>=`). They are **stop-gradient, not an error**:
  the `0`/`1` result is locally constant, so it enters the graph as a
  constant and `0` flows into both operands. That is what makes §6.1's
  data-dependent branches work *and* what makes a comparison-masked
  select `(a .< b) .* a .+ (a .>= b) .* b` differentiate — the
  cotangent reaches whichever operand the mask selected.
- `argmax`, `argmin`.
- Integer indexing where the index depends on a differentiable value.
  (A **fixed** index does trace — see below.)
- A **partial** index into a rank > 1 tensor (`x[0]` on an `[N, M]`,
  which yields a sub-tensor rather than an element) leaves the graph.
  Reduce instead (`sum` / `mean` / the `*_along` family), or address a
  single element with a full index.
- `variance_along` and `pull_to_mean_along` have no VJP yet; the rest of
  the `*_along` family (`sum_along`, `mean_along`, `max_along`,
  `min_along`) does.
- The remaining scalar-math builtins — `abs`, `atan`, `atan2`, `hypot`,
  `floor`, `ceil`, `round`, … — still leave the graph.
- Any op the backend cannot differentiate through raises an
  unsupported-in-`@grad` diagnostic. (A general `nondiff` IR marker is
  planned but not yet a distinct flag; enforcement is best-effort.)

**What *does* trace that once did not** (interpreter, first order —
`@grad @grad` through any of these errors cleanly):

- `sqrt`, `exp`, `log`, `sin`, `cos`, `tan` **on a traced scalar**, with
  their elementary derivatives. `sqrt(sum(w .* w))` — the Euclidean norm
  every SDF primitive ends in — differentiates to `w / ‖w‖`.
- **Scalar `!` parameters.** `@grad fn f(!a: f32) -> f32 { a*a*a }`
  differentiates and `Grads` carries a scalar `g.a`. (The JIT `@grad`
  subset still requires tensor `!` parameters; such a function runs on
  the interpreter.)
- **Element reads with a fixed index**, `x[i]` and `x[i, j]`, whose
  gradient scatters back to that slot. Component-wise vector math —
  `sqrt(w[0]*w[0] + w[1]*w[1])` — is now written the natural way; the
  outer-product broadcast (`r @ ones_p`) is no longer needed.
- **Comparison masks**, as described above.
- **`max_along` / `min_along`**, with the per-lane subgradient.
- **Captured `mut` bindings** — tensor *and* scalar — read directly in
  the body, which come back as their own `Grads` fields (§2.1).

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
