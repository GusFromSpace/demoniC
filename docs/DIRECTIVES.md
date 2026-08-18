# demoniC — Directives

**Companion to:** `docs/SPEC.md §2.6` (Directives), `§4.10` (Directive blocks),
`§6.2` (`@grad` — autodiff), `§7` (Execution model).

A directive is a `@`-prefixed identifier that attaches semantics to a
following declaration, block, or expression. The set is **closed** —
no user-defined directives. Adding one requires a spec revision.

---

## 1. Catalog

| Directive          | Attaches to        | Effect (summary)                                    | Section           | Implementation status |
| ------------------ | ------------------ | ----------------------------------------------------- | ----------------- | --------------------- |
| `@grad`            | `fn`               | emits forward + backward                              | `docs/AUTODIFF.md`     | Fully implemented |
| `@cast(t)`         | block / expr       | runs everything inside in dtype `t`                    | `docs/SPEC.md §7.1` (Mixed-precision scopes (`@cast`))    | Value-preserving float casts (bf16/f16/tf32/f32/f64) implemented; quantized dtypes are f32-backed no-ops (fp8) or JIT-rejected (int) — dequant-during-compute not implemented |
| `@host`            | `match`            | comptime hardware dispatch                             | `docs/SPEC.md §7.2` (Hardware dispatch (`@host`))    | Interpreter only; JIT lowering pending |
| `@deterministic`   | block               | bit-exact reproducibility contract                     | `docs/SPEC.md §7.3` (Determinism contract (`@deterministic`))    | Fully implemented |
| `@recompute(...)`  | block               | activation-budget checkpointing                        | `docs/AUTODIFF.md §3`  | Parse-accepted no-op; not implemented |
| `@inplace`         | stmt                | fail if a write would CoW                              | —  | Parse-accepted; no enforcement |
| `@shard(...)`      | `let` / expr       | marks a tensor for sharding along a named axis (parsed only) | `docs/SPEC.md §2.6` (Directives) | Parsed and type-checked; does not alter code generation in this version |
| `@tp(...)`         | `let` / expr       | marks a tensor-parallel weight or op (parsed only)     | `docs/SPEC.md §2.6` (Directives) | Parsed and type-checked; does not alter code generation in this version |
| `@pp(...)`         | `fn`                | pipeline-parallel function with `stage K:` body        | `docs/SPEC.md §2.6` (Directives) | Body-validated; the interpreter executes stages sequentially, threading `_` between them; JIT lowering pending |
| `@fuse`            | block / expr       | force single-kernel emission                           | `docs/SPEC.md §7.4` (Forced fusion (`@fuse`))    | Fully implemented |
| `@comptime`        | block / `fn`        | evaluates contents; does not yet force comptime folding | `docs/SPEC.md §7.5` (Comptime evaluation (`@comptime`))    | Block form runs as an ordinary block (no folding enforced); `@comptime fn` is inert; the compiler warns on both forms |

**Note:** `@inplace` and `@recompute` are **parse-accepted no-ops** —
the compiler does nothing with them today, and the type checker emits
a warning (`directive @X is not implemented — it is parsed but has no
effect`) so they aren't silent. `@comptime`'s block form evaluates its
contents like an ordinary block without yet enforcing compile-time
folding; a `@comptime fn` declaration is inert and warns the same way.
`@host` is functional in the interpreter: `@host match { .feature =>
... }` performs host-feature dispatch, with JIT lowering pending.

---

## 2. Argument forms

```
@grad
@cast(bf16)
@host
@deterministic
@recompute(budget=4G)
@inplace
@shard(…)
@tp(…)
@pp(…)
@fuse
@comptime
```

All arguments are comptime. Size arguments accept plain integer literals
and the binary-suffix forms `K`/`M`/`G` (= 1024^1/2/3, so `4G` = 4294967296;
`Ki`/`Mi`/`Gi` aliases too).

---

## 3. Stacking

Directives stack from outside in; the **innermost** directive is the
one closest to the wrapped construct:

```
@deterministic @cast(bf16) @fuse {
    q @ k'
}
```

Stacking is meaningful in this order:

1. **Comptime-collapsing** (`@comptime`) — innermost; evaluates first, before any other directive sees the expression.
2. **Fusion-binding** (`@fuse`) — wraps the now-resolved expression as a single kernel.
3. **Hardware-dispatch** (`@host`) — selects per-target lowering for the fused unit.
4. **Semantic-altering** (`@cast`) — middle.
5. **Contract-enforcing** (`@deterministic`, `@recompute`) — outermost.

`@grad @grad` is a **legal** stack — it is the second-order autodiff
form (`docs/SPEC.md §6.2` (`@grad` — autodiff), `docs/AUTODIFF.md §7`). Stacking a third `@grad` is
not implemented.

Illegal stacks (specified as compile-time errors; **not yet
enforced** — the checker currently accepts them):

- `@cast(t1) @cast(t2)` directly nested — inner wins, but explicit nesting is rejected to avoid confusion.
- `@inplace` on anything other than an assignment statement.
- `@shard` on a value whose type cannot accept the sharding annotation.
- `@fuse @fuse` — idempotent, redundant.
- `@fuse` wrapping an expression whose ops cannot be collapsed on the host — fails as `fuse-infeasible`, not a stacking error per se but reported at the same compile stage.
- `@comptime` wrapping an expression containing any non-comptime operand — fails as `comptime-non-static`.

---

## 4. Scope rules

Directives have **lexical scope only**. They do not flow through
function calls. A `@cast(bf16) { foo(x) }` block does *not* re-emit
`foo` in bf16 — the cast applies only to the loads of `foo`'s inputs
and the stores of its return.

This is intentional. The alternative (interprocedural directive flow)
would make every function's emitted code dependent on every callsite,
breaking the monomorphization-by-shape rule.

If you want a function to run in bf16, write it with bf16 in the type:

```
fn mlp_bf16(x: Tensor[bf16, [B, D]]) -> Tensor[bf16, [B, D]] { ... }
```

### 4.1 Exotic-float casts and stdlib builtins

`@cast(t)` where `t` is an **exotic float** — `bf16`, `f16`, `tf32`,
`fp8_e4m3`, `fp8_e5m2` — is a **no-op on numeric values**. These types are
f32-backed in both backends: the interpreter computes
in f32 and retags only the result's nominal dtype, and the JIT ignores the
cast and computes in f32. The directive neither rounds inputs to `t`'s
precision nor rounds intermediate accumulations. There is no
"accumulate-in-bf16" mode.

This fixes the contract for a **generic-`T` stdlib builtin** (`rms_norm`,
`rope`, `attn_gqa`, `softmax`, `silu`, …) invoked under `@cast(t)`:

> A stdlib builtin called inside `@cast(bf16) { … }` executes its internal
> reductions and accumulations in **f32**, not bf16. bf16 appears only as a
> nominal type tag — never as a compute or rounding width.

So `@cast(bf16) { rms_norm(x) }` is numerically identical to `rms_norm(x)`:
the wrapper does not change the answer. If you need genuine bf16 storage or
precision, it comes from the **typed data path** — e.g. bf16 weights loaded
via `load_npz` and upconverted to f32 for compute — not from the
`@cast` directive. Worked example and regression test (interp + JIT parity):
`examples/cast_bf16_boundary.dmc`.

**Integer and `Model` casts are not no-ops.** The scalar form `x as i64`
converts in both backends and is the recommended integer cast. The *block*
form over a tensor, `@cast(i64) { t }`, truncates toward zero in the
interpreter; the JIT has no elementwise-truncation kernel for it and so
**rejects it as unsupported** (`dmc jit` errors rather than silently
computing a different answer). Use `x as i64`, or `dmc run` for the
truncating block form. The overlay casts `@cast(u8) { "text" }` (string →
byte tensor) and `@cast(Model) { bytes }` are value-transforming by design.

---

## 5. Adding a directive

Procedure (binding on future PRs):

1. Open an issue tagged `directive-proposal`.
2. Write the `docs/SPEC.md` amendment.
3. Write the entry in this catalog.
4. Add at least one example under `examples/`.
5. The implementation must be a parser change **and** a JIT lowering;
   parse-only directives are rejected. (Three grandfathered
   exceptions predate this rule: `@comptime`, `@inplace`, `@recompute`
   — see the catalog's status column.)

A directive that only logs, only warns, or only renames an existing
behavior does not earn a slot. The set stays small on purpose.
