# demoniC — Operator Catalog

**Companion to:** `SPEC.md §4.4` (Math operators).

Every operator demoniC recognizes, sorted by precedence (tightest first).
Each entry lists the intended hardware lowering. Lowerings target x86_64
AVX2 baseline with AVX-512 and aarch64 NEON fast paths.

---

## 1. Precedence table

Higher binds tighter. Same row = same precedence; associativity in the
third column.

| Prec | Operators                                    | Assoc |
| ---: | -------------------------------------------- | ----- |
|   19 | `A'` (postfix transpose), `A[...]` (index), `A?` (propagate), `as` (type cast) | left |
|   18 | `\>` `\<` (unary activation)                 | right |
|   17 | unary `-`, unary `!`, unary `*`, unary `~`   | right |
|   16 | `.^` `.**` `**`                              | right |
|   15 | `@`                                          | left  |
|   14 | `.*` `./` `*` `/` `%`                        | left  |
|   13 | `.+` `.-` `+` `-`                            | left  |
|   12 | `<<` `>>` (bitwise shifts)                   | left  |
|   11 | `&` (bitwise AND)                            | left  |
|   10 | `^` (bitwise XOR)                            | left  |
|    9 | `\|` (bitwise OR)                            | left  |
|    8 | `..` `..=`                                   | none  |
|    7 | `==` `!=`                                    | none  |
|    6 | `<` `<=` `>` `>=`                            | none  |
|    5 | `&&`                                         | left  |
|    4 | `\|\|`                                       | left  |
|    3 | `\|>` (pipe / chain)                         | left  |
|    2 | `<-` (stream append)                         | right |
|    1 | `=` `:=` `+=` `-=` `*=` `/=` `&=` `\|=` `^=` | right |

Parentheses override everything. No user-defined operator overloading;
the table above is the universe.

**Note on `==`/`!=` vs `<` `<=` `>` `>=`:** equality binds
**tighter** than the ordering comparisons here — the reverse of C/Python/Rust.
So `a < b == c` parses as `a < (b == c)`. This matches the grammar
(`compare = equality {…}`) and the parser, which have agreed since the
codebase's first commit; the table previously listed the conventional
ordering and was the stale artifact. Resolved as implementation-wins:
changing the parser now would silently break programs already written
against the current precedence. Mixed comparison/equality chains are rare;
parenthesize when in doubt.

---

## 2. Math operators

### 2.1 Matrix multiply: `A @ B`

- **Type:** `Tensor[T, [..., M, K]] @ Tensor[T, [..., K, N]] -> Tensor[T, [..., M, N]]`
- **Lowering:** tiled SGEMM/BGEMM emitted by the JIT, parameterized on `(M, K, N, T)`. Small shapes (M,N,K ≤ 16) get a fully unrolled AVX-512 microkernel. Larger shapes fall through to a blocked algorithm with prefetching and a packing pass.
- **Notes:** broadcasts on leading batch dimensions. Reduces only the innermost two axes. Differentiable.

### 2.2 Transpose: `A'`

- **Type:** `Tensor[T, [..., M, N]]' -> View[T, [..., N, M]]`
- **Lowering:** **does not move bytes**. Returns a new `View` with the last two strides swapped. Materialization only happens if the result feeds an op that cannot consume non-contiguous strides.

### 2.3 Broadcasted elementwise: `.+ .- .* ./ .^ .**`

- **Type:** numpy/julia broadcasting.
- **Lowering:** fused SIMD loop. Chained broadcast ops inside a `\|>` pipeline fuse into a single kernel — no intermediate Forge allocation.

**On integer elements these are integer arithmetic at the element width**
(`docs/SPEC.md §3.1` (Scalar types)). A tensor's element type carries its
width the way a scalar's does, so `a .+ b` on two `Tensor[i32, [N]]` wraps at
32 bits, `a ./ b` is integer division (`7 ./ 2` is 3, and division by zero is
0), and unsigned elements wrap unsigned. `MIN ./ -1` is the same runtime
error the scalar `/` raises, naming the width. The JIT does not lower integer
elementwise ops at all today — it refuses them (`elementwise ops are
f32-only`) rather than computing a second answer.

### 2.4 Scalar math: `+ - * / % **`

Native instructions. No surprises.

**Note on `^`:** `^` is **bitwise XOR** (prec 10, left-associative), not a
power operator. Use `**` for scalar exponentiation (right-associative),
`.^` / `.**` for broadcasted elementwise power.

---

## 3. Activations

| Op    | Math               | Lowering                                    | Diff |
| ----- | ------------------ | ------------------------------------------- | ---- |
| `\>x` | `max(x, 0)` (ReLU) | `vmaxps zmm, zmm, zero` per lane            | yes  |
| `\<x` | GeLU `x·Φ(x)`      | tanh-approximation kernel per lane          | yes  |

`\<x` is **GeLU** (Gaussian Error Linear Unit), computed with the
standard tanh approximation — identical to the stdlib `gelu(x)`. It is
**not** min-with-zero; the negative-clamp idiom is `x .- \>x`
(`x - max(x,0) = min(x,0)`).

Both fuse into adjacent elementwise ops in a `\|>` chain. SiLU and
softmax live in the standard library — they require more than one SIMD
instruction. The fused stdlib primitives (`attn`, `softmax`,
`rms_norm`, `layer_norm`) carry a normative single-kernel guarantee
documented in `STDLIB.md`.

---

## 4. Pipe / chain: `\|>`

```
y = x \|> ln \|> linear[d, 4d] \|> \> \|> linear[4d, d]
```

`x \|> f` ≡ `f(x)`. Chained elementwise ops fuse into a single kernel
pass. `\|>` is canonical; the bare `|>` is the same operator and the same
node — `dmcfmt` normalizes it to `\|>` (`TOKENIZER.md §2–§3`).

`>>` is not a pipe spelling. It is the arithmetic right shift (§8a), so
`x >> f` is a shift on an integer, never a pipe.

---

## 5. Stream append: `<-`

```
cache <- v       # cache: KV[T, [..., ~, ...]]
```

- **Type:** `KV[T, S] <- View[T, S_inner] -> nil`. `S_inner` matches `S`
  with the streaming axis dropped, or set to `v`'s extent along that
  axis.
- **Lowering:** one bump-pointer advance on the stream cursor, one
  hardware `rep movsq` (x86) or NEON-aligned copy. **Never** reallocates.
- **Semantics:** see `SPEC.md §4.8` (Stream append: `<-`). Exhausting the
  capacity panics.
- **Differentiable:** yes; the backward is a slice + copy of the
  gradient region.

---

## 6. Error propagation: `?`

```
let bytes = read_file(path)?
```

- **Type:** `(T, Err) -> T` inside a function whose return is `(_, Err)`.
- **Lowering:** trivial: read the `Err` field, branch on non-`nil`,
  return the pair early.
- **Differentiable:** no. Use of `?` inside a `@grad fn` is a compile
  error.

---

## 7. Type cast: `as`

```
let x_int = x as i32
let y_f32 = y as f32
```

- **Syntax:** `expr as Type` — postfix, non-associative at the highest
  precedence level alongside `'`, `[...]`, and `?`.
- **Type:** converts `expr` to the target `Type`. Numeric conversions
  follow C truncation/extension rules for integer widths; float-to-int
  truncates toward zero. Out-of-range values are implementation-defined.
- **Differentiable:** no. `as` is a compile-time type reinterpretation;
  the backward is not defined.
- **Note:** `^` is NOT a power operator — it is bitwise XOR (see §8a).
  Power is `**`, `.^`, or `.**`.

---

## 8a. Bitwise operators: `&`, `|`, `^`, `<<`, `>>`

Operate on integer scalar types only (`i8`–`i64`, `u8`–`u64`). Scalar only —
neither shift broadcasts over a tensor.

| Op   | Math           | Notes                                  |
| ---- | -------------- | -------------------------------------- |
| `~`  | bitwise NOT    | prefix; integer types only             |
| `<<` | left shift     | fills zeros on the right               |
| `>>` | right shift    | **arithmetic**: the sign bit is copied in on the left, so `-8 >> 1` is -4 and `-1 >> n` is -1. It floors; `-7 >> 1` is -4 while `-7 / 2` is -3. |
| `&`  | bitwise AND    |                                        |
| `^`  | bitwise XOR    | **not** power; power is `**` / `.^`    |
| `\|` | bitwise OR     |                                        |

The **shift amount must be in `0..=w-1`**, where `w` is the width of the left
operand's type: `0..=63` for an `i64` — the type an unsuffixed integer literal
adopts — and `0..=31` for an `i32`. Outside that range the JIT rejects a
literal count at compile time and traps a computed one, naming the range it
judged that count against; never a silent mod-`w` wrap. The interpreter raises
a clean runtime error against the same range.

**A shift is performed at the left operand's width.** An `i32` shift is a
32-bit shift and keeps the low 32 bits, exactly as `i32` `+` and `*` already
wrap:

```dmc
let a: i32 = 1
let n: i32 = 31
let x = a << n        # -2147483648 — i32::MIN, on both backends
```

Both backends answer at the operand's width, so a narrow shift is a parity
case and not a backend split. `i32` and `i64` are the only integer widths the
JIT compiles; the narrower kinds are refused outright, never silently widened.

Compound assignment forms: `&=`, `|=`, `^=`. There is no `<<=` or `>>=`
compound-assignment form.

---

## 9. Range: `..`, `..=`

- `a..b` — half-open, `[a, b)`.
- `a..=b` — inclusive, `[a, b]`.
- Strides come from a third colon-separated position: `A[0:100:2]` (start,
  stop, step). Note this uses `:` separators, not `..` — `A[0..100:2]` is a
  different, invalid form (a range expression as the start bound).

---

## 10. Mutation: `=`, `:=`, `+=`, …

- `=` — assignment. LHS must be a `mut` binding or a mutable view.
- `:=` — re-bind (shadow). Useful in loops.
- `+= -= *= /=` — compound arithmetic. Subject to copy-on-write semantics
  when the target has aliases.
- `&= |= ^=` — compound bitwise. Integer types only.

No walrus, no `++`, no `--`.

---

## 11. Logical: `&& || !`

Short-circuit. `!` is prefix-only when used as logical-not. Postfix on
a binding (`let !x = ...`) means "mut" (`SPEC.md §3.4`, Mutability). Lexer
disambiguates by position; the parser never backtracks.

---

## 12. Function arrow, match arrow, pipeline stage

- `->` — function return type: `fn f() -> T`.
- `=>` — match arm: `pat => expr`.
- `stage K:` — pipeline stage marker inside `@pp` functions.

Punctuation, not operators. Listed here for completeness.

---

## 13. Tokenizer notes

`\>`, `\<`, `\|>`, `<-`, `>>`, `.+`, `.-`, `.*`, `./`, `..`, `..=`
have canonical spacing requirements documented in `TOKENIZER.md §2`.
The lexer accepts no-space variants; the formatter normalizes.

---

## 14. What is deliberately absent

| Not an operator | Why                                                       |
| --------------- | --------------------------------------------------------- |
| `??`            | `nil` is not a hole in `T`; `?` is the only error sugar   |
| `=>` in expr    | Lambdas are `fn(x) { ... }`. Explicit.                    |
| `.` for method  | No user-defined methods outside `model` `fn` slots; but `x.f(args)` desugars to `f(x, args)` via UFCS (`SPEC.md §4.11`, UFCS (method-call sugar)), and strings/`@grad` fns expose built-in method surfaces. |
| `::`            | Unused; modules ship via `use ... as alias` + dot access (`SPEC.md §6.6`, Modules and imports). |
| `++` `--`       | Use `+= 1`.                                               |
| `*` on tensors  | Scalar multiply — gives a type error on tensors. Use `.*` |
| `>>=` `<<=`     | No compound-assignment form for either shift. Write `x = x >> n`. |

---

## 15. Tensor operator quick-reference

A cheat sheet for the operators writers most often look up.

### Elementwise (dot-ops)

| You want              | Operator | Example              |
| --------------------- | -------- | -------------------- |
| Add two tensors       | `.+`     | `a .+ b`             |
| Subtract tensors      | `.-`     | `a .- b`             |
| Multiply elementwise  | `.*`     | `a .* b`             |
| Divide elementwise    | `./`     | `a ./ b`             |
| Elementwise power     | `.^`     | `a .^ 2.0`           |
| Compare elements      | `.< .>`  | `a .< b` (0/1 mask)  |

**Do not use `*` between tensors** — `*` is scalar multiply and will error on tensor operands. The compiler emits: `Add/Mul on tensors — did you mean the dotted form?`

### Activations (unary prefix)

| You want          | Operator | Equivalent          | Note                    |
| ----------------- | -------- | ------------------- | ----------------------- |
| ReLU              | `\>`     | `max(x, 0)`         | Differentiable          |
| GeLU              | `\<` or `gelu(x)` | `x·Φ(x)` (tanh approx) | Differentiable |
| Negative clamp    | `x .- \>x` | `min(x, 0)`       | Idiom — no dedicated operator |
| Sigmoid           | `sigmoid(x)` | stdlib builtin | S-shaped [0,1] output   |

**`\<` is GeLU, NOT min-with-zero.** For the negative clamp use the
idiom `x .- \>x` (i.e., `x - max(x,0) = min(x,0)`).

### Pipeline / composition

| You want                | Operator | Example                      |
| ----------------------- | -------- | ---------------------------- |
| Pipe value into fn      | `\|>`    | `x \|> softmax`              |
| Matmul                  | `@`      | `q @ k'`                     |
| Transpose               | `'`      | `k'` (postfix)               |

**`>>` is the right shift, not a pipe** — it is the arithmetic right shift (§8a), so `x >> 2` means what it looks like. Piping into a *value* rather than a callable (`x \|> 2`) is a check-time error with a hint; for an elementwise stage, pipe into a `_`-placeholder: `x \|> _ .+ y`.
