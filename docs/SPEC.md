# demoniC — Language Specification

**Version:** 0.0.4-draft
**Status:** Draft. Breaking changes expected on every revision until 0.1.

This document specifies the behavior implemented by the compiler in this
repository. Where an example disagrees with the spec, the spec wins.

---

## 0. Conventions

- Source files use the `.dmc` extension.
- Source is UTF-8. Identifiers may use any Unicode XID character.
- Indentation is not semantic. Whitespace separates tokens; nothing else.
- Statements are terminated by newline or `;`. Both are legal; neither is
  required inside `( )` or `[ ]`.
- A **function signature** may wrap across lines: newlines are insignificant
  inside the parameter list `( )`, and one newline between `)` and `->` is
  also allowed, so the return arrow may begin its own line. The body's `{`
  must follow the signature on the same line.
- Anything starting with `@` is a **directive** (§2.6). Directives are part of
  the language, not user-defined annotations.
- Notation: `*` = zero or more, `+` = one or more, `?` = optional,
  `|` = alternation.

## 1. Design constraints

1. **Execution over legibility.** If a symbol maps to a SIMD instruction, it
   wins over a word: `A @ B` over `matmul(A, B)`.
2. **JIT as a first-class citizen.** The runtime may rewrite machine code at
   function boundaries during execution.
3. **Immutable by default.** All bindings, tensors, and views are immutable
   unless mutation is granted via `mut` or `!`.
4. **No hidden allocation.** Every byte has an arena. Every allocation is a
   pointer bump. No GC, no reference counting.
5. **Stable tokenization.** Source bytes are chosen to tokenize predictably
   under modern byte-pair-encoding tokenizers. demoniC is designed to be
   written and emitted by humans, editors, and language models.

## 2. Lexical structure

### 2.1 Comments

```
# line comment
#{ block comment, may nest #{ ... }# }#
```

### 2.2 Keywords

```
fn      let     mut     match   if      else
for     while   loop    break   continue return
vault   forge   stream  view    shape   dtype   as
true    false   nil
model   stage   self    type    extern  enum
pub     use
```

`vault`, `forge`, and `stream` are both keywords and the names of built-in
arenas. They cannot be shadowed.

### 2.3 Identifiers

```
ident ::= (XID_Start | "_") (XID_Continue | "_")* "!"?
```

Identifiers ending in `!` denote mutating functions (convention). The `!` is
part of the name.

### 2.4 Literals

| Form                                | Type             |
| ----------------------------------- | ---------------- |
| `42`, `0xff`, `0b1010`, `1_000_000` | `i64` (default)  |
| `42i32`, `42u8`, `42i16`            | sized integer    |
| `3.14`, `1e-9`, `.5`                | `f64` (default)  |
| `3.14f64`, `1.0bf16`, `1.0fp8_e4m3` | sized float      |
| `"abc"`, `"line\n"`                 | `str` (UTF-8)    |
| `c"A"`, `c"\n"`, `c"0"`             | `u32` (Unicode scalar value) |
| `b'A'`, `b'\n'`, `b'0'`             | `i64` (byte value 0–255, ASCII) |
| `true`, `false`                     | `bool`           |
| `nil`                               | the unit value   |

Byte literals `b'x'` are disambiguated from `b'` (the postfix transpose of a
tensor named `b`) by one token of lookahead: if the character after `b'` is a
single character or escape sequence followed by a closing `'`, it is a byte
literal; otherwise `b` is an identifier and `'` is the transpose operator.

### 2.5 Operators and punctuation

Recognized as single tokens by the lexer:

```
@   '   \>   \<   .+   .-   .*   ./   .^   .**
+   -   *   /   %   ^   **
==  !=  <   >   <=  >=
&&  ||  !
=   :=  +=  -=  *=  /=  &=  |=  ^=
<<  >>  &   |
..  ..= ::
->  =>  \|>
!   ?   <-   ~
```

- `<-` — **append**, on stream-typed values (§3.6).
- `?` — **propagate**, on `(T, Err)`-returning calls (§4.9).
- `~` — **streaming axis marker** inside a shape literal (§3.6); in expression
  position, prefix **bitwise NOT** on integer scalars.
- `**` is power; `^` is XOR. `>>` is the arithmetic right shift
  (`OPERATORS.md §8a`), not a pipe.

### 2.6 Directives

A directive is a `@`-prefixed identifier followed optionally by `(args)` and
then a function declaration, a block, an expression, or a match. The set is
closed and versioned; there are no user-defined directives. Recognized by the
parser:

```
@grad       @cast(t)       @deterministic   @recompute(…)
@inplace    @host          @shard(…)        @tp(…)
@pp(…)      @fuse          @comptime
```

Semantics-carrying directives are specified in §6.2 (`@grad`) and §7
(`@cast`, `@host`, `@deterministic`, `@fuse`, `@comptime`). `@pp` functions
are body-validated and their stages execute sequentially in the interpreter
(threading `_` between stages), without pipeline parallelism. The remaining
members of the set (`@recompute`, `@inplace`, `@shard`, `@tp`) parse but do
not alter code generation in this version.

## 3. Types

demoniC is statically and structurally typed. Every type is known at JIT
compile time.

### 3.1 Scalar types

```
i8  i16  i32  i64                   # signed integers
u8  u16  u32  u64                   # unsigned integers
int4  int8                          # packed signed 4-bit / 8-bit
f16 f32 f64 bf16 tf32               # floats
fp8_e4m3  fp8_e5m2                  # 8-bit floats (E4M3 and E5M2)
bool
str                                 # UTF-8 byte sequence, immutable
nil                                 # unit type
```

All implicit numeric conversions are forbidden. Casts are explicit:
`x as f32`. Two distinct concrete scalar types are never mutually assignable —
`i64 → i32`, `f64 → f32`, `f64 → i8` all require an `as` cast.

The exotic float types (`bf16`, `f16`, `tf32`, `fp8_e4m3`, `fp8_e5m2`) are
f32-backed: values compute in f32 and carry the narrow type as a tag. `int4`
and `int8` are storage tags with the same convention.

**Untyped numeric literals.** A bare integer literal (`5`) or float literal
(`5.0`) is untyped and adopts the type of its context — the annotation, return
type, parameter, or other operand at its use site:

```
let x: i32 = 5        # 5 adopts i32
fn f() -> f64 { 5.0 } # 5.0 adopts f64
let y = 1.0 .* t      # 1.0 adopts t's element type
```

An integer literal may adopt any integral context but not a float one
(`let y: f32 = 5` is rejected — write `5.0` or `5 as f32`). A float literal
may adopt any float context but not an integral one. When a literal is bound
without a numeric context it defaults to the 64-bit type of its family:
integer → `i64`, float → `f64`. A directly written integer literal whose
magnitude does not fit the narrow integral type it adopts is a compile-time
error (`let x: i8 = 300`). Only syntactic literals (and a leading unary `-`)
are range-checked; literal arithmetic like `200 - 100` is never falsely
flagged.

An **explicit type suffix** on an integer literal (`42u32`, `0xffi16`)
types the literal concretely, exactly like a suffixed float: it binds an
unannotated `let` at the suffix type, conflicts with a different
annotation or parameter type as a normal type error, and is
range-checked against its own width (`300u8` is a compile-time error).
The 64-bit suffixes are exempt from the range check: hex and binary
literals store as 64-bit bit patterns, so any such mask fits.

**`str` operations:** indexing `s[i]` returns the UTF-8 byte at position `i`
as `i64`. Concatenation uses `+`. Integer-to-string conversion: `n as str`.
Length: `len(s)`.

**Map type:** `map_new()` (alias: `map()`) creates a mutable `str → any` hash
map with reference semantics — mutations are visible through all aliases. Key
operations: `map_set(m, key, val)`, `map_get(m, key)`, `map_contains(m, key)`,
`map_del(m, key)`. All operations are in-place; the map pointer is stable.

**Runtime type introspection** (`dmc run`): `typeof(x) -> str` returns the
runtime kind (`"int"`, `"float"`, `"str"`, `"bool"`, `"list"`, `"map"`,
`"nil"`, `"fn"`, `"Tensor"`). Predicates `is_int`, `is_float`, `is_str`,
`is_bool`, `is_list`, `is_map`, `is_nil`, `is_fn`, `is_tensor` each return
`bool`.

**Safe numeric parse:** `is_numeric(s) -> bool` tests whether a string parses
as a number without aborting. `try_to_int(s)` / `try_to_float(s)` return
`(value, Err)` (§3.9): `Err` is `nil` on success or a `str` message on
failure. (Contrast `to_int`/`to_float`, which abort on malformed input.)

**The `any` type (dynamic escape hatch):** `any` is the one type whose value's
kind is not fixed at compile time. It is accepted in any type position and is
compatible with every other type in both directions. This lets a value whose
type varies at runtime cross a function boundary:

```
fn atom(tok: str) -> any {
    if is_numeric(tok) { return to_float(tok) }   # f64 on one path,
    tok                                            # str on another — both ok
}
```

Recover the concrete kind at the use site with the introspection predicates.
`any` is interpreter-only (`dmc run`): an `any`-typed parameter, return, or
model field is rejected by the JIT with a `use dmc run` hint, preserving the
invariant that JIT-compiled code is fully statically typed.

### 3.2 Tensors

```
Tensor[T, [d1, d2, ..., dN]]
```

Shapes may be **static** (`Tensor[f32, [256, 768]]`), **symbolic**
(`Tensor[f32, [B, 768]]` introducing `B`), or **dynamic**
(`Tensor[f32, [?, ?]]`, escape hatch only).

Shape mismatches between static or symbolic dimensions are compile-time
errors.

### 3.3 Views

A `View[T, [...]]` is a stack-allocated descriptor pointing into an arena,
with shape, strides, and an arena tag. `Tensor[T, S]` is a `View` with
contiguous strides in row-major layout. Slicing a `Tensor` yields a `View`;
slicing never copies.

### 3.4 Mutability

```
let !w = forge.zeros[f32, [768, 768]]   # mut binding, mut data
let  w = forge.zeros[f32, [768, 768]]   # immut binding, immut data
let mut x = ...                          # long form of `let !x`, except at a
                                         # model field — see below
```

Binding a mutable copy of an existing value (`let !y = x`) copies by value:
mutating `y` leaves `x` untouched.

**Bindings from model fields.** `let !x = m.field` follows the field's kind:
a **tensor** field binds a live alias — every write through the binding
(element write, whole-binding assignment, compound assignment, stream
append) reaches the field; every other field kind (scalars above all) binds
its current **value**, a snapshot per the copy rule above.

**`!` and `mut` are not interchangeable.** Both spellings make the binding
and its data mutable, and everywhere else in the language they mean the same
thing. They part company at exactly one construct — the field binding above.
`let !x = m.field` takes the alias; `let mut x = m.field` binds the field's
current **value** for every field kind, tensor included. Take the long form
when the point is to snapshot a tensor field.

### 3.5 Functions

```
fn name[shape_vars](arg: Type, ...) -> Ret { body }
```

Shape parameters in `[ ]` are comptime; value parameters in `( )` are
runtime. Functions monomorphize over their shape parameters — every distinct
shape produces a distinct JIT-emitted machine code body.

Functions may reference each other regardless of definition order within a
file. Top-level `let` bindings are evaluated in source order and may not
forward-reference other `let` bindings.

**Entry point.** `fn main()` is optional. When present, it is called after
all top-level `let` bindings are evaluated. When absent, the program
evaluates its top-level items in source order and returns `nil`.

#### 3.5.1 Function types and first-class functions

A function type is written `fn(T1, T2) -> R`. It can appear as a parameter
type, a local annotation, or a return type:

```
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
```

A function literal (anonymous lambda / closure) binds parameters and returns
a value of that type:

```
let double = fn(x: i64) -> i64 { x * 2 }
apply(double, 21)                          # => 42
apply(fn(x: i64) -> i64 { x + 1 }, 41)     # inline
```

A named function used as a value is written bare (no call syntax):

```
fn add(x: i64, y: i64) -> i64 { x + y }
let f = add   # f: fn(i64, i64) -> i64
f(3, 4)       # => 7
```

Calling through a map or opaque `i64` pointer requires an explicit type
annotation so the runtime dispatch knows the arity and calling convention:

```
let !env = map_new()
map_set(env, "+", add)
let f: fn(i64, i64) -> i64 = map_get(env, "+")
f(10, 3)   # => 13
```

The annotation is **required for `dmc jit`** and optional under `dmc run`.
Without it, `map_get` returns a plain `i64`: `dmc --check` still passes and
the interpreter still dispatches the call (the example above prints `13`
unannotated), but the JIT refuses the program with `jit unsupported: unknown
function`. Annotate it, and both backends accept it. Both non-capturing and
capturing closures are supported in `dmc run` and `dmc jit`.

### 3.6 Streaming types (KV caches)

A streaming axis is marked `~` in a shape:

```
KV[T, [B, H, ~, D]]
```

Streaming axes:

- Live in the `stream` arena (or its sub-region inside `forge`).
- Reserve trailing capacity at allocation: `forge.kv[T, S](capacity = N)`.
- Grow by **append**, written `<-` (§4.8). One bump-pointer advance plus one
  hardware copy. Never reallocates.
- May appear at exactly one axis per type.

A `KV[T, S]` is convertible to `View[T, S]` (with the current `~` value
frozen as a concrete dimension) for any read-only operation. The conversion
is free.

### 3.7 Random as a value

```
Rng                  # an opaque type
Rng.seed(seed: u64) -> Rng
```

`Rng` is a linear value: every consumption returns a new `Rng` plus the
sampled output:

```
let (rng, x) = rng.normal[f32, [B, D]]
let (rng, u) = rng.uniform[f32, [B, D]]
let (rng, i) = rng.uniform_int[i32, [B]](low, high)
```

All three draws run in the interpreter (none is JIT-lowered). `uniform_int`
samples the half-open range `[low, high)` and requires `high > low`; the
result tensor is integer-tagged, so element reads round-trip as integers.

The `Rng` value is the canonical API. Seeded global helpers (`rand_seed(n)`,
`rand_float()`, `rand_int(lo, hi)`) also exist as a convenience surface; they
draw from one process-wide generator and offer none of the linear-value
replay guarantees. The same `Rng` call sequence with the same seed produces
bit-identical bytes on the same hardware.

### 3.8 Errors as values

There is one built-in sum type for fallible operations:

```
(T, Err)
```

By convention, `Err` is `nil` on success and a `str`-tagged error value on
failure. Functions returning `(T, Err)` may use `?` to propagate (§4.9). No
exceptions exist anywhere in the language.

### 3.9 Models

```
model Name[shape_params] {
    field1: Type
    field2: Type

    fn method(self, ...) -> Ret { ... }
}
```

A `model` is a struct with:

- **Deterministic field order.** Layout matches source order.
- **An opinionated `forward` slot.** If a `forward` method exists,
  `instance(x, ...)` desugars to `instance.forward(x, ...)`.
- **Fixed-size model arrays.** `[M; N]` is a field/parameter type holding `N`
  instances of model `M`, shape-parameterized forms included
  (`blocks: [Block[D, H]; L]`). Allocated with `forge.uninit[M, [N]]`,
  indexed `blocks[i]`, iterated with `for`. `N` is comptime.
- **Shape parameters, in every position they can be written.** A model's own
  shape parameters are in scope throughout its methods and bind from the
  instance: `Box[2, 3]` gives `H = 2`, `W = 3`. A method may declare shape
  parameters of its own — `fn blit![SH, SW](self, src: Tensor[u32, [SH, SW]])`
  — bound by an explicit bracket between the method name and the call
  (`b.blit![4, 5](src)`, positional or named `b.blit![SH=4, SW=5](src)`) or
  inferred from the arguments (`b.blit!(src)`); a method shape parameter that
  reuses one of the model's names shadows it for that call. A
  shape-parameterized model type is legal as a field type
  (`!surface: Surface[H, W]`) and as a parameter type (`!b: Box[H, W]`): a
  model literal carries the shape arguments it was written with, so
  `Surface[4, 5] { … }` unifies with a field expecting `Surface[H, W]` under
  `H = 4, W = 5`, and an argument's own shape arguments bind the parameter's
  shape names. A literal written *without* a bracket (`Surface { … }`) is not
  silent: its shape arguments are recovered from the fields it was given, so
  `Surface { px: <a 4x5 tensor> }` unifies as `Surface[4, 5]` and a 7x7 one
  does not. An empty argument list is never a wildcard — a bare literal whose
  fields pin *nothing* is rejected in a field slot, naming the field.

  Three caveats, all pre-existing: **reading** such a field yields the declared
  type *unsubstituted* (`o.surf` types as `Inner[H, W]`, not `Inner[4, 5]`), so
  it satisfies a generic consumer but not a concrete expectation — as a tensor
  field already does; **assignment** to a model field is not shape-checked at
  all (`o.surf = Inner[9, 9] { … }` on an `Outer[2, 2]` is accepted); and a
  lowercase shape argument (`Box[h, w]`) checks but can only be
  supplied explicitly at the call, since inference binds uppercase-initial
  names only, for models and tensors alike.
- **No inheritance, no traits, no methods other than those declared in the
  model body.** (Exception: UFCS, §4.11.)

A `model` introduces both a type (`Name[...]`) and a constructor
(`Name { field1: ..., field2: ... }`). Field initialization order must match
declaration order. Models are not heap objects; a `model` value is a
stack-held descriptor over its field arena. For weight I/O use
`vault.load` / `vault.load_npz`.

### 3.10 Raw pointers

```
*i8   *u8   *f32   *f64   ...   *nil
```

A `*T` is an opaque machine pointer to storage holding values of scalar type
`T` (or `nil`, for void pointers). Raw pointers exist only to cross the
foreign-function boundary declared by `extern fn` (§6.7). A `*T` has no
shape or length; it cannot appear as a tensor or model field element type,
cannot be bound outside an `extern fn` call site, and cannot be dereferenced
in source. The JIT materializes `*T` values from demoniC tensors at the call
site; the source program never constructs one.

### 3.11 Ports

```
Port[python]
```

A `Port[L]` is a handle to a foreign runtime `L`. The foreign language never
enters demoniC's type system: values cross the boundary as canonical JSON
strings, through three builtins that return errors as values (§3.8):

```
port_open(lang: str) -> (Port[lang], Err)
port_call(p: Port[L], name: str, payload: str) -> (str, Err)
port_close(p: Port[L]) -> (nil, Err)
```

The `python` process port is implemented: `name` is a dotted
import path (`math.sqrt`, `json.dumps`) or a bare builtin name, the
JSON-decoded `payload` is the call's argument vector — a JSON array of
positional arguments, or an `{args, kwargs}` object — and the result is
re-encoded to canonical JSON. Other runtimes report a `port-open` error. Both
backends lower port calls and drive one shared port registry, so the wire
protocol and the error tags are the same code under `dmc run` and `dmc jit`.

A handle stays opaque in both. It compares only against `nil` — `p == nil` is
how a failed `port_open` is detected — never against another handle or a `str`,
and a `str` is not a handle: handle text is ordinary text, so accepting one
would let a program forge a handle it never opened. It renders, for display
only, as `<opaque port#<id>:<lang>>`. `Port[L]` is writable in a parameter or
return position, so a handle may be passed and returned — and `L` is checked
there: binding a handle to a `Port[L]` position compares `L` against the
runtime the handle was opened for, and a mismatch is a located runtime error
in both backends. A handle may *not* appear in a model field, a tensor element
type, or a Vault constant; the checker rejects the model-field case, so both
backends refuse the same program at the same span. Semantics, the value
boundary, and the error tags are specified in `PORTS.md`.

## 4. Expressions

### 4.1 Literals and identifiers

An integer literal has type `i64`, a float literal `f64`, `true`/`false`
`bool`, string literals `str`, and `nil` has type `nil`, subject to the
context-adoption rules of §3.1.

### 4.2 Tuples and tensor literals

```
(a, b, c)               # tuple, heterogeneous, stack
[1, 2, 3, 4]            # 1-D tensor literal, in nearest arena
[[1,2],[3,4]]           # 2-D tensor literal
```

Tensor literals are for small constants (test vectors, small
initializations). A tensor literal with more than 256 elements is a
**compile-time error** — the count is of leaves through the full inferred
shape, and the diagnostic names the replacement spelling (`TOKENIZER.md
§8a`). For bulk data use the arena constructors:

```
forge.zeros[f32, [768, 768]]    # zero-initialised, in forge arena
forge.ones[f32, [1024]]         # one-filled
forge.uninit[f32, [B, S, D]]    # uninitialized (fast; caller fills)
vault.load[f32, [50257, 768]]("embed.bin")   # load from disk into vault
```

Rule of thumb: if the literal would not fit on one source line, use `forge`
or `vault` instead.

### 4.3 Indexing and slicing

```
A[i]            A[i, j]         A[0:100]
A[0:100:2]      A[:, 3]         A[.., -1]       A[0..]
```

Slicing never copies. Negative indices count from the end. Static
out-of-bounds is a compile-time error; dynamic out-of-bounds is a runtime
panic.

One construct does not end-resolve a negative index. A slice whose start is
a runtime value but whose extent is statically known (`t[i..i+K]` for a
compile-time `K`) treats a negative resolved `i` as dynamic out-of-bounds and
panics, rather than counting from the end: the static extent already fixes
the window's size, so end-resolving the start would silently change which
elements a fixed-size window reads. A runtime scalar index (`t[i]`) and a
runtime slice bound assigned into (`t[a..b] = rhs`) both still end-resolve a
negative value as everywhere else in this section. Both backends agree on
the exception.

### 4.4 Math operators

- `A @ B` — matrix multiply. Both operands must have rank ≥ 2.
- `m'` — transpose (postfix).
- `.+  .-  .*  ./` — broadcasted elementwise arithmetic. Bare `+ - * /` on
  tensors is an error; elementwise tensor ops are always dotted.
- `.>  .<  .>=  .<=` — elementwise comparison, producing a 0.0/1.0 mask.
- `\>(x)` — ReLU (prefix). `\<(x)` — GeLU (prefix), the tanh
  approximation; **not** min-with-zero. For the negative clamp write
  `x .- \>x` (`OPERATORS.md §3`).
- `**` — power on scalars. `^` is XOR, not power.

### 4.5 Pattern matching

`match` is an expression and must be exhaustive against the scrutinee's type.
Patterns: `_`, literal, identifier (binding), `pat if cond` (guard), the
`x @ pat` bind form (matches iff `pat` matches, binding the name to the whole
value), and the `..` rest pattern — standalone it is a catch-all, and inside
a tuple it absorbs the unmatched middle (`(a, ..)`, `(a, .., z)`; at most one
per tuple). Shape patterns (`[2, 3]`) are rejected at check time.

Exhaustiveness is enforced for every scalar scrutinee. Closed sets must be
fully covered or carry a catch-all: `bool` (both `true`/`false`) and `enum`
(every variant). Open scalars (`i64`, `str`, floats, narrow ints) must carry
a catch-all `_` or bare-identifier binding; a `match` on an `i64` with no
catch-all is a compile error.

**Enums.** A closed set of named variants:

```
enum Token { Assign, Eq, Ident }        # variants are i64 ordinals, in order

fn name(t: Token) -> str {
    match t {                           # exhaustive — every variant or `_`
        Token.Assign => "ASSIGN",       # qualified `Enum.Variant`, or…
        Eq           => "EQ",           # …bare (resolved by scrutinee type)
        Token.Ident  => "IDENT",
    }
}
```

An enum value is its variant's `i64` ordinal in declaration order
(`Token.Eq as i64 == 1`). Enums are nominal — distinct from `i64` and from
other enums; an int does not implicitly flow into an enum. In pattern
position a bare variant name is resolved against the scrutinee's enum; a bare
identifier that is not a variant is an ordinary catch-all binding. Enums
lower through the JIT at parity with the interpreter.

**Payload-carrying variants (tagged unions).** A variant may carry positional
data, and a `match` arm binds it:

```
enum Shape { Circle(f32), Rect(f32, f32), Empty }

fn area(s: Shape) -> f32 {
    match s {
        Circle(r)        => 3.14159 * r * r,
        Shape.Rect(w, h) => w * h,
        Empty            => 0.0,
    }
}
let s = Shape.Circle(2.0)               # construction is qualified
```

`as i64` still yields the variant's ordinal. Scalar payloads
(`i32`/`i64`/`f32`/`f64`/`bool`) run at interpreter/JIT parity; other payload
types run under `dmc run`. Recursive payloads
(`enum Expr { Add(Expr, Expr) }`) are not expressible — a variant cannot
embed its own enum by value. Express recursive sum types as a tagged `model`
(a `kind: i64` discriminant plus the union of variant fields) with children
stored by index into a model array.

**Match arms compare literals, not named constants.** A bare identifier in a
match arm is a binding pattern — `match k { TOK_EQ => ... }` matches
everything and binds a new `TOK_EQ`. Discriminate with literal arms
(`match k { 1 => ... }`) or a guard (`x if x == TOK_EQ`).

### 4.6 Pipe / chain

`x \|> f` ≡ `f(x)`. `\|>` is canonical; the bare `|>` is an accepted
alternate spelling the formatter normalizes (`TOKENIZER.md §2–§3`). Chains
of elementwise ops fuse into a single kernel pass.

`>>` is not a spelling of this operator. It is the arithmetic right shift
(`OPERATORS.md §8a`).

### 4.7 Control flow as expressions

`if`, `match`, and block `{ ... }` are expressions; their value is the last
expression in the chosen arm. There is no `?:` ternary — use `if`.

### 4.8 Stream append: `<-`

For any `KV[T, S]`-typed binding `c`:

```
c <- v       # v: View[T, S_inner], where S_inner matches S with the
             #    streaming axis dropped (or set to v's matching dim)
```

Semantics: bump `c`'s `~` cursor by `v`'s extent along the streaming axis,
copy `v`'s bytes into the freshly reserved region, return `nil`. If the
reservation is exhausted, the program exits with a clear
exceeded-declared-capacity error in both backends — no reallocation, ever.
`<-` is the only way to extend a streaming axis.

### 4.9 Error propagation: `?`

Postfix on any expression of type `(T, Err)`:

```
let bytes = read_file(path)?
```

Sugar for:

```
let (bytes, e) = read_file(path)
if e != nil { return (default[T], e) }
```

`default[T]` is a compiler intrinsic producing the zero value of `T`
(0 / 0.0 / false / nil / zero-tensor), used only in this desugaring. `?` is
legal only inside a function whose return type is also `(_, Err)`; elsewhere
it is a compile-time error.

### 4.10 Directive blocks

A directive may scope an expression:

```
let y = @cast(bf16) { mlp(x) }
let l = @deterministic { train_step(...) }
let z = @fuse { x .* g .+ b }
```

The block is an expression evaluating to its last expression. Directives may
stack; stacking order is innermost-first when semantically meaningful.

### 4.11 UFCS (method-call sugar)

`x.f(args)` desugars post-parse to `f(x, args)` when `f` names a known free
function or builtin and the receiver is not a model. Not UFCS: genuine model
methods, the `@grad` call-surface methods (`.fwd`, `.grad`, `.fwd_bwd`,
`.fwd_bwd_bwd`), and `str` built-in methods (`.split`, `.trim`, …), which
dispatch as builtins. If `f` is not a known free function or builtin, the
compiler reports an unknown-function error on the desugared form.

`nil` has no methods, and neither backend answers one with a value. Under
`dmc run` a nil receiver is a located runtime error — ``cannot call method
`f` on nil`` — for every `f`; `dmc jit` raises the same error at the same
span for the methods it lowers (today `.starts_with` on a `str`), and refuses
the program at compile time, naming the method, for one it does not. They
differ in when they speak, not in whether. This matters most for the
`(_, Err)` convention: `e.starts_with("…")` on an `Err` that is `nil` because
nothing went wrong must say so, not answer "no".

## 5. Statements

```
let pat = expr                       # binding (immut)
let !pat = expr                      # binding (mut)
let mut pat = expr                   # binding (mut, long form; §3.4 — not
                                     # a synonym at a model-field binding)

expr = expr                          # assignment (LHS must be mut)
expr += expr                         # compound assignment

if cond { ... } else { ... }
match expr { arm, arm, ... }
for ident in iter { ... }
while cond { ... }
loop { ... }
break    continue    return expr
```

There are no exceptions. `panic` aborts the process without unwinding.
Recoverable errors use `(T, Err)` and `?`.

### 5.1 `:=` vs `=` in nested blocks

Inside `forge`, `if`, `for`, `while`, or `@cast` blocks:

- `:=` introduces a new shadow binding scoped to that block. The outer
  binding is not updated.
- `=` updates the nearest enclosing `let !` / `let mut` binding.

The checker rejects a `:=` shadow that immediately goes out of scope without
being read — the most common form of this mistake is a hard error.

## 6. Items

### 6.1 Functions

```
fn forward[B, S](x: Tensor[f32, [B, S, 768]]) -> Tensor[f32, [B, S, 768]] {
    x \|> ln \|> attn \|> ln \|> mlp
}
```

### 6.2 `@grad` — autodiff

```
@grad fn loss(!W: Tensor[f32, [D, H]], x: Tensor[f32, [B, D]]) -> f32 { ... }
```

A `@grad fn` declares both a forward and a backward. Callers see five call
forms:

- `loss(...)` — forward only, returns the scalar.
- `loss.fwd(...)` — explicit forward-only alias.
- `loss.grad(...)` — backward only, returns `Grads` without the loss.
- `loss.fwd_bwd(...)` — returns `(f32, Grads)`: the loss and a gradient
  struct with one field per `!` parameter in declaration order. Destructure
  as `let (loss, g) = loss.fwd_bwd(...)`, then read `g.W`, `g.b`, ….
- `loss.fwd_bwd_bwd(...)` — second order; requires `@grad @grad fn`.
  Returns `(value, second_grads)`.

Gradients flow to `!` (mut) parameters. A `@grad fn` with no `!` parameter is
a compile-time error. Conditions of `if`/`match` and loop trip counts inside
the differentiated body are evaluated concretely and are non-differentiable;
only the executed path is recorded (define-by-run). The scalar-math builtins
`sqrt`, `exp`, `log`, `sin`, `cos`, `tan` on a traced scalar stay on the
tape with their elementary derivatives (first order, interpreter). Other
scalar builtins, indexed reads `x[i]`, and `argmax`/`argmin` leave the
gradient graph.

The interpreter supports the full `@grad` semantics, including shape-generic
functions. The JIT lowers a concrete-shape subset (matmul, elementwise ops,
activations, fused `softmax`/`rms_norm`/`layer_norm`/`rope`/`attn`).

### 6.3 Arenas as scopes

```
vault { ... }      forge { ... }      stream { ... }
```

Inside such a block, all implicit allocations target the named arena. Outside
any block the default is `forge`. `forge` is a bump-allocated scratch arena;
`forge.reset()` frees everything allocated in it since program start (the
per-training-step idiom). `vault` holds long-lived data (weights);
`stream` holds streaming (KV) data.

### 6.4 Models

```
model Block[D, H] {
    ln: Tensor[f32, [D]]
    qkv: Tensor[f32, [D, 3*D]]
    out: Tensor[f32, [D, D]]

    fn forward(self, x: Tensor[f32, [B, S, D]]) -> Tensor[f32, [B, S, D]] {
        let qkv = rms_norm(x, self.ln) @ self.qkv
        let (q, k, v) = qkv.split[3, axis=-1]   # n-tuple of equal pieces
        x .+ softmax((q @ k') .* (1.0/sqrt(D as f32)), -1) @ v @ self.out
    }
}
```

See §3.9 for model semantics.

### 6.5 Type aliases

```
type Hidden = Tensor[f32, [768]]
type Batch[B] = Tensor[f32, [B, 768]]
```

No newtypes. No traits. No generics beyond shape parameters and element type.

### 6.6 Modules and imports

```
use "path/to/file.dmc"           # unqualified — all pub items enter scope
use "path/to/file.dmc" as alias  # qualified — items accessed as alias.name
```

- Paths are relative to the importing file.
- Imports resolve statically at compile time; circular imports are a
  compile-time error (files compile leaf-first over the import DAG).
- Only `pub`-annotated items (`pub fn`, `pub model`, `pub type`) are
  exported; everything else is file-private.

### 6.7 Foreign function declarations (`extern fn`)

```
extern fn cblas_sgemm(order: i32, transA: i32, transB: i32,
                      M: i32, N: i32, K: i32,
                      alpha: f32, A: *f32, lda: i32,
                      B: *f32, ldb: i32, beta: f32,
                      C: *f32, ldc: i32) -> nil
```

An `extern fn` declares a function whose body is provided by a foreign ABI;
the signature is the declaration. `extern fn name(...)` uses the platform C
ABI — the JIT looks up `name` in the process's dynamic symbol table.
`extern "cuda"` / `extern "hip"` name device ABIs.

Rules (compile-time): no body, no shape parameters, no `self`, no mutability
markers. Parameter and return types are restricted to scalar types, raw
pointers `*T`, and `nil` — tensors reach a foreign call only as a data
pointer plus separately passed scalar shape arguments. An `extern fn` is
always visible to importers without `pub` (and `pub` on one is an error). An
`extern fn` may not be called from `@comptime`, `@grad fn`, `@fuse`, or
`@deterministic`; autodiff treats any `extern fn` call as a hard
non-differentiable barrier. Inside `@comptime` the ban is a consequence of
§7.5's total ban on calls, and is reported as `comptime-non-static` naming
the `extern fn`.

Both are enforced at check time, under their own diagnostic tags: a
parameter or return type the boundary does not admit is `extern-boundary`,
and a call from a forbidden context is `extern-context`. An implementation may
additionally refuse, as a *backend* limitation rather than a language one,
an admissible type whose machine width in that backend is not the width the
annotation names — `str`, and the widths a backend emulates.

**Passing a tensor.** A tensor argument at a `*T` parameter is the
materialization §3.10 describes. The value passed is the address of element
zero of the tensor's data in the Forge, row-major, with no header and no
padding ahead of it, at the tensor's **storage** width — the width the
element type has in the running backend, not necessarily the width the `*T`
annotation names. The pointee type is not checked against the tensor's
element type; `*nil` accepts any tensor.

The pointer is valid for the **dynamic extent of the call only**. A callee
that retains it is wrong, and `forge.reset()` invalidates every pointer
previously materialized from a Forge tensor. Access past the tensor's
element count is out of bounds and unchecked: the callee is told the extents
by scalar arguments the caller wrote by hand, and nothing verifies they
agree with the tensor.

The boundary is unsafe by design: the JIT performs no shape, alignment,
lifetime, or aliasing check across the call. A miscompiled call segfaults;
the language offers no mitigation. `extern` is the marking, and there is no
second one. A symbol the process cannot resolve is a JIT refusal
(`jit-extern`) naming the declaration, not a crash inside the code
generator.

The interpreter parses and type-checks `extern fn` declarations but rejects
calls to them at runtime — it has no pointer representation for a tensor to
hand across. Foreign calls require `dmc jit`.

## 7. Execution model

The `dmc` binary provides both backends:

| Command             | Behavior                                            |
| ------------------- | --------------------------------------------------- |
| `dmc run f.dmc`     | tree-walking interpreter — full semantics           |
| `dmc jit f.dmc`     | Cranelift JIT — statically typed subset, fast       |
| `dmc f.dmc`         | full pipeline: lex, parse, check, run               |
| `dmc --check f.dmc` | type/shape check only, no execution                 |
| `dmc test path`     | run every zero-arg `fn test_*() -> bool`            |
| `dmc test --jit`    | additionally run JIT-eligible tests on both backends and compare |
| `dmc fmt f.dmc`     | canonical pretty-print                              |
| `dmc selftest`      | generate random well-typed programs, run both backends, diff |

The interpreter is the reference semantics. The JIT compiles the statically
typed subset; constructs outside it (e.g. `any`, shape-generic `@grad`,
`Rng` draws) report a clear error directing to `dmc run`.

JIT stages: parse → type/shape-check the whole program (shape errors report
as a batch; nothing runs until the program checks end to end) → lower to a
typed IR → monomorphize over shape parameters at first call → emit machine
code through Cranelift.

### 7.1 Mixed-precision scopes (`@cast`)

`@cast(t) { expr }` rewrites the IR for `expr` so every op runs in `t`.
Loads cast in at block entry; stores cast out at block exit; the cost is one
vectorized cast pass per boundary, never per-op. Nested `@cast` blocks
override the outer one.

Only value-preserving float casts carry semantics: `t` ∈ {`bf16`, `f16`,
`tf32`, `f32`, `f64`}. The exotic floats are f32-backed in both backends —
computed in f32 and retagged, not rounded to `t`'s storage precision. The
fp8 tags are likewise f32-backed no-ops. A tensor `@cast` to an integer type
(`int4`, `int8`, `i32`, …) is rejected by the JIT as value-lossy; the
interpreter truncates each element toward zero and retags the result as an
integer tensor — it does not run the enclosed ops in integer arithmetic.

### 7.2 Hardware dispatch (`@host`)

```
@host match {
    .avx512 => kernel_a(),
    .avx2   => kernel_b(),
    .neon   => kernel_c(),
}
```

The arm chosen at JIT time is the only one whose opcodes are emitted. The
host-feature set is interrogated once at startup.

A feature name is a property of the host the program runs on, not
necessarily of its CPU. Alongside the ISA flags — `avx512`, `avx2`, `neon`
and the rest of the per-target set — the name **`accelerate`** means a BLAS
`cblas_sgemm` is resolvable in this process's dynamic symbol table, and
therefore that a large f32 matmul can reach the platform's matrix unit. It
is present or absent for the same reason an ISA flag is: a program's arms
are chosen by what this machine can actually do. Adding a feature name is
additive and does not change the meaning of any existing one.

### 7.2b Float accumulation contract

`f32` **reductions** (`sum`, `mean`, `variance`, the `*_along` family)
accumulate at f32 width in index order, rounding after every add. `sum(t)` is
therefore the same number as the loop it stands for:

```
let !s = 0.0f32
for i in 0..len(t) { s = s + t[i] }
```

`f32` **matmul** (`@`) accumulates over the inner dimension ascending and
**contracts**: each multiply-add is one fused operation with a single
rounding. It is consequently *not* the same number as a hand-written
`s = s + a[i]*b[i]` loop, which rounds twice per step — matmul is a
primitive with a defined accumulation, not shorthand for that loop.
Contraction is the more accurate of the two, and it is host-independent:
where the hardware has no FMA unit the backends fall back to a correctly
rounded software implementation rather than splitting the operation.

Both rules hold on **both backends and every kernel**. In particular the
answer does not depend on which kernel a shape selects.

This is a per-op contract, not a claim that demoniC's float output matches
another implementation's. `docs/NUMERICS.md` states what is and is not
promised across versions, including the edges this contract does not cover.

### 7.3 Determinism contract (`@deterministic`)

Inside `@deterministic { expr }`: given identical inputs and identical `Rng`
state, the bytes of every output are bit-exact on repeated execution on the
same host. Reduction orders are fixed, kernel selection ignores
non-deterministic fast paths, and all RNG paths consume `Rng` linearly.
Outside the block, faster non-deterministic kernels may be selected.

"Ignores non-deterministic fast paths" is enforced against the one such path
that exists: the `--blas` GEMM offload (`docs/NUMERICS.md §2.2`) is
unreachable inside the block, whatever the flag says.

This is the strongest cross-run reproducibility promise the language makes.
`docs/NUMERICS.md` states how it composes with the rest of the numerics
contract, and what it does *not* promise — in particular that it fixes
demoniC's own accumulation order and says nothing about agreement with an
external reference implementation.

### 7.4 Forced fusion (`@fuse`)

`@fuse { expr }` instructs the JIT to emit `expr` as a single kernel with no
materialized intermediates. If the contract cannot be honored on the host,
compilation fails with a `fuse-infeasible` diagnostic citing the operations
that could not be collapsed.

### 7.5 Comptime evaluation (`@comptime`)

Shape parameters in `[ ]` and integer literals are implicitly comptime: a
function monomorphizes over its shape parameters, so shape arithmetic folds per
instantiation.

`@comptime { expr }` is the explicit form for derived values. It evaluates
`expr` at compile time with the reference interpreter, before either backend
lowers the program, and the result is a compile-time constant in the
surrounding monomorphization. A block whose body cannot be fully evaluated at
compile time is a compile-time error (`comptime-non-static`) citing the first
non-comptime operand encountered.

**The fold set is integers, booleans, and shape arithmetic.** This is
normative, not an implementation status. A block whose body evaluates to an
integer or a boolean is replaced by that literal before check completes; a
block reading a shape parameter is folded per monomorphization by whichever
backend compiles the call. Everything else is `comptime-non-static`: floats
of every width, tensors, strings, and every other value the language has.

The float cut is deliberate. Folding a float would decide whether a
compile-time float must equal a run-time one, and nothing `@comptime` exists
for needs that decided. Widening the fold set is a spec amendment, not an
implementation change.

The body admits literals, names the block binds, shape parameters of the
enclosing `fn` or `model`, arithmetic, comparison, boolean logic, `if`/`else`,
`while`/`loop`/`for` over integer ranges, and assignment to a binding the
block itself made. It admits **no calls** — not to a user `fn`, a builtin, a
port, or an `extern fn`. That total ban is what makes the effect restrictions
of `docs/PORTS.md §5` and §6.7's `extern fn` rule hold without
interprocedural analysis: nothing reachable from a legal body can perform an
effect.

Because the body may loop, compile-time evaluation is bounded. Exhausting the
step budget is a compile-time error (`comptime-budget`) naming the block; a
non-terminating `@comptime` fails the compile rather than hanging it.

`@comptime` attaches to a block. On a `fn`, an item, or a bare statement it is
rejected, with a hint to the supported spelling. `dmc fmt` prints `@comptime`
as written: folding is a compilation step, not a source rewrite.

## 8. Errors and diagnostics

Every diagnostic reports the `file:line:col` of the offending token, a
one-line summary, the inferred shapes of any tensors involved, and the
directive stack if non-empty. There is no error-code system.

Categories: lexical, syntactic, shape, dtype, mutability, aliasing, arena,
directive, autodiff.
