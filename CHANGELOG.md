# Changelog

This file tracks releases of the public demoniC repository. Releases are
periodic snapshots of ongoing development, so each entry batches multiple
changes rather than corresponding to a single commit.

## 2026-09-13 — compiler: per-tenant arena budgets, memory-safety fixes, located diagnostics

### Added

- **Per-tenant arena budgets.** A `Budget` is a host-lifetime handle on a
  region of Forge. `forge.root` names the process-wide budget that `--forge`
  sizes; `forge.carve(parent, bytes)` takes a child out of a parent's free
  capacity; `forge.reap(b)` releases a budget's region and credits its whole
  extent — spent and unspent alike — back to the meter it was carved from;
  `forge.enter(b) { … }` pushes `b` as the entered budget for a lexical
  extent and pops it on every exit edge, including `return`, `break`,
  `continue` and `?`, reaping nothing; and a trailing `in b` on a single
  constructor (`forge.zeros[f32, [N]] in b`) overrides the entered budget for
  that constructor only. All of it lowers on **both** backends. The refusals
  are named: `budget-carve` (a carve past the parent's free capacity) and
  `budget-reap` (a reap with a live child) are runtime refusals, because
  `--forge` is a runtime flag and no static parent capacity exists at check
  time; `budget-arena` is the check-time class for an allocation whose bytes
  no budget can own — `forge.kv`/`stream.kv` inside a budgeted extent, and
  `vault.<ctor>[…] in b`. The reap message conjugates identically on both
  backends. The complement is what makes a tenant's weights work: a
  `vault.*` constructor with **no** `in b`, written inside
  `forge.enter(b) { … }`, charges the root and therefore survives
  `forge.reap(b)`. A program that never carves runs on the root budget and
  behaves exactly as it did before. Specified in `docs/MEMORY.md §3.2` and
  `docs/SPEC.md §6.3` (Arenas as scopes); the acceptance program is
  `examples/tenant_budgets.dmc`, whose every figure is a refusal boundary
  bisected on both backends, with the refused half of each section pinned in
  `tools/jit_probes.py` and the two-sided bracket in
  `compiler/tests/budgets_acceptance.rs`.
- **`tools/rss_gate.py` — a peak-resident-memory gate**, and it now runs in
  public CI. Every other gate in this repository pins bytes *charged* or
  program *output*, so an allocator change that touches every page of a chunk
  while charging the same number passes all of them; this one measures the
  peak resident set of three tiny runs against generous ceilings. It caught
  exactly that: an intermediate page-aligned chunk allocator eagerly touched
  all 512 MiB of its default chunk, taking an unflagged `dmc jit` run from
  ~10 MB to ~547 MB while every existing gate stayed green.
- **A large tranche of JIT lowering.** Methods on models lower as free
  functions, one symbol per instantiation for shape-generic models, and
  `m(x)` reaches `m.forward(x)` through the same rewrite. Multi-file programs
  are merged into one unit before lowering, so imports resolve the way
  `--check` resolves them instead of failing as a check error. Arena blocks
  lower as lexical scopes, and a block in statement position lowers for
  effect. Also lowered: `?` propagation and the `(T, Err)` nil idiom; `!`
  mutating parameters outside `@grad fn`; `u8`/`u32`/`u64` tensor element
  types; the math constants; whole-field scalar stores on model bindings;
  `write_file`, `read_file`, `append_file` and `file_exists` as host calls;
  `map_get`/`map_set`/`len` over maps; a `View` parameter binding its shape
  from a `Tensor` or `KV` argument; and top-level type aliases, which are
  transparent rather than refused.
- `dmc test --fail-empty` exits 1 on a suite that ran zero tests, so a
  filter that matches nothing is a failure rather than a silent pass.
- New examples for the newly specified behavior: `tenant_budgets.dmc`,
  `u64_semantics.dmc`, `u64_unsigned_order.dmc`, `mut_params.dmc`,
  `mut_param_type_conventions.dmc`, `cast_model_overlay_byte_order.dmc`, and
  `generic_model_shape_spellings.dmc`.
- `docs/STDLIB.md` gains `len`'s signature and checked domain, a section on
  the unsigned scalar builtins (`abs`, `min`, `max`, `clamp` read a `u64`
  operand unsigned), and a File I/O section covering the four file builtins
  on both backends.

### Changed

- **`u64` is read unsigned wherever the answer depends on signedness.** Both
  backends hold an integer in 64 bits, so a `u64` carries the same
  two's-complement pattern an `i64` does — but `/`, `%`, `>>` and the
  ordering comparisons now re-interpret that pattern as unsigned, and so does
  printing: `0u64 - 1u64` divides, shifts, compares and prints as
  18446744073709551615, not as -1. `+ - * << & | ^` need no
  re-interpretation, which is why they already agreed. `as i64` remains a
  reinterpretation. All three literal spellings above `i64::MAX` are accepted
  with the `u64` suffix — decimal, hex and binary — and `-1u64` is a checker
  error rather than a wrap.
- **Every fixed-width integer kind now compiles in the JIT**, and integer
  elementwise tensor ops lower there at every element width that is a whole
  number of bytes: the integer kernel is the scalar op run per element, so it
  is the same code and the same answers. What the JIT still refuses rather
  than answering differently: `.^` on integer elements, integer tensor
  comparisons, a broadcast between differently-shaped integer tensors, and
  any op on the sub-byte `int4` element.
- **The `@cast(Model)` overlay's byte layout is normative** (`docs/SPEC.md
  §3.10`, Models): successive byte offsets in field order, no padding, no
  alignment, multi-byte scalars little-endian, fixed-shape tensor fields
  row-major. The admissible field domain is enforced at `--check` and names
  the offending field.
- **`!` parameter conventions are normative per type** (`docs/SPEC.md
  §3.4.1`, Mutating parameters): a tensor parameter is copied in and written back through the
  argument's place, so it may be rebound; a model or `KV` parameter *is* the
  caller's instance and must not be rebound; every other type is a
  compile-time error. A `!` argument must be a place, and a `!`-param
  function is not usable as an `fn(...) -> R` value. Exclusivity is not
  adopted. Alongside it, `let mut x = m.field` snapshots a **tensor** field
  where `let !x = m.field` aliases it — a model-typed field aliases under
  both spellings, and a model instance is never copied by either.
- **A plain `let` no longer reaches a `!` tensor or view parameter.** A `!`
  parameter writes through its argument, and a plain `let` binds immutable
  data (`docs/SPEC.md §3.4`, Mutability), so passing one is now a
  compile-time error that names the binding and the fix: *passing `x` to a
  `!` parameter writes through it; `x` needs `let !x`*. The fix is one word
  at the binding — `let !x = …`, or the long form `let mut x = …`. It reaches
  a tensor field through a plain `let` model too, so `let m = M { … }` then
  `bump(m.w)` is refused for the same reason `m.w[0] = 42.0` already was.
  Unaffected: a **model** or **`KV`** argument, which may be any place
  because the parameter *is* the caller's instance; and a `@grad fn`, where
  `!` marks a differentiable input rather than a mutation. Specified in
  `docs/SPEC.md §3.4.1` (Mutating parameters), ¶Argument form.
- **A `pub let` is a compile-time error**, not a silent no-op, and an
  importer cannot read an imported module's top-level `let` at all;
  constants cross a module boundary as a `pub fn` accessor.
- A type-arg-less annotation of a shape-parameterized model (`type B = Buf`,
  `fn f(b: Buf)`, `forge.zeros[Surface, [N]]`) is refused in every position,
  not just a literal's field slot. It is a warning in this draft and becomes
  an error in the next — one version of warning, not zero, not more.
- A tensor's, view's, or `KV`'s element type must be one of the scalar types;
  a model there is a located checker error instead of a silent acceptance.
- Every bound of a range or slice — start, end and step, in either spelling
  — is an integer expression; a bound with a known non-integer type is a
  check-time error.
- A named dimension in a type alias's body or in any other annotation
  resolves against a top-level literal `let` of that name, on both sides of
  the compiler and in method annotations as well as function ones. A mutable
  top-level binding is never a dimension.
- **The arena meters are hierarchical, not monotonic**, and both backends now
  meter **requested** bytes at the declared element width: an implicit
  intermediate, a builtin's fresh result, a `!` parameter's copy-in and an
  `@grad` pass's activations are all charged, where the interpreter
  previously charged only explicit constructors at its own f64 storage width.
  A fresh string is charged at its UTF-8 byte length wherever one
  materializes, including `+`'s concatenation arm, which was previously
  charged by the JIT and not at all by the interpreter — an unbounded
  divergence. `forge.restore(mark)` rewinds the bump cursor and returns
  nothing to the meter, on both backends. The JIT's chunk commitment asks the
  meter for the allocation's own need and is not itself the meter, so the
  refusal decision and the "free" figure in an exhaustion message are read
  off the requested-byte meter alone. `docs/MEMORY.md §1.1` carries the
  measured residuals that remain between the two backends.
- `dmc test --jit --all` collects every refusal in a file rather than
  stopping at the first, and reports per-file blocker depth.
- The interpreter-vs-JIT whole-example differential treats a timeout as a
  failure with its own bucket rather than folding it into a skip, so the gate
  reports one number under load instead of a different one per run. Public CI
  passes an explicit `--timeout 900` to suit the slowest machine the corpus
  meets.

### Fixed

- **A map value was reinterpreted as a pointer.** The JIT's coercion path
  turned an optional integer — what `map_get` yields — into a pointer when
  the context wanted one, so a program that compared or used an absent map
  value dereferenced an integer. Coercion now refuses instead of
  reinterpreting; comparing an absent map value answers the way the
  interpreter answers, and every other use is a located error.
- **Three JIT traps escaped their error envelope.** The chunk-allocation
  fatal wrote straight to stderr and exited, and the nil and match traps
  bypassed the `--json` envelope, so a tool consuming machine-readable
  diagnostics got a truncated stream and a raw message instead of a
  diagnostic. All three now go out through the same out-of-band path.
- **A bare model argument against a shape-parameterized callee read out of
  bounds** instead of refusing, and a model's shape arguments are now
  compared the way a tensor's are. A tensor-field read through a
  type-arg-less generic receiver refuses cleanly rather than dying inside the
  code generator.
- **A shape expression's error built a 4-element tensor.** An error raised
  while evaluating a shape expression was swallowed and the constructor
  proceeded with a default extent; it propagates now.
- **The constructor dtype table is total.** `fp8_e4m3`/`fp8_e5m2` now meter
  at one byte per element rather than silently falling through to f32, an
  unknown type name is an error instead of quietly becoming f32, and the
  lexer keeps the underscore in `fp8_e4m3`/`fp8_e5m2` when they appear as a
  constructor type argument. Overlay decoding reads `fp8` and `tf32` at their
  own widths, and the element-type table is exhaustive.
- **The `@cast(Model)` overlay decoded multi-byte scalars in the wrong byte
  order.** It is little-endian, matching the one byte order this language
  fixes.
- **A flat `let` of an arena constructor allocated twice under the JIT** and
  once under the interpreter (`let t = forge.ones[…]`). The JIT now lands the
  tensor once, like the block form, so the two backends' minimums agree.
- **Import failures are located.** A missing or cyclic import is a `resolve`
  diagnostic reported at the `use` — `import-unresolved` and `import-cycle` —
  instead of a bare message with no span, on both the resolver and JIT paths.
- **`dmc test`'s tally no longer zeroes on a file-level failure.** A file
  that failed to load reset the passed count for the whole run, so a real
  suite reported fewer passes than it had.
- Interpreter control flow and storage: `return`, `break` and `continue`
  propagate out of a block instead of being swallowed; a `let` inside a block
  is scoped to that block; the main file's top-level `let`s evaluate once;
  compound assignment to an indexed tensor element computes and stores in
  place; a tensor element store no longer clones the whole tensor;
  write-backs apply on bracketed method calls; `?` returns
  `(default[T], e)` on the error path as specified; a `type` alias is
  transparent at every run-time width origin; and every top-level name
  resolves against the file that mentions it.
- Checker coverage that was silently missing: the stream-iteration aliasing
  lint reaches call arguments and into `in b` operands; range bounds must be
  integers; a map-tailed function must declare `-> map`; `nil` is not a map
  key; map builtins carry a `str` key parameter type; and a method annotation
  resolves a top-level constant dimension the way a function's does, while a
  mutable `let` is never a dimension.
- Diagnostics spell types the way the parser accepts them — one canonical
  spelling table for every user-facing scalar type, and `TyType` renders
  every type in a form that can be pasted back into source.
- Folded shape dimensions go through the same integer arithmetic as
  everything else, so wrap and division by zero have one rule rather than two.
- A coerced tuple whose bytes do not change reuses its slot instead of
  allocating a fresh one.
- Byte-size formatting truncates at a unit boundary rather than rounding up,
  so an exhaustion message never claims a budget is larger than it is.
- The `extern fn` pre-flight tests resolve `libm`/`libc` symbols rather than
  the test binary's own exports, which resolved on macOS and not on Linux.

### Documentation

- `docs/SPEC.md`, `docs/MEMORY.md`, `docs/STDLIB.md`, `docs/OPERATORS.md`,
  `docs/DIRECTIVES.md`, `docs/AUTODIFF.md`, `docs/PORTS.md` and
  `docs/GRAMMAR.ebnf` are brought level with the behavior above. The spec
  draft is `0.0.5-draft`.
- `docs/GRAMMAR.ebnf` gains `budget_block` and the arena-constructor spine
  that carries the `in` tail, with the side condition written into the
  production rather than left in prose: `in` is admissible only after a
  constructor spine, and its operand is a postfix expression, never general
  arithmetic.
- `docs/AUTODIFF.md`'s capture rule no longer points at `pub let !x`, which
  is now itself an error; an imported module's file-private `let !x` is not
  a capture, and a `!` parameter is the way to reach one.

## 2026-09-04 — compiler: the real Rust floor, `@comptime` folding, backend-parity fixes

### Changed

- **The stated Rust requirement is 1.94, not 1.75.** The declared floor was
  never real: the Cranelift crates in the committed lockfile each require
  1.94, so a build on 1.75 produced a wall of dependency errors instead of
  the documented requirement. `compiler/Cargo.toml` and `README.md` now state
  the version the crate actually builds on.
- The compiler crate is on the **Rust 2024 edition**. No behavior change; the
  test count is the same on either edition.
- `@comptime` folds rather than parsing and doing nothing. It evaluates its
  block at compile time and replaces it with the resulting literal, before
  either backend lowers the program. The fold set is normative and narrow —
  integers, booleans, and shape arithmetic; floats, tensors, strings, and
  every call are refused as `comptime-non-static`, and a non-terminating body
  fails the compile as `comptime-budget` rather than hanging it. Previously
  `dmc run` evaluated the block and warned while `dmc jit` refused the same
  program outright, so a program that ran was a program that would not
  compile. Specified in `docs/SPEC.md §7.8` (Comptime evaluation), catalogued
  in `docs/DIRECTIVES.md §1` and `§3`.
- `extern fn` is enforced, not just specified. A parameter or return type the
  boundary does not admit is `extern-boundary`; a call from `@grad fn`,
  `@fuse`, or `@deterministic` is `extern-context`; a symbol the process
  cannot resolve is a `jit-extern` refusal naming the declaration, where it
  previously panicked inside the code generator with no span. `docs/SPEC.md
  §6.8` gains the tensor-to-`*T` materialization rule and the pointer's
  call-scoped lifetime.
- JIT refusals are classified more accurately: 64 sites that reported an
  unclassified `jit error` are lowering gaps and now say so, across five new
  refusal codes (`jit-element-type`, `jit-rank`, `jit-type-form`,
  `jit-slice`, `jit-conversion`). Nothing about acceptance moves — every site
  refuses, or succeeds, exactly as before. The differential tools score an
  unclassified error as a divergence, so the misclassification was hiding
  real gaps from the batteries that watch for them.

### Added

- `docs/NUMERICS.md` — the numerics contract: what the language promises
  about float accumulation, backend parity, integer wrap-at-width, and port
  wire numerics across versions, and what it deliberately does not promise.
- `docs/STABILITY.md` — the stability policy. There is no frozen version
  line, but surface that has shipped as documented, normative behavior does
  not move silently: a breaking change to the table in §2 requires a
  deprecation window, changelog entries, and a test that pins the
  replacement.
- `docs/SPEC.md §7.4b` — the float accumulation contract, which both
  backends have honored all along and no section stated: f32 reductions
  accumulate at f32 width in index order, and f32 matmul accumulates
  ascending over the inner dimension with one fused multiply-add per step,
  on both backends and every kernel. `docs/NUMERICS.md` is built on it.
- `dmc jit --blas` routes an f32 matmul of at least 2^17 MACs to the host's
  `cblas_sgemm`, reached by dynamic-symbol lookup rather than by linking a
  framework. It is **off by default** and **never** selected inside a
  `@deterministic` block: BLAS accumulates in its own blocked order, so the
  result is tolerance-equal to the default kernel's, not bit-equal. With the
  flag off, every matmul lowers exactly as it did before. The same detection
  is exposed as the `@host` feature name `accelerate`.
- Examples for the newly gated behavior: `comptime_fold.dmc`,
  `shift_ops.dmc`, `int_tensor_widths.dmc`, `int_tensor_dotted_widths.dmc`,
  `embed_index_type.dmc`, and wider coverage in `generic_slice_bounds.dmc`.

### Fixed

- **The release build was broken on aarch64.** An aarch64-gated SIMD matmul
  kernel was missed by the edition migration; on ARM under `-D warnings` its
  intrinsics and raw-pointer work were 26 hard errors. x86-64 hosts never
  compile that kernel, so it built clean there.
- **A match-arm or loop binding no longer destroys an outer local.** The JIT
  held one flat map of locals per function, so an enum-payload binding, a
  catch-all arm binding, or a `for` index wrote over an outer name of the
  same spelling and never restored it — including from an arm that never
  executed, since the map is consulted while lowering. The clobbered local
  then read as zero: both backends exited 0 and printed different numbers.
- **Integer elementwise tensor ops run at the element width.** The
  interpreter's dtype tag did not carry the width, so `.+ .- .* ./ .^` on a
  narrow-integer tensor fell through to f64 lanes: `2147483647 .+ 1` on two
  `Tensor[i32]` answered 2147483648 — a value that does not fit the element
  type — while the scalar spelling of the same addition answered
  -2147483648. Integer `./` is integer division, division by zero is 0, and
  unsigned elements wrap unsigned, exactly as the scalar operators do.
- **`\<` and `gelu()` compute one number.** The prefix operator evaluated an
  unrounded f64 kernel while the builtin rounded to f32, so `\<(x) ==
  gelu(x)` was false in the interpreter for the same input; the JIT compiled
  both through one kernel all along.
- An untyped float literal in an enum payload match arm adopts the payload's
  width. `match s { Circle(r) => r * r, Empty => 0.0 }` in an `-> f32`
  function — the specification's own worked example — ran under `dmc run`
  and was refused by `dmc jit` as arms disagreeing on type.
- A slab whose bound is a shape parameter (`x[0..S]`) keeps its tensor type
  instead of collapsing to the element type. The colon spelling of the same
  slab already typed correctly, so the two spellings disagreed.
- A trailing directive block yields the value of an `if` or `match`
  statement inside it, in the checker and the interpreter, instead of typing
  the enclosing function body as `nil`. The JIT already did.
- A negative resolved start on a runtime-start, statically-sized slice
  (`t[i..i+K]`) is out-of-bounds rather than counted from the end — the
  static extent already fixes the window's size. `docs/SPEC.md §4.3`
  (Indexing and slicing) states the exception; a scalar index and a slice
  bound assigned into still end-resolve as before.
- `embed` refuses a non-integer `ids` tensor at check time
  (`embed-index-type`). A float `ids` passed `--check`, after which the
  interpreter gathered rows by a truncated float index and `dmc jit` refused.
- `<-` accepts an appendee with the streaming axis dropped — the spelling
  `docs/SPEC.md §3.6` leads with, and the one a decode loop writes. Only the
  full-rank spelling ever worked; the dropped form died inside the array
  library. The capacity check on that path also read the appended extent off
  the wrong axis.
- `f16` narrowing carries the NaN payload across instead of collapsing every
  NaN onto two bit patterns, so 2044 of 65536 half patterns that did not
  survive a round trip now do.

### Documentation

- `docs/SPEC.md`, `docs/DIRECTIVES.md`, `docs/PORTS.md`, `docs/OPERATORS.md`,
  and `docs/ASSIMILATE.md` are brought level with the behavior above.
- Seven claims the shipped binary contradicted were corrected across the
  reference docs: `>>` is the arithmetic right shift rather than a second
  pipe spelling (and moves in the grammar accordingly), `\<` is GeLU rather
  than an inverted ReLU, and the JIT does lower port calls.
- `docs/PORTS.md §3.1`'s whole-value int/float wire decision is ratified and
  dated: the canonical writer emits `2.0` as `2`, the ambiguity is permanent
  by design, and a caller needing the distinction carries it in the payload.
- `docs/ASSIMILATE.md §4` documents `schema` as an integer *token* — `1.0`
  and `1e0` are rejected, which is what the reader already did.

## 2026-08-22 — compiler: assimilation, the port argument ABI, decode performance

### Added

- `dmc assimilate`: generate port-wrapper bindings from a JSON descriptor,
  or introspect a live python module (`dmc assimilate python:<module>`) into
  a draft descriptor — `--bindings` carries it through to wrappers in one
  command. Specified in `docs/ASSIMILATE.md`; generation is deterministic
  and descriptor-stamped.
- The port argument ABI (`docs/PORTS.md` §2): a `port_call` payload decodes
  as a positional array, an `{args, kwargs}` envelope, or a bare value.
- `examples/port_python.dmc` — the process-port floor end to end: every
  argument mode of the ABI, the error tags, and close, against a live
  python runtime.
- Port-effect enforcement: a port call inside a `@grad fn` is a
  compile-time error on both backends (`docs/PORTS.md` §5).

### Changed

- `dmc jit` type-checks the program before lowering, so an ill-typed
  program is refused with a diagnostic instead of failing mid-lowering.
- bool tensors support element assignment and element reads identically
  under the interpreter and the JIT.
- Explicit shape arguments bind the named form too, instead of silently
  falling back to an opaque tensor.
- Decode-path performance: GPU matmul dispatches are batched and
  pipelined, and the CPU bf16 decode GEMV is column-split across cores.
  Outputs are bit-identical to the previous kernels.
- Dependencies: cranelift 0.134.3, ureq 3.4.

## 2026-08-19 — docs: memory model, ports, package manifests, the grammar

### Added

- `docs/MEMORY.md` (arenas, copy-on-write, snapshots, the Stream arena),
  `docs/PORTS.md` (the foreign-runtime boundary), and `docs/PACKAGES.md`
  (the `demoni.json` manifest format), alongside `demoni.json` itself and
  `tools/validate_manifest.py`, which now runs in CI.
- `docs/GRAMMAR.ebnf` — the full grammar, reviewed against the parser (that
  pass's finding: the grammar *omitted* the implemented elementwise
  comparisons `.>` `.<` `.>=` `.<=`; now included). The review was not
  exhaustive — later passes found further divergences.
- The spec gains §3.11 (Ports): the `Port[L]` type and the three port
  builtins were implemented but unspecified.

## 2026-08-18 — docs: five reference docs, this changelog, per-example conformance gate

### Added

- Reference documentation alongside the spec: `docs/OPERATORS.md`,
  `docs/STDLIB.md`, `docs/AUTODIFF.md`, `docs/DIRECTIVES.md`,
  `docs/TOKENIZER.md`.
- This changelog, including the pre-release development timeline.
- `tools/example_runner.py` with a per-example conformance baseline
  (`tools/example_baseline.json`): every example is gated individually on
  its test results, on both backends, in CI.
- CI now also runs `lint_dmc.py`, a `diff_fuzz.py` smoke pass, and
  `numpy_oracle.py`.
- `examples/blends/tictactoe_ackermann.dmc` gains tests (previously the
  only example without them).

## 2026-08-18 — compiler: f32 correctness at run time, branch unification, JIT error classification

### Fixed

- Scalar `f32` values are real at run time on both the interpreter and the
  JIT; they no longer silently widen to `f64`.
- `f32` tensor reductions accumulate at `f32` width instead of promoting to
  `f64`.
- Matrix multiplication contracts its shared dimension consistently across
  all matmul code paths.
- Numeric literal suffixes (`f32`, `i64`, etc.) lex as a single token and
  stop at a token boundary, instead of merging with adjacent characters.

### Changed

- `if`/`else` branch types are unified by the type checker.

### Added

- JIT errors are now classified as deliberate refusal (a construct outside
  the JIT-supported subset) versus failure (a bug), a distinction the
  differential testing tools use.

Also includes diagnostic and comment cleanup.

## 2026-08-16 — jit: tuples, to_str, and a fix to the if-lowering join

### Added

- Tuples as a first-class JIT type: construction, destructuring (including
  `_` and `..` patterns), nesting, and tuple returns.
- `to_str` / `to_string` lower as coercion to `Str`.

### Fixed

- A join-block parameter bug in if-lowering.
- Tensor literals now take their element type from the surrounding binding.

Backend parity test coverage grew from 141 to 262 tests; the `examples/games`
set from 0 to 55.

## 2026-08-14 — tools: ship the interp-vs-JIT verification gates

### Added

- `tools/diff_backends.py`, `jit_probes.py`, `diff_fuzz.py`, `numpy_oracle.py`,
  `diff_demonic_lexer.py`, and `lint_dmc.py` — the correctness gates used to
  check the JIT against the interpreter, which is treated as the reference
  semantics.

## 2026-08-14 — compiler: field-binding semantics, integer suffixes, wrapped signatures, cross-arena enforcement

### Changed

- Field-binding semantics, integer literal suffixes, wrapped function
  signatures, and cross-arena enforcement in the compiler front end.
- Lists no longer deep-copy on every builtin call (a 40x speedup on
  list-building workloads).

### Added

- The self-hosted lexer example gains line/column spans, an error path,
  byte literals, and character escapes.

## 2026-08-10 — examples: ship the audited corpus (103 files) + CI gates

### Added

- An audited corpus of 103 example programs, gated in CI on every push:
  `dmc test examples` on the interpreter and `dmc test --jit examples` for
  backend parity.

## 2026-08-07 — demoniC 0.0.4-draft: initial public release

### Added

- Initial public release: the compiler (tree-walking interpreter and
  Cranelift JIT), README, language specification (`docs/SPEC.md`), under the
  Apache-2.0 license.

## Pre-release development (2026-05-24 → 2026-08-07)

Records the language's development leading up to the first public release
above. Dates are development timestamps from the project's internal history;
each entry condenses that day's work to its milestones.

### By the numbers (as of 2026-06-29)

Built over **37 days** (2026-05-24 → 2026-06-29):

- **413** commits
- **193** tracked issues resolved — bugs, feature proposals, spec audits
- **~52,100** lines of Rust across the compiler (lexer, type checker,
  interpreter, JIT)
- **1,300** tests
- **206** demoniC programs (`.dmc`), with reference translations to C, C++,
  and Python

### Highlights

- **Language toolchain**: a complete front-to-back pipeline — lexer,
  three-pass type checker, tree-walking interpreter, and a Cranelift JIT —
  for a tensor-first language with shape-parametric types, arena memory
  (`forge` / `vault`), reverse- and second-order autodiff (`@grad`), and
  ML-shaped builtins (attention, RoPE, softmax, grouped-query attention).
- **JIT backend**: native code generation covering scalars, control flow,
  tensors, SIMD kernels, fused elementwise chains, autodiff, KV-cache
  attention, closures, hash maps, and weight-file I/O — reaching parity with
  the interpreter on the example corpus.
- **Examples**: a corpus of runnable programs (games, a fantasy console, an
  SDF raymarcher, bytecode VMs, a Scheme interpreter) plus reference
  translations to mainstream languages.

---

The final three weeks (2026-07-17 → 2026-08-07) covered release
preparation: assembling the public repository tree and the tooling that
generates it.

### 2026-07-16

- The JIT now reclaims gradient-call memory automatically: each compiled
  `@grad` call snapshots the allocator on entry and compacts back to that
  point on return, so training loops no longer need manual arena
  management.

### 2026-07-13

- Fused attention (`attn`, `attn_gqa`) now differentiates on both the
  interpreter and the JIT, closing the last fused-op gradient gap.
- Fixed: the JIT silently ignored the mask argument on plain `attn`,
  computing unmasked attention while the interpreter masked correctly.

### 2026-07-08

- Fixed: integer `@cast` blocks silently produced wrong values on the JIT
  instead of truncating like the interpreter; the JIT now rejects them as
  unsupported rather than computing a wrong answer.

### 2026-06-29

- Upgraded the Cranelift code-generation dependency to 0.133.1.

### 2026-06-26

- CI began additionally running the JIT backend against the example test
  suite, catching interpreter/JIT divergences in ground-truth tests, not
  just compile failures.

### 2026-06-25

- Backend conformance gating moved from two aggregate pass/fail counts to
  per-example ground truth, so a single regressing example fails the build
  without a broad improvement masking it.
- Enum variants may now carry positional data (tagged unions), with
  construction and pattern matching checked against each variant's declared
  fields, across the interpreter, formatter, and JIT.

### 2026-06-21

- C-like enums (declared with `enum`) now compile end-to-end through the
  JIT, not just the interpreter.
- Fixed: the JIT rendered a boolean coerced to a string as `1`/`0` instead
  of `true`/`false`, diverging from the interpreter.

### 2026-06-20

- Added an experimental GPU backend path (Metal, macOS) behind a build
  flag; the CPU path remains the default and the correctness reference.
- Cut decode latency roughly 30x by caching a loop-invariant computation
  instead of recomputing it every step.

### 2026-06-19

- `match` on an open scalar type (`i64`, `str`, floats, etc.) now requires a
  catch-all arm at compile time, instead of risking a runtime "no arm
  matched" panic.
- Added closed-set `enum` declarations with compile-time match-coverage
  checking against all variants (interpreter only at this point).
- Out-of-range integer literals (e.g. assigning 300 to an `i8`) are now
  caught at compile time.
- Fixed: the JIT's bf16 tensor transpose upconverted to a much larger f32
  buffer before transposing; it now transposes at native width, reducing
  peak memory for large weight tensors.

### 2026-06-18

- Method-call syntax (`x.f(args)`) now desugars to the equivalent
  free-function call across all backends.
- New builtins: `sort`, `gcd`, `median`, and string-conversion aliases.
- Cache-blocked and register-blocked the matmul kernel and multithreaded
  large matrix multiplications.

### 2026-06-17

- Added Rust-style byte literals (`b'A'`).

### 2026-06-16

- Implicit numeric type conversions are now rejected; untyped numeric
  literals adopt the type they're used in, and unconstrained ones default
  to 64-bit.

### 2026-06-12

- Grouped-query attention lowers in the JIT against streaming key/value
  caches (not just dense tensors), with a runtime-determined history
  length.
- The activation functions `relu`, `sigmoid`, `tanh`, and `gelu` —
  previously JIT-only — are now implemented identically on the interpreter
  and type checker; added `silu`.

### 2026-06-10

- Added general strided tensor slicing (any mix of scalar/range/full-axis
  indices), multi-axis `argmax`/`argmin`, an embedding-row-gather builtin,
  and integer-element tensors, to the JIT.
- Fixed several interpreter/JIT divergences: scalar dotted arithmetic,
  tensor `max`/`min`/`argmax`/`argmin` reductions, negative tensor indices,
  a zero-valued `main` return being printed as nothing, and
  compile-time-constant top-level `let` bindings — all now agree between
  backends.

### 2026-06-09

- Fixed: several soundness and safety issues found in a targeted
  correctness sweep — mismatched match-arm types, under-checked tuple
  destructuring arity, and incomplete model-constructor field checks all
  passed type-checking despite being wrong; integer division/modulo by
  `i64::MIN` and `-1` and out-of-bounds float-to-int casts on the JIT no
  longer crash the process; invalid regex patterns now raise a clean
  runtime error instead of silently matching nothing.
- Fixed: JIT tensor bindings and struct fields aliased their backing memory
  instead of copying, so a later write could silently corrupt an earlier
  "snapshot" — value semantics now match the interpreter.
- `f32`-declared tensors now compute at true `f32` precision in the
  interpreter by default (previously f64-backed), matching the JIT.
- Second-order autodiff (`f.fwd_bwd_bwd(...)`) is now reachable from source
  on both backends.

### 2026-06-08

- Added two more opt-in static-analysis lints: identity-operand arithmetic
  (`x + 0`, `x * 1`, etc., which can hide an infinite loop on a loop
  counter) and dead self-assignment (`x = x`).

### 2026-06-07

- Added `tools/diff_backends.py`, an interpreter-vs-JIT differential test
  that runs every JIT-runnable example under both backends and asserts the
  output matches.
- Fixed: JIT tensor indexing lowered to an unchecked raw memory access;
  out-of-range indices now trap deterministically instead of corrupting
  memory.
- Added a `Map` type distinct from `list`, so the checker can flag
  iterating directly over a map (maps aren't iterable; iterate over
  `map_keys`/`map_vals` instead).
- Fixed: JIT tensor printing used a different format than the interpreter;
  both now match byte-for-byte.
- Fixed: the interpreter silently ignored the `*=`, `/=`, `&=`, `|=`, and
  `^=` compound assignment operators.

### 2026-06-05

- Extended the method-call lint to flag any call to an unsupported method,
  and unsupported `str` methods specifically, at compile time instead of
  failing at runtime.

### 2026-06-04

- Added `read_bytes`, an exact binary file reader, for lossless
  weight-file loading.
- Added demon mode (`--demon`), which suppresses the opt-in lint family for
  unrestricted runs; hard type errors still fire.
- Added lints for method-call syntax on builtins (which type-checks but
  resolves to a nonsensical value at runtime, since demoniC has no
  method-call syntax) and for truncated-vs-floored `%` semantics on
  negative operands.
- Added the `any` type, a dynamic escape hatch for heterogeneous function
  boundaries (interpreter only).

### 2026-06-03

- The JIT gained native KV-cache attention end to end, capturing closures,
  runtime string keys, model staging (including fixed-size model arrays),
  and several scalar-math builtins.
- Added native balanced-ternary tensors (`forge.trit`) with a specialized
  JIT matmul kernel.
- Added runtime type introspection (`typeof`, `is_int`, etc.) and safe
  string-to-number parsing.

### 2026-06-02

- The JIT gained hash maps, first-class function pointers, for-loops,
  `match`, strings, model structs, and weight-file I/O (`vault.load` /
  `vault.load_npz`).

### 2026-06-01

- Added second-order autodiff (`@grad @grad`) with re-taping through
  activation functions and first-order MAML-style plumbing.

### 2026-05-31

- The JIT gained autodiff, elementwise-chain fusion (`@fuse`),
  multi-parameter `@grad`, SIMD ReLU, C FFI (`extern fn`), and 4D batched
  matmul/attention/RoPE/grouped-query attention.
- Added `solve`, `inv`, and `lstsq` linear-algebra builtins to the
  interpreter.
- Fixed: interpreter recursion no longer aborts the process on stack
  exhaustion.

### 2026-05-30

- Expanded formatter, checker, and interpreter test coverage; added
  reference translations to C, C++, and Python for existing example
  programs.

### 2026-05-29

- Added a small fantasy game console (assembler, a Snake cartridge,
  live-input builtins) and an SDF ray-marching renderer, plus example
  programs including a bytecode stack VM.
- The JIT gained tensor operations: ReLU, `forge.zeros`/`ones`
  constructors, reshape, axis broadcast, and comparison masks.

### 2026-05-28

- The JIT backend gained SIMD tensor kernels, register-tiled matmul, a
  `.fwd_bwd` training-loop entry point, and `@grad` reverse-mode autodiff.
- Added `extern fn` (C ABI) support across the lexer, parser, checker, and
  interpreter.
- Added a differential JIT-vs-interpreter test harness.

### 2026-05-27

- Added a module system (multi-file programs, `pub` visibility) and kicked
  off the JIT backend (Cranelift-based scalar and control-flow lowering).
- Added `@grad` autodiff to the interpreter and type checker, and
  substantially expanded the standard library (collections, string/math
  builtins, file I/O, JSON, regex, date/time, and more).
- Added `dmc fmt` (a pretty-printer), a profiling mode, and the `dmc test`
  runner.
- Added models and methods, batched matmul, distribution directives, and a
  streaming tensor axis.

### 2026-05-26

- The tree-walking interpreter went from scaffolding to actually running
  programs, alongside a three-pass type checker.
- Added reverse-mode autodiff (finite-difference `@grad`), `match`
  expression evaluation, and core tensor/string builtins.

### 2026-05-25

- Project start: the initial language spec, memory model, operators, and
  grammar, plus a Rust-based lexer bootstrap and the first example
  programs.
