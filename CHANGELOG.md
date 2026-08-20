# Changelog

This file tracks releases of the public demoniC repository. Releases are
periodic snapshots of ongoing development, so each entry batches multiple
changes rather than corresponding to a single commit.

## 2026-08-19 — docs: memory model, ports, package manifests, the grammar

### Added

- `docs/MEMORY.md` (arenas, copy-on-write, snapshots, the Stream arena),
  `docs/PORTS.md` (the foreign-runtime boundary), and `docs/PACKAGES.md`
  (the `demoni.json` manifest format), alongside `demoni.json` itself and
  `tools/validate_manifest.py`, which now runs in CI.
- `docs/GRAMMAR.ebnf` — the full grammar, verified production-by-production
  against the parser (the audit's one finding: the grammar *omitted* the
  implemented elementwise comparisons `.>` `.<` `.>=` `.<=`; now included).
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
