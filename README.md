# demoniC

demoniC is a tensor-first systems language. Three things are in the language
itself rather than in a library on top of it: reverse-mode automatic
differentiation (`@grad`), tensor shapes checked at compile time, and arena
memory with no garbage collector.

Two backends run those semantics — a tree-walking interpreter, which is the
reference, and a Cranelift JIT over a statically typed subset. The compiler
is written in Rust.

This project — code, tests, and documentation — is written and maintained by
AI, directed by a human maintainer. Multiple models have contributed:
primarily Claude, with work from Gemini, Codex, Grok, GLM, Qwen, Mistral,
and others.

## Building

Requires stable Rust.

```
cd compiler
cargo build --release
```

The binary is `compiler/target/release/dmc`.

## Usage

| Command             | Behavior                                            |
| ------------------- | --------------------------------------------------- |
| `dmc run f.dmc`     | tree-walking interpreter — full semantics           |
| `dmc jit f.dmc`     | Cranelift JIT — statically typed subset             |
| `dmc f.dmc`         | full pipeline: lex, parse, check, run               |
| `dmc --check f.dmc` | type and shape check only, no execution             |
| `dmc test path`     | run every zero-arg `fn test_*() -> bool` found      |
| `dmc test --jit path` | additionally run JIT-eligible tests on both backends and compare |
| `dmc fmt f.dmc`     | canonical pretty-print                              |
| `dmc selftest`      | generate random well-typed programs, run both backends, diff the results |

## Example

```
@grad fn loss[D](!w: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
    let d = w .- x
    sum(d .* d)
}

fn main() -> nil {
    let x = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
    let !w = forge.zeros[f32, [4]]
    for step in 0..20 {
        let (l, g) = loss.fwd_bwd(w, x)
        w = w .- g.w .* 0.1f32
    }
    print(loss(w, x)); print("\n")
    nil
}
```

```
$ dmc run example.dmc
0.003987707299529575
```

`@grad` generates the backward pass; `loss.fwd_bwd` returns the loss and a
gradient struct with one field per `!` parameter. Twenty gradient-descent
steps fit `w` to `x`.

Points where demoniC differs from what its syntax may suggest:

- Comments start with `#`, not `//`.
- Mutability is `let !x` (or `let mut x`); plain `let` bindings and their
  data are immutable.
- Elementwise tensor arithmetic is dotted (`.+ .- .* ./`); bare `+` on
  tensors is an error. `@` is matrix multiply, `'` is postfix transpose.
- `**` is power; `^` is XOR. `>>` is a pipe operator, not right shift.
- Allocation goes through arenas (`forge`, `vault`, `stream`) with
  bump-pointer semantics; there is no garbage collector.

The full language definition is in [docs/SPEC.md](docs/SPEC.md).

## Correctness

The code here is written by AI, which is no longer unusual. So rather than ask
for trust, this section says what can be checked. Everything named below ships
in this repository and runs in CI on every pull request and every push to
`main`; the commands are under [Testing](#testing).

- **Two implementations, differentially tested per example.** The interpreter
  and the JIT are independent implementations of the same semantics.
  `dmc test --jit examples` runs every JIT-eligible `test_*` on both backends
  and compares the results; `tools/diff_backends.py` diffs whole-program output
  between them; `tools/jit_probes.py` covers edge cases the example corpus does
  not reach — NaN and inf propagation, integer division and modulo by zero,
  saturating casts, negative and out-of-bounds indexing, degenerate reductions,
  matmul edge shapes. A JIT that disagrees with the interpreter is a bug even
  when it is faster.
- **An external oracle, not two of our own implementations agreeing.** A wrong
  result computed identically by both backends passes every check above.
  `tools/numpy_oracle.py` compares demoniC tensor results against an
  independent NumPy implementation written from each op's definition in the
  spec rather than from demoniC's Rust. Its `--meta-test` mode checks a correct
  demoniC result against a deliberately wrong reference and fails if the
  mismatch goes uncaught, so an oracle that has quietly stopped comparing
  anything is itself a CI failure.
- **Generated programs, not only the ones we thought to write.** `dmc selftest`
  generates random well-typed programs, runs each through both backends and
  diffs them, in-process. `tools/diff_fuzz.py` does the same with each program
  in its own process, so a native JIT crash is reported rather than taking the
  harness down with it. Both are seeded: a failure prints the seed and the
  source for a one-command repro, and past repros stay pinned as regression
  seeds.
- **A recorded baseline, not a green light.** `tools/example_runner.py` gates
  every example individually against `tools/example_baseline.json`. An example
  counts only if it carries executable `test_*` assertions, they pass under the
  interpreter, and the two backends agree wherever it is JIT-runnable — an
  example that merely compiles is reported as unverified, never as passing. A
  file whose passing-test or JIT-parity count drops fails the build, a new
  example has to be recorded deliberately, and a vacuous test — one with no
  call or comparison in it — is refused.

The limits, stated plainly: no tool decides whether an arbitrary program is
correct. Agreement between the backends is evidence about the JIT, not proof
about the semantics, which is why the NumPy oracle exists; the oracle covers
tensor ops rather than the whole language; the fuzzers generate scalar
programs. And `dmc --check` is type- and shape-only — it accepts code that is
type-clean and computes the wrong number.

## Testing

```
cd compiler
cargo test --all
```

`dmc selftest` is the compiler-derived differential suite: it generates
random well-typed programs, runs each through both the interpreter and the
JIT, and reports any divergence.

`tools/` holds the correctness gates. The interpreter is the reference
semantics, so most of them check that the JIT agrees with it. CI runs all of
these except the lexer differ, which is run on demand. What each gate proves —
and the difference between "both backends agree" and "independently
validated" — is laid out in [docs/VERIFICATION.md](docs/VERIFICATION.md):

```
python3 tools/example_runner.py       # per-example gate vs tools/example_baseline.json
python3 tools/diff_backends.py        # interpreter vs JIT, whole-example output
python3 tools/jit_probes.py           # interpreter vs JIT, curated edge cases
python3 tools/diff_fuzz.py            # generated programs, both backends diffed
python3 tools/numpy_oracle.py         # tensor ops vs an independent NumPy reference
python3 tools/diff_demonic_lexer.py   # the demoniC-in-demoniC lexer vs the Rust one
python3 tools/lint_dmc.py             # .dmc style checks
```

`numpy_oracle.py` needs NumPy; the rest need only Python 3 and a built `dmc`.

If you change the JIT, run `diff_backends.py` and `jit_probes.py`: a change
that makes the JIT disagree with the interpreter is a bug even if it is
faster, and these are what catch it.

## Status

Pre-0.1 draft. Breaking changes are expected on every revision. The
interpreter is the reference semantics; the JIT compiles a statically typed
subset and reports a clear error for constructs outside it.

## License

Apache-2.0. See [LICENSE](LICENSE).
