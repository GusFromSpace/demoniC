# demoniC

demoniC is a tensor-first systems language with built-in reverse-mode
automatic differentiation, a tree-walking interpreter, and a Cranelift-based
JIT compiler. The compiler is written in Rust.

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

## Testing

```
cd compiler
cargo test --all
```

`dmc selftest` is the compiler-derived differential suite: it generates
random well-typed programs, runs each through both the interpreter and the
JIT, and reports any divergence.

`tools/` holds the correctness gates. The interpreter is the reference
semantics, so most of them check that the JIT agrees with it. CI runs the
first two on every change; the rest are run on demand:

```
python3 tools/diff_backends.py        # interpreter vs JIT, whole-example output
python3 tools/jit_probes.py           # interpreter vs JIT, curated edge cases
python3 tools/diff_fuzz.py            # generated programs, both backends diffed
python3 tools/numpy_oracle.py         # tensor ops vs an independent NumPy reference
python3 tools/diff_demonic_lexer.py   # the demoniC-in-demoniC lexer vs the Rust one
python3 tools/lint_dmc.py             # .dmc style checks
```

If you change the JIT, run `diff_backends.py` and `jit_probes.py`: a change
that makes the JIT disagree with the interpreter is a bug even if it is
faster, and these are what catch it.

## Status

Pre-0.1 draft. Breaking changes are expected on every revision. The
interpreter is the reference semantics; the JIT compiles a statically typed
subset and reports a clear error for constructs outside it.

## License

Apache-2.0. See [LICENSE](LICENSE).
