# demoniC — Verification

**Companion to:** `README.md` (Testing), `docs/SPEC.md §7` (Execution model),
`docs/DIRECTIVES.md §1` (implementation status).

This document states what is verified, by what, and what is not claimed. It
exists because "the two backends agree" and "the behavior is independently
validated" are different strengths of evidence, and a reader auditing the
project should not have to reverse-engineer which gate provides which.

Everything below is reproducible from a fresh clone with stable Rust and
Python 3 (NumPy for the oracle). The toolchain is pinned: `rust-version` in
`compiler/Cargo.toml`, `Cargo.lock` committed.

---

## 1. Chain of authority

```
docs/SPEC.md  (normative)
   ↑ tested by: compiler/tests/spec_probes/   — probes trace to spec sections
interpreter   (reference implementation, `dmc run`)
   ↑ tested by: differential gates            — JIT must match the interpreter
JIT           (`dmc jit`)
```

- **The spec wins.** Where an implementation disagrees with the spec, the
  implementation is wrong. `compiler/tests/spec_probes/` holds minimal
  programs each tracing to a specific spec section; a probe the interpreter
  currently fails is parked in `spec_probes/PENDING.md` and tracked as an
  implementation gap, not re-specified around.
- **The interpreter is the reference implementation.** Machine code cannot be
  diffed against a document, so the JIT is verified mechanically against the
  interpreter, and the interpreter is verified against the spec.
- **The JIT refuses rather than guesses.** Constructs outside its statically
  typed subset produce a classified, deliberate error directing to `dmc run`.
  The differential tools distinguish this refusal (*jit-gap*, informational)
  from a wrong answer (*divergence*, always a failing bug).

## 2. Levels of evidence

Ordered weakest to strongest. A claim in this repository should be read at
the highest level a gate actually establishes, and no higher.

| Level | Claim | Establishes | Cannot establish |
| ----- | ----- | ----------- | ---------------- |
| 0 | it type-checks (`dmc --check`) | shape/dtype consistency | numeric correctness — type-clean-but-wrong code passes |
| 1 | ground truth | `fn test_*() -> bool` assertions with hand-computed expected values pass under the interpreter | correctness of untested paths |
| 2 | internal consistency | interpreter and JIT agree observationally | a shared bug computed identically by both backends |
| 3 | external validation | output matches an independent NumPy reimplementation written from the spec math, not from the Rust | correctness beyond the ops and tolerances the oracle covers |

Level 2 is where most of the volume is (hundreds of parity assertions plus
fuzzing); level 3 is what makes a shared-bug escape visible. Both are needed:
neither subsumes the other.

## 3. The gates

| Gate | Compares | Level | In CI |
| ---- | -------- | ----- | ----- |
| `cargo test --all` | unit + integration tests, incl. spec probes | 0–2 | yes |
| `dmc test examples` | every example's `test_*` assertions, interpreter | 1 | yes |
| `dmc test --jit examples` | the same assertions on both backends, compared | 1 + 2 | yes |
| `tools/example_runner.py` | per-example conformance vs `tools/example_baseline.json` — each file gated individually, so one regression cannot hide behind aggregate improvement | 1 + 2 | yes |
| `dmc selftest` | generated random well-typed programs, both backends diffed | 2 | yes |
| `tools/diff_backends.py` | whole-example output, `dmc run` vs `dmc jit` | 2 | yes |
| `tools/jit_probes.py` | curated edge cases (NaN/inf, div/mod by zero, saturating casts, index bounds, degenerate reductions) on both backends | 2 | yes |
| `tools/diff_fuzz.py` | generated programs, both backends diffed; pinned regression seeds | 2 | yes |
| `tools/numpy_oracle.py` | tensor-op results vs an independent NumPy reference, explicit rtol/atol | 3 | yes |
| `tools/diff_demonic_lexer.py` | the self-hosted demoniC lexer vs the Rust lexer, token by token | 2 | on demand |
| `tools/lint_dmc.py` | `.dmc` style rules | — | yes |

See `.github/workflows/ci.yml` for the authoritative CI list.

## 4. Determinism and reproduction

- **Fuzzing is seeded, not random.** `diff_fuzz.py` derives every program
  from `--seed` plus its index; a failing program prints its exact seed and
  source, and `--repro SEED` re-emits and reruns that one program. Minimal
  historical repros are pinned in `REGRESSION_SEEDS` so fixed bugs stay
  covered. `dmc selftest` is likewise seeded (seed base printed in its
  summary line).
- **The harnesses prove they have teeth.** `diff_fuzz.py --meta-test` and
  `numpy_oracle.py --meta-test` check a correct result against a
  deliberately wrong reference and assert the mismatch is caught. CI runs
  both meta-tests, so a silently toothless gate is itself a CI failure.
- **Known divergences are visible, not hidden.** `diff_backends.py` carries
  an explicit allowlist where each entry names the tracking issue; the gate
  stays green while an issue is open, and any *new* divergence fails loudly.
  `jit_probes.py` prints its jit-gap count and warns per-probe on any gap
  not in its allowlist; only a divergence fails the gate.

A skeptic's minimal session:

```
cd compiler && cargo build --release && cargo test --all && cd ..
compiler/target/release/dmc test --jit examples     # levels 1 + 2
compiler/target/release/dmc selftest                # level 2, generated
python3 tools/numpy_oracle.py                       # level 3, external
```

## 5. What is not claimed

- **Parity is not proof of correctness.** A wrong formula implemented
  identically twice passes every level-2 gate; only ground-truth tests and
  the NumPy oracle can catch it, and they cover what they cover — the
  oracle's op list and tolerances are in `tools/numpy_oracle.py`.
- **Float comparisons are tolerance-based.** The oracle uses explicit
  rtol/atol; bit-exactness is promised only inside `@deterministic`
  (`docs/SPEC.md §7.3`) and only on the same host.
- **The JIT covers a subset.** Skipped-as-outside-subset counts are printed
  by `dmc test --jit` and the differential tools; a refusal is working as
  designed, a divergence never is.
- **Some specified surfaces are not implemented.** The per-directive status
  column in `docs/DIRECTIVES.md §1` and `spec_probes/PENDING.md` are the
  honest inventory; nothing in this document upgrades them.
- **No performance claims are made here.** Every gate above is about
  behavior, not speed.
