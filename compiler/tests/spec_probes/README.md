# Spec-coverage probe suite

47 minimal `.dmc` programs, each ~5–15 lines, each tracing to a specific
section of `docs/SPEC.md` (or the auxiliary docs). The integration test
in `../spec_probes.rs` walks this directory and runs every probe through
the full `dmc` pipeline (lex → parse → check → run); every probe must
exit cleanly.

## Purpose

This is a **regression** suite, not a feature-completeness suite. Every
probe here is known to pass on `main`. If any probe starts failing, a
recent change broke a spec-promised behavior.

Pairs nicely with the inline unit tests in `src/*_tests.rs`, which cover
library API and IR shapes. The probes here cover the user-facing
end-to-end contract: "if I write the syntax the spec describes, does
the binary do the right thing?"

## How to read a probe filename

```
pNN_short_name.dmc
```

`NN` is roughly the chronological order of authorship; not a priority or
section number. Each file opens with a header comment naming the spec
section being exercised. Example:

```dmc
# Probe 27 — Spec §4.5: match on shapes (the spec's primary match use case)
```

## How to add a probe

1. Pick a spec feature not already covered (see grouping below).
2. Write the smallest demoniC program that exercises it, including a
   `fn main() -> nil` that produces visible output.
3. Run it: `cargo run --release -- path/to/probe.dmc` must print
   `Run OK`.
4. Drop the file in this directory. The integration test auto-discovers
   it.

## Current coverage by spec section

| Spec § | Probes |
|---|---|
| §2 (lexical, literals, comments, idents, strings) | p01, p02, p03, p04, p38, p39 |
| §3 (types: aliases, models, tuples, dynamic shape, Rng, Mesh) | p12, p23, p23b, p24, p25, p29, p32, p34 |
| §4 (expressions: slicing, transpose, broadcast, ReLU, pipe, power, control flow, match, range, cast, tensor literals) | p05, p06, p07, p08, p09, p11, p13, p14, p15, p16, p26, p27, p28, p36, p40, p40b, p43 |
| §5–6 (statements, items, arenas, shape arithmetic) | p10, p17, p35, p45 |
| §7 (JIT model, directives) | p18, p19, p20, p21, p22, p33, p37, p42, p44 |
| `STDLIB.md` (fused primitives) | p46 |

## Excluded probes — see PENDING.md

Two probes from the original 49-probe set live in `PENDING.md` instead
of this suite because their failures are interpreter-side gaps still
under development, not regressions in implemented behavior.
