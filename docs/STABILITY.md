# demoniC — Stability Policy

**Companion to:** `docs/NUMERICS.md`; `docs/PORTS.md §3.1`;
`docs/VERIFICATION.md` (the gates named in §2 are described there).

This document is normative.

## 1. Policy

demoniC has no frozen version line. The requirement set is still being
discovered, so the language surface stays open, and additions are never
blocked by this document.

What this document fixes is discipline on what has already shipped as
documented, normative surface — the table in §2 lists it. Shipped surface does
not shift silently: a breaking change to it requires a deprecation window, a
`CHANGELOG.md` entry, and a test that pins the replacement behavior (§3). This
is stability by policy, not by version number.

## 2. Stable surfaces

| Surface | Specified in | Drift detection today |
| --- | --- | --- |
| Grammar | `docs/GRAMMAR.ebnf`, cross-checked by `docs/SPEC.md` | The example corpus, through `dmc test examples`, `dmc test --jit examples`, and `tools/example_runner.py` against its recorded per-example baseline. These catch a change to already-shipped syntax only where an example exercises it; a new construct with no example is not caught. |
| Port ABI: framing, typed decode, error tags | `docs/PORTS.md §2, §3.1, §6` | One shared port registry backs both backends, so they cannot independently drift from each other; `tools/diff_backends.py`, `tools/jit_probes.py`, and `dmc test --jit` exercise it. No check diffs the ABI against a prior shipped version. |
| Port ABI: tensor envelope | `docs/PORTS.md §3.2` | Same shared registry; gated by the differential tools, the example corpus, and interpreter-vs-JIT byte-identical checks for bool and integer tensors. |
| Whole-value int/float wire decision | `docs/PORTS.md §3.1` (ratified 2026-09-02) | Same as the port ABI above. The decision itself is permanent by design (§3.1) — not merely gated, closed. |
| Assimilation descriptor format | `docs/ASSIMILATE.md §4` | The descriptor reader refuses an unknown or mismatched `schema` field at parse time. That enforces the contract at run time; no check compares the document's prose against the constant. |
| `demoni.json` manifest | `docs/PACKAGES.md §3, §4` | `tools/validate_manifest.py`, which runs in CI, including a `manifest-schema` error on the wrong `schema` value. |
| Numerics promises (§1 of `docs/NUMERICS.md`) | `docs/NUMERICS.md §1`; `docs/SPEC.md §7.2b` (Float accumulation contract), `§7.3` (Determinism contract (`@deterministic`)) | `tools/numpy_oracle.py` (tolerance-bound external reference check), `tools/diff_backends.py` and `tools/jit_probes.py` and `dmc selftest` and `tools/diff_fuzz.py` (interpreter-vs-JIT agreement), `tools/example_runner.py` (per-example results pinned). |

A row whose mechanism is partial is not an excuse to skip the procedure in
§3 — it means a reviewer has to do by hand what a check would otherwise
catch, and it is a candidate for a future gate.

`docs/VERIFICATION.md` describes what each gate above establishes, and the
difference between "the two backends agree" and "independently validated".

## 3. Changing a shipped surface

Breaking a row in §2 — not adding to it — requires all four:

1. **Deprecation window: 14 calendar days minimum**, starting from the
   `CHANGELOG.md` entry that announces the deprecation. Nothing here is
   tagged by release and the changelog is batched irregularly, so a fixed
   calendar window is the only unit both a maintainer and an outside reader
   can check without guessing at a release cadence. 14 days spans at least
   one observed release without being long enough to stall a
   one-maintainer repository.
2. **A `CHANGELOG.md` entry** at deprecation time — what is changing, why,
   and the replacement — and a second at removal time.
3. **A test that pins the replacement behavior**, not just the old
   behavior's removal. It lives beside the surface's existing gate (§2): a
   port fixture for the ABI, an example for grammar, a
   `tools/validate_manifest.py` or `tools/numpy_oracle.py` case for the
   manifest or numerics.
4. **A dated decision record.** In this repository the two `CHANGELOG.md`
   entries of (2) are that record: between them they name the surface, the
   old and the new behavior, and the dates.

Skipping any of the four is grounds for rejecting the change, the same way a
change that contradicts the specification is rejected.

## 4. What is NOT stable

- **The JIT-supported subset.** Versioned, not frozen: which constructs the
  JIT lowers changes every release by design. A construct outside the subset
  is a classified refusal directing to `dmc run`, never a wrong answer.
- **Numerics non-promises** (`docs/NUMERICS.md §2`): matching another
  implementation's summation order, and the `--blas` GEMM offload's numerics
  — which is off by default, so a run that does not ask for it is unaffected.
- **Directives marked "not yet" or "parse-accepted no-op"** in
  `docs/DIRECTIVES.md §1` — `@inplace` and `@recompute` today. A directive
  that computes nothing makes no behavioral promise, whatever its name
  suggests (`docs/NUMERICS.md §2.4`). A directive that leaves that set gains
  one: `@comptime` did, and every stacking rule in `docs/DIRECTIVES.md §3`
  is now enforced, `comptime-non-static` included.
- **Experimental backends.** The GPU/Metal backend (`--features gpu`, macOS)
  is experimental and outside this policy until it is not.
- Anything not listed in §2. Silence is not a promise; it means the surface
  has not been reviewed for this table yet, not that it is frozen.

## 5. Why not a version line (2026-09-02)

Cutting a version line now would freeze surface before the language has
enough real use to know what the surface should be. The lesson this project
takes from languages that froze early is to freeze narrow — freeze what an
external tool could actually depend on — and not to freeze for optics.

This document is the discipline that implies, without a version cut: shipped
surface is protected by procedure, not by declaring a number stable and
hoping nothing under it needs to move.
