# demoniC — Tokenizer Stability

**Companion to:** `docs/SPEC.md §2` (Lexical structure), `docs/SPEC.md §1`
(Design constraints) — constraint 5, stable tokenization.

This document defines the rules demoniC source bytes must follow so
that the language tokenizes stably under modern BPE tokenizers. A
language designed to be emitted by models has to care how it looks to
the tokenizer that emits it.

This is normative for the **canonical** syntax. Alternate spellings
(see §3) are legal but discouraged in checked-in code.

---

## 1. Goal

For each demoniC operator and keyword, define the canonical source
byte sequence and assert that, under the BPE vocabularies of the major
open and closed model families circa the spec version's date, the
sequence:

1. Tokenizes to a small, stable number of tokens (ideally 1).
2. Does **not** split mid-operator in a way that fuses with adjacent
   identifiers (the "`A.+B` collapses to one token" failure mode).
3. Is reachable from natural code-prompt prefixes — i.e., a model
   that has seen Python and Julia in pretraining can emit the
   operator without having to fight its sampling distribution.

---

## 2. Canonical operators

| Operator | Canonical bytes | Notes                                    |
| -------- | --------------- | ---------------------------------------- |
| `@`      | `@`             | single token in essentially every BPE    |
| `'`      | `'`             | postfix transpose; ensure no quoted-str confusion at lex |
| `\>`     | `\>` with a **leading space** required when in expr position | The leading space is **part of the canonical syntax** to keep `\>` lex-stable across BPEs that fuse adjacent identifiers |
| `\<`     | `\<` with leading space | same reasoning |
| `.+`     | `.+`            | always tokenized after a space or `)` or `]` in canonical style |
| `.-`     | `.-`            | same                                     |
| `.*`     | `.*`            | same                                     |
| `./`     | `./`            | same                                     |
| `.^`     | `.^`            |                                          |
| `.**`    | `.**`           |                                          |
| `\|>`    | ` \|> ` (spaces required) | reads as pipe in every Julia-aware tokenizer |
| `>>`     | ` >> ` (spaces required) | otherwise risks fusing into `>>=` or shift |
| `<-`     | ` <- ` (spaces required) | distinct from `<=` and `->`              |
| `?`      | `?`             | postfix only; the lexer rejects `??`     |
| `~`      | `~`             | legal inside shape literals; also a prefix bitwise-NOT in expression position (`~x` on an integer scalar) |
| `<<`     | `<<`            | left shift; always 2 tokens             |
| `..` `..=` | `..` / `..=` | distinct from `.+`, `.*` by following char |

**Required spaces** above are part of the canonical syntax in this
spec version. The lexer accepts the no-space form but `dmcfmt` (the
canonical formatter, future tool) inserts spaces to match the table.

---

## 3. Alternate spellings

Each of these is **legal** input but rewritten by `dmcfmt`:

| Alt          | Canonical |
| ------------ | --------- |
| `relu(x)`                   | `\> x`  |

(`transpose(A)`, `matmul(A, B)`, and `pipe(x, f)` were earlier listed
here as legal alternates; they are undefined identifiers — `A'`, `A @ B`,
and `x \|> f` are the only spellings.)
| `x \|> f` (bare, no backslash) | `x \|> f` |

Bare `|>` (without the leading backslash) tokenizes identically to `\|>`
and is accepted by the lexer; `dmcfmt` normalizes it to the canonical
`\|>` form.

These exist for human-typed code and search-engine indexing.
Production source is canonical.

---

## 4. Keywords

All keywords from `docs/SPEC.md §2.2` (Keywords) are pure ASCII lowercase. They split
to single tokens under every BPE we have surveyed. The list will not
be extended with non-ASCII keywords.

The `extern` keyword tokenizes to a single token in every surveyed BPE.
The ABI-string form `extern "cuda"` and `extern "hip"` relies on the
standard string-literal lex path; no new lexical state is introduced.

---

## 5. Comments

`#` is the line-comment marker — a single token under every BPE worth
caring about. The block-comment form `#{ ... }#` adds two characters at
each boundary; in practice this tokenizes to 2 tokens at each end and
is fine.

---

## 5b. Byte literals

`b'x'` tokenizes to `IntLit(x as i64)`, where `x` is a single ASCII
character or escape sequence. Supported escapes: `\n`, `\r`, `\t`,
`\\`, `\'`, `\"`, `\0`. The lexer disambiguates `b'` (a byte literal
opening) from `b` followed by `'` (transpose of a tensor named `b`) by
lookahead: if the character after `b'` is a printable ASCII character or
a `\` escape followed by a closing `'`, it is a byte literal; otherwise
`b` is an identifier and `'` is the postfix transpose operator.

---

## 6. Identifiers

Identifiers should be:

- Lowercase `snake_case` for functions and bindings.
- `PascalCase` for types and `model` declarations.
- ASCII when possible; Unicode XID is permitted but tokenizer-hostile.

This is convention. The compiler does not enforce it.

---

## 7. Audit procedure

When proposing a new operator or directive:

1. Tokenize the canonical byte sequence under the current snapshot of
   the major BPE vocabularies (this tooling does not exist yet).
2. Reject if the operator splits in a way that fuses with a likely
   adjacent identifier (e.g., a hypothetical `.@` would fuse with
   `.@gradient` calls).
3. Reject if the operator collides with an existing operator under
   any reasonable preceding-character context.

The audit log lives in this document. When 0.1 ships, the audit log
must cover all operators in `docs/OPERATORS.md`.

---

## 8a. Tensor literal size and BPE stability

A tensor literal `[1.0, 2.0, ..., 1000.0]` with hundreds of elements
tokenizes to hundreds or thousands of tokens and is hostile to every
language-model consumer of the source. It also defeats the "fits on one
screen" readability contract.

The normative limit is **256 elements** per tensor literal. Literals larger
than this are accepted in the current release but produce a warning; a
future release will make them a hard error. Use `forge.zeros`, `forge.ones`,
`forge.uninit`, or `vault.load` for bulk data — these are single tokens each,
and their shapes are expressed as static type arguments that are cheap to
tokenize.

---

## 8. Why this matters

A language that takes 4 tokens to write what it means is a language
the model has to "spend" attention on, every time. Brutalism applies
to the tokenizer too. Every saved token is real compute saved at
inference time across every program ever emitted.
