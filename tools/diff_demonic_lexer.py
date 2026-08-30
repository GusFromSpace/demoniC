#!/usr/bin/env python3
"""diff_demonic_lexer.py — differential test for the self-hosted demoniC lexer.

Runs every .dmc source through two lexers and compares, token by token, ALL of:

  * the raw slice   — the exact source bytes each token covers (boundaries),
  * the token kind  — what each lexer decided that slice *is*, and
  * the line:col    — where each lexer says the token starts.

  1. reference:   dmc --lex <file>                        (Rust `TokenKind`)
  2. self-hosted: dmc run examples/demonic_lexer.dmc <f>  (JSON rows)

The two sides use different vocabularies on purpose — the self-hosted lexer
emits string tags ("kw_fn", "type_i32", "dotle") and the reference emits Rust
`TokenKind` variants (`Fn`, `I32`, `DotLe`). KIND_MAP below is the explicit
bridge between them; a self-hosted tag with no entry is itself a failure.

WHAT THIS PROVES: for every file checked, the two lexers cut the source at the
same byte boundaries, agree on the classification of every token, and agree on
every token's reported source position.

WHAT THIS DOES NOT PROVE:
  * Nothing about token *payloads*. `IntLit(42, Some("i32"))` and `IntLit(7,
    None)` both normalize to `IntLit` here — the reference parses literals into
    values and suffixes, so decoded values and numeric suffixes are compared
    only insofar as they change the token's extent.
  * Nothing about *error* behaviour beyond the fact of rejection. A file the
    reference rejects yields an empty reference stream; the self-hosted dumper
    prints an empty array when its own `lex` returns an `Err`. So the two agree
    when both reject, and a length divergence flags a file only one rejects —
    but the diagnostics themselves (message, fault position) are not compared.
  * Nothing about constructs absent from the corpus. Coverage is whatever
    examples/ happens to contain, plus the small PROBES below for shapes the
    corpus does not reach.

    python3 tools/diff_demonic_lexer.py
    python3 tools/diff_demonic_lexer.py --coverage      # + unexercised mappings
    python3 tools/diff_demonic_lexer.py --dmc /path/to/dmc --file examples/x.dmc

This harness is not run by CI. As of #454 it exits 0 on the whole corpus plus
the probes below, so wiring it in is now a question of runtime, not of red.

Notes on parsing the reference: each `dmc --lex` line is
`line:col  {kind:?}  {raw:?}`. We pull the trailing Rust-debug-quoted raw with a
regex and un-escape it back to the original bytes (so unicode and embedded
quotes compare correctly); the `line:col` prefix is captured, and everything
between it and the raw is the kind, normalized to its bare variant name by
dropping any `(payload)`.

The self-hosted dumper emits `[kind, lo, hi, line, col]` rows. Three-element
`[kind, lo, hi]` rows are still accepted, and simply skip the position check —
so an older dumper does not turn into a wall of false failures.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"
SELF_LEXER = REPO / "examples" / "demonic_lexer.dmc"
LEXER_RS = REPO / "compiler" / "src" / "lexer.rs"

# Trailing Rust-debug-quoted raw on each `--lex` line: ... "the\traw".
RAW_RE = re.compile(r'"((?:[^"\\]|\\.)*)"\s*$')
# `line:col  ` prefix that precedes the kind field.
POS_RE = re.compile(r"^\s*(\d+):\s*(\d+)\s\s(.*)$")

# ── Kind vocabulary bridge ───────────────────────────────────────────────────
# self-hosted tag (examples/demonic_lexer.dmc, `tok()` call sites)
#   -> reference variant name (compiler/src/lexer.rs, `enum TokenKind`)
#
# The table is the specification of the correspondence, not a transcript of
# today's behaviour: an entry may exist for a kind the self-hosted lexer cannot
# yet emit, and --coverage names the ones nothing exercises.
KIND_MAP: dict[str, str] = {
    # ── literals ──
    "int": "IntLit",
    "float": "FloatLit",
    "str": "StrLit",
    "char": "CharLit",

    # ── keywords ──
    "kw_fn": "Fn",
    "kw_let": "Let",
    "kw_mut": "Mut",
    "kw_match": "Match",
    "kw_if": "If",
    "kw_else": "Else",
    "kw_for": "For",
    "kw_while": "While",
    "kw_loop": "Loop",
    "kw_break": "Break",
    "kw_continue": "Continue",
    "kw_return": "Return",
    "kw_vault": "Vault",
    "kw_forge": "Forge",
    "kw_stream": "Stream",
    "kw_view": "View",
    "kw_shape": "Shape",
    "kw_dtype": "Dtype",
    "kw_as": "As",
    "kw_model": "Model",
    "kw_stage": "Stage",
    "kw_self": "SelfKw",
    "kw_type": "Type",
    "kw_enum": "Enum",
    "kw_use": "Use",
    "kw_pub": "Pub",
    "kw_extern": "Extern",
    "kw_true": "True",
    "kw_false": "False",
    "kw_nil": "Nil",

    # ── scalar type keywords ──
    "type_i8": "I8", "type_i16": "I16", "type_i32": "I32", "type_i64": "I64",
    "type_u8": "U8", "type_u16": "U16", "type_u32": "U32", "type_u64": "U64",
    "type_int4": "Int4", "type_int8": "Int8",
    "type_f16": "F16", "type_bf16": "Bf16", "type_tf32": "Tf32",
    "type_f32": "F32", "type_f64": "F64",
    "type_fp8_e4m3": "Fp8E4M3", "type_fp8_e5m2": "Fp8E5M2",
    "type_trit": "Trit", "type_bool": "Bool", "type_str": "Str",

    # ── identifier / directive ──
    "ident": "Ident",
    "at": "At",

    # ── postfix ──
    "transpose": "Transpose",
    "query": "Query",

    # ── elementwise arithmetic ──
    "dotplus": "DotAdd",
    "dotminus": "DotSub",
    "dotstar": "DotMul",
    "dotslash": "DotDiv",
    "dotcaret": "DotPow",

    # ── elementwise comparison ──
    "dotgt": "DotGt",
    "dotlt": "DotLt",
    "dotge": "DotGe",
    "dotle": "DotLe",

    # ── activation primitives ──
    "relu": "ReLU",
    "gelu": "GeLU",

    # ── bitwise ──
    "amp": "Amp",
    "bar": "Bar",
    "lshift": "LtLt",
    "ampeq": "AmpEq",
    "bareq": "BarEq",
    "careteq": "CaretEq",

    # ── pipes / streams ──
    "pipe": "Pipe",          # both `|>` and the canonical `\|>`
    # `>>` — the right shift since #530. It was the pipe until #501 ruling S1a
    # and a lex error in between, which is why this row went away and is back.
    "rshift": "RShift",
    "stream": "StreamArrow",  # `<-` (distinct from the `stream` keyword above)

    # ── arithmetic ──
    "plus": "Plus",
    "minus": "Minus",
    "star": "Star",
    "slash": "Slash",
    "pct": "Percent",
    "caret": "Caret",
    "starstar": "StarStar",

    # ── comparison ──
    "eqeq": "EqEq",
    "neq": "BangEq",
    "lt": "Lt",
    "gt": "Gt",
    "le": "LtEq",
    "ge": "GtEq",

    # ── logic ──
    "andand": "AndAnd",
    "oror": "OrOr",
    "bang": "Bang",

    # ── assignment ──
    "eq": "Eq",
    "coloneq": "ColonEq",
    "pluseq": "PlusEq",
    "minuseq": "MinusEq",
    "stareq": "StarEq",
    "slasheq": "SlashEq",

    # ── ranges / arrows / paths ──
    "dotdot": "DotDot",
    "dotdoteq": "DotDotEq",
    "coloncolon": "ColonColon",
    "arrow": "Arrow",
    "fatarrow": "FatArrow",

    # ── shapes ──
    "tilde": "Tilde",

    # ── punctuation ──
    "lparen": "LParen",
    "rparen": "RParen",
    "lbrack": "LBracket",
    "rbrack": "RBracket",
    "lbrace": "LBrace",
    "rbrace": "RBrace",
    "comma": "Comma",
    "semi": "Semicolon",
    "colon": "Colon",
    "dot": "Dot",
    "newline": "Newline",

    # ── meta ──
    # Mapped for completeness but never compared: both dumpers stop at EOF, so
    # --coverage always reports "eof" as unexercised.
    "eof": "Eof",

    # NOTE: "bad" is deliberately unmapped, and as of #454 the self-hosted lexer
    # no longer emits it -- what the reference rejects, it now rejects too,
    # returning an `Err` instead of inventing a token. The entry stays absent so
    # that a reintroduced "bad" tag fails here rather than passing silently.
}

# Synthetic inputs for shapes the examples/ corpus does not exercise. Each is
# lexed by both front-ends exactly like a corpus file. Keep them minimal and
# valid — they only need to LEX, not to type-check or run.
PROBES: dict[str, str] = {
    # `b'x'` is a byte literal in the reference (looks_like_byte_lit /
    # lex_byte_lit, #334) but the corpus's only `b'` sits inside a comment, so
    # nothing here is covered by examples/.
    "byte_lit": (
        "fn main() -> nil {\n"
        "    let a = b'A'\n"
        "    let nl = b'\\n'\n"
        "    let q = b'\\''\n"
        "    nil\n"
        "}\n"
    ),
    # The disambiguation the byte literal must NOT break: `b` transposed.
    "byte_lit_vs_transpose": (
        "fn main() -> nil {\n"
        "    let c = a @ b'\n"
        "    nil\n"
        "}\n"
    ),
    # Operators, keywords and scalar types the corpus never reaches (see
    # --coverage). Lexically valid; it is never parsed, so it need not typecheck.
    "operator_and_type_spread": (
        "fn probe() -> nil {\n"
        "    a & b\n"
        "    a &= b\n"
        "    a | b\n"
        "    a |= b\n"
        "    a ^= b\n"
        "    x := 1\n"
        "    a .>= b\n"
        "    a?\n"
        "    a >> b\n"
        "    a << b\n"
        "    a /= b\n"
        "    \\< a\n"
        "    continue\n"
        "    loop\n"
        "    view\n"
        "    dtype\n"
        "    type\n"
        "    enum\n"
        "    let t: i8 = 0\n"
        "    let u: i16 = 0\n"
        "    let v: u16 = 0\n"
        "    let w: int4 = 0\n"
        "    let y: int8 = 0\n"
        "    let z: f16 = 0\n"
        "    let p: tf32 = 0\n"
        "    let q: trit = 0\n"
        "    let r: fp8_e4m3 = 0\n"
        "    let s: fp8_e5m2 = 0\n"
        "    nil\n"
        "}\n"
    ),
    # #501: `.**` is no longer an operator — the reference hard-errors on it.
    # The self-hosted lexer must reject too, not tokenize `.*` `*`: rejection
    # means an empty stream on BOTH sides, so a self-hosted stream with tokens
    # here is a LEN divergence. Guards against the second spelling creeping back.
    "dotstarstar_rejected": (
        "fn probe() -> nil {\n"
        "    a .** b\n"
        "    nil\n"
        "}\n"
    ),
    # #464: numeric suffixes are matched WHOLE and must end at a token
    # boundary, so a literal butted against a keyword or identifier yields two
    # tokens, not one. The corpus never writes a literal without a space after
    # it, which is exactly how the self-hosted lexer's greedy scanner survived.
    # Both halves are probed: the shapes that must NOT take a suffix, and the
    # suffixes that must still attach.
    "numeric_suffix_boundaries": (
        "fn probe() -> nil {\n"
        "    1if\n"
        "    1in\n"
        "    1i32abc\n"
        "    1i32_x\n"
        "    4Gx\n"
        "    4KiB\n"
        "    0xffK\n"
        "    1.0f32abc\n"
        "    1.0for\n"
        "    .5f64abc\n"
        "    nil\n"
        "}\n"
    ),
    "numeric_suffix_whole": (
        "fn probe() -> nil {\n"
        "    1i8 1i16 1i32 1i64 1u8 1u16 1u32 1u64\n"
        "    1.0f16 1.0f32 1.0f64 1.0bf16 1.0tf32\n"
        "    1.0fp8_e4m3 1.0fp8_e5m2\n"
        "    .5f64\n"
        "    4K 4M 4G 4Ki 4Mi 4Gi\n"
        "    0x65u8 0b1010u8\n"
        "    nil\n"
        "}\n"
    ),
}


def rust_variants() -> set[str] | None:
    """Variant names of `enum TokenKind`, for typo-checking KIND_MAP. None if
    lexer.rs is not where we expect (the harness still runs, just unchecked)."""
    try:
        src = LEXER_RS.read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"enum TokenKind \{(.*?)\n\}", src, re.S)
    if not m:
        return None
    body = re.sub(r"//.*", "", m.group(1))          # line comments
    body = re.sub(r"#\[[^\]]*\]", "", body)         # attributes
    body = re.sub(r"\([^)]*\)", "", body)           # variant payloads
    # Variants may share a line (`I8, I16, I32, I64,`), so split on commas too.
    return {p for p in (x.strip() for x in re.split(r"[,\n]", body))
            if re.fullmatch(r"[A-Z]\w*", p)}


def unescape(s: str) -> str:
    """Rust debug string body -> original UTF-8 text."""
    try:
        return (bytes(s, "utf-8").decode("unicode_escape")
                .encode("latin-1").decode("utf-8", "replace"))
    except Exception:
        return s


def ref_tokens(dmc: str, f: str) -> list[tuple[str, str, int, int]]:
    """[(kind_variant, raw, line, col)] from `dmc --lex` (EOF excluded, as it
    prints). An empty list means the reference REJECTED the file."""
    out = subprocess.run([dmc, "--lex", f], capture_output=True, text=True).stdout
    toks = []
    for line in out.splitlines():
        m = RAW_RE.search(line)
        if not m:
            continue
        pos = POS_RE.match(line[:m.start()])
        if not pos:
            continue
        # Drop any `(payload)`: `Ident("x")` -> `Ident`, `IntLit(1, None)` -> `IntLit`.
        kind = pos.group(3).rstrip().split("(", 1)[0]
        toks.append((kind, unescape(m.group(1)), int(pos.group(1)), int(pos.group(2))))
    return toks


def self_tokens(dmc: str, f: str) -> list[tuple[str, int, int, int | None, int | None]] | None:
    """[(kind_tag, lo, hi, line, col)] from the self-hosted lexer's JSON dump.
    Rows may be `[kind, lo, hi]` (older dumper, no positions -> line/col None)
    or `[kind, lo, hi, line, col]`."""
    p = subprocess.run([dmc, "run", str(SELF_LEXER), f], capture_output=True, text=True)
    try:
        rows = json.loads(p.stdout)
    except json.JSONDecodeError:
        return None
    out = []
    for row in rows:
        if not isinstance(row, list):
            return None
        if len(row) == 3:
            out.append((row[0], row[1], row[2], None, None))
        elif len(row) == 5:
            out.append(tuple(row))
        else:
            return None
    return out


class Divergence:
    """One disagreement, in a form that can be aggregated across files."""

    def __init__(self, kind: str, idx: int, lo: int, hi: int,
                 text: str, ref: str, slf: str):
        self.kind = kind      # "RAW" | "KIND" | "POS" | "LEN"
        self.idx = idx
        self.lo, self.hi = lo, hi
        self.text = text
        self.ref, self.slf = ref, slf

    def __str__(self) -> str:
        where = f"tok#{self.idx} [{self.lo}..{self.hi}] {self.text!r}"
        return f"{self.kind:4} {where}: ref={self.ref} self={self.slf}"


def compare(src: bytes, ref: list[tuple[str, str, int, int]],
            slf: list[tuple[str, int, int, int | None, int | None]]) -> list[Divergence]:
    """Compare both streams. A raw divergence desynchronizes the streams, so we
    stop there; kind- and position-only divergences leave them aligned, so we
    keep going and collect them all."""
    out: list[Divergence] = []
    for i in range(min(len(ref), len(slf))):
        r_kind, r_raw, r_line, r_col = ref[i]
        s_tag, lo, hi, s_line, s_col = slf[i]
        s_raw = src[lo:hi].decode("utf-8", "replace")
        if r_raw != s_raw:
            out.append(Divergence("RAW", i, lo, hi, s_raw,
                                  f"{r_raw!r} ({r_kind})", f"{s_raw!r} ({s_tag})"))
            return out  # boundaries diverged; everything after is shifted noise
        expect = KIND_MAP.get(s_tag)
        if expect is None:
            out.append(Divergence("KIND", i, lo, hi, s_raw,
                                  r_kind, f"{s_tag} <no KIND_MAP entry>"))
        elif expect != r_kind:
            out.append(Divergence("KIND", i, lo, hi, s_raw,
                                  r_kind, f"{s_tag} -> {expect}"))
        if s_line is not None and (s_line, s_col) != (r_line, r_col):
            out.append(Divergence("POS", i, lo, hi, s_raw,
                                  f"{r_line}:{r_col}", f"{s_line}:{s_col}"))
    if len(ref) != len(slf):
        i = min(len(ref), len(slf))
        tail = ref[i][1] if i < len(ref) else src[slf[i][1]:slf[i][2]].decode("utf-8", "replace")
        lo = slf[i][1] if i < len(slf) else len(src)
        out.append(Divergence("LEN", i, lo, lo, tail,
                              f"{len(ref)} tokens", f"{len(slf)} tokens"))
    return out


def check(dmc: str, label: str, path: str, src: bytes,
          max_report: int, seen: set[str]) -> tuple[bool, list[Divergence]]:
    ref = ref_tokens(dmc, path)
    slf = self_tokens(dmc, path)
    if slf is None:
        print(f"FAIL {label}: self-hosted lexer produced no/invalid JSON")
        return False, []
    seen.update(row[0] for row in slf)
    divs = compare(src, ref, slf)
    if not divs:
        return True, []
    print(f"FAIL {label}: {len(divs)} divergence(s)")
    for d in divs[:max_report]:
        print(f"       {d}")
    if len(divs) > max_report:
        print(f"       ... and {len(divs) - max_report} more")
    return False, divs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmc", default=str(DEFAULT_DMC))
    ap.add_argument("--file", help="check a single file instead of the corpus")
    ap.add_argument("--include-probes", action="store_true",
                    help="also check tests/spec_probes/*.dmc if present")
    ap.add_argument("--no-synthetic", action="store_true",
                    help="skip the built-in PROBES snippets")
    ap.add_argument("--max-report", type=int, default=5,
                    help="divergences printed per file (default 5)")
    ap.add_argument("--coverage", action="store_true",
                    help="also list KIND_MAP entries no input exercised")
    args = ap.parse_args()

    unknown = set()
    variants = rust_variants()
    if variants:
        unknown = {v for v in KIND_MAP.values() if v not in variants}
        if unknown:
            print(f"WARN KIND_MAP names not found in TokenKind: {sorted(unknown)}")

    if args.file:
        files = [args.file]
    else:
        files = sorted(glob.glob(str(REPO / "examples" / "**" / "*.dmc"), recursive=True))
        if args.include_probes:
            files += sorted(glob.glob(str(REPO / "compiler" / "tests" / "spec_probes" / "**" / "*.dmc"),
                                      recursive=True))

    ok, failed, seen = 0, [], set()
    for f in files:
        if str(Path(f).resolve()) == str(SELF_LEXER.resolve()):
            continue
        try:  # --file may point outside the repo; fall back to the path as given
            rel = str(Path(f).resolve().relative_to(REPO))
        except ValueError:
            rel = f
        good, divs = check(args.dmc, rel, f, Path(f).read_bytes(),
                           args.max_report, seen)
        if good:
            ok += 1
        else:
            failed.append((rel, divs))

    if not args.file and not args.no_synthetic:
        with tempfile.TemporaryDirectory() as td:
            for name, text in PROBES.items():
                p = Path(td) / f"{name}.dmc"
                p.write_text(text, encoding="utf-8")
                label = f"<probe:{name}>"
                good, divs = check(args.dmc, label, str(p),
                                   p.read_bytes(), args.max_report, seen)
                if good:
                    ok += 1
                else:
                    failed.append((label, divs))

    # Aggregate the kind disagreements: this is the work list for the lexer.
    classes: dict[tuple[str, str], list[str]] = {}
    for label, divs in failed:
        for d in divs:
            if d.kind == "KIND":
                classes.setdefault((d.slf, d.ref), []).append(f"{label} {d.text!r}")
    if classes:
        print("\nKIND DISAGREEMENTS (self-hosted -> reference), by class:")
        for (slf, ref), hits in sorted(classes.items(), key=lambda kv: -len(kv[1])):
            print(f"  {slf:34} vs {ref:12} {len(hits):5} tok  e.g. {hits[0]}")

    if args.coverage:
        cold = sorted(set(KIND_MAP) - seen)
        print(f"\nCOVERAGE: {len(KIND_MAP) - len(cold)}/{len(KIND_MAP)} mapped "
              f"kinds exercised; never emitted by any input:")
        print("  " + (", ".join(cold) if cold else "(none)"))

    total = ok + len(failed)
    print(f"\nSELF-HOSTED LEXER DIFF (raws + kinds + line:col): "
          f"{ok}/{total} inputs MATCH")
    return 1 if (failed or unknown) else 0


if __name__ == "__main__":
    sys.exit(main())
