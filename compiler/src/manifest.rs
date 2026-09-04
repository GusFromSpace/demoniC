/// #463: the compiler-side reader for `demoni.json`, the package manifest
/// specified in `docs/PACKAGES.md`. This is the first place the compiler
/// reads a file outside the source graph, so the boundary is stated here
/// and held: **the manifest configures advisory output only** — lints,
/// diagnostics, formatting — **never semantics** (PACKAGES.md §1). A
/// program compiles to the same result whether or not a manifest exists.
///
/// Consequences of that invariant:
///   - A missing, unreadable, or malformed manifest is silently ignored —
///     the lint stays off. `tools/validate_manifest.py` owns manifest
///     validation and its diagnostics; the compiler never duplicates them.
///   - Resolution walks up from the source file to the **nearest**
///     `demoni.json`; that manifest governs even when the key it is asked
///     for is absent. A nested package's manifest shadows its parent's.
///
/// The JSON reader below is deliberately minimal: full JSON grammar
/// (objects, arrays, strings with escapes, numbers, literals) with no
/// extensions, no spans, no error messages — parse failure is `None`.
/// A serde dependency for one integer key is not worth the build cost.

use std::path::{Path, PathBuf};

/// `lints.max_file_lines` from the nearest manifest, or `None` — because
/// there is no manifest up the tree, it has no such dial, or it does not
/// parse. `None` means the file-size lint does not fire (PACKAGES.md §4).
pub fn max_file_lines_for(source: &Path) -> Option<usize> {
    let manifest = nearest_manifest(source)?;
    let text = std::fs::read_to_string(manifest).ok()?;
    max_file_lines_in(&text)
}

/// Nearest `demoni.json`, walking up from the source file's directory.
fn nearest_manifest(source: &Path) -> Option<PathBuf> {
    // Canonicalize so a bare `dmc run foo.dmc` walks up from the real
    // location, not from an empty relative parent.
    let source = source
        .canonicalize()
        .ok()
        .or_else(|| std::env::current_dir().ok().map(|d| d.join(source)))?;
    for dir in source.ancestors().skip(1) {
        let candidate = dir.join("demoni.json");
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// `lints.max_file_lines` out of one manifest's text: present, a positive
/// integer → the dial; anything else → `None`.
pub fn max_file_lines_in(text: &str) -> Option<usize> {
    let root = Json::parse(text)?;
    let lints = root.get("lints")?;
    match lints.get("max_file_lines")? {
        // #518/#514: `Json::Int` only — a `Json::Num` here is a `1e1` or
        // `10.0` token, and `tools/validate_manifest.py` fails `make check`
        // on either. Arming the lint from one anyway was the divergence:
        // a manifest `make check` rejects must not still work for the
        // compiler's own reader.
        Json::Int(n) if *n >= 1 => Some(*n as usize),
        _ => None,
    }
}

// ─── Minimal JSON ────────────────────────────────────────────────────────────

#[derive(Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    /// A number token with a `.` and/or an exponent — a JSON *float* literal,
    /// even when its value happens to be integral (`10.0`, `1e1`). Kept
    /// distinct from `Int` (#518/#514) so a manifest dial that wants "an
    /// integer" can require the written *token form*, matching
    /// `tools/validate_manifest.py`'s `isinstance(value, int)` (Python's
    /// `json` module makes the identical distinction — `10` decodes to
    /// `int`, `10.0`/`1e1` to `float`, regardless of value).
    Num(f64),
    /// A number token with neither a `.` nor an exponent — a JSON *integer*
    /// literal. See `Num`.
    Int(i64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn parse(text: &str) -> Option<Json> {
        let mut p = JsonParser { src: text.as_bytes(), pos: 0 };
        p.skip_ws();
        let v = p.value()?;
        p.skip_ws();
        if p.pos != p.src.len() { return None; } // trailing garbage
        Some(v)
    }

    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
}

impl JsonParser<'_> {
    fn peek(&self) -> Option<u8> { self.src.get(self.pos).copied() }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) { self.pos += 1; }
    }

    fn eat(&mut self, b: u8) -> Option<()> {
        if self.peek() == Some(b) { self.pos += 1; Some(()) } else { None }
    }

    fn eat_word(&mut self, w: &str) -> Option<()> {
        if self.src[self.pos..].starts_with(w.as_bytes()) {
            self.pos += w.len();
            Some(())
        } else { None }
    }

    fn value(&mut self) -> Option<Json> {
        match self.peek()? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Json::Str),
            b't' => { self.eat_word("true")?; Some(Json::Bool(true)) }
            b'f' => { self.eat_word("false")?; Some(Json::Bool(false)) }
            b'n' => { self.eat_word("null")?; Some(Json::Null) }
            b'-' | b'0'..=b'9' => self.number(),
            _ => None,
        }
    }

    fn object(&mut self) -> Option<Json> {
        self.eat(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.eat(b'}').is_some() { return Some(Json::Obj(pairs)); }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.eat(b':')?;
            self.skip_ws();
            pairs.push((key, self.value()?));
            self.skip_ws();
            if self.eat(b',').is_some() { continue; }
            self.eat(b'}')?;
            return Some(Json::Obj(pairs));
        }
    }

    fn array(&mut self) -> Option<Json> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.eat(b']').is_some() { return Some(Json::Arr(items)); }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            if self.eat(b',').is_some() { continue; }
            self.eat(b']')?;
            return Some(Json::Arr(items));
        }
    }

    fn string(&mut self) -> Option<String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek()? {
                b'"' => { self.pos += 1; return Some(out); }
                b'\\' => {
                    self.pos += 1;
                    match self.peek()? {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            self.pos += 1;
                            let hi = self.hex4()?;
                            // Surrogate pair: a low surrogate must follow.
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                self.eat(b'\\')?;
                                self.eat(b'u')?;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) { return None; }
                                0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                            } else {
                                hi
                            };
                            out.push(char::from_u32(cp)?);
                            continue; // pos already past the escape
                        }
                        _ => return None,
                    }
                    self.pos += 1;
                }
                _ => {
                    // Consume one UTF-8 scalar, not one byte.
                    let rest = std::str::from_utf8(&self.src[self.pos..]).ok()?;
                    let c = rest.chars().next()?;
                    out.push(c);
                    self.pos += c.len_utf8();
                }
            }
        }
    }

    /// Four hex digits, cursor left after them.
    fn hex4(&mut self) -> Option<u32> {
        let s = self.src.get(self.pos..self.pos + 4)?;
        let s = std::str::from_utf8(s).ok()?;
        let v = u32::from_str_radix(s, 16).ok()?;
        self.pos += 4;
        Some(v)
    }

    /// #518/#514: strict JSON number grammar — `-? INT FRAC? EXP?` where
    /// `INT` is `0` or a non-zero digit followed by more digits (no leading
    /// zeros: `007` is not a JSON number), `FRAC` is `.` plus one-or-more
    /// digits, and `EXP` is `[eE] [+-]? ` plus one-or-more digits. The
    /// previous version of this scanner accepted any run of
    /// `[0-9.eE+-]` — which read straight through a leading-zero `007` the
    /// grammar forbids (Rust's `f64::parse` accepts it as 7.0, so nothing
    /// downstream caught it either) — and, the leniency #514/#518 named
    /// directly, could not tell `10` from `10.0`/`1e1` at all: every number
    /// token became the same `Json::Num(f64)`, so `max_file_lines_in` had no
    /// way to require the plain-integer token `tools/validate_manifest.py`
    /// requires. Returns `Json::Int` for a token with no `.`/exponent,
    /// `Json::Num` otherwise — see the `Json` enum doc.
    fn number(&mut self) -> Option<Json> {
        let start = self.pos;
        if self.peek() == Some(b'-') { self.pos += 1; }
        match self.peek() {
            Some(b'0') => { self.pos += 1; }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
            }
            _ => return None,
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            let frac_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
            if self.pos == frac_start { return None; } // `.` needs >=1 digit
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) { self.pos += 1; }
            let exp_start = self.pos;
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
            if self.pos == exp_start { return None; } // exponent needs >=1 digit
        }
        let s = std::str::from_utf8(&self.src[start..self.pos]).ok()?;
        if is_float {
            s.parse::<f64>().ok().filter(|n| n.is_finite()).map(Json::Num)
        } else {
            s.parse::<i64>().ok().map(Json::Int)
        }
    }
}
