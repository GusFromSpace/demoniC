/// demoniC lexer — spec 0.0.4-draft
///
/// Companion to: GRAMMAR.ebnf §Lexical, SPEC.md §2, TOKENIZER.md
///
/// Token stream is fully eager (no lazy modes). The lexer is the only
/// pass that touches raw bytes; everything above operates on Tokens.

use std::fmt;

// ─── Token types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // ── Literals ──────────────────────────────────────────────────────────
    IntLit(i64),
    FloatLit(f64, Option<String>),
    StrLit(String),
    /// Char literal `c"x"` -- Unicode scalar value; type is `u32`.
    CharLit(char),
    True,
    False,
    Nil,

    // ── Keywords ──────────────────────────────────────────────────────────
    Fn,
    Let,
    Mut,
    Match,
    If,
    Else,
    For,
    While,
    Loop,
    Break,
    Continue,
    Return,
    Vault,
    Forge,
    Stream,
    View,
    Shape,
    Dtype,
    As,
    Model,
    Stage,
    SelfKw,  // `self` is a keyword, not an ident
    Type,
    Enum,
    Use,
    Pub,
    Extern,

    // ── Scalar type keywords ───────────────────────────────────────────────
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Int4, Int8,
    F16, Bf16, Tf32, F32, F64,
    Fp8E4M3, Fp8E5M2,
    Trit,
    Bool,
    Str,

    // ── Identifier ────────────────────────────────────────────────────────
    /// Includes trailing `!` if present (mutating-function convention)
    Ident(String),

    // ── Directives ────────────────────────────────────────────────────────
    At,           // `@` before ident — directive prefix

    // ── Operators (per TOKENIZER.md canonical table) ───────────────────────
    // Transpose / postfix
    Transpose,     // `'`
    Query,         // `?`   postfix option-propagation

    // Elementwise arithmetic
    DotAdd,        // `.+`
    DotSub,        // `.-`
    DotMul,        // `.*`
    DotDiv,        // `./`
    DotPow,        // `.^`
    DotPow2,       // `.**`

    // Elementwise comparison
    DotGt,         // `.>`
    DotLt,         // `.<`
    DotGe,         // `.>=`
    DotLe,         // `.<=`

    // Unary activation primitives (SPEC §2.5, TOKENIZER §2)
    ReLU,          // `\>`
    GeLU,          // `\<`

    // Bitwise operators
    Amp,           // `&`
    Bar,           // `|`
    LtLt,          // `<<`
    AmpEq,         // `&=`
    BarEq,         // `|=`
    CaretEq,       // `^=`

    // Pipe operators
    Pipe,          // `|>`
    RShift,        // `>>`  (bitwise right shift / pipeline fan-out)

    // Stream assignment
    StreamArrow,   // `<-`

    // Standard arithmetic
    Plus,          // `+`
    Minus,         // `-`
    Star,          // `*`
    Slash,         // `/`
    Percent,       // `%`
    Caret,         // `^`
    StarStar,      // `**`

    // Comparison
    EqEq,          // `==`
    BangEq,        // `!=`
    Lt,            // `<`
    Gt,            // `>`
    LtEq,          // `<=`
    GtEq,          // `>=`

    // Logic
    AndAnd,        // `&&`
    OrOr,          // `||`
    Bang,          // `!`

    // Assignment
    Eq,            // `=`
    ColonEq,       // `:=`
    PlusEq,        // `+=`
    MinusEq,       // `-=`
    StarEq,        // `*=`
    SlashEq,       // `/=`

    // Range / matmul / etc.
    DotDot,        // `..`
    DotDotEq,      // `..=`
    ColonColon,    // `::`
    Arrow,         // `->`
    FatArrow,      // `=>`

    // Matrix multiply
    #[allow(dead_code)]  // emitted by parser, not the lexer; kept for future backend use
    Matmul,        // `@`  (when not directly followed by ident — directive vs matmul
                    //       distinction is context-sensitive; parser resolves)

    // Shape literal
    Tilde,         // `~`  (only legal inside shape literals per spec)

    // ── Punctuation ────────────────────────────────────────────────────────
    LParen,        // `(`
    RParen,        // `)`
    LBracket,      // `[`
    RBracket,      // `]`
    LBrace,        // `{`
    RBrace,        // `}`
    Comma,         // `,`
    Semicolon,     // `;`
    Colon,         // `:`
    Dot,           // `.`  (field access)
    Newline,       // significant newline (statement terminator)

    // ── Meta ───────────────────────────────────────────────────────────────
    Eof,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub col: usize,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    /// Raw source slice (for error messages, not semantic use)
    pub raw: String,
}

// ─── Lexer error ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LexError {
    pub msg: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lex error at {}:{}: {}", self.line, self.col, self.msg)
    }
}

// ─── Lexer ───────────────────────────────────────────────────────────────────

pub struct Lexer<'src> {
    src: &'src [u8],
    pos: usize,
    line: usize,
    col: usize,
    /// True while we are inside balanced `( )` or `[ ]` — newlines are not
    /// significant there (SPEC §0: "neither is required inside `( )` or `[ ]`")
    paren_depth: usize,
    bracket_depth: usize,
}

impl<'src> Lexer<'src> {
    pub fn new(src: &'src str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
            paren_depth: 0,
            bracket_depth: 0,
        }
    }

    // ── Primitives ────────────────────────────────────────────────────────

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<u8> {
        self.src.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.src.get(self.pos).copied()?;
        self.pos += 1;
        if ch == b'\n' {
            self.line += 1;
            self.col = 1;
        } else if ch & 0xC0 != 0x80 {
            // #281: count columns in characters, not bytes — don't bump on
            // UTF-8 continuation bytes (0b10xx_xxxx), or every column after
            // a multi-byte char (idents/strings are UTF-8 per SPEC §0) is
            // wrong in diagnostics.
            self.col += 1;
        }
        Some(ch)
    }

    fn eat(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn span_from(&self, start: usize, start_line: usize, start_col: usize) -> Span {
        Span { start, end: self.pos, line: start_line, col: start_col }
    }

    fn raw_slice(&self, start: usize) -> String {
        String::from_utf8_lossy(&self.src[start..self.pos]).into_owned()
    }

    fn err(&self, msg: impl Into<String>) -> LexError {
        LexError { msg: msg.into(), line: self.line, col: self.col }
    }

    // ── Comments ──────────────────────────────────────────────────────────

    /// Consume a `#` line comment or `#{ ... }#` block comment.
    /// Nested block comments are supported (SPEC §2.1).
    fn skip_comment(&mut self) -> Result<(), LexError> {
        // Consume the `#`
        self.advance();

        if self.peek() == Some(b'{') {
            // Block comment — may nest
            self.advance(); // eat `{`
            let mut depth = 1usize;
            loop {
                match self.advance() {
                    None => return Err(self.err("unterminated block comment")),
                    Some(b'#') if self.peek() == Some(b'{') => {
                        self.advance();
                        depth += 1;
                    }
                    Some(b'}') if self.peek() == Some(b'#') => {
                        self.advance();
                        depth -= 1;
                        if depth == 0 { break; }
                    }
                    _ => {}
                }
            }
        } else {
            // Line comment — eat until newline (but don't consume the \n itself;
            // it might be a statement terminator)
            while let Some(c) = self.peek() {
                if c == b'\n' { break; }
                self.advance();
            }
        }
        Ok(())
    }

    // ── Whitespace ────────────────────────────────────────────────────────

    fn skip_horizontal_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == b' ' || c == b'\t' || c == b'\r' { self.advance(); } else { break; }
        }
    }

    // ── String literals ───────────────────────────────────────────────────

    fn lex_string(&mut self) -> Result<TokenKind, LexError> {
        // Opening `"` already consumed by caller
        let mut s = String::new();
        loop {
            match self.advance() {
                None | Some(b'\n') => return Err(self.err("unterminated string literal")),
                Some(b'"') => break,
                Some(b'\\') => {
                    let escaped = match self.advance() {
                        Some(b'n')  => '\n',
                        Some(b'r')  => '\r',
                        Some(b't')  => '\t',
                        Some(b'\\') => '\\',
                        Some(b'"')  => '"',
                        Some(b'0')  => '\0',
                        _ => return Err(self.err("invalid escape sequence")),
                    };
                    s.push(escaped);
                }
                Some(c) => {
                    if c < 0x80 {
                        s.push(c as char);
                    } else {
                        // Multi-byte UTF-8: collect continuation bytes and decode.
                        let mut buf = vec![c];
                        while let Some(next) = self.peek() {
                            if next & 0xC0 == 0x80 {
                                buf.push(next);
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        match std::str::from_utf8(&buf) {
                            Ok(ch_str) => s.push_str(ch_str),
                            Err(_) => return Err(self.err("invalid UTF-8 in string literal")),
                        }
                    }
                }
            }
        }
        Ok(TokenKind::StrLit(s))
    }

    // -- Char literals `c"x"`

    /// Lex a char literal body. Opening `"` already consumed.
    /// Exactly one Unicode scalar value must appear before the closing `"`.
    fn lex_char_lit(&mut self) -> Result<TokenKind, LexError> {
        let mut s = String::new();
        loop {
            match self.advance() {
                None | Some(b'\n') => return Err(self.err("unterminated char literal")),
                Some(b'"') => break,
                Some(b'\\') => {
                    let escaped = match self.advance() {
                        Some(b'n')  => '\n',
                        Some(b'r')  => '\r',
                        Some(b't')  => '\t',
                        Some(b'\\') => '\\',
                        Some(b'"')  => '"',
                        Some(b'0')  => '\0',
                        _ => return Err(self.err("invalid escape sequence in char literal")),
                    };
                    s.push(escaped);
                }
                Some(c) => {
                    if c < 0x80 {
                        s.push(c as char);
                    } else {
                        let mut buf = vec![c];
                        while let Some(next) = self.peek() {
                            if next & 0xC0 == 0x80 {
                                buf.push(next);
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        match std::str::from_utf8(&buf) {
                            Ok(ch_str) => s.push_str(ch_str),
                            Err(_) => return Err(self.err("invalid UTF-8 in char literal")),
                        }
                    }
                }
            }
        }
        let mut chars = s.chars();
        let ch = match chars.next() {
            Some(c) => c,
            None => return Err(self.err("char literal must contain exactly one character")),
        };
        if chars.next().is_some() {
            return Err(self.err("char literal must contain exactly one character"));
        }
        Ok(TokenKind::CharLit(ch))
    }

    /// Disambiguate `b'x'` (byte literal) from `b'` (transpose of a tensor named
    /// `b`). `self.pos` is just past the `b`. Only the unambiguous byte-literal
    /// shape fires — `b' <ascii> '` or an escape `b'\…'`; anything else (a bare
    /// `b'` followed by an operator, newline, EOF, or a multibyte char) stays a
    /// `b` identifier + transpose, so `a @ b'` is untouched.
    fn looks_like_byte_lit(&self) -> bool {
        if self.peek() != Some(b'\'') {
            return false;
        }
        match self.src.get(self.pos + 1).copied() {
            Some(b'\\') => true, // escape form; lex_byte_lit validates it
            Some(c) if c < 0x80 && c != b'\'' && c != b'\n' => {
                self.src.get(self.pos + 2) == Some(&b'\'')
            }
            _ => false,
        }
    }

    /// Lex a byte literal `'x'` (opening `'` already consumed) → `IntLit(byte)`.
    fn lex_byte_lit(&mut self) -> Result<TokenKind, LexError> {
        let val: u8 = match self.advance() {
            None | Some(b'\n') => return Err(self.err("unterminated byte literal")),
            Some(b'\'') => return Err(self.err("empty byte literal")),
            Some(b'\\') => match self.advance() {
                Some(b'n')  => b'\n',
                Some(b'r')  => b'\r',
                Some(b't')  => b'\t',
                Some(b'\\') => b'\\',
                Some(b'\'') => b'\'',
                Some(b'"')  => b'"',
                Some(b'0')  => 0,
                _ => return Err(self.err("invalid escape sequence in byte literal")),
            },
            Some(c) => c,
        };
        match self.advance() {
            Some(b'\'') => Ok(TokenKind::IntLit(val as i64)),
            _ => Err(self.err("byte literal must contain exactly one byte")),
        }
    }

    // -- Numeric literals ──────────────────────────────────────────────────

    fn lex_number(&mut self, first: u8) -> Result<TokenKind, LexError> {
        let start = self.pos - 1; // already advanced past `first`

        // Check for `0x` / `0b` prefix
        if first == b'0' {
            if self.peek() == Some(b'x') || self.peek() == Some(b'X') {
                self.advance(); // eat `x`
                // Grammar (#291.6): `"0x" hex_digit { hex_digit | "_" }` — a hex
                // digit must immediately follow `0x`. An underscore may only sit
                // *between* digits, so `0x_ff` is rejected (also catches `0x`).
                if !self.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                    return Err(self.err("hex literal must start with a hex digit"));
                }
                while let Some(c) = self.peek() {
                    if c.is_ascii_hexdigit() || c == b'_' { self.advance(); } else { break; }
                }
                let raw = self.raw_slice(start);
                let digits: String = raw[2..].chars().filter(|&c| c != '_').collect();
                // #282: parse as a 64-bit *bit pattern* — masks like
                // 0xffff_ffff_ffff_ffff or 0x8000_0000_0000_0000 must be
                // writable even though they exceed i64::MAX.
                let v = u64::from_str_radix(&digits, 16)
                    .map_err(|_| self.err("hex literal out of range"))? as i64;
                self.try_eat_int_suffix();
                return Ok(TokenKind::IntLit(v));
            } else if self.peek() == Some(b'b') || self.peek() == Some(b'B') {
                self.advance(); // eat `b`
                // Grammar (#291.6): `"0b" bin_digit { bin_digit | "_" }` — a
                // binary digit must immediately follow `0b`; `0b_11` is rejected
                // (also catches the empty `0b`).
                if !matches!(self.peek(), Some(b'0') | Some(b'1')) {
                    return Err(self.err("binary literal must start with 0 or 1"));
                }
                while let Some(c) = self.peek() {
                    if c == b'0' || c == b'1' || c == b'_' { self.advance(); } else { break; }
                }
                let raw = self.raw_slice(start);
                let digits: String = raw[2..].chars().filter(|&c| c != '_').collect();
                // #282: same bit-pattern rule as hex above.
                let v = u64::from_str_radix(&digits, 2)
                    .map_err(|_| self.err("binary literal out of range"))? as i64;
                self.try_eat_int_suffix();
                return Ok(TokenKind::IntLit(v));
            }
        }

        // Decimal integer or float
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'_' { self.advance(); } else { break; }
        }

        // Float if `.` follows (but not `..` or `.identifier`)
        let is_float = self.peek() == Some(b'.') && {
            let next = self.peek2();
            next.map_or(false, |c| c.is_ascii_digit() || c == b'_')
                || next == None  // `3.` is a float
        };

        // Also float if `e`/`E` exponent follows
        let has_exp = matches!(self.peek(), Some(b'e') | Some(b'E'));

        if is_float || has_exp {
            if is_float {
                self.advance(); // eat `.`
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() || c == b'_' { self.advance(); } else { break; }
                }
            }
            if matches!(self.peek(), Some(b'e') | Some(b'E')) {
                self.advance();
                if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.advance(); }
                let exp_start = self.pos;
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() { self.advance(); } else { break; }
                }
                if self.pos == exp_start {
                    return Err(self.err("empty exponent in float literal"));
                }
            }
            let suffix = self.try_eat_float_suffix().map(|s| s.to_string());
            let raw = self.raw_slice(start);
            // Strip type suffix and underscores for parsing
            let clean: String = raw.chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == 'E' || *c == '+' || *c == '-' || *c == '_')
                .filter(|c| *c != '_')
                .collect();
            let v: f64 = clean.parse().map_err(|_| self.err("invalid float literal"))?;
            return Ok(TokenKind::FloatLit(v, suffix));
        }

        let _suffix = self.try_eat_int_suffix();
        // #401: optional binary size suffix (K/M/G) for byte counts, e.g.
        // `@recompute(budget=4G)`. Consumed here so it doesn't split off as a
        // separate `Ident` token.
        let size_mult = self.try_eat_size_suffix();
        let raw = self.raw_slice(start);
        // Strip suffix and underscores
        let clean: String = raw.chars()
            .take_while(|c| c.is_ascii_digit() || *c == '_')
            .filter(|c| *c != '_')
            .collect();
        let v: i64 = clean.parse().map_err(|_| self.err("integer literal out of range"))?;
        let v = v.checked_mul(size_mult).ok_or_else(|| self.err("integer literal out of range"))?;
        Ok(TokenKind::IntLit(v))
    }

    /// Consume an optional binary size suffix and return its multiplier (1 if
    /// none). `K`/`M`/`G` = 1024^1/2/3 (byte counts, e.g. `@recompute(budget=4G)`);
    /// `Ki`/`Mi`/`Gi` are accepted as explicit-binary aliases. Like the type
    /// suffixes, the suffix must end at a token boundary, so `4Gx` stays
    /// `IntLit(4)` + `Ident("Gx")` rather than silently splitting.
    fn try_eat_size_suffix(&mut self) -> i64 {
        const KI: i64 = 1024;
        const MI: i64 = 1024 * 1024;
        const GI: i64 = 1024 * 1024 * 1024;
        // Longer aliases first so `Ki` is not matched as `K` + `i`.
        let opts: &[(&str, i64)] = &[
            ("Ki", KI), ("Mi", MI), ("Gi", GI),
            ("K", KI), ("M", MI), ("G", GI),
        ];
        for (s, mult) in opts {
            let bytes = s.as_bytes();
            if self.src[self.pos..].starts_with(bytes) {
                let next = self.src.get(self.pos + bytes.len()).copied();
                let at_boundary = next.map_or(true, |c| !c.is_ascii_alphanumeric() && c != b'_');
                if at_boundary {
                    for _ in 0..bytes.len() { self.advance(); }
                    return *mult;
                }
            }
        }
        1
    }

    // Consume optional integer type suffix (i8, i16, i32, i64, u8, u16, u32, u64)
    fn try_eat_int_suffix(&mut self) -> Option<&'static str> {
        // #280: match whole suffixes only (like try_eat_float_suffix). The
        // old scanner ate any leading `i`/`u` unconditionally, corrupting an
        // adjacent token: `1if` lexed as IntLit(1) + Ident("f"). A suffix
        // must also end at a token boundary so `1i32abc` stays IntLit(1) +
        // Ident("i32abc") rather than silently splitting.
        let suffixes = ["i16", "i32", "i64", "i8", "u16", "u32", "u64", "u8"];
        for s in &suffixes {
            let bytes = s.as_bytes();
            if self.src[self.pos..].starts_with(bytes) {
                let next = self.src.get(self.pos + bytes.len()).copied();
                let at_boundary = next.map_or(true, |c| !c.is_ascii_alphanumeric() && c != b'_');
                if at_boundary {
                    for _ in 0..bytes.len() { self.advance(); }
                    return Some(s);
                }
            }
        }
        None
    }

    fn try_eat_float_suffix(&mut self) -> Option<&'static str> {
        // f16 | bf16 | tf32 | f32 | f64 | fp8_e4m3 | fp8_e5m2
        let suffixes = ["fp8_e4m3", "fp8_e5m2", "bf16", "tf32", "f16", "f32", "f64"];
        for s in &suffixes {
            let bytes = s.as_bytes();
            if self.src[self.pos..].starts_with(bytes) {
                for _ in 0..bytes.len() { self.advance(); }
                return Some(s);
            }
        }
        None
    }

    // ── Float literal starting with `.` (`.5`, `.5f64`) ─────────────────

    fn lex_leading_dot_float(&mut self) -> Result<TokenKind, LexError> {
        // `.` already consumed, next char is a digit
        let start = self.pos - 1;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == b'_' { self.advance(); } else { break; }
        }
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            self.advance();
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.advance(); }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() { self.advance(); } else { break; }
            }
        }
        let suffix = self.try_eat_float_suffix().map(|s| s.to_string());
        let raw = self.raw_slice(start);
        let clean: String = raw.chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == 'e' || *c == 'E' || *c == '+' || *c == '-' || *c == '_')
            .filter(|c| *c != '_')
            .collect();
        let v: f64 = clean.parse().map_err(|_| self.err("invalid float literal"))?;
        Ok(TokenKind::FloatLit(v, suffix))
    }

    // ── Identifiers and keywords ──────────────────────────────────────────

    fn lex_ident_or_keyword(&mut self, _first: u8) -> TokenKind {
        let start = self.pos - 1;
        while let Some(c) = self.peek() {
            // Accept ASCII alphanumeric, underscore, and any byte that is part
            // of a multi-byte UTF-8 sequence (>= 0x80) for Unicode XID_Continue.
            if c.is_ascii_alphanumeric() || c == b'_' || c >= 0x80 { self.advance(); } else { break; }
        }
        // Check for trailing `!` (mutating function convention)
        let has_bang = self.peek() == Some(b'!');
        if has_bang { self.advance(); }

        let raw = self.raw_slice(start);

        // Keywords — strip trailing `!` for matching (it's only a name convention)
        let kw = if has_bang { &raw[..raw.len()-1] } else { raw.as_str() };

        match kw {
            "fn"       => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Fn },
            "let"      => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Let },
            "mut"      => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Mut },
            "match"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Match },
            "if"       => if has_bang { TokenKind::Ident(raw) } else { TokenKind::If },
            "else"     => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Else },
            "for"      => if has_bang { TokenKind::Ident(raw) } else { TokenKind::For },
            "while"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::While },
            "loop"     => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Loop },
            "break"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Break },
            "continue" => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Continue },
            "return"   => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Return },
            "vault"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Vault },
            "forge"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Forge },
            "stream"   => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Stream },
            "view"     => if has_bang { TokenKind::Ident(raw) } else { TokenKind::View },
            "shape"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Shape },
            "dtype"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Dtype },
            "as"       => if has_bang { TokenKind::Ident(raw) } else { TokenKind::As },
            "model"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Model },
            "stage"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Stage },
            "self"     => if has_bang { TokenKind::Ident(raw) } else { TokenKind::SelfKw },
            "type"     => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Type },
            "enum"     => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Enum },
            "use"      => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Use },
            "pub"      => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Pub },
            "extern"   => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Extern },
            "true"     => if has_bang { TokenKind::Ident(raw) } else { TokenKind::True },
            "false"    => if has_bang { TokenKind::Ident(raw) } else { TokenKind::False },
            "nil"      => if has_bang { TokenKind::Ident(raw) } else { TokenKind::Nil },
            // Scalar type names (also usable as types in type position, not reserved elsewhere)
            "i8"  => TokenKind::I8,   "i16" => TokenKind::I16,
            "i32" => TokenKind::I32,  "i64" => TokenKind::I64,
            "u8"  => TokenKind::U8,   "u16" => TokenKind::U16,
            "u32" => TokenKind::U32,  "u64" => TokenKind::U64,
            "int4"     => TokenKind::Int4,
            "int8"     => TokenKind::Int8,
            "f16"      => TokenKind::F16, "bf16" => TokenKind::Bf16,
            "tf32"     => TokenKind::Tf32, "f32" => TokenKind::F32,
            "f64"      => TokenKind::F64,
            "fp8_e4m3" => TokenKind::Fp8E4M3,
            "fp8_e5m2" => TokenKind::Fp8E5M2,
            "trit"     => TokenKind::Trit,
            "bool"     => TokenKind::Bool,
            "str"      => TokenKind::Str,
            _          => TokenKind::Ident(raw),
        }
    }

    // ── Main token dispatch ───────────────────────────────────────────────

    pub fn next_token(&mut self) -> Result<Token, LexError> {
        loop {
            // Skip horizontal whitespace (newlines handled specially)
            self.skip_horizontal_ws();

            let sp = self.pos;
            let sl = self.line;
            let sc = self.col;

            let ch = match self.peek() {
                None => {
                    return Ok(Token {
                        kind: TokenKind::Eof,
                        span: self.span_from(sp, sl, sc),
                        raw: String::new(),
                    });
                }
                Some(c) => { self.advance(); c }
            };

            let kind: TokenKind = match ch {
                // Comments
                b'#' => {
                    // put back the `#` advance
                    // We already advanced in `advance()`, un-do by adjusting pos
                    self.pos -= 1; self.col -= 1;
                    self.skip_comment()?;
                    continue; // restart token scan
                }

                // Newline — significant unless inside parens/brackets
                b'\n' => {
                    if self.paren_depth == 0 && self.bracket_depth == 0 {
                        TokenKind::Newline
                    } else {
                        continue; // insignificant inside balanced delimiters
                    }
                }

                // String literal
                b'"' => self.lex_string()?,

                // Numeric literal starting with digit
                c if c.is_ascii_digit() => self.lex_number(c)?,

                // Numeric literal starting with `.` — must be `.5` style
                b'.' => {
                    if self.peek().map_or(false, |c| c.is_ascii_digit()) {
                        self.lex_leading_dot_float()?
                    } else if self.peek() == Some(b'.') {
                        self.advance();
                        if self.eat(b'=') { TokenKind::DotDotEq } else { TokenKind::DotDot }
                    } else if self.peek() == Some(b'+') {
                        self.advance(); TokenKind::DotAdd
                    } else if self.peek() == Some(b'-') {
                        self.advance(); TokenKind::DotSub
                    } else if self.peek() == Some(b'*') {
                        self.advance();
                        if self.eat(b'*') { TokenKind::DotPow2 } else { TokenKind::DotMul }
                    } else if self.peek() == Some(b'/') {
                        self.advance(); TokenKind::DotDiv
                    } else if self.peek() == Some(b'^') {
                        self.advance(); TokenKind::DotPow
                    } else if self.peek() == Some(b'>') {
                        self.advance();
                        if self.eat(b'=') { TokenKind::DotGe } else { TokenKind::DotGt }
                    } else if self.peek() == Some(b'<') {
                        self.advance();
                        if self.eat(b'=') { TokenKind::DotLe } else { TokenKind::DotLt }
                    } else {
                        TokenKind::Dot
                    }
                }

                // Identifiers and keywords (ASCII and UTF-8 XID_Start)
                // Special case: `c"..."` is a char literal; bare `c` is an ident.
                b'c' if self.peek() == Some(b'"') => {
                    self.advance(); // eat the opening `"`
                    self.lex_char_lit()?
                }
                // `b'x'` — Rust-style byte literal → the byte's integer value
                // (#334: models echoing Rust/C reach for this). Guarded so it never
                // steals `b'` = transpose of a tensor named `b`, which is common in
                // ML code: only the unambiguous `b' <byte> '` / `b'\<esc>'` shape fires.
                b'b' if self.looks_like_byte_lit() => {
                    self.advance(); // eat the opening `'`
                    self.lex_byte_lit()?
                }
                c if c.is_ascii_alphabetic() || c == b'_' || c >= 0x80 => self.lex_ident_or_keyword(c),

                // `@` — directive prefix or matmul operator; parser disambiguates
                b'@' => TokenKind::At,

                // `'` — transpose (postfix)
                b'\'' => TokenKind::Transpose,

                // `?` — postfix option propagation
                b'?' => TokenKind::Query,

                // `~` — growable axis inside shapes
                b'~' => TokenKind::Tilde,

                // `\` — `\>` (ReLU), `\<` (GeLU), `\|>` (pipe canonical form per TOKENIZER.md §2)
                b'\\' => {
                    match self.peek() {
                        Some(b'>') => { self.advance(); TokenKind::ReLU }
                        Some(b'<') => { self.advance(); TokenKind::GeLU }
                        Some(b'|') if self.peek2() == Some(b'>') => {
                            self.advance(); // `|`
                            self.advance(); // `>`
                            TokenKind::Pipe
                        }
                        _ => return Err(self.err("unexpected `\\` — expected `\\>`, `\\<`, or `\\|>`")),
                    }
                }

                // `|` — `||`, `|>`, `|=`, or bitwise `|`
                b'|' => {
                    if self.eat(b'|') { TokenKind::OrOr }
                    else if self.eat(b'>') { TokenKind::Pipe }
                    else if self.eat(b'=') { TokenKind::BarEq }
                    else { TokenKind::Bar }
                }

                // `>` — `>=`, `>>`, `>`
                b'>' => {
                    if self.eat(b'=') { TokenKind::GtEq }
                    else if self.eat(b'>') { TokenKind::RShift }
                    else { TokenKind::Gt }
                }

                // `<` — `<=`, `<-`, `<<`, `<`
                b'<' => {
                    if self.eat(b'=') { TokenKind::LtEq }
                    else if self.eat(b'-') { TokenKind::StreamArrow }
                    else if self.eat(b'<') { TokenKind::LtLt }
                    else { TokenKind::Lt }
                }

                // `=` — `==`, `=>`
                b'=' => {
                    if self.eat(b'=') { TokenKind::EqEq }
                    else if self.eat(b'>') { TokenKind::FatArrow }
                    else { TokenKind::Eq }
                }

                // `!` — `!=` or unary `!`
                b'!' => {
                    if self.eat(b'=') { TokenKind::BangEq } else { TokenKind::Bang }
                }

                // `+` — `+=`
                b'+' => {
                    if self.eat(b'=') { TokenKind::PlusEq } else { TokenKind::Plus }
                }

                // `-` — `-=`, `->`
                b'-' => {
                    if self.eat(b'=') { TokenKind::MinusEq }
                    else if self.eat(b'>') { TokenKind::Arrow }
                    else { TokenKind::Minus }
                }

                // `*` — `*=`, `**`
                b'*' => {
                    if self.eat(b'=') { TokenKind::StarEq }
                    else if self.eat(b'*') { TokenKind::StarStar }
                    else { TokenKind::Star }
                }

                // `/` — `/=`
                b'/' => {
                    if self.eat(b'=') { TokenKind::SlashEq } else { TokenKind::Slash }
                }

                // `%`
                b'%' => TokenKind::Percent,

                // `^` — `^=` or `^` (bitwise XOR)
                b'^' => {
                    if self.eat(b'=') { TokenKind::CaretEq } else { TokenKind::Caret }
                }

                // `&` — `&&`, `&=`, or bitwise `&`
                b'&' => {
                    if self.eat(b'&') { TokenKind::AndAnd }
                    else if self.eat(b'=') { TokenKind::AmpEq }
                    else { TokenKind::Amp }
                }

                // `:` — `:=`, `::`, `:`
                b':' => {
                    if self.eat(b'=') { TokenKind::ColonEq }
                    else if self.eat(b':') { TokenKind::ColonColon }
                    else { TokenKind::Colon }
                }

                // Brackets / parens / braces
                b'(' => { self.paren_depth   += 1; TokenKind::LParen   }
                b')' => { self.paren_depth    = self.paren_depth.saturating_sub(1); TokenKind::RParen   }
                b'[' => { self.bracket_depth += 1; TokenKind::LBracket }
                b']' => { self.bracket_depth  = self.bracket_depth.saturating_sub(1); TokenKind::RBracket }
                b'{' => TokenKind::LBrace,
                b'}' => TokenKind::RBrace,

                b',' => TokenKind::Comma,
                b';' => TokenKind::Semicolon,

                other => {
                    let ch_str = if other < 0x80 {
                        format!("`{}`", other as char)
                    } else {
                        format!("`{}`", String::from_utf8_lossy(&[other]))
                    };
                    return Err(self.err(format!("unexpected character {}", ch_str)));
                }
            };

            let raw = self.raw_slice(sp);
            return Ok(Token { kind, span: self.span_from(sp, sl, sc), raw });
        }
    }

    /// Collect all tokens into a Vec. Returns on first error.
    pub fn tokenize(mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::new();
        loop {
            let tok = self.next_token()?;
            let is_eof = tok.kind == TokenKind::Eof;
            tokens.push(tok);
            if is_eof { break; }
        }
        Ok(tokens)
    }
}
