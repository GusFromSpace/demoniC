//! Process ports — the `PORTS.md §7.1` portability floor, shared by both backends.
//!
//! #402 put the process port in the interpreter; the JIT refused any program
//! that touched one. Both backends now drive the *same* code: this module owns
//! the child process, the line-oriented JSON framing, the argument-vector
//! envelope (`PORTS.md §2`), and the `PORTS.md §6` error tags. `interp.rs` and
//! `jit.rs` only translate between their own value representations and the
//! `Result<String, String>` this module returns — `Ok` is the canonical-JSON
//! result, `Err` is a tagged error string. Neither backend re-implements a byte
//! of the wire protocol, so they cannot drift on it.

use std::collections::HashMap;
use std::io::{BufRead, Write};

use crate::interp::{decode_kind, json_encode_str, json_encode_value, json_parse, Value};

/// #402: the python process-port harness (PORTS.md §7.1), passed to
/// `python3 -c`. Request per line: `{"name": str, "payload": str}` where
/// `name` is a dotted import path (`math.sqrt`, `json.dumps`, or a bare
/// builtin like `len`) and `payload` is JSON for the function's single
/// argument — `null` or empty calls with no arguments. Response per line:
/// `{"ok": result}` or `{"err": message}`.
///
/// The harness is also the python half of §3.2 copy mode: it rehydrates every
/// tensor envelope in the argument vector on the way in and re-encodes every
/// array in the result on the way out. With numpy installed a tensor arrives
/// as an `ndarray`, so the foreign function is ordinary numpy code; without
/// it, as a `_Tensor` carrying the same metadata and payload, so the round
/// trip still works and demoniC's own test gates do not need the dependency.
///
/// Neither path lets the host's own type inventory reach the wire: numpy has
/// no `bfloat16`, so a `bf16` envelope must widen to be computable there, and
/// the widening is undone on the way out. Otherwise the same program's
/// `dmc.echo` would return `bf16` or `f32` depending on whether numpy happened
/// to be installed on the far side.
const PY_PORT_HARNESS: &str = include_str!("py/port_harness.py");

/// #402: an open process port (PORTS.md §7.1) — a child runtime speaking
/// line-oriented JSON over its own stdin/stdout. The demoniC side holds the
/// pipes; the language side carries only an opaque `port#<id>:<lang>` handle.
struct PortProc {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    stdout: std::io::BufReader<std::process::ChildStdout>,
}

/// The open ports of one program run. Ids are never reused, so a call on a
/// closed handle reports `port-closed` instead of reaching a stranger.
///
/// Each backend keeps exactly one of these — the interpreter as a field, the
/// JIT as a thread-local behind its `extern "C"` helpers.
pub struct PortRegistry {
    ports: HashMap<i64, PortProc>,
    next_id: i64,
}

impl Default for PortRegistry {
    fn default() -> Self { Self::new() }
}

/// The opaque handle text a successful `port_open` hands back: `port#<id>:<lang>`.
/// Both backends carry it as their own opaque/str value and pass it straight back.
fn handle_text(id: i64, lang: &str) -> String {
    format!("port#{}:{}", id, lang)
}

/// #402: recover the registry id from an opaque `port#<id>:<lang>` handle.
/// `None` for anything that is not one — the caller reports that as the
/// hard "not a Port handle" error, never as a tagged port error.
pub fn handle_id(s: &str) -> Option<i64> {
    s.strip_prefix("port#")?.split(':').next()?.parse().ok()
}

/// The runtime name in a handle's text: `port#1:python` → `python`.
pub fn handle_lang(s: &str) -> Option<&str> {
    s.strip_prefix("port#")?.split_once(':').map(|(_, lang)| lang)
}

/// The `L` in a `Port[L]` type annotation, or `None` for a bare `Port` or a
/// type that is not a port at all — `is_port_type` answers that question.
///
/// `L` is written as a comptime identifier (SPEC §3.11), which reaches the AST
/// either as a named type or as a bare identifier expression depending on how
/// the annotation was parsed; both spellings mean the same thing here.
pub fn port_type_lang(ty: &crate::ast::Type) -> Option<String> {
    let crate::ast::Type::Named { name, args, .. } = ty else { return None };
    if name != "Port" { return None; }
    match args.first()? {
        crate::ast::TypeArg::Type(crate::ast::Type::Named { name, .. }) => Some(name.clone()),
        crate::ast::TypeArg::Expr(e) => match e.as_ref() {
            crate::ast::Expr::Ident(n, _) => Some(n.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// Is this annotation a port handle type (`Port` or `Port[L]`)?
pub fn is_port_type(ty: &crate::ast::Type) -> bool {
    matches!(ty, crate::ast::Type::Named { name, .. } if name == "Port")
}

/// The `Port[L]` mismatch message, shared so both backends word it identically.
///
/// `L` was decoration before this: a `Port[lua]` parameter took a python handle
/// without complaint in either backend, so the annotation implied a check that
/// did not exist. The handle knows its own runtime at run time, which is where
/// both backends now compare it.
pub fn lang_mismatch(want: &str, handle_text: &str) -> Option<String> {
    let got = handle_lang(handle_text)?;
    if got == want { return None; }
    Some(format!(
        "port lang mismatch: expected a `Port[{}]` handle, got one opened for `{}`",
        want, got))
}

impl PortRegistry {
    pub fn new() -> Self {
        PortRegistry { ports: HashMap::new(), next_id: 1 }
    }

    /// `port_open(lang)`. `Ok(handle_text)` or `Err("port-open: …")`.
    ///
    /// Only the python runtime is wired up so far; others report `port-open`
    /// ("runtime could not be started") rather than pretending.
    pub fn open(&mut self, lang: &str) -> Result<String, String> {
        if lang != "python" {
            return Err(format!(
                "port-open: unsupported runtime `{}` — the process-port floor implements `python`",
                lang));
        }
        match std::process::Command::new("python3")
            .args(["-u", "-c", PY_PORT_HARNESS])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
        {
            Ok(mut child) => {
                let stdin = child.stdin.take().expect("piped stdin");
                let stdout = std::io::BufReader::new(child.stdout.take().expect("piped stdout"));
                let id = self.next_id;
                self.next_id += 1;
                self.ports.insert(id, PortProc { child, stdin, stdout });
                Ok(handle_text(id, lang))
            }
            Err(e) => Err(format!("port-open: failed to start python3: {}", e)),
        }
    }

    /// `port_call(p, name, payload)`. `Ok(canonical JSON result)` or `Err(tag: …)`
    /// where the tag is one of `port-closed`, `port-protocol`, `port-call`.
    pub fn call(&mut self, id: i64, name: &str, payload: &str) -> Result<String, String> {
        let port = match self.ports.get_mut(&id) {
            Some(p) => p,
            None => return Err("port-closed: handle was already closed".to_string()),
        };
        let req = format!("{{\"name\":{},\"payload\":{}}}",
            json_encode_str(name), json_encode_str(payload));
        if let Err(e) = writeln!(port.stdin, "{}", req).and_then(|_| port.stdin.flush()) {
            return Err(format!("port-protocol: write failed: {}", e));
        }
        let mut line = String::new();
        match port.stdout.read_line(&mut line) {
            Ok(0) => return Err("port-protocol: runtime closed the pipe".to_string()),
            Ok(_) => {}
            Err(e) => return Err(format!("port-protocol: read failed: {}", e)),
        }
        match json_parse(line.trim()) {
            Ok(Value::Map(m)) => {
                let m = m.borrow();
                let tagged = |tag: &str, v: &Value| {
                    let msg = match v {
                        Value::Str(s) => s.clone(),
                        other => json_encode_value(other),
                    };
                    format!("{}: {}", tag, msg)
                };
                // `perr` is a malformed-payload signal from the harness
                // (PORTS.md §6 port-protocol); `err` is a foreign-runtime
                // failure (port-call). Distinguished so code can match tags.
                if let Some(perr) = m.get("perr") {
                    Err(tagged("port-protocol", perr))
                } else if let Some(err) = m.get("err") {
                    Err(tagged("port-call", err))
                } else if let Some(ok) = m.get("ok") {
                    // Re-encode through the crate's canonical JSON writer,
                    // so the result str is canonical regardless of the
                    // foreign runtime's formatting.
                    Ok(json_encode_value(ok))
                } else {
                    Err("port-protocol: response has neither `ok` nor `err`".to_string())
                }
            }
            Ok(_) => Err("port-protocol: response is not a JSON object".to_string()),
            Err(e) => Err(format!("port-protocol: {}", e)),
        }
    }

    /// `port_close(p)`. `Ok(())` or `Err("port-closed: …")` for a double close.
    pub fn close(&mut self, id: i64) -> Result<(), String> {
        match self.ports.remove(&id) {
            Some(port) => {
                let PortProc { stdin, mut child, .. } = port;
                drop(stdin);               // EOF ends the harness loop
                let _ = child.wait();      // reap; exit status is not an error surface
                Ok(())
            }
            None => Err("port-closed: handle was already closed".to_string()),
        }
    }
}

// ─── §3.2 tensor copy mode ───────────────────────────────────────────────────
//
// A tensor does not become a JSON array (that would hide the allocation and
// erase the shape). It crosses as one canonical-JSON object — the *tensor
// envelope* — carrying the §3.2 metadata `{dtype, shape, layout}` plus the
// payload buffer as base64:
//
// ```json
// {"data":"AQAAAAAAAAAC…","dmc_tensor":1,"dtype":"i64",
//  "layout":"row_major","shape":[2,3]}
// ```
//
// Everything about that text lives in this module: the dtype table, the
// element widths, the base64 alphabet, the key order, and the decode rules.
// The two backends hand in their own storage (the interpreter's f64 array, the
// JIT's native bytes) and get the same bytes out, so copy mode cannot drift
// between them any more than the framing above can.

/// The element types the §3.2 wire format names. The **byte width here is the
/// width on the wire**, and it is the whole point: #292 was a load that read
/// 2-byte bf16 into an f32-typed buffer, so every downstream op strode it 2×.
/// A wire `dtype` therefore always states the width of the bytes actually in
/// `data`, never the width of a demoniC annotation that computes f32-backed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WireDType { I64, I32, F64, F32, Bf16, F16, Bool }

impl WireDType {
    pub fn name(self) -> &'static str {
        match self {
            WireDType::I64 => "i64",   WireDType::I32 => "i32",
            WireDType::F64 => "f64",   WireDType::F32 => "f32",
            WireDType::Bf16 => "bf16", WireDType::F16 => "f16",
            WireDType::Bool => "bool",
        }
    }

    /// Bytes per element **on the wire**.
    pub fn bytes(self) -> usize {
        match self {
            WireDType::I64 | WireDType::F64 => 8,
            WireDType::I32 | WireDType::F32 => 4,
            WireDType::Bf16 | WireDType::F16 => 2,
            WireDType::Bool => 1,
        }
    }

    /// The integer spelling the JIT's `extern "C"` helpers pass. Codes are
    /// append-only: they are an internal ABI between `jit.rs` and this module,
    /// not part of the wire format (the wire carries `name()`).
    pub fn code(self) -> i64 {
        match self {
            WireDType::I64 => 0, WireDType::I32 => 1,
            WireDType::F64 => 2, WireDType::F32 => 3,
            WireDType::Bf16 => 4, WireDType::F16 => 5,
            WireDType::Bool => 6,
        }
    }

    pub fn from_code(c: i64) -> Option<Self> {
        Some(match c {
            0 => WireDType::I64, 1 => WireDType::I32,
            2 => WireDType::F64, 3 => WireDType::F32,
            4 => WireDType::Bf16, 5 => WireDType::F16,
            6 => WireDType::Bool,
            _ => return None,
        })
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Some(match s {
            "i64" => WireDType::I64, "i32" => WireDType::I32,
            "f64" => WireDType::F64, "f32" => WireDType::F32,
            "bf16" => WireDType::Bf16, "f16" => WireDType::F16,
            "bool" => WireDType::Bool,
            _ => return None,
        })
    }

    fn is_float(self) -> bool {
        matches!(self, WireDType::F64 | WireDType::F32 | WireDType::Bf16 | WireDType::F16)
    }
}

/// The envelope's format version. A reader that does not know the version it
/// finds refuses the value rather than guessing at the field set — the
/// manifest rule (§5.1), not the descriptor's additive one.
pub const TENSOR_ENVELOPE_VERSION: i64 = 1;

/// The only `layout` version 1 defines: C-contiguous, last axis fastest.
pub const TENSOR_LAYOUT: &str = "row_major";

/// The envelope's discriminator key. A JSON object carrying it is a tensor;
/// one that does not is an ordinary object. A tag, not a shape heuristic:
/// nothing about `{dtype, shape, layout, data}` is unguessable, and a reader
/// that sniffed for those field names would claim someone else's object.
pub const TENSOR_TAG: &str = "dmc_tensor";

const B64: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// RFC 4648 §4 base64, standard alphabet, always padded. The payload buffer is
/// bytes and the transport is JSON text, so it has to be *some* text encoding;
/// base64 is the one that survives the round trip byte-exactly. A JSON number
/// array would not: it would re-print every float through a decimal writer and
/// hand back a different bit pattern than it was given.
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[(n >> 18) as usize & 63] as char);
        out.push(B64[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    out
}

fn b64_val(c: u8) -> Option<u32> {
    Some(match c {
        b'A'..=b'Z' => (c - b'A') as u32,
        b'a'..=b'z' => (c - b'a') as u32 + 26,
        b'0'..=b'9' => (c - b'0') as u32 + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return None,
    })
}

/// Strict base64 decode: padded, no whitespace, no alternate alphabet. Strict
/// because a decoder that accepts sloppy input makes the format two formats.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    if b.len() % 4 != 0 { return None; }
    let mut out = Vec::with_capacity(b.len() / 4 * 3);
    for chunk in b.chunks(4) {
        let pad = chunk.iter().filter(|&&c| c == b'=').count();
        if pad > 2 { return None; }
        // Padding is only ever at the very end of the whole string.
        if pad > 0 && !std::ptr::eq(chunk.as_ptr(), b[b.len() - 4..].as_ptr()) { return None; }
        if pad == 2 && chunk[2] != b'=' { return None; }
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            let v = if c == b'=' {
                if i < 4 - pad { return None; }
                0
            } else {
                b64_val(c)?
            };
            n = (n << 6) | v;
        }
        out.push((n >> 16) as u8);
        if pad < 2 { out.push((n >> 8) as u8); }
        if pad < 1 { out.push(n as u8); }
    }
    Some(out)
}

/// The most axes an envelope may carry. Not a JSON limit — a limit on how much
/// a reader must be willing to allocate before it has checked anything, which
/// is what a hostile `shape` array otherwise buys for free. Eight is past every
/// rank the language itself constructs.
pub const TENSOR_MAX_RANK: usize = 8;

/// Element count of a shape, or `None` on overflow, a non-positive extent, or a
/// rank past `TENSOR_MAX_RANK`.
fn shape_numel(shape: &[i64]) -> Option<usize> {
    if shape.is_empty() || shape.len() > TENSOR_MAX_RANK { return None; }
    let mut n: i64 = 1;
    for &d in shape {
        if d <= 0 { return None; }
        n = n.checked_mul(d)?;
    }
    usize::try_from(n).ok()
}

/// Build the §3.2 envelope text. Keys are emitted in the **canonical writer's
/// order** — lexicographic, the order `json_encode_value` sorts a map into —
/// so an envelope demoniC writes and one that comes back through `port_call`'s
/// re-encode are the same bytes.
fn envelope(dtype: WireDType, shape: &[i64], payload: &[u8]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!(
        "{{\"data\":{},\"{}\":{},\"dtype\":{},\"layout\":{},\"shape\":[{}]}}",
        json_encode_str(&base64_encode(payload)),
        TENSOR_TAG, TENSOR_ENVELOPE_VERSION,
        json_encode_str(dtype.name()),
        json_encode_str(TENSOR_LAYOUT),
        dims.join(","),
    )
}

/// The wire dtype a demoniC-produced tensor of storage type `store` crosses as.
///
/// Integer tensors always cross as `i64`. The interpreter has exactly one
/// integer tensor dtype and the JIT has two (`i32`/`i64`), so any narrower
/// choice would make the same program emit different bytes on the two
/// backends — the drift this module exists to prevent. Widening is lossless,
/// costs only payload size, and leaves `i32` a dtype a *foreign* producer
/// (numpy `int32`) may still send in.
pub fn wire_dtype_for(store: WireDType) -> WireDType {
    match store { WireDType::I32 => WireDType::I64, other => other }
}

/// Why a tensor cannot be written as an envelope at all.
///
/// Deliberately untagged. `port_tensor_encode` is a local value transform that
/// returns a `str` — no runtime is involved and there is no `Err` half to carry
/// a tag — so its failures surface as ordinary runtime errors, the way encoding
/// a `trit` tensor already does. §3.2 reserves `port-protocol` for an envelope
/// *sent* to a runtime, which is a later step and a different failure.
fn unencodable_shape(shape: &[i64]) -> String {
    format!(
        "shape {:?} has no envelope — `shape` is 1 to {} extents, every one positive \
         (PORTS.md §3.2)", shape, TENSOR_MAX_RANK)
}

/// Encode a tensor whose elements are already native little-endian bytes of
/// `store` width — the JIT's storage, which *is* the wire layout.
pub fn pack_from_raw(store: WireDType, shape: &[i64], bytes: &[u8]) -> Result<String, String> {
    let numel = shape_numel(shape).ok_or_else(|| unencodable_shape(shape))?;
    if bytes.len() != numel * store.bytes() {
        return Err(format!(
            "tensor payload is {} bytes but shape {:?} of `{}` needs {}",
            bytes.len(), shape, store.name(), numel * store.bytes()));
    }
    let wire = wire_dtype_for(store);
    if wire == store {
        return Ok(envelope(wire, shape, bytes));
    }
    let mut out = Vec::with_capacity(numel * wire.bytes());
    for i in 0..numel {
        let x = read_elem(store, &bytes[i * store.bytes()..]);
        write_elem(wire, x, &mut out);
    }
    Ok(envelope(wire, shape, &out))
}

/// Encode a tensor the interpreter holds as f64 elements tagged with `store`.
pub fn pack_from_f64(store: WireDType, shape: &[i64], vals: &[f64]) -> Result<String, String> {
    let numel = shape_numel(shape).ok_or_else(|| unencodable_shape(shape))?;
    if vals.len() != numel {
        return Err(format!(
            "tensor has {} elements but shape {:?} needs {}", vals.len(), shape, numel));
    }
    let wire = wire_dtype_for(store);
    let mut out = Vec::with_capacity(numel * wire.bytes());
    for &x in vals { write_elem(wire, x, &mut out); }
    Ok(envelope(wire, shape, &out))
}

/// One element out of a little-endian buffer, widened to the f64 pivot.
///
/// The pivot is only ever crossed on conversions that are exact in f64:
/// `i32 → i64` and every float widening. An identity decode is a byte copy and
/// never reaches here, which is what keeps a full-width `i64` exact.
fn read_elem(dt: WireDType, b: &[u8]) -> f64 {
    match dt {
        WireDType::I64 => i64::from_le_bytes(b[..8].try_into().unwrap()) as f64,
        WireDType::I32 => i32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
        WireDType::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()),
        WireDType::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
        WireDType::Bf16 => crate::jit::bf16_bits_to_f32(
            u16::from_le_bytes(b[..2].try_into().unwrap())) as f64,
        WireDType::F16 => crate::jit::f16_bits_to_f32(
            u16::from_le_bytes(b[..2].try_into().unwrap())) as f64,
        WireDType::Bool => if b[0] != 0 { 1.0 } else { 0.0 },
    }
}

/// Append one element to a little-endian buffer.
fn write_elem(dt: WireDType, x: f64, out: &mut Vec<u8>) {
    match dt {
        WireDType::I64 => out.extend_from_slice(&(x as i64).to_le_bytes()),
        WireDType::I32 => out.extend_from_slice(&(x as i32).to_le_bytes()),
        WireDType::F64 => out.extend_from_slice(&x.to_le_bytes()),
        WireDType::F32 => out.extend_from_slice(&(x as f32).to_le_bytes()),
        WireDType::Bf16 => out.extend_from_slice(
            &(((x as f32).to_bits() >> 16) as u16).to_le_bytes()),
        WireDType::F16 => out.extend_from_slice(&f32_to_f16_bits(x as f32).to_le_bytes()),
        WireDType::Bool => out.push(if x != 0.0 { 1 } else { 0 }),
    }
}

/// Narrow an f32 to IEEE 754 half bits (round-to-nearest-even). Only reached
/// when a caller asks for an `f16` payload, never on a decode.
fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        if mant == 0 {
            return sign | 0x7c00; // ±inf
        }
        // NaN. Carry the payload across instead of flattening every NaN to one
        // pattern: `| 0x0200` mapped all 2046 half NaNs onto 0x7e00 / 0xfe00,
        // so 2044 of them did not survive a round trip (#534, item 2 — the
        // half of the domain a sampled test never looks at).
        //
        // The half's 10 mantissa bits are the single's top 10, which keeps the
        // quiet bit in place and so keeps a quiet NaN quiet. Truncation can
        // leave all ten zero — the payload lived below the cut — and a zero
        // mantissa here would silently become an infinity, so that case sets
        // the quiet bit to stay a NaN. The value is the same NaN either way;
        // only the payload of an already-payload-losing narrowing differs.
        let payload = (mant >> 13) as u16;
        return sign | 0x7c00 | if payload != 0 { payload } else { 0x0200 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f { return sign | 0x7c00; }          // overflow → ±inf
    if e <= 0 {
        if e < -10 { return sign; }                  // underflow → ±0
        let m = (mant | 0x0080_0000) >> (1 - e) as u32;
        let round = ((m & 0x1fff) > 0x1000) as u32
            | (((m & 0x3fff) == 0x3000) as u32);     // ties-to-even
        return sign | (((m >> 13) + round) as u16);
    }
    let round = ((mant & 0x1fff) > 0x1000) as u32
        | (((mant & 0x3fff) == 0x3000) as u32);
    sign | ((((e as u32) << 10) | (mant >> 13)) as u16) + round as u16
}

/// May a value that arrived as `wire` be read into a `want` tensor?
///
/// The §3.1 rule, applied to element types: a typed decode never coerces
/// across a kind and never narrows. An integer wire dtype is readable as a
/// wider integer, a float as a wider float, `bool` only as `bool`. `1` is
/// still not `true`, and an f64 payload is still not an f32 tensor.
fn decode_widens(wire: WireDType, want: WireDType) -> bool {
    if wire == want { return true; }
    match want {
        WireDType::I64 => wire == WireDType::I32,
        WireDType::F64 => wire.is_float(),
        WireDType::F32 => matches!(wire, WireDType::Bf16 | WireDType::F16),
        _ => false,
    }
}

/// Decode a §3.2 envelope into `want`-width little-endian bytes for a tensor
/// of exactly `want_shape`.
///
/// Errors carry the §3.1 decode tags, not the `port-` family: this runs after
/// the call returned, so a bad envelope is a contract failure by the caller or
/// the foreign runtime, not a port that failed. `decode-parse` means the text
/// is not JSON; everything else — not an envelope, unknown version, wrong
/// dtype, wrong shape, bad base64, short payload — is `decode-type`.
pub fn unpack_raw(text: &str, want: WireDType, want_shape: &[i64]) -> Result<Vec<u8>, String> {
    let numel = shape_numel(want_shape)
        .ok_or_else(|| format!("decode-type: undecodable tensor shape {:?}", want_shape))?;
    let v = json_parse(text).map_err(|e| format!("decode-parse: {}", e))?;
    let Value::Map(m) = &v else {
        return Err(format!(
            "decode-type: expected a `{}` tensor envelope, got {}",
            TENSOR_TAG, decode_kind(&v)));
    };
    let m = m.borrow();
    match m.get(TENSOR_TAG) {
        Some(Value::Int(v, _)) if *v == TENSOR_ENVELOPE_VERSION => {}
        Some(Value::Int(v, _)) => return Err(format!(
            "decode-type: tensor envelope version {} is not {}", v, TENSOR_ENVELOPE_VERSION)),
        // The tag is present but is not a version number. Say that: `"1"` is a
        // str, and reporting a plain JSON object would deny the reader the one
        // fact it needs — the object *is* claiming to be an envelope.
        Some(other) => return Err(format!(
            "decode-type: tensor envelope version must be an integer, got {}",
            decode_kind(other))),
        None => return Err(format!(
            "decode-type: expected a `{}` tensor envelope, got a plain JSON object", TENSOR_TAG)),
    }
    // §3.2's manifest rule reaches the fields, not only the version: a key
    // this version does not define is a document written to a format this
    // reader does not know, and reading it anyway would be silently reading
    // the wrong *numbers*. AGENTS.md §2.6 — take the less forgiving option.
    // Sorted, so the message is the same on both backends and on every run:
    // the underlying map has no order of its own.
    let mut unknown: Vec<&str> = m.keys()
        .map(|k| k.as_str())
        .filter(|k| !matches!(*k, "data" | "dtype" | "layout" | "shape") && *k != TENSOR_TAG)
        .collect();
    if !unknown.is_empty() {
        unknown.sort_unstable();
        return Err(format!(
            "decode-type: tensor envelope has unknown field(s) `{}`", unknown.join("`, `")));
    }
    let layout = match m.get("layout") {
        Some(Value::Str(s)) => s.clone(),
        _ => return Err("decode-type: tensor envelope has no `layout` str".to_string()),
    };
    if layout != TENSOR_LAYOUT {
        return Err(format!(
            "decode-type: tensor layout `{}` is not `{}`", layout, TENSOR_LAYOUT));
    }
    let wire = match m.get("dtype") {
        Some(Value::Str(s)) => WireDType::from_name(s).ok_or_else(|| format!(
            "decode-type: unknown tensor dtype `{}`", s))?,
        _ => return Err("decode-type: tensor envelope has no `dtype` str".to_string()),
    };
    if !decode_widens(wire, want) {
        return Err(format!(
            "decode-type: expected a `{}` tensor, got `{}`", want.name(), wire.name()));
    }
    let shape: Vec<i64> = match m.get("shape") {
        Some(Value::List(xs)) => {
            let mut dims = Vec::with_capacity(xs.len());
            for x in xs.iter() {
                match x {
                    Value::Int(d, _) if *d > 0 => dims.push(*d),
                    _ => return Err(
                        "decode-type: tensor `shape` must be positive integers".to_string()),
                }
            }
            dims
        }
        _ => return Err("decode-type: tensor envelope has no `shape` array".to_string()),
    };
    if shape != want_shape {
        return Err(format!(
            "decode-type: expected tensor shape {:?}, got {:?}", want_shape, shape));
    }
    let data = match m.get("data") {
        Some(Value::Str(s)) => base64_decode(s)
            .ok_or_else(|| "decode-type: tensor `data` is not valid base64".to_string())?,
        _ => return Err("decode-type: tensor envelope has no `data` str".to_string()),
    };
    if data.len() != numel * wire.bytes() {
        return Err(format!(
            "decode-type: tensor payload is {} bytes but `{}`{:?} needs {}",
            data.len(), wire.name(), shape, numel * wire.bytes()));
    }
    if wire == want { return Ok(data); }
    let mut out = Vec::with_capacity(numel * want.bytes());
    for i in 0..numel {
        write_elem(want, read_elem(wire, &data[i * wire.bytes()..]), &mut out);
    }
    Ok(out)
}

/// Read a little-endian payload of `dt` elements back out as f64 — the
/// interpreter's tensor storage. The JIT keeps the bytes as they are.
pub fn raw_to_f64(dt: WireDType, bytes: &[u8]) -> Vec<f64> {
    (0..bytes.len() / dt.bytes())
        .map(|i| read_elem(dt, &bytes[i * dt.bytes()..]))
        .collect()
}

/// Is python3 on PATH? The process port is the only runtime the floor
/// implements, so every protocol test needs it; they skip (loudly) without it.
/// Shared with `jit`'s test module so both report the skip the same way.
#[cfg(test)]
pub(crate) fn have_python() -> bool {
    std::process::Command::new("python3")
        .arg("-c").arg("pass")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Is numpy importable by the python3 on PATH? The harness has two code paths
/// and this decides which one the ambient run takes; a test that asserts the
/// *difference* between them needs to know.
#[cfg(test)]
fn have_numpy() -> bool {
    std::process::Command::new("python3")
        .arg("-c").arg("import numpy")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Drive the real harness with a controlled `PYTHONPATH` and return each
/// response, re-encoded through the canonical writer exactly as
/// `PortRegistry::call` does — `Ok` for an `ok` payload, `Err` for the tagged
/// `perr`/`err` string, so a test can assert a refusal as well as a result.
///
/// Spawned directly rather than through `PortRegistry` because the point is to
/// vary the child's environment, and a test that mutated the process-wide
/// environment would race every other test in the binary.
///
/// Strictly lockstep — write one request, read its response, then the next —
/// which is also what `PortRegistry::call` does. Writing every request first
/// and reading afterwards deadlocks as soon as one response outgrows the pipe
/// buffer, and the whole-space bf16 sweep below is a 175 KB response.
#[cfg(test)]
fn drive_harness(
    pythonpath: Option<&std::path::Path>,
    reqs: &[(&str, String)],
) -> Vec<Result<String, String>> {
    let mut cmd = std::process::Command::new("python3");
    cmd.args(["-u", "-c", PY_PORT_HARNESS]);
    if let Some(p) = pythonpath { cmd.env("PYTHONPATH", p); }
    let mut child = cmd
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn python3");
    let mut stdin = child.stdin.take().expect("piped stdin");
    let mut stdout = std::io::BufReader::new(child.stdout.take().expect("piped stdout"));
    let mut out = Vec::with_capacity(reqs.len());
    for (name, payload) in reqs {
        writeln!(stdin, "{{\"name\":{},\"payload\":{}}}",
            json_encode_str(name), json_encode_str(payload)).expect("write request");
        stdin.flush().expect("flush request");
        let mut line = String::new();
        stdout.read_line(&mut line).expect("read response");
        let v = json_parse(line.trim()).unwrap_or_else(|e| panic!("{}: {}", e, line));
        let Value::Map(m) = &v else { panic!("not a response object: {}", line) };
        let m = m.borrow();
        out.push(match (m.get("ok"), m.get("perr"), m.get("err")) {
            (Some(ok), _, _) => Ok(json_encode_value(ok)),
            (_, Some(e), _) => Err(format!("port-protocol: {}", json_encode_value(e))),
            (_, _, Some(e)) => Err(format!("port-call: {}", json_encode_value(e))),
            _ => panic!("response has neither ok nor err: {}", line),
        });
    }
    drop(stdin);
    let _ = child.wait();
    out
}

/// A directory holding a `numpy.py` that refuses to import — the reviewer's
/// trick for forcing the harness's no-numpy path on a machine that has numpy.
#[cfg(test)]
fn numpy_blocking_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("dmc_nonumpy_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(dir.join("numpy.py"), "raise ImportError(\"no module named 'numpy'\")\n")
        .expect("write stub");
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_round_trips_through_its_text() {
        assert_eq!(handle_text(7, "python"), "port#7:python");
        assert_eq!(handle_id("port#7:python"), Some(7));
        // A str that is not a handle is not silently read as one.
        assert_eq!(handle_id("port#"), None);
        assert_eq!(handle_id("7:python"), None);
        assert_eq!(handle_id("port#x:python"), None);
        assert_eq!(handle_id(""), None);
    }

    #[test]
    fn unsupported_runtime_reports_port_open_verbatim() {
        let mut reg = PortRegistry::new();
        assert_eq!(
            reg.open("lua").unwrap_err(),
            "port-open: unsupported runtime `lua` — the process-port floor implements `python`",
        );
    }

    #[test]
    fn call_on_an_unknown_id_reports_port_closed_verbatim() {
        let mut reg = PortRegistry::new();
        assert_eq!(
            reg.call(1, "len", "[[1,2]]").unwrap_err(),
            "port-closed: handle was already closed",
        );
        assert_eq!(reg.close(1).unwrap_err(), "port-closed: handle was already closed");
    }

    #[test]
    fn ids_are_never_reused_after_close() {
        if !have_python() { eprintln!("skipped: python3 not on PATH"); return; }
        let mut reg = PortRegistry::new();
        let h1 = reg.open("python").unwrap();
        assert_eq!(h1, "port#1:python");
        reg.close(handle_id(&h1).unwrap()).unwrap();
        let h2 = reg.open("python").unwrap();
        assert_eq!(h2, "port#2:python");
        // The stale handle does not reach the new child.
        assert_eq!(
            reg.call(handle_id(&h1).unwrap(), "len", "[[1]]").unwrap_err(),
            "port-closed: handle was already closed",
        );
        reg.close(handle_id(&h2).unwrap()).unwrap();
    }

    #[test]
    fn the_four_wire_outcomes_carry_their_tags() {
        if !have_python() { eprintln!("skipped: python3 not on PATH"); return; }
        let mut reg = PortRegistry::new();
        let id = handle_id(&reg.open("python").unwrap()).unwrap();

        // ok: canonical JSON, re-encoded by demoniC's writer (1024.0 -> "1024").
        assert_eq!(reg.call(id, "math.pow", "[2, 10]").unwrap(), "1024");
        // no arguments at all.
        assert_eq!(reg.call(id, "list", "").unwrap(), "[]");
        // the {args, kwargs} envelope.
        assert_eq!(
            reg.call(id, "round", "{\"args\":[3.14159],\"kwargs\":{\"ndigits\":2}}").unwrap(),
            "3.14");
        // port-call: the foreign runtime raised.
        let e = reg.call(id, "math.sqrt", "[\"sixteen\"]").unwrap_err();
        assert!(e.starts_with("port-call: "), "{}", e);
        // port-protocol: a bare scalar is not an argument vector.
        assert_eq!(
            reg.call(id, "math.sqrt", "16").unwrap_err(),
            "port-protocol: payload must be a JSON array, an {args, kwargs} object, or null",
        );
        // The port survives both error kinds.
        assert_eq!(reg.call(id, "math.gcd", "[462, 1071]").unwrap(), "21");
        reg.close(id).unwrap();
    }

    /// A name carrying JSON metacharacters must reach the harness intact —
    /// the request line is built with the canonical string escaper, not by
    /// pasting the name between quotes.
    #[test]
    fn a_quote_in_the_name_does_not_break_the_frame() {
        if !have_python() { eprintln!("skipped: python3 not on PATH"); return; }
        let mut reg = PortRegistry::new();
        let id = handle_id(&reg.open("python").unwrap()).unwrap();
        let e = reg.call(id, "math.\"sqrt\\", "[]").unwrap_err();
        // Resolution fails inside python — a `port-call`, not a mangled frame.
        assert!(e.starts_with("port-call: "), "{}", e);
        // …and the port is still usable, i.e. the framing stayed in sync.
        assert_eq!(reg.call(id, "math.gcd", "[462, 1071]").unwrap(), "21");
        reg.close(id).unwrap();
    }

    /// A payload carrying a quote/newline crosses intact too: the harness sees
    /// the *string* `he said "hi"` plus a newline, so `len` counts 13.
    #[test]
    fn a_quote_in_the_payload_survives_the_envelope() {
        if !have_python() { eprintln!("skipped: python3 not on PATH"); return; }
        let mut reg = PortRegistry::new();
        let id = handle_id(&reg.open("python").unwrap()).unwrap();
        assert_eq!(reg.call(id, "len", "[\"he said \\\"hi\\\"\\n\"]").unwrap(), "13");
        reg.close(id).unwrap();
    }

    // ── §3.2 copy mode ──────────────────────────────────────────────────────

    /// The exact envelope text, byte for byte. This is the ABI: if it changes,
    /// every reader on the other side of a port changes with it, so it is
    /// pinned here rather than described.
    #[test]
    fn the_copy_mode_envelope_is_exactly_this_text() {
        assert_eq!(
            pack_from_f64(WireDType::I64, &[2, 3], &[1.0, 2.0, 3.0, -4.0, 5.0, 6.0]).unwrap(),
            "{\"data\":\"AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA/P////////8FAAAAAAAAAAYAAAAAAAAA\",\
             \"dmc_tensor\":1,\"dtype\":\"i64\",\"layout\":\"row_major\",\"shape\":[2,3]}",
        );
        // Keys are in the canonical writer's (lexicographic) order, so an
        // envelope that comes back through `port_call`'s re-encode is the same
        // bytes as one demoniC wrote.
        assert_eq!(
            json_encode_value(&json_parse(
                &pack_from_f64(WireDType::Bool, &[2], &[1.0, 0.0]).unwrap()).unwrap()),
            pack_from_f64(WireDType::Bool, &[2], &[1.0, 0.0]).unwrap(),
        );
    }

    /// The dtype on the wire and the width of the bytes in `data` agree, for
    /// every dtype. This is the #292 invariant: that bug was 2-byte bf16 read
    /// into an f32-typed buffer, and it is exactly what a `dtype` that lies
    /// about its payload width reintroduces at the port boundary.
    #[test]
    fn every_wire_dtype_payload_is_its_own_width() {
        for (dt, want_bytes) in [
            (WireDType::I64, 8), (WireDType::I32, 4),
            (WireDType::F64, 8), (WireDType::F32, 4),
            (WireDType::Bf16, 2), (WireDType::F16, 2),
            (WireDType::Bool, 1),
        ] {
            assert_eq!(dt.bytes(), want_bytes, "{}", dt.name());
            // 4 elements, so the base64 payload decodes to 4 * width bytes.
            let text = envelope(dt, &[4], &vec![0u8; 4 * dt.bytes()]);
            let v = json_parse(&text).unwrap();
            let Value::Map(m) = v else { panic!("envelope is not an object") };
            let m = m.borrow();
            let Some(Value::Str(name)) = m.get("dtype") else { panic!("no dtype") };
            let Some(Value::Str(data)) = m.get("data") else { panic!("no data") };
            assert_eq!(name, dt.name());
            assert_eq!(base64_decode(data).unwrap().len(), 4 * want_bytes, "{}", dt.name());
        }
    }

    /// An integer tensor always crosses as `i64`, whichever backend wrote it —
    /// the interpreter has one integer tensor dtype and the JIT has two.
    #[test]
    fn integer_tensors_cross_as_i64_whatever_the_backend_stores() {
        assert_eq!(wire_dtype_for(WireDType::I32), WireDType::I64);
        assert_eq!(wire_dtype_for(WireDType::I64), WireDType::I64);
        // A JIT `Tensor[i32, [2]]` (4-byte storage) and an interpreter integer
        // tensor of the same values produce identical bytes.
        let from_i32 = pack_from_raw(
            WireDType::I32, &[2], &[7, 0, 0, 0, 0xf9, 0xff, 0xff, 0xff]).unwrap();
        let from_f64 = pack_from_f64(WireDType::I64, &[2], &[7.0, -7.0]).unwrap();
        assert_eq!(from_i32, from_f64);
        // f32/f64/bool storage is already the wire width — a straight copy.
        assert_eq!(wire_dtype_for(WireDType::F32), WireDType::F32);
        assert_eq!(
            pack_from_raw(WireDType::F32, &[1], &1.5f32.to_le_bytes()).unwrap(),
            pack_from_f64(WireDType::F32, &[1], &[1.5]).unwrap());
    }

    /// Every dtype round-trips its own payload byte-for-byte, including the
    /// full-width `i64` that an f64 pivot would have rounded.
    #[test]
    fn a_round_trip_is_byte_identical() {
        let big = i64::MAX - 1;                 // 9223372036854775806
        let raw: Vec<u8> = [big, i64::MIN, 0].iter()
            .flat_map(|v| v.to_le_bytes()).collect();
        let text = pack_from_raw(WireDType::I64, &[3], &raw).unwrap();
        assert_eq!(unpack_raw(&text, WireDType::I64, &[3]).unwrap(), raw);

        for dt in [WireDType::F64, WireDType::F32, WireDType::Bf16,
                   WireDType::F16, WireDType::Bool] {
            let raw = vec![0x3cu8; 4 * dt.bytes()];
            let text = pack_from_raw(dt, &[2, 2], &raw).unwrap();
            assert_eq!(unpack_raw(&text, dt, &[2, 2]).unwrap(), raw, "{}", dt.name());
        }
    }

    /// The §3.1 no-coercion rule, at element granularity: widen inside a kind,
    /// never across one and never downward.
    #[test]
    fn a_typed_tensor_decode_widens_but_never_coerces() {
        let i32_wire = envelope(WireDType::I32, &[1], &5i32.to_le_bytes());
        assert_eq!(unpack_raw(&i32_wire, WireDType::I64, &[1]).unwrap(),
                   5i64.to_le_bytes().to_vec());
        let f32_wire = envelope(WireDType::F32, &[1], &0.5f32.to_le_bytes());
        assert_eq!(unpack_raw(&f32_wire, WireDType::F64, &[1]).unwrap(),
                   0.5f64.to_le_bytes().to_vec());
        // bf16/f16 widen into f32 the way `vault.load_npz` widens them (#292).
        let bf16_wire = envelope(WireDType::Bf16, &[1], &[0x00, 0x3f]);
        assert_eq!(unpack_raw(&bf16_wire, WireDType::F32, &[1]).unwrap(),
                   0.5f32.to_le_bytes().to_vec());

        // …and the refusals, each `decode-type`.
        let f64_wire = envelope(WireDType::F64, &[1], &0.5f64.to_le_bytes());
        for (text, want, msg) in [
            (&f64_wire, WireDType::F32, "expected a `f32` tensor, got `f64`"),
            (&f64_wire, WireDType::I64, "expected a `i64` tensor, got `f64`"),
            (&i32_wire, WireDType::Bool, "expected a `bool` tensor, got `i32`"),
            (&i32_wire, WireDType::F64, "expected a `f64` tensor, got `i32`"),
        ] {
            assert_eq!(unpack_raw(text, want, &[1]).unwrap_err(),
                       format!("decode-type: {}", msg));
        }
        // `1` is not `true` here either.
        let bool_wire = envelope(WireDType::Bool, &[1], &[1]);
        assert_eq!(unpack_raw(&bool_wire, WireDType::I64, &[1]).unwrap_err(),
                   "decode-type: expected a `i64` tensor, got `bool`");
    }

    /// Everything a malformed envelope can be, and the tag it earns. The split
    /// is §6's: `decode-parse` is "not JSON", every other failure is a
    /// well-formed document that is not the tensor that was asked for.
    #[test]
    fn a_malformed_envelope_carries_the_decode_tags() {
        let ok = pack_from_f64(WireDType::I64, &[2], &[1.0, 2.0]).unwrap();
        assert!(unpack_raw("{oops", WireDType::I64, &[2])
            .unwrap_err().starts_with("decode-parse: "));
        for (text, msg) in [
            ("[1,2]", "expected a `dmc_tensor` tensor envelope, got list".to_string()),
            ("7", "expected a `dmc_tensor` tensor envelope, got i64".to_string()),
            ("{\"a\":1}",
             "expected a `dmc_tensor` tensor envelope, got a plain JSON object".to_string()),
            (&ok.replace("\"dmc_tensor\":1", "\"dmc_tensor\":2"),
             "tensor envelope version 2 is not 1".to_string()),
            (&ok.replace("row_major", "col_major"),
             "tensor layout `col_major` is not `row_major`".to_string()),
            (&ok.replace("\"i64\"", "\"i128\""), "unknown tensor dtype `i128`".to_string()),
            (&ok.replace("[2]", "[3]"), "expected tensor shape [2], got [3]".to_string()),
            (&ok.replace("\"data\":\"", "\"data\":\"!"),
             "tensor `data` is not valid base64".to_string()),
            // The tag is there, so "a plain JSON object" would be a lie: the
            // document is claiming to be an envelope and failing at it.
            (&ok.replace("\"dmc_tensor\":1", "\"dmc_tensor\":\"1\""),
             "tensor envelope version must be an integer, got str".to_string()),
            (&ok.replace("\"dmc_tensor\":1", "\"dmc_tensor\":null"),
             "tensor envelope version must be an integer, got nil".to_string()),
            // Manifest-strict on fields, not descriptor-additive: an unknown
            // key is a format this reader does not know (§3.2). Two of them,
            // to pin that the list is sorted rather than map-ordered.
            (&ok.replace("\"dtype\"", "\"evil\":{\"a\":1},\"beast\":9,\"dtype\""),
             "tensor envelope has unknown field(s) `beast`, `evil`".to_string()),
        ] {
            assert_eq!(unpack_raw(text, WireDType::I64, &[2]).unwrap_err(),
                       format!("decode-type: {}", msg), "for {}", text);
        }
        // A payload that is valid base64 but the wrong length for the shape.
        let short = envelope(WireDType::I64, &[2], &1i64.to_le_bytes());
        assert_eq!(unpack_raw(&short, WireDType::I64, &[2]).unwrap_err(),
                   "decode-type: tensor payload is 8 bytes but `i64`[2] needs 16");
    }

    /// base64 is the payload encoding, so it is strict in both directions:
    /// padded, standard alphabet, nothing else accepted.
    #[test]
    fn base64_is_rfc4648_and_strict() {
        for (bytes, text) in [
            (&b""[..], ""), (&b"f"[..], "Zg=="), (&b"fo"[..], "Zm8="),
            (&b"foo"[..], "Zm9v"), (&b"foob"[..], "Zm9vYg=="),
            (&b"\xff\xfe\xfd"[..], "//79"),
        ] {
            assert_eq!(base64_encode(bytes), text);
            assert_eq!(base64_decode(text).unwrap(), bytes);
        }
        for bad in ["Zg=", "Zg", "Z g==", "Zm9v!", "Zg===", "Z===", "=m9v", "Zg=v"] {
            assert!(base64_decode(bad).is_none(), "accepted `{}`", bad);
        }
    }

    /// The `shape` bound is part of the format, and a shape that falls outside
    /// it is a *local* failure of `port_tensor_encode` — no runtime is involved
    /// and there is no `Err` half — so it carries no `§6` tag. `port-protocol`
    /// belongs to an envelope sent to a runtime, which this never became.
    #[test]
    fn an_unencodable_shape_is_untagged_and_states_the_bound() {
        for shape in [&[0i64][..], &[2, 0][..], &[-1][..], &[][..],
                      &[1, 1, 1, 1, 1, 1, 1, 1, 1][..]] {
            let e = pack_from_f64(WireDType::I64, shape, &[]).unwrap_err();
            assert!(e.starts_with("shape "), "for {:?}: {}", shape, e);
            assert!(e.contains("1 to 8 extents, every one positive"), "for {:?}: {}", shape, e);
            for tag in ["port-protocol", "port-call", "decode-type", "decode-parse"] {
                assert!(!e.contains(tag), "for {:?}: {} carries `{}`", shape, e, tag);
            }
        }
        // Rank 8 is inside the bound, so it encodes.
        assert!(pack_from_f64(WireDType::I64, &[1; 8], &[0.0]).is_ok());
    }

    /// The wire dtype is the tensor's, never the far side's inventory. numpy
    /// has no `bfloat16`, so the harness widens a `bf16` payload to f32 to make
    /// it computable — and must undo that on the way out, or the same program's
    /// `dmc.echo` would answer `bf16` or `f32` depending on whether numpy
    /// happened to be installed. Both harness paths are driven here: the
    /// ambient one, and one forced to the no-numpy branch by a stub `numpy`
    /// that refuses to import. The repo's gates are numpy-free, which is
    /// exactly why this divergence was invisible until it was looked for.
    /// Sampling two ordinary values is what let the first version of this fix
    /// pass while still corrupting 126 patterns, so the gate is the whole
    /// space: all 65536 bf16 bit patterns, in one tensor, through the real
    /// harness, on both paths. sNaN is the band that broke — widening through
    /// a Python float quiets it, because a Python float is a C double and
    /// f32 → f64 → f32 is not the identity on a signaling NaN.
    #[test]
    fn a_bf16_round_trip_does_not_depend_on_numpy() {
        if !have_python() { eprintln!("skipped: python3 not on PATH"); return; }
        let all: Vec<u8> = (0u32..65536).flat_map(|h| (h as u16).to_le_bytes()).collect();
        let want = envelope(WireDType::Bf16, &[65536], &all);
        let reqs = [
            ("dmc.echo", format!("[{}]", want)),
            ("dmc.dtype", format!("[{}]", want)),
        ];
        let blocked = numpy_blocking_dir("bf16");
        for (path, what) in [(None, "as installed"), (Some(blocked.as_path()), "numpy blocked")] {
            let got = drive_harness(path, &reqs);
            let echoed = got[0].as_ref().unwrap_or_else(|e| panic!("{}: {}", what, e));
            // Compare the decoded payloads, so a failure names the pattern
            // rather than dumping 175 KB of base64 into the log.
            let back = unpack_raw(echoed, WireDType::Bf16, &[65536])
                .unwrap_or_else(|e| panic!("{}: {}", what, e));
            let bad: Vec<String> = (0..65536)
                .filter(|&i| back[i * 2..i * 2 + 2] != all[i * 2..i * 2 + 2])
                .map(|i| format!("{:#06x}->{:#06x}", i,
                    u16::from_le_bytes([back[i * 2], back[i * 2 + 1]])))
                .collect();
            assert!(bad.is_empty(),
                "{}: {} of 65536 bf16 patterns changed on an untouched round trip: {:?}",
                what, bad.len(), &bad[..bad.len().min(8)]);
            assert_eq!(got[1].as_deref(), Ok("\"bf16\""), "far-side dtype, {}", what);
        }
        let _ = std::fs::remove_dir_all(&blocked);
        // The complement of the rule: data the far side newly allocated is the
        // far side's storage, and crosses as what it now is.
        if !have_numpy() { eprintln!("note: numpy path covered only by the blocked run"); return; }
        let two = envelope(WireDType::Bf16, &[2], &[0x00, 0x3f, 0xa0, 0xbf]);
        let new = drive_harness(None, &[("numpy.negative", format!("[{}]", two))]);
        assert_eq!(new[0].as_deref(), Ok(envelope(WireDType::F32, &[2], &[
            0x00, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xa0, 0x3f]).as_str()));
    }

    /// Python's `bool` is a subclass of `int` and `1.0 == 1`, so a bare
    /// equality on the version tag accepted `true` and `1.0` — and the harness
    /// then echoed them back normalised while `unpack_raw` refused them. §3.2
    /// says the two readers refuse the same documents; this is that sentence,
    /// gated. The `shape` check had the same hole, where the wrong tag was the
    /// visible symptom: `[true]` survived the ABI check and died later inside
    /// `reshape`, arriving as `port-call` where §3.2 promises `port-protocol`.
    #[test]
    fn both_readers_refuse_the_same_envelopes() {
        if !have_python() { eprintln!("skipped: python3 not on PATH"); return; }
        let ok = envelope(WireDType::I64, &[2], &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
        let blocked = numpy_blocking_dir("readers");
        for bad in [
            ok.replace("\"dmc_tensor\":1", "\"dmc_tensor\":true"),
            ok.replace("\"dmc_tensor\":1", "\"dmc_tensor\":1.0"),
            ok.replace("\"shape\":[2]", "\"shape\":[true]"),
        ] {
            // The local reader refuses it, with the decode family.
            let here = unpack_raw(&bad, WireDType::I64, &[2]).unwrap_err();
            assert!(here.starts_with("decode-type: "), "for {}: {}", bad, here);
            // And so does the far side, before any foreign code runs.
            for path in [None, Some(blocked.as_path())] {
                let got = drive_harness(path, &[("dmc.echo", format!("[{}]", bad))]);
                let e = got[0].as_ref().expect_err(&format!("harness accepted {}", bad));
                assert!(e.starts_with("port-protocol: "), "for {}: {}", bad, e);
            }
        }
        let _ = std::fs::remove_dir_all(&blocked);
    }

    /// #534, item 2: where the domain is 16 bits, sweep it rather than sample
    /// it. The port's own bf16 gate above is the precedent — two sampled
    /// values passed while 126 patterns were being corrupted. These in-process
    /// narrowings are the same shape and had no gate at all.
    ///
    /// The rule these pin is not blanket identity, because the encoder cannot
    /// offer it. `write_elem` takes an `f64` — the interpreter carries every
    /// tensor element as one — so a half reaching the wire has been through
    /// f32 -> f64 and back, and that hop quiets a signaling NaN. So the rule
    /// is: **every pattern round-trips unchanged except a signaling NaN, which
    /// arrives as itself with the quiet bit set, and nothing else moves.**
    ///
    /// Written this way the gate still fails on the two things worth catching:
    /// any ordinary value that shifts, and any change to the size or shape of
    /// the signaling band. An `assert!(bad.is_empty())` could not be written
    /// here at all, and asserting only a count would pass on the wrong 126.
    ///
    /// `exp_mask` selects the exponent field, `mant_mask` the mantissa, and
    /// `quiet` the mantissa's top bit.
    fn assert_only_snans_are_quieted(
        label: &str, dt: WireDType, widen: fn(u16) -> f32,
        exp_mask: u16, mant_mask: u16, quiet: u16,
    ) {
        let is_snan = |h: u16| {
            h & exp_mask == exp_mask && h & mant_mask != 0 && h & quiet == 0
        };
        let mut moved = 0usize;
        let mut wrong: Vec<String> = Vec::new();
        for h in 0u32..65536 {
            let h = h as u16;
            let mut out = Vec::new();
            write_elem(dt, widen(h) as f64, &mut out);
            let back = u16::from_le_bytes([out[0], out[1]]);
            if back == h {
                // An sNaN that came back untouched would mean the band moved.
                if is_snan(h) {
                    wrong.push(format!("{:#06x} is sNaN but survived", h));
                }
                continue;
            }
            moved += 1;
            if !is_snan(h) || back != h | quiet {
                wrong.push(format!("{:#06x}->{:#06x}", h, back));
            }
        }
        assert!(wrong.is_empty(),
            "{}: {} pattern(s) broke the rule (an sNaN quiets, nothing else moves): {:?}",
            label, wrong.len(), &wrong[..wrong.len().min(8)]);
        // The signaling band is every all-ones-exponent value with a non-zero
        // mantissa whose top bit is clear, both signs.
        let want = 2 * ((mant_mask as usize + 1) / 2 - 1);
        assert_eq!(moved, want,
            "{}: expected exactly the {} signaling-NaN patterns to quiet, saw {}",
            label, want, moved);
    }

    #[test]
    fn only_signaling_bf16_nans_change_through_write_elem() {
        assert_only_snans_are_quieted(
            "bf16", WireDType::Bf16, crate::jit::bf16_bits_to_f32,
            0x7f80, 0x007f, 0x0040);
    }

    /// Before the payload fix this moved 2044 of 65536 — every half NaN was
    /// flattened onto 0x7e00 / 0xfe00. Now only the signaling band moves, and
    /// it moves by exactly the quiet bit.
    #[test]
    fn only_signaling_f16_nans_change_through_write_elem() {
        assert_only_snans_are_quieted(
            "f16", WireDType::F16, crate::jit::f16_bits_to_f32,
            0x7c00, 0x03ff, 0x0200);
    }

    /// The JIT's own narrowing, with no f64 hop in the way. Truncating the low
    /// 16 bits is exactly the inverse of the widener's `<< 16`, so this one
    /// does hold across the whole space, every NaN payload included — which is
    /// what makes the f64 hop, not the narrowing, the thing costing the port
    /// path its signaling NaNs.
    #[test]
    fn every_bf16_pattern_survives_the_jit_narrowing() {
        let bad: Vec<String> = (0u32..65536).filter_map(|h| {
            let h = h as u16;
            let back = (crate::jit::bf16_bits_to_f32(h).to_bits() >> 16) as u16;
            (back != h).then(|| format!("{:#06x}->{:#06x}", h, back))
        }).collect();
        assert!(bad.is_empty(),
            "{} of 65536 bf16 patterns changed through the JIT narrowing: {:?}",
            bad.len(), &bad[..bad.len().min(8)]);
    }
}
