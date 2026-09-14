/// demoniC tree-walking interpreter — Phase 3.5 (pre-alpha).
///
/// Walks the AST and evaluates it directly. Uses `ndarray` for tensor
/// storage on CPU. Slow but correct — serves as the reference
/// implementation forever (Phase 4 Metal backend cross-checks against it).
///
/// Pre-alpha scope: enough to run hello.dmc and small custom programs
/// end-to-end. The interpreter is general-purpose: scalar arithmetic
/// (full integer + float hierarchy), tensor ops (2D matmul, elementwise
/// broadcast, sum), strings, control flow, function calls, pattern matching
/// on idents/wildcards/tuples.
///
/// Deferred to Phase 4 / later:
///   - @grad autodiff
///   - @shard / @tp / @pp distribution
///   - KV[~] streaming axes: `<-` appends along the declared `~` axis
///   - Higher-rank matmul (only 2D for now)
///   - Models / methods (only top-level fns)
///   - Match: shape patterns and bind (@) patterns not yet supported
///   - Most stdlib (print/sum/sqrt/exp/log only)

use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::cell::{Cell, RefCell};

use ndarray::{ArrayD, Axis, Dimension, IxDyn};
extern crate crossterm;

use crate::ast::*;
use crate::lexer::Span;

// ─── Runtime errors ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RuntimeError {
    pub msg: String,
    pub span: Option<Span>,
    /// Set by the postfix `?` operator on an error value. It is not a real
    /// error — it carries the ERROR HALF (`e`) of the `(T, Err)` tuple that
    /// tripped, and is intercepted at the nearest function-call boundary
    /// (SPEC §4.9's `return (default[T], e)`), where the `T` half is rebuilt
    /// from the ENCLOSING function's declared return type. `None` for
    /// ordinary errors.
    ///
    /// #628: this used to carry the callee's *whole* tuple, which the catch
    /// site returned verbatim — so the caller's success slot held the
    /// callee's error-path value (`777`, not `0`), and when the two functions'
    /// `T` differed it was not even the right type.
    ///
    /// "The nearest function-call boundary" above means `call_fn_in_module`
    /// (the body half of `call_fn_with_shapes`) and `call_lambda` specifically — every OTHER frame this signal might
    /// cross is not such a boundary and does not catch it. #647 (still open):
    /// the `@grad` calling conventions (`.fwd_bwd`/`.grad`/`.fwd`, evaluated
    /// via `eval_block_with_tape`) have no catch of their own, so a `?` inside
    /// a `@grad fn -> (T, Err)` escapes that frame uncaught and is instead
    /// rebuilt against whichever named fn or lambda called `.fwd_bwd(...)` —
    /// the WRONG `T` whenever the two differ. Not fixed here (different
    /// category/PR); see `call_grad`'s own note near its `eval_block_with_tape`
    /// call.
    pub propagate: Option<Box<Value>>,
    /// #639: a `return` / `break` / `continue` that left a block *expression*.
    /// Like `propagate`, it is not a real error — `eval_expr` can only hand
    /// back a `Value`, so this is the carrier that lifts an escape out of an
    /// expression context up to the nearest statement boundary, where
    /// `eval_stmt` turns it back into the `Flow` an `if` body would have
    /// produced. Private: only `escaping_flow` builds one, and `catch_flow`
    /// is the only thing that opens it. `None` for ordinary errors.
    ///
    /// #628 × #639: the two carriers are DISJOINT and are caught at different
    /// boundaries, so neither may be tested for by "is this an error?". A `?`
    /// signal has `propagate: Some(_), flow: None` and is caught only at a
    /// function-call boundary (`finish_query_propagation`); a block escape has
    /// `flow: Some(_), propagate: None` and is caught at the nearest statement
    /// or block boundary (`catch_flow`), one frame below. Every site that
    /// intercepts one guards on ITS OWN field — `e.propagate.is_some()` /
    /// `e.flow.is_some()` — so the other rides through untouched to the
    /// boundary that owns it.
    flow: Option<Box<Escape>>,
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(sp) = &self.span {
            write!(f, "runtime error at {}:{}: {}", sp.line, sp.col, self.msg)
        } else {
            write!(f, "runtime error: {}", self.msg)
        }
    }
}

impl RuntimeError {
    pub fn msg(s: impl Into<String>) -> Self { Self { msg: s.into(), span: None, propagate: None, flow: None } }
    pub fn at(s: impl Into<String>, sp: Span) -> Self { Self { msg: s.into(), span: Some(sp), propagate: None, flow: None } }
    /// `?`-operator early-return signal carrying the error half `e` to return
    /// alongside `default[T]` (SPEC §4.9). The span passed in is whatever
    /// `PostfixOp::Query`'s eval site hands it — `parse_postfix` (`parser.rs`)
    /// starts that span at the OPERAND, not the `?` token, so it covers the
    /// whole `expr?`, not just the operator. Even so, a refusal to build
    /// `default[T]` points at this postfix expression rather than at the
    /// call boundary that intercepts it.
    ///
    /// `msg` here is a last-resort diagnostic, not the one users actually see:
    /// every reachable entry point (`main`/`test_*`, always invoked through
    /// `call_fn`/`call_fn_with_shapes`) guarantees this signal reaches a
    /// `finish_query_propagation` call before anything ever calls `Display`
    /// on it — that call either rebuilds `(default[T], e)` or replaces this
    /// error outright with its own "`?` is only legal inside…" diagnostic. So
    /// `msg` is intentionally an internal-invariant message, not a duplicate
    /// of that user-facing one: seeing it would mean a `?` signal escaped
    /// every enclosing function-call boundary, which is an interpreter bug.
    ///
    /// #639's block-flow carrier is a different signal on the same channel:
    /// this one is `propagate: Some(_), flow: None`, so `catch_flow` (which
    /// reads `flow`) lets it past every statement and block boundary on its way
    /// to the function-call boundary that owns it.
    fn propagate(err_half: Value, sp: Span) -> Self {
        Self {
            msg: "internal error: a `?` early-return signal escaped every \
                  enclosing function-call boundary uncaught (this is an \
                  interpreter bug, not a `?` misuse — `finish_query_propagation` \
                  should have intercepted it first)".into(),
            span: Some(sp),
            propagate: Some(Box::new(err_half)),
            flow: None,
        }
    }
    /// #639: the block-flow carrier. Every path that can raise one also
    /// catches it — `eval_stmt` and `eval_block` between them cover every
    /// expression position, and `call_fn_with_shapes` / `call_grad` close the
    /// two entry points that bypass `eval_block` — so the message is a
    /// backstop for a context that swallows the `Err` instead, and for the one
    /// position that has no enclosing statement at all: a module-level
    /// `let G = { return 5 }`. All three wordings are reachable from there, and
    /// they match what `eval_if` already says about the same three keywords.
    /// The span is the escaping keyword's own, so the backstop points at the
    /// `return` / `break` / `continue` rather than at nothing.
    fn escaping_flow(esc: Escape) -> Self {
        let (msg, sp) = match &esc {
            Escape::Return(_, sp) => ("`return` outside a function", sp.clone()),
            Escape::Break(sp) => ("`break` outside loop", sp.clone()),
            Escape::Continue(sp) => ("`continue` outside loop", sp.clone()),
        };
        Self {
            msg: msg.into(),
            span: Some(sp),
            propagate: None,
            flow: Some(Box::new(esc)),
        }
    }
}

/// #639: re-open the block-flow carrier. A `return` / `break` / `continue`
/// that escaped a block expression rides the error channel (the only channel
/// an `Expr` has) until the nearest statement boundary; here it becomes the
/// `Flow` again, so the enclosing loop or function sees exactly what it would
/// have seen from an `if` body. Ordinary errors pass through untouched — and
/// so does #628's `?` signal, which is a `propagate` carrier with `flow: None`
/// and belongs to a function-call boundary further up, not to this one.
fn catch_flow(r: EvalResult<Flow>) -> EvalResult<Flow> {
    match r {
        Err(e) => match e.flow {
            Some(esc) => Ok(esc.into_flow()),
            None => Err(e),
        },
        ok => ok,
    }
}

pub type EvalResult<T> = Result<T, RuntimeError>;

/// #505: why a `@comptime` fold did not produce a value (`SPEC.md §7.8`).
///
/// Kept apart from `RuntimeError` because the two become different
/// diagnostics: `Budget` is `comptime-budget` (the operands were static, the
/// evaluation just did not finish) while `Failed` is `comptime-non-static`
/// (the body did something a compile-time evaluation cannot do). A caller
/// that collapsed them would have to re-derive the distinction from prose.
#[derive(Debug)]
pub enum ComptimeStop {
    /// The step budget ran out — the block may not terminate.
    Budget,
    /// The body raised an ordinary evaluation error (overflow, division by
    /// zero, a value the fold set has no representation for).
    Failed(RuntimeError),
}

// ─── Value ───────────────────────────────────────────────────────────────────

/// Coarse runtime dtype tag for tensors. The interpreter stores all tensor
/// data as `f64` (pre-alpha pragma), but it must remember whether a tensor
/// holds *integers* so that a scalar element read returns `Value::Int` rather
/// than `Value::Float`. Without this, integer division, `%`, bitwise ops and
/// range/loop bounds sourced from a tensor element silently misbehave (#125).
///
/// `Int` carries the element's WIDTH (#571), the way the float side splits
/// `F32`/`F64`: a tensor element's width is part of its type, so an element
/// read out of a `Tensor[i32, …]` is an `i32` and `a .+ b` over two of them
/// wraps at 32 bits. Before this the tag was width-less, the elementwise path
/// computed in f64, and `2147483647 .+ 1` answered 2147483648 — a value that
/// does not fit the element type the checker assigned it. The width comes from
/// the constructor or cast that named the element type; a source that names no
/// width (a bare integer tensor literal, `read_bytes`, `uniform_int`) is `I64`,
/// which wraps at 64 bits and so is the identity on everything below it.
///
/// #241: floats track their *width*, because the two widths have different
/// JIT-parity semantics. `F32` covers f32 and the narrower float-likes that
/// the JIT computes as f32 (the cast-no-op convention, #179) and that #834
/// does not give their own declared metering width — `f16`, `bf16`, `tf32`:
/// the data stays f64-backed but is rounded through f32 whenever a tensor is
/// produced or an element is stored, so `dmc run` matches `dmc jit` numerics
/// by default. `F64` keeps full f64, which is what the JIT does for
/// declared-f64 tensors. `F32` is the default float tag — tensor literals,
/// op results, and rng constructors are f32 on the JIT too. `Bool` is
/// f64-backed like everything else (0.0/1.0), but element reads yield
/// `Value::Bool` and element writes accept one — matching the JIT, where a
/// bool tensor is a 1-byte 0/1 lane and `m[i,j]` is a real bool.
///
/// `Fp8E4M3`/`Fp8E5M2` (#834) are ALSO f32-backed nominal tags computationally
/// — every arithmetic/rounding decision that keys off `F32` must key off
/// these two the same way (`DType::rounds_as_f32`), and `default_tensor_dtype`/
/// `arena_ctor_dtype` build them from the identical `ScalarType` table
/// (`dtype_for_scalar_elem`) — but they carry their OWN one-byte
/// `byte_width`, unlike `f16`/`bf16`/`tf32` above: SPEC §3.1 declares
/// `fp8_e4m3`/`fp8_e5m2` as genuinely 8-bit floats, and #790's metering
/// charges the DECLARED width, not the arithmetic one. Before #834 the
/// interpreter had no representation for either at all and silently reused
/// `F32`'s tag (and so its 4-byte meter) for both — the bug #834 fixes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType { Int(IW), F32, F64, Trit, Bool, Fp8E4M3, Fp8E5M2 }

impl DType {
    /// Declared element width in bytes (#781) — what `--vault`/`--forge`
    /// charge per element when a tensor of this dtype is materialized. This
    /// is the width `T` names in `Tensor[T, …]`, NOT the interpreter's f64
    /// storage width every `TensorVal` uses regardless of `dtype` (`DType`'s
    /// own doc comment). Matches the JIT's `ScalarKind::store_bytes`
    /// (`jit.rs`) element-for-element, so the two backends charge the same
    /// program the same requested bytes — except `Fp8E4M3`/`Fp8E5M2`, which
    /// the JIT never lowers at all (refused as an `ElementType` gap), so
    /// there is no JIT-side kind to match against.
    fn byte_width(self) -> u64 {
        match self {
            DType::F64 => 8,
            DType::F32 => 4,
            // #834: fp8's declared width is one byte (SPEC §3.1's 8-bit
            // floats) — same as `Bool`/`Trit`, just for an unrelated reason.
            DType::Bool | DType::Trit | DType::Fp8E4M3 | DType::Fp8E5M2 => 1,
            DType::Int(w) => match w {
                IW::I64 | IW::U64 => 8,
                IW::I32 | IW::U32 => 4,
                IW::I16 | IW::U16 => 2,
                IW::I8  | IW::U8  | IW::I4 => 1,
            },
        }
    }

    /// Does `self` round through f32 the same way plain `DType::F32` does?
    /// (#834) `Fp8E4M3`/`Fp8E5M2` are nominal f32-backed tags (SPEC §3.1:
    /// "computed in f32 and retagged, never rounded to their own narrower
    /// precision") that differ from `F32` ONLY in declared metering width
    /// (`byte_width`) — every other site in this file that used to test
    /// `== DType::F32` to decide "does this tensor's arithmetic round at f32
    /// precision" widens to this predicate instead, so an fp8 tensor keeps
    /// computing exactly like an f32 one now tags it distinctly for
    /// metering. Before #834 fp8 tensors were mistakenly tagged plain
    /// `DType::F32` outright, so this predicate's `true` answer for them is
    /// not new behavior — only the separate `byte_width` is.
    fn rounds_as_f32(self) -> bool {
        matches!(self, DType::F32 | DType::Fp8E4M3 | DType::Fp8E5M2)
    }
}

/// Width of a scalar float (#473) — the scalar counterpart of `DType`'s
/// `F32`/`F64` split for tensors.
///
/// `F32` values are rounded through f32 after every operation that produces
/// one. #241 established this is not an approximation: for `+ - * /`,
/// computing in f64 and rounding once through f32 is *bit-exact* equal to
/// native f32 arithmetic, because double rounding is harmless when the
/// intermediate carries at least 2p+2 bits and f64's 53 > 2*24+2. That is what
/// makes f32 tensors match the JIT today; scalars match it the same way,
/// without needing an f32-native evaluator.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FW { F64, F32 }

impl FW {
    /// Width of a binary op's result: f32 only when BOTH operands are f32,
    /// mirroring the tensor rule in #241 ("result is F64 iff an operand is
    /// F64"). An f64 operand keeps the result f64, so mixed arithmetic never
    /// silently narrows.
    #[inline]
    fn join(a: FW, b: FW) -> FW {
        if a == FW::F32 && b == FW::F32 { FW::F32 } else { FW::F64 }
    }
    /// Round a value to this width.
    #[inline]
    fn round(self, x: f64) -> f64 {
        match self { FW::F32 => quantize_f32(x), FW::F64 => x }
    }
}

/// Round a scalar through f32 (the F32-tensor store semantics).
#[inline]
fn quantize_f32(x: f64) -> f64 {
    x as f32 as f64
}

/// The width a value contributes to an arithmetic result. Ints and non-floats
/// count as f64 so that `1 + 0.5f32` stays f64 — mixed arithmetic never
/// silently narrows (#473).
#[inline]
fn float_width(v: &Value) -> FW {
    match v { Value::Float(_, w) => *w, _ => FW::F64 }
}

/// Width of a scalar integer (#540) — the integer twin of `FW`.
///
/// SPEC §3.1: arithmetic on a fixed-width integer type wraps at that width.
/// The interpreter is dynamically typed, so the only way an add can wrap at 32
/// bits is for the value to carry the width the annotation gave it — exactly
/// how #473 made an `f32` annotation real at run time. Without it `let c: i32 =
/// a + b` stores an i64 while the JIT wraps its `cl::I32`, and the backends
/// part company on every `i32` expression that leaves the 32-bit range.
///
/// #544 widened this from `{I64, I32}` to every fixed-width integer kind the
/// language has. Each variant is a WIDTH plus a SIGNEDNESS; the backing
/// storage is an i64 in all cases, holding the wrapped value sign-extended
/// (signed kinds) or zero-extended (unsigned kinds). Zero-extension is what
/// makes `0u32 - 1u32` the unsigned 4294967295 rather than -1, and it also
/// keeps `/ % >> < >` correct without a separate unsigned instruction set:
/// a zero-extended `u8`/`u16`/`u32` is a non-negative i64, so the signed
/// operation IS the unsigned one.
///
/// `u64` (#597) carries the 64-bit two's-complement PATTERN in the same
/// i64-backed slot `i64` uses — there is no wider machine word to zero-extend
/// into, so a `u64` value at or above 2^63 is stored as a negative `i64`. The
/// `U32`/`U16`/`U8` trick above (a signed instruction on a zero-extended,
/// always-non-negative pattern IS the unsigned one) stops working here: at 64
/// bits the pattern can be genuinely negative, so `/ % >>` and the ordering
/// comparisons re-interpret the pattern as `u64` explicitly wherever the width
/// is unsigned (see `int_arith`, `IW::wrap`'s `BitShr` caller, and
/// `scalar_compare`) instead of reusing the signed op. `int8` is a signed
/// 8-bit type (SPEC §3.1) and shares `I8`; `int4` is `I4`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IW { I64, I32, I16, I8, I4, U64, U32, U16, U8 }

impl IW {
    /// Width of an integer binary op's result. Unlike `FW::join`, the *narrow*
    /// width wins: `check.rs` types `i32 OP i32` as `i32` and forbids an `i32`
    /// meeting a real `i64` (§3.1, no mutual assignability), so the only value
    /// that reaches a narrow operand tagged `I64` is an untyped integer
    /// literal — which §3.1 says adopts "the type of its context … or other
    /// operand at its use site". That is the same rule the JIT applies in
    /// `adopt_int_literal_kind` (#541), resolved here against the operand's
    /// runtime tag instead of its lowered kind. Two genuinely different narrow
    /// kinds never meet (the checker refuses `u8 + i16`), so the fall-through
    /// only has to be total, not clever: take the narrower.
    ///
    /// #597 (review): "take the narrower" means a hypothetical `u8`-vs-`i64`
    /// pair joins to `U8`, and any caller that then branches on
    /// `w.is_signed()` would read a genuine i64 unsigned. `int_arith` is safe
    /// from that by construction — it `wrap`s both operands into `w` before
    /// looking at the signedness, so an unsigned `w` always sees a
    /// zero-extended, non-negative pattern — and `scalar_compare`, which does
    /// NOT wrap, keys on its operands' own tags instead of on this join. Any
    /// future signedness decision must do one or the other.
    #[inline]
    fn join(a: IW, b: IW) -> IW {
        if a == b { return a; }
        if a == IW::I64 { return b; }
        if b == IW::I64 { return a; }
        if a.bits() <= b.bits() { a } else { b }
    }
    /// Wrap a value into this width, two's-complement (#300: overflow wraps).
    /// Unsigned kinds land in `0..2^w`, signed kinds in `-2^(w-1)..2^(w-1)`.
    ///
    /// `pub(crate)` (#718 rework, R1): the JIT's shape-dimension fold for a
    /// local `let` needs to wrap a compile-time-folded initializer at the
    /// binding's own width exactly as this does, and reusing it directly
    /// is what keeps the two backends from drifting apart on the rule.
    #[inline]
    pub(crate) fn wrap(self, x: i64) -> i64 {
        match self {
            // #597: `u64` occupies the whole 64-bit slot, same as `i64` — there
            // is nothing narrower to mask into, so the pattern passes through
            // unchanged. It is `/ % >>` and the comparisons, not `wrap`, that
            // read the pattern differently for `U64` (see `int_arith`,
            // `apply_binop`'s `BitShr` arm, and `scalar_compare`).
            IW::I64 | IW::U64 => x,
            IW::I32 => x as i32 as i64,
            IW::I16 => x as i16 as i64,
            IW::I8  => x as i8  as i64,
            // No Rust primitive is 4 bits: mask, then sign-extend bit 3.
            IW::I4  => ((x & 0xf) ^ 0x8) - 0x8,
            IW::U32 => x as u32 as i64,
            IW::U16 => x as u16 as i64,
            IW::U8  => x as u8  as i64,
        }
    }
    /// Bit width, for the `/` overflow and `<<` range diagnostics, which name
    /// the range actually violated (`0..=31` for an i32 shift).
    #[inline]
    fn bits(self) -> u32 {
        match self {
            IW::I64 | IW::U64 => 64,
            IW::I32 | IW::U32 => 32,
            IW::I16 | IW::U16 => 16,
            IW::I8  | IW::U8  => 8,
            IW::I4 => 4,
        }
    }
    /// Whether the width is signed. Only a signed width has a `MIN / -1`.
    #[inline]
    fn is_signed(self) -> bool {
        matches!(self, IW::I64 | IW::I32 | IW::I16 | IW::I8 | IW::I4)
    }
    /// The most negative value at this width — the `MIN / -1` overflow case.
    /// `0` at an unsigned width, which `is_signed` keeps out of that guard.
    #[inline]
    fn min(self) -> i64 {
        if !self.is_signed() { return 0; }
        match self {
            IW::I64 => i64::MIN,
            _ => -(1i64 << (self.bits() - 1)),
        }
    }
    /// How the width names itself in a diagnostic (`exceeds the i32 range`).
    #[inline]
    fn name(self) -> &'static str {
        match self {
            IW::I64 => "i64", IW::I32 => "i32", IW::I16 => "i16",
            // `i4`, not `int4`: the diagnostic names the WIDTH that overflowed,
            // and the JIT's `__dmc_div_overflow_trap` spells it `i{bits}`.
            IW::I8 => "i8", IW::I4 => "i4",
            IW::U64 => "u64", IW::U32 => "u32", IW::U16 => "u16", IW::U8 => "u8",
        }
    }
}

/// The width a value contributes to an integer arithmetic result. Non-ints
/// count as `I64`, so they never drag a result narrow (#540).
#[inline]
fn int_width(v: &Value) -> IW {
    match v { Value::Int(_, w) => *w, _ => IW::I64 }
}

/// The f32-family scalar types. `f16`/`bf16`/`tf32`/`fp8_*` are f32-backed in
/// both backends by the #179 convention — computed in f32 and retagged, never
/// rounded to their own narrower precision.
#[inline]
fn scalar_is_f32_family(t: &ScalarType) -> bool {
    use ScalarType::*;
    matches!(t, F32 | F16 | Bf16 | Tf32 | Fp8E4M3 | Fp8E5M2)
}


/// Apply a declared scalar type's float WIDTH to a value (#473).
///
/// The binding sites — `let x: f32 = ...`, a typed parameter, a typed return —
/// are where an annotation stops being documentation and starts being
/// semantics. Without this, `let a: f32 = 0.1` keeps the f64 0.1 while the JIT
/// lowers the f32 one, and the two backends part company on the next operation.
/// Only floats are touched; every other value passes through untouched.
/// `Port[L]`: the annotation names a runtime, so a handle entering the
/// position has to be one (SPEC §3.11).
///
/// Applied at the same places `coerce_scalar_width` is — a typed parameter, a
/// typed `let`, a declared return — because those are exactly where an
/// annotation becomes semantics, and they are the positions the JIT's
/// `coerce_to` covers. Checking only parameters would have left `let q:
/// Port[lua] = p` and `fn give() -> Port[lua]` refused by the JIT and accepted
/// here, which is the divergence this whole check exists to remove.
///
/// A null/nil handle is not a lang error: whatever reads it next reports it as
/// the missing handle it is.
fn check_port_lang(v: &Value, ty: Option<&Type>, sp: &Span) -> EvalResult<()> {
    let (Some(ty), Value::Opaque(h)) = (ty, v) else { return Ok(()) };
    let Some(want) = crate::ports::port_type_lang(ty) else { return Ok(()) };
    match crate::ports::lang_mismatch(&want, h) {
        Some(msg) => Err(RuntimeError::at(msg, sp.clone())),
        None => Ok(()),
    }
}

/// The integer WIDTH a declared scalar type carries at run time (#540, widened
/// to every fixed-width kind by #544; `u64` given its own tag by #597).
///
/// `u64` maps to its own `IW::U64` rather than sharing `I64`: the storage is
/// still one i64-backed slot (there is no wider machine word), but the tag now
/// records that the pattern is read unsigned by `/ % >>` and the ordering
/// comparisons (see `IW`'s doc comment).
/// `int8`/`int4` are the packed signed 8-/4-bit types of SPEC §3.1.
/// The JIT's `narrow_scalar_kind` is the mirror of this table — the two must
/// stay in lockstep or the backends wrap at different widths.
#[inline]
fn scalar_int_width(t: &ScalarType) -> Option<IW> {
    use ScalarType::*;
    match t {
        I32 => Some(IW::I32),
        I16 => Some(IW::I16),
        I8 | Int8 => Some(IW::I8),
        Int4 => Some(IW::I4),
        U64 => Some(IW::U64),
        U32 => Some(IW::U32),
        U16 => Some(IW::U16),
        U8 => Some(IW::U8),
        I64 => Some(IW::I64),
        _ => None,
    }
}

/// The integer WIDTH a tensor element type carries at run time (#571), keyed by
/// the spelling a constructor or cast names it with (`forge.zeros[i32, …]`,
/// `t as u8`). The name-keyed twin of `scalar_int_width`, and the reason
/// `is_int_type_name` and this table can never disagree about which names are
/// integers: the predicate is defined in terms of this function.
#[inline]
fn int_type_name_width(name: &str) -> Option<IW> {
    Some(match name {
        // `int8` is the packed signed 8-bit type and `int4` the packed signed
        // 4-bit one (SPEC §3.1); they share `I8`/`I4` with the scalar table.
        "i8" | "int8" => IW::I8,
        "i16" => IW::I16,
        "i32" => IW::I32,
        "int4" => IW::I4,
        "u8"  => IW::U8,
        "u16" => IW::U16,
        "u32" => IW::U32,
        // #597: `u64` gets its own tag now, matching `scalar_int_width` — a
        // tensor element's backing store is still f64 (`cast_tensor_to_int_width`
        // truncates the same way regardless), so this only changes which
        // width the element is TAGGED, not the stored magnitude.
        "u64" => IW::U64,
        "i64" => IW::I64,
        _ => return None,
    })
}

/// Narrow every element of a tensor into an integer element width, for a cast
/// that names an integer element type (#571).
///
/// This is the table `apply_cast` has used for the `as` spelling since #298:
/// Rust's `as` out of the f64 lane, which SATURATES a float source at the
/// range end (`4294967298.0 as i32` is 2147483647) — the documented
/// counterpart to the wrapping integer cast. `I4` has no Rust primitive, so it
/// goes through `i8` and then `IW::wrap`, the one wrap rule.
///
/// `I64`/`U64` truncate only: the f64 lane cannot represent the whole 64-bit
/// range, so pushing it through `as i64` would saturate values that survive
/// today. That is the pre-existing representational limit #544 names, not
/// this fix.
fn cast_tensor_to_int_width(data: &ArrayD<f64>, w: IW) -> ArrayD<f64> {
    match w {
        IW::I64 | IW::U64 => data.mapv(|x| x.trunc()),
        IW::I8  => data.mapv(|x| x as i8  as f64),
        IW::I16 => data.mapv(|x| x as i16 as f64),
        IW::I32 => data.mapv(|x| x as i32 as f64),
        IW::U8  => data.mapv(|x| x as u8  as f64),
        IW::U16 => data.mapv(|x| x as u16 as f64),
        IW::U32 => data.mapv(|x| x as u32 as f64),
        IW::I4  => data.mapv(|x| IW::I4.wrap(x as i8 as i64) as f64),
    }
}

fn coerce_scalar_width(v: Value, ty: Option<&Type>) -> Value {
    let Some(Type::Scalar(st, _)) = ty else { return v };
    match v {
        Value::Float(x, _) if scalar_is_f32_family(st) => {
            Value::Float(quantize_f32(x), FW::F32)
        }
        Value::Float(x, _) if matches!(st, ScalarType::F64) => Value::Float(x, FW::F64),
        // #540: an integer annotation is a width, not documentation. This is
        // the same set of sites #473 uses for floats — a typed `let`, a typed
        // parameter, a declared return, a declared model field — because those
        // are where an annotation becomes semantics on both backends.
        Value::Int(n, w0) => match scalar_int_width(st) {
            Some(w) => Value::Int(w.wrap(n), w),
            None => Value::Int(n, w0),
        },
        other => other,
    }
}

/// The parser builds a 1-element `Tuple` for plain parenthesised grouping, so
/// `(T)` must read transparently as `T`. `jit.rs`'s `ty_from_ast` peels it the
/// same way; the two must agree or `?` sees a tuple on one backend and a
/// scalar on the other.
fn peel_grouping(ty: &Type) -> &Type {
    match ty {
        Type::Tuple(els, _) if els.len() == 1 => peel_grouping(&els[0]),
        _ => ty,
    }
}

/// SPEC §4.9's `default[T]` for a scalar `T`: "0 / 0.0 / false / nil".
///
/// `str`'s zero value is `nil`, not `""` — the spec's own enumeration has no
/// empty string in it, and the JIT gives a `str` slot the NULL pointer, which
/// IS `nil` in every nilable-`str` position the language has (`s == nil`, a
/// closed `Port`, a decode's absent error). Widths follow the declared type,
/// as at every other binding site (#473/#540): `default[f32]` is an f32 zero.
fn default_scalar_value(st: &ScalarType) -> Value {
    match st {
        ScalarType::Str | ScalarType::Nil => Value::Nil,
        ScalarType::Bool => Value::Bool(false),
        ScalarType::F64 => Value::Float(0.0, FW::F64),
        s if scalar_is_f32_family(s) => Value::Float(0.0, FW::F32),
        // A trit's zero is an ordinary integer 0, inside {-1, 0, +1}.
        ScalarType::Trit => Value::Int(0, IW::I64),
        s => Value::Int(0, scalar_int_width(s).unwrap_or(IW::I64)),
    }
}

/// The `DType` a scalar type carries as a TENSOR ELEMENT — one table so a
/// constructor's bracket name (`arena_ctor_dtype`) and a resolved type
/// annotation (`default_tensor_dtype`, below) can never tag the same
/// `ScalarType` two different widths (#834, the bug this closes: fp8 used
/// to silently fall back to `DType::F32`'s 4-byte width in the former and
/// was grouped into the same fallback in the latter).
///
/// Returns `None` for `str`/`nil` only — #817 is an open ruling on whether
/// either has a real tensor-element representation at all; this table does
/// not decide it, and `default_tensor_dtype` keeps its own pre-existing
/// `DType::F32` fallback for the two, unchanged by this fix. `arena_ctor_dtype`
/// never calls this with `Str`/`Nil` in the first place — its own
/// `ScalarType::from_ctor_name` table excludes both names in constructor
/// value position, so they take the `DType::F32` fallback there (the
/// checker still admits both; #817 is the open ruling).
fn dtype_for_scalar_elem(st: &ScalarType) -> Option<DType> {
    match st {
        ScalarType::Trit => Some(DType::Trit),
        ScalarType::Bool => Some(DType::Bool),
        ScalarType::F64 => Some(DType::F64),
        // #834: fp8's declared metering width is genuinely one byte (SPEC
        // §3.1) — listed as their own arms rather than folding into the
        // `F32`-family arm below, which both fp8 kinds would also match
        // (they are still nominal f32-backed tags computationally,
        // `DType::rounds_as_f32`), so this table states their width plainly
        // instead of relying on arm order to pick it out.
        ScalarType::Fp8E4M3 => Some(DType::Fp8E4M3),
        ScalarType::Fp8E5M2 => Some(DType::Fp8E5M2),
        ScalarType::F32 | ScalarType::F16 | ScalarType::Bf16 | ScalarType::Tf32 => Some(DType::F32),
        // Exhaustive rather than a trailing `s => scalar_int_width(s)...`
        // catch-all (#834 rework): every fixed-width integer kind is named
        // here so a future `ScalarType` variant is a compile error, not a
        // silent fall-through. `scalar_int_width` still supplies the width —
        // this arm only fixes which variants reach it.
        ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        | ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64
        | ScalarType::Int4 | ScalarType::Int8 => scalar_int_width(st).map(DType::Int),
        ScalarType::Str | ScalarType::Nil => None,
    }
}

/// The `DType` tag a declared tensor element type carries, for `default[T]`'s
/// zero tensor. The type-keyed twin of `arena_ctor_dtype` (which reads the
/// same information off a constructor's bracket name) — both now resolve
/// through the one shared `dtype_for_scalar_elem` table (#834).
fn default_tensor_dtype(st: &ScalarType) -> DType {
    dtype_for_scalar_elem(st).unwrap_or(DType::F32)
}

/// A short spelling of a declared type, for `?`'s diagnostics. Deliberately
/// coarse — it names the type well enough to point back at the signature;
/// `fmt.rs` owns canonical type printing.
fn type_label(ty: &Type) -> String {
    match ty {
        Type::Scalar(s, _) => crate::fmt::scalar_type_str(s).to_string(),
        Type::Tensor(e, _, _) => format!("Tensor[{}, ..]", type_label(e)),
        Type::View(e, _, _) => format!("View[{}, ..]", type_label(e)),
        Type::KV(e, _, _) => format!("KV[{}, ..]", type_label(e)),
        Type::Tuple(els, _) => {
            format!("({})", els.iter().map(type_label).collect::<Vec<_>>().join(", "))
        }
        Type::Array(e, _, _) => format!("[{}; N]", type_label(e)),
        Type::RawPtr(e, _) => format!("*{}", type_label(e)),
        Type::Fn(_, r, _) => format!("fn(..) -> {}", type_label(r)),
        Type::Mesh(_, _) => "Mesh".to_string(),
        Type::Named { name, .. } => name.clone(),
    }
}

/// Round every element of a tensor's data through f32.
#[inline]
fn quantize_f32_arr(data: &mut ArrayD<f64>) {
    data.mapv_inplace(|x| x as f32 as f64);
}

/// Sum a sequence at f32 width: round the running total after every add (#481).
///
/// This is #241's technique applied to a REDUCTION. `acc + x` on two f32-valued
/// f64s is exact, so rounding the result through f32 gives exactly the f32 add,
/// and a fold of them gives exactly a native-f32 sequential sum. The JIT reduces
/// f32 tensors that way and — since #473 — so does a hand-written
/// `let !s = 0.0f32   for i { s = s + t[i] }` loop on both backends. Accumulating
/// in f64 and rounding once at the end, which is what `ndarray::sum` does, is a
/// *different* number: verified across sizes 2..1000, `sum(t)` disagreed with
/// the loop it is documented to mean.
#[inline]
fn sum_f32<I: Iterator<Item = f64>>(it: I) -> f64 {
    it.fold(0.0, |acc, x| quantize_f32(acc + x))
}

/// Sum at the width the tensor's dtype implies (#481): f32 tensors round every
/// add, everything else accumulates in f64 as before.
#[inline]
fn tensor_sum(t: &TensorVal) -> f64 {
    if t.dtype.rounds_as_f32() { sum_f32(t.iter().copied()) } else { t.iter().sum() }
}

/// `sum(t) / n` at the tensor's width (#481). The JIT converts the count to f32
/// and divides in f32, so an f32 tensor must do the same rather than dividing
/// the f32 total by an f64 count.
#[inline]
fn tensor_mean(t: &TensorVal) -> f64 {
    let total = tensor_sum(t);
    let n = t.len() as f64;
    if t.dtype.rounds_as_f32() { quantize_f32(total / quantize_f32(n)) } else { total / n }
}

/// #674: the interpreter's
/// stamp on a tensor's origin budget. `gen` is the budget's generation AT THE
/// MOMENT this tensor was stamped; `live` is the table entry's *own*
/// generation cell, shared by every tensor ever stamped from that entry.
/// Stale iff `gen != live.get()` — a pure predicate needing no table lookup
/// and no `&Interpreter`, which is what lets `check_live` run at a walker
/// (`Debug`, `json_encode_value`) or the AD tape's backward pass, neither of
/// which holds one.
///
/// **PR #821 rework — the tag is held behind an `Rc`, not inlined.** Written
/// as the design spells it (`origin: Option<BudgetTag>`, two words) it added
/// **16 B** to `TensorVal` and so to `Value`, which is passed and returned by
/// value on every one of the interpreter's hot paths: `size_of::<Value>()`
/// went 120 → 136, and the measured wrapper cost was an order of magnitude
/// over §3.2's price. `OriginTag` below is the same predicate in **one**
/// pointer-sized, niche-optimized field: `Value` is 128 B, and the only path
/// that pays the extra indirection is the *budgeted* one, which is also the
/// only one §3.2 prices above a tenth of a nanosecond. Cloning the `Rc` is
/// what propagates the tag to a deep copy of a budgeted tensor — it shares
/// its source's `live` cell and goes stale at the same reap, exactly the
/// propagation B6 requires, before a re-stamp site (§6 Q5) updates it to
/// whatever budget is current at the copy.
pub struct BudgetTag {
    pub generation: u32,
    pub live: Rc<Cell<u32>>,
}

/// #674 slice 3b: a tensor's stamped origin — `None` for the root (never
/// stale), `Some` for a budget carved off it. **One** pointer wide: `Rc`'s
/// non-null niche makes the `Option` free, so this costs `TensorVal` a single
/// word rather than the two an inlined `BudgetTag` would (see `BudgetTag`'s
/// doc for the measurement that forced it).
pub type OriginTag = Option<Rc<BudgetTag>>;

/// #674 slice 3b: the interpreter's per-budget table (`Interpreter::budgets`,
/// one entry per live budget, laid out below). Entry 0 is `forge.root`,
/// `parent(0) = 0` (a fixed point `reap` refuses to touch, §4.1), bound to
/// `--forge` if the flag was given, unbounded otherwise (`extent: None`,
/// `free: None`).
struct BudgetEntry {
    /// Index of the parent budget (`parent(0) = 0`).
    parent: u32,
    /// Bytes this budget was carved with. `None` only for entry 0 under no
    /// `--forge` — "no ceiling" (§4.1): PRE and PRE-A both go vacuous there.
    extent: Option<u64>,
    /// Bytes allocated (not carved) directly in this budget so far — what
    /// `forge_used` used to be, for entry 0.
    used: u64,
    /// Remaining capacity, spent by BOTH `allocate` and `carve` alike — the
    /// table's own `free`, never `ForgeArena::exceeds_budget` (§3.1).  `None`
    /// iff `extent` is `None`.
    free: Option<u64>,
    /// Live-child count; `reap` refuses (`budget-reap`, R2) while this is
    /// nonzero.
    children: u32,
    /// This entry's own generation cell — every `TensorVal` stamped from
    /// this budget shares it via `BudgetTag::live`. `reap` bumps it, which is
    /// the whole of B6's invalidation: no table lookup, no walk, one `Cell`
    /// write reaches every live and future-cloned copy.
    live: Rc<Cell<u32>>,
}

impl BudgetEntry {
    fn root(extent: Option<u64>) -> Self {
        BudgetEntry {
            parent: 0,
            extent,
            used: 0,
            free: extent,
            children: 0,
            live: Rc::new(Cell::new(0)),
        }
    }
}

/// #674 slice 3b: 20 bits of index (1 048 575 live budgets), 44 of
/// generation, one `i64` — a reaped index is reused without the generation
/// wrapping. The interpreter's own generation counter is `u32`
/// (`BudgetTag`/`BudgetEntry::live`), comfortably inside the 44-bit field the
/// descriptor reserves for it.
const BUDGET_INDEX_BITS: u32 = 20;
const BUDGET_INDEX_MASK: i64 = (1i64 << BUDGET_INDEX_BITS) - 1;

fn budget_descriptor(generation: u32, index: u32) -> i64 {
    ((generation as i64) << BUDGET_INDEX_BITS) | (index as i64 & BUDGET_INDEX_MASK)
}
fn budget_index(descriptor: i64) -> usize {
    (descriptor & BUDGET_INDEX_MASK) as usize
}
fn budget_generation(descriptor: i64) -> u32 {
    ((descriptor >> BUDGET_INDEX_BITS) & 0xFFFF_FFFF) as u32
}

thread_local! {
    /// #674: the budget `TensorVal::new`
    /// stamps a fresh tensor's `origin` from — the single construction
    /// chokepoint has no `&Interpreter`, only this ambient thread-local,
    /// which `enter`/`in b`/frame management push and pop (`jit.rs`'s
    /// `FORGE`/`FORGE_LIMIT` pattern). `None` is entry 0 (root): a tensor so
    /// tagged can never go stale, because `forge.root` can never be reaped
    /// (§4.1). Slice 3d is what finally pushes this to `Some` from surface
    /// syntax: `Expr::BudgetBlock` (`forge.enter(b) { … }`) and
    /// `Expr::AllocIn` (`<ctor> in b`) both open a `BudgetGuard`, which is
    /// the only writer outside `VaultTagGuard`'s row-19 clear.
    static CUR_BUDGET_TAG: RefCell<OriginTag> = const { RefCell::new(None) };

    /// #674: the ENTERED-BUDGET REGISTER —
    /// the table index PRE-A charges against (`charge_arena`'s Forge arm).
    /// Thread-local rather than an `Interpreter` field for one reason: B5
    /// requires it to be restored on **every** exit edge of
    /// `forge.enter(b) { … }` — normal exit, `return`, `break`, `continue`
    /// and a `?` early return — and the only shape that covers all five
    /// without a restore line per path is RAII (`BudgetGuard`, below), whose
    /// `Drop` has no `&mut Interpreter` to write a field through. It is
    /// reset to 0 by `Interpreter::new`, so a fresh interpreter on a reused
    /// thread (`dmc test` runs every file on one) always starts at the root.
    static CUR_BUDGET: Cell<usize> = const { Cell::new(0) };
}

/// #674 slice 3d: read the entered-budget register. `0` is `forge.root`.
#[inline(always)]
fn cur_budget() -> usize { CUR_BUDGET.with(|c| c.get()) }

/// #674: while live, the entered
/// budget is the one this guard was opened on; on drop — on **every** exit
/// edge, the `?`/carrier ones included — the previous budget and its stamp
/// come back. The design names the trap this shape exists to avoid in as
/// many words: `Expr::ArenaBlock`'s own arm writes `self.eval_block(&ab.
/// body)?`, and a copy of that shape with a restore line after it leaks the
/// register whenever the body raises or escapes. Sibling of
/// `VaultTagGuard`, which is the same discipline over the stamp alone.
struct BudgetGuard {
    saved_index: usize,
    saved_tag: OriginTag,
}
impl BudgetGuard {
    /// Push `index` (and the stamp every tensor constructed under it
    /// inherits — `None` for the root, which can never go stale, §4.1).
    fn enter(index: usize, tag: OriginTag) -> Self {
        let saved_index = CUR_BUDGET.with(|c| c.replace(index));
        let saved_tag = CUR_BUDGET_TAG.with(|c| c.replace(tag));
        BudgetGuard { saved_index, saved_tag }
    }
}
impl Drop for BudgetGuard {
    fn drop(&mut self) {
        // The OUTER budget, not the root: an `enter` nested inside another
        // `enter` restores what it shadowed (the nested-enter restore rule above).
        CUR_BUDGET.with(|c| c.set(self.saved_index));
        CUR_BUDGET_TAG.with(|c| *c.borrow_mut() = self.saved_tag.take());
    }
}

/// #674: while live,
/// holds `CUR_BUDGET_TAG` cleared to `None`. `try_arena_constructor`'s Vault
/// arm keeps one for the rest of the call — see the call site's doc — and
/// restores the PREVIOUS tag on drop, covering every exit (an early
/// `return`, a `?` from `charge_ctor`/shape validation) the same way a
/// `MutexGuard` would.
struct VaultTagGuard(OriginTag);
impl VaultTagGuard {
    fn new() -> Self {
        VaultTagGuard(CUR_BUDGET_TAG.with(|c| c.borrow_mut().take()))
    }
}
impl Drop for VaultTagGuard {
    fn drop(&mut self) {
        CUR_BUDGET_TAG.with(|c| *c.borrow_mut() = self.0.take());
    }
}

/// #674 slice 3b: true iff `t`'s stamped generation no longer matches its
/// origin budget's live generation — the budget was reaped after `t` was
/// stamped (or cloned from a stamped tensor). A pure predicate: no table
/// lookup, no `&Interpreter`, so it runs anywhere there is an error channel —
/// or, for the two walkers (`Debug`, `json_encode_value`), anywhere there
/// isn't one.
#[inline(always)]
fn tensor_is_stale(t: &TensorVal) -> bool {
    match &t.origin {
        Some(tag) => tag.generation != tag.live.get(),
        None => false,
    }
}

/// #674 slice 3b: the `budget-stale` error, built out of line. Every caller
/// is one branch off a hot path — `eval_expr`'s per-node check above all —
/// so the `String` allocation, the `Span` and the `RuntimeError` all belong
/// behind a `#[cold]` call the branch predictor never takes. `#[inline(never)]`
/// as well as `#[cold]`, because `eval_expr` is `#[inline(always)]` and would
/// otherwise carry this into every one of its callers.
#[cold]
#[inline(never)]
fn stale_error(span: Span) -> RuntimeError {
    RuntimeError::at(
        "budget-stale: this tensor was allocated in a budget that \
         has already been reaped".to_string(),
        span,
    )
}

/// #674: re-stamp `v`'s origin to whatever
/// budget is current AT THIS COPY SITE, reading the same ambient thread-local
/// `TensorVal::new` stamps a fresh construction from. `#[derive(Clone)]`
/// would otherwise carry the SOURCE tensor's tag into a private copy
/// (`let !y = x`, a `!`-param copy-in) that `charge_intermediate` charges as
/// a fresh materialization — the copy's origin must match what it is charged
/// to, not what `x` was built under, or a copy taken OUT of a budget before
/// it is reaped would incorrectly go stale (or a copy taken at the root of a
/// tensor built inside a budget would incorrectly survive its reap).
fn restamp_current_budget(v: &mut Value) {
    if let Value::Tensor(t) = v {
        t.origin = CUR_BUDGET_TAG.with(|c| c.borrow().clone());
    }
}

/// A tensor value: f64-backed storage plus a coarse dtype tag. Derefs to the
/// underlying `ArrayD<f64>` so the bulk of the interpreter (shape queries,
/// iteration, elementwise math) operates on it unchanged.
#[derive(Clone)]
pub struct TensorVal {
    pub data: ArrayD<f64>,
    pub dtype: DType,
    /// #674 slice 3b: `None` means this tensor was constructed under the
    /// root (never stale — root can never be reaped); `Some(tag)` means it
    /// was constructed under a budget `carve`d from the root, and
    /// `check_live` (every retrieval site the interpreter checks for a live
    /// budget handle) reports `budget-stale` once that budget is reaped.
    pub origin: OriginTag,
}

impl TensorVal {
    pub fn new(mut data: ArrayD<f64>, dtype: DType) -> Self {
        // #226/#241: F32 tensors are kept f32-rounded. This is the single
        // construction chokepoint, so every op output / literal / constructor
        // inherits the rounding — which makes elementwise chains round per-op,
        // bit-exact with the JIT's native-f32 `+ - * /` (f64's 53-bit mantissa
        // exceeds the 2p+2 double-rounding bound for f32's 24). Int/Trit are
        // never rounded (an i64 tensor element must round-trip exactly); F64
        // keeps full width, matching the JIT's f64 tensors.
        if dtype.rounds_as_f32() {
            quantize_f32_arr(&mut data);
        }
        // #674 slice 3b: stamp from the ambient thread-local (§3.2 M1) — the
        // single construction chokepoint, so every op output / literal /
        // constructor inherits the stamp the same way it inherits f32
        // rounding above.
        let origin = CUR_BUDGET_TAG.with(|c| c.borrow().clone());
        TensorVal { data, dtype, origin }
    }
    /// Width a scalar read out of this tensor carries (#473). Elements of an
    /// `F32` tensor are already f32-rounded by #241's store-time rule; the
    /// width is what keeps them f32 through the arithmetic that follows.
    pub fn float_width(&self) -> FW {
        if self.dtype.rounds_as_f32() { FW::F32 } else { FW::F64 }
    }
}

impl std::ops::Deref for TensorVal {
    type Target = ArrayD<f64>;
    fn deref(&self) -> &ArrayD<f64> { &self.data }
}
impl std::ops::DerefMut for TensorVal {
    fn deref_mut(&mut self) -> &mut ArrayD<f64> { &mut self.data }
}
/// Default conversion tags the tensor `Float` — the historical behavior.
/// Integer sources construct `TensorVal::new(.., DType::Int)` explicitly.
impl From<ArrayD<f64>> for TensorVal {
    fn from(data: ArrayD<f64>) -> Self { TensorVal::new(data, DType::F32) }
}
impl fmt::Display for TensorVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.data) }
}

/// Runtime value. Tensors are normalized to f64 internally (pre-alpha
/// pragma: avoid an enum-of-typed-tensors until we have a typed-AST), with a
/// coarse `DType` tag so integer element reads round-trip as integers.
#[derive(Clone)]
pub enum Value {
    /// #540: a scalar integer carries its declared WIDTH, the way `Float`
    /// carries `FW`. Without it an `i32` annotation is inert at run time: the
    /// interpreter computes every integer in i64 while the JIT wraps its
    /// `cl::I32`, so the two backends disagree on `i32` overflow.
    Int(i64, IW),
    /// #473: a scalar float carries its declared WIDTH, the way `TensorVal`
    /// carries `DType`. Without it an `f32` annotation is inert at runtime: the
    /// interpreter computes every float in f64 while the JIT computes true
    /// f32, so the two backends disagree on `0.1f32 + 0.2f32`.
    Float(f64, FW),
    Bool(bool),
    Str(String),
    Nil,
    Tensor(TensorVal),
    Tuple(Vec<Value>),
    /// Named-field aggregate (model instance or @grad bundle).
    /// Wrapped in Rc<RefCell<...>> so all clones (including `self` passed to
    /// methods) share the same field storage — mutations persist to the caller.
    Struct(Rc<RefCell<Vec<(String, Value)>>>),
    /// Mutable reference to a named field inside a shared struct.
    /// Created by `let !alias = recv.field` when recv is a model instance;
    /// indexed writes (`alias[i] = val`) propagate back through the Rc.
    FieldRef {
        rc: Rc<RefCell<Vec<(String, Value)>>>,
        field: String,
    },
    /// `..` / `..=` range — stored as inclusive-end semantics
    Range { start: i64, end: i64, inclusive: bool },
    /// User-defined function (top-level decl, captured by reference).
    /// Pre-alpha: no closures, just a name lookup at call time.
    Fn(String),
    /// User-defined function with explicit shape params pre-bound (from `f[N]` syntax).
    BoundFn { name: String, shape_bindings: Vec<(String, i64)> },
    /// Anonymous function literal with captured environment.
    /// `captured_env` is a snapshot of every local binding visible at the
    /// point the `fn(...)` literal was evaluated. Values are cloned at capture
    /// time (value semantics — not reference capture).
    Lambda {
        lit: std::sync::Arc<crate::ast::FnLit>,
        captured_env: HashMap<String, Value>,
        /// The file the literal was written in; its body resolves top-level
        /// names there, not where it is eventually called.
        module: usize,
    },
    /// Built-in stdlib stub.
    Builtin(String),
    /// A C-like enum value (#336): the enum's name and the variant name. Tagged
    /// (vs. a bare `Int`) so a type-erased `match` can resolve bare-variant
    /// patterns and so two distinct enums never compare equal. The i64 ordinal
    /// (for `as i64`) is the variant's index in the enum's registry entry.
    ///
    /// `payload` (#350 Part 2) holds a payload variant's positional data
    /// (`Shape.Circle(2.0)` → `[Float(2.0)]`); empty for a tag-only variant.
    EnumVal { enum_name: String, variant: String, payload: Vec<Value> },
    /// Opaque value (model instances, arena handles, etc.) — for things
    /// the interpreter doesn't model yet.
    Opaque(String),
    /// Dynamic list — heterogeneous ordered collection.
    ///
    /// A VALUE type (unlike `Map` below): `let ys = xs` snapshots, and no
    /// mutation of `xs` is ever visible through `ys`. The `Rc` is a
    /// copy-on-write buffer, not shared mutable state — every writer goes
    /// through `Rc::make_mut`, which clones the backing `Vec` the moment more
    /// than one value holds it. Sharing is therefore unobservable; it only
    /// buys us the case where the buffer *is* uniquely owned, and then the
    /// write lands in place. That is what makes the `xs = list_push(xs, v)`
    /// build-a-list loop O(1) amortized instead of O(n) per push (#452).
    /// Use `Value::list(vec)` to construct.
    List(Rc<Vec<Value>>),
    /// Dynamic map — string-keyed heterogeneous collection.
    /// Wrapped in Rc<RefCell<...>> so all clones share the same backing
    /// HashMap — mutations (map_set, map_del) are visible through every copy,
    /// matching the JIT's pointer/reference semantics.
    Map(Rc<RefCell<HashMap<String, Value>>>),
    /// Module reference
    Module { alias: String, path: std::path::PathBuf },
    /// #674: a host-lifetime budget handle — one opaque machine word, the
    /// descriptor `(gen << 20) | index`. No arithmetic, no ordering, no
    /// literal; `Clone` copies the
    /// descriptor, and both copies name the SAME budget (`MEMORY.md §4`'s
    /// CoW does not apply — there is no buffer here to copy). `forge.root`
    /// evaluates to `Value::Budget(0)` (index 0, generation 0 — the root is
    /// never reaped, so its generation never moves).
    Budget(i64),
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // #597: a `u64` at or above 2^63 is a negative `i64` pattern —
            // render it as the unsigned magnitude it names, not the signed
            // reinterpretation (`0u64 - 1u64` prints 18446744073709551615, not
            // -1). Every other width's pattern is already non-negative, so
            // this only changes `U64`'s output.
            Value::Int(n, IW::U64) => write!(f, "{}", *n as u64),
            Value::Int(n, _)   => write!(f, "{}", n),
            // #473: an f32 renders as the f64 expansion of its f32-rounded
            // value ("12.368000030517578"), NOT the shortest f32 form
            // ("12.368"). That is #368's rule, it is what f32 *tensors* have
            // always printed, and the JIT widens f32→f64 before formatting to
            // match — so the same dtype renders the same way whether it sits in
            // a scalar or an element. Pinned by
            // `jit::tests::as_cast_derived_f32_str_matches_interp`.
            Value::Float(x, _) => write!(f, "{}", x),
            Value::Bool(b)  => write!(f, "{}", b),
            Value::Str(s)   => write!(f, "{:?}", s),
            Value::Nil      => write!(f, "nil"),
            // #674: a
            // `fmt` has no error channel and needs none — a rendering that is
            // not a number cannot BE a wrong number, so a stale tensor prints
            // a marker instead of stale bytes.
            Value::Tensor(t) if tensor_is_stale(t) => write!(f, "Tensor<reaped>"),
            Value::Tensor(t) => write!(f, "Tensor{:?} {}",  t.shape(), t),
            Value::Tuple(t) => {
                write!(f, "(")?;
                for (i, v) in t.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", v)?;
                }
                write!(f, ")")
            }
            Value::Struct(fields) => {
                write!(f, "{{ ")?;
                for (i, (k, v)) in fields.borrow().iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    // The hidden tag holds the model's table key; print the
                    // name the program wrote.
                    match v {
                        Value::Str(tag) if k == "__model__" =>
                            write!(f, "{}: {:?}", k, bare_name(tag))?,
                        _ => write!(f, "{}: {:?}", k, v)?,
                    }
                }
                write!(f, " }}")
            }
            Value::FieldRef { rc, field } => {
                let borrowed = rc.borrow();
                match borrowed.iter().find(|(k, _)| k == field) {
                    Some((_, v)) => write!(f, "&{}={:?}", field, v),
                    None => write!(f, "&{}=<undefined>", field),
                }
            }
            Value::Range { start, end, inclusive } => {
                write!(f, "{}..{}{}", start, if *inclusive { "=" } else { "" }, end)
            }
            Value::Fn(n)      => write!(f, "<fn {}>", bare_name(n)),
            Value::BoundFn { name, .. } => write!(f, "<fn {}>", bare_name(name)),
            Value::Lambda { .. } => write!(f, "<lambda>"),
            Value::Builtin(n) => write!(f, "<builtin {}>", n),
            Value::EnumVal { enum_name, variant, payload } => {
                write!(f, "{}.{}", enum_name, variant)?;
                if !payload.is_empty() {
                    let inner: Vec<String> = payload.iter().map(|v| v.to_string()).collect();
                    write!(f, "({})", inner.join(", "))?;
                }
                Ok(())
            }
            Value::Opaque(n)  => write!(f, "<opaque {}>", bare_name(n)),
            Value::List(vs) => {
                write!(f, "[")?;
                for (i, v) in vs.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", v)?;
                }
                write!(f, "]")
            }
            Value::Map(m) => {
                write!(f, "{{")?;
                for (i, (k, v)) in m.borrow().iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {:?}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Module { alias, path } => write!(f, "<module {} {:?}>", alias, path),
            Value::Budget(d) => write!(f, "<budget #{} gen {}>", budget_index(*d), budget_generation(*d)),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Str(s) => write!(f, "{}", s),  // unquoted for `print`
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Unwrap a list's copy-on-write buffer into an owned `Vec`.
///
/// Moves the buffer out when this handle is the only one left, and falls back
/// to a clone when the list is still shared — so consuming a freshly built or
/// uniquely owned list costs nothing, and a shared one keeps value semantics.
fn into_list_vec(rc: Rc<Vec<Value>>) -> Vec<Value> {
    Rc::try_unwrap(rc).unwrap_or_else(|rc| (*rc).clone())
}

impl Value {
    /// Construct a `Float`-tagged tensor value (the common case).
    fn tensor(data: ArrayD<f64>) -> Value { Value::Tensor(TensorVal::new(data, DType::F32)) }
    /// Construct a tensor value with an explicit dtype tag.
    fn tensor_dt(data: ArrayD<f64>, dtype: DType) -> Value { Value::Tensor(TensorVal::new(data, dtype)) }
    /// Construct a list value from an owned `Vec` (wraps it in the COW buffer).
    pub fn list(items: Vec<Value>) -> Value { Value::List(Rc::new(items)) }
    fn as_int(&self) -> Option<i64> {
        match self { Value::Int(n, _) => Some(*n), _ => None }
    }
    fn as_float(&self) -> Option<f64> {
        match self {
            // #597: a `u64` value carries the 64-bit PATTERN in an i64-backed
            // slot, so `*n as f64` read u64::MAX's all-ones pattern as -1 —
            // `print(x as f64)` printed `-1` instead of 1.8446744073709552e19,
            // and every f64-mediated builtin (min/max, the mixed int/float
            // comparison path) inherited the same sign error. The narrower
            // unsigned widths are stored zero-extended, so their signed and
            // unsigned readings already coincide and this arm never fires.
            Value::Int(n, IW::U64) => Some(*n as u64 as f64),
            Value::Int(n, _) => Some(*n as f64),
            Value::Float(x, _) => Some(*x),
            _ => None,
        }
    }
    fn as_bool(&self) -> Option<bool> {
        match self { Value::Bool(b) => Some(*b), _ => None }
    }
    fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_, _)     => "int",
            Value::Float(_, _)   => "float",
            Value::Bool(_)    => "bool",
            Value::Str(_)     => "str",
            Value::Nil        => "nil",
            Value::Tensor(_)  => "Tensor",
            Value::Tuple(_)   => "tuple",
            Value::Struct(_)  => "struct",
            Value::FieldRef { .. } => "field-ref",
            Value::Range { .. } => "range",
            Value::Fn(_)      => "fn",
            Value::BoundFn { .. } => "fn",
            Value::Lambda { .. } => "fn",
            Value::Builtin(_) => "builtin",
            Value::EnumVal { .. } => "enum",
            Value::Opaque(_)  => "opaque",
            Value::List(_)    => "list",
            Value::Map(_)     => "map",
            Value::Module { .. } => "module",
            Value::Budget(_)  => "Budget",
        }
    }
}

// ─── Control-flow signals ────────────────────────────────────────────────────

/// Statements can produce control-flow signals — `break`, `continue`,
/// `return`. We thread them up through eval_block. Each signal carries the
/// span of the keyword that raised it (#639), so a diagnostic reached far from
/// that keyword — the module-level backstop, say — can still point at it.
#[derive(Debug, Clone)]
enum Flow {
    Normal(Value),
    Break(Span),
    Continue(Span),
    Return(Value, Span),
}

/// #639: the part of a `Flow` that can *escape a block expression*. A block's
/// `Normal` case is a value, not control flow, so it is not representable
/// here — which is why nothing downstream (`escaping_flow`'s message table,
/// `Escape::into_flow`) needs an unreachable fourth arm.
#[derive(Debug, Clone)]
enum Escape {
    Return(Value, Span),
    Break(Span),
    Continue(Span),
}

impl Escape {
    fn into_flow(self) -> Flow {
        match self {
            Escape::Return(v, sp) => Flow::Return(v, sp),
            Escape::Break(sp) => Flow::Break(sp),
            Escape::Continue(sp) => Flow::Continue(sp),
        }
    }

    /// The keyword, for a diagnostic that names it.
    fn keyword(&self) -> &'static str {
        match self {
            Escape::Return(..) => "return",
            Escape::Break(_) => "break",
            Escape::Continue(_) => "continue",
        }
    }

    fn span(&self) -> &Span {
        match self {
            Escape::Return(_, sp) | Escape::Break(sp) | Escape::Continue(sp) => sp,
        }
    }
}

impl Flow {
    /// Split a `Flow` into a block expression's VALUE, or the escape that has
    /// to leave that expression on the carrier. Total by construction: `Normal`
    /// is the only case that is a value.
    fn into_value_or_escape(self) -> Result<Value, Escape> {
        match self {
            Flow::Normal(v) => Ok(v),
            Flow::Return(v, sp) => Err(Escape::Return(v, sp)),
            Flow::Break(sp) => Err(Escape::Break(sp)),
            Flow::Continue(sp) => Err(Escape::Continue(sp)),
        }
    }
}

// ─── Op profiling ────────────────────────────────────────────────────────────

#[derive(Default, Debug)]
pub struct OpProfile {
    pub tensor_ops: u64,
    pub tensor_elements: u64,
    pub scalar_ops: u64,
    pub fn_calls: u64,
    pub allocs: u64,
}

#[derive(Clone)]
pub struct InterpModuleEnv {
    pub scopes: Vec<HashMap<String, Value>>,
    pub fns: HashMap<String, FnDecl>,
    pub models: HashMap<String, String>,
    pub enums: HashMap<String, Vec<String>>,
    pub type_aliases: HashMap<String, TypeAlias>,
    pub public_items: std::collections::HashSet<String>,
    /// The module number the snapshotted file was loaded under; an importer
    /// finds the file's `pub` items under `module_key(epoch, name)`.
    pub epoch: usize,
}

// ─── Interpreter ─────────────────────────────────────────────────────────────

/// Recursive-call ceiling for the tree-walking interpreter. Each user call burns
/// ~100 KB of native stack across its eval sub-frames; the interpreter runs on a
/// 256 MB thread (see `main.rs` / `RUST_MIN_STACK` in `.cargo/config.toml`), so the
/// hard native ceiling is ~2,500 calls. We trip well below that so a runaway
/// recursion surfaces as a catchable error instead of a SIGABRT. Raising this
/// REQUIRES raising the interpreter thread stack proportionally.
const MAX_CALL_DEPTH: usize = 2000;

/// RAII guard for `Interpreter::call_depth`. Created on every user-fn / lambda
/// entry; its `Drop` decrements the counter so `?` early-returns and panics on the
/// error path can't desync it. Holds a cloned `Rc<Cell<…>>` rather than borrowing
/// the interpreter, so the call body can still use `&mut self` freely.
struct CallDepthGuard(Rc<Cell<usize>>);

impl Drop for CallDepthGuard {
    fn drop(&mut self) {
        self.0.set(self.0.get().saturating_sub(1));
    }
}

/// The outcome of trying to read a postfix bracket as a shape-bracketed method
/// call (#474). Either it was one and has been called, or it was not — and the
/// receiver, which had to be evaluated to find that out, comes back so the
/// ordinary postfix path can reuse it rather than evaluate it a second time.
enum BracketedCall {
    Dispatched(Value),
    NotAMethod(Value),
}

pub struct Interpreter {
    /// Lexical scope stack: each scope a name→value map.
    scopes: Vec<HashMap<String, Value>>,
    /// #517: parallel to `scopes` — the subset of each frame's names that
    /// were bound as SHAPE PARAMETERS (`fn f[S](...)`, inferred from an arg's
    /// tensor shape, or a model's `__shape_P__` field forwarded into a
    /// method call) rather than an ordinary `let`/`for`/param binding. A
    /// plain `HashMap<String, Value>` can't tell the two apart after the
    /// fact — `bind_shape_param` tags the name here at the moment it's
    /// bound. Used only to match the JIT's `shape_env` (jit.rs): a shape
    /// param resolves to a compile-time constant there, so a slice bound
    /// built from one must be classified the same way here (`affine_form`),
    /// not treated as a runtime term the way an ordinary variable is.
    shape_param_scopes: Vec<std::collections::HashSet<String>>,
    /// Lexical scope stack for declared streaming axes on KV-like bindings.
    stream_axes: Vec<HashMap<String, usize>>,
    /// Declared capacity (max streaming-axis length) for each KV binding.
    /// Populated from the `capacity = N` arg in `stream.kv[...](capacity = N)`.
    kv_capacities: HashMap<String, usize>,
    /// Top-level fn declarations (collected from program before any eval).
    fns: HashMap<String, FnDecl>,
    /// Top-level model declarations — referenced as opaque values at
    /// runtime (e.g. `Transformer[L=24, ...]`), so we just need the name set.
    /// Model names as each file sees them → the declaring file's key. A
    /// `pub model` imported under two spellings (`Point`, `math.Point`) maps
    /// both to one canonical key, which is what instances carry in their
    /// `__model__` tag and what `model_methods` / `model_fields` /
    /// `model_shape_params` are keyed by.
    models: HashMap<String, String>,
    /// Shape parameter names for each model, used to bind them into method scopes.
    /// e.g. `model S[P]` → `model_shape_params["S"] = ["P"]`
    model_shape_params: HashMap<String, Vec<String>>,
    /// Methods for each model, keyed by model name then method name.
    model_methods: HashMap<String, HashMap<String, FnDecl>>,
    /// Field declarations (name + type, in declaration order) for each model.
    /// Used by `@cast(Model){bytes}` to overlay a raw byte tensor onto a
    /// struct: it needs the field types to compute byte offsets and sizes.
    model_fields: HashMap<String, Vec<(String, crate::ast::Type)>>,
    /// #336: top-level enum declarations — ordered variant names, so a
    /// variant's value is its index here (the i64 ordinal).
    ///
    /// Keyed by `module_key`, exactly as `fns` and `models` are: an enum is a
    /// per-file name, and a file sees only the enums it declares plus the
    /// `pub` ones it imports. One flat map let two files' private `enum E`
    /// share a slot, so the last file loaded decided what `E`'s variants were
    /// everywhere — `E.Gamma` in the file that declares it reported `undefined
    /// identifier E` when another file's two-variant `E` had overwritten it.
    enums: HashMap<String, Vec<String>>,
    /// Top-level `type` aliases (SPEC §6.6) — name → declaration, collected
    /// in `collect_item` exactly as `enums` is. An alias is transparent: a
    /// value annotated `Role` where `type Role = u8` is a `u8`, and SPEC §3.1
    /// makes a declared integer type a run-time WIDTH. Every site where an
    /// annotation becomes semantics (`coerce_declared`, `check_declared_port`)
    /// resolves through this table first; before it existed, `let r: Role =
    /// 255` kept `r` a full i64 and `r + 1` was 256 under `dmc run` while the
    /// checker — which already resolves aliases — had accepted the program as
    /// the `u8` one that wraps to 0.
    ///
    /// Keyed by `module_key`, exactly as `fns`, `models` and `enums` are: SPEC
    /// §9 states ONE visibility rule for every top-level name, so an alias is
    /// visible where a fn of the same provenance would be — in the file that
    /// declares it, and in a file that imported it because it was `pub`. One
    /// flat map made the last-loaded declaration win everywhere, so an
    /// importer's alias silently re-typed the fn bodies of the library it
    /// imported — `libbump(255)` ran as `u8` and printed 0 where the checker
    /// had type-checked it as `i64` and meant 256 (and the reverse orientation
    /// printed 256 for the `u8` the checker had accepted).
    ///
    /// #623 first fixed that with a per-name list filtered by load order
    /// (`module <= alias_module`, highest wins). That admitted an alias a file
    /// never imported, merely one declared earlier in the topological order;
    /// registering it on import instead is the same rule the other three
    /// tables follow, and it needs no second module field.
    type_aliases: HashMap<String, TypeAlias>,
    /// Module-level mutable bindings (`let !w = …` / `let mut w = …`), in
    /// declaration order. A `@grad fn` that reads one of these captures it, and
    /// AUTODIFF.md §2 makes that capture a differentiable input: `call_grad`
    /// tapes it and returns its adjoint in `Grads` under this name (#398).
    /// Order is the source order of the `let`s, which fixes the order the
    /// capture fields follow the `!` params in `Grads`.
    ///
    /// Each entry is tagged with the **module** that declared it. One
    /// `Interpreter` loads every file in the dependency graph through repeated
    /// `load_program`, so a single flat name list let one module's `let !alpha`
    /// make an unrelated module's *immutable* `alpha` differentiable — a
    /// `Grads` field for a binding no checker rule ever admitted. The checker
    /// never had that hole: an imported `pub let !` is not even in scope as a
    /// bare name, so a `@grad fn` can only ever capture its **own** module's
    /// mut bindings. The tag is what lets the tape agree with that.
    mut_globals: Vec<(usize, String)>,
    /// The module whose code is executing right now — the one quantity, in one
    /// field. Every top-level name a body mentions resolves through
    /// `module_key(cur_module, name)`: a file's own items and its imports are
    /// registered under the file's number, so a file-private `helper` in two
    /// files never shares a slot (SPEC §6.7).
    ///
    /// `load_program` mints the next number into it (one per file, in the
    /// resolver's leaf-first order) and it stays there while the file's items
    /// are collected and its top-level `let`s run; every call then sets it to
    /// the `module` its decl carries and restores it on the way out.
    cur_module: usize,
    /// xorshift64 RNG state, advanced by every `rng.normal[...]` /
    /// `rng.uniform[...]`. Seeded from `Rng.seed(N)` at the call site; this
    /// global is the fallback so programs that never call seed still get a
    /// reproducible (per-process) random stream rather than zeros.
    rng_state: u64,
    /// Monotonic start time for `time_ms()`.
    start_time: std::time::Instant,
    /// Command-line arguments passed to the program (for `argv()`).
    argv: Vec<String>,
    /// #402: the open process ports of this run. The registry — child process,
    /// wire framing, and PORTS.md §6 tags — lives in `crate::ports`, shared with
    /// the JIT so the two backends cannot drift on the protocol.
    ports: crate::ports::PortRegistry,
    /// Optional op-count profile; Some only when --profile is active.
    pub profile: Option<OpProfile>,
    pub interp_modules: HashMap<std::path::PathBuf, InterpModuleEnv>,
    pub public_items: std::collections::HashSet<String>,
    /// CPU / ISA features detected at startup; used by `@host match` arm dispatch.
    host_features: std::collections::HashSet<String>,
    /// Names of extern fn declarations; calls produce a runtime error (no JIT backend yet).
    extern_fns: std::collections::HashSet<String>,
    /// Write-back slots populated by the most recent `call_fn_with_shapes` call.
    /// Each entry is `(param_index, final_value)` for every `!`-prefixed parameter.
    /// The call site reads this immediately after the call and writes the values
    /// back to the corresponding caller-side identifier (if the arg was a bare ident).
    pending_writebacks: Vec<(usize, Value)>,
    /// #474: the receiver of a shape-bracketed postfix call, evaluated once.
    ///
    /// `recv.name[…](args)` cannot be told apart from `recv.field[i](args)`
    /// without knowing what `recv` *is*, so the dispatch has to evaluate it
    /// before it can decide. When the answer is "not a method after all", the
    /// ordinary postfix path takes over — and used to walk the very same
    /// receiver expression a second time. That is not a slower route to the
    /// same answer: `get(h).subs[0](5)` called `get` twice, printing twice and
    /// allocating a second receiver in the arena.
    ///
    /// So the value is parked here instead, keyed by the identity of the AST
    /// node it came from, and the one `Field` evaluation that follows takes it.
    /// Node addresses are stable and unique for the program's lifetime, and the
    /// only way to reach that `Field` is through the call expression that parks
    /// the value — which re-parks a fresh one on every evaluation — so a parked
    /// value can never be read by a later run of the same expression.
    parked_receiver: Option<(*const Expr, Value)>,
    /// Current recursive-call depth, bumped via `CallDepthGuard` on every user-fn /
    /// lambda entry. Trips `MAX_CALL_DEPTH` with a catchable error before the native
    /// stack is exhausted. `Rc<Cell<…>>` so the guard can decrement on drop without
    /// holding a borrow of the interpreter across the call body.
    call_depth: Rc<Cell<usize>>,
    /// #452: name of the binding that the statement in flight will overwrite
    /// with the result of the builtin call being evaluated. Set only for the
    /// exact shape `name = builtin(name, …)`, `None` everywhere else. The call
    /// takes this hint and, once its arguments are evaluated, drops that
    /// binding's handle on the list buffer it was handed — the binding is dead
    /// the moment the call returns, so releasing it early leaves the buffer
    /// uniquely owned and lets the write land in place instead of copying.
    /// See `release_move_target`.
    move_hint: Option<String>,
    /// #478 type-directed literal evaluation: the width an *unsuffixed* float
    /// literal should adopt in the value position about to be evaluated
    /// (SPEC.md §"Untyped numeric literals" — a bare literal takes the type of
    /// its context, including "the other operand at its use site"). Armed by
    /// `eval_if_flow` / `eval_match` from a sibling branch's static type and
    /// `take()`n at the top of `eval_expr` / `eval_block`, so it reaches
    /// exactly the position it was armed for. Mirrors `jit::Translator`'s
    /// `float_lit_hint` rule by rule: without it `dmc run` and `dmc jit`
    /// disagree on `if c { x_f32 } else { 0.1 }` whenever the else arm is the
    /// one taken — the checker calls that literal an f32, and only the JIT
    /// used to be told.
    float_lit_hint: Option<FW>,
    /// #400: per-arena byte budgets from `--vault=` / `--forge=` (`MEMORY.md
    /// §1.1`). `None` on an arena means unbounded.
    arena_limits: crate::arena::ArenaLimits,
    /// Bytes charged against each sized arena so far. The interpreter has no
    /// bump pointer — it hands every tensor to Rust's allocator — so these
    /// counters *are* the arena watermarks: they meter every materialized
    /// tensor — explicit `vault.*`/`forge.*` constructors AND implicit
    /// intermediates alike (#781) — at its DECLARED element width
    /// (`DType::byte_width`), not the interpreter's f64 storage width. Both
    /// counters are monotonic: nothing here reclaims memory, so nothing here
    /// returns budget. In particular a `forge { … }` block does not rewind
    /// `forge_used` on exit — see `Expr::ArenaBlock` (#400).
    ///
    /// `vault_used` stays a flat counter — Vault has no `carve` (§3.5). Forge
    /// gained per-budget storage in slice 3b: what was `forge_used` is now
    /// `budgets[0].used`, entry 0 of the table below.
    vault_used: u64,
    /// #674: the interpreter's budget table.
    /// Entry 0 is `forge.root`, sized by `--forge` (installed by
    /// `set_arena_limits`) — `budgets[0].used` is what `forge_used` used to
    /// be, and a program that never carves changes nothing (§3.1, the
    /// compatibility gate slices 3b/3c hold to). `forge.carve` pushes;
    /// `forge.reap` bumps a `gen`, credits the parent's `free`, and frees the
    /// entry for reuse (the descriptor's index, never the `Vec` slot itself).
    budgets: Vec<BudgetEntry>,
    /// #674 slice 3b: indices `reap` has freed and not yet handed back out.
    /// `carve` pops one before growing `budgets` — "entry freed for reuse"
    /// (§3.1) — reusing the SLOT's own `live` cell (already bumped past the
    /// generation the reaped occupant's tensors were stamped with), so a
    /// stale `Value::Budget` descriptor naming the old occupant can never
    /// match the new one's generation at the same index (§6 Q8's whole
    /// point: an index is reused, but never its generation).
    budget_free: Vec<usize>,
    // #674 slice 3d: the budget every allocation charges (PRE-A,
    // `charge_arena`'s Forge arm) used to be a field here. It is now the
    // `CUR_BUDGET` thread-local beside `CUR_BUDGET_TAG` — see that cell's
    // doc for why B5's five exit edges force RAII, and RAII forces the
    // register out of `&mut Interpreter`.
    /// #781 (reviewer rework): true while evaluating a tensor-literal
    /// expression whose VALUE is consumed by something other than being
    /// the tensor a `charge_intermediate` call should bill — a row inside
    /// an enclosing nested literal (`[[16 f32]×4]`: the inner `[16 f32]`
    /// materializes the row array, but only the assembled outer array is
    /// the tensor that survives; charging both double-bills it), or a
    /// bracket list read as a SHAPE (`a.reshape[[4096]]`: `[4096]` never
    /// becomes a tensor a program can observe — it is consumed for its
    /// integers and discarded, and the JIT never charges it either). Set
    /// around exactly those two call sites (`Expr::TensorLit`'s row loop,
    /// `reshape`'s shape-argument eval) and restored on every path,
    /// including an `Err` return, so nesting depth cannot leak it.
    lit_charge_suppressed: bool,
    /// #781 (reviewer rework): running total of the Forge bytes charged by an
    /// EXPLICIT constructor — a `forge.*` spelling, or an `rng.*` draw —
    /// since this interpreter started. Strictly monotonic, and never itself a
    /// budget: it exists only so a compiler-inserted rewind can subtract two
    /// snapshots of it and learn how much of the window it is about to
    /// release was spelled out by the program.
    ///
    /// A rewind exists to hand a body's compiler-inserted intermediates back:
    /// `MEMORY.md §1.1` returns budget only for a temporary *the compiler
    /// itself* inserted and reclaimed, and a tensor the program asked for by
    /// name is not one. A constructor's result can escape the body and stay
    /// live — assigned to a model field (`m.w = forge.zeros[…]`), written
    /// through a `!` param, stored into a module-level `let !` — at which
    /// point rewinding past it un-charges memory the program still holds. So
    /// both AD rewinds (the forward-only `@grad` mark in
    /// `call_fn_in_module`, and the `.fwd_bwd` epilogue) land at `mark +
    /// (now − then)` rather than at `mark`.
    ///
    /// A monotonic counter read at two points, rather than a scoped
    /// accumulator: nesting composes with no save/restore (an inner rewind's
    /// kept bytes are inside the outer window's difference too, which is
    /// correct — they are live under the outer scope as well), and no error
    /// path out of a marked body can leave a flag set.
    ///
    /// Conservative by construction: it keeps every constructor charged
    /// inside the window, whether or not that particular result escapes —
    /// so a scratch constructor used and dropped in the body over-charges
    /// its full size on every call, unbounded across calls (#796); a
    /// COMPUTED tensor that escapes is not kept (#795). Both resolve by
    /// tracking escapes at the assignment/return site.
    /// Over-keeping bills memory the program named; under-keeping loses
    /// memory the program still holds, which is the failure `MEMORY.md`'s
    /// budgets exist to prevent. There is no parity cost — the JIT refuses an
    /// explicit constructor inside a `@grad` body outright (measured: it
    /// refuses each of the three escape shapes), so no JIT number moves
    /// either way.
    ///
    /// `vault.*` needs no term here: `vault_used` is never rewound at all.
    ctor_forge_bytes: u64,
    /// #505: remaining `@comptime` step budget, or `None` when this
    /// interpreter is executing a program rather than folding a block. Armed
    /// by `eval_comptime` and cleared by it, so no ordinary run is ever
    /// metered. One step is one `eval_expr` entry.
    comptime_fuel: Option<u64>,
    /// #628 follow-up: the declared return type of every named `fn`/method
    /// whose body is currently on the (dynamic) call stack. Pushed and popped
    /// by `call_fn_with_shapes` around its `call_fn_in_module` call — the same
    /// wrapper shape #652 gave `cur_module`, so it covers every exit that
    /// function has rather than only the ones spelled out today.
    ///
    /// `call_lambda` reads the top of this stack when the lambda itself has
    /// no declared `(T, Err)` return type, so an untyped `?`-using lambda's
    /// early return can be rebuilt against the same "enclosing function"
    /// `check.rs`'s `current_fn_ret` means (`PostfixOp::Query`, `src/check.rs`):
    /// there, "enclosing" is LEXICAL — the named `fn` whose body textually
    /// contains the lambda literal, fixed once at check time regardless of
    /// where the lambda is later called from. This stack is a DYNAMIC stand-in:
    /// it agrees with the lexical answer whenever the lambda is invoked from
    /// within the extent of the named fn that (lexically) contains it — true
    /// for every corpus/test use (a lambda built and called inline) — but can
    /// disagree if a lambda escapes its definition site (returned or stored)
    /// and is later invoked while a *different* named fn is on the stack, or
    /// none at all. There is no lexical alternative available here: the
    /// interpreter has no per-`FnLit` "enclosing fn" table (unlike `check.rs`,
    /// which gets it for free from the AST recursion order), and building one
    /// was out of scope for this fix — `check.rs` already refuses every case
    /// where the *lexically* enclosing fn is not `(T, Err)` (`in_err_fn`), so
    /// the escape-hatch programs this stack could get wrong are ones the
    /// checker already accepted; this stack only has to agree with it, not
    /// invent legality on its own.
    named_fn_ret_stack: Vec<Option<Type>>,
}

struct AxisSelection {
    indices: Vec<usize>,
    reduce_rank: bool,
}

/// An index element whose sub-expressions have already been evaluated to
/// integers. Splitting index *evaluation* (which re-enters `eval_expr`) from
/// index *resolution* (pure arithmetic against the target's shape) lets a
/// caller finish all interpretation before it takes a `borrow_mut()` on the
/// destination's backing store. #443: `let !c = self.cells; c[self.table[0]] = 1`
/// used to evaluate the index under the field's live mutable borrow, so the
/// `self.table` read hit `RefCell already mutably borrowed` and aborted the
/// whole process — a panic, not a diagnostic.
#[derive(Clone)]
enum IndexVal {
    /// `..` — full axis
    FullSlice,
    /// a range expression in index position: `a..b` / `a..=b`. `dyn_extent`
    /// is `Some(n)` when `start`/`end` share the same runtime idents with
    /// matching coefficients (`i..i+1`), so the WIDTH `n` is fixed by
    /// construction even though `start` is only known at runtime (#517;
    /// mirrors `jit::classify_index`'s `DynRange` split). When set, the
    /// window `[start, start+n)` is bounds-checked and traps on OOB rather
    /// than being independently clamped — see `resolve_index_values`.
    Range { start: Option<i64>, end: Option<i64>, inclusive: bool, dyn_extent: Option<i64> },
    /// slice syntax with an optional step
    Slice { start: Option<i64>, end: Option<i64>, step: i64 },
    /// a scalar index (may be negative — normalized against the axis length)
    Index(i64),
}

impl IndexVal {
    /// Whether this element keeps its axis (used for the keepdims rule below).
    fn is_slice(&self) -> bool {
        !matches!(self, IndexVal::Index(_))
    }
}


/// Coerce an evaluated index sub-expression to an integer, naming the position
/// (`"tensor index"`, `"slice step"`, …) in the error.
fn index_int(val: &Value, what: &str, span: &Span) -> EvalResult<i64> {
    val.as_int().ok_or_else(|| RuntimeError::at(
        format!("{} must be integer, got {}", what, val.type_name()),
        span.clone(),
    ))
}

fn assign_scalar_to_selection(arr: &mut ArrayD<f64>, selections: &[AxisSelection], val: f64) {
    let mut target_shape = Vec::new();
    for sel in selections {
        if !sel.reduce_rank {
            target_shape.push(sel.indices.len());
        }
    }

    if target_shape.is_empty() {
        let mut src_coords = Vec::new();
        for sel in selections {
            src_coords.push(sel.indices[0]);
        }
        arr[IxDyn(&src_coords)] = val;
        return;
    }

    let mut target_coords = vec![0; target_shape.len()];
    let target_len: usize = target_shape.iter().product();
    for _ in 0..target_len {
        let mut src_coords = Vec::with_capacity(selections.len());
        let mut target_axis = 0;
        for sel in selections {
            if sel.reduce_rank {
                src_coords.push(sel.indices[0]);
            } else {
                src_coords.push(sel.indices[target_coords[target_axis]]);
                target_axis += 1;
            }
        }

        arr[IxDyn(&src_coords)] = val;

        // Advance target coordinate
        for axis in (0..target_shape.len()).rev() {
            target_coords[axis] += 1;
            if target_coords[axis] < target_shape[axis] {
                break;
            }
            target_coords[axis] = 0;
        }
    }
}

impl Interpreter {
    /// #517: the affine form `const + Σ coeff·ident` of a range-bound
    /// expression — used only to tell whether a `Range`'s `start` and `end`
    /// differ by a fixed constant (`i..i+1`: both are `1*i + k`), i.e.
    /// whether the slice's WIDTH is fixed by construction independent of
    /// any runtime value. Mirrors `jit::affine_index_expr`, INCLUDING its
    /// shape-env lookup: an identifier currently bound as a SHAPE PARAMETER
    /// (`is_shape_param`) resolves to its concrete value as a constant here
    /// too — exactly like the JIT's `shape_env` — rather than becoming a
    /// term. Without this, `fn f[S](x: Tensor[f32,[S]]) { x[S..S+1] }`
    /// classified as dynamic here but static in the JIT (which folds `S` to
    /// a literal at monomorphization), so the two backends picked different
    /// arms of `resolve_index_values` for the very same range. An ordinary
    /// `let`/`for`-bound identifier that merely holds an int right now is
    /// NOT a shape param and still becomes a term (coefficient 1) — that's
    /// what makes `i..i+1` with a loop var `i` dynamic. Conservative: `None`
    /// whenever the expression isn't affine in its idents.
    fn affine_form(&self, e: &Expr) -> Option<(i64, HashMap<String, i64>)> {
        match e {
            Expr::Literal(Literal::Int(n, _), _) => Some((*n, HashMap::new())),
            Expr::Ident(name, _) => {
                if self.is_shape_param(name) {
                    let v = self.lookup(name)?.as_int()?;
                    Some((v, HashMap::new()))
                } else {
                    Some((0, HashMap::from([(name.clone(), 1i64)])))
                }
            }
            // `(e)` — the parser builds a 1-tuple for grouping.
            Expr::Tuple(es, _) if es.len() == 1 => self.affine_form(&es[0]),
            Expr::UnOp { op: UnOp::Neg, operand, .. } => {
                let (c, mut t) = self.affine_form(operand)?;
                for v in t.values_mut() { *v = -*v; }
                Some((-c, t))
            }
            Expr::BinOp { op, lhs, rhs, .. } => {
                let (lc, lt) = self.affine_form(lhs)?;
                let (rc, rt) = self.affine_form(rhs)?;
                match op {
                    BinOp::Add | BinOp::Sub => {
                        let sign = if matches!(op, BinOp::Add) { 1 } else { -1 };
                        let mut t = lt;
                        for (k, v) in rt { *t.entry(k).or_insert(0) += sign * v; }
                        t.retain(|_, v| *v != 0);
                        Some((lc + sign * rc, t))
                    }
                    BinOp::Mul if rt.is_empty() => {
                        let mut t = lt;
                        for v in t.values_mut() { *v *= rc; }
                        t.retain(|_, v| *v != 0);
                        Some((lc * rc, t))
                    }
                    BinOp::Mul if lt.is_empty() => {
                        let mut t = rt;
                        for v in t.values_mut() { *v *= lc; }
                        t.retain(|_, v| *v != 0);
                        Some((lc * rc, t))
                    }
                    BinOp::Div if rt.is_empty() && lt.is_empty() && rc != 0 => {
                        Some((lc / rc, HashMap::new()))
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// #517: does `start..end` (`end` required, `start` optional — an
    /// omitted start is the literal `0`) have a WIDTH fixed by construction
    /// even though `start` may be a runtime expression? `Some(extent)` iff
    /// the two bounds' affine forms share identical ident terms (so
    /// subtracting them cancels every runtime ident, leaving a constant)
    /// AND that shared term set is non-empty (i.e. this isn't just two
    /// literals-or-shape-params — that case keeps the existing
    /// independent-clamp behavior, #291.4's JIT counterpart). Returns the
    /// extent with the `..=` bump already applied, floored at 0.
    fn dyn_range_extent(&self, start: Option<&Expr>, end: Option<&Expr>, inclusive: bool) -> Option<i64> {
        let end = end?;
        let sa = match start {
            Some(e) => self.affine_form(e),
            None => Some((0, HashMap::new())),
        };
        let ea = self.affine_form(end);
        match (sa, ea) {
            (Some((sc, st)), Some((ec, et))) if st == et && !st.is_empty() => {
                let mut extent = ec - sc;
                if inclusive { extent += 1; }
                Some(extent.max(0))
            }
            _ => None,
        }
    }

    /// Evaluate every expression inside an index into an integer, leaving a
    /// form `resolve_index_values` can consume without the interpreter. This
    /// is the only half that can re-enter `eval_expr`, so a caller that is
    /// about to hold a `RefCell` borrow of the destination must run it *first*
    /// (#443).
    fn eval_index_elems(&mut self, elems: &[IndexElem], span: &Span) -> EvalResult<Vec<IndexVal>> {
        let mut out = Vec::with_capacity(elems.len());
        for elem in elems {
            out.push(match elem {
                IndexElem::Expr(Expr::Range { start, end, inclusive, .. }) => {
                    // #517: classify BEFORE evaluating — the affine check is
                    // purely syntactic (over the un-evaluated bound exprs).
                    let dyn_extent = self.dyn_range_extent(start.as_deref(), end.as_deref(), *inclusive);
                    let start = match start {
                        Some(e) => Some(index_int(&self.eval_expr(e)?, "range start", span)?),
                        None => None,
                    };
                    let end = match end {
                        Some(e) => Some(index_int(&self.eval_expr(e)?, "range end", span)?),
                        None => None,
                    };
                    IndexVal::Range { start, end, inclusive: *inclusive, dyn_extent }
                }
                IndexElem::Expr(e) => {
                    IndexVal::Index(index_int(&self.eval_expr(e)?, "tensor index", span)?)
                }
                IndexElem::FullSlice(_) => IndexVal::FullSlice,
                IndexElem::Slice { start, end, step, .. } => {
                    let step = match step {
                        Some(e) => index_int(&self.eval_expr(e)?, "slice step", span)?,
                        None => 1,
                    };
                    let start = match start {
                        Some(e) => Some(index_int(&self.eval_expr(e)?, "slice start", span)?),
                        None => None,
                    };
                    let end = match end {
                        Some(e) => Some(index_int(&self.eval_expr(e)?, "slice end", span)?),
                        None => None,
                    };
                    IndexVal::Slice { start, end, step }
                }
            });
        }
        Ok(out)
    }

    fn resolve_index_selection(&mut self, arr_shape: &[usize], elems: &[IndexElem], span: Span) -> EvalResult<Vec<AxisSelection>> {
        let vals = self.eval_index_elems(elems, &span)?;
        Self::resolve_index_values(arr_shape, &vals, &span)
    }

    /// Turn pre-evaluated index values into per-axis selections. Takes no
    /// `self`: pure arithmetic against `arr_shape`, safe to call under a live
    /// borrow of the tensor being indexed.
    fn resolve_index_values(arr_shape: &[usize], vals: &[IndexVal], span: &Span) -> EvalResult<Vec<AxisSelection>> {
        let ndim = arr_shape.len();
        let mut selections = Vec::new();

        let mut padded_elems = vals.to_vec();
        while padded_elems.len() < ndim {
            padded_elems.push(IndexVal::FullSlice);
        }

        if padded_elems.len() > ndim {
            return Err(RuntimeError::at(
                format!("index has {} dims but tensor has {}", padded_elems.len(), ndim),
                span.clone(),
            ));
        }

        for (axis, elem) in padded_elems.iter().enumerate() {
            let dim = arr_shape[axis];
            match elem {
                IndexVal::Range { start, end: _, inclusive: _, dyn_extent: Some(extent) } => {
                    // #517: a dynamic start whose WIDTH is fixed by
                    // construction (`i..i+1`) is dynamic OOB per SPEC §4.3
                    // when the window `[start, start+extent)` doesn't fit
                    // the axis — a NEGATIVE start included; it does NOT
                    // resolve from the end the way a static bound does (no
                    // resolution preserves both the fixed extent and the
                    // value actually seen — see
                    // `jit::bounds_check_slice_start`'s doc comment for why
                    // the JIT makes the same call). The independent
                    // per-bound clamp below is for STATIC bounds only
                    // (#291.4) and does not run for this arm. Message
                    // matches `__dmc_slice_oob_trap` verbatim (jit.rs
                    // `dmc_slice_oob_trap`) so both backends emit the same
                    // diagnostic and exit(1).
                    let start_val = start.expect("dyn_extent implies a non-omitted start (#517)");
                    let extent = *extent;
                    if start_val < 0 || start_val + extent > dim as i64 {
                        return Err(RuntimeError::msg(format!(
                            "slice start {} with extent {} out of bounds for axis of size {}",
                            start_val, extent, dim
                        )));
                    }
                    let indices: Vec<usize> = (start_val..start_val + extent).map(|v| v as usize).collect();
                    selections.push(AxisSelection {
                        indices,
                        reduce_rank: false,
                    });
                }
                IndexVal::Range { start, end, inclusive, dyn_extent: None } => {
                    let start_val = *start;
                    let end_val = *end;

                    let start_normalized = start_val.unwrap_or(0);
                    let mut end_normalized = end_val.unwrap_or(dim as i64);
                    if *inclusive {
                        end_normalized += 1;
                    }

                    let normalize = |val: i64| -> i64 {
                        if val < 0 {
                            std::cmp::max(0, dim as i64 + val)
                        } else {
                            std::cmp::min(dim as i64, val)
                        }
                    };

                    let start_idx = normalize(start_normalized);
                    let end_idx = normalize(end_normalized);
                    let mut indices = Vec::new();
                    let mut curr = start_idx;
                    while curr < end_idx {
                        indices.push(curr as usize);
                        curr += 1;
                    }

                    selections.push(AxisSelection {
                        indices,
                        reduce_rank: false,
                    });
                }
                IndexVal::Index(n) => {
                    let n = *n;
                    let i = if n < 0 { (dim as i64 + n) as usize } else { n as usize };
                    if i >= dim {
                        return Err(RuntimeError::at(format!("index {} out of bounds for axis {} of size {}", i, axis, dim), span.clone()));
                    }
                    let has_before = padded_elems[..axis].iter().any(IndexVal::is_slice);
                    let has_after = padded_elems[axis + 1..].iter().any(IndexVal::is_slice);
                    let keepdims = has_before && has_after;

                    selections.push(AxisSelection {
                        indices: vec![i],
                        reduce_rank: !keepdims,
                    });
                }
                IndexVal::FullSlice => {
                    selections.push(AxisSelection {
                        indices: (0..dim).collect(),
                        reduce_rank: false,
                    });
                }
                IndexVal::Slice { start, end, step } => {
                    let step_val = *step;
                    if step_val == 0 {
                        return Err(RuntimeError::at("slice step cannot be zero".to_string(), span.clone()));
                    }

                    let start_val = *start;
                    let end_val = *end;

                    let start_normalized = start_val.unwrap_or(if step_val > 0 { 0 } else { dim as i64 - 1 });
                    let end_normalized = end_val.unwrap_or(if step_val > 0 { dim as i64 } else { -1 });

                    let normalize = |val: i64| -> i64 {
                        if val < 0 {
                            std::cmp::max(0, dim as i64 + val)
                        } else {
                            std::cmp::min(dim as i64, val)
                        }
                    };

                    let mut indices = Vec::new();
                    if step_val > 0 {
                        let start_idx = normalize(start_normalized);
                        let end_idx = normalize(end_normalized);
                        let mut curr = start_idx;
                        while curr < end_idx {
                            indices.push(curr as usize);
                            curr += step_val;
                        }
                    } else {
                        let start_idx = if start_normalized < 0 { dim as i64 + start_normalized } else { start_normalized };
                        let start_idx = std::cmp::max(0, std::cmp::min(dim as i64 - 1, start_idx));
                        let end_idx = if end_normalized < 0 && end_normalized != -1 { dim as i64 + end_normalized } else { end_normalized };

                        let mut curr = start_idx;
                        while curr > end_idx {
                            indices.push(curr as usize);
                            curr += step_val;
                        }
                    }

                    selections.push(AxisSelection {
                        indices,
                        reduce_rank: false,
                    });
                }
            }
        }

        Ok(selections)
    }

    fn read_tensor_selection(&self, arr: &ArrayD<f64>, selections: &[AxisSelection]) -> ArrayD<f64> {
        let mut out_shape = Vec::new();
        for sel in selections {
            if !sel.reduce_rank {
                out_shape.push(sel.indices.len());
            }
        }

        if out_shape.is_empty() {
            let mut out = ArrayD::zeros(IxDyn(&[]));
            let mut src_coords = Vec::new();
            for sel in selections {
                src_coords.push(sel.indices[0]);
            }
            out[IxDyn(&[])] = arr[IxDyn(&src_coords)];
            return out;
        }

        let mut out = ArrayD::zeros(IxDyn(&out_shape));
        let out_len = out.len();

        let mut coords = vec![0; out_shape.len()];
        for _ in 0..out_len {
            let mut src_coords = Vec::with_capacity(selections.len());
            let mut out_axis = 0;
            for sel in selections {
                if sel.reduce_rank {
                    src_coords.push(sel.indices[0]);
                } else {
                    src_coords.push(sel.indices[coords[out_axis]]);
                    out_axis += 1;
                }
            }

            out[IxDyn(&coords)] = arr[IxDyn(&src_coords)];

            for axis in (0..out_shape.len()).rev() {
                coords[axis] += 1;
                if coords[axis] < out_shape[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }

        out
    }

    /// The `type` declaration a name resolves to **from the module whose body
    /// is running** — the file's own declaration, or the `pub` one it
    /// imported, which is exactly what `resolve_fn` and `resolve_model` ask of
    /// their own tables. A name the running file neither declares nor imported
    /// is not an alias here, which is what stops an importer's `type Role =
    /// u8` from re-typing the library it imported.
    fn alias_decl<'a>(&'a self, name: &str) -> Option<&'a TypeAlias> {
        self.type_aliases.get(&module_key(self.cur_module, name))
    }

    /// Resolve a declared type through the top-level `type` aliases (SPEC
    /// §6.6) to the type it spells, chasing a chain (`type A = B  type B =
    /// u8`). Anything that is not an alias name comes back as itself.
    ///
    /// A shape-generic alias resolves only when its target is a SCALAR
    /// (`type W[N] = u8`): a shape argument cannot change a scalar's width, so
    /// `let r: W[3] = 255` is the `u8` the checker resolved it to and wraps to
    /// 0. It printed 256 before, which was the same #540 hole one indirection
    /// further out. A shape-generic alias over a shaped type
    /// (`type Batch[B] = Tensor[f32,[B,768]]`) is still left alone:
    /// substituting `B` here is separate work, and no run-time width hides
    /// behind it.
    ///
    /// The loop bound (rather than a seen-set) is enough to stop a cycle: the
    /// checker refuses a recursive alias (`check.rs::resolve_type_alias`), so
    /// this is only a cap for an interpreter run that skipped the check.
    fn resolve_alias_ty<'a>(&'a self, ty: &'a Type) -> &'a Type {
        let mut cur = ty;
        for _ in 0..=self.type_aliases.len() {
            match cur {
                Type::Named { name, args, .. } => match self.alias_decl(name) {
                    Some(alias)
                        if alias.shape_params.is_empty() && args.is_empty() => cur = &alias.ty,
                    Some(alias)
                        if !alias.shape_params.is_empty()
                            && matches!(alias.ty, Type::Scalar(_, _)) => cur = &alias.ty,
                    _ => break,
                },
                _ => break,
            }
        }
        cur
    }

    /// `coerce_scalar_width` behind alias resolution. This — not the free
    /// function — is what every annotation site calls (a typed `let`, a typed
    /// parameter, a declared return, a declared model field), so a width
    /// spelled through an alias narrows exactly as the direct spelling does.
    #[inline]
    fn coerce_declared(&self, v: Value, ty: Option<&Type>) -> Value {
        coerce_scalar_width(v, ty.map(|t| self.resolve_alias_ty(t)))
    }

    /// `check_port_lang` behind alias resolution, for the same sites and the
    /// same reason as `coerce_declared`: `type Py = Port[python]` names a
    /// runtime just as the direct spelling does.
    #[inline]
    fn check_declared_port(&self, v: &Value, ty: Option<&Type>, sp: &Span) -> EvalResult<()> {
        check_port_lang(v, ty.map(|t| self.resolve_alias_ty(t)), sp)
    }

    /// The declared type of `field` on a struct value, via its hidden
    /// `__model__` tag (#473). `None` when the value carries no model tag or
    /// the model declares no such field.
    fn struct_field_ty(
        &self,
        fields: &Rc<RefCell<Vec<(String, Value)>>>,
        field: &str,
    ) -> Option<crate::ast::Type> {
        let model = {
            let b = fields.borrow();
            match b.iter().find(|(k, _)| k == "__model__") {
                Some((_, Value::Str(n))) => n.clone(),
                _ => return None,
            }
        };
        self.model_fields.get(&model)?
            .iter()
            .find(|(n, _)| n == field)
            .map(|(_, t)| t.clone())
    }

    /// Store `rval` into `arr` at already-evaluated indices. Deliberately takes
    /// no `self`, so it cannot re-enter the interpreter: callers may hold a
    /// `RefCell` borrow of the destination across this call (#443).
    fn assign_to_tensor_resolved(arr: &mut ArrayD<f64>, dtype: DType, idx_vals: &[IndexVal], rval: Value, span: Span) -> EvalResult<()> {
        // #226/#241: element writes mutate the backing array directly
        // (bypassing TensorVal::new), so round the written value through f32
        // here when the target is an F32 tensor. F64/Int/Trit pass through.
        let rval = match dtype {
            // #834: fp8 elements round through f32 on write exactly like F32
            // elements do (`DType::rounds_as_f32`).
            DType::F32 | DType::Fp8E4M3 | DType::Fp8E5M2 => match rval {
                Value::Float(x, _) => Value::Float(quantize_f32(x), FW::F64),
                Value::Tensor(mut t) => { quantize_f32_arr(&mut t.data); Value::Tensor(t) }
                other => other,
            },
            // #571: the same rule one width down. A narrow integer element is
            // as wide as its type says, so what lands in the buffer is the
            // wrapped value — the JIT stores into a 4-byte `i32` lane and
            // truncates, and an unwrapped store here would read back a number
            // that does not fit the element type.
            // `I64` is left alone rather than wrapped-as-identity: at 64 bits
            // the wrap changes no integer, and routing through `x as i64` would
            // truncate a float written into a width-less integer tensor, which
            // is a different (pre-existing) question than this one.
            DType::Int(w) if w != IW::I64 => match rval {
                Value::Int(n, _) => Value::Int(w.wrap(n), w),
                Value::Tensor(mut t) => {
                    t.data.mapv_inplace(|x| w.wrap(x as i64) as f64);
                    Value::Tensor(t)
                }
                other => other,
            },
            _ => rval,
        };
        let selections = Self::resolve_index_values(arr.shape(), idx_vals, &span)?;

        let mut target_shape = Vec::new();
        for sel in selections.iter() {
            if !sel.reduce_rank {
                target_shape.push(sel.indices.len());
            }
        }

        match rval {
            Value::Tensor(rhs_arr) => {
                if target_shape != rhs_arr.shape() {
                    return Err(RuntimeError::at(format!(
                        "slice assignment shape mismatch: {:?} ← {:?}",
                        target_shape, rhs_arr.shape()
                    ), span));
                }

                let mut target_coords = vec![0; target_shape.len()];
                let target_len: usize = target_shape.iter().product();
                for _ in 0..target_len {
                    let mut src_coords = Vec::with_capacity(selections.len());
                    let mut target_axis = 0;
                    for sel in selections.iter() {
                        if sel.reduce_rank {
                            src_coords.push(sel.indices[0]);
                        } else {
                            src_coords.push(sel.indices[target_coords[target_axis]]);
                            target_axis += 1;
                        }
                    }

                    arr[IxDyn(&src_coords)] = rhs_arr[IxDyn(&target_coords)];

                    for axis in (0..target_shape.len()).rev() {
                        target_coords[axis] += 1;
                        if target_coords[axis] < target_shape[axis] {
                            break;
                        }
                        target_coords[axis] = 0;
                    }
                }
            }
            Value::Int(n, _) => {
                assign_scalar_to_selection(arr, &selections, n as f64);
            }
            Value::Float(val, _) => {
                assign_scalar_to_selection(arr, &selections, val);
            }
            Value::Bool(b) => {
                assign_scalar_to_selection(arr, &selections, if b { 1.0 } else { 0.0 });
            }
            other => return Err(RuntimeError::at(format!(
                "cannot assign value of type {} to tensor slice",
                other.type_name()
            ), span)),
        }
        Ok(())
    }

    /// The scalar/elementwise `BinOp` pair a compound-assignment operator
    /// dispatches to — the same choice the scalar-ident arms (`x += v`, just
    /// below) and `trace_reassign` make: elementwise (`Dot*`) when either
    /// side is a tensor, scalar otherwise. Shared here so #693's indexed
    /// compound-assign path picks exactly the `apply_binop` the scalar path
    /// would.
    fn compound_binops(op: &AssignOp) -> (BinOp, BinOp) {
        match op {
            AssignOp::PlusEq  => (BinOp::Add, BinOp::DotAdd),
            AssignOp::MinusEq => (BinOp::Sub, BinOp::DotSub),
            AssignOp::StarEq  => (BinOp::Mul, BinOp::DotMul),
            AssignOp::SlashEq => (BinOp::Div, BinOp::DotDiv),
            other => unreachable!("compound_binops called with non-arithmetic {:?}", other),
        }
    }

    /// Compound-assign (`+=`/`-=`/`*=`/`/=`) into a single tensor element at
    /// already-evaluated indices (#693). Before this, no match arm at all
    /// handled a compound op against an indexed target: `t[i] += v` fell to
    /// the catch-all no-op — `--check` accepted it, nothing stored, nothing
    /// errored, and `t[i]` read back its old value (0 where 5 was expected).
    ///
    /// Mirrors the plain indexed store's in-place path (#689/#692): no clone
    /// of `arr`, mutate through `&mut`. The current element is read out
    /// dtype-aware exactly as the ordinary index *read* expression does (an
    /// `Int` tensor yields a width-tagged `Value::Int`, `Bool` a
    /// `Value::Bool`, else a width-tagged `Value::Float` — see the
    /// `PostfixOp::Index` read arm), combined via `apply_binop` — the same
    /// function the scalar `x += v` arm calls — and the result is stored
    /// back through `assign_to_tensor_resolved`, which already applies the
    /// same f32-quantize / narrow-int-wrap a plain `t[i] = v` store does.
    /// That store-time quantization is why this arm's f32 answer matches
    /// the plain indexed store and the JIT (`0.1f32 += 0.2f32` here reads
    /// back `0.30000001192092896`) rather than the SCALAR `x += v` arm,
    /// which skips it and reads back `0.30000000149011613` — a pre-existing
    /// interp/jit divergence in the scalar path, not something this arm
    /// reproduces (filed as #713; not fixed here).
    ///
    /// Errors — never silently no-ops — when the index does not name a
    /// single scalar element. Compound assignment to a tensor *slice*
    /// (`t[.., 0] += v`) is a different, broader feature this fix does not
    /// add; refusing it loudly beats guessing at elementwise-broadcast
    /// semantics nobody has specified yet.
    fn compound_assign_tensor_elem(
        arr: &mut ArrayD<f64>,
        dtype: DType,
        scalar_op: BinOp,
        dot_op: BinOp,
        idx_vals: &[IndexVal],
        rval: Value,
        span: Span,
    ) -> EvalResult<()> {
        let selections = Self::resolve_index_values(arr.shape(), idx_vals, &span)?;
        if !selections.iter().all(|s| s.reduce_rank) {
            return Err(RuntimeError::at(
                "compound assignment to a tensor slice is not supported — index every \
                 axis so the target is a single element (`t[i] += v`, not a range or \
                 partial index)".to_string(),
                span,
            ));
        }
        let coords: Vec<usize> = selections.iter().map(|s| s.indices[0]).collect();
        let x = arr[IxDyn(&coords)];
        let cur = match dtype {
            // #571: width-tagged, matching the plain read arm.
            DType::Int(w) => Value::Int(w.wrap(x as i64), w),
            DType::Bool => Value::Bool(x != 0.0),
            _ => Value::Float(x, if dtype.rounds_as_f32() { FW::F32 } else { FW::F64 }),
        };
        let op = if matches!(rval, Value::Tensor(_)) { dot_op } else { scalar_op };
        let new_val = apply_binop(op, &cur, &rval)?;
        Self::assign_to_tensor_resolved(arr, dtype, idx_vals, new_val, span)
    }

    pub fn new() -> Self {
        // #674 slice 3d: the entered-budget register and its stamp are
        // thread-local (see `CUR_BUDGET`'s doc), and this interpreter's
        // budget table starts at entry 0 alone — so the register must name
        // entry 0 too. `BudgetGuard` restores on every exit edge, so a
        // clean run always leaves it at 0 anyway; this is the belt for a
        // thread that a previous run left mid-`enter` (a panic, a
        // `std::process::exit` from a nested harness) rather than a
        // load-bearing reset.
        CUR_BUDGET.with(|c| c.set(0));
        CUR_BUDGET_TAG.with(|c| *c.borrow_mut() = None);
        Self {
            scopes: vec![HashMap::new()],
            shape_param_scopes: vec![std::collections::HashSet::new()],
            stream_axes: vec![HashMap::new()],
            kv_capacities: HashMap::new(),
            model_shape_params: HashMap::new(),
            fns: HashMap::new(),
            models: HashMap::new(),
            model_methods: HashMap::new(),
            model_fields: HashMap::new(),
            enums: HashMap::new(),
            type_aliases: HashMap::new(),
            mut_globals: Vec::new(),
            cur_module: 0,
            rng_state: 0x9E3779B97F4A7C15,  // SplitMix64 golden-ratio seed
            start_time: std::time::Instant::now(),
            argv: Vec::new(),
            ports: crate::ports::PortRegistry::new(),
            profile: None,
            interp_modules: HashMap::new(),
            public_items: std::collections::HashSet::new(),
            host_features: detect_host_features(),
            extern_fns: std::collections::HashSet::new(),
            pending_writebacks: Vec::new(),
            parked_receiver: None,
            call_depth: Rc::new(Cell::new(0)),
            move_hint: None,
            float_lit_hint: None,
            arena_limits: crate::arena::ArenaLimits::default(),
            vault_used: 0,
            budgets: vec![BudgetEntry::root(None)],
            budget_free: Vec::new(),
            lit_charge_suppressed: false,
            ctor_forge_bytes: 0,
            comptime_fuel: None,
            named_fn_ret_stack: Vec::new(),
        }
    }

    /// #505: evaluate a `@comptime` body at compile time (`SPEC.md §7.8`).
    ///
    /// This is the whole of "the reference interpreter is the comptime
    /// evaluator": the arithmetic a folded block sees is the arithmetic every
    /// other demoniC expression sees, so a fold cannot disagree with a run on
    /// truncating division, `%` sign, or per-width integer wrapping (#540).
    /// A hand-written const-folder beside the reference implementation is
    /// exactly the drift the single-evaluator design avoids.
    ///
    /// What keeps compile-time evaluation out of runtime state is not this
    /// entry point but `comptime.rs`'s static gate, which admits no call of
    /// any kind — so a legal body cannot name the port registry, the RNG,
    /// `argv`, the clock, the arenas, or the filesystem, and the caller can
    /// throw the whole `Interpreter` away after the block.
    ///
    /// `budget` bounds termination: the body may loop, so a non-terminating
    /// one must fail the compile rather than hang it.
    pub fn eval_comptime(&mut self, block: &Block, budget: u64)
        -> Result<Value, ComptimeStop>
    {
        self.comptime_fuel = Some(budget);
        self.push_scope();
        let r = self.eval_block(block);
        self.pop_scope();
        let exhausted = self.comptime_fuel == Some(0);
        self.comptime_fuel = None;
        match r {
            Ok(Flow::Normal(v)) | Ok(Flow::Return(v, _)) => Ok(v),
            Ok(_) => Ok(Value::Nil),
            Err(e) if exhausted => { let _ = e; Err(ComptimeStop::Budget) }
            Err(e) => Err(ComptimeStop::Failed(e)),
        }
    }

    /// #400: install the `--vault=` / `--forge=` byte budgets (`MEMORY.md §1.1`).
    /// Absent flags leave the corresponding arena unbounded, which is what every
    /// program got before the flags were accepted.
    pub fn set_arena_limits(&mut self, limits: crate::arena::ArenaLimits) {
        self.arena_limits = limits;
        // #674 slice 3b: entry 0's extent/free mirror `--forge` — called
        // before any allocation, so `used` is still 0 and `free` starts at
        // the whole limit (or stays `None`, unbounded, absent the flag).
        self.budgets[0].extent = limits.forge;
        self.budgets[0].free = limits.forge;
    }

    // ── #674 slice 3b test-only harness ─────────────────────────────────────
    //
    // `forge.enter`/`in b` do not lower until slice 3d (they stay refused,
    // located, in this slice), so no DEMONIC PROGRAM can stamp a tensor to a
    // non-root budget yet. The retrieval sites that must check for a stale
    // budget still need covering NOW, at the mechanism level, which is what
    // these give
    // `interp_tests.rs`: the exact push/pop discipline `enter` will use in
    // 3d, exposed only under `#[cfg(test)]` and not reachable from any
    // surface syntax. `pub(crate)` (not `pub`) because `interp_tests` is a
    // sibling module of `interp`, not a descendant — plain field/method
    // privacy would not reach it.

    /// TEST-ONLY: evaluate `e` directly — `eval_expr`'s wrapper (`check_live`
    /// included), no source text, no scope setup beyond what the caller has
    /// already bound.
    #[cfg(test)]
    pub(crate) fn eval_expr_for_test(&mut self, e: &Expr) -> EvalResult<Value> {
        self.eval_expr(e)
    }

    /// TEST-ONLY: bind `name` in the current (outermost) scope, so a
    /// hand-built expression string can refer to a `Value` the test
    /// constructed some other way (typically under `with_budget_for_test`).
    #[cfg(test)]
    pub(crate) fn bind_for_test(&mut self, name: &str, v: Value) {
        self.bind(name, v);
    }

    /// TEST-ONLY: park `v` as the already-evaluated receiver for the
    /// expression node `recv_expr`, exactly as `call_bracketed_method`'s
    /// `NotAMethod` arm does. It is the only way to pin §3.2 hole-table
    /// row 3's *take* path on its own: reaching it through surface syntax
    /// means going through `call_bracketed_method`, whose own receiver read
    /// is expression position and refuses an already-stale tensor first
    /// (row 1). What this reaches instead is the window row 3 actually owns
    /// — a receiver parked live and taken stale.
    #[cfg(test)]
    pub(crate) fn park_receiver_for_test(&mut self, recv_expr: &Expr, v: Value) {
        self.park_receiver(recv_expr, v);
    }

    /// #674 slice 3d: the stamp every tensor constructed under budget
    /// `index` inherits (`TensorVal::new`, §3.2 M1). `None` for entry 0 —
    /// the root can never be reaped, so a tensor tagged from it can never go
    /// stale (§4.1) — and the entry's OWN generation cell otherwise, shared
    /// so one `reap` reaches every tensor ever stamped from it.
    fn budget_stamp(&self, index: usize) -> OriginTag {
        if index == 0 {
            None
        } else {
            let e = &self.budgets[index];
            Some(Rc::new(BudgetTag { generation: e.live.get(), live: Rc::clone(&e.live) }))
        }
    }

    /// #674 slice 3d: resolve a `Budget` handle and open the register on it.
    /// The one entry point for both `forge.enter(b) { … }`
    /// (`Expr::BudgetBlock`) and `<ctor> in b` (`Expr::AllocIn`), so the two
    /// cannot drift on either refusal: a value that is not a `Budget` is a
    /// well-formedness error naming `what` (the shape `forge.carve`/
    /// `forge.reap` already use for their own arguments — no new tag, so
    /// B7's "exactly three refusal classes" count is untouched), and a
    /// handle whose generation no longer matches its entry is
    /// **`budget-stale`**, the runtime error §2.3's table lists at
    /// "carve / reap / enter / `in b`".
    fn open_budget(&self, v: &Value, span: &Span, what: &str) -> EvalResult<BudgetGuard> {
        let Value::Budget(d) = v else {
            return Err(RuntimeError::at(format!(
                "{what}: the budget operand must be a `Budget`, got {}",
                v.type_name()), span.clone()));
        };
        let index = budget_index(*d);
        let generation = budget_generation(*d);
        if index >= self.budgets.len() || generation != self.budgets[index].live.get() {
            return Err(RuntimeError::at(
                "budget-stale: this budget handle names a budget that was \
                 already reaped".to_string(), span.clone()));
        }
        Ok(BudgetGuard::enter(index, self.budget_stamp(index)))
    }

    /// TEST-ONLY: push the entered-budget register and its stamp to `index`,
    /// run `f`, then restore both — the same `BudgetGuard` `Expr::
    /// BudgetBlock` and `Expr::AllocIn` open, reached without surface
    /// syntax for the slice-3b hole-table cases that predate `enter`.
    #[cfg(test)]
    pub(crate) fn with_budget_for_test<R>(&mut self, index: usize, f: impl FnOnce(&mut Self) -> R) -> R {
        let guard = BudgetGuard::enter(index, self.budget_stamp(index));
        let r = f(self);
        drop(guard);
        r
    }

    /// TEST-ONLY: the entered-budget register, for the slice-3d exit-edge
    /// cases — one per edge, each asserting the register came back to the
    /// OUTER budget after the block (the required coverage for each exit
    /// edge).
    #[cfg(test)]
    pub(crate) fn cur_budget_for_test() -> usize { cur_budget() }

    /// TEST-ONLY: decode a `Value::Budget`'s table index, for a test that
    /// carved a budget and needs the raw index to pass to
    /// `with_budget_for_test`.
    #[cfg(test)]
    pub(crate) fn budget_index_for_test(v: &Value) -> usize {
        match v {
            Value::Budget(d) => budget_index(*d),
            other => panic!("expected Value::Budget, got {:?}", other),
        }
    }

    /// Enter a user-function call frame: bump the recursion counter and hand back a
    /// guard that decrements it on drop. Errors (catchably) if `MAX_CALL_DEPTH` would
    /// be exceeded, converting unbounded recursion from a native SIGABRT into a
    /// reportable runtime error. Call once at the top of each user-fn / lambda entry.
    fn enter_call(&self, sp: &Span) -> EvalResult<CallDepthGuard> {
        let depth = self.call_depth.get();
        if depth >= MAX_CALL_DEPTH {
            return Err(RuntimeError::at(
                format!("call stack depth exceeded {MAX_CALL_DEPTH} (unbounded recursion?)"),
                sp.clone(),
            ));
        }
        self.call_depth.set(depth + 1);
        Ok(CallDepthGuard(Rc::clone(&self.call_depth)))
    }

    /// Enable op profiling. After `run()`, inspect `self.profile` for counters.
    pub fn enable_profile(&mut self) {
        self.profile = Some(OpProfile::default());
    }

    /// Increment a profile counter if profiling is active.
    #[inline]
    fn prof_tensor_op(&mut self, elements: u64) {
        if let Some(p) = self.profile.as_mut() {
            p.tensor_ops += 1;
            p.tensor_elements += elements;
        }
    }

    #[inline]
    fn prof_scalar_op(&mut self) {
        if let Some(p) = self.profile.as_mut() {
            p.scalar_ops += 1;
        }
    }

    #[inline]
    fn prof_fn_call(&mut self) {
        if let Some(p) = self.profile.as_mut() {
            p.fn_calls += 1;
        }
    }

    #[inline]
    fn prof_alloc(&mut self) {
        if let Some(p) = self.profile.as_mut() {
            p.allocs += 1;
        }
    }

    /// #400: charge `nbytes` against a sized arena, or report exhaustion.
    ///
    /// `arena` is `None` for allocations that land somewhere the sizing flags
    /// do not reach — only the Stream arena (`forge.kv` / `stream.kv`, which
    /// carries its own `capacity`) is exempt (#781: every OTHER materialized
    /// tensor, explicit constructor or implicit intermediate, is charged —
    /// see `charge_intermediate`).
    fn charge_arena(
        &mut self,
        arena: Option<crate::arena::Arena>,
        nbytes: u64,
        span: Option<&Span>,
    ) -> EvalResult<()> {
        use crate::arena::Arena;
        let Some(arena) = arena else { return Ok(()) };
        match arena {
            // Vault has no `carve` (§3.5) — a flat per-process meter,
            // unchanged by slice 3b.
            Arena::Vault => self.charge_vault(nbytes, span),
            // #674 slice 3b: PRE-A now reads the budget table's `free`
            // (`charge_budget`), never a bare `forge_used` counter — see
            // that fn's doc.
            Arena::Forge => self.charge_budget(cur_budget(), nbytes, span),
        }
    }

    /// The Vault half of `charge_arena`, unchanged by slice 3b: a flat
    /// per-process meter, `vault_used`, never rewound and never part of the
    /// budget table.
    fn charge_vault(&mut self, nbytes: u64, span: Option<&Span>) -> EvalResult<()> {
        use crate::arena::{fmt_bytes, Arena};
        let Some(limit) = self.arena_limits.vault else {
            // Unbounded: still track the watermark so a later `--vault` run
            // reports the same numbers, but never fail.
            self.vault_used = self.vault_used.saturating_add(nbytes);
            return Ok(());
        };
        let after = self.vault_used.saturating_add(nbytes);
        if after > limit {
            let msg = format!(
                "{} arena exhausted: this allocation needs {}, but only {} of the \
                 {} {} budget is free",
                Arena::Vault.name(),
                fmt_bytes(nbytes),
                fmt_bytes(limit.saturating_sub(self.vault_used)),
                fmt_bytes(limit),
                Arena::Vault.flag(),
            );
            return Err(match span {
                Some(sp) => RuntimeError::at(msg, sp.clone()),
                None => RuntimeError::msg(msg),
            });
        }
        self.vault_used = after;
        Ok(())
    }

    /// #674: PRE-A — charge `nbytes`
    /// against budget table entry `index`'s `free`, never `ForgeArena::
    /// exceeds_budget`'s shape and never a bare counter. Entry 0's
    /// exhaustion sentence is byte-identical to what `charge_arena`'s Forge
    /// arm produced before this slice (the compatibility gate slices 3b/3c
    /// hold to, §3.1); any other entry names "the entered budget" and its
    /// own extent instead of `--forge`, the §1 correction's PRE-A sentence —
    /// unreachable from any demoniC program in 3b, since nothing yet pushes
    /// `cur_budget` off 0, but wired now so 3d's `enter` needs no second
    /// charging path.
    fn charge_budget(&mut self, index: usize, nbytes: u64, span: Option<&Span>) -> EvalResult<()> {
        use crate::arena::{fmt_bytes, Arena};
        let (free, used, extent) = {
            let e = &self.budgets[index];
            (e.free, e.used, e.extent)
        };
        let Some(free) = free else {
            // Unbounded (only entry 0 under no `--forge`, §4.1): still
            // track the watermark, never fail.
            self.budgets[index].used = used.saturating_add(nbytes);
            return Ok(());
        };
        if nbytes > free {
            let msg = if index == 0 {
                format!(
                    "{} arena exhausted: this allocation needs {}, but only {} of the \
                     {} {} budget is free",
                    Arena::Forge.name(),
                    fmt_bytes(nbytes),
                    fmt_bytes(free),
                    fmt_bytes(extent.unwrap_or(0)),
                    Arena::Forge.flag(),
                )
            } else {
                format!(
                    "forge arena exhausted: this allocation needs {}, but only {} of the \
                     entered budget's {} extent is free",
                    fmt_bytes(nbytes),
                    fmt_bytes(free),
                    fmt_bytes(extent.unwrap_or(0)),
                )
            };
            return Err(match span {
                Some(sp) => RuntimeError::at(msg, sp.clone()),
                None => RuntimeError::msg(msg),
            });
        }
        let e = &mut self.budgets[index];
        e.free = Some(free - nbytes);
        e.used = used.saturating_add(nbytes);
        Ok(())
    }

    /// #674: `forge.carve(parent, bytes)`.
    /// **(PRE)** `bytes ≤ free(parent)` — vacuous when `parent` is unbounded
    /// (§4.1: `forge.root` under no `--forge`). Reserves METER, not bytes:
    /// `free(parent)` falls by `n` and nothing is committed (§3.1's "a
    /// region carved from the parent's storage is true of the BUDGET, not
    /// the bytes" — the interpreter has no chunks to commit regardless).
    fn budget_carve(&mut self, parent: &Value, bytes: &Value, span: &Span) -> EvalResult<Value> {
        use crate::arena::fmt_bytes;
        let Value::Budget(pd) = parent else {
            return Err(RuntimeError::at(format!(
                "forge.carve: the first argument must be a `Budget`, got {}",
                parent.type_name()), span.clone()));
        };
        let p_index = budget_index(*pd);
        let p_generation = budget_generation(*pd);
        if p_index >= self.budgets.len() || p_generation != self.budgets[p_index].live.get() {
            return Err(RuntimeError::at(
                "budget-stale: this budget handle names a budget that was \
                 already reaped".to_string(), span.clone()));
        }
        let Some(n) = bytes.as_int() else {
            return Err(RuntimeError::at(format!(
                "forge.carve: the second argument must be an integer byte \
                 count, got {}", bytes.type_name()), span.clone()));
        };
        if n < 0 {
            return Err(RuntimeError::at(
                "forge.carve: the byte count must not be negative".to_string(),
                span.clone()));
        }
        let n = n as u64;
        let parent_extent = self.budgets[p_index].extent;
        if let Some(free) = self.budgets[p_index].free {
            if n > free {
                return Err(RuntimeError::at(format!(
                    "budget-carve: this carve needs {}, but only {} of the \
                     parent budget's {} extent is free",
                    fmt_bytes(n), fmt_bytes(free), fmt_bytes(parent_extent.unwrap_or(0))),
                    span.clone()));
            }
        }
        // (PRE) held (or was vacuous) — apply the transition.
        if let Some(f) = self.budgets[p_index].free.as_mut() { *f -= n; }
        self.budgets[p_index].children += 1;
        let new_index = if let Some(idx) = self.budget_free.pop() {
            let live = Rc::clone(&self.budgets[idx].live);
            self.budgets[idx] = BudgetEntry {
                parent: p_index as u32, extent: Some(n), used: 0, free: Some(n),
                children: 0, live,
            };
            idx
        } else {
            let idx = self.budgets.len();
            if idx > BUDGET_INDEX_MASK as usize {
                return Err(RuntimeError::at(
                    "forge.carve: the interpreter's budget table is full \
                     (1 048 575 live budgets)".to_string(), span.clone()));
            }
            self.budgets.push(BudgetEntry {
                parent: p_index as u32, extent: Some(n), used: 0, free: Some(n),
                children: 0, live: Rc::new(Cell::new(0)),
            });
            idx
        };
        let generation = self.budgets[new_index].live.get();
        Ok(Value::Budget(budget_descriptor(generation, new_index as u32)))
    }

    /// #674: `forge.reap(b) -> nil`.
    /// **(LOC)** refused while `b` has a live child (R2, `budget-reap`); the
    /// root is a fixed point `reap` refuses outright (`parent(0) = 0` would
    /// otherwise credit the root's `free` to itself and underflow its own
    /// `children`). Credits `b`'s WHOLE extent — spent and unspent alike —
    /// to `parent(b)`, because releasing it is what a credit means; bumps
    /// `b`'s generation, which is the entirety of B6's invalidation (every
    /// `TensorVal` stamped from `b`, live or already cloned, shares this
    /// cell via `BudgetTag::live`).
    fn budget_reap(&mut self, b: &Value, span: &Span) -> EvalResult<Value> {
        let Value::Budget(bd) = b else {
            return Err(RuntimeError::at(format!(
                "forge.reap: the argument must be a `Budget`, got {}",
                b.type_name()), span.clone()));
        };
        let index = budget_index(*bd);
        let generation = budget_generation(*bd);
        if index >= self.budgets.len() || generation != self.budgets[index].live.get() {
            return Err(RuntimeError::at(
                "budget-stale: this budget handle names a budget that was \
                 already reaped".to_string(), span.clone()));
        }
        if index == 0 {
            return Err(RuntimeError::at(
                "budget-reap: `forge.root` can never be reaped".to_string(),
                span.clone()));
        }
        let children = self.budgets[index].children;
        if children != 0 {
            return Err(RuntimeError::at(format!(
                "budget-reap: {}",
                crate::arena::budget_live_child_clause(children as u64)),
                span.clone()));
        }
        let parent = self.budgets[index].parent as usize;
        let extent = self.budgets[index].extent.unwrap_or(0);
        if let Some(f) = self.budgets[parent].free.as_mut() { *f = f.saturating_add(extent); }
        self.budgets[parent].children = self.budgets[parent].children.saturating_sub(1);
        let live = &self.budgets[index].live;
        live.set(live.get().wrapping_add(1));
        self.budget_free.push(index);
        Ok(Value::Nil)
    }

    /// #674 slice 3b: move budget table entry `index`'s `used` DIRECTLY to
    /// `new_used`, crediting (or charging) the difference to `free` too, so
    /// (LOC) stays exact. The only two callers are the AD forward-only and
    /// `fwd_bwd` rewinds (`call_fn_in_module`, `call_grad_in_module`), which
    /// move what was `forge_used` to a point in the past rather than route a
    /// fresh allocation through `charge_arena` — see `ctor_forge_bytes`'s
    /// doc for why the landing point is `mark + kept_ctor_bytes`, not `mark`.
    fn set_budget_used(&mut self, index: usize, new_used: u64) {
        let e = &mut self.budgets[index];
        if new_used <= e.used {
            let credit = e.used - new_used;
            if let Some(f) = e.free.as_mut() { *f = f.saturating_add(credit); }
        } else {
            let charge = new_used - e.used;
            if let Some(f) = e.free.as_mut() { *f = f.saturating_sub(charge); }
        }
        e.used = new_used;
    }

    /// #781: charge an EXPLICIT constructor (`vault.zeros[…]`,
    /// `forge.uninit[…]`, an `rng.*` draw) against its arena.
    ///
    /// Identical to `charge_arena` except that a successful Forge charge is
    /// also added to the `ctor_forge_bytes` running total, so an AD rewind
    /// can keep these bytes rather than un-charging a tensor the program
    /// named and may still hold — see that field's doc.
    fn charge_ctor(
        &mut self,
        arena: Option<crate::arena::Arena>,
        nbytes: u64,
        span: Option<&Span>,
    ) -> EvalResult<()> {
        self.charge_arena(arena, nbytes, span)?;
        if matches!(arena, Some(crate::arena::Arena::Forge)) {
            self.ctor_forge_bytes = self.ctor_forge_bytes.saturating_add(nbytes);
        }
        Ok(())
    }

    /// #781: charge a tensor materialized OUTSIDE an explicit `vault.*`/
    /// `forge.*` constructor — a computed intermediate (`let C = A @ B`, a
    /// dotted elementwise op, a tensor literal, a transpose, a slice that
    /// copies, a reshape) — against the Forge arena, at its DECLARED element
    /// width (`DType::byte_width`), not the interpreter's f64 storage width.
    ///
    /// #804: also charges a STRING materialized the same way. Two forms
    /// reach this arm: `apply_binop`'s `+` (`s + "x"`, `"lit" + to_str(f)`)
    /// and `apply_cast`'s `as str` (`x as str`) — both string-producing
    /// forms that live outside `call_builtin` and so miss #802's
    /// `charge_builtin_result` (whose own `Value::Str` arm charges directly,
    /// not through here — see that fn's doc for why the two must NOT share
    /// this suppressible path). Charged at the fresh string's UTF-8 byte
    /// length, the same number `charge_builtin_result` charges a builtin's
    /// fresh `Value::Str` result at, so a `Value::Str` is billed identically
    /// wherever it is produced. `call_str_method` (`.trim`/`.upper`/
    /// `.lower`/`.replace`/`.split`/`.lines`, …) is a THIRD string-producing
    /// form outside `call_builtin`; it charges its result through
    /// `charge_builtin_result` directly rather than through this arm — see
    /// its call site.
    ///
    /// Every such value charges Forge unconditionally, never Vault: an
    /// implicit intermediate has no `vault.`/`forge.` spelling to derive an
    /// arena from, and `MEMORY.md §1`'s Forge row — "activations, gradients,
    /// scratch" — is exactly what these are; only an explicit `vault.*`
    /// constructor can put bytes in Vault (`charge_arena`'s other caller).
    /// A no-op for any other value.
    fn charge_intermediate(&mut self, v: &Value, span: Option<&Span>) -> EvalResult<()> {
        if self.lit_charge_suppressed {
            return Ok(());
        }
        match v {
            Value::Tensor(t) => {
                let nbytes = (t.len() as u64).saturating_mul(t.dtype.byte_width());
                self.charge_arena(Some(crate::arena::Arena::Forge), nbytes, span)?;
            }
            Value::Str(s) => {
                let nbytes = s.len() as u64;
                self.charge_arena(Some(crate::arena::Arena::Forge), nbytes, span)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// #781, #650, SPEC §3.4: charge the PRIVATE COPY a `!` binding
    /// (a `let !y = x` from a bare identifier, a top-level `let !alpha = x`,
    /// or a `!t` parameter) makes of its bare-identifier source — parity
    /// with what the JIT actually copies at that binding, NOT with what the
    /// interpreter itself allocates to give `y` independent identity.
    ///
    /// For a `Value::Tensor` those agree: the JIT copies the bytes, so this
    /// is `charge_intermediate` unchanged. For a `Value::Str` they diverge —
    /// on the JIT a `str` is a pointer (SPEC §3.1), so `let !t = s` binds a
    /// second pointer at the private binding and copies nothing; the JIT
    /// requests 0 bytes for it. The interpreter's `Value::Str` is an owned
    /// `String`, so cloning it for `t` IS a real interpreter allocation —
    /// but the rule this charge exists to enforce is parity with the JIT's
    /// requested bytes, not "charge every interpreter allocation," and the
    /// JIT requests none here. (PR #815 review: a `read_file` result copied
    /// by 100 `let !t = c` bindings ran at 21 513 B against the JIT's
    /// 384 B — 100 phantom copies the JIT never makes, because a comment at
    /// each of these three call sites asserted "the JIT charges the copy"
    /// for every mutating binding without carving out the one value kind
    /// where that is false.) So `Value::Str` is excluded here and left
    /// uncharged; every other value kind charges exactly as
    /// `charge_intermediate` would.
    fn charge_private_copy(&mut self, v: &Value, span: Option<&Span>) -> EvalResult<()> {
        if matches!(v, Value::Str(_)) {
            return Ok(());
        }
        self.charge_intermediate(v, span)
    }

    /// Evaluate `expr` with `lit_charge_suppressed` held true for the
    /// duration — see the field doc. Restores the PREVIOUS value (not
    /// unconditionally `false`) on every path, `Err` included, so a
    /// suppressed evaluation nested inside another suppressed evaluation
    /// (a literal-of-literals row, or a reshape whose shape argument is
    /// itself a nested literal) leaves the outer suppression intact
    /// instead of clearing it early.
    fn eval_expr_lit_charge_suppressed(&mut self, expr: &Expr) -> EvalResult<Value> {
        let saved = self.lit_charge_suppressed;
        self.lit_charge_suppressed = true;
        let result = self.eval_expr(expr);
        self.lit_charge_suppressed = saved;
        result
    }

    /// Set command-line arguments for `argv()` builtin.
    pub fn set_argv(&mut self, args: Vec<String>) {
        self.argv = args;
    }

    /// Advance the xorshift64 state and return the next u64.
    fn rand_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        // Ensure non-zero (xorshift dies at zero).
        if x == 0 { x = 0xDEADBEEFCAFEBABE; }
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    /// Uniform [0, 1) double from the next 53 bits of state.
    fn rand_uniform(&mut self) -> f64 {
        (self.rand_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box-Muller from two uniforms.
    fn rand_normal(&mut self) -> f64 {
        let u1 = self.rand_uniform().max(f64::MIN_POSITIVE);
        let u2 = self.rand_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn run(&mut self, program: &Program, path: Option<&std::path::Path>) -> EvalResult<Value> {
        self.load_program(program, path)?;
        // Pass 3: if main() exists in the file just loaded, call it; otherwise
        // return Nil. A `fn main` in a dependency is that file's own.
        if let Some(key) = self.resolve_fn("main") {
            let main = self.fns[&key].clone();
            self.call_fn(&main, Vec::new(), main.span.clone())
        } else {
            Ok(Value::Nil)
        }
    }

    pub fn load_program(&mut self, program: &Program, path: Option<&std::path::Path>) -> EvalResult<()> {
        // One epoch per file (#398). Everything this file can name — its own
        // items and the `pub` items it imports — is registered under this
        // number, so two files' name tables never overlap; `cur_module` stays
        // on it while the file's top-level `let`s run and until the next load.
        self.cur_module += 1;
        // Files are loaded one after another and every call restores
        // `cur_module` on its way out, so at the top of a load it still holds
        // the previously loaded file's number and the bump always mints a
        // fresh one. A call that leaked its save would hand this file a number
        // another file already owns and silently merge their name tables; this
        // is the cheap proof that none does.
        debug_assert!(
            self.fns.values().all(|f| f.module < self.cur_module),
            "module numbering went backwards — a call did not restore `cur_module`",
        );
        self.public_items = crate::ast::collect_public_items(program);
        self.process_imports(program, path)?;
        // Pass 1: collect fn and model decls (forward references)
        for item in &program.items {
            self.collect_item(item);
        }
        // Pass 2: evaluate any top-level let stmts (initializing globals)
        for item in &program.items {
            self.eval_top_level_item(item)?;
        }
        Ok(())
    }

    pub fn get_module_env(&self) -> InterpModuleEnv {
        InterpModuleEnv {
            scopes: self.scopes.clone(),
            fns: self.fns.clone(),
            models: self.models.clone(),
            enums: self.enums.clone(),
            type_aliases: self.type_aliases.clone(),
            public_items: self.public_items.clone(),
            epoch: self.cur_module,
        }
    }

    fn process_imports(&mut self, program: &Program, path: Option<&std::path::Path>) -> EvalResult<()> {
        for item in &program.items {
            if let Item::Use(us) = item {
                if let Some(current_path) = path {
                    let parent_dir = current_path.parent().unwrap_or_else(|| std::path::Path::new(""));
                    let import_path = parent_dir.join(&us.path);
                    let canonical_path = match import_path.canonicalize() {
                        Ok(p) => p,
                        Err(e) => {
                            return Err(RuntimeError::at(
                                format!("error resolving import {:?} in file {:?}: {}", us.path, current_path, e),
                                us.span.clone(),
                            ));
                        }
                    };
                    if let Some(imported_env) = self.interp_modules.get(&canonical_path).cloned() {
                        // A `pub` item lives in the imported file's snapshot
                        // under that file's number; it enters THIS file's
                        // table under this file's number, bare or under the
                        // alias. One rule for all four kinds of top-level name
                        // (SPEC §9): a fn, a model, an enum, a `type` alias.
                        // Model entries point at the declaring file's key, so
                        // an instance built here dispatches to the methods
                        // registered there; methods, fields and shape params
                        // are looked up through that key only, so they are not
                        // copied. A fn's own home module travels on its
                        // `FnDecl` (`f.module`), so both spellings of an
                        // imported fn still run in the file that wrote it.
                        let spell = |name: &str| match &us.alias {
                            Some(alias) => format!("{}.{}", alias, name),
                            None => name.to_string(),
                        };
                        for name in &imported_env.public_items {
                            let src = module_key(imported_env.epoch, name);
                            let dst = module_key(self.cur_module, &spell(name));
                            if let Some(f) = imported_env.fns.get(&src) {
                                self.fns.insert(dst, f.clone());
                            } else if let Some(canonical) = imported_env.models.get(&src) {
                                self.models.insert(dst, canonical.clone());
                            } else if let Some(variants) = imported_env.enums.get(&src) {
                                // The checker does not yet carry a `pub enum`
                                // across a `use` — it refuses the importer's
                                // `Color.Red` as an undefined identifier — so
                                // no program reaches this arm today. It is
                                // here because the loop states one rule for
                                // all four kinds, and dropping the enum out of
                                // it would leave the interpreter with a
                                // different rule than the one it enforces
                                // everywhere else.
                                self.enums.insert(dst, variants.clone());
                            } else if let Some(alias) = imported_env.type_aliases.get(&src) {
                                self.type_aliases.insert(dst, alias.clone());
                            }
                        }
                        if let Some(alias) = &us.alias {
                            self.bind(alias.clone(), Value::Module { alias: alias.clone(), path: canonical_path });
                            // #623's `fn_module` carry-over onto the `alias.name`
                            // spelling is unnecessary here: the home module now
                            // travels on the `FnDecl` itself (`f.module`), which
                            // the loop above clones under both spellings.
                            if let Some(imported_scope) = imported_env.scopes.get(0) {
                                for (name, val) in imported_scope {
                                    if !imported_env.public_items.contains(name) {
                                        continue;
                                    }
                                    if let Some(top_scope) = self.scopes.get_mut(0) {
                                        top_scope.insert(format!("{}.{}", alias, name), val.clone());
                                    }
                                }
                            }
                        } else {
                            if let Some(imported_scope) = imported_env.scopes.get(0) {
                                for (name, val) in imported_scope {
                                    if !imported_env.public_items.contains(name) {
                                        continue;
                                    }
                                    if let Some(top_scope) = self.scopes.get_mut(0) {
                                        top_scope.insert(name.clone(), val.clone());
                                    }
                                }
                            }
                        }
                    } else {
                        return Err(RuntimeError::at(
                            format!("imported module not found / loaded yet: {:?}", canonical_path),
                            us.span.clone(),
                        ));
                    }
                } else {
                    return Err(RuntimeError::at(
                        "cannot resolve imports without a file path context".to_string(),
                        us.span.clone(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Call a fn by the name the last-loaded file knows it by — a `dmc test`
    /// entry point, or a harness probe.
    pub fn call_named_fn(&mut self, name: &str, args: Vec<Value>) -> EvalResult<Value> {
        let f = self.resolve_fn(name).map(|k| self.fns[&k].clone()).ok_or_else(|| {
            RuntimeError::msg(format!("undefined fn `{}`", name))
        })?;
        self.call_fn(&f, args, f.span.clone())
    }

    fn collect_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f)       => {
                let mut f = f.clone();
                f.module = self.cur_module;
                self.fns.insert(module_key(self.cur_module, &f.name), f);
            }
            Item::ExternFn(e) => { self.extern_fns.insert(e.name.clone()); }
            Item::Model(m) => {
                let key = module_key(self.cur_module, &m.name);
                self.models.insert(key.clone(), key.clone());
                let shape_param_names: Vec<String> =
                    m.shape_params.iter().map(|sp| sp.name.clone()).collect();
                self.model_shape_params.insert(key.clone(), shape_param_names);
                let mut methods: HashMap<String, FnDecl> = HashMap::new();
                let mut fields: Vec<(String, crate::ast::Type)> = Vec::new();
                for member in &m.members {
                    match member {
                        ModelMember::Method(f) => {
                            let mut f = f.clone();
                            f.module = self.cur_module;
                            methods.insert(f.name.clone(), f);
                        }
                        ModelMember::Field { name, ty, .. } => { fields.push((name.clone(), ty.clone())); }
                    }
                }
                self.model_methods.insert(key.clone(), methods);
                self.model_fields.insert(key, fields);
            }
            // Under this file's number, like every other top-level name: a
            // private `enum E` in two files no longer shares one slot.
            Item::Enum(e) => {
                let variants: Vec<String> = e.variants.iter().map(|v| v.name.clone()).collect();
                self.enums.insert(module_key(self.cur_module, &e.name), variants);
            }
            // Likewise for a `type` alias, which SPEC §9 makes visible by the
            // same rule (#623 scoped it by load order instead; see the field).
            Item::TypeAlias(t) => {
                self.type_aliases.insert(module_key(self.cur_module, &t.name), t.clone());
            }
            // A module-level mutable binding is what a `@grad fn` can capture
            // (#398). Record the name (and the module's declaration order,
            // which orders the capture fields in `Grads`). Collection runs over
            // every item before any body does; the checker separately requires
            // the binding to be declared *above* the fn that reads it, so this
            // list is only ever consulted for names already in scope.
            Item::Let(l) if l.mutating || l.is_mut => {
                if let Pattern::Ident(name, _) = &l.pattern {
                    let entry = (self.cur_module, name.clone());
                    if !self.mut_globals.contains(&entry) {
                        self.mut_globals.push(entry);
                    }
                }
            }
            Item::Directive { inner, .. } => self.collect_item(inner),
            Item::Pub(inner) => self.collect_item(inner),
            _ => {}
        }
    }

    fn eval_top_level_item(&mut self, item: &Item) -> EvalResult<()> {
        match item {
            Item::Let(l) => {
                let v = self.eval_expr(&l.value)?;
                self.check_declared_port(&v, l.ty.as_ref(), &l.span)?;
                let mut v = self.coerce_declared(v, l.ty.as_ref());
                // #781, SPEC §3.4: same private-copy charge as the
                // statement-level `let !y = x` below (`Stmt::Let`) — a
                // module-level `let !alpha = x` is the same binding form,
                // just at top-level scope. `charge_private_copy` (not
                // `charge_intermediate`) excludes `Value::Str` — see that
                // fn's doc for why a string copy stays uncharged here.
                if l.mutating && matches!(l.value, Expr::Ident(_, _)) {
                    self.charge_private_copy(&v, Some(&l.span))?;
                    // #674: `#[derive(Clone)]`
                    // carried the SOURCE's origin tag; re-stamp to the budget
                    // current at THIS copy, which is what it is charged to.
                    restamp_current_budget(&mut v);
                }
                self.bind_pattern(&l.pattern, v);
                if let Some(axis) = l.ty.as_ref().and_then(streaming_axis_from_type) {
                    self.bind_stream_axis_pattern(&l.pattern, axis);
                }
            }
            Item::Directive { inner, .. } => self.eval_top_level_item(inner)?,
            Item::Pub(inner) => self.eval_top_level_item(inner)?,
            _ => {}
        }
        Ok(())
    }

    /// Call any callable `Value` (Fn, BoundFn, Builtin) with positional args.
    /// Used by higher-order list builtins (map, filter, reduce, sort_by).
    fn call_value(&mut self, callee: Value, args: Vec<Value>, sp: Span) -> EvalResult<Value> {
        match callee {
            Value::Fn(name) => {
                if self.extern_fns.contains(&name) {
                    return Err(RuntimeError::at(format!("extern fn `{}` cannot be called by the interpreter — a foreign call crosses an unchecked machine boundary this backend has no pointer representation for, so the spec confines `extern fn` calls to the JIT; run it with `dmc jit`", name), sp));
                }
                let f = self.fns.get(&name).cloned().ok_or_else(|| {
                    RuntimeError::at(format!("undefined fn `{}`", bare_name(&name)), sp.clone())
                })?;
                self.call_fn(&f, args, sp)
            }
            Value::BoundFn { ref name, ref shape_bindings } => {
                let f = self.fns.get(name).cloned().ok_or_else(|| {
                    RuntimeError::at(format!("undefined fn `{}`", bare_name(name)), sp.clone())
                })?;
                let bindings = shape_bindings.clone();
                self.call_fn_with_shapes(&f, args, &bindings, sp)
            }
            Value::Lambda { lit, captured_env, module } =>
                self.call_lambda(&lit.clone(), captured_env, module, args, sp),
            Value::Builtin(name) => self.call_builtin(&name, args, sp),
            Value::Opaque(name) if name == "\\>" || name == "\\<" => {
                if args.len() != 1 {
                    return Err(RuntimeError::at(
                        format!("activation {} expects 1 arg, got {}", name, args.len()),
                        sp,
                    ));
                }
                let op = if name == "\\>" { crate::ast::UnOp::ReLU } else { crate::ast::UnOp::GeLU };
                apply_unop(op, &args[0])
            }
            other => Err(RuntimeError::at(
                format!("expected callable, got {}", other.type_name()), sp,
            )),
        }
    }

    fn call_lambda(&mut self, lit: &crate::ast::FnLit, captured_env: HashMap<String, Value>, module: usize, args: Vec<Value>, sp: Span) -> EvalResult<Value> {
        if args.len() != lit.params.len() {
            return Err(RuntimeError::at(
                format!("lambda expects {} args, got {}", lit.params.len(), args.len()),
                sp,
            ));
        }
        let _depth = self.enter_call(&sp)?;
        self.push_scope();
        // Inject captured bindings first so that parameters shadow them when
        // the same name appears in both (standard lexical closure rule).
        for (name, val) in captured_env {
            self.bind(name, val);
        }
        for (p, v) in lit.params.iter().zip(args) {
            self.bind(&p.name, self.coerce_declared(v, p.ty.as_ref()));
        }
        let saved = std::mem::replace(&mut self.cur_module, module);
        let result = self.eval_block(&lit.body);
        // Same `?` rebuild as `call_fn_in_module`, and for the same reason:
        // the lambda's own declared `(T, Err)` names the `T`, and its scope
        // has to still be standing to resolve a shape-parametric one.
        //
        // #628 follow-up: an UNTYPED lambda (no `-> (T, Err)` of its own) has
        // no `T` here to rebuild against — but `check.rs`'s `PostfixOp::Query`
        // validates a `?` inside one against the LEXICALLY enclosing named
        // fn's declared return type, not the lambda's (a lambda literal never
        // touches `current_fn_ret`). Left alone, this catch site disagreed
        // with the checker it is supposed to defend: `dmc --check` accepted
        // `fn f() -> (str, str) { let c = fn() { g()? }; c() }` reading `?` as
        // legal against `f`'s `(str, str)`, and then `dmc run` refused it at
        // the lambda's own (nonexistent) return type. Falling back to
        // `named_fn_ret_stack`'s top — see its doc comment for exactly how
        // close a dynamic stand-in gets to that lexical rule — closes the gap.
        let fallback_ret: Option<Type> = if lit.ret_type.is_none() {
            self.named_fn_ret_stack.last().cloned().flatten()
        } else {
            None
        };
        let lambda_ret = lit.ret_type.as_ref().or(fallback_ret.as_ref());
        let result = self.finish_query_propagation(result, lambda_ret, "<lambda>", &sp);
        // #652 × #628: restore `cur_module` AFTER the rebuild, not before. The
        // `T` this rebuilds against comes from the lambda's own signature (or,
        // untyped, from the named fn on `named_fn_ret_stack`), and a
        // shape-parametric or aliased `T` resolves its `type` aliases through
        // `cur_module` — which for a lambda is the file the literal was
        // WRITTEN in, not the file that called it. Restoring first would look
        // the alias up in the caller's module and miss it.
        self.cur_module = saved;
        self.pop_scope();
        match result {
            // #473: a lambda's declared return type binds its width too.
            Ok(Flow::Normal(v)) | Ok(Flow::Return(v, _)) => {
                Ok(self.coerce_declared(v, lit.ret_type.as_ref()))
            }
            Ok(_) => Ok(Value::Nil),
            Err(e) => Err(e),
        }
    }

    /// Turn a `?` propagation signal into the value SPEC §4.9 says the
    /// enclosing function returns: `(default[T], e)`, with `T` the enclosing
    /// function's own declared success type and `e` the error that tripped.
    ///
    /// #628: the interpreter used to hand back the CALLEE's whole tuple, so
    /// the caller's success slot carried the callee's error-path value. The
    /// callee's `T` is not the caller's `T` — when the two differ it was not
    /// even the right type — and the JIT (`lower_query`) already builds the
    /// spec's tuple, so the backends disagreed on every `?` error path.
    ///
    /// Must be called while the callee's scope is still standing: a
    /// shape-parametric `T` (`-> (Tensor[f32, [N]], Err)`) resolves `N` out of
    /// that scope.
    ///
    /// Its two callers, `call_fn_in_module` and `call_lambda`, are the only
    /// two catch sites — NOT every function-call boundary has one.
    /// `call_grad_in_module`
    /// (`.fwd_bwd`/`.grad`/`.fwd`) does not call this at all (#647, open,
    /// deliberately not fixed here): a `?` inside a `@grad fn` escapes that
    /// frame uncaught and surfaces at whichever of these two catches it next,
    /// rebuilt against that CALLER's `T` rather than the `@grad fn`'s own.
    ///
    /// #639 added a THIRD body-evaluation path with the same shape, inside
    /// `call_fn_in_module` itself: a `@pp` fn's `stage K:` bodies return
    /// early, above this call, so a `?` that trips in a stage escapes the
    /// `@pp` fn uncaught and is likewise rebuilt against its CALLER's `T`
    /// (`@pp fn -> (i64, Err)` called from a `-> (str, Err)` fn answers `nil`,
    /// not `0`). Same category as #647 and left with it — #639 made that path
    /// catch the block-flow carrier, which is a different signal with a
    /// different owner, and widening it to `?` is a behaviour change belonging
    /// to #647's fix, not to a merge.
    fn finish_query_propagation(
        &mut self,
        result: EvalResult<Flow>,
        ret_type: Option<&Type>,
        owner: &str,
        sp: &Span,
    ) -> EvalResult<Flow> {
        // #639: the guard is on `propagate` specifically, not on "is this an
        // `Err`?". A block-flow carrier is an `Err` with `propagate: None`, and
        // it belongs to `catch_flow` at the statement/block boundary below this
        // one — by the time control reaches here `eval_block` has already
        // re-opened it, and an uncaught one (the `@pp` stage path) has to keep
        // travelling rather than be rebuilt as a `?` early return.
        let signal = match result {
            Err(e) if e.propagate.is_some() => e,
            other => return other,
        };
        // The `?` operator's own span, set by `RuntimeError::propagate`.
        let sp = &signal.span.clone().unwrap_or_else(|| sp.clone());
        let err_half = *signal.propagate.expect("checked just above");
        // `?` is legal only inside a function whose own return type is
        // `(_, Err)` (SPEC §4.9), and `check.rs` refuses the rest before
        // either backend runs — so this arm is a defense, exactly like the
        // matching fallback in the JIT's `lower_query`.
        // #652: the signature may spell the whole `(T, Err)` as a top-level
        // `type` alias, and an alias means what the module whose body is
        // running spells it — which is why both call sites rebuild before
        // restoring `cur_module`.
        let ret_resolved = ret_type.map(|t| self.resolve_alias_ty(peel_grouping(t)).clone());
        let t = match ret_resolved.as_ref().map(peel_grouping) {
            Some(Type::Tuple(els, _)) if els.len() == 2 => els[0].clone(),
            _ => {
                return Err(RuntimeError::at(
                    format!(
                        "`?` is only legal inside a function returning `(T, Err)`; \
                         `{}` returns `{}`",
                        owner,
                        ret_type.map(type_label).unwrap_or_else(|| "nothing".to_string()),
                    ),
                    sp.clone(),
                ))
            }
        };
        let default_t = self.query_default_value(&t, sp)?;
        // #639 gave `Flow::Return` a span; the one that fits a `?` early return
        // is the `?` postfix expression's own, which is the `sp` narrowed just
        // above. Nothing reads it on this path today (both catch sites match
        // `Flow::Return(v, _)`), but a `Flow` that carries no span is one a
        // later diagnostic cannot point at.
        Ok(Flow::Return(Value::Tuple(vec![default_t, err_half]), sp.clone()))
    }

    /// SPEC §4.9's `default[T]` intrinsic: "the zero value of type `T` (0 /
    /// 0.0 / false / nil / zero-tensor), used only in the `?` desugaring."
    ///
    /// Scoped to exactly that enumeration — scalars, `str`/`Err`/`nil` (whose
    /// zero value IS `nil`), and an all-zero tensor. A model, an enum, a map,
    /// a port, an array, a nested tuple has no default the spec defines, so it
    /// is refused rather than invented. That is the same set `jit.rs`'s
    /// `default_value_for` constructs and refuses, so the two backends agree
    /// on what `?` can propagate through; the JIT refuses while lowering, and
    /// the interpreter — which has no lowering pass — refuses when the error
    /// path is actually taken.
    fn query_default_value(&mut self, ty: &Type, sp: &Span) -> EvalResult<Value> {
        // #652 × #628: `T` may be spelled as a top-level `type` alias
        // (`pub type Slot = u8` → `-> (Slot, Err)`), and SPEC §6.6 scopes an
        // alias to the module that declared it. Resolving it here is what
        // makes `default[Slot]` the `u8` zero rather than a refusal of a name
        // that is a perfectly good scalar one indirection away — and it is
        // what makes the ORDER at the two call sites load-bearing: both rebuild
        // while `cur_module` still names the file whose signature this is, so
        // the lookup lands in that file's alias table and not the caller's.
        // Diagnostics below keep the original spelling: `Slot` is what the
        // signature says, and what the reader has to go and look at.
        let resolved = self.resolve_alias_ty(peel_grouping(ty)).clone();
        match peel_grouping(&resolved) {
            Type::Scalar(st, _) => Ok(default_scalar_value(st)),
            // `Err` (SPEC §3.9) is `nil` on success and a `str` message on
            // failure, so its zero value is `nil` — the same value `str` gets,
            // and the same call `jit.rs` makes (`Err` lowers to `TyKind::Str`).
            Type::Named { name, .. } if name == "Err" => Ok(Value::Nil),
            Type::Tensor(elem, shape, _) => {
                // An aliased ELEMENT type (`type P = f32` → `Tensor[P, [N]]`)
                // is the same indirection one level in.
                let elem_resolved = self.resolve_alias_ty(peel_grouping(elem)).clone();
                let Type::Scalar(est, _) = peel_grouping(&elem_resolved) else {
                    return Err(RuntimeError::at(
                        format!("`?`: tensor element type of `{}` is not a scalar", type_label(ty)),
                        sp.clone(),
                    ));
                };
                let mut dims = Vec::with_capacity(shape.elems.len());
                for e in &shape.elems {
                    match e {
                        // #628 follow-up: this axis is evaluated STRICTLY and
                        // inline, refusing loudly exactly like the wildcard-
                        // axis arm just below, rather than through
                        // `eval_dim` — `default[T]`'s zero tensor has to have
                        // the shape `T` actually declares, and this call
                        // site wants its own message (naming `T` and citing
                        // `?`) and its own span (the `?`'s, not the failing
                        // sub-expression's) rather than `eval_dim`'s. #839
                        // made `eval_dim` itself strict too — it no longer
                        // has a loose default to differ from — so this inline
                        // copy survives only for its own wording and span,
                        // not because `eval_dim` would get it wrong.
                        crate::ast::ShapeElem::Expr(ex) => match self.eval_expr(ex) {
                            Ok(Value::Int(n, _)) if n >= 0 => dims.push(n as usize),
                            Ok(Value::Float(x, _)) if x >= 0.0 => dims.push(x as usize),
                            Ok(v) => return Err(RuntimeError::at(
                                format!(
                                    "`?`'s early return needs a concrete non-negative \
                                     dimension for `{}`, but a shape axis evaluated to \
                                     `{}` ({})",
                                    type_label(ty), v, v.type_name(),
                                ),
                                sp.clone(),
                            )),
                            Err(e) => return Err(e),
                        },
                        // A wildcard/spread/streaming axis has no size, so
                        // there is no zero tensor to build.
                        _ => {
                            return Err(RuntimeError::at(
                                format!(
                                    "`?`'s early return needs a zero tensor for `{}`, \
                                     but its declared shape has an axis with no size",
                                    type_label(ty),
                                ),
                                sp.clone(),
                            ))
                        }
                    }
                }
                self.prof_alloc();
                Ok(Value::tensor_dt(
                    ArrayD::<f64>::zeros(IxDyn(&dims)),
                    default_tensor_dtype(est),
                ))
            }
            _ => Err(RuntimeError::at(
                format!(
                    "`?`'s early return needs a default value for `{}`, which has none: \
                     SPEC §4.9 defines `default[T]` as 0 / 0.0 / false / nil / zero-tensor, \
                     so only scalars, `str`/`Err` and tensors have one",
                    // The signature's own spelling, alias and all — that is
                    // the line the reader has to go and change.
                    type_label(peel_grouping(ty)),
                ),
                sp.clone(),
            )),
        }
    }

    // ── Scoping ───────────────────────────────────────────────────────────

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.shape_param_scopes.push(std::collections::HashSet::new());
        self.stream_axes.push(HashMap::new());
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.shape_param_scopes.pop();
        self.stream_axes.pop();
    }

    fn bind(&mut self, name: impl Into<String>, v: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), v);
        }
    }

    /// #517: like `bind`, but also tags `name` as a SHAPE PARAMETER in the
    /// current frame (`shape_param_scopes`) — a value inferred from an arg's
    /// tensor shape or supplied via `f[N]`/`f[S=4]`, matching what the JIT's
    /// `shape_env` holds for the same call. Every shape-param binding site
    /// (`call_fn_with_shapes`, `@grad`'s own shape-param setup) uses this
    /// instead of `bind` so `affine_form` can tell a shape param apart from
    /// an ordinary `let`/`for`/plain-param binding that merely happens to
    /// hold an int right now.
    fn bind_shape_param(&mut self, name: impl Into<String>, v: Value) {
        let name = name.into();
        if let Some(scope) = self.shape_param_scopes.last_mut() {
            scope.insert(name.clone());
        }
        self.bind(name, v);
    }

    /// #517: is `name`, as `lookup` would currently resolve it (innermost
    /// scope first), a SHAPE PARAMETER rather than an ordinary binding? Must
    /// mirror `lookup`'s own scan order: an inner `let x = ...` shadowing an
    /// outer shape param `x` is correctly NOT a shape param, and vice versa.
    fn is_shape_param(&self, name: &str) -> bool {
        for (scope, shape_names) in self.scopes.iter().rev().zip(self.shape_param_scopes.iter().rev()) {
            if scope.contains_key(name) {
                return shape_names.contains(name);
            }
        }
        false
    }

    fn bind_stream_axis(&mut self, name: impl Into<String>, axis: usize) {
        if let Some(scope) = self.stream_axes.last_mut() {
            scope.insert(name.into(), axis);
        }
    }

    fn lookup(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(v) = scope.get(name) { return Some(v.clone()); }
        }
        None
    }

    /// The `fns` key a source-level fn name denotes in the executing file:
    /// one of its own fns, or a `pub` fn it imported (bare or `alias.name`).
    fn resolve_fn(&self, name: &str) -> Option<String> {
        let key = module_key(self.cur_module, name);
        self.fns.contains_key(&key).then_some(key)
    }

    /// The declaring file's key for a source-level model name in the
    /// executing file — what instances are tagged with.
    fn resolve_model(&self, name: &str) -> Option<String> {
        self.models.get(&module_key(self.cur_module, name)).cloned()
    }

    /// Whether `@cast(name)` names a model to overlay rather than a dtype to
    /// convert to. The two `@cast` sites must ask this and not
    /// `model_fields.contains_key(name)` directly: `model_fields` is keyed by
    /// the DECLARING file's key, so a source-level name never matched it and
    /// the overlay silently fell through to `apply_cast_by_name`, handing back
    /// the source byte tensor unchanged.
    fn is_overlay_model(&self, name: &str) -> bool {
        self.resolve_model(name)
            .is_some_and(|key| self.model_fields.contains_key(&key))
    }

    /// `lookup` + FieldRef resolution (#444): a binding created by
    /// `let !alias = recv.field` reads as the field's CURRENT value anywhere
    /// a plain value would. Callers that need the raw alias (write-through,
    /// stream append) keep using `lookup`.
    fn lookup_deref(&self, name: &str) -> Option<Value> {
        match self.lookup(name) {
            Some(Value::FieldRef { rc, field }) => {
                let borrowed = rc.borrow();
                borrowed.iter().find(|(k, _)| k == &field).map(|(_, v)| v.clone())
            }
            other => other,
        }
    }

    fn lookup_stream_axis(&self, name: &str) -> Option<usize> {
        for scope in self.stream_axes.iter().rev() {
            if let Some(axis) = scope.get(name) { return Some(*axis); }
        }
        None
    }

    fn assign(&mut self, name: &str, v: Value) -> EvalResult<()> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                // A FieldRef binding is an alias of a model field (#444):
                // every assignment through it writes the FIELD and keeps the
                // alias — matching element writes and `self.field = …`.
                // Previously this rebound the local, so `b = …` after
                // `let !b = self.buf` silently never reached the field.
                // `v` is always fully evaluated before assign is called, so
                // no interpreter re-entry can occur under this borrow (#443).
                if let Some(Value::FieldRef { rc, field }) = scope.get(name) {
                    let rc = rc.clone();
                    let field = field.clone();
                    let mut borrowed = rc.borrow_mut();
                    return match borrowed.iter_mut().find(|(k, _)| k == &field) {
                        Some((_, slot)) => { *slot = v; Ok(()) }
                        None => Err(RuntimeError::msg(
                            format!("undefined field `{}`", field))),
                    };
                }
                scope.insert(name.to_string(), v);
                return Ok(());
            }
        }
        Err(RuntimeError::msg(format!("cannot assign to undefined `{}`", name)))
    }

    /// #452: release the about-to-be-overwritten binding's handle on a list
    /// buffer that was just passed to a builtin.
    ///
    /// `toks = list_push(toks, t)` hands the builtin a second handle on the
    /// buffer while the binding still holds the first, so a copy-on-write push
    /// would copy every time — O(n) per element, O(n²) to build the list. But
    /// the binding is dead: the statement overwrites it with this call's
    /// result. Clearing it here, after the arguments are evaluated and before
    /// the builtin runs, leaves the buffer uniquely owned and the push lands in
    /// place.
    ///
    /// Deliberately narrow — it fires only when all of these hold:
    ///   * this call is the *entire* right-hand side of `name = …`, so nothing
    ///     else in the statement can still read the old value (`move_hint`, set
    ///     in `Stmt::Expr`, which is where that shape is matched),
    ///   * the call's first positional argument is that same bare `name`,
    ///   * the binding still holds the *identical* buffer that was passed in.
    /// If the builtin then fails, the binding is left `Nil`; a runtime error
    /// unwinds out of the program, so no demoniC code can observe it.
    fn release_move_target(&mut self, hint: Option<&str>, first_arg: Option<&CallArg>, arg_vals: &[Value]) {
        let Some(target) = hint else { return };
        let Some(CallArg::Positional(Expr::Ident(argname, _))) = first_arg else { return };
        if argname != target { return; }
        let Some(Value::List(passed)) = arg_vals.first() else { return };
        // Raw pointer, not a clone — an extra handle here would defeat the point.
        let passed = Rc::as_ptr(passed);
        for scope in self.scopes.iter_mut().rev() {
            if let Some(slot) = scope.get_mut(target) {
                if matches!(slot, Value::List(held) if std::ptr::eq(Rc::as_ptr(held), passed)) {
                    *slot = Value::Nil;
                }
                return;
            }
        }
    }

    fn bind_pattern(&mut self, pat: &Pattern, v: Value) {
        match pat {
            // #336: a bare ident naming a variant of an enum value is a variant
            // match, not a binding — don't shadow the variant name.
            Pattern::Ident(n, _)
                if matches!(&v, Value::EnumVal { enum_name, .. }
                    if self.enum_variant_ordinal(enum_name, n).is_some()) => {}
            // #350 Part 2: a payload-variant pattern binds its sub-patterns to
            // the value's payload, positionally. Tag-only EnumVariant patterns
            // (empty `bindings`) fall through to the no-op arm below.
            Pattern::EnumVariant { bindings, .. } if !bindings.is_empty() => {
                if let Value::EnumVal { payload, .. } = v {
                    for (i, b) in bindings.iter().enumerate() {
                        let pv = payload.get(i).cloned().unwrap_or(Value::Nil);
                        self.bind_pattern(b, pv);
                    }
                } else {
                    for b in bindings { self.bind_pattern(b, Value::Nil); }
                }
            }
            // (the `_` -> {} no-op arm below also covers tag-only `EnumVariant`)
            Pattern::Ident(n, _) if n != "_" => self.bind(n, v),
            Pattern::Wildcard(_) => {}
            Pattern::Tuple(pats, _) => {
                if let Value::Tuple(vals) = v {
                    let (before, after, has_rest) = crate::ast::tuple_rest_split(pats);
                    let ok = if has_rest { vals.len() >= before.len() + after.len() }
                             else { pats.len() == vals.len() };
                    if ok {
                        if has_rest {
                            for (p, val) in before.iter().zip(&vals) {
                                self.bind_pattern(p, val.clone());
                            }
                            let tail = vals.len() - after.len();
                            for (p, val) in after.iter().zip(&vals[tail..]) {
                                self.bind_pattern(p, val.clone());
                            }
                        } else {
                            for (p, v) in pats.iter().zip(vals) {
                                self.bind_pattern(p, v);
                            }
                        }
                        return;
                    }
                }
                // Shape mismatch — leave the named bindings nil (`..` binds nothing).
                for p in pats {
                    if !matches!(p, Pattern::Rest(_)) { self.bind_pattern(p, Value::Nil); }
                }
            }
            // #393: `x @ pat` — bind the binder name to the whole value, then
            // recurse into the sub-pattern for any nested bindings it introduces.
            Pattern::Bind(binder, sub, _) => {
                self.bind_pattern(binder, v.clone());
                self.bind_pattern(sub, v);
            }
            _ => {}  // shape patterns, literals — pre-alpha no-op
        }
    }

    fn bind_stream_axis_pattern(&mut self, pat: &Pattern, axis: usize) {
        match pat {
            Pattern::Ident(name, _) if name != "_" => self.bind_stream_axis(name, axis),
            Pattern::Tuple(pats, _) => {
                for p in pats {
                    self.bind_stream_axis_pattern(p, axis);
                }
            }
            Pattern::Bind(inner, _, _) => self.bind_stream_axis_pattern(inner, axis),
            _ => {}
        }
    }

    // ── @grad ─────────────────────────────────────────────────────────────

    /// Dispatch for the @grad calling conventions:
    ///   `f.fwd_bwd(args)` → `(loss, {param: grad, ...})`
    ///   `f.grad(args)`    → `{param: grad, ...}`     (no forward value)
    ///   `f.fwd(args)`     → forward value only       (same as `f(args)`)
    ///
    /// Real reverse-mode autodiff via tape:
    ///   1. Walk the function body forward, recording every tensor-producing
    ///      op into a tape (op-kind + input-node-ids + output-value).
    ///   2. After forward, seed dL/d(output) = 1.0 (the function must return a
    ///      scalar loss).
    ///   3. Walk the tape in reverse, applying the VJP rule for each op to
    ///      accumulate gradients into input nodes.
    ///   4. Hand back the gradients for each `!`-marked tensor parameter.
    ///
    /// This is O(forward) — the right complexity for autodiff.
    /// Same module re-scoping as `call_fn_with_shapes`: a `@grad fn`'s params
    /// and loss are annotation sites too, and they mean what the module that
    /// declared the fn spells.
    fn call_grad(
        &mut self,
        fn_name: &str,
        method: &str,
        args: Vec<Value>,
        span: Span,
    ) -> EvalResult<Value> {
        // The taped body resolves names — and its annotations' `type` aliases
        // — in the fn's own file, like any call.
        let home = self.fns.get(fn_name).map_or(self.cur_module, |f| f.module);
        let saved = std::mem::replace(&mut self.cur_module, home);
        let result = self.call_grad_in_module(fn_name, method, args, span);
        self.cur_module = saved;
        result
    }

    fn call_grad_in_module(
        &mut self,
        fn_name: &str,
        method: &str,
        args: Vec<Value>,
        span: Span,
    ) -> EvalResult<Value> {
        let f = self.fns.get(fn_name).cloned().ok_or_else(||
            RuntimeError::at(format!("@grad: undefined fn `{}`", bare_name(fn_name)), span.clone()))?;
        if args.len() != f.params.len() {
            return Err(RuntimeError::at(
                format!("@grad `{}` expects {} args, got {}", bare_name(fn_name), f.params.len(), args.len()),
                span,
            ));
        }
        // `fwd` is the cheap path — no tape needed.
        if method == "fwd" {
            return self.call_fn(&f, args, span);
        }
        // Second-order autodiff requires stacked `@grad @grad` (matching the
        // JIT, which only emits a `$fwd_bwd_bwd` entry for grad_count >= 2).
        if method == "fwd_bwd_bwd"
            && f.directives.iter().filter(|d| d.name == "grad").count() < 2
        {
            return Err(RuntimeError::at(format!(
                "`{}.fwd_bwd_bwd` (second-order autodiff) requires stacked \
                 `@grad @grad` on fn `{}`", fn_name, fn_name), span));
        }
        // Forward + tape build. The function body must be straight-line:
        // a sequence of let-bindings followed by a tail expression. Control
        // flow (if/for/while/match) inside @grad isn't supported yet — the
        // tape design doesn't branch.
        self.push_scope();
        // Bind shape params from arg shapes (same as call_fn).
        let mut inferred: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (p, v) in f.params.iter().zip(&args) {
            let Some(ty) = &p.ty else { continue; };
            let Some(spec) = type_shape_spec(ty) else { continue; };
            collect_idents_in_spec(spec, &mut referenced);
            let Value::Tensor(t) = v else { continue; };
            infer_shape_from_arg(spec, t.shape(), &mut inferred, &f.name, &span)?;
        }
        for sp_decl in &f.shape_params { referenced.insert(sp_decl.name.clone()); }
        for name in &referenced {
            match inferred.get(name) {
                Some(&dim) => self.bind_shape_param(name, Value::Int(dim, IW::I64)),
                None => {
                    self.pop_scope();
                    return Err(RuntimeError::at(format!(
                        "@grad `{}`: shape param `{}` cannot be inferred from args",
                        f.name, name), span));
                }
            }
        }

        // #791: the AD boundary this call's activations are reclaimed at, the
        // interpreter's twin of `jit.rs`'s `grad_call_prologue`/
        // `grad_call_epilogue` `InternalMark` pair (#614, PR #779). Everything
        // the tape materializes from here on — forward activations, every
        // adjoint the backward pass builds — is a compiler-inserted temporary
        // the user never asked to keep, so it is metered while it is live and
        // handed back at the epilogue below, leaving only what actually
        // escapes: the published gradients and the loss.
        //
        // Taken before the tape exists so the window covers the whole pass.
        // `vault_used` is deliberately not part of it — see the same carve-out
        // in `call_fn_in_module`'s forward-only mark.
        // #674 slice 3d: the ENTERED budget, not entry 0 — a `.fwd_bwd`
        // called inside `forge.enter(b) { … }` charges its temporaries to
        // `b`, so `b` is what this window releases. `BudgetGuard` pops on
        // every exit edge, so the register reads the same here and at the
        // matching `set_budget_used` below (the same mark/rewind rule the
        // JIT's own grad prologue/epilogue pair follows, applied here).
        let ad_budget = cur_budget();
        let ad_mark = self.budgets[ad_budget].used;
        // #781: the same explicit-constructor carve-out the forward-only mark
        // takes (`call_fn_in_module`, and see `ctor_forge_bytes`). A `forge.*`
        // constructor written inside a `.fwd_bwd` body is not one of the
        // pass's own temporaries, and its result can escape the pass — into a
        // model field, a module-level `let !` — and stay live after the
        // epilogue. Those bytes are added back at the epilogue rather than
        // released.
        let ad_ctor_at_mark = self.ctor_forge_bytes;
        let mut tape = Tape::new();
        let mut var_node: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        // Bind params: tensor params become Input nodes; the `mutating` flag
        // tells the backward pass which ones to return gradients for.
        // A **scalar `!` parameter** is an Input node too (#420 item 2), so
        // `@grad fn f(!a: f32) -> f32 { a*a*a }` differentiates without having
        // to route the value through a one-element tensor. Only `!` float
        // scalars join the tape: an immutable scalar has no gradient to
        // return, and an integer parameter isn't differentiable.
        for (i, (p, v)) in f.params.iter().zip(&args).enumerate() {
            let bound = self.coerce_declared(v.clone(), p.ty.as_ref());
            self.bind(&p.name, bound.clone());
            let traced = matches!(bound, Value::Tensor(_))
                || (p.mutating && matches!(bound, Value::Float(_, _)));
            if traced {
                let node_id = tape.push(TapeNode {
                    op: TapeOp::Input { param_idx: i, mutating: p.mutating },
                    inputs: Vec::new(),
                    value: bound,
                });
                var_node.insert(p.name.clone(), node_id);
            }
        }

        // AUTODIFF.md §2 / #398: a captured `mut` binding is a differentiable
        // input too. Seed one tape input per captured binding, holding its value
        // at call time, so reads of it in the body are traced and its adjoint
        // comes back in `Grads` under the binding's own name. The node ids are
        // kept here rather than read back out of `var_node` at the end: a body
        // that reassigns the capture (`b = b .+ 1`) repoints `var_node`, and the
        // gradient we owe is the one w.r.t. the value the call started from.
        //
        // A capture resolves to the **module binding**, never to whatever
        // `lookup` finds on the way out through the caller's frames. The
        // interpreter shares one scope stack across calls, so `go()` holding a
        // local of the same name used to have that local taped under the
        // capture's name — same `@grad fn`, same module state, a different
        // gradient depending on who called it. Any such intervening binding is
        // lifted out for the duration of the call and put back below, which
        // leaves scope 0 (the module scope) as the only `cap` in sight: reads
        // in the body and the value on the tape are then the same thing by
        // construction, and a reassignment in the body still writes through to
        // the module binding as AUTODIFF §2.1 says it does.
        let capture_names = self.grad_capture_names(&f);
        // Scope 0 is the module scope (what we want) and the last scope is this
        // call's own frame, already holding the params and shape params — a
        // binding there shadows legitimately. Only the CALLER frames in between
        // are masked.
        let mut masked: Vec<(usize, String, Value)> = Vec::new();
        let caller_frames = self.scopes.len().saturating_sub(1);
        for name in &capture_names {
            for i in 1..caller_frames {
                if let Some(v) = self.scopes[i].get(name) {
                    masked.push((i, name.clone(), v.clone()));
                }
            }
        }
        for (i, name, _) in &masked {
            self.scopes[*i].remove(name);
        }
        let mut capture_nodes: Vec<(String, usize)> = Vec::new();
        for name in capture_names {
            let Some(v) = self.scopes.first().and_then(|g| g.get(&name)).cloned() else { continue };
            if !value_is_differentiable(&v) { continue; }
            let node_id = tape.push(TapeNode {
                op: TapeOp::Captured { name: name.clone() },
                inputs: Vec::new(),
                value: v.clone(),
            });
            var_node.insert(name.clone(), node_id);
            capture_nodes.push((name, node_id));
        }

        // Forward pass: evaluate the body, recording tensor ops to the tape.
        //
        // #647 (open, not fixed here — different category/PR): unlike
        // `call_fn_in_module`/`call_lambda`, there is no
        // `finish_query_propagation` call on `result` before the scope pops.
        // A `?` inside a `@grad fn -> (T, Err)` therefore escapes this frame
        // uncaught and is rebuilt by whichever of those two catches it next —
        // against THAT frame's `T`, not this `@grad fn`'s own.
        let result = self.eval_block_with_tape(&f.body, &mut tape, &mut var_node, &span);
        self.pop_scope();
        // Put back whatever the capture masking lifted out, before any `?`
        // below can leave through an error path. The backward pass reads
        // captures off `capture_nodes`, not the scope stack, so restoring here
        // is early enough to be unconditional and late enough to be correct.
        for (i, name, v) in masked {
            if let Some(scope) = self.scopes.get_mut(i) {
                scope.insert(name, v);
            }
        }
        // #639: the tape walker hands a `Expr::Block` to plain `eval_expr`, so
        // the block-flow carrier can reach this boundary — and a `@grad` body
        // is a function boundary, the second of the two that bypass
        // `eval_block`. Without this catch a `return` inside the body rode `?`
        // straight out of `call_grad` and returned from the CALLER (a
        // check-clean program silently answered the caller's frame away).
        //
        // It is refused rather than honoured: the carrier holds a `Value`, not
        // a tape node, so a value that left through it never joined the
        // gradient graph and there is nothing to differentiate. That is the
        // same line `trace_stmt` already takes on a bare `return` statement in
        // a `@grad` body, and `.fwd` — which goes through `call_fn` — keeps
        // running such a body normally. #615's separate rule (a block ON the
        // gradient path is not traced, so the loss does not participate) has
        // its own diagnostic below and is untouched.
        let result = match result {
            Err(e) if e.flow.is_some() => {
                let esc = e.flow.expect("just checked");
                Err(RuntimeError::at(format!(
                    "@grad `{}`: a `{}` inside the differentiated body is not \
                     supported — it leaves the body with a value the tape never \
                     recorded, so there is no gradient to compute. Supported: \
                     let, reassignment of a named accumulator, while/for loops, \
                     and if/match expressions; break/continue/return and the \
                     unbounded `loop` are not wired.",
                    f.name, esc.keyword()), esc.span().clone()))
            }
            other => other,
        };
        let (loss_value, loss_node) = result?;

        // The function's tail must produce a scalar (real-valued loss). The
        // tape-recording layer returns the node id only when the tail is a
        // tensor-tracked computation. For scalar losses, find the most recent
        // tracked tensor node (the one whose `Sum`/`ScalarDiv` produced the
        // scalar); the tape's `loss_node` field holds it.
        let loss_node = loss_node.ok_or_else(|| RuntimeError::at(format!(
            "@grad `{}`: function body must produce a scalar derived from a \
            differentiable input (got a value that doesn't participate in the \
            gradient graph).\n\
            hint: gradients flow from `!` parameters — tensors, and (#420) f32 \
            scalars. A partial index `x[i]` on a rank > 1 tensor yields a \
            sub-tensor and exits the graph; reduce with a traced reduction \
            (`sum(x)` / `mean(x)`, e.g. `sum(x .* x)`) or read a single element \
            with a full index.\n\
            `argmax`/`argmin` and fused `embed` don't trace yet.\n\
            See examples/autograd.dmc for working patterns.", f.name),
            span.clone()))?;

        // Backward pass. First-order: seed dL/dL = 1 and propagate values.
        // Second-order (`fwd_bwd_bwd`): replay the backward pass symbolically
        // onto the tape, reduce the first `!` param's gradient to a scalar
        // via sum, and seed the ordinary backward there — yielding
        // g2.p = ∇p(sum(∇p₀ L)) for each `!` param p, the same Hessian
        // row-sum semantics as the JIT's `$fwd_bwd_bwd` entry.
        let mut grads: Vec<Option<Value>>;
        if method == "fwd_bwd_bwd" {
            // #791: `backward_symbolic` replays the backward pass ONTO the
            // tape, so the nodes it appends carry freshly computed tensors
            // (`push_dotmul`, `push_negate`, the `ActivationGrad` and mask
            // nodes) that no interpreter path metered. Everything the
            // FORWARD pass put on the tape was already charged where it was
            // computed, so only the appended range is charged here.
            let symbolic_from = tape.nodes.len();
            let adj = tape.backward_symbolic(loss_node, &span)?;
            for i in symbolic_from..tape.nodes.len() {
                let v = tape.nodes[i].value.clone();
                self.charge_intermediate(&v, Some(&span))?;
            }
            let p0 = f.params.iter()
                .find(|p| p.mutating)
                .and_then(|p| var_node.get(&p.name).copied().map(|id|
                    (format!("`!` param `{}`", p.name), id)))
                // No `!` param: a captured `mut` binding is the differentiable
                // input the second-order pass reduces over instead (#398).
                .or_else(|| capture_nodes.first().map(|(name, id)|
                    (format!("captured `mut` binding `{}`", name), *id)));
            let (p0_label, p0_node) = p0.ok_or_else(|| RuntimeError::at(format!(
                "@grad `{}`: fwd_bwd_bwd needs a `!` tensor parameter or a \
                 captured `mut` binding",
                f.name), span.clone()))?;
            let g1 = adj[p0_node].ok_or_else(|| RuntimeError::at(format!(
                "@grad `{}`: the second-order pass found no first-order gradient \
                 for {} to differentiate again — either the loss does not depend \
                 on it, or its only path to the loss is one the second-order \
                 replay does not cover (the scalar `*` multiplier operand is the \
                 known gap; first order is unaffected)",
                f.name, p0_label), span.clone()))?;
            let loss2 = match &tape.nodes[g1].value {
                Value::Tensor(t) => {
                    let total: f64 = t.iter().sum();
                    tape.push(TapeNode {
                        op: TapeOp::Sum, inputs: vec![g1], value: Value::Float(total, FW::F64),
                    })
                }
                _ => g1,
            };
            grads = vec![None; tape.nodes.len()];
            grads[loss2] = Some(Value::Float(1.0, FW::F64));
        } else {
            grads = vec![None; tape.nodes.len()];
            grads[loss_node] = Some(Value::Float(1.0, FW::F64));
        }
        tape.backward(&mut grads, &span)?;
        // #791: `Tape::backward` builds its adjoints with ndarray directly and
        // never re-enters the interpreter, so not one of them can reach
        // `charge_intermediate` where it is made. They are metered here
        // instead, once the pass is complete — nothing releases an adjoint
        // while the pass runs, so the set standing at the end IS the pass's
        // high-water mark and charging it in one go bills the same bytes an
        // allocation-by-allocation charge would have.
        for i in 0..grads.len() {
            if let Some(g) = grads[i].clone() {
                self.charge_intermediate(&g, Some(&span))?;
            }
        }

        // Collect gradients for each `!`-marked parameter (tensor, or — since
        // #420 item 2 — f32 scalar).
        let mut grad_struct: Vec<(String, Value)> = Vec::new();
        for p in f.params.iter() {
            if !p.mutating { continue; }
            let Some(node_id) = var_node.get(&p.name) else { continue; };
            let g = grads[*node_id].clone().unwrap_or_else(|| {
                // No gradient flowed back — this parameter doesn't affect
                // the loss. Surface it as zeros of the param shape, since
                // that IS the mathematically correct answer (∂L/∂x=0 when
                // L doesn't depend on x), not a placeholder. Shaped from the
                // tape's own input value, which is the *coerced* parameter —
                // handing back the parameter itself would read as a gradient.
                match &tape.nodes[*node_id].value {
                    Value::Tensor(t) => Value::tensor(ArrayD::zeros(t.raw_dim())),
                    Value::Float(_, w) => Value::Float(0.0, *w),
                    other => other.clone(),
                }
            });
            grad_struct.push((p.name.clone(), g));
        }
        // Captured `mut` bindings follow the `!` params, in the order the
        // module declares them (AUTODIFF.md §2). Same zero-gradient rule as a
        // parameter: a capture the loss doesn't depend on gets zeros of its own
        // shape, which is the correct ∂L/∂x = 0, not a placeholder.
        for (name, node_id) in &capture_nodes {
            let g = grads[*node_id].clone().unwrap_or_else(|| {
                match &tape.nodes[*node_id].value {
                    Value::Tensor(t) => Value::tensor(ArrayD::zeros(t.raw_dim())),
                    _ => Value::Float(0.0, FW::F64),
                }
            });
            grad_struct.push((name.clone(), g));
        }
        // #791: the AD epilogue — `jit.rs`'s `grad_call_epilogue` rewind,
        // spelled in the interpreter's own terms. Every byte charged since
        // `ad_mark` was a compiler-inserted temporary of this pass (forward
        // activations, the adjoints for nodes nothing publishes), so the whole
        // window is handed back; then the values that genuinely OUTLIVE the
        // call are charged again, because they are what the caller now holds:
        // the published gradients, and the loss when it is a tensor rather
        // than the usual scalar. That leaves the interpreter charging one
        // gradient set per `fwd_bwd` — the same monotonic slope the JIT's
        // `capture_grad` leaves standing outside its own rewind — instead of a
        // whole pass per step.
        //
        // The re-charge cannot fail the budget: it re-bills bytes that were
        // already inside the window just released, so `forge_used` lands at or
        // below the peak this pass had already passed the check at. It is
        // still routed through `charge_intermediate` rather than added
        // straight to the counter, so one rule keeps deciding what a
        // materialization costs.
        self.set_budget_used(ad_budget,
            ad_mark.saturating_add(self.ctor_forge_bytes.saturating_sub(ad_ctor_at_mark)));
        for (_, g) in grad_struct.iter() {
            let g = g.clone();
            self.charge_intermediate(&g, Some(&span))?;
        }
        self.charge_intermediate(&loss_value, Some(&span))?;
        let grad_value = Value::Struct(Rc::new(RefCell::new(grad_struct)));
        // #473's rule reaches the loss too. An ordinary call already narrows
        // its result to the declared return type (`call_fn`, same helper); the
        // `@grad` entry handed back the raw tape value, so one `@grad fn f(..)
        // -> f32 { sum(w) + 0.1 }` produced TWO different numbers depending on
        // how it was called — `f(w)` gave the f32, `f.fwd_bwd(w)` the f64 — and
        // the JIT, whose tape is f32 throughout, matched only the first. That
        // is an AUTODIFF §6.1 divergence in the loss (the gradients were always
        // f32 and always agreed), and it is why #422 wall 2's neighbourhood
        // still disagreed after the malformed IR was fixed.
        let loss_value = self.coerce_declared(loss_value, f.ret_type.as_ref());
        match method {
            "grad" => Ok(grad_value),
            "fwd_bwd" | "fwd_bwd_bwd" => Ok(Value::Tuple(vec![loss_value, grad_value])),
            _ => unreachable!(),
        }
    }

    /// The captured `mut` bindings a `@grad fn` differentiates (AUTODIFF.md §2,
    /// #398): the module-level mutable bindings its body reads *directly*,
    /// minus its own parameter names (a param shadows the binding, so a read
    /// there is not a capture).
    ///
    /// Reads that occur only inside a closure literal are excluded: the tape
    /// never enters a closure body, so such a capture has no adjoint. The
    /// checker rejects that program outright (`check_fn`'s #398 rule) and both
    /// sides use the same scan, so what the interpreter tapes is exactly what
    /// the checker admitted. The related case — a capture read by a fn the body
    /// *calls*, whose gradient path the tape also cannot see — is enforced by
    /// the checker alone (it owns the call graph); the interpreter would
    /// otherwise hand back that capture's body-only half.
    ///
    /// Candidates are drawn from the fn's **own module** only. A `@grad fn` can
    /// capture nothing else: the checker does not put an imported module's
    /// file-private `let !` in scope under its bare name, so a cross-module
    /// name match was never a capture the checker admitted — only one the
    /// flat name list invented.
    ///
    /// Result order is the module's declaration order, which fixes the order
    /// the capture fields take in `Grads`.
    fn grad_capture_names(&self, f: &FnDecl) -> Vec<String> {
        if self.mut_globals.is_empty() {
            return Vec::new();
        }
        // The fn's home module, stamped on the decl when its file was loaded.
        let home = f.module;
        let params: std::collections::HashSet<&str> =
            f.params.iter().map(|p| p.name.as_str()).collect();
        let refs = crate::check::collect_body_idents(&f.body);
        self.mut_globals.iter()
            .filter(|(module, n)| {
                *module == home
                    && !params.contains(n.as_str())
                    // `reads_free` excludes names the body BINDS itself: a
                    // body-local `let cap` shadows the module binding, so there
                    // is no capture to tape and no `g.cap` to hand back.
                    && refs.reads_free(n)
                    && refs.direct.contains(n)
                    && !refs.in_closure.contains(n)
            })
            .map(|(_, n)| n.clone())
            .collect()
    }

    /// Evaluate a block while recording tensor ops to a tape. Returns the
    /// final value and (optionally) the tape node id that produced it.
    fn eval_block_with_tape(
        &mut self,
        block: &Block,
        tape: &mut Tape,
        var_node: &mut std::collections::HashMap<String, usize>,
        span: &Span,
    ) -> EvalResult<(Value, Option<usize>)> {
        self.push_scope();
        let result = (|| -> EvalResult<(Value, Option<usize>)> {
            let mut last: (Value, Option<usize>) = (Value::Nil, None);
            for stmt in &block.stmts {
                last = self.trace_stmt(stmt, tape, var_node, span)?;
            }
            if let Some(tail) = &block.tail_expr {
                last = self.eval_expr_with_tape(tail, tape, var_node, span)?;
            }
            Ok(last)
        })();
        self.pop_scope();
        result
    }

    /// Trace a single statement onto the tape (#368). Differentiable statement
    /// subset: `let`, expression statements, plain/compound reassignment of a
    /// named accumulator (`acc = …`, `acc += …`), and `while` / `for` loops.
    /// Loops are **unrolled** onto the tape by concrete execution — the forward
    /// pass runs for real, so trip counts are fixed at trace time and each
    /// iteration appends its ops (accumulators chain through `var_node`).
    /// `break` / `continue` / `return` and the unbounded `loop` are not wired
    /// (they'd need genuine control-flow nodes on the tape).
    fn trace_stmt(
        &mut self,
        stmt: &Stmt,
        tape: &mut Tape,
        var_node: &mut std::collections::HashMap<String, usize>,
        span: &Span,
    ) -> EvalResult<(Value, Option<usize>)> {
        match stmt {
            Stmt::Let(l) => {
                let (v, n) = self.eval_expr_with_tape(&l.value, tape, var_node, span)?;
                // Bind the simple-ident pattern only. Tuple destructuring,
                // shape patterns, etc. inside @grad are out of scope.
                if let Pattern::Ident(name, _) = &l.pattern {
                    if name != "_" {
                        self.bind(name, v.clone());
                        match n {
                            Some(node_id) => { var_node.insert(name.clone(), node_id); }
                            None => { var_node.remove(name); }
                        }
                    }
                } else if let Pattern::Wildcard(_) = &l.pattern {
                    // discard
                } else {
                    return Err(RuntimeError::at(
                        "@grad: only simple `let name = ...` bindings are supported \
                        in the differentiated body (got a more complex pattern)".to_string(),
                        span.clone()));
                }
                Ok((Value::Nil, None))
            }
            Stmt::Expr { lhs, assign: None, .. } => {
                self.eval_expr_with_tape(lhs, tape, var_node, span)
            }
            // Reassignment of a named accumulator threads `var_node` so the
            // gradient flows through the updated node (`acc += f(w)` etc.).
            Stmt::Expr { lhs: Expr::Ident(name, _), assign: Some((op, rhs)), .. } => {
                self.trace_reassign(name, op, rhs, tape, var_node, span)?;
                Ok((Value::Nil, None))
            }
            // Loops unroll: run concretely, tracing each iteration's body onto
            // the same tape / var_node so accumulators chain up.
            Stmt::While { cond, body, .. } => {
                loop {
                    let c = self.eval_expr(cond)?;
                    if !c.as_bool().unwrap_or(false) { break; }
                    self.eval_block_with_tape(body, tape, var_node, span)?;
                }
                Ok((Value::Nil, None))
            }
            Stmt::For { pattern, iter, body, .. } => {
                let iter_v = self.eval_expr(iter)?;
                let items = expand_iter(&iter_v)?;
                for item in items {
                    self.push_scope();
                    self.bind_pattern(pattern, item);  // loop index: concrete, non-differentiable
                    let r = self.eval_block_with_tape(body, tape, var_node, span);
                    self.pop_scope();
                    r?;
                }
                Ok((Value::Nil, None))
            }
            // A block-form `if`/`match` in tail position parses as a statement,
            // not the block's `tail_expr` — route it through the expression
            // tracers so it differentiates the same as the let-bound form (#421).
            Stmt::If(ie) => self.eval_if_with_tape(ie, tape, var_node, span),
            Stmt::Match(me) => self.eval_match_with_tape(me, tape, var_node, span),
            _ => {
                let kind = match stmt {
                    Stmt::Loop { .. } => "an unbounded `loop`",
                    Stmt::Stage { .. } => "a `@pp` stage",
                    Stmt::Directive { .. } | Stmt::DirectiveBlock { .. } => "a directive",
                    Stmt::Expr { assign: Some(_), .. } => "a non-identifier assignment target",
                    _ => "this statement form",
                };
                Err(RuntimeError::at(format!(
                    "@grad: {kind} is not supported in the differentiated body. \
                    Supported: let, reassignment of a named accumulator, while/for \
                    loops, and if/match expressions; break/continue/return and the \
                    unbounded `loop` are not wired."), span.clone()))
            }
        }
    }

    /// Trace `name (op)= rhs` onto the tape and rebind `name`'s value + node.
    /// Reuses `tape_binop` for the compound forms so the VJP mapping stays in
    /// one place. Only the differentiable ops (`=`, `+=`, `-=`, `*=`, `/=`) are
    /// supported; bitwise / stream compound-assign has no VJP.
    fn trace_reassign(
        &mut self,
        name: &str,
        op: &AssignOp,
        rhs: &Expr,
        tape: &mut Tape,
        var_node: &mut std::collections::HashMap<String, usize>,
        span: &Span,
    ) -> EvalResult<()> {
        let (new_val, new_node) = match op {
            AssignOp::Eq => self.eval_expr_with_tape(rhs, tape, var_node, span)?,
            AssignOp::PlusEq | AssignOp::MinusEq | AssignOp::StarEq | AssignOp::SlashEq => {
                let (rval, rnode) = self.eval_expr_with_tape(rhs, tape, var_node, span)?;
                let cur = self.lookup(name).unwrap_or(Value::Nil);
                // #674: `trace_reassign` reads the binding directly too.
                self.check_live(&cur, span.clone())?;
                let cur_node = var_node.get(name).copied();
                // Same scalar-vs-tensor dispatch as eval_stmt's compound assign.
                let is_tensor = matches!(cur, Value::Tensor(_)) || matches!(rval, Value::Tensor(_));
                let binop = match op {
                    AssignOp::PlusEq  => if is_tensor { BinOp::DotAdd } else { BinOp::Add },
                    AssignOp::MinusEq => if is_tensor { BinOp::DotSub } else { BinOp::Sub },
                    AssignOp::StarEq  => if is_tensor { BinOp::DotMul } else { BinOp::Mul },
                    AssignOp::SlashEq => if is_tensor { BinOp::DotDiv } else { BinOp::Div },
                    _ => unreachable!(),
                };
                self.tape_binop(tape, &binop, &cur, cur_node, &rval, rnode, span)?
            }
            _ => return Err(RuntimeError::at(format!(
                "@grad: compound assignment `{:?}` is not differentiable \
                (bitwise / stream ops have no VJP)", op), span.clone())),
        };
        if self.assign(name, new_val.clone()).is_err() {
            self.bind(name, new_val);
        }
        match new_node {
            Some(id) => { var_node.insert(name.to_string(), id); }
            None => { var_node.remove(name); }
        }
        Ok(())
    }

    /// True for the comparison operators — dotted (tensor mask) and scalar
    /// alike. Their output is a 0/1 indicator that is constant almost
    /// everywhere in its operands, so inside `@grad` they are stop-gradient
    /// (AUTODIFF.md §5) rather than an unsupported-op error (#434).
    fn is_comparison(op: &BinOp) -> bool {
        matches!(op,
            BinOp::DotGt | BinOp::DotLt | BinOp::DotGe | BinOp::DotLe
            | BinOp::Gt | BinOp::Lt | BinOp::GtEq | BinOp::LtEq
            | BinOp::Eq | BinOp::NotEq)
    }

    /// Push a binary-op node onto the tape, reusing one op→VJP mapping for both
    /// the `BinOp` expression arm and compound reassignment. Returns the forward
    /// value and its node id (None when neither operand is tracked).
    fn tape_binop(
        &mut self,
        tape: &mut Tape,
        op: &BinOp,
        lv: &Value, ln: Option<usize>,
        rv: &Value, rn: Option<usize>,
        span: &Span,
    ) -> EvalResult<(Value, Option<usize>)> {
        let out = apply_binop(op.clone(), lv, rv)?;
        // #791: one of the two places the tape evaluator materializes a
        // tensor without going through `eval_expr` / `call_builtin` (the
        // other is `UnOp` in `eval_expr_with_tape`). Every other traced
        // form — the reductions, the activations, `softmax`, `rms_norm`,
        // `layer_norm`, `rope`, `attn`, `reshape`, the transpose — computes
        // its forward value by calling one of those two, so it is already
        // metered at the choke point they share (`charge_intermediate`,
        // `charge_builtin_result`); only `apply_binop`/`apply_unop` bypass
        // it. Charged unconditionally, before the two untracked
        // early-returns below: a value that leaves the gradient graph still
        // occupies the bytes.
        self.charge_intermediate(&out, Some(span))?;
        if ln.is_none() && rn.is_none() {
            return Ok((out, None));
        }
        // Comparisons are stop-gradient (AUTODIFF.md §5): the 0/1 result is
        // locally constant in its operands, so it leaves the tape as a value
        // rather than erroring (#434). That is what makes comparison-masked
        // select — `(a .< b) .* a .+ (a .>= b) .* b` — differentiable: the
        // mask multiplies in as a constant and `DotMul`'s product rule routes
        // the whole cotangent to whichever branch the mask selected, exactly
        // the treatment `\>`'s ReLU mask already gets.
        if Self::is_comparison(op) {
            return Ok((out, None));
        }
        let tape_op = match op {
            BinOp::Matmul => Some(TapeOp::Matmul),
            BinOp::DotAdd => Some(TapeOp::DotAdd),
            BinOp::DotSub => Some(TapeOp::DotSub),
            BinOp::DotMul => Some(TapeOp::DotMul),
            BinOp::DotDiv => Some(TapeOp::DotDiv),
            BinOp::Div    => Some(TapeOp::ScalarDiv),
            BinOp::Mul    => Some(TapeOp::ScalarMul),
            BinOp::Add | BinOp::Sub => Some(TapeOp::ScalarAddSub(op.clone())),
            _ => None,
        };
        match tape_op {
            Some(top) => {
                let lnid = ln.unwrap_or_else(|| tape.push_const(lv.clone()));
                let rnid = rn.unwrap_or_else(|| tape.push_const(rv.clone()));
                let id = tape.push(TapeNode { op: top, inputs: vec![lnid, rnid], value: out.clone() });
                Ok((out, Some(id)))
            }
            None => Err(RuntimeError::at(format!(
                "@grad: BinOp `{:?}` not supported inside differentiated body \
                (no VJP rule yet).", op), span.clone())),
        }
    }

    /// Trace a fully-qualified element read `base[i, j, ...]` onto the tape
    /// (#420 item 3). The gradient of an element read scatters: the adjoint
    /// lands on element `[i, j, ...]` of a zero tensor shaped like `base`,
    /// which is what lets vector component math (`w[0]*w[0] + w[1]*w[1]`, the
    /// natural spelling of an SDF) differentiate without the outer-product
    /// broadcast workaround.
    ///
    /// Returns `None` — leaving the caller's untracked path in charge — when
    /// this isn't an element read of a traced float tensor: a bracket form
    /// that isn't indexing at all (arena constructors, `.split`, `.reshape`),
    /// a non-constant or non-integer subscript, a partial index (which yields
    /// a sub-tensor, not an element), or an int/bool tensor. Indices
    /// themselves are always concrete — they select *which* element, they are
    /// not differentiable.
    ///
    /// Bailing out after the base has been traced re-evaluates it in the
    /// caller and leaves an unreferenced node on the tape. Both are inert: the
    /// admitted base forms (ident / arithmetic, never a call) are pure, and a
    /// node no adjoint reaches contributes nothing to the backward pass.
    fn tape_index_read(
        &mut self,
        base: &Expr,
        elems: &[IndexElem],
        tape: &mut Tape,
        var_node: &mut std::collections::HashMap<String, usize>,
        span: &Span,
    ) -> EvalResult<Option<(Value, Option<usize>)>> {
        // Only bracket forms that are genuinely subscripts. A `Field`/`Postfix`
        // base is `vault.zeros[...]`, `t.split[n]`, `t.reshape[[..]]` and
        // friends — tracing those bases would re-evaluate a constructor.
        if !matches!(base, Expr::Ident(_, _) | Expr::BinOp { .. } | Expr::UnOp { .. } | Expr::Tuple(_, _)) {
            return Ok(None);
        }
        if elems.is_empty() {
            return Ok(None);
        }
        let mut raw_idx: Vec<i64> = Vec::with_capacity(elems.len());
        for elem in elems {
            let IndexElem::Expr(ie) = elem else { return Ok(None) };
            match self.eval_expr(ie)? {
                Value::Int(n, _) => raw_idx.push(n),
                _ => return Ok(None),
            }
        }
        let (bv, bn) = self.eval_expr_with_tape(base, tape, var_node, span)?;
        let Value::Tensor(arr) = &bv else { return Ok(None) };
        if !arr.dtype.rounds_as_f32() && arr.dtype != DType::F64 {
            return Ok(None);
        }
        if raw_idx.len() != arr.ndim() {
            return Ok(None);      // partial index → sub-tensor; not wired
        }
        let mut idx: Vec<usize> = Vec::with_capacity(raw_idx.len());
        for (axis, &n) in raw_idx.iter().enumerate() {
            let dim = arr.shape()[axis] as i64;
            let resolved = if n < 0 { dim + n } else { n };
            if resolved < 0 || resolved >= dim {
                return Err(RuntimeError::at(
                    format!("index {} out of bounds for axis {} of size {}", n, axis, dim),
                    span.clone()));
            }
            idx.push(resolved as usize);
        }
        let out = Value::Float(arr[IxDyn(&idx)], arr.float_width());
        let Some(nid) = bn else { return Ok(Some((out, None))) };
        let id = tape.push(TapeNode {
            op: TapeOp::IndexRead(idx), inputs: vec![nid], value: out.clone(),
        });
        Ok(Some((out, Some(id))))
    }

    /// Evaluate an expression while building the tape. Returns (value,
    /// node_id) where node_id is Some iff the value is a tracked tensor or
    /// a scalar derived from one (which can be the loss).
    fn eval_expr_with_tape(
        &mut self,
        e: &Expr,
        tape: &mut Tape,
        var_node: &mut std::collections::HashMap<String, usize>,
        span: &Span,
    ) -> EvalResult<(Value, Option<usize>)> {
        use Expr::*;
        match e {
            Ident(name, _) => {
                let v = self.eval_expr(e)?;
                let n = var_node.get(name).copied();
                Ok((v, n))
            }
            Literal(_, _) | Nil(_) => Ok((self.eval_expr(e)?, None)),
            Cast { .. } => Ok((self.eval_expr(e)?, None)),
            // Parenthesized expressions parse as 1-element tuples — unwrap them
            // transparently so we don't lose tape tracking through `(...)`.
            Tuple(elems, _) if elems.len() == 1 => {
                self.eval_expr_with_tape(&elems[0], tape, var_node, span)
            }
            BinOp { op, lhs, rhs, .. } => {
                let (lv, ln) = self.eval_expr_with_tape(lhs, tape, var_node, span)?;
                let (rv, rn) = self.eval_expr_with_tape(rhs, tape, var_node, span)?;
                self.tape_binop(tape, op, &lv, ln, &rv, rn, span)
            }
            UnOp { op, operand, .. } => {
                let (v, n) = self.eval_expr_with_tape(operand, tape, var_node, span)?;
                let out = apply_unop(op.clone(), &v)?;
                // #791: the second of the tape evaluator's two direct
                // materializations — see `tape_binop`'s comment.
                self.charge_intermediate(&out, Some(span))?;
                let track = n.is_some();
                let tape_op = match op {
                    crate::ast::UnOp::ReLU   => Some(TapeOp::ReLU),
                    crate::ast::UnOp::Neg    => Some(TapeOp::Negate),
                    crate::ast::UnOp::BitNot => None,  // not differentiable
                    _ => None,
                };
                if track {
                    if let Some(top) = tape_op {
                        let nid = n.unwrap();
                        let id = tape.push(TapeNode { op: top, inputs: vec![nid], value: out.clone() });
                        return Ok((out, Some(id)));
                    } else {
                        return Err(RuntimeError::at(
                            format!("@grad: UnOp `{:?}` not supported inside differentiated body (no VJP rule yet).", op),
                            span.clone()));
                    }
                }
                Ok((out, None))
            }
            Postfix { expr, op, span: psp } => match op {
                PostfixOp::Transpose => {
                    let (v, n) = self.eval_expr_with_tape(expr, tape, var_node, span)?;
                    let out = self.eval_expr(e)?;     // re-eval is cheap for the transpose-view
                    let _ = v;                          // silence unused
                    if let Some(nid) = n {
                        let id = tape.push(TapeNode { op: TapeOp::Transpose, inputs: vec![nid], value: out.clone() });
                        return Ok((out, Some(id)));
                    }
                    Ok((out, None))
                }
                PostfixOp::Call(args) => {
                    // `sum(...)` and `mean(...)` are the tracked reductions.
                    // `mean(x)` is `sum(x) / N`, taped as a Sum node followed by
                    // a ScalarDiv by the element count (#253) — its VJP then
                    // falls out of the existing rules. Any other call inside
                    // @grad falls back to opaque eval — fine if the result isn't
                    // part of the gradient chain; an error if it is.
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if (fname == "sum" || fname == "mean") && args.len() == 1 {
                            if let CallArg::Positional(arg_expr) = &args[0] {
                                let (v, n) = self.eval_expr_with_tape(arg_expr, tape, var_node, span)?;
                                let out = self.call_builtin(fname, vec![v.clone()], psp.clone())?;
                                if let Some(nid) = n {
                                    // The raw Sum value (mean's numerator).
                                    let sum_val = Value::Float(as_tensor(&v)?.iter().sum(), FW::F64);
                                    let sum_id = tape.push(TapeNode {
                                        op: TapeOp::Sum, inputs: vec![nid],
                                        value: if fname == "sum" { out.clone() } else { sum_val },
                                    });
                                    if fname == "mean" {
                                        let count = as_tensor(&v)?.len() as f64;
                                        let cnode = tape.push_const(Value::Float(count, FW::F64));
                                        let mean_id = tape.push(TapeNode {
                                            op: TapeOp::ScalarDiv,
                                            inputs: vec![sum_id, cnode], value: out.clone(),
                                        });
                                        return Ok((out, Some(mean_id)));
                                    }
                                    return Ok((out, Some(sum_id)));
                                }
                                return Ok((out, None));
                            }
                        }
                    }
                    // The axis reductions (axis dropped) differentiate through
                    // the tape: sum_along / mean_along broadcast the adjoint
                    // back along the axis (#307); max_along / min_along route
                    // it to each lane's extreme element (#434).
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        let along: Option<fn(usize) -> TapeOp> = match fname.as_str() {
                            "sum_along"  => Some(TapeOp::SumAlong),
                            "mean_along" => Some(TapeOp::MeanAlong),
                            "max_along"  => Some(TapeOp::MaxAlong),
                            "min_along"  => Some(TapeOp::MinAlong),
                            _ => None,
                        };
                        if let (Some(mk), 2) = (along, args.len()) {
                            if let (CallArg::Positional(xe), CallArg::Positional(ae)) = (&args[0], &args[1]) {
                                let (v, n) = self.eval_expr_with_tape(xe, tape, var_node, span)?;
                                let axis_lit = self.eval_expr(ae)?.as_int().unwrap_or(0);
                                let out = self.call_builtin(fname, vec![v.clone(), Value::Int(axis_lit, IW::I64)], psp.clone())?;
                                if let Some(nid) = n {
                                    let ndim = as_tensor(&v)?.ndim();
                                    let axis = normalize_axis(axis_lit, ndim, span)?;
                                    let id = tape.push(TapeNode { op: mk(axis), inputs: vec![nid], value: out.clone() });
                                    return Ok((out, Some(id)));
                                }
                                return Ok((out, None));
                            }
                        }
                    }
                    // softmax(x, axis?) differentiates through the tape (#307):
                    // VJP `dx = y .* (g - rowsum(g .* y, axis))`. The axis is
                    // normalized and stored on the node.
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if fname == "softmax" && (args.len() == 1 || args.len() == 2) {
                            if let CallArg::Positional(arg_expr) = &args[0] {
                                let (v, n) = self.eval_expr_with_tape(arg_expr, tape, var_node, span)?;
                                let axis_lit = if args.len() == 2 {
                                    match &args[1] {
                                        CallArg::Positional(ae) => self.eval_expr(ae)?.as_int().unwrap_or(-1),
                                        _ => -1,
                                    }
                                } else { -1 };
                                let out = self.call_builtin(fname, vec![v.clone(), Value::Int(axis_lit, IW::I64)], psp.clone())?;
                                if let Some(nid) = n {
                                    let ndim = as_tensor(&v)?.ndim();
                                    let axis = normalize_axis(axis_lit, ndim, span)?;
                                    let id = tape.push(TapeNode { op: TapeOp::Softmax(axis), inputs: vec![nid], value: out.clone() });
                                    return Ok((out, Some(id)));
                                }
                                return Ok((out, None));
                            }
                        }
                    }
                    // rms_norm(x, gain, eps?) differentiates through the tape
                    // (#307): inputs [x, gain], eps carried on the node.
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if fname == "rms_norm" && (args.len() == 2 || args.len() == 3) {
                            if let (CallArg::Positional(xe), CallArg::Positional(ge)) = (&args[0], &args[1]) {
                                let (xv, xn) = self.eval_expr_with_tape(xe, tape, var_node, span)?;
                                let (gv, gn) = self.eval_expr_with_tape(ge, tape, var_node, span)?;
                                let eps = if args.len() == 3 {
                                    match &args[2] {
                                        CallArg::Positional(ee) => self.eval_expr(ee)?.as_float().unwrap_or(1e-6),
                                        _ => 1e-6,
                                    }
                                } else { 1e-6 };
                                let out = self.call_builtin(fname, vec![xv.clone(), gv.clone(), Value::Float(eps, FW::F64)], psp.clone())?;
                                if xn.is_some() || gn.is_some() {
                                    let xnid = xn.unwrap_or_else(|| tape.push_const(xv.clone()));
                                    let gnid = gn.unwrap_or_else(|| tape.push_const(gv.clone()));
                                    let id = tape.push(TapeNode { op: TapeOp::RmsNorm(eps), inputs: vec![xnid, gnid], value: out.clone() });
                                    return Ok((out, Some(id)));
                                }
                                return Ok((out, None));
                            }
                        }
                    }
                    // layer_norm(x, gain, bias, eps?) differentiates through the
                    // tape (#307): inputs [x, gain, bias], eps carried on the node.
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if fname == "layer_norm" && (args.len() == 3 || args.len() == 4) {
                            if let (CallArg::Positional(xe), CallArg::Positional(ge), CallArg::Positional(be))
                                = (&args[0], &args[1], &args[2]) {
                                let (xv, xn) = self.eval_expr_with_tape(xe, tape, var_node, span)?;
                                let (gv, gn) = self.eval_expr_with_tape(ge, tape, var_node, span)?;
                                let (bv, bn) = self.eval_expr_with_tape(be, tape, var_node, span)?;
                                let eps = if args.len() == 4 {
                                    match &args[3] {
                                        CallArg::Positional(ee) => self.eval_expr(ee)?.as_float().unwrap_or(1e-5),
                                        _ => 1e-5,
                                    }
                                } else { 1e-5 };
                                let out = self.call_builtin(fname, vec![xv.clone(), gv.clone(), bv.clone(), Value::Float(eps, FW::F64)], psp.clone())?;
                                if xn.is_some() || gn.is_some() || bn.is_some() {
                                    let xnid = xn.unwrap_or_else(|| tape.push_const(xv.clone()));
                                    let gnid = gn.unwrap_or_else(|| tape.push_const(gv.clone()));
                                    let bnid = bn.unwrap_or_else(|| tape.push_const(bv.clone()));
                                    let id = tape.push(TapeNode { op: TapeOp::LayerNorm(eps), inputs: vec![xnid, gnid, bnid], value: out.clone() });
                                    return Ok((out, Some(id)));
                                }
                                return Ok((out, None));
                            }
                        }
                    }
                    // rope(x, cos, sin) differentiates through the tape (#368):
                    // RoPE is a per-pair orthogonal rotation, so the VJP is the
                    // inverse rotation `dx = rope(g, cos, -sin)`. cos/sin are
                    // read-only position tables — recorded as const nodes; the
                    // gradient flows only to x.
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if fname == "rope" && args.len() == 3 {
                            if let (CallArg::Positional(xe), CallArg::Positional(ce), CallArg::Positional(se))
                                = (&args[0], &args[1], &args[2]) {
                                let (xv, xn) = self.eval_expr_with_tape(xe, tape, var_node, span)?;
                                let cv = self.eval_expr(ce)?;
                                let sv = self.eval_expr(se)?;
                                let out = self.call_builtin("rope", vec![xv.clone(), cv.clone(), sv.clone()], psp.clone())?;
                                if let Some(xnid) = xn {
                                    let cnid = tape.push_const(cv);
                                    let snid = tape.push_const(sv);
                                    let id = tape.push(TapeNode { op: TapeOp::Rope, inputs: vec![xnid, cnid, snid], value: out.clone() });
                                    return Ok((out, Some(id)));
                                }
                                return Ok((out, None));
                            }
                        }
                    }
                    // attn(q, k, v, mask?) / attn_gqa(...) differentiate through
                    // the tape (#368): fused scaled-dot-product attention. The
                    // VJP flows to q, k, and v; the optional mask is a read-only
                    // const node (masked positions have softmax weight 0, so no
                    // gradient crosses them).
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if (fname == "attn" || fname == "attn_gqa")
                            && (args.len() == 3 || args.len() == 4)
                            && args.iter().all(|a| matches!(a, CallArg::Positional(_)))
                        {
                            let pos = |a: &CallArg| match a {
                                CallArg::Positional(e) => e.clone(),
                                _ => unreachable!(),
                            };
                            let (qe, ke, ve) = (pos(&args[0]), pos(&args[1]), pos(&args[2]));
                            let (qv, qn) = self.eval_expr_with_tape(&qe, tape, var_node, span)?;
                            let (kv, kn) = self.eval_expr_with_tape(&ke, tape, var_node, span)?;
                            let (vv, vn) = self.eval_expr_with_tape(&ve, tape, var_node, span)?;
                            let mask = if args.len() == 4 {
                                // `nil` means no mask — same as the 3-arg form.
                                match self.eval_expr(&pos(&args[3]))? {
                                    Value::Nil => None,
                                    m => Some(m),
                                }
                            } else { None };
                            let mut call_args = vec![qv.clone(), kv.clone(), vv.clone()];
                            if let Some(m) = &mask { call_args.push(m.clone()); }
                            let out = self.call_builtin(fname, call_args, psp.clone())?;
                            if qn.is_some() || kn.is_some() || vn.is_some() {
                                let qnid = qn.unwrap_or_else(|| tape.push_const(qv.clone()));
                                let knid = kn.unwrap_or_else(|| tape.push_const(kv.clone()));
                                let vnid = vn.unwrap_or_else(|| tape.push_const(vv.clone()));
                                let mut inputs = vec![qnid, knid, vnid];
                                if let Some(m) = mask { inputs.push(tape.push_const(m)); }
                                let id = tape.push(TapeNode { op: TapeOp::Attn, inputs, value: out.clone() });
                                return Ok((out, Some(id)));
                            }
                            return Ok((out, None));
                        }
                    }
                    // Elementwise activation builtins differentiate through the
                    // tape (#306): `relu` reuses the ReLU VJP; sigmoid/tanh/gelu/
                    // silu get an Activation node (smooth, derivative from input).
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        let act_op = if fname == "relu" {
                            Some(TapeOp::ReLU)
                        } else {
                            ActKind::from_name(fname).map(TapeOp::Activation)
                        };
                        if let Some(top) = act_op {
                            if args.len() == 1 {
                                if let CallArg::Positional(arg_expr) = &args[0] {
                                    let (v, n) = self.eval_expr_with_tape(arg_expr, tape, var_node, span)?;
                                    let out = self.call_builtin(fname, vec![v.clone()], psp.clone())?;
                                    if let Some(nid) = n {
                                        let id = tape.push(TapeNode { op: top, inputs: vec![nid], value: out.clone() });
                                        return Ok((out, Some(id)));
                                    }
                                    return Ok((out, None));
                                }
                            }
                        }
                    }
                    // Scalar-math builtins on a traced scalar stay on the tape
                    // (#420): sqrt/exp/log/sin/cos/tan previously left the
                    // graph, making every Euclidean-norm-shaped SDF loss
                    // untraceable. Scalars only — these builtins don't
                    // broadcast over tensors, and a tensor arg errors in the
                    // forward call itself.
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if let Some(kind) = ScalarMathKind::from_name(fname) {
                            if args.len() == 1 {
                                if let CallArg::Positional(arg_expr) = &args[0] {
                                    let (v, n) = self.eval_expr_with_tape(arg_expr, tape, var_node, span)?;
                                    let out = self.call_builtin(fname, vec![v.clone()], psp.clone())?;
                                    if let Some(nid) = n {
                                        let id = tape.push(TapeNode {
                                            op: TapeOp::ScalarMath(kind), inputs: vec![nid], value: out.clone(),
                                        });
                                        return Ok((out, Some(id)));
                                    }
                                    return Ok((out, None));
                                }
                            }
                        }
                    }
                    // Global reductions variance / max / min differentiate
                    // through the tape (#307, Tier C). Each takes a single tensor
                    // arg and returns a scalar; max/min stay variadic-scalar when
                    // called with >1 arg (those don't reach this single-arg arm).
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        let red_op = match fname.as_str() {
                            "variance" => Some(TapeOp::Variance),
                            "max" => Some(TapeOp::MaxReduce),
                            "min" => Some(TapeOp::MinReduce),
                            _ => None,
                        };
                        if let Some(top) = red_op {
                            if args.len() == 1 {
                                if let CallArg::Positional(arg_expr) = &args[0] {
                                    let (v, n) = self.eval_expr_with_tape(arg_expr, tape, var_node, span)?;
                                    // Only a tensor arg has a per-element gradient;
                                    // a scalar max/min is just an identity passthrough.
                                    if matches!(v, Value::Tensor(_)) {
                                        let out = self.call_builtin(fname, vec![v.clone()], psp.clone())?;
                                        if let Some(nid) = n {
                                            let id = tape.push(TapeNode { op: top, inputs: vec![nid], value: out.clone() });
                                            return Ok((out, Some(id)));
                                        }
                                        return Ok((out, None));
                                    }
                                }
                            }
                        }
                    }
                    // Anything else: evaluate normally, untracked.
                    let v = self.eval_expr(e)?;
                    Ok((v, None))
                }
                PostfixOp::Index(elems) => {
                    // `x.reshape[[..]]` parses as Index over a `.reshape` Field.
                    // Trace it so the adjoint reshapes back to x's shape (#307).
                    if let Postfix { expr: recv, op: PostfixOp::Field(method), .. } = expr.as_ref() {
                        if method == "reshape" {
                            let (_rv, rn) = self.eval_expr_with_tape(recv, tape, var_node, span)?;
                            let out = self.eval_expr(e)?;
                            if let Some(nid) = rn {
                                let id = tape.push(TapeNode { op: TapeOp::Reshape, inputs: vec![nid], value: out.clone() });
                                return Ok((out, Some(id)));
                            }
                            return Ok((out, None));
                        }
                    }
                    // `w[i]` / `w[i, j]` on a traced tensor stays on the tape
                    // (#420 item 3): the adjoint scatters back to element `i`.
                    if let Some(hit) = self.tape_index_read(expr, elems, tape, var_node, span)? {
                        return Ok(hit);
                    }
                    Ok((self.eval_expr(e)?, None))
                }
                _ => Ok((self.eval_expr(e)?, None)),
            },
            // Control flow as an expression (#368). The interpreter runs
            // concretely, so a data-dependent branch is differentiable by
            // define-by-run: the condition/scrutinee is non-differentiable
            // (evaluated for real) and only the *taken* branch is traced onto
            // the tape. The gradient flows through the executed path exactly as
            // in eager frameworks; the untaken branch contributes nothing.
            If(ie) => self.eval_if_with_tape(ie, tape, var_node, span),
            Match(me) => self.eval_match_with_tape(me, tape, var_node, span),
            _ => Ok((self.eval_expr(e)?, None)),
        }
    }

    /// Trace a `match` expression onto the tape. Like `if`, the scrutinee,
    /// patterns, and guards are non-differentiable (evaluated concretely);
    /// only the taken arm's body is recorded (define-by-run).
    fn eval_match_with_tape(
        &mut self,
        me: &MatchExpr,
        tape: &mut Tape,
        var_node: &mut std::collections::HashMap<String, usize>,
        span: &Span,
    ) -> EvalResult<(Value, Option<usize>)> {
        let scrutinee = self.eval_expr(&me.scrutinee)?;
        for arm in &me.arms {
            if self.pattern_matches(&arm.pattern, &scrutinee) {
                self.push_scope();
                self.bind_pattern(&arm.pattern, scrutinee.clone());
                let guard_ok = match &arm.guard {
                    Some(g) => self.eval_expr(g)?.as_bool().unwrap_or(false),
                    None => true,
                };
                if guard_ok {
                    let out = self.eval_expr_with_tape(&arm.body, tape, var_node, span);
                    self.pop_scope();
                    return out;
                }
                self.pop_scope();
            }
        }
        Err(RuntimeError::at(format!(
            "@grad match: no arm matched scrutinee {:?}", scrutinee), span.clone()))
    }

    /// Trace an `if` expression onto the tape (#368). The condition is
    /// non-differentiable (evaluated concretely); only the taken block is
    /// recorded, so the gradient follows the executed path (define-by-run).
    /// `else if` chains recurse; a bare `if` with no `else` yields Nil.
    fn eval_if_with_tape(
        &mut self,
        ie: &IfExpr,
        tape: &mut Tape,
        var_node: &mut std::collections::HashMap<String, usize>,
        span: &Span,
    ) -> EvalResult<(Value, Option<usize>)> {
        let c = self.eval_expr(&ie.cond)?;
        if c.as_bool().unwrap_or(false) {
            self.eval_block_with_tape(&ie.then_branch, tape, var_node, span)
        } else {
            match &ie.else_branch {
                Some(ElseBranch::Block(b)) => self.eval_block_with_tape(b, tape, var_node, span),
                Some(ElseBranch::If(nested)) => self.eval_if_with_tape(nested, tape, var_node, span),
                None => Ok((Value::Nil, None)),
            }
        }
    }

    // ── Match ─────────────────────────────────────────────────────────────

    /// Returns true if `val` structurally matches `pat` (no side effects).
    fn pattern_matches(&self, pat: &Pattern, val: &Value) -> bool {
        match pat {
            Pattern::Wildcard(_) => true,
            // `..` standalone is a catch-all (matches anything); inside a tuple it
            // is handled by the Tuple arm below via `tuple_rest_split`.
            Pattern::Rest(_) => true,
            // `.variant` patterns (hardware/ISA feature flags): check against the set
            // detected at interpreter startup.  Plain (non-dotted) ident patterns always match.
            Pattern::Ident(name, _) => {
                if let Some(feature) = name.strip_prefix('.') {
                    self.host_features.contains(feature)
                } else if let Value::EnumVal { enum_name, variant, .. } = val {
                    // #336: a bare ident naming a variant of the scrutinee's enum
                    // is a *variant* match; any other ident is a catch-all bind.
                    if self.enum_variant_ordinal(enum_name, name).is_some() {
                        name == variant
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
            // #336/#350: `Color.Red` / `Circle(r)` — match an enum value by
            // variant. An empty pattern `enum_name` is the bare payload form,
            // resolved against the scrutinee's own enum (the checker validated
            // the variant belongs to it). Bindings don't affect matching.
            Pattern::EnumVariant { enum_name, variant, .. } => {
                matches!(val, Value::EnumVal { enum_name: vn, variant: vv, .. }
                    if vv == variant && (enum_name.is_empty() || vn == enum_name))
            }
            Pattern::Literal(lit, _) => {
                let lv = lit_value(lit);
                match (&lv, val) {
                    (Value::Int(a, _),   Value::Int(b, _))   => a == b,
                    (Value::Float(a, _), Value::Float(b, _)) => a == b,
                    // Cross-match Int/Float like `==` (scalar_compare) and
                    // `list_contains` already do (#291.2): a `0` literal pattern
                    // matches a `0.0` scrutinee, so match agrees with equality.
                    (Value::Int(a, _),   Value::Float(b, _)) => (*a as f64) == *b,
                    (Value::Float(a, _), Value::Int(b, _))   => *a == (*b as f64),
                    (Value::Bool(a),  Value::Bool(b))  => a == b,
                    (Value::Str(a),   Value::Str(b))   => a == b,
                    (Value::Nil,      Value::Nil)       => true,
                    _ => false,
                }
            }
            Pattern::Tuple(pats, _) => {
                if let Value::Tuple(vals) = val {
                    let (before, after, has_rest) = crate::ast::tuple_rest_split(pats);
                    if has_rest {
                        // `(a, .., z)` matches any tuple long enough to cover the
                        // fixed head and tail; the rest absorbs the middle.
                        vals.len() >= before.len() + after.len()
                            && before.iter().zip(vals.iter())
                                .all(|(p, v)| self.pattern_matches(p, v))
                            && after.iter().zip(vals[vals.len() - after.len()..].iter())
                                .all(|(p, v)| self.pattern_matches(p, v))
                    } else {
                        pats.len() == vals.len()
                            && pats.iter().zip(vals).all(|(p, v)| self.pattern_matches(p, v))
                    }
                } else {
                    false
                }
            }
            // #393: `x @ pat` matches iff the sub-pattern matches; the binder is
            // a name, handled in `bind_pattern`. So test the sub-pattern here.
            Pattern::Bind(_binder, sub, _) => self.pattern_matches(sub, val),
            // Shape patterns are still unmatched (rejected at check time, #393).
            Pattern::Shape(_, _) => false,
        }
    }

    /// #336: the i64 ordinal of `variant` within `enum_name` **as the running
    /// file spells it**, or None if the file cannot see such an enum or the
    /// enum has no such variant. This is the resolution point for every enum
    /// name, so `resolve_fn` / `resolve_model`'s rule reaches enums here: a
    /// name the file neither declares nor imported is not an enum to it.
    fn enum_variant_ordinal(&self, enum_name: &str, variant: &str) -> Option<i64> {
        self.enums.get(&module_key(self.cur_module, enum_name))
            .and_then(|vs| vs.iter().position(|v| v == variant))
            .map(|p| p as i64)
    }

    /// #478's `if_branch_hint` for a `match`: what an unsuffixed float literal
    /// in arm `taken` adopts, read off the other arms.
    fn match_arm_hint(&self, me: &MatchExpr, taken: usize, inherited: Option<FW>) -> Option<FW> {
        if !has_untyped_float_lit(&me.arms[taken].body) { return None; }
        let other = me.arms.iter().enumerate()
            .filter(|(i, _)| *i != taken)
            .find_map(|(_, a)| self.static_float_width(&a.body));
        if other == Some(FW::F32) { Some(FW::F32) } else { inherited.filter(|w| *w == FW::F32) }
    }

    fn eval_match(&mut self, me: &MatchExpr) -> EvalResult<Value> {
        // #478: the hint belongs to the arm body, never to the scrutinee.
        let inherited = self.float_lit_hint.take();
        let scrutinee = self.eval_expr(&me.scrutinee)?;
        for (idx, arm) in me.arms.iter().enumerate() {
            if self.pattern_matches(&arm.pattern, &scrutinee) {
                self.push_scope();
                self.bind_pattern(&arm.pattern, scrutinee.clone());
                // #639: the arm's scope is popped on EVERY exit, error paths
                // included. A `return` / `break` / `continue` that leaves a
                // block inside a guard or an arm body travels as an `Err`
                // carrier and is caught one frame up, so an early `?` here
                // would leave the arm's scope on the stack and desynchronise
                // every later `pop_scope`.
                let guard_ok = match &arm.guard {
                    Some(guard) => {
                        let g = self.eval_expr(guard);
                        match g {
                            Ok(v) => v.as_bool().unwrap_or(false),
                            Err(e) => { self.pop_scope(); return Err(e); }
                        }
                    }
                    None => true,
                };
                if guard_ok {
                    self.float_lit_hint = self.match_arm_hint(me, idx, inherited);
                    let val = self.eval_expr(&arm.body);
                    self.pop_scope();
                    return val;
                }
                self.pop_scope();
            }
        }
        // Spec §4.5: match must be exhaustive; no-match is a runtime panic.
        Err(RuntimeError::msg(format!(
            "match: no arm matched scrutinee {:?}",
            scrutinee
        )))
    }

    // ── Function call ─────────────────────────────────────────────────────

    fn call_fn(&mut self, f: &FnDecl, args: Vec<Value>, sp: Span) -> EvalResult<Value> {
        self.call_fn_with_shapes(f, args, &[], sp)
    }

    /// Bind a shape-argument bracket — `f[N]`, `f[S=4]`, `b.blit![H, W]` —
    /// against a declaration's shape parameters. Positional args bind in
    /// declaration order, named args by name. Every failure is an error here
    /// rather than a fallthrough: an opaque stand-in silently poisons the
    /// trailing call (the `<opaque bracket(...)>` mode, and #474's ghost
    /// method). Free functions and methods share it so both report alike.
    fn bind_shape_bracket(
        &mut self,
        owner: &str,
        shape_params: &[ShapeParam],
        bracket: &[(Option<&str>, &Expr, Span)],
        span: &Span,
    ) -> EvalResult<Vec<(String, i64)>> {
        let mut bindings: Vec<(String, i64)> = Vec::new();
        let mut pos_idx = 0usize;
        for (arg_name, value_expr, aspan) in bracket {
            let sp_name = match arg_name {
                None => {
                    let Some(sp) = shape_params.get(pos_idx) else {
                        return Err(RuntimeError::at(format!(
                            "`{}` declares {} shape parameter(s), got more bracket args",
                            owner, shape_params.len()), span.clone()));
                    };
                    pos_idx += 1;
                    sp.name.clone()
                }
                Some(n) => {
                    if !shape_params.iter().any(|sp| sp.name == *n) {
                        return Err(RuntimeError::at(format!(
                            "`{}` is not a shape parameter of `{}` (declared: {})",
                            n, owner,
                            shape_params.iter().map(|sp| sp.name.as_str())
                                .collect::<Vec<_>>().join(", ")), aspan.clone()));
                    }
                    (*n).to_string()
                }
            };
            if bindings.iter().any(|(b, _)| *b == sp_name) {
                return Err(RuntimeError::at(format!(
                    "shape parameter `{}` of `{}` bound twice", sp_name, owner), aspan.clone()));
            }
            let v = self.eval_expr(value_expr)?;
            let Some(n) = v.as_int() else {
                return Err(RuntimeError::at(format!(
                    "shape argument `{}` of `{}` must be an integer, got {}",
                    sp_name, owner, v.type_name()), aspan.clone()));
            };
            bindings.push((sp_name, n));
        }
        Ok(bindings)
    }

    /// #474: dispatch `recv.method![shape args](args)`.
    ///
    /// Deciding whether this *is* a method call needs the receiver's value, so
    /// the receiver is evaluated here — exactly once, on every path. When the
    /// answer turns out to be no (the receiver is not a model instance, or the
    /// name is one of its *fields* being indexed, `b.handlers[i](x)`), the
    /// value comes back out with `NotAMethod` so the caller can hand it to the
    /// ordinary postfix path instead of running the expression again. When the
    /// receiver is a model instance and the name is neither a method nor a
    /// field, this is #441's undefined method reached through a bracket, and it
    /// errors here instead of evaluating to a silent opaque.
    fn call_bracketed_method(
        &mut self,
        recv_expr: &Expr,
        method: &str,
        bracket: &[(Option<&str>, &Expr, Span)],
        arg_vals: &[Value],
        span: &Span,
    ) -> EvalResult<BracketedCall> {
        let recv = self.eval_expr(recv_expr)?;
        let Value::Struct(ref fields) = recv else { return Ok(BracketedCall::NotAMethod(recv)) };
        let Some(mname) = struct_model_name(fields) else {
            return Ok(BracketedCall::NotAMethod(recv));
        };
        let Some(f) = self.model_methods.get(&mname)
            .and_then(|ms| ms.get(method)).cloned()
        else {
            // A field of that name may hold an indexable value; leave it to
            // the ordinary path. Otherwise there is nothing to call (#441).
            if fields.borrow().iter().any(|(k, _)| k == method) {
                return Ok(BracketedCall::NotAMethod(recv));
            }
            return Err(RuntimeError::at(format!(
                "no method `{}` on model `{}`", method, bare_name(&mname)), span.clone()));
        };
        let owner = format!("{}.{}", bare_name(&mname), method);
        let explicit = self.bind_shape_bracket(&owner, &f.shape_params, bracket, span)?;
        // The model's own shape args, then the method's. A method that reuses
        // one of the model's shape-param names shadows it with what the call
        // site wrote, which is the only reading that makes the bracket mean
        // anything.
        let mut shape_bindings = model_shape_bindings(fields);
        shape_bindings.retain(|(n, _)| !explicit.iter().any(|(e, _)| e == n));
        shape_bindings.extend(explicit);
        let mut call_args = vec![recv.clone()];
        call_args.extend(arg_vals.iter().cloned());
        Ok(BracketedCall::Dispatched(
            self.call_fn_with_shapes(&f, call_args, &shape_bindings, span.clone())?))
    }

    /// Park an already-evaluated receiver for the `Field` evaluation that the
    /// ordinary postfix path is about to perform on the same node.
    fn park_receiver(&mut self, recv_expr: &Expr, recv: Value) {
        self.parked_receiver = Some((recv_expr as *const Expr, recv));
    }

    /// Take the parked receiver for `expr`, if this is the node it was parked
    /// for. A value parked for some other node belongs to an outer call still
    /// on its way down and is left alone.
    fn take_parked_receiver(&mut self, expr: &Expr) -> Option<Value> {
        let key = expr as *const Expr;
        match &self.parked_receiver {
            Some((parked, _)) if *parked == key =>
                self.parked_receiver.take().map(|(_, v)| v),
            _ => None,
        }
    }

    /// Harvest shape-param bindings from one argument against its declared
    /// parameter type. Three sources, all landing in the same `inferred` map
    /// through `bind_shape_name`, so two arguments that disagree report the
    /// same way whichever pair of sources they came from:
    ///
    ///   - a tensor / view / kv argument's shape against a `ShapeSpec` — the
    ///     original rule, and for a long time the only one;
    ///   - a model array's length against `[T; N]` (#459). The array knows how
    ///     many slots it has, so `N` is every bit as inferable as a tensor dim;
    ///     without this `fn build[N](!ns: [Node; N], …)` type-checked and then
    ///     died at the call site.
    ///   - a model argument's own bound shape args against a parameterized
    ///     model parameter type (#474, parameter position). An instance carries
    ///     its shape args as hidden `__shape_P__` fields, so `!b: Box[H, W]`
    ///     can bind H and W instead of making every call site spell out
    ///     `f![2, 2](b, …)`.
    fn harvest_shape_params(
        &self,
        ty: &Type,
        v: &Value,
        inferred: &mut std::collections::HashMap<String, i64>,
        fn_name: &str,
        sp: &Span,
    ) -> EvalResult<()> {
        match ty {
            Type::Tensor(_, spec, _) | Type::View(_, spec, _) | Type::KV(_, spec, _) => {
                if let Value::Tensor(t) = v {
                    infer_shape_from_arg(spec, t.shape(), inferred, fn_name, sp)?;
                }
            }
            // A model array is a List of slots at runtime (see the
            // `forge.uninit[Model, [N]]` path); its length is the `N`.
            Type::Array(_, size, _) => {
                if let Value::List(items) = v {
                    bind_shape_ident_expr(size, items.len() as i64, inferred, fn_name, sp)?;
                }
            }
            Type::Named { name, args, .. } if !args.is_empty() => {
                let Value::Struct(fields) = v else { return Ok(()); };
                let Some(model) = struct_model_name(fields) else { return Ok(()); };
                // The annotation names the model as the callee's file spells
                // it; the instance carries the declaring file's key.
                if self.resolve_model(name).as_deref() != Some(model.as_str()) { return Ok(()); }
                let Some(decl) = self.model_shape_params.get(&model) else { return Ok(()); };
                // The instance tags its dims under the model's *declaration*
                // names; the parameter type names them positionally.
                for (decl_name, arg) in decl.iter().zip(args) {
                    let Some(target) = type_arg_shape_var(arg) else { continue; };
                    let key = format!("__shape_{}__", decl_name);
                    let actual = fields.borrow().iter()
                        .find(|(k, _)| *k == key)
                        .and_then(|(_, v)| v.as_int());
                    let Some(dim) = actual else { continue; };
                    bind_shape_name(target, dim, inferred, fn_name, sp)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// A fn body's annotations resolve their `type` aliases in the module the
    /// fn was DECLARED in, not the one that called it — `f.module` is the
    /// same tag `grad_capture_names` asks for. Restored on every exit path,
    /// including the error ones, by wrapping the body rather than threading
    /// the save through each `return`.
    ///
    /// #628 follow-up: `named_fn_ret_stack` is saved and restored the same
    /// way, and for the same reason. It used to be pushed inside the body and
    /// popped at each of five `return`s, which is exactly the shape that grows
    /// a leak the next time someone adds a sixth — the `harvest_shape_params`
    /// exit was already such a hole. One push here and one pop after
    /// `call_fn_in_module` returns covers every exit that function has,
    /// including the ones it does not have yet.
    fn call_fn_with_shapes(&mut self, f: &FnDecl, args: Vec<Value>, explicit_shapes: &[(String, i64)], sp: Span) -> EvalResult<Value> {
        // The body — and the shape harvest over its parameter types — resolves
        // top-level names, and its annotations' `type` aliases, in the file the
        // fn was declared in.
        let saved = std::mem::replace(&mut self.cur_module, f.module);
        // See `named_fn_ret_stack`'s doc: a lambda called anywhere in this
        // fn's dynamic extent (including a `?` in its body) can fall back to
        // THIS declared return type when the lambda has none of its own.
        self.named_fn_ret_stack.push(f.ret_type.clone());
        let result = self.call_fn_in_module(f, args, explicit_shapes, sp);
        self.named_fn_ret_stack.pop();
        // Restored LAST: `call_fn_in_module` ends with
        // `finish_query_propagation`, whose `T` may name a `type` alias
        // declared in this fn's own file.
        self.cur_module = saved;
        result
    }

    fn call_fn_in_module(&mut self, f: &FnDecl, args: Vec<Value>, explicit_shapes: &[(String, i64)], sp: Span) -> EvalResult<Value> {
        if args.len() != f.params.len() {
            return Err(RuntimeError::at(
                format!("fn `{}` expects {} args, got {}", f.name, f.params.len(), args.len()),
                sp,
            ));
        }
        let _depth = self.enter_call(&sp)?;
        self.prof_fn_call();
        self.push_scope();
        // Shape params: bind from explicit args first, then infer from the
        // arguments themselves. Bind all explicit shape params upfront so they
        // are visible in the method body even when not referenced in any
        // parameter type annotation (e.g. model shape params that only appear
        // in the body as bare identifiers like `D`).
        for (name, dim) in explicit_shapes {
            self.bind_shape_param(name, Value::Int(*dim, IW::I64));
        }
        let mut inferred: std::collections::HashMap<String, i64> =
            explicit_shapes.iter().cloned().collect();
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (p, v) in f.params.iter().zip(&args) {
            let Some(ty) = &p.ty else { continue; };
            if let Some(spec) = type_shape_spec(ty) {
                collect_idents_in_spec(spec, &mut referenced);
            }
            // An exit like the two below it, so it unwinds like them. This was
            // a bare `?` before, which left this fn's scope standing on the way
            // out — a pre-existing leak on main, not introduced by `?`'s
            // rework, but on the very lines that rework restructures, so it is
            // closed here. Invisible while every error out of here is fatal,
            // which is exactly how the same hole survived in `eval_match`'s arm
            // scope until #639; it stops being invisible now that two of the
            // things travelling this channel are not errors at all.
            if let Err(e) = self.harvest_shape_params(ty, v, &mut inferred, &f.name, &sp) {
                self.pop_scope();
                return Err(e);
            }
        }
        for sp_decl in &f.shape_params {
            referenced.insert(sp_decl.name.clone());
        }
        for name in &referenced {
            match inferred.get(name) {
                Some(&dim) => self.bind_shape_param(name, Value::Int(dim, IW::I64)),
                None => {
                    self.pop_scope();
                    return Err(RuntimeError::at(format!(
                        "fn `{}`: shape param `{}` cannot be inferred from any argument. \
                        Either pass a tensor, model array or model whose declared type uses `{}`, \
                        or make the value explicit in the program.",
                        f.name, name, name), sp));
                }
            }
        }
        // A dim harvested from a source no annotation named (#459/#474: a model
        // array's length, a model argument's own shape args) still belongs in
        // scope — the body may reference it even where `referenced` did not.
        for (name, dim) in &inferred {
            if !referenced.contains(name) {
                self.bind_shape_param(name, Value::Int(*dim, IW::I64));
            }
        }
        for (p, v) in f.params.iter().zip(args) {
            // `Port[L]`: the annotation names a runtime, so the handle has to
            // be one. `L` was decoration before this — a `Port[lua]` parameter
            // took a python handle without complaint in either backend, so the
            // signature implied a check that did not exist.
            if let Err(e) = self.check_declared_port(&v, p.ty.as_ref(), &sp) {
                self.pop_scope();
                return Err(e);
            }
            let mut coerced = self.coerce_declared(v, p.ty.as_ref());
            // #781, #650: a `!` param is a PRIVATE COPY, not an
            // alias of the caller's tensor — the JIT actually copies the
            // bytes in at the CALL SITE, before the callee's own frame
            // exists. That placement is why this charge is monotonic on
            // both backends even where the callee's activations are not:
            // the buffer belongs to the caller's scope, so no
            // compiler-inserted rewind inside the callee covers it (the
            // forward-only `@grad` mark taken below is deliberately taken
            // AFTER this loop for exactly that reason). The copy-OUT is a
            // memcpy back into storage the caller already owns — no second
            // buffer, so nothing to charge for it (#791; see the writeback
            // loop below).
            //
            // An ordinary (non-`!`) param binds no differently from any
            // other variable read and stays uncharged, exactly as before.
            //
            // `charge_private_copy` (not `charge_intermediate`) excludes
            // `Value::Str`: a `str` is a pointer on the JIT (SPEC §3.1), so
            // the copy-in at the call site this comment describes for a
            // tensor is, for a string, a second pointer to the SAME bytes —
            // the JIT requests 0 bytes for it, where charging here as if it
            // copied the string's contents charged a copy that never
            // happens on either backend's wire (PR #815 review).
            if p.mutating {
                if let Err(e) = self.charge_private_copy(&coerced, Some(&sp)) {
                    self.pop_scope();
                    return Err(e);
                }
                // #674: re-stamp — see the
                // `Item::Let`/`Stmt::Let` sites' doc.
                restamp_current_budget(&mut coerced);
            }
            self.bind(&p.name, coerced);
            if let Some(axis) = p.ty.as_ref().and_then(streaming_axis_from_type) {
                self.bind_stream_axis(&p.name, axis);
            }
        }
        // @pp(stages=N): evaluate each `stage K: expr` in sequence, binding `_`
        // to the previous stage's output so the next stage can reference it.
        let is_pp = f.directives.iter().any(|d| d.name == "pp");
        if is_pp {
            // #639: a stage body is evaluated with `eval_expr`, NOT `eval_block`,
            // so this is one of the two entry points where the block-flow
            // carrier would otherwise cross a call frame. It is caught at the
            // stage boundary because the stage boundary IS a function boundary:
            // a `return` here returns from the `@pp` fn, exactly as it does
            // from an ordinary body. Left uncaught, the carrier rode `?` out of
            // the callee and landed in the CALLER — a `return` returned from
            // the caller, a `break` broke the caller's loop — and the same `?`
            // skipped the `pop_scope` below, leaking one frame per escape.
            //
            // #628's `?` signal is NOT caught here. It is the other carrier on
            // the same channel, and this early return sits above the
            // `finish_query_propagation` call that owns it — so a `?` that
            // trips in a stage still escapes the `@pp` fn and is rebuilt
            // against its caller's `T`. That is #647's open category (see
            // `finish_query_propagation`'s doc), unchanged by this merge; what
            // IS fixed here is the leak — the signal now pops the scope on its
            // way out instead of skipping it.
            let mut prev = Value::Nil;
            let mut escaped: Option<Escape> = None;
            for stmt in &f.body.stmts {
                if let Stmt::Stage { body, .. } = stmt {
                    self.bind("_", prev.clone());
                    match self.eval_expr(body) {
                        Ok(v) => prev = v,
                        Err(e) => match e.flow {
                            Some(esc) => { escaped = Some(*esc); break; }
                            // #628: not a block escape — a real error, or a `?`
                            // signal. Either way it leaves this frame, so the
                            // scope pops on the way out.
                            None => {
                                self.pop_scope();
                                return Err(e);
                            }
                        },
                    }
                }
            }
            self.pop_scope();
            return match escaped {
                None => Ok(prev),
                Some(Escape::Return(v, _)) => Ok(v),
                // No loop inside the stage to catch these. `call_fn` already
                // refuses the same thing on an ordinary body; a `@pp` stage is
                // not a loop either, so it gets the same backstop rather than
                // reaching out to whatever loop the caller happened to be in.
                Some(esc) => Err(RuntimeError::at(
                    "break/continue outside loop", esc.span().clone())),
            };
        }
        // #781 (#674 slice 1) / #791: the interpreter-side twin of
        // `jit.rs`'s `fwd_only_scope` — the forward-only entry of a `@grad
        // fn` (a plain `f(..)` call, or `f.fwd(..)`, never `.fwd_bwd`) is
        // wrapped by the JIT in an `InternalMark` whose restore hands back
        // every byte the body allocated. Its return value is a scalar, so
        // nothing the body materialized can escape through it, and the
        // activations are exactly the compiler-reclaimed intermediates
        // `MEMORY.md §1.1` carves out of the monotonic rule.
        //
        // Same gate as the JIT, rule for rule: `@grad` on the fn AND a
        // declared non-`nil` SCALAR return. An ordinary fn is never
        // wrapped; a `@grad fn` returning a tensor would hand back a tensor
        // the rewind had just un-charged.
        //
        // Taken here rather than before the parameter loop on purpose: the
        // `!` copy-in belongs to the CALLER's scope on both backends (see
        // that loop's comment), so it must sit outside this mark.
        //
        // This is NOT a general rewind. It moves `forge_used` only, only
        // across a compiler-inserted scope, and only on the path that
        // actually returns. The language-level `forge.restore` stays
        // monotonic (#614), and `vault_used` is never rewound at all — a
        // `vault.*` constructor is `MEMORY.md §1`'s persistent store, not a
        // temporary the compiler is entitled to reclaim.
        let fwd_only_mark: Option<u64> = if f.directives.iter().any(|d| d.name == "grad")
            && matches!(f.ret_type, Some(Type::Scalar(ref k, _)) if *k != ScalarType::Nil)
        {
            Some(self.budgets[cur_budget()].used)
        } else {
            None
        };
        // #781: snapshot the explicit-constructor running total alongside the
        // mark. A constructor's result can ESCAPE the body — into a model
        // field, through a `!` param, into a module-level `let !` — so the
        // rewind below must not un-charge it; the difference across the body
        // is exactly what it has to keep. See `ctor_forge_bytes`.
        let ctor_at_mark = self.ctor_forge_bytes;
        let result = self.eval_block(&f.body);
        let kept_ctor_bytes = self.ctor_forge_bytes.saturating_sub(ctor_at_mark);
        // Before popping the fn scope, capture the final values of all `!` params.
        // The call site uses these to write back to the caller's bindings.
        // #791: this is the copy-OUT half of #650's private-copy
        // semantics, and it is NOT charged. The bytes go back into the
        // buffer the caller already owns and already paid for — the JIT
        // emits a `memcpy` into the incoming pointer
        // (`emit_mut_param_copy_out`), not a fresh allocation — so there is
        // no second materialization to meter. An earlier pass of #781
        // charged it, which billed every `!`-param call twice; the surviving
        // charge is the copy-IN, in the parameter loop above.
        //
        // #781: captured UNCONDITIONALLY, on every exit path. An `Err` here
        // is not only a failure — a `?` early-return arrives as one (it
        // carries `finish_query_propagation`'s marker and is rebuilt into a
        // normal `(default[T], e)` return a few lines below), and so does a
        // block-flow escape. Gating this on `result.is_ok()` silently
        // dropped the writebacks of a fn that wrote a `!` tensor and then
        // returned early via `?`, so `dmc run` printed the untouched
        // caller-side value where `dmc jit` — which emits the copy-out on
        // every path out of the callee — printed the written one.
        let mut writebacks: Vec<(usize, Value)> = Vec::new();
        for (i, p) in f.params.iter().enumerate().filter(|(_, p)| p.mutating) {
            let Some(v) = self.scopes.last().and_then(|s| s.get(&p.name)).cloned() else {
                continue;
            };
            writebacks.push((i, v));
        }
        self.pending_writebacks = writebacks;
        // The rewind itself: after the copy-out has been captured (it reads
        // the callee's buffer, so it has to happen first — the JIT orders
        // `emit_mut_param_copy_out` ahead of its own restore for the same
        // reason) and only when the body actually produced a value.
        //
        // #781: the landing point is `mark + kept_ctor_bytes`, not `mark`.
        // Everything between the two is what an explicit constructor inside
        // the body charged, and those bytes stay charged — the body's
        // COMPILER-inserted intermediates are what this scope is entitled to
        // reclaim, and a tensor the program spelled out is not one of them.
        if let (Some(mark), true) = (fwd_only_mark, result.is_ok()) {
            // #674 slice 3d: the entered budget, for the same reason
            // `call_grad_in_module`'s `ad_budget` is — and it reads the
            // same value the mark above was taken under, because `enter`
            // pops on every exit edge.
            self.set_budget_used(cur_budget(), mark.saturating_add(kept_ctor_bytes));
        }
        // `?` early-return: rebuild SPEC §4.9's `(default[T], e)` against THIS
        // function's declared `T` — before the scope goes, because a
        // shape-parametric `T` (`-> (Tensor[f32, [N]], Err)`) still needs `N`.
        let result = self.finish_query_propagation(result, f.ret_type.as_ref(), &f.name, &sp);
        self.pop_scope();
        match result {
            // #473: a declared return type is one of the binding sites where an
            // annotation becomes semantics, alongside a typed `let`, a typed
            // param, a typed model field and an `as` cast. Without this
            // `fn k() -> f32 { 0.1 }` hands
            // back an f64 while the JIT — whose signature carries the declared
            // f32 — returns the f32, and the two backends part company on the
            // next operation. A `?` propagation returns a (T, Err) tuple, not a
            // T, so `coerce_scalar_width` (scalar-typed returns only) leaves it
            // alone by construction.
            //
            // #639: the `Flow::Return` a `?` arrives as is built by
            // `finish_query_propagation` just above, so its span is the `?`'s
            // own and is not needed again here.
            Ok(Flow::Normal(v)) | Ok(Flow::Return(v, _)) => {
                self.check_declared_port(&v, f.ret_type.as_ref(), &sp)?;
                Ok(self.coerce_declared(v, f.ret_type.as_ref()))
            }
            Ok(Flow::Break(sp)) | Ok(Flow::Continue(sp)) =>
                Err(RuntimeError::at("break/continue outside loop", sp)),
            // Every `?` signal was already turned into `(default[T], e)` by
            // `finish_query_propagation`; anything still an `Err` here is a
            // real error (or #639's block-flow carrier, which `eval_block` has
            // already re-opened, so it cannot be one either).
            Err(e) => Err(e),
        }
    }

    // ── Block ─────────────────────────────────────────────────────────────

    fn eval_block(&mut self, block: &Block) -> EvalResult<Flow> {
        self.push_scope();
        let result = self.eval_block_body(block);
        self.pop_scope();
        // #639: a carrier that reached the block boundary is this block's own
        // escaping control flow — re-open it so every caller of `eval_block`
        // (a function body, a lambda body, a loop body, an `if` branch, a
        // `@comptime` fold) keeps seeing a `Flow` and nothing else.
        catch_flow(result)
    }

    /// The statements and value of `block`, evaluated in the CURRENT scope.
    /// `eval_block` wraps this in a scope of its own; the trailing
    /// directive-block arm below calls it directly so the inner `let`s stay
    /// visible to the enclosing block, and so a body that ends in an `if` /
    /// `match` STATEMENT (keyword-led, so parsed as a stmt rather than a tail
    /// expr) yields that statement's value exactly as a plain block does —
    /// which is what the checker types it as.
    fn eval_block_body(&mut self, block: &Block) -> EvalResult<Flow> {
        // #478: a block is value-transparent through its tail, but its
        // statements are not — take the hint now, re-arm it on the tail below.
        let want = self.float_lit_hint.take();
        // Defer the last stmt if it's a yielding form AND no explicit tail.
        let n = block.stmts.len();
        let deferred_last = if block.tail_expr.is_none() && n > 0 {
            matches!(&block.stmts[n - 1],
                Stmt::DirectiveBlock { .. } | Stmt::Expr { assign: None, .. }
                | Stmt::Stage { .. } | Stmt::If(_) | Stmt::Match(_))
        } else { false };

        let stop = if deferred_last { n - 1 } else { n };
        for stmt in &block.stmts[..stop] {
            match self.eval_stmt(stmt)? {
                Flow::Normal(_) => {}
                other => return Ok(other),
            }
        }
        if let Some(tail) = &block.tail_expr {
            self.float_lit_hint = want;
            // #639: the tail is an expression like any other, so a block in
            // tail position may hand back control flow rather than a value.
            catch_flow(self.eval_expr(tail).map(Flow::Normal))
        } else if deferred_last {
            match &block.stmts[n - 1] {
                Stmt::DirectiveBlock { directives, body, .. } => {
                    // Walk the body in the outer scope, taking its value the
                    // way any block does; the hint is re-armed on its tail.
                    self.float_lit_hint = want;
                    let raw = match self.eval_block_body(body)? {
                        Flow::Normal(v) => v,
                        other => return Ok(other),
                    };
                    // Apply @cast directive if present (same logic as Expr::DirectiveBlock).
                    let v = if let Some(cast_dir) = directives.iter().find(|d| d.name == "cast") {
                        if let Some(crate::ast::DArg::Positional(type_expr)) = cast_dir.args.first() {
                            if let Expr::Ident(ty_name, _) = type_expr {
                                if self.is_overlay_model(ty_name) {
                                    self.cast_to_struct(&raw, ty_name)?
                                } else {
                                    apply_cast_by_name(&raw, ty_name)
                                }
                            } else { raw }
                        } else { raw }
                    } else { raw };
                    Ok(Flow::Normal(v))
                }
                Stmt::If(ie) => { self.float_lit_hint = want; catch_flow(self.eval_if_flow(ie)) }
                // #639: these three are the value-yielding last statement of a
                // block, so they are expression positions — control flow that
                // left a block inside one rides the carrier and is re-opened
                // here, the same as at any other statement boundary.
                Stmt::Expr { lhs, assign: None, .. } => {
                    self.float_lit_hint = want;
                    catch_flow(self.eval_expr(lhs).map(Flow::Normal))
                }
                Stmt::Stage { body, .. } => catch_flow(self.eval_expr(body).map(Flow::Normal)),
                Stmt::Match(me) => {
                    self.float_lit_hint = want;
                    catch_flow(self.eval_match(me).map(Flow::Normal))
                }
                _ => unreachable!(),
            }
        } else {
            Ok(Flow::Normal(Value::Nil))
        }
    }

    // ── Stmt ──────────────────────────────────────────────────────────────

    /// #639: the statement boundary is where a `Flow` that escaped a block
    /// *expression* — a `let` initializer, a call argument, an assignment RHS,
    /// a bare block statement — becomes a `Flow` again. Every statement form
    /// evaluates expressions somewhere, so catching once here covers all of
    /// them; `eval_block_body` then propagates it to the enclosing loop or
    /// function exactly as it does for an `if` body.
    fn eval_stmt(&mut self, stmt: &Stmt) -> EvalResult<Flow> {
        catch_flow(self.eval_stmt_inner(stmt))
    }

    fn eval_stmt_inner(&mut self, stmt: &Stmt) -> EvalResult<Flow> {
        match stmt {
            Stmt::Let(l) => {
                // `let !alias = recv.field` — create a FieldRef so mutations
                // to alias propagate back through the shared struct Rc.
                if l.mutating {
                    if let Expr::Postfix { expr: inner, op: PostfixOp::Field(fname), .. } = &l.value {
                        // #336: `let !l = Color.Red` is an enum value, not a field
                        // alias — skip the FieldRef path so `Color` isn't evaluated
                        // as a (non-existent) value binding.
                        let is_enum_variant = matches!(inner.as_ref(), Expr::Ident(base, _)
                            if self.enum_variant_ordinal(base, fname).is_some());
                        if !is_enum_variant {
                            let recv = self.eval_expr(inner)?;
                            if let Value::Struct(rc) = recv {
                                if let Pattern::Ident(alias, _) = &l.pattern {
                                    // #444: only TENSOR fields bind as a live
                                    // alias (the element-write-through idiom).
                                    // Every other field kind — scalars above
                                    // all — binds its current VALUE, per the
                                    // spec's `let !y = x` copy rule; aliasing
                                    // a scalar made every "snapshot this
                                    // field" read a moving target.
                                    let fval = {
                                        let borrowed = rc.borrow();
                                        borrowed.iter()
                                            .find(|(k, _)| k == fname)
                                            .map(|(_, v)| v.clone())
                                    };
                                    match fval {
                                        Some(Value::Tensor(_)) => {
                                            self.bind(alias, Value::FieldRef { rc, field: fname.clone() });
                                            return Ok(Flow::Normal(Value::Nil));
                                        }
                                        Some(v) => {
                                            self.bind(alias, v);
                                            return Ok(Flow::Normal(Value::Nil));
                                        }
                                        None => {} // not a field — fall through
                                    }
                                }
                            }
                        }
                    }
                }
                let v = self.eval_expr(&l.value)?;
                self.check_declared_port(&v, l.ty.as_ref(), &l.span)?;
                let mut v = self.coerce_declared(v, l.ty.as_ref());
                // #781, SPEC §3.4: `let !y = x` binds a PRIVATE, independently
                // mutable copy of `x` — not an alias — so the JIT charges the
                // copy for a tensor (a `Value::Str` copy is excluded by
                // `charge_private_copy` — see that fn's doc: a `str` is a
                // pointer on the JIT, so the copy the JIT makes there is a
                // second pointer, not a byte-for-byte copy, and it requests
                // 0 bytes for it). Narrowed to a bare-identifier RHS (the
                // paradigm case, and the one the FieldRef branch above
                // doesn't already handle): `let !y = a + b` etc. already
                // charged its fresh result as an implicit intermediate
                // inside `eval_expr` (`BinOp`, a tensor literal, …), so
                // charging it again here would double-bill a value that was
                // never aliased in the first place — only a bare `x`
                // denotes storage that already has an owner and therefore
                // needs an actual copy to give `y` independent identity.
                if l.mutating && matches!(l.value, Expr::Ident(_, _)) {
                    self.charge_private_copy(&v, Some(&l.span))?;
                    // #674: re-stamp — see
                    // the identical `Item::Let` site's doc.
                    restamp_current_budget(&mut v);
                }
                self.bind_pattern(&l.pattern, v);
                // #399: resolve the streaming axis and declared capacity from the
                // explicit `KV[..~..]` type annotation OR, failing that, the
                // constructor value itself. Previously both were gated on the
                // annotation, so the common unannotated form
                // `let !c = forge.kv[T, [.., ~, ..]](capacity = N)` went untracked:
                // `<-` then guessed the append axis (wrong for equal-shaped frames)
                // and silently overflowed the reservation (Spec §3.6 mandates a
                // panic on overflow).
                let stream_axis = l.ty.as_ref().and_then(streaming_axis_from_type)
                    .or_else(|| streaming_axis_from_kv_value(&l.value));
                if let Some(axis) = stream_axis {
                    self.bind_stream_axis_pattern(&l.pattern, axis);
                }
                if let Some(cap) = extract_kv_capacity(&l.value) {
                    if let Pattern::Ident(name, _) = &l.pattern {
                        self.kv_capacities.insert(name.clone(), cap);
                    }
                }
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::Expr { lhs, assign, span } => {
                if let Some((op, rhs)) = assign {
                    // #452: flag `name = builtin(name, …)` — and *only* that
                    // exact shape — for the in-place path (see
                    // `release_move_target`). Requiring the call to be the
                    // whole right-hand side is what makes the release safe: the
                    // binding it clears is guaranteed to be overwritten by that
                    // same call's result, and no other part of the expression
                    // can still be waiting to read the old value. `xs = if
                    // list_len(xs) > 2 { list_slice(xs, 0, 2) } else { xs }`
                    // reads `xs` again after the call, so it does not qualify.
                    // Cleared unconditionally below — the hint must never
                    // outlive this one right-hand side.
                    if let (AssignOp::Eq, Expr::Ident(name, _)) = (&*op, &*lhs) {
                        if let Expr::Postfix { expr: callee, op: PostfixOp::Call(cargs), .. } = &*rhs {
                            let callee_is_builtin = matches!(callee.as_ref(),
                                Expr::Ident(f, _) if is_builtin(f) && self.resolve_fn(f).is_none());
                            let first_arg_is_target = matches!(cargs.first(),
                                Some(CallArg::Positional(Expr::Ident(a, _))) if a == name);
                            if callee_is_builtin && first_arg_is_target {
                                self.move_hint = Some(name.clone());
                            }
                        }
                    }
                    let rval = self.eval_expr(rhs);
                    self.move_hint = None;
                    let rval = rval?;
                    match (op, lhs) {
                        // Simple assignment to ident
                        (AssignOp::Eq, Expr::Ident(name, _)) => {
                            // Try to assign to existing binding; create in current scope if new.
                            if self.assign(name, rval.clone()).is_err() {
                                self.bind(name, rval);
                            }
                        }
                        (AssignOp::PlusEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup_deref(name).unwrap_or(Value::Nil);
                            // #674: this reads the binding's CURRENT value
                            // directly, bypassing `eval_expr`'s wrapper.
                            self.check_live(&cur, span.clone())?;
                            // Tensor ops dispatch to the elementwise variant;
                            // pure-scalar dispatch to the scalar variant. The
                            // language doesn't distinguish += vs .+= at the
                            // surface.
                            let op = if matches!(cur, Value::Tensor(_)) || matches!(rval, Value::Tensor(_)) {
                                BinOp::DotAdd
                            } else { BinOp::Add };
                            let new = apply_binop(op, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::MinusEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup_deref(name).unwrap_or(Value::Nil);
                            // #674: this reads the binding's CURRENT value
                            // directly, bypassing `eval_expr`'s wrapper.
                            self.check_live(&cur, span.clone())?;
                            let op = if matches!(cur, Value::Tensor(_)) || matches!(rval, Value::Tensor(_)) {
                                BinOp::DotSub
                            } else { BinOp::Sub };
                            let new = apply_binop(op, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        // #222: `*=` / `/=` were unhandled and silently no-op'd
                        // (so `acc *= i` left acc unchanged — interp diverged from
                        // the JIT, which applies them). Dispatch scalar vs tensor
                        // like `+=`/`-=`.
                        (AssignOp::StarEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup_deref(name).unwrap_or(Value::Nil);
                            // #674: this reads the binding's CURRENT value
                            // directly, bypassing `eval_expr`'s wrapper.
                            self.check_live(&cur, span.clone())?;
                            let op = if matches!(cur, Value::Tensor(_)) || matches!(rval, Value::Tensor(_)) {
                                BinOp::DotMul
                            } else { BinOp::Mul };
                            let new = apply_binop(op, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::SlashEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup_deref(name).unwrap_or(Value::Nil);
                            // #674: this reads the binding's CURRENT value
                            // directly, bypassing `eval_expr`'s wrapper.
                            self.check_live(&cur, span.clone())?;
                            let op = if matches!(cur, Value::Tensor(_)) || matches!(rval, Value::Tensor(_)) {
                                BinOp::DotDiv
                            } else { BinOp::Div };
                            let new = apply_binop(op, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        // Bitwise compound-assign (int only; no tensor variant).
                        (AssignOp::AmpEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup_deref(name).unwrap_or(Value::Nil);
                            // #674: this reads the binding's CURRENT value
                            // directly, bypassing `eval_expr`'s wrapper.
                            self.check_live(&cur, span.clone())?;
                            let new = apply_binop(BinOp::BitAnd, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::BarEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup_deref(name).unwrap_or(Value::Nil);
                            // #674: this reads the binding's CURRENT value
                            // directly, bypassing `eval_expr`'s wrapper.
                            self.check_live(&cur, span.clone())?;
                            let new = apply_binop(BinOp::BitOr, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::CaretEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup_deref(name).unwrap_or(Value::Nil);
                            // #674: this reads the binding's CURRENT value
                            // directly, bypassing `eval_expr`'s wrapper.
                            self.check_live(&cur, span.clone())?;
                            let new = apply_binop(BinOp::BitXor, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::StreamArrow, Expr::Ident(name, _)) => {
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
                            // #674: the stream-append slot read.
                            self.check_live(&cur, span.clone())?;
                            match (cur, rval) {
                                (Value::Tensor(existing), Value::Tensor(new_data)) => {
                                    let axis = self.lookup_stream_axis(name)
                                        .unwrap_or_else(|| {
                                            // Find the axis where shapes differ — that is the
                                            // streaming axis. If shapes differ on exactly one
                                            // axis use it; otherwise fall back to axis 0.
                                            let ex_shape = existing.shape();
                                            let new_shape = new_data.shape();
                                            if ex_shape.len() == new_shape.len() {
                                                let differing: Vec<usize> = ex_shape.iter()
                                                    .zip(new_shape.iter())
                                                    .enumerate()
                                                    .filter(|(_, (a, b))| a != b)
                                                    .map(|(i, _)| i)
                                                    .collect();
                                                if differing.len() == 1 {
                                                    return differing[0];
                                                }
                                            }
                                            0
                                        });
                                    if axis >= existing.ndim() {
                                        return Err(RuntimeError::msg(format!(
                                            "<- stream append axis {} out of bounds for {}-d tensor",
                                            axis,
                                            existing.ndim()
                                        )));
                                    }
                                    // SPEC §4.8 also spells the appendee with the streaming
                                    // axis dropped; bring it up to rank first so the capacity
                                    // check below counts frames and not feature width (p31).
                                    let new_arr = promote_stream_appendee(
                                        &existing.data, &new_data.data, axis,
                                    ).map_err(|e| RuntimeError::msg(
                                        format!("<- stream append failed: {}", e)))?;
                                    // Enforce declared capacity (Spec §3.6: panics on overflow).
                                    if let Some(&cap) = self.kv_capacities.get(name.as_str()) {
                                        let current_len = existing.shape()[axis];
                                        let append_len = new_arr.shape()[axis];
                                        if current_len + append_len > cap {
                                            return Err(RuntimeError::msg(format!(
                                                "stream `<-` append exceeded declared capacity {} \
                                                 for `{}` (axis {} has {} frames, appending {} more)",
                                                cap, name, axis, current_len, append_len
                                            )));
                                        }
                                    }
                                    let cat = ndarray::concatenate(
                                        Axis(axis),
                                        &[existing.view(), new_arr.view()],
                                    ).map_err(|e| RuntimeError::msg(
                                        format!("<- stream append failed: {}", e)
                                    ))?;
                                    let grown = Value::tensor_dt(cat, existing.dtype);
                                    // #781: `name <- rhs` concatenates into a
                                    // freshly allocated, wider array — an
                                    // implicit intermediate, EXCEPT when
                                    // `name` is a declared `forge.kv`/
                                    // `stream.kv` binding: MEMORY.md §1
                                    // keeps the Stream arena uncharged by
                                    // design (it carries its own
                                    // `capacity`, already enforced above).
                                    // `kv_capacities` is populated only for
                                    // such a binding (`extract_kv_capacity`
                                    // at the `let` site), so its absence
                                    // here means `name` is an ordinary
                                    // Forge tensor being grown, and the JIT
                                    // has no lowering for that yet either.
                                    if !self.kv_capacities.contains_key(name.as_str()) {
                                        self.charge_intermediate(&grown, None)?;
                                    }
                                    self.assign(name, grown).ok();
                                }
                                // `let !q = self.field; q <- x` — q is a FieldRef
                                // alias. Append through it back into the field.
                                (Value::FieldRef { rc, field }, Value::Tensor(new_data)) => {
                                    let mut borrowed = rc.borrow_mut();
                                    let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == &field) else {
                                        return Err(RuntimeError::msg(format!("unknown field `{}` in <- append", field)));
                                    };
                                    let (ex_data, dtype, ndim) = match slot {
                                        Value::Tensor(t) => {
                                            // #674: the
                                            // other stream-append slot read.
                                            self.check_tensor_live(t, span.clone())?;
                                            (t.data.clone(), t.dtype, t.ndim())
                                        }
                                        other => return Err(RuntimeError::msg(
                                            format!("<- stream-arrow requires a tensor field, got {}", other.type_name()))),
                                    };
                                    let axis = stream_axis_for(&ex_data, &new_data.data);
                                    if axis >= ndim {
                                        return Err(RuntimeError::msg(format!(
                                            "<- stream append axis {} out of bounds for {}-d tensor", axis, ndim)));
                                    }
                                    let new_arr = promote_stream_appendee(&ex_data, &new_data.data, axis)
                                        .map_err(|e| RuntimeError::msg(format!("<- stream append failed: {}", e)))?;
                                    let cat = ndarray::concatenate(Axis(axis), &[ex_data.view(), new_arr.view()])
                                        .map_err(|e| RuntimeError::msg(format!("<- stream append failed: {}", e)))?;
                                    *slot = Value::tensor_dt(cat, dtype);
                                }
                                _ => return Err(RuntimeError::msg(
                                    "<- stream-arrow requires tensor on both sides"
                                )),
                            }
                        }
                        // Stream append through a struct field: `self.cache <- x`.
                        // The field-access LHS doesn't match the Ident arm above, so
                        // without this the append silently no-ops (the field never
                        // grows). Concatenate along the streaming axis and write the
                        // grown tensor back into the shared struct slot.
                        (AssignOp::StreamArrow, Expr::Postfix { expr: struct_expr, op: PostfixOp::Field(field_name), .. }) => {
                            let new_data = match rval {
                                Value::Tensor(t) => t,
                                other => return Err(RuntimeError::at(
                                    format!("<- stream-arrow requires a tensor on the right, got {}", other.type_name()),
                                    span.clone())),
                            };
                            let recv = self.eval_expr(struct_expr)?;
                            let Value::Struct(fields) = recv else {
                                return Err(RuntimeError::at(
                                    format!("<- stream-arrow target must be a struct field, got {}", recv.type_name()),
                                    span.clone()));
                            };
                            let mut borrowed = fields.borrow_mut();
                            let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == field_name) else {
                                return Err(RuntimeError::at(
                                    format!("unknown field `{}` in <- append", field_name), span.clone()));
                            };
                            let (ex_data, dtype, ndim) = match slot {
                                Value::Tensor(t) => {
                                    // #674: the struct-field
                                    // spelling's own slot read.
                                    self.check_tensor_live(t, span.clone())?;
                                    (t.data.clone(), t.dtype, t.ndim())
                                }
                                other => return Err(RuntimeError::at(
                                    format!("<- stream-arrow requires a tensor field, got {}", other.type_name()),
                                    span.clone())),
                            };
                            let axis = stream_axis_for(&ex_data, &new_data.data);
                            if axis >= ndim {
                                return Err(RuntimeError::at(format!(
                                    "<- stream append axis {} out of bounds for {}-d tensor", axis, ndim),
                                    span.clone()));
                            }
                            let cat = ndarray::concatenate(Axis(axis), &[ex_data.view(), new_data.data.view()])
                                .map_err(|e| RuntimeError::at(format!("<- stream append failed: {}", e), span.clone()))?;
                            *slot = Value::tensor_dt(cat, dtype);
                        }
                        // Tensor element write: t[i] = val  or  t[i, j] = val
                        (AssignOp::Eq, Expr::Postfix { expr: base_expr, op: PostfixOp::Index(idx_elems), .. }) => {
                            if let Expr::Ident(name, _) = base_expr.as_ref() {
                                // #689: a plain tensor binding takes the in-place path —
                                // `lookup` below deep-clones a Tensor's backing `ArrayD`
                                // (unlike `Value::List`, `Tensor` is not `Rc`-wrapped), so
                                // routing `t[i] = v` through it made every element store
                                // O(N) in the tensor's size and an N-element fill O(N²).
                                // Peeking the variant through `lookup_ref` (no clone) lets
                                // a Tensor skip straight to mutating the scope slot's
                                // `ArrayD` through `&mut`; List/FieldRef/other keep the
                                // original lookup-and-reassign shape below (List's "clone"
                                // there is a cheap `Rc::clone`, not a data copy).
                                if matches!(self.lookup_ref(name), Some(Value::Tensor(_))) {
                                    // #674: this fast path
                                    // mutates the scope slot's `ArrayD`
                                    // directly, bypassing `eval_expr`.
                                    if let Some(Value::Tensor(t)) = self.lookup_ref(name) {
                                        self.check_tensor_live(t, span.clone())?;
                                    }
                                    // #443: resolve indices before taking the mutable scope
                                    // borrow — an index expr can read `name` itself (e.g.
                                    // `t[t[0] as i64] = 1`), and holding `&mut` on the slot
                                    // across a re-entrant `eval_expr` would double-borrow.
                                    let idx_vals = self.eval_index_elems(idx_elems, &span)?;
                                    for scope in self.scopes.iter_mut().rev() {
                                        match scope.get_mut(name) {
                                            Some(Value::Tensor(t)) => {
                                                let dt = t.dtype;
                                                Self::assign_to_tensor_resolved(&mut t.data, dt, &idx_vals, rval, span.clone())?;
                                                break;
                                            }
                                            // Not shadowing: the peek above already
                                            // confirmed `name` was a tensor in this same
                                            // scope, so a non-tensor here means the index
                                            // expression we just evaluated (`eval_index_elems`
                                            // above) retyped this very binding — e.g. an
                                            // index expr that reassigns the indexed name to
                                            // a non-tensor. Error like the sibling arms
                                            // below instead of silently dropping the store.
                                            Some(other) => {
                                                return Err(RuntimeError::at(
                                                    format!("indexed assignment requires a tensor, got {}", other.type_name()),
                                                    span.clone(),
                                                ));
                                            }
                                            None => continue,
                                        }
                                    }
                                    return Ok(Flow::Normal(Value::Nil));
                                }
                                let cur = self.lookup(name).unwrap_or(Value::Nil);
                                match cur {
                                    // #266: model-array (List) write — `arr[i] = instance`
                                    // stores the value into slot i (parity with the JIT's
                                    // ModelArray store). Single scalar index; negatives wrap.
                                    Value::List(mut buf) => {
                                        let raw = match idx_elems.as_slice() {
                                            [IndexElem::Expr(e)] => self.eval_expr(e)?.as_int().ok_or_else(|| {
                                                RuntimeError::at("model-array index must be an integer", span.clone())
                                            })?,
                                            _ => return Err(RuntimeError::at(
                                                "model-array assignment requires a single scalar index", span.clone())),
                                        };
                                        let len = buf.len() as i64;
                                        let i = if raw < 0 { len + raw } else { raw };
                                        if i < 0 || i >= len {
                                            return Err(RuntimeError::at(
                                                format!("index {} out of bounds for axis 0 of size {}", raw, len), span.clone()));
                                        }
                                        // Drop the binding's handle on the buffer first:
                                        // `make_mut` copies while a second holder exists,
                                        // which would make an element-by-element fill of a
                                        // model array quadratic (#452). The slot is
                                        // reassigned on the next line either way.
                                        self.assign(name, Value::Nil).ok();
                                        Rc::make_mut(&mut buf)[i as usize] = rval;
                                        self.assign(name, Value::List(buf)).ok();
                                    }
                                    Value::FieldRef { rc, field } => {
                                        // #443: resolve the index *before* borrowing the
                                        // struct's fields. An index that reads the same
                                        // struct (`c[self.table[0]] = 1`, or a method call)
                                        // re-enters the interpreter, and the member-access
                                        // path borrows this same RefCell — under a live
                                        // `borrow_mut()` that is a process-killing panic,
                                        // not a diagnostic. `rval` is already evaluated
                                        // above, which is why `self` in value position
                                        // was always safe.
                                        let idx_vals = self.eval_index_elems(idx_elems, &span)?;
                                        let mut borrowed = rc.borrow_mut();
                                        if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == &field) {
                                            if let Value::Tensor(arr) = slot {
                                                let dt = arr.dtype;
                                                Self::assign_to_tensor_resolved(arr, dt, &idx_vals, rval, span.clone())?;
                                            } else if matches!(slot, Value::List(_)) {
                                                // #476: same model-array store as the
                                                // `m.field[i] =` path below, for the
                                                // aliased-field spelling.
                                                assign_into_model_array(slot, &idx_vals, rval, &span)?;
                                            } else {
                                                return Err(RuntimeError::at(
                                                    format!("indexed assignment requires a tensor, got {}", slot.type_name()),
                                                    span.clone(),
                                                ));
                                            }
                                        }
                                    }
                                    other => {
                                        return Err(RuntimeError::at(
                                            format!("indexed assignment requires a tensor, got {}", other.type_name()),
                                            span.clone(),
                                        ));
                                    }
                                }
                            } else if let Expr::Postfix { expr: struct_expr, op: PostfixOp::Field(field_name), .. } = base_expr.as_ref() {
                                // m.field[i] = val — tensor element write through struct field
                                let recv = self.eval_expr(struct_expr)?;
                                if let Value::Struct(fields) = recv {
                                    // #443: same hazard as the FieldRef arm — the index may
                                    // read `m` itself, so evaluate it before the borrow.
                                    let idx_vals = self.eval_index_elems(idx_elems, &span)?;
                                    let mut borrowed = fields.borrow_mut();
                                    if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == field_name) {
                                        if let Value::Tensor(arr) = slot {
                                            let dt = arr.dtype;
                                            Self::assign_to_tensor_resolved(arr, dt, &idx_vals, rval, span.clone())?;
                                        } else if matches!(slot, Value::List(_)) {
                                            // #476 spelling 3: `h.cells[i] = Cell { .. }`
                                            // — a model-array field. The local form
                                            // (`cs[i] = ..`) has always worked; through a
                                            // field it hit the tensor-only path and failed
                                            // with a message about tensors and lists,
                                            // neither of which the author wrote. Writing
                                            // the slot here goes through the struct's own
                                            // `Rc<RefCell<..>>`, so unlike the
                                            // copy-out-and-fill spelling it really lands.
                                            assign_into_model_array(slot, &idx_vals, rval, &span)?;
                                        } else {
                                            return Err(RuntimeError::at(
                                                format!("indexed assignment requires a tensor, got {}", slot.type_name()),
                                                span.clone(),
                                            ));
                                        }
                                    }
                                } else {
                                    return Err(RuntimeError::at(
                                        format!("indexed assignment requires a tensor, got {}", recv.type_name()),
                                        span.clone(),
                                    ));
                                }
                            }
                        }
                        // #693: compound assignment to a tensor element —
                        // `t[i] += v`, `t[i, j] -= v`, `m.t[i] *= v`, and
                        // through a live field alias (`let !y = m.t; y[i] /=
                        // v`). Before this arm existed, every one of these
                        // fell to the catch-all no-op below: `--check`
                        // accepted the form, nothing stored, and the element
                        // read back unchanged (SPEC §3.4 requires a compound
                        // assignment through a tensor-field alias to reach
                        // the field, same as a plain element write).
                        (AssignOp::PlusEq | AssignOp::MinusEq | AssignOp::StarEq | AssignOp::SlashEq,
                         Expr::Postfix { expr: base_expr, op: PostfixOp::Index(idx_elems), .. }) => {
                            let (scalar_op, dot_op) = Self::compound_binops(op);
                            if let Expr::Ident(name, _) = base_expr.as_ref() {
                                // Same in-place fast path #692 uses for `=`:
                                // peek the variant with no clone before
                                // touching the index.
                                if matches!(self.lookup_ref(name), Some(Value::Tensor(_))) {
                                    // #674: a compound indexed
                                    // assign READS the element before
                                    // writing it back — the paradigm case
                                    // this check exists for.
                                    if let Some(Value::Tensor(t)) = self.lookup_ref(name) {
                                        self.check_tensor_live(t, span.clone())?;
                                    }
                                    let idx_vals = self.eval_index_elems(idx_elems, &span)?;
                                    for scope in self.scopes.iter_mut().rev() {
                                        match scope.get_mut(name) {
                                            Some(Value::Tensor(t)) => {
                                                let dt = t.dtype;
                                                Self::compound_assign_tensor_elem(
                                                    &mut t.data, dt, scalar_op, dot_op,
                                                    &idx_vals, rval, span.clone(),
                                                )?;
                                                break;
                                            }
                                            // See the `=` arm above: the peek
                                            // already confirmed a tensor, so a
                                            // non-tensor here means the index
                                            // expression itself retyped `name`.
                                            Some(other) => {
                                                return Err(RuntimeError::at(
                                                    format!("indexed assignment requires a tensor, got {}", other.type_name()),
                                                    span.clone(),
                                                ));
                                            }
                                            None => continue,
                                        }
                                    }
                                    return Ok(Flow::Normal(Value::Nil));
                                }
                                let cur = self.lookup(name).unwrap_or(Value::Nil);
                                match cur {
                                    // SPEC §3.4: `let !y = m.t` is a live alias
                                    // into the model field — `y[i] op= v` must
                                    // reach `m.t`, same as `y[i] = v`.
                                    Value::FieldRef { rc, field } => {
                                        let idx_vals = self.eval_index_elems(idx_elems, &span)?;
                                        let mut borrowed = rc.borrow_mut();
                                        if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == &field) {
                                            if let Value::Tensor(arr) = slot {
                                                // #674: the field's tensor,
                                                // checked before a compound assign combines it.
                                                self.check_tensor_live(arr, span.clone())?;
                                                let dt = arr.dtype;
                                                Self::compound_assign_tensor_elem(
                                                    &mut arr.data, dt, scalar_op, dot_op,
                                                    &idx_vals, rval, span.clone(),
                                                )?;
                                            } else {
                                                return Err(RuntimeError::at(
                                                    format!("indexed assignment requires a tensor, got {}", slot.type_name()),
                                                    span.clone(),
                                                ));
                                            }
                                        }
                                    }
                                    // A model-array (List) element, or any
                                    // other non-tensor binding, has no
                                    // compound-arithmetic meaning — name it
                                    // and refuse rather than drop the store.
                                    other => {
                                        return Err(RuntimeError::at(
                                            format!("indexed assignment requires a tensor, got {}", other.type_name()),
                                            span.clone(),
                                        ));
                                    }
                                }
                            } else if let Expr::Postfix { expr: struct_expr, op: PostfixOp::Field(field_name), .. } = base_expr.as_ref() {
                                // m.t[i] op= val — tensor element compound-write
                                // through a struct field.
                                let recv = self.eval_expr(struct_expr)?;
                                if let Value::Struct(fields) = recv {
                                    let idx_vals = self.eval_index_elems(idx_elems, &span)?;
                                    let mut borrowed = fields.borrow_mut();
                                    if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == field_name) {
                                        if let Value::Tensor(arr) = slot {
                                            // #674: the field's tensor,
                                            // checked before a compound assign combines it.
                                            self.check_tensor_live(arr, span.clone())?;
                                            let dt = arr.dtype;
                                            Self::compound_assign_tensor_elem(
                                                &mut arr.data, dt, scalar_op, dot_op,
                                                &idx_vals, rval, span.clone(),
                                            )?;
                                        } else {
                                            return Err(RuntimeError::at(
                                                format!("indexed assignment requires a tensor, got {}", slot.type_name()),
                                                span.clone(),
                                            ));
                                        }
                                    }
                                } else {
                                    return Err(RuntimeError::at(
                                        format!("indexed assignment requires a tensor, got {}", recv.type_name()),
                                        span.clone(),
                                    ));
                                }
                            } else {
                                return Err(RuntimeError::at(
                                    "compound assignment to this indexed target is not supported".to_string(),
                                    span.clone(),
                                ));
                            }
                        }
                        // Direct field write: m.field = expr  (including self.field = expr in methods)
                        (AssignOp::Eq, Expr::Postfix { expr: base_expr, op: PostfixOp::Field(field_name), .. }) => {
                            let recv = self.eval_expr(base_expr)?;
                            if let Value::Struct(fields) = recv {
                                // #473: a declared scalar field type binds the
                                // stored value's float width here too, so a
                                // later `m.x = 0.1` holds the same f32 that
                                // `M { x: 0.1 }` stored at construction.
                                let declared = self.struct_field_ty(&fields, field_name);
                                let mut borrowed = fields.borrow_mut();
                                if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == field_name) {
                                    *slot = self.coerce_declared(rval, declared.as_ref());
                                }
                            }
                        }
                        // `:=` — shadow/rebind: create a new binding in current scope.
                        (AssignOp::ColonEq, Expr::Ident(name, _)) => {
                            self.bind(name, rval);
                        }
                        // #693 audit, follow-up: the only OTHER target shape
                        // reachable by this arm's own operator family (`+= -=
                        // *= /=`) was a direct field compound-assign with no
                        // index — `m.field += v` (incl. `self.field += v` in
                        // a method, and through a live struct alias / nested
                        // `o.i.n += v`, both of which resolve here the same
                        // way `eval_expr(base_expr)` resolves any other field
                        // chain). This used to name the operator with `{:?}`
                        // and refuse outright ("assignment `PlusEq` to a
                        // struct field is not supported" — leaking the Rust
                        // enum variant to users) on every shape, including
                        // the ordinary scalar case a bare compound-assign
                        // targets almost every time.
                        //
                        // Coordinator decision on the #693 follow-up review:
                        // compute it instead, mirroring the `(Eq,
                        // Postfix{Field})` arm just above — read the field's
                        // current value, combine via `apply_binop` (the same
                        // dispatch the plain-ident compound arms and #693's
                        // indexed-tensor-element arm use), and store back
                        // through the same slot the `=` arm writes, coerced
                        // to the field's declared width so `m.x += 1` rounds
                        // an f32 field exactly as `m.x = m.x + 1` would
                        // (SPEC §3.4: compound assignment must reach the
                        // field, same as a plain write, including through an
                        // alias binding). A tensor-typed field with a
                        // whole-tensor RHS is the same trivial elementwise
                        // replace the plain-ident `y += tensor` arms already
                        // do; a tensor field combined with a non-tensor RHS
                        // has no established broadcast semantics here, so
                        // that shape is refused with a clear, non-leaking
                        // message (`assignop_str`, not `{:?}`) rather than
                        // guessed at.
                        (AssignOp::PlusEq | AssignOp::MinusEq | AssignOp::StarEq | AssignOp::SlashEq,
                         Expr::Postfix { expr: base_expr, op: PostfixOp::Field(field_name), .. }) => {
                            let (scalar_op, dot_op) = Self::compound_binops(op);
                            let recv = self.eval_expr(base_expr)?;
                            if let Value::Struct(fields) = recv {
                                let declared = self.struct_field_ty(&fields, field_name);
                                let mut borrowed = fields.borrow_mut();
                                if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == field_name) {
                                    match (&*slot, &rval) {
                                        (Value::Tensor(_), Value::Tensor(_)) => {
                                            // #674: the field's tensor,
                                            // checked before a compound assign combines it.
                                            self.check_live(slot, span.clone())?;
                                            let cur = slot.clone();
                                            *slot = apply_binop(dot_op, &cur, &rval)?;
                                        }
                                        (Value::Tensor(_), _) => {
                                            return Err(RuntimeError::at(
                                                format!(
                                                    "compound assignment `{}` to a whole tensor-typed \
                                                     field needs a tensor on the right (broadcasting a \
                                                     scalar into a field this way is not supported) — \
                                                     got {}",
                                                    crate::fmt::assignop_str(op), rval.type_name()
                                                ),
                                                span.clone(),
                                            ));
                                        }
                                        _ => {
                                            let cur = slot.clone();
                                            let new_val = apply_binop(scalar_op, &cur, &rval)?;
                                            *slot = self.coerce_declared(new_val, declared.as_ref());
                                        }
                                    }
                                }
                            }
                        }
                        // Every other combination reaching here — `:=` on a
                        // non-ident target (#685's sibling bug, deliberately
                        // NOT fixed here: one bug category per PR, AGENTS.md
                        // §3), bitwise compound (`&= |= ^=`) against a
                        // field/index target (a *silent no-op* under `dmc
                        // run` — same defect class as #693, different
                        // operator family, filed as #714 and deliberately
                        // not fixed here: one bug category per PR), and `<-`
                        // stream-append to an INDEXED local (`caches[i] <-
                        // frame`, #176's KV-array append — real,
                        // JIT-supported, exercised by
                        // `jit_kv_array_indexed_append_advances_len` /
                        // `jit_kv_array_literal_and_pass`, which is why this
                        // is not hardened into an error alongside the arm
                        // above; the interpreter has no arm for it at all —
                        // a silent no-op under `dmc run` against a real JIT
                        // append, an unresolved #176-family divergence filed
                        // as #715) — is unrelated to #693 and left exactly as
                        // it was: evaluate the lhs for side effects, no-op.
                        _ => {
                            let _ = self.eval_expr(lhs)?;
                        }
                    }
                } else {
                    let _ = self.eval_expr(lhs)?;
                }
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::If(if_e) => self.eval_if_flow(if_e),
            Stmt::Match(me) => Ok(Flow::Normal(self.eval_match(me)?)),
            Stmt::For { pattern, iter, body, .. } => {
                let iter_v = self.eval_expr(iter)?;
                let items = expand_iter(&iter_v)?;
                for item in items {
                    self.push_scope();
                    self.bind_pattern(pattern, item);
                    let r = self.eval_block(body);
                    self.pop_scope();
                    match r? {
                        Flow::Break(_) => break,
                        Flow::Continue(_) | Flow::Normal(_) => continue,
                        Flow::Return(v, rsp) => return Ok(Flow::Return(v, rsp)),
                    }
                }
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::While { cond, body, .. } => {
                loop {
                    let c = self.eval_expr(cond)?;
                    if !c.as_bool().unwrap_or(false) { break; }
                    match self.eval_block(body)? {
                        Flow::Break(_) => break,
                        Flow::Return(v, rsp) => return Ok(Flow::Return(v, rsp)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::Loop { body, .. } => {
                loop {
                    match self.eval_block(body)? {
                        Flow::Break(_) => break,
                        Flow::Return(v, rsp) => return Ok(Flow::Return(v, rsp)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::Stage { body, .. } => {
                let _ = self.eval_expr(body)?;
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::Directive { inner, .. } => self.eval_stmt(inner),
            Stmt::DirectiveBlock { body, .. } => {
                // Treat as evaluating the inner block; result discarded
                // unless this is the tail (handled in eval_block).
                // #639: a directive block is a block, so a `return` / `break` /
                // `continue` inside it belongs to the enclosing fn or loop; the
                // VALUE is what gets discarded here, never the control flow.
                match self.eval_block(body)? {
                    Flow::Normal(_) => Ok(Flow::Normal(Value::Nil)),
                    other => Ok(other),
                }
            }
            Stmt::Break(sp)    => Ok(Flow::Break(sp.clone())),
            Stmt::Continue(sp) => Ok(Flow::Continue(sp.clone())),
            Stmt::Return { value, span } => {
                let v = if let Some(e) = value { self.eval_expr(e)? } else { Value::Nil };
                Ok(Flow::Return(v, span.clone()))
            }
        }
    }

    /// Evaluate an if-expression in *flow-aware* form. break/continue/return
    /// inside an if-body propagate out via the returned Flow. Callers that
    /// only want a Value should unwrap (or treat non-Normal as an error in
    /// expression context).
    /// #478: the float width this expression will evaluate to, answered
    /// **without evaluating it** — so no side effects, and no commitment to a
    /// branch that will not be taken. The mirror of
    /// `jit::Translator::static_float_kind`; deliberately partial, and a
    /// `None` answer changes nothing (the literal keeps #209's f64 default).
    fn static_float_width(&self, e: &Expr) -> Option<FW> {
        match e {
            // A suffix is the width, full stop (#473). An *unsuffixed* literal
            // is the thing being decided, so it answers nothing.
            Expr::Literal(Literal::Float(_, Some(t)), _) =>
                Some(if scalar_is_f32_family(t) { FW::F32 } else { FW::F64 }),
            Expr::Ident(name, _) => match self.lookup_ref(name)? {
                Value::Float(_, fw) => Some(*fw),
                _ => None,
            },
            Expr::Cast { ty, .. } => match self.resolve_alias_ty(ty) {
                Type::Scalar(s, _) if scalar_is_f32_family(s) => Some(FW::F32),
                Type::Scalar(ScalarType::F64, _) => Some(FW::F64),
                _ => None,
            },
            Expr::UnOp { op: UnOp::Neg, operand, .. } => self.static_float_width(operand),
            Expr::Tuple(els, _) if els.len() == 1 => self.static_float_width(&els[0]),
            Expr::Block(b) => self.static_float_width_of_block(b),
            Expr::If(ie) => self.static_float_width_of_if(ie),
            Expr::Match(me) => me.arms.iter().find_map(|a| self.static_float_width(&a.body)),
            Expr::BinOp { op: BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod,
                          lhs, rhs, .. } => {
                match (self.static_float_width(lhs), self.static_float_width(rhs)) {
                    (Some(a), Some(b)) if a != b => None,
                    (a, b) => a.or(b),
                }
            }
            // A fully-subscripted read of a tensor binding is an element of
            // that tensor, and carries the tensor's width (#473).
            Expr::Postfix { expr, op: PostfixOp::Index(elems), .. } => {
                let Expr::Ident(name, _) = &**expr else { return None };
                let Value::Tensor(t) = self.lookup_ref(name)? else { return None };
                let all_points = elems.iter().all(|ie| matches!(ie, IndexElem::Expr(_)));
                if !all_points || elems.len() != t.shape().len() { return None; }
                Some(t.float_width())
            }
            _ => None,
        }
    }

    fn static_float_width_of_block(&self, b: &Block) -> Option<FW> {
        if let Some(tail) = &b.tail_expr { return self.static_float_width(tail); }
        // #421: a trailing block-form `if`/`match` parses as a statement.
        match b.stmts.last() {
            Some(Stmt::If(ie)) => self.static_float_width_of_if(ie),
            Some(Stmt::Match(me)) => me.arms.iter().find_map(|a| self.static_float_width(&a.body)),
            _ => None,
        }
    }

    fn static_float_width_of_if(&self, ie: &IfExpr) -> Option<FW> {
        self.static_float_width_of_block(&ie.then_branch).or_else(|| match &ie.else_branch {
            Some(ElseBranch::Block(b)) => self.static_float_width_of_block(b),
            Some(ElseBranch::If(nested)) => self.static_float_width_of_if(nested),
            None => None,
        })
    }

    /// Non-cloning scope lookup, for the peek above only — `lookup` clones the
    /// whole `Value`, which for a tensor binding would make an `if` inside a
    /// hot loop copy its data just to ask what width it is.
    fn lookup_ref(&self, name: &str) -> Option<&Value> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    /// #478: the width an unsuffixed float literal in the *taken* branch of
    /// `ie` should adopt — read off the branch NOT taken, falling back to an
    /// enclosing join's answer. Gated on the taken branch actually containing
    /// such a literal, so the common `if` pays one shallow AST walk and no
    /// scope lookups at all.
    fn if_branch_hint(&self, ie: &IfExpr, then_taken: bool, inherited: Option<FW>) -> Option<FW> {
        let taken_has_lit = if then_taken {
            block_has_untyped_float_lit(&ie.then_branch)
        } else {
            else_has_untyped_float_lit(&ie.else_branch)
        };
        if !taken_has_lit { return None; }
        let other = if then_taken {
            match &ie.else_branch {
                Some(ElseBranch::Block(b)) => self.static_float_width_of_block(b),
                Some(ElseBranch::If(nested)) => self.static_float_width_of_if(nested),
                None => None,
            }
        } else {
            self.static_float_width_of_block(&ie.then_branch)
        };
        if other == Some(FW::F32) { Some(FW::F32) } else { inherited.filter(|w| *w == FW::F32) }
    }

    fn eval_if_flow(&mut self, ie: &IfExpr) -> EvalResult<Flow> {
        // #478: the hint belongs to the branch value, never to the condition.
        let inherited = self.float_lit_hint.take();
        let c = self.eval_expr(&ie.cond)?;
        if c.as_bool().unwrap_or(false) {
            self.float_lit_hint = self.if_branch_hint(ie, true, inherited);
            self.eval_block(&ie.then_branch)
        } else {
            let hint = self.if_branch_hint(ie, false, inherited);
            match &ie.else_branch {
                Some(ElseBranch::Block(b)) => { self.float_lit_hint = hint; self.eval_block(b) }
                Some(ElseBranch::If(nested)) => {
                    self.float_lit_hint = hint;
                    self.eval_if_flow(nested)
                }
                None => Ok(Flow::Normal(Value::Nil)),
            }
        }
    }

    /// Value-returning if for expression contexts. Non-Normal flow becomes
    /// an error here because break/continue/return don't make sense as values.
    fn eval_if(&mut self, ie: &IfExpr) -> EvalResult<Value> {
        match self.eval_if_flow(ie)? {
            Flow::Normal(v) | Flow::Return(v, _) => Ok(v),
            Flow::Break(sp)    => Err(RuntimeError::at("`break` outside loop".to_string(), sp)),
            Flow::Continue(sp) => Err(RuntimeError::at("`continue` outside loop".to_string(), sp)),
        }
    }

    // ── Expr ──────────────────────────────────────────────────────────────

    /// #674: the single
    /// expression evaluator, wrapped in a liveness check total over
    /// expression position. `eval_expr_lit_charge_suppressed`,
    /// `eval_expr_tolerant` and `@grad`'s `eval_expr_with_tape` all delegate
    /// to THIS fn for every value-producing form — there is no second
    /// expression evaluator — so a `budget-stale` tensor is caught here
    /// before it can reach any of the container/postfix routes the hole
    /// table (§3.2) closes "by construction": the routes all bottom out in
    /// expression position, so this one check covers every one of them.
    ///
    /// **PR #821 rework — the shape of the wrapper is the price, not the
    /// predicate.** Spelled as §3.2's edit-list item 4 writes it —
    /// `let v = self.eval_expr_inner(e)?; self.check_live(&v, e.span_of())?;
    /// Ok(v)` — this cost **+34 % to +66 %** wall-clock on `dmc run`, ~450 ×
    /// the +0.028 ns/node the table prices. None of it was the compare.
    /// Three things were, and this spelling removes all three:
    ///
    ///   1. **A frame.** An out-of-line `eval_expr` between every caller and
    ///      `eval_expr_inner` is a call, a prologue and an epilogue per node.
    ///      `#[inline(always)]` puts the wrapper back inside its caller; the
    ///      body it guards, `eval_expr_inner`, stays out of line exactly as
    ///      before, so nothing else about code size moves.
    ///   2. **Two moves of a `Value`.** `let v = …?` moves the returned
    ///      `Value` out of the `Result`, and `Ok(v)` moves it back in — at
    ///      128 B a piece, twice per node. Reading the value *through* the
    ///      `Result` by reference and returning `r` itself is a check that
    ///      moves nothing.
    ///   3. **A `Span` per node.** `expr.span_of()` is a 19-arm match plus a
    ///      32 B copy, and the old spelling ran it on **every** node to build
    ///      an argument only the error path ever reads. It now runs only on
    ///      the stale path, inside `stale_error`.
    ///
    /// What is left in the hot path is one discriminant test against
    /// `Value::Tensor` and, for a tensor, one null test on `origin` — the
    /// `nontensor` and `root` arms `gencheck.rs` prices at +0.028 and
    /// +0.041 ns. The `budget-stale` construction itself is `#[cold]` and
    /// out of line.
    #[inline(always)]
    fn eval_expr(&mut self, expr: &Expr) -> EvalResult<Value> {
        let r = self.eval_expr_inner(expr);
        if let Ok(Value::Tensor(t)) = &r {
            if tensor_is_stale(t) {
                return Err(stale_error(expr.span_of()));
            }
        }
        r
    }

    /// #674 slice 3b: `budget-stale` iff `v` is a tensor whose stamped
    /// generation no longer matches its origin budget's live generation
    /// (`tensor_is_stale`) — the check `eval_expr`'s wrapper runs at every
    /// expression, and the assignment family / stream-append slot reads run
    /// directly, since neither passes through expression position.
    ///
    /// Unlike `eval_expr`'s own inlined form above, this one takes the `Span`
    /// eagerly: its callers run once per statement, per render or per
    /// backward pass, never once per node, so the 19-arm `span_of` match is
    /// not on any hot path here.
    fn check_live(&self, v: &Value, span: Span) -> EvalResult<()> {
        if let Value::Tensor(t) = v {
            self.check_tensor_live(t, span)?;
        }
        Ok(())
    }

    /// #674 slice 3b: `check_live`'s predicate, callable directly on a
    /// `&TensorVal` — the assignment family and the two stream-append slot
    /// reads (§3.2 hole-table rows 1, 13) hold a `TensorVal` without first
    /// wrapping it in a `Value`, so they call this instead of `check_live`.
    fn check_tensor_live(&self, t: &TensorVal, span: Span) -> EvalResult<()> {
        if tensor_is_stale(t) {
            return Err(stale_error(span));
        }
        Ok(())
    }

    fn eval_expr_inner(&mut self, expr: &Expr) -> EvalResult<Value> {
        // #505: the `@comptime` step budget. Armed only by `eval_comptime`, so
        // `dmc run` pays one predictable branch on a `None` field and nothing
        // else. A `@comptime` body may loop, so this is
        // what turns a non-terminating fold into a failed compile instead of a
        // hung one. The message never reaches a user: `eval_comptime` sees the
        // exhausted counter and reports `comptime-budget` against the block.
        if let Some(fuel) = self.comptime_fuel.as_mut() {
            if *fuel == 0 {
                return Err(RuntimeError::msg("comptime step budget exhausted"));
            }
            *fuel -= 1;
        }
        // #478: consume the type-direction hint here, once. Forms that are
        // *value-transparent* — they yield one of their sub-expressions
        // unchanged — re-arm it before recursing; everything else drops it, so
        // it can only ever reach a literal that really is the value of the
        // branch position it was armed for. Mirrors `jit::Translator::lower_expr`.
        let want = self.float_lit_hint.take();
        match expr {
            Expr::Literal(lit, _) => Ok(lit_value_hinted(lit, want)),
            Expr::Nil(_)          => Ok(Value::Nil),
            Expr::Underscore(_)   => {
                // In @pp stage bodies `_` refers to the previous stage's output.
                // Look it up in scope; fall back to Nil for other uses (pipe placeholders).
                Ok(self.lookup("_").unwrap_or(Value::Nil))
            }
            Expr::Spread(_)       => Ok(Value::Nil),
            Expr::Ident(name, span) => self.eval_ident(name, span.clone()),
            Expr::Tuple(elems, _) => {
                if elems.len() == 1 {
                    self.float_lit_hint = want;
                    return self.eval_expr(&elems[0]);
                }
                let vs: Result<Vec<_>, _> = elems.iter().map(|e| self.eval_expr(e)).collect();
                Ok(Value::Tuple(vs?))
            }
            Expr::TensorLit(elems, lit_span) => {
                if elems.is_empty() {
                    self.prof_alloc();
                    return Ok(Value::tensor(ArrayD::zeros(IxDyn(&[0]))));
                }
                // #781: suppress charging while evaluating `elems[0]` — if it
                // turns out to be a tensor (the `Value::Tensor(first_row)` arm
                // below), it is a ROW of this literal, not an independent
                // materialization, and only the assembled outer array should
                // be charged (see `lit_charge_suppressed`'s doc). Harmless for
                // the `Int`/`Float` arms: a scalar element never reaches
                // `charge_intermediate`.
                let first = self.eval_expr_lit_charge_suppressed(&elems[0])?;
                let result = match first {
                    Value::Tensor(first_row) => {
                        // 2D+ nested tensor literal: stack rows along a new leading axis.
                        let row_dtype = first_row.dtype;
                        let row_shape = first_row.shape().to_vec();
                        let row_elems: usize = row_shape.iter().product();
                        let n = elems.len();
                        let mut data: Vec<f64> = Vec::with_capacity(n * row_elems);
                        let first_std = first_row.as_standard_layout();
                        data.extend_from_slice(first_std.as_slice().unwrap_or(&[]));
                        for e in elems.iter().skip(1) {
                            // #781: same suppression as `elems[0]` above — a
                            // later row is not an independent materialization
                            // either.
                            let v = self.eval_expr_lit_charge_suppressed(e)?;
                            match v {
                                Value::Tensor(row) => {
                                    // Every row must match the first row's shape;
                                    // a ragged literal (e.g. `[[1,2],[3]]`) is a
                                    // located error, not a leaked ndarray string.
                                    if row.shape() != row_shape.as_slice() {
                                        return Err(RuntimeError::at(format!(
                                            "tensor literal: rows have mismatched shapes \
                                             (row 0 is {:?}, a later row is {:?}) — every row \
                                             must have the same shape",
                                            row_shape, row.shape()), lit_span.clone()));
                                    }
                                    let row_std = row.as_standard_layout();
                                    data.extend_from_slice(row_std.as_slice().unwrap_or(&[]));
                                }
                                other => return Err(RuntimeError::at(
                                    format!("tensor literal: expected tensor row, got {}", other.type_name()),
                                    lit_span.clone(),
                                )),
                            }
                        }
                        let mut full_shape = vec![n];
                        full_shape.extend_from_slice(&row_shape);
                        let arr = ArrayD::from_shape_vec(IxDyn(&full_shape), data)
                            .map_err(|e| RuntimeError::at(format!("tensor literal: {}", e), lit_span.clone()))?;
                        // Nested literal inherits its rows' dtype.
                        Ok(Value::tensor_dt(arr, row_dtype))
                    }
                    Value::Int(f0, _) => {
                        let mut data = vec![f0 as f64];
                        for e in elems.iter().skip(1) {
                            let v = self.eval_expr(e)?;
                            data.push(v.as_float().ok_or_else(|| RuntimeError::msg(
                                format!("tensor literal element must be numeric, got {}", v.type_name())
                            ))?);
                        }
                        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data)
                            .map_err(|e| RuntimeError::msg(format!("tensor literal: {}", e)))?;
                        // An all-integer literal is an integer tensor (#125).
                        // The literal names no width, so its elements are the
                        // 64-bit default of the integer family (§3.1, #571).
                        Ok(Value::tensor_dt(arr, DType::Int(IW::I64)))
                    }
                    Value::Float(f0, _) => {
                        let mut data = vec![f0];
                        for e in elems.iter().skip(1) {
                            let v = self.eval_expr(e)?;
                            data.push(v.as_float().ok_or_else(|| RuntimeError::msg(
                                format!("tensor literal element must be numeric, got {}", v.type_name())
                            ))?);
                        }
                        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data)
                            .map_err(|e| RuntimeError::msg(format!("tensor literal: {}", e)))?;
                        Ok(Value::tensor(arr))
                    }
                    other => Err(RuntimeError::msg(
                        format!("tensor literal element must be numeric, got {}", other.type_name())
                    )),
                };
                if let Ok(v) = &result {
                    // #781: a tensor literal is an implicit intermediate — no
                    // `vault.`/`forge.` spelling names an arena for it — so it
                    // charges Forge at its DECLARED element width. Charged
                    // only for the OUTERMOST literal: a nested literal's rows
                    // reach this same arm through the suppressed evals above,
                    // so `charge_intermediate` is a no-op for them, and only
                    // the assembled `[[..]×n]` array (or a shape-argument
                    // list evaluated through `eval_expr_lit_charge_suppressed`
                    // by its caller) is skipped — the ordinary top-level call
                    // still charges once, here.
                    self.charge_intermediate(v, Some(lit_span))?;
                    self.prof_alloc();
                }
                result
            }
            Expr::Block(b) => {
                self.float_lit_hint = want;
                // #639: a block is not a function boundary and not a loop
                // boundary. Its VALUE is its tail — but a `return` / `break` /
                // `continue` inside it belongs to the enclosing fn or loop, so
                // it leaves on the carrier instead of collapsing into the
                // block's value (`return`) or into nil (`break`/`continue`),
                // which is what the JIT has always done.
                self.eval_block(b)?.into_value_or_escape()
                    .map_err(RuntimeError::escaping_flow)
            }
            Expr::ArenaBlock(ab) => {
                self.float_lit_hint = want;
                // #400: the meter does **not** rewind at the end of a
                // `forge { … }` block. The interpreter has no bump pointer to
                // reset — every tensor is an ordinary Rust allocation kept
                // alive by whatever binding holds it, and the block's value
                // escapes as the block's result. Rewinding `forge_used` here
                // made `forge { … }` a one-token bypass of `--forge`: wrapping
                // each allocation in a block let a recursive fn hold 80 MiB of
                // live tensors under `--forge=4M`, while the same program
                // unwrapped was correctly refused. The JIT takes the same line
                // — `ForgeArena::restore` rewinds the cursor but never returns
                // committed bytes to the budget — so metering is monotonic on
                // both backends and the flag means what it says.
                //
                // #639: an arena block is a block, so control flow leaves it
                // the same way — on the carrier, to the enclosing fn or loop.
                self.eval_block(&ab.body)?.into_value_or_escape()
                    .map_err(RuntimeError::escaping_flow)
            }
            // #674: `forge.
            // enter(b) { … }` pushes `b` as the entered budget and pops it on
            // EVERY exit edge — normal exit, `return`, `break`, `continue`
            // and a `?` early return. Nothing is reaped; `b` outlives the
            // block.
            //
            // The push/pop is a `BudgetGuard`, not a restore line after the
            // body, and the trap that forces it is one arm up: `Expr::
            // ArenaBlock` writes `self.eval_block(&ab.body)?`, so a restore
            // written after that `?` would be skipped by the error path AND
            // by all three carriers (a `return`/`break`/`continue` leaves a
            // block expression as `Err(escaping_flow)`, #639) — the register
            // would leak into the rest of the program and mis-charge it with
            // no diagnostic. The guard drops on all five, and restores the
            // OUTER budget rather than the root, so `enter` nests.
            Expr::BudgetBlock(bb) => {
                let operand = self.eval_expr(&bb.operand)?;
                let _guard = self.open_budget(&operand, &bb.operand.span_of(), "forge.enter")?;
                self.float_lit_hint = want;
                // #639, exactly as `Expr::ArenaBlock` above: a budget block
                // is a block, so control flow leaves it on the carrier.
                self.eval_block(&bb.body)?.into_value_or_escape()
                    .map_err(RuntimeError::escaping_flow)
            }
            // #674 slice 3d: `<ctor> in b` overrides the entered budget for
            // exactly one constructor (B3). The same guard as `enter`, held
            // across the constructor's whole evaluation, so BOTH halves land
            // in `b`: PRE-A (`charge_arena`'s Forge arm reads `cur_budget()`)
            // and the stamp (`TensorVal::new` reads `CUR_BUDGET_TAG`). The
            // budget operand is evaluated FIRST and outside the guard — it is
            // not part of the allocation it names.
            Expr::AllocIn { expr, budget, .. } => {
                let b = self.eval_expr(budget)?;
                let _guard = self.open_budget(&b, &budget.span_of(), "`in`")?;
                self.float_lit_hint = want;
                self.eval_expr(expr)
            }
            Expr::If(ie) => { self.float_lit_hint = want; self.eval_if(ie) }
            Expr::Match(me) => { self.float_lit_hint = want; self.eval_match(me) }
            Expr::FnLit(lit) => {
                // Snapshot all locals visible at this point (outer → inner so
                // inner bindings win on duplicate names, matching lookup order).
                let mut captured_env: HashMap<String, Value> = HashMap::new();
                for scope in &self.scopes {
                    for (k, v) in scope {
                        captured_env.insert(k.clone(), v.clone());
                    }
                }
                Ok(Value::Lambda {
                    lit: std::sync::Arc::new((**lit).clone()),
                    captured_env,
                    module: self.cur_module,
                })
            }
            Expr::StructLit { name, type_args, fields, .. } => {
                // The declaring file's key for this model — what method
                // dispatch, field types and shape params are registered under.
                // An unregistered name keeps its spelling: a plain struct.
                let name = &self.resolve_model(name).unwrap_or_else(|| name.clone());
                let mut field_vals: Vec<(String, Value)> = Vec::with_capacity(fields.len() + 1);
                // Store model name as a hidden tag so method dispatch can find it.
                field_vals.push(("__model__".to_string(), Value::Str(name.clone())));
                // Store shape param bindings (e.g. P=5) so methods can use them at runtime.
                if let Some(param_names) = self.model_shape_params.get(name).cloned() {
                    for (pname, arg_expr) in param_names.iter().zip(type_args.iter()) {
                        // #639: a shape argument that does not evaluate is
                        // simply not recorded — but the block-flow carrier is
                        // not "does not evaluate", it is the enclosing fn's
                        // control flow passing through, and it keeps going.
                        if let Some(v) = self.eval_expr_tolerant(arg_expr)? {
                            field_vals.push((format!("__shape_{}__", pname), v));
                        }
                    }
                }
                // #473: a declared scalar field type binds the stored value's
                // float width, exactly like a typed `let` or parameter. The JIT
                // gives an `f32` field 4-byte storage, so `M { x: 0.1 }` reads
                // back as the f32 0.100000001490116 there; without this the
                // interpreter would hand back the full-width f64 and the two
                // backends would part company on the next operation.
                let field_tys = self.model_fields.get(name).cloned();
                for (fname, fexpr) in fields {
                    let v = self.eval_expr(fexpr)?;
                    let declared = field_tys.as_ref()
                        .and_then(|fs| fs.iter().find(|(n, _)| n == fname))
                        .map(|(_, t)| t);
                    field_vals.push((fname.clone(), self.coerce_declared(v, declared)));
                }
                Ok(Value::Struct(Rc::new(RefCell::new(field_vals))))
            }
            Expr::DirectiveBlock { directives, body, .. } => {
                self.float_lit_hint = want;
                // #639: control flow leaves a directive block the way it leaves
                // any other block — on the carrier, unchanged by `@cast`.
                let v = self.eval_block(body)?.into_value_or_escape()
                    .map_err(RuntimeError::escaping_flow)?;
                // Apply @cast directive to the block result if present.
                if let Some(cast_dir) = directives.iter().find(|d| d.name == "cast") {
                    if let Some(crate::ast::DArg::Positional(type_expr)) = cast_dir.args.first() {
                        let target = match type_expr {
                            Expr::Ident(name, _) => Some(name.as_str()),
                            _ => None,
                        };
                        if let Some(ty_name) = target {
                            if self.is_overlay_model(ty_name) {
                                return self.cast_to_struct(&v, ty_name);
                            }
                            return Ok(apply_cast_by_name(&v, ty_name));
                        }
                    }
                }
                Ok(v)
            }
            Expr::BinOp { op, lhs, rhs, span } => {
                // `x |> f` and `x >> f` both call f(x).  The RHS must be a callable;
                // we dispatch through call_value so Fn, BoundFn, and Builtin all work.
                if matches!(op, BinOp::Pipe) {
                    let arg = self.eval_expr(lhs)?;
                    // Placeholder-fusion form: `x |> _ .+ b` — the RHS is a stage
                    // expression that uses `_` for the piped value, not a callable.
                    // Bind `_` to the piped value and evaluate the stage directly.
                    if expr_contains_underscore(rhs) {
                        self.push_scope();
                        self.bind("_", arg);
                        let out = self.eval_expr(rhs);
                        self.pop_scope();
                        return out;
                    }
                    let callee = self.eval_expr(rhs)?;
                    return self.call_value(callee, vec![arg], span.clone());
                }
                // #285: `&&` / `||` short-circuit (OPERATORS.md §11) — the RHS
                // must not be evaluated when the LHS decides. Matches the JIT
                // (`lower_short_circuit`). Truthiness mirrors apply_binop:
                // non-bool operands coerce via as_bool().unwrap_or(false).
                if matches!(op, BinOp::And | BinOp::Or) {
                    let l = self.eval_expr(lhs)?.as_bool().unwrap_or(false);
                    let result = match op {
                        BinOp::And if !l => false,
                        BinOp::Or if l => true,
                        _ => self.eval_expr(rhs)?.as_bool().unwrap_or(false),
                    };
                    return Ok(Value::Bool(result));
                }
                let l = self.eval_expr(lhs)?;
                let r = self.eval_expr(rhs)?;
                let result = apply_binop(op.clone(), &l, &r)?;
                // #781: a binop that produces a NEW tensor (matmul, a dotted
                // elementwise op) is an implicit intermediate — `let C = A @
                // B` names no `vault.`/`forge.` arena — so it charges Forge
                // at its declared element width, the same as any other
                // materialized tensor.
                self.charge_intermediate(&result, Some(span))?;
                // Profile: count tensor ops vs scalar ops.
                match op {
                    BinOp::Matmul | BinOp::DotAdd | BinOp::DotSub | BinOp::DotMul | BinOp::DotDiv
                    | BinOp::DotPow | BinOp::DotGt | BinOp::DotLt
                    | BinOp::DotGe | BinOp::DotLe => {
                        let elems = match &result {
                            Value::Tensor(t) => t.len() as u64,
                            _ => 1,
                        };
                        self.prof_tensor_op(elems);
                    }
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod
                    | BinOp::Pow | BinOp::StarStar => {
                        match (&l, &r) {
                            (Value::Tensor(_), _) | (_, Value::Tensor(_)) => {
                                let elems = match &result {
                                    Value::Tensor(t) => t.len() as u64,
                                    _ => 1,
                                };
                                self.prof_tensor_op(elems);
                            }
                            (Value::Int(_, _), _) | (Value::Float(_, _), _)
                            | (_, Value::Int(_, _)) | (_, Value::Float(_, _)) => {
                                self.prof_scalar_op();
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
                Ok(result)
            }
            Expr::UnOp { op, operand, .. } => {
                // `-0.1` is a negated literal, so the hint passes through the sign.
                if matches!(op, UnOp::Neg) { self.float_lit_hint = want; }
                let v = self.eval_expr(operand)?;
                apply_unop(op.clone(), &v)
            }
            Expr::Postfix { expr, op, span } => self.eval_postfix(expr, op, span.clone()),
            Expr::Cast { expr, ty, span } => {
                let v = self.eval_expr(expr)?;
                // #336: `Color.Red as i64` (or any numeric) → the variant ordinal.
                // Resolve to an Int here, where the enum registry is in scope,
                // then let the generic cast widen/narrow it.
                // An `as` cast is a width origin (SPEC §3.1), and an alias
                // (§6.6) is transparent: `300 as Role` where `type Role = u8`
                // wraps to 44 exactly as `300 as u8` does.
                let ty = self.resolve_alias_ty(ty);
                if let Value::EnumVal { enum_name, variant, .. } = &v {
                    let ord = self.enum_variant_ordinal(enum_name, variant).unwrap_or(0);
                    return Ok(apply_cast(&Value::Int(ord, IW::I64), ty));
                }
                let out = apply_cast(&v, ty);
                // #781: a tensor `as` cast materializes a new element array
                // at the target width (`apply_cast`'s `Value::Tensor` arm
                // always allocates — `cast_tensor_to_int_width` or
                // `t.data.clone()`) — an implicit intermediate like any
                // other. Uncharged on the JIT too today; it refuses tensor
                // `as` casts outright, so there is nothing to stay in parity
                // with yet — this charges ahead of the JIT lowering it.
                self.charge_intermediate(&out, Some(&span))?;
                Ok(out)
            }
            Expr::Range { start, end, inclusive, .. } => {
                // A bound that does not evaluate to an int reads as 0 — the
                // range is still a well-formed value. #639: that tolerance is
                // for *values*, and must not eat the block-flow carrier, or a
                // `return` written inside a bound would silently become a 0
                // bound instead of leaving the enclosing function.
                let s = self.eval_range_bound(start.as_deref())?;
                let e = self.eval_range_bound(end.as_deref())?;
                Ok(Value::Range { start: s, end: e, inclusive: *inclusive })
            }
        }
    }

    /// Evaluate `e` in one of the few positions that TOLERATE an ordinary
    /// failure and carry on: a range bound that is not an integer, a model
    /// shape argument that does not evaluate, a `.split` piece count that has
    /// its own diagnostic. `None` is "this expression did not produce a
    /// value"; the caller supplies its own fallback or message.
    ///
    /// #639: the block-flow carrier is the one failure that is never absorbed.
    /// It is not an error *about this expression* — it is the enclosing
    /// function's or loop's control flow passing through — so swallowing it
    /// either loses a `return` silently or, worse, reports it as this
    /// position's own complaint (`.split` blamed a bad piece count for a
    /// `return` that should have left the function).
    fn eval_expr_tolerant(&mut self, e: &Expr) -> EvalResult<Option<Value>> {
        match self.eval_expr(e) {
            Ok(v) => Ok(Some(v)),
            Err(err) if err.flow.is_some() => Err(err),
            Err(_) => Ok(None),
        }
    }

    /// One bound of a `a..b` range: its integer value, or 0 for an absent or
    /// non-integer bound (see `eval_expr_tolerant` for what is not absorbed).
    fn eval_range_bound(&mut self, bound: Option<&Expr>) -> EvalResult<i64> {
        let Some(e) = bound else { return Ok(0) };
        Ok(self.eval_expr_tolerant(e)?.and_then(|v| v.as_int()).unwrap_or(0))
    }

    fn eval_ident(&mut self, name: &str, span: Span) -> EvalResult<Value> {
        if let Some(v) = self.lookup(name) {
            // Transparently dereference FieldRef for reads.
            if let Value::FieldRef { ref rc, ref field } = v {
                let borrowed = rc.borrow();
                return match borrowed.iter().find(|(k, _)| k == field) {
                    Some((_, fval)) => Ok(fval.clone()),
                    None => Err(RuntimeError::at(format!("undefined field `{}`", field), span)),
                };
            }
            return Ok(v);
        }
        if let Some(key) = self.resolve_fn(name) { return Ok(Value::Fn(key)); }
        if self.extern_fns.contains(name) { return Ok(Value::Fn(name.to_string())); }
        // `pi` is a well-known math constant
        if name == "pi"  { return Ok(Value::Float(std::f64::consts::PI, FW::F64)); }
        if name == "tau" { return Ok(Value::Float(std::f64::consts::TAU, FW::F64)); }
        if name == "e"   { return Ok(Value::Float(std::f64::consts::E, FW::F64)); }
        if name == "inf" { return Ok(Value::Float(f64::INFINITY, FW::F64)); }
        if name == "nan" { return Ok(Value::Float(f64::NAN, FW::F64)); }
        if let Some(key) = self.resolve_model(name) {
            // Model names appear in `Transformer[L=24, ...].load(...)` patterns;
            // the interpreter doesn't realize methods, so it's opaque-but-known.
            return Ok(Value::Opaque(key));
        }
        if is_builtin(name) { return Ok(Value::Builtin(name.to_string())); }
        // Type-level / arena / mesh-axis names — opaque at runtime.
        if matches!(name,
            "vault" | "forge" | "stream" | "~" | "self" |
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" |
            "int4" | "int8" | "f16" | "bf16" | "tf32" | "f32" | "f64" |
            "fp8_e4m3" | "fp8_e5m2" | "bool" | "str" | "nil" |
            "Tensor" | "View" | "KV" | "Mesh" | "Rng" | "Weights" |
            "dp" | "tp" | "pp" | "ep" | "sp" |
            "\\>" | "\\<"
        ) {
            return Ok(Value::Opaque(name.to_string()));
        }
        Err(RuntimeError::at(format!("undefined identifier `{}`", name), span))
    }

    fn eval_postfix(&mut self, expr: &Expr, op: &PostfixOp, span: Span) -> EvalResult<Value> {
        match op {
            PostfixOp::Call(args) => {
                // #452: claim the enclosing statement's move hint before any
                // argument is evaluated. The hint is set only for a call that
                // *is* the whole right-hand side, and that call is the first
                // one entered, so taking it here hands it to its rightful owner
                // and leaves nested calls with `None`.
                let move_hint = self.move_hint.take();
                // Evaluate args first (call-by-value, left-to-right).
                let mut arg_vals = Vec::with_capacity(args.len());
                for a in args {
                    match a {
                        CallArg::Positional(e) => arg_vals.push(self.eval_expr(e)?),
                        CallArg::Named { value, .. } => arg_vals.push(self.eval_expr(value)?),
                        CallArg::Spread(_) => {}
                    }
                }
                // #395: `rng.uniform_int[T, [shape]](low, high)` — the integer
                // draw. Its range args live on this outer call, so the bracket
                // ctor path (`try_arena_constructor`) can't handle it.
                if let Expr::Postfix { expr: idx_inner, op: bracket_op, .. } = expr {
                    let bracket_args: Option<Vec<&Expr>> = match bracket_op {
                        PostfixOp::Index(elems) => Some(elems.iter().filter_map(|e| match e {
                            IndexElem::Expr(e) => Some(e),
                            _ => None,
                        }).collect()),
                        PostfixOp::BracketArgs(bargs) => Some(bargs.iter().filter_map(|a| match a {
                            CallArg::Positional(e) => Some(e),
                            _ => None,
                        }).collect()),
                        _ => None,
                    };
                    if let Some(bargs) = bracket_args {
                        if let Expr::Postfix { expr: rng_expr, op: PostfixOp::Field(m), .. } = idx_inner.as_ref() {
                            if m == "uniform_int" {
                                return self.eval_rng_uniform_int(rng_expr, &bargs, &arg_vals, span);
                            }
                        }
                    }
                }
                // #350 Part 2: `Shape.Circle(args)` — payload-variant
                // construction. Intercept before the method-call machinery
                // tries to `eval_expr` the enum name as a value.
                if let Expr::Postfix { expr: inner_expr, op: PostfixOp::Field(variant), .. } = expr {
                    if let Expr::Ident(en, _) = inner_expr.as_ref() {
                        if self.enum_variant_ordinal(en, variant).is_some() {
                            return Ok(Value::EnumVal {
                                enum_name: en.clone(),
                                variant: variant.clone(),
                                payload: arg_vals,
                            });
                        }
                    }
                }
                // #474 — the ghost method. `b.blit![2, 2](src)` parses as
                // Call(Index(Field(b, "blit!"), [2, 2]), [src]): the shape
                // bracket sits between the method name and the call, so the
                // Field-shaped dispatch just below never sees a method. The
                // callee fell through to `Value::Opaque("index")`, and calling
                // an opaque yields another opaque — a defined, `--check`-clean
                // method that did nothing whatsoever, no error and no effect.
                // That is #441's ghost-method mode surviving in the shape-
                // parameterized case, and a test asserting the call did
                // nothing could not tell the difference.
                //
                // Dispatch it here. Both halves of the body's shape scope are
                // reachable: the receiver carries the *model's* shape args as
                // hidden `__shape_P__` fields, and the bracket names the
                // method's *own*.
                if let Expr::Postfix { expr: bracketed, op: bracket_op, .. } = expr {
                    if let Some(bracket) = shape_bracket_args(bracket_op, &span) {
                        if let Expr::Postfix { expr: recv_expr, op: PostfixOp::Field(method), .. } =
                            bracketed.as_ref()
                        {
                            match self.call_bracketed_method(
                                recv_expr, method, &bracket, &arg_vals, &span)?
                            {
                                BracketedCall::Dispatched(v) => {
                                    // #671: a method dispatched through a shape
                                    // bracket (`b.fillg![2](o2)`) used to return
                                    // here without the write-back step the plain
                                    // method-call and forward-desugar paths both
                                    // perform (#475). The receiver occupies
                                    // callee param 0 with no call-arg slot, same
                                    // offset (`self_params = 1`) as those paths.
                                    apply_writebacks(
                                        &mut self.scopes,
                                        &mut self.pending_writebacks,
                                        args,
                                        1,
                                    );
                                    return Ok(v);
                                }
                                // Not a method after all. The receiver is
                                // already evaluated; hand it to the ordinary
                                // path below instead of running it again.
                                BracketedCall::NotAMethod(recv) =>
                                    self.park_receiver(recv_expr, recv),
                            }
                        }
                    }
                }
                // String method calls: s.split(","), s.trim(), s.upper(), etc.
                // Also handles model instance method calls: m.method(args).
                if let Expr::Postfix { expr: inner_expr, op: PostfixOp::Field(method), .. } = expr {
                    let recv = self.eval_expr(inner_expr)?;
                    if let Value::Str(s) = &recv {
                        let v = call_str_method(s, method, arg_vals, span.clone())?;
                        // #804 review (PR #815): `.trim`/`.upper`/`.lower`/
                        // `.replace` materialize a fresh string and `.split`/
                        // `.lines` a fresh list of them — this call sat
                        // outside `call_builtin` and so missed #802's
                        // choke point entirely (0 B charged on `dmc run` for
                        // any of them; the JIT refuses all but
                        // `.starts_with`, so there was no parity to break,
                        // only a real materialization going uncounted).
                        // `charge_builtin_result` already recurses into a
                        // `List` result, so `.split`/`.lines` charge every
                        // piece, not just the list itself.
                        self.charge_builtin_result(&v, &span)?;
                        return Ok(v);
                    }
                    // `nil` has no methods. This used to fall through to the
                    // forward-compat "unknown receiver" path and hand back
                    // `Value::Opaque(".starts_with(...)")`, which is falsy — so
                    // the PORTS.md §6 idiom written without its nil guard,
                    // `e.starts_with("port-open")` on a no-error `Err`, quietly
                    // answered "no" instead of saying the receiver was nil. The
                    // JIT raises here (a null str pointer is `nil`); raise the
                    // same error, with the same wording, rather than diverge.
                    if matches!(recv, Value::Nil) {
                        return Err(RuntimeError::at(
                            format!("cannot call method `{}` on nil", method), span));
                    }
                    // A `Port` handle has no methods either. PORTS.md §7.1: a
                    // handle "compares only against `nil`, … and is otherwise
                    // inert: it has no length, no elements, and no equality".
                    // Without this it took the same forward-compat path and
                    // `p.starts_with("port#")` on a *live* handle answered with
                    // a falsy `Opaque`, rc 0 — the handle-forging check the
                    // opacity exists to make impossible, quietly reported as
                    // "no". The JIT refuses this at compile time; `run` cannot
                    // (the receiver's type is not known until it is evaluated),
                    // so it raises here, located, rather than answering.
                    if let Value::Opaque(ref o) = recv {
                        if crate::ports::handle_id(o).is_some() {
                            return Err(RuntimeError::at(format!(
                                "cannot call method `{}` on a Port handle — a handle is \
                                 opaque and has no methods (PORTS §7.1)", method), span));
                        }
                    }
                    // Model instance method dispatch: m.other_method(args).
                    // The forward-desugar path (m() → m.forward()) is handled later via
                    // Value::Struct in the callee match. But m.method(args) for any named
                    // method must be intercepted here before eval_expr produces Opaque.
                    if let Value::Struct(ref fields) = recv {
                        let model_name = {
                            let borrowed = fields.borrow();
                            borrowed.iter()
                                .find(|(k, _)| k == "__model__")
                                .and_then(|(_, v)| if let Value::Str(s) = v { Some(s.clone()) } else { None })
                        };
                        if let Some(mname) = model_name {
                            if let Some(f) = self.model_methods.get(&mname)
                                .and_then(|ms| ms.get(method.as_str())).cloned()
                            {
                                let shape_bindings: Vec<(String, i64)> = {
                                    let borrowed = fields.borrow();
                                    borrowed.iter()
                                        .filter_map(|(k, v)| {
                                            k.strip_prefix("__shape_")
                                                .and_then(|s| s.strip_suffix("__"))
                                                .and_then(|pname| v.as_int().map(|n| (pname.to_string(), n)))
                                        })
                                        .collect()
                                };
                                let mut call_args = vec![recv.clone()];
                                call_args.extend(arg_vals);
                                let result = self.call_fn_with_shapes(&f, call_args, &shape_bindings, span)?;
                                // #475: `!` params on a method alias exactly as
                                // they do on a free function. `self` occupies
                                // callee param 0 and has no call-arg slot, hence
                                // the offset of 1.
                                apply_writebacks(&mut self.scopes, &mut self.pending_writebacks, args, 1);
                                return Ok(result);
                            }
                            // #394: model serialization `instance.save(...)` /
                            // `.load(...)` is specified (SPEC §3.10) but not
                            // implemented. With no user method of that name it
                            // used to fall through to a silent opaque (no file
                            // written, no error). Error loudly instead.
                            if matches!(method.as_str(), "save" | "load") {
                                return Err(RuntimeError::at(format!(
                                    "model serialization `{}.{}` is specified \
                                     but not yet implemented", bare_name(&mname), method), span));
                            }
                            // #441: no such method on this model. A field of
                            // that name may still hold a callable, so only
                            // error when neither exists — otherwise this fell
                            // through to a silent no-op in statement position.
                            let has_field = fields.borrow().iter()
                                .any(|(k, _)| k == method.as_str());
                            if !has_field {
                                return Err(RuntimeError::at(format!(
                                    "no method `{}` on model `{}`", method, bare_name(&mname)), span));
                            }
                        }
                    }
                }
                // #394: `Name.load(...)` (model deserialization) on a model name —
                // specified (SPEC §3.10) but not implemented. The receiver is a
                // model name (`Value::Opaque`), so without this it falls through
                // to a silent opaque. Error loudly. (`Name.save` is guarded too
                // for symmetry, though `save` is an instance operation.)
                if let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr {
                    if matches!(method.as_str(), "save" | "load") {
                        // Peel shape args so `Name[D=4].load(...)` resolves to
                        // `Name`. Both the named-arg form (`Name[D=4]`, BracketArgs)
                        // and the positional/empty form (`Name[N]`, `Name[]`, which
                        // parse as Index) must peel — otherwise `Name[].load(...)`
                        // slipped past the guard to a silent opaque (#394). The
                        // `self.models.contains` check below keeps this from firing
                        // on a genuine indexed value like `arr[i].load`.
                        let mut base = inner.as_ref();
                        while let Expr::Postfix {
                            expr: b, op: PostfixOp::BracketArgs(_) | PostfixOp::Index(_), ..
                        } = base {
                            base = b.as_ref();
                        }
                        if let Expr::Ident(name, _) = base {
                            if self.resolve_model(name).is_some() {
                                return Err(RuntimeError::at(format!(
                                    "model serialization `{}.{}` is specified \
                                     but not yet implemented", name, method), span));
                            }
                        }
                    }
                }
                // #396: `allreduce.sum(...)` etc. — a distributed collective in
                // method form (desugar deliberately leaves it un-rewritten). Left
                // to resolve generically it yields a silent opaque; the earlier
                // UFCS bug silently computed 0.0. Report a real error naming the
                // collective and the op.
                if let Expr::Postfix { expr: inner, op: PostfixOp::Field(op), .. } = expr {
                    if let Expr::Ident(coll, _) = inner.as_ref() {
                        if matches!(coll.as_str(),
                            "allreduce" | "allgather" | "reducescatter" | "broadcast")
                        {
                            return Err(RuntimeError::at(format!(
                                "distributed collective `{}.{}` is not executable here; \
                                 collectives are unimplemented on a single node", coll, op), span));
                        }
                    }
                }
                // #674: `forge.carve(...)`
                // / `forge.reap(...)` — intercepted here, before the generic
                // ghost-opaque no-op every unrecognized method call on an
                // `Opaque` receiver would otherwise fall to, the same
                // syntactic spot 3a's not-yet-implemented refusal used.
                if let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr {
                    if let Expr::Ident(fname, _) = inner.as_ref() {
                        if fname == "forge" && method == "carve" {
                            let nil = Value::Nil;
                            let parent = arg_vals.first().unwrap_or(&nil);
                            let bytes = arg_vals.get(1).unwrap_or(&nil);
                            return self.budget_carve(parent, bytes, &span);
                        }
                        if fname == "forge" && method == "reap" {
                            let nil = Value::Nil;
                            let b = arg_vals.first().unwrap_or(&nil);
                            return self.budget_reap(b, &span);
                        }
                    }
                }
                // @grad call patterns: `<fn>.fwd_bwd(args)` or `<fn>.grad(args)`.
                // Intercept before generic field-access, since `.fwd_bwd` on a
                // Value::Fn would otherwise just yield Opaque.
                if let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr {
                    if let Expr::Ident(fn_name, _) = inner.as_ref() {
                        if matches!(method.as_str(), "fwd_bwd" | "grad" | "fwd" | "fwd_bwd_bwd") {
                            if let Some(key) = self.resolve_fn(fn_name) {
                                return self.call_grad(&key, method, arg_vals, span);
                            }
                        }
                        // `Rng.seed(N)` seeds the interpreter's PRNG and returns
                        // an opaque rng handle. State is process-global rather
                        // than threaded through values; the language only uses
                        // `rng` as an opaque token at this layer.
                        if fn_name == "Rng" && method == "seed" {
                            let seed = arg_vals.first().and_then(|v| v.as_int())
                                .unwrap_or(0) as u64;
                            // SplitMix64 step on the seed so seed=0 still
                            // gives a non-zero xorshift state.
                            let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
                            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                            self.rng_state = z ^ (z >> 31);
                            return Ok(Value::Opaque("rng".into()));
                        }
                    }
                }
                // Resolve callee
                let callee = self.eval_expr(expr)?;
                match callee {
                    Value::Fn(name) => {
                        if self.extern_fns.contains(&name) {
                            return Err(RuntimeError::at(format!("extern fn `{}` cannot be called by the interpreter — a foreign call crosses an unchecked machine boundary this backend has no pointer representation for, so the spec confines `extern fn` calls to the JIT; run it with `dmc jit`", name), span));
                        }
                        let f = self.fns.get(&name).cloned().ok_or_else(|| {
                            RuntimeError::at(format!("undefined fn `{}`", bare_name(&name)), span.clone())
                        })?;
                        let result = self.call_fn(&f, arg_vals, span)?;
                        apply_writebacks(&mut self.scopes, &mut self.pending_writebacks, args, 0);
                        Ok(result)
                    }
                    Value::BoundFn { ref name, ref shape_bindings } => {
                        let f = self.fns.get(name).cloned().ok_or_else(|| {
                            RuntimeError::at(format!("undefined fn `{}`", bare_name(name)), span.clone())
                        })?;
                        let bindings = shape_bindings.clone();
                        let result = self.call_fn_with_shapes(&f, arg_vals, &bindings, span)?;
                        apply_writebacks(&mut self.scopes, &mut self.pending_writebacks, args, 0);
                        Ok(result)
                    }
                    Value::Lambda { lit, captured_env, module } =>
                        self.call_lambda(&lit.clone(), captured_env, module, arg_vals, span),
                    Value::Builtin(name) => {
                        self.release_move_target(move_hint.as_deref(), args.first(), &arg_vals);
                        self.call_builtin(&name, arg_vals, span)
                    }
                    Value::Opaque(name) => {
                        // Calling a type constructor / arena helper — pre-alpha: nil
                        Ok(Value::Opaque(format!("{}(...)", name)))
                    }
                    // `forge.kv[T, shape](capacity = N)` already produced the tensor
                    // at the BracketArgs step; the trailing call just carries
                    // allocation hints. Same for any tensor-returning chain.
                    Value::Tensor(t) => Ok(Value::Tensor(t)),
                    // Model instance call: dispatch to the model's `forward` method
                    // (Spec §6.4 forward-desugar: `m(x)` ≡ `m.forward(x)`).
                    Value::Struct(ref fields) => {
                        let model_name = {
                            let borrowed = fields.borrow();
                            borrowed.iter()
                                .find(|(k, _)| k == "__model__")
                                .and_then(|(_, v)| if let Value::Str(s) = v { Some(s.clone()) } else { None })
                        };
                        if let Some(mname) = model_name {
                            if let Some(f) = self.model_methods.get(&mname)
                                .and_then(|ms| ms.get("forward")).cloned()
                            {
                                let shape_bindings: Vec<(String, i64)> = {
                                    let borrowed = fields.borrow();
                                    borrowed.iter()
                                        .filter_map(|(k, v)| {
                                            k.strip_prefix("__shape_")
                                                .and_then(|s| s.strip_suffix("__"))
                                                .and_then(|pname| v.as_int().map(|n| (pname.to_string(), n)))
                                        })
                                        .collect()
                                };
                                let mut call_args = vec![callee.clone()];
                                call_args.extend(arg_vals);
                                let result = self.call_fn_with_shapes(&f, call_args, &shape_bindings, span)?;
                                // #475: same as the named-method path — the
                                // forward-desugar `m(x)` ≡ `m.forward(x)` puts
                                // the receiver in param 0 with no call-arg slot.
                                apply_writebacks(&mut self.scopes, &mut self.pending_writebacks, args, 1);
                                return Ok(result);
                            }
                        }
                        Err(RuntimeError::at(
                            format!("cannot call struct value (no `forward` method found)"), span,
                        ))
                    }
                    other => Err(RuntimeError::at(
                        format!("cannot call value of type {}", other.type_name()), span,
                    )),
                }
            }
            PostfixOp::Field(name) => {
                // #336: `Color.Red` — a qualified enum-variant value. Resolve
                // before evaluating the base (the enum name is not a binding).
                if let Expr::Ident(base, _) = expr {
                    if self.enum_variant_ordinal(base, name).is_some() {
                        return Ok(Value::EnumVal { enum_name: base.clone(), variant: name.clone(), payload: Vec::new() });
                    }
                }
                // #674: `forge.root`
                // is entry 0 of the budget table — a `Budget` value, so
                // `forge.carve(forge.root, n)` runs. Its generation is always
                // `budgets[0].live.get()` (root is never reaped, so that
                // cell never moves off 0), read rather than hard-coded so
                // this stays correct if that ever changes. Intercepted here,
                // syntactically, before it would otherwise fall through to
                // the generic ghost-opaque field fallback below.
                if let Expr::Ident(base, _) = expr {
                    if base == "forge" && name == "root" {
                        return Ok(Value::Budget(budget_descriptor(self.budgets[0].live.get(), 0)));
                    }
                }
                // #474: a shape-bracketed call that turned out not to be a
                // method already evaluated this very node. Reuse that value —
                // evaluating it again would repeat every side effect.
                let recv = match self.take_parked_receiver(expr) {
                    // #674 slice 3b, PR #821 review: §3.2's hole-table row 3
                    // closed this arm with "`take_parked_receiver` (4496)
                    // feeds `eval_expr`" — it does not. It hands the parked
                    // value straight to the member dispatch below, so a
                    // receiver that was live when it was parked and stale by
                    // the time it is taken would answer `.shape` / `.rank`
                    // off a reaped tensor with no diagnostic. The `None` arm
                    // is expression position and checks itself; this one has
                    // to say so.
                    Some(v) => { self.check_live(&v, expr.span_of())?; v }
                    None => self.eval_expr(expr)?,
                };
                match (&recv, name.as_str()) {
                    (Value::Module { alias, .. }, _) => {
                        let qual_name = format!("{}.{}", alias, name);
                        if let Some(key) = self.resolve_fn(&qual_name) {
                            return Ok(Value::Fn(key));
                        }
                        if let Some(v) = self.lookup(&qual_name) {
                            return Ok(v.clone());
                        }
                        if let Some(key) = self.resolve_model(&qual_name) {
                            return Ok(Value::Opaque(key));
                        }
                        Err(RuntimeError::at(format!("no member `{}` in module `{}`", name, alias), span))
                    }
                    (Value::Tensor(t), "shape") => {
                        let dims: Vec<Value> = t.shape().iter()
                            .map(|&d| Value::Int(d as i64, IW::I64)).collect();
                        Ok(Value::Tuple(dims))
                    }
                    (Value::Tensor(_), "rank" | "ndim") => Ok(Value::Int(
                        if let Value::Tensor(t) = &recv { t.ndim() as i64 } else { 0 }
                    , IW::I64)),
                    (Value::Struct(fields), _) => {
                        for (k, v) in fields.borrow().iter() {
                            if k == name { return Ok(v.clone()); }
                        }
                        Ok(Value::Opaque(format!(".{}", name)))
                    }
                    // Weight loading (raw binary and NPZ) is JIT-only; the
                    // opaque stand-in the interpreter would otherwise yield
                    // silently poisons any downstream use (e.g. an attn mask
                    // that never applies).
                    (Value::Opaque(arena), "load" | "load_npz")
                        if arena == "vault" || arena == "forge" =>
                    {
                        Err(RuntimeError::at(format!(
                            "`{}.{}` is not supported by the interpreter \
                             (`dmc run`); run this program under `dmc jit`", arena, name), span))
                    }
                    // #476: an unwritten model-array slot is `nil` (that is how
                    // `forge.uninit[Cell, [N]]` fills its list), so a field read
                    // off one used to yield a silent `Opaque`. The program then
                    // failed at whatever line first *used* the value, with a
                    // complaint about "opaque" — a word that appears nowhere in
                    // the program or the spec — naming neither the field, nor
                    // the array, nor initialization. Name the real cause here,
                    // at the read.
                    (Value::Nil, _) => Err(RuntimeError::at(format!(
                        "read of uninitialized model-array element: field `.{}` was \
                         read from an empty slot. `forge.uninit`/`vault.uninit` leaves \
                         every element unwritten — assign a whole model literal to the \
                         slot (`arr[i] = M {{ .. }}`) before reading it (MEMORY §2)",
                        name), span)),
                    _ => Ok(Value::Opaque(format!(".{}", name))),
                }
            }
            PostfixOp::Transpose => {
                let v = self.eval_expr(expr)?;
                if let Value::Tensor(t) = v {
                    let n = t.ndim();
                    if n >= 2 {
                        let mut axes: Vec<usize> = (0..n).collect();
                        axes.swap(n - 1, n - 2);
                        let out = Value::tensor_dt(t.data.clone().permuted_axes(IxDyn(&axes)), t.dtype);
                        // #781: a transpose materializes a new, permuted copy
                        // of the backing storage — an implicit intermediate.
                        self.charge_intermediate(&out, Some(&span))?;
                        return Ok(out);
                    }
                    Ok(Value::Tensor(t))
                } else {
                    Err(RuntimeError::at("transpose requires tensor".to_string(), span))
                }
            }
            PostfixOp::Query => {
                let val = self.eval_expr(expr)?;
                match val {
                    Value::Tuple(ref elems) if elems.len() == 2 => {
                        match &elems[1] {
                            // No error → unwrap to the value.
                            Value::Nil => Ok(elems[0].clone()),
                            // Error set → early-return the enclosing function
                            // with `(default[T], e)` (SPEC §4.9), where `T` is
                            // the ENCLOSING function's declared success type.
                            // Only `e` travels; the `T` half is built at the
                            // function-call boundary, which is the only place
                            // that knows the enclosing signature. The callee's
                            // own error-path value is deliberately dropped here
                            // (#628) — it is not the caller's value, and when
                            // the two `T`s differ it is not even its type.
                            e => Err(RuntimeError::propagate(e.clone(), span)),
                        }
                    }
                    other => Err(RuntimeError::at(
                        format!("`?` requires a (T, Err) tuple, got {}", other.type_name()), span,
                    )),
                }
            }
            PostfixOp::BracketArgs(args) => {
                // Tensor `.split[n, axis=k]` (SPEC §6.4): split into n equal
                // pieces along `axis`, returned as an n-tuple.
                if let Expr::Postfix { expr: recv, op: PostfixOp::Field(fname), .. } = expr {
                    if fname == "split" {
                        return self.eval_tensor_split(recv, args, span);
                    }
                }
                // Try the <X>.<method>[T, shape] constructor pattern first
                // (named-args form, e.g. `Transformer[L=24, D=2048]`).
                if let Some(v) = self.try_arena_constructor(
                    expr,
                    args.iter().filter_map(|a| match a {
                        CallArg::Positional(e) => Some(e),
                        _ => None,
                    }),
                )? {
                    return Ok(v);
                }
                // `f[N]` / `f[S=4]` with user fn: bind explicit shape params.
                // Positional args bind in declaration order; named args bind
                // by name. A binding that can't be made errors loudly here —
                // falling through to the opaque stand-in would silently poison
                // the trailing call (the `<opaque bracket(...)>` failure mode).
                if let Expr::Ident(fn_name, _) = expr {
                    if let Some(f) = self.resolve_fn(fn_name).map(|k| self.fns[&k].clone()) {
                        if !f.shape_params.is_empty() {
                            // A spread is not a shape argument; the pre-#474
                            // loop skipped it and this keeps that.
                            let bracket: Vec<(Option<&str>, &Expr, Span)> = args.iter()
                                .filter_map(|a| match a {
                                    CallArg::Positional(e) => Some((None, e, span.clone())),
                                    CallArg::Named { name, value, span: nspan } =>
                                        Some((Some(name.as_str()), value, nspan.clone())),
                                    CallArg::Spread(_) => None,
                                })
                                .collect();
                            let shape_bindings =
                                self.bind_shape_bracket(fn_name, &f.shape_params, &bracket, &span)?;
                            if !shape_bindings.is_empty() {
                                let name = module_key(f.module, &f.name);
                                return Ok(Value::BoundFn { name, shape_bindings });
                            }
                        }
                    }
                }
                let _ = self.eval_expr(expr)?;
                Ok(Value::Opaque("bracket".into()))
            }
            PostfixOp::Constructor(fields) => {
                let mut evaluated_fields = Vec::with_capacity(fields.len());
                for (name, expr) in fields {
                    evaluated_fields.push((name.clone(), self.eval_expr(expr)?));
                }
                Ok(Value::Struct(Rc::new(RefCell::new(evaluated_fields))))
            }
            PostfixOp::Index(elems) => {
                // `<X>.<method>[T, shape]` parses as Index when all args are
                // positional (the common form, e.g. `vault.zeros[f32, [768, 3072]]`).
                if let Some(v) = self.try_arena_constructor(
                    expr,
                    elems.iter().filter_map(|e| match e {
                        IndexElem::Expr(e) => Some(e),
                        _ => None,
                    }),
                )? {
                    return Ok(v);
                }
                // Tensor `.split[n]` (SPEC §6.4): the single-positional form (no
                // named `axis=`) parses as Index, so it must dispatch here too —
                // otherwise the pieces fall through to opaque indexing and a
                // destructure reads silent nils. axis defaults to -1; the
                // `[n, axis=k]` form parses as BracketArgs and is handled there.
                if let Expr::Postfix { expr: recv, op: PostfixOp::Field(fname), .. } = expr {
                    if fname == "split" {
                        let positional: Vec<&Expr> = elems.iter().filter_map(|e| match e {
                            IndexElem::Expr(x) => Some(x),
                            _ => None,
                        }).collect();
                        return self.eval_tensor_split_parts(recv, &positional, None, span);
                    }
                }
                // tensor.reshape[[d0, d1, ...]] — reshape to a new shape given as a list
                if let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr {
                    if method == "reshape" {
                        let recv = self.eval_expr(inner)?;
                        if let Value::Tensor(t) = recv {
                            // #781: `[4096]` here is a shape spec, not a
                            // tensor — it is consumed for its integers and
                            // discarded, never returned to the program, and
                            // the JIT never charges it either. Suppress the
                            // literal's own charge (`eval_expr` would
                            // otherwise route it through `Expr::TensorLit`
                            // and bill it as an implicit intermediate).
                            let shape_val = match elems.first() {
                                Some(IndexElem::Expr(e)) => self.eval_expr_lit_charge_suppressed(e)?,
                                _ => return Err(RuntimeError::at(
                                    "reshape: expected shape list in brackets", span)),
                            };
                            let new_shape: Vec<usize> = match shape_val {
                                Value::List(vs) => vs.iter()
                                    .filter_map(|v| v.as_int().map(|n| n as usize))
                                    .collect(),
                                Value::Tuple(vs) => vs.iter()
                                    .filter_map(|v| v.as_int().map(|n| n as usize))
                                    .collect(),
                                // [2, 3, 4] in demoniC evaluates as a Tensor literal
                                Value::Tensor(t) => t.iter().map(|&x| x as usize).collect(),
                                Value::Int(n, _) => vec![n as usize],
                                _ => return Err(RuntimeError::at(
                                    "reshape: shape must be a list of integers", span)),
                            };
                            let dtype = t.dtype;
                            let out = t.data.clone().into_shape_with_order(ndarray::IxDyn(&new_shape))
                                .map(|a| Value::tensor_dt(a, dtype))
                                .map_err(|e| RuntimeError::at(
                                    format!("reshape failed: {}", e), span.clone()))?;
                            // #781: reshape clones the backing storage into
                            // the new shape — an implicit intermediate.
                            self.charge_intermediate(&out, Some(&span))?;
                            return Ok(out);
                        }
                    }
                }
                let base = self.eval_expr(expr)?;
                // Collect scalar integer indices from the bracket elems.
                // Negative indices are resolved after we know the tensor shape.
                let mut raw_idx: Vec<i64> = Vec::new();
                let mut all_scalar = true;
                for elem in elems {
                    match elem {
                        IndexElem::Expr(e) => {
                            match self.eval_expr(e)? {
                                Value::Int(n, _)   => raw_idx.push(n),
                                Value::Float(x, _) => raw_idx.push(x as i64),
                                _ => { all_scalar = false; break; }
                            }
                        }
                        _ => { all_scalar = false; break; }
                    }
                }
                match base {
                    Value::Tensor(arr) if all_scalar => {
                        // Resolve negative indices against the tensor shape, with a
                        // clean bounds check (#207). Raw ndarray indexing panics on
                        // out-of-range (and a too-negative index wraps to a huge
                        // usize), so validate each axis first and return a
                        // RuntimeError — matching the tensor *write* path and the
                        // tuple/string read branches below.
                        let mut idx: Vec<usize> = Vec::with_capacity(raw_idx.len());
                        for (axis, &n) in raw_idx.iter().enumerate() {
                            let dim = arr.shape().get(axis).copied().unwrap_or(1) as i64;
                            let resolved = if n < 0 { dim + n } else { n };
                            if resolved < 0 || resolved >= dim {
                                return Err(RuntimeError::at(
                                    format!("index {} out of bounds for axis {} of size {}", n, axis, dim),
                                    span,
                                ));
                            }
                            idx.push(resolved as usize);
                        }
                        if idx.len() == arr.ndim() {
                            // Full index → scalar element. Integer tensors yield
                            // an Int so downstream `/`, `%`, `&`, range bounds etc.
                            // keep integer semantics (#125).
                            let x = arr[IxDyn(&idx)];
                            match arr.dtype {
                                // #571: the element's width comes with it. The
                                // JIT loads an `i32` lane as an `I32`, so a
                                // width-less load here made `t[i] + 1` wrap on
                                // one backend and not the other.
                                DType::Int(w) => Ok(Value::Int(w.wrap(x as i64), w)),
                                DType::Bool => Ok(Value::Bool(x != 0.0)),
                                _ => Ok(Value::Float(x, arr.float_width())),
                            }
                        } else if idx.len() < arr.ndim() {
                            // Partial index → sub-tensor slice along first axes;
                            // a sub-tensor keeps its parent's dtype.
                            let mut view = arr.view();
                            for &i in &idx {
                                view = view.index_axis_move(Axis(0), i);
                            }
                            let out = Value::tensor_dt(view.to_owned().into_dyn(), arr.dtype);
                            // #781: a partial index copies a sub-tensor's
                            // backing storage — an implicit intermediate
                            // ("slices that copy").
                            self.charge_intermediate(&out, Some(&span))?;
                            Ok(out)
                        } else {
                            Err(RuntimeError::at(
                                format!("index has {} dims but tensor has {}", idx.len(), arr.ndim()),
                                span,
                            ))
                        }
                    }
                    Value::Tuple(vs) if all_scalar && raw_idx.len() == 1 => {
                        let len = vs.len() as i64;
                        let n = raw_idx[0];
                        let i = if n < 0 { (len + n) as usize } else { n as usize };
                        vs.into_iter().nth(i).ok_or_else(|| RuntimeError::at(
                            format!("tuple index {} out of range", i), span,
                        ))
                    }
                    // #266: model-array (List) read — `arr[i]` returns the slot
                    // (a model Struct, or Nil if unfilled). Negative indices wrap.
                    Value::List(vs) if all_scalar && raw_idx.len() == 1 => {
                        let len = vs.len() as i64;
                        let n = raw_idx[0];
                        let resolved = if n < 0 { len + n } else { n };
                        if resolved < 0 || resolved >= len {
                            return Err(RuntimeError::at(
                                format!("index {} out of bounds for axis 0 of size {}", n, len), span));
                        }
                        vs.get(resolved as usize).cloned().ok_or_else(|| RuntimeError::at(
                            format!("list index {} out of range", resolved), span))
                    }
                    // Multi-axis slicing: t[.., n, ..] — mix of FullSlice and scalar indices.
                    // FullSlice (..) keeps the entire axis; a scalar index normally reduces
                    // rank by 1. Exception (keepdims): if the scalar index is "sandwiched"
                    // between FullSlice ops on both sides (e.g. t[.., n, ..]), the axis is
                    // preserved with size 1 — matching the KV-cache read pattern [B, S, D].
                    Value::Tensor(arr) if !all_scalar => {
                        let selections = self.resolve_index_selection(arr.shape(), elems, span.clone())?;
                        let result = self.read_tensor_selection(&arr, &selections);
                        let out = Value::tensor_dt(result, arr.dtype);
                        // #781: a multi-axis slice copies the selected
                        // elements into a new tensor — "slices that copy".
                        self.charge_intermediate(&out, Some(&span))?;
                        Ok(out)
                    }
                    // `s[i]` on a string — return the byte value at index i.
                    // Matches JIT behaviour: returns an i64 ASCII/UTF-8 byte.
                    Value::Str(ref s) if all_scalar && raw_idx.len() == 1 => {
                        let len = s.len() as i64;
                        let n = raw_idx[0];
                        let i = if n < 0 { (len + n) as usize } else { n as usize };
                        match s.as_bytes().get(i) {
                            Some(&b) => Ok(Value::Int(b as i64, IW::I64)),
                            None => Err(RuntimeError::at(
                                format!("string index {} out of range (len {})", i, s.len()),
                                span,
                            )),
                        }
                    }
                    // `f[N]` on a user fn: bind explicit shape params
                    Value::Fn(ref name) if all_scalar => {
                        if let Some(f) = self.fns.get(name).cloned() {
                            if !f.shape_params.is_empty() {
                                let shape_bindings: Vec<(String, i64)> = f.shape_params.iter()
                                    .zip(raw_idx.iter())
                                    .map(|(sp, &n)| (sp.name.clone(), n))
                                    .collect();
                                return Ok(Value::BoundFn { name: name.clone(), shape_bindings });
                            }
                        }
                        Ok(Value::Opaque("index".into()))
                    }
                    // Indexing `nil` is not an index. It used to fall through
                    // to the forward-compat `Opaque("index")` below, so `e[0]`
                    // on a nil `Err` printed `<opaque index>` and exited 0 —
                    // silent, and the JIT had raised on the same program. Say
                    // it, at the span, in the JIT's words.
                    Value::Nil => Err(RuntimeError::at("cannot index nil", span)),
                    // A `Port` handle has no elements (PORTS.md §7.1), so
                    // indexing one is an error rather than an opaque.
                    Value::Opaque(ref o) if crate::ports::handle_id(o).is_some() =>
                        Err(RuntimeError::at(
                            "cannot index a Port handle — a handle is opaque and has \
                             no elements (PORTS §7.1)", span)),
                    _ => Ok(Value::Opaque("index".into())),
                }
            }
        }
    }

    /// Evaluate an expression as a dimension count. #839: this used to
    /// substitute a bare 4 on ANY error, so `forge.zeros[f32, [n()]]` with a
    /// raising `n()` — an arena exhaustion, a `budget-stale` retrieval, any
    /// runtime error in user code — silently built a wrong-sized tensor with
    /// no diagnostic. That default served no live caller: the one legitimate
    /// symbol it was meant to catch, the KV streaming axis `~`, is resolved
    /// by the caller (the `is_kv` check in `try_arena_constructor`) before
    /// `eval_dim` is ever reached, and an unbound shape param already errors
    /// earlier, in `call_fn_with_shapes`'s inference pass. So a real
    /// evaluation error now propagates, located at the failing sub-
    /// expression's own span, and a value that evaluates cleanly but isn't a
    /// usable dimension (negative, or not a number) is its own located error
    /// rather than a silent 4.
    fn eval_dim(&mut self, e: &Expr) -> EvalResult<usize> {
        match self.eval_expr(e)? {
            Value::Int(n, _)   if n >= 0 => Ok(n as usize),
            Value::Float(x, _) if x >= 0.0 => Ok(x as usize),
            v => Err(RuntimeError::at(
                format!("a tensor shape dimension must be a non-negative number, \
                         got `{}` ({})", v, v.type_name()),
                e.span_of())),
        }
    }

    /// Concrete dimensions of a `@cast(Model)` tensor field's shape. Every axis
    /// must be a static non-negative integer: the overlay's byte extent is
    /// element count × element width, so a dimension that is not pinned has no
    /// extent to size (#598). Unlike `eval_dim` this never substitutes a
    /// default — guessing an axis here would mis-size every field after it.
    fn overlay_dims(&mut self, shape: &crate::ast::ShapeSpec, model: &str, field: &str)
        -> EvalResult<Vec<usize>>
    {
        use crate::ast::ShapeElem;
        let mut dims = Vec::with_capacity(shape.elems.len());
        for e in &shape.elems {
            let n = match e {
                ShapeElem::Expr(ex) => match self.eval_expr(ex) {
                    Ok(Value::Int(n, _)) if n >= 0 => n as usize,
                    _ => return Err(overlay_no_layout(model, field,
                        "a tensor field overlays only at a static shape")),
                },
                _ => return Err(overlay_no_layout(model, field,
                    "a tensor field overlays only at a static shape")),
            };
            dims.push(n);
        }
        Ok(dims)
    }

    /// Overlay a raw byte tensor onto a model's fields (`@cast(Model){bytes}`).
    /// Fields are read in declaration order at successive byte offsets, each at
    /// the width its DECLARED type states, and each decoded back INTO that type
    /// — an `f32` field is an f32, a `bool` field is a bool, a
    /// `Tensor[u8,[24]]` field is a 24-byte row-major tensor (#598). Returns a
    /// struct value whose fields read like any model instance's.
    ///
    /// Byte order: multi-byte fields are assembled little-endian (#606,
    /// `SPEC.md §3.10`) — the one byte order this language fixes
    /// (`PORTS.md`), and the same order every other byte path in this crate
    /// already used.
    fn cast_to_struct(&mut self, raw: &Value, model_name: &str) -> EvalResult<Value> {
        use crate::ast::Type;
        let key = self.resolve_model(model_name);
        let fields_def = match key.as_ref().and_then(|k| self.model_fields.get(k)) {
            Some(f) => f.clone(),
            None => return Err(RuntimeError::msg(format!("@cast: `{}` is not a known model", model_name))),
        };
        // The instance carries the DECLARING file's key — that is what method
        // dispatch and field types are registered under — while every
        // diagnostic below keeps the name the program actually wrote.
        let tag = key.unwrap_or_else(|| model_name.to_string());
        // One byte per ELEMENT: `u8` tensors and views are the ordinary source,
        // and `read_bytes` hands back the same stream as one i64 per byte
        // (STDLIB.md). `--check` refuses a source of wider elements.
        let bytes: Vec<u8> = match raw {
            Value::Tensor(t) => t.data.iter().map(|&x| x as i64 as u8).collect(),
            other => return Err(RuntimeError::msg(format!(
                "@cast({}) expects a byte tensor to overlay, got {}", model_name, other.type_name()))),
        };
        let mut offset = 0usize;
        let mut out: Vec<(String, Value)> =
            vec![("__model__".to_string(), Value::Str(tag))];
        for (fname, fty) in &fields_def {
            // #598: the spec's only wire-layout guarantee for a model is field
            // ORDER (SPEC.md §3.10), so the overlay reaches exactly as far as
            // each field's own type fixes a byte extent — a scalar of known
            // width, or a tensor of such scalars at a static shape.
            // `--check` refuses everything else at the `@cast` site
            // (`check_cast_model_overlay`); these arms are the interpreter's
            // own backstop for a caller that skipped the checker, so an
            // unlayoutable field fails loud here instead of silently reading
            // as a corrupted `Int(0)` (the old behavior, since
            // `ScalarType::Str` had a byte size of 0).
            let val = match fty {
                Type::Scalar(st, _) => {
                    let Some(w) = st.overlay_byte_width() else {
                        return Err(overlay_no_layout(model_name, fname,
                            st.overlay_refusal_reason().unwrap_or("no byte layout")));
                    };
                    let slice = read_overlay_bytes(&bytes, offset, w, model_name, fname)?;
                    offset += w;
                    decode_overlay_scalar(st, slice)
                }
                Type::Tensor(elem, shape, _) => {
                    let Type::Scalar(est, _) = &**elem else {
                        return Err(overlay_no_layout(model_name, fname,
                            "a tensor field overlays only with a scalar element type"));
                    };
                    let Some(w) = est.overlay_byte_width() else {
                        return Err(overlay_no_layout(model_name, fname,
                            est.overlay_refusal_reason().unwrap_or("no byte layout")));
                    };
                    let dims = self.overlay_dims(shape, model_name, fname)?;
                    let count: usize = dims.iter().product();
                    // Row-major, the one element order the spec's tensor
                    // convention states — `ArrayD::from_shape_vec` reads this
                    // vector in exactly that order.
                    let mut data: Vec<f64> = Vec::with_capacity(count);
                    for _ in 0..count {
                        let slice = read_overlay_bytes(&bytes, offset, w, model_name, fname)?;
                        offset += w;
                        data.push(overlay_scalar_as_f64(est, slice));
                    }
                    let arr = ArrayD::from_shape_vec(IxDyn(&dims), data)
                        .map_err(|e| RuntimeError::msg(format!("@cast({}): {}", model_name, e)))?;
                    let field_val = Value::tensor_dt(arr, overlay_elem_dtype(est)?);
                    // #781: the overlay reads raw bytes into a freshly
                    // allocated element array per tensor field — an implicit
                    // intermediate. No span reaches this decode loop (the
                    // `@cast` call site's is not threaded through); charged
                    // unlocated, the same as any other unbounded-arena
                    // watermark update. Uncharged on the JIT too today; it
                    // refuses the `@cast` overlay outright.
                    self.charge_intermediate(&field_val, None)?;
                    field_val
                }
                other => return Err(overlay_no_layout(model_name, fname,
                    &format!("a field of type {:?} has no byte layout", other))),
            };
            out.push((fname.clone(), val));
        }
        Ok(Value::Struct(Rc::new(RefCell::new(out))))
    }

    /// Detects the `<X>.<method>[T, shape]` arena/rng constructor pattern and,
    /// if matched, returns a real tensor (or `(rng, tensor)` for rng ctors).
    /// Returns `Ok(None)` if `expr` doesn't look like a constructor.
    ///
    /// Recognised methods:
    ///   - `zeros`, `ones`, `uninit`, `kv` → tensor of zeros (or ones)
    ///   - `normal`, `uniform`             → (rng_state, tensor) tuple
    ///   - `identity`                      → N×N identity matrix (scalar N arg)
    /// Tensor `.split[n, axis=k]` (SPEC §6.4): split a tensor into `n` equal
    /// pieces along `axis` (default the last), returned as an `n`-tuple. `axis`
    /// may be negative (numpy-style). Errors if the axis length is not divisible
    /// by `n` or the axis is out of range.
    fn eval_tensor_split(&mut self, recv: &Expr, args: &[CallArg], span: Span) -> EvalResult<Value> {
        // n = first positional arg; axis = named `axis=` (default -1).
        let mut positional: Vec<&Expr> = Vec::new();
        let mut axis_expr: Option<&Expr> = None;
        for a in args {
            match a {
                CallArg::Positional(e) => positional.push(e),
                CallArg::Named { name, value, .. } if name == "axis" => axis_expr = Some(value),
                _ => {}
            }
        }
        self.eval_tensor_split_parts(recv, &positional, axis_expr, span)
    }

    /// Shared `.split` core, taking the receiver, positional args, and optional
    /// `axis=` expr directly — so both the `BracketArgs` form (`[n, axis=k]`) and
    /// the `Index` form (`[n]`, single positional, no named arg) can reach it.
    fn eval_tensor_split_parts(
        &mut self,
        recv: &Expr,
        positional: &[&Expr],
        axis_expr: Option<&Expr>,
        span: Span,
    ) -> EvalResult<Value> {
        let recv_val = self.eval_expr(recv)?;
        let t = match &recv_val {
            Value::Tensor(t) => t,
            other => return Err(RuntimeError::at(
                format!("`.split` expects a tensor, got {}", other.type_name()), span)),
        };
        // #639: the piece count has a diagnostic of its own for anything that
        // is not a positive integer — but an escaping `return` / `break` /
        // `continue` is not a bad piece count, so it travels on rather than
        // being reported as one.
        let count = match positional.first() {
            Some(e) => self.eval_expr_tolerant(e)?,
            None => None,
        };
        let n = count
            .and_then(|v| v.as_int())
            .filter(|&n| n > 0)
            .ok_or_else(|| RuntimeError::at(
                "`.split` requires a positive integer piece count, e.g. `t.split[3, axis=-1]`",
                span.clone()))? as usize;
        let ndim = t.data.ndim();
        let axis_raw = match axis_expr {
            Some(e) => self.eval_expr(e)?.as_int().unwrap_or(-1),
            None => -1,
        };
        let axis_norm = if axis_raw < 0 { ndim as i64 + axis_raw } else { axis_raw };
        if axis_norm < 0 || axis_norm as usize >= ndim {
            return Err(RuntimeError::at(
                format!("`.split`: axis {} is out of range for a rank-{} tensor", axis_raw, ndim), span));
        }
        let axis = axis_norm as usize;
        let dim = t.data.shape()[axis];
        if n == 0 || dim % n != 0 {
            return Err(RuntimeError::at(
                format!("`.split`: axis {} of length {} is not divisible into {} equal pieces", axis, dim, n), span));
        }
        let w = dim / n;
        let dtype = t.dtype;
        let mut pieces = Vec::with_capacity(n);
        for i in 0..n {
            let slice = t.data
                .slice_axis(Axis(axis), ndarray::Slice::from((i * w)..((i + 1) * w)))
                .to_owned();
            let piece = Value::Tensor(TensorVal::new(slice, dtype));
            // #781: each piece is its own copied-out backing array — an
            // implicit intermediate, one per piece. Uncharged on the JIT
            // too today; it refuses `.split` outright.
            self.charge_intermediate(&piece, Some(&span))?;
            pieces.push(piece);
        }
        Ok(Value::Tuple(pieces))
    }

    /// `rng.uniform_int[T, [shape]](low, high)` (SPEC §3.8, #395): integer
    /// draws in the half-open range `[low, high)`, matching the `rand_int`
    /// convention. Returns the linear-value pair `(rng, tensor)` like the
    /// float rng constructors; the tensor is dtype-tagged Int so element
    /// reads round-trip as integers.
    fn eval_rng_uniform_int(
        &mut self,
        rng_expr: &Expr,
        bracket_args: &[&Expr],
        call_args: &[Value],
        span: Span,
    ) -> EvalResult<Value> {
        let lo = call_args.first().and_then(|v| v.as_int()).ok_or_else(|| RuntimeError::at(
            "rng.uniform_int: low argument required (integer)".to_string(), span.clone()))?;
        let hi = call_args.get(1).and_then(|v| v.as_int()).ok_or_else(|| RuntimeError::at(
            "rng.uniform_int: high argument required (integer)".to_string(), span.clone()))?;
        if hi <= lo {
            return Err(RuntimeError::at(format!(
                "rng.uniform_int: high ({}) must be greater than low ({})", hi, lo), span));
        }
        let dims: Vec<usize> = match bracket_args.iter().find_map(|e| match e {
            Expr::TensorLit(d, _) =>
                Some(d.iter().map(|x| self.eval_dim(x)).collect::<EvalResult<Vec<usize>>>()),
            _ => None,
        }) {
            Some(r) => r?,
            None => Vec::new(),
        };
        let nelems = checked_shape_elems(&dims).ok_or_else(|| RuntimeError::at(format!(
            "uniform_int: tensor shape {:?} is too large — its element count \
            overflows the address space", dims), span.clone()))?;
        let range = (hi - lo) as u64;
        let mut data: Vec<f64> = Vec::with_capacity(nelems);
        for _ in 0..nelems {
            data.push((lo + (self.rand_u64() % range) as i64) as f64);
        }
        let arr = ArrayD::from_shape_vec(IxDyn(&dims), data)
            .map_err(|e| RuntimeError::at(format!("uniform_int: {}", e), span.clone()))?;
        self.prof_alloc();
        let rng_state = self.eval_expr(rng_expr)?;
        // `uniform_int` names no element type, so its elements are i64-wide.
        Ok(Value::Tuple(vec![rng_state, Value::tensor_dt(arr, DType::Int(IW::I64))]))
    }

    fn try_arena_constructor<'a>(
        &mut self,
        expr: &Expr,
        positional: impl Iterator<Item = &'a Expr>,
    ) -> EvalResult<Option<Value>> {
        let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr else {
            return Ok(None);
        };

        let is_tensor_ctor = matches!(method.as_str(),
            "zeros" | "ones" | "uninit" | "identity" | "kv" | "trit");
        let is_rng_ctor = matches!(method.as_str(), "normal" | "uniform");
        // #395: `rng.uniform_int` reaching here means the bracket form was used
        // without the range call — the draw is handled at the call site
        // (`eval_rng_uniform_int`), which needs `(low, high)`.
        if method == "uniform_int" {
            return Err(RuntimeError::at(
                "`rng.uniform_int` needs its range: \
                 `rng.uniform_int[T, [shape]](low, high)`".to_string(),
                expr.span_of(),
            ));
        }
        if !is_tensor_ctor && !is_rng_ctor { return Ok(None); }

        // #400: which sized arena this constructor charges against. `vault.*`
        // is Vault, `forge.*` is Forge; `.kv` on either lands in the Stream
        // arena (`MEMORY.md §9`), which has no sizing flag. An `rng.*` draw
        // (`r.normal[…]`) has no arena base in its spelling either — `r` names
        // the RNG value, not an arena — but it still materializes a fresh
        // tensor; #781 charges it to Forge unconditionally, the same as any
        // other implicit intermediate (`charge_intermediate`'s reasoning:
        // `MEMORY.md §1`'s Forge row is "activations, gradients, scratch").
        // See `charge_arena`.
        let charge_to = if is_rng_ctor {
            Some(crate::arena::Arena::Forge)
        } else {
            match inner.as_ref() {
                Expr::Ident(base, _) if method != "kv" => match base.as_str() {
                    "vault" => Some(crate::arena::Arena::Vault),
                    "forge" => Some(crate::arena::Arena::Forge),
                    _ => None,
                },
                _ => None,
            }
        };
        // #674: the Vault
        // and Forge arms both route through the same `Value::tensor_dt` →
        // `TensorVal::new` chokepoint the budget stamp is applied at, so a
        // `vault.zeros[…]` evaluated while a budget is current would
        // otherwise inherit that budget's tag and read `budget-stale` once
        // the budget is reaped — though its bytes live in `vault_used`,
        // never the budget's.
        //
        // Reviewer edit (slice 3d, §3.5): this is a flag, not a guard held
        // for the rest of the call. The earlier shape opened the guard HERE,
        // before the shape arguments below are evaluated, so a forge tensor
        // allocated inside `vault.zeros[f32, [n()]]`'s own argument — with
        // `n()` allocating — was charged to the entered budget by
        // `charge_arena` (which reads `cur_budget`) but stamped ROOT by
        // `TensorVal::new` (which reads the cleared `CUR_BUDGET_TAG`): index
        // and stamp disagreed, so the tensor's bytes died at `reap(b)` while
        // it still read as live. §3.5 splits the two: the CONSTRUCTOR's own
        // allocation charges root and stamps root; its ARGUMENTS are
        // ordinary code in the entered budget and charge `b` and stamp `b`.
        // So the guard is now opened at each of the three points the
        // constructor's own bytes are charged and built, below, and never
        // over argument evaluation. It is the interpreter's half of the same
        // seam `Jit::forge_alloc_root` closes on the JIT.
        let vault_stamps_root = matches!(charge_to, Some(crate::arena::Arena::Vault));

        // Collect args once so we can read both the element type (for the
        // dtype tag) and the shape literal. The element type is the leading
        // bracket arg, e.g. the `u64` in `forge.zeros[u64, [128]]`.
        let pos_vec: Vec<&Expr> = positional.collect();

        // #266: a *model* element type (`forge.uninit[Layer, [N]]`, bare or
        // parameterized `Layer[D]`) builds a model array — represented as a List
        // of N slots (Nil until filled), reaching parity with the JIT's
        // ModelArray. `arr[i] = instance` writes and `arr[i].field` reads are
        // handled in the index-assign / index-read paths. Tensor element types
        // fall through to the numeric path below.
        if matches!(method.as_str(), "zeros" | "ones" | "uninit") {
            if let Some(base) = pos_vec.first().and_then(|e| model_ctor_base_name(e)) {
                if self.resolve_model(base).is_some() {
                    let n = match pos_vec.iter().find_map(|e| match e {
                        Expr::TensorLit(dims, _) => dims.first().map(|d| self.eval_dim(d)),
                        _ => None,
                    }) {
                        Some(r) => r?,
                        None => return Err(RuntimeError::msg(format!(
                            "`{}[{}, [N]]` requires a 1D size", method, base))),
                    };
                    self.prof_alloc();
                    return Ok(Some(Value::list(vec![Value::Nil; n])));
                }
            }
        }

        // Random draws are float-valued by definition regardless of the
        // declared type; everything else inherits the declared element type.
        let dtype = if is_rng_ctor { DType::F32 } else { arena_ctor_dtype(&pos_vec, &expr.span_of())? };

        // `identity` builds an N×N matrix with 1.0 on the diagonal. It accepts
        // either a scalar dimension (`vault.identity[T, N]`) or a square shape
        // literal (`vault.identity[T, [N, N]]`). Completes the zeros/ones/
        // identity additive-and-multiplicative-identity triple.
        if method == "identity" {
            let n = match pos_vec.iter().find_map(|e| match e {
                Expr::TensorLit(dims, _) => dims.first().map(|d| self.eval_dim(d)),
                Expr::Literal(Literal::Int(k, _), _) if *k >= 0 => Some(Ok(*k as usize)),
                _ => None,
            }) {
                Some(r) => r?,
                None => 0,
            };
            if checked_shape_elems(&[n, n]).is_none() {
                return Err(RuntimeError::msg(format!(
                    "identity: size {} is too large — the {}×{} element count overflows the address space",
                    n, n, n)));
            }
            // #781: charge at the DECLARED element width, not the
            // interpreter's f64 storage width — an f32 `identity` bills half
            // what an f64 one does, matching the JIT.
            // #674 slice 3d (§3.5): the constructor's OWN bytes — charged
            // to Vault's flat meter and stamped ROOT, whatever budget is
            // entered. Opened here, after the shape arguments above were
            // evaluated in the entered budget.
            let _vault_tag_guard = vault_stamps_root.then(VaultTagGuard::new);
            self.charge_ctor(
                charge_to,
                (n as u64).saturating_mul(n as u64).saturating_mul(dtype.byte_width()),
                Some(&expr.span_of()),
            )?;
            let mut a = ArrayD::<f64>::zeros(IxDyn(&[n, n]));
            for i in 0..n { a[[i, i]] = 1.0; }
            self.prof_alloc();
            return Ok(Some(Value::tensor_dt(a, dtype)));
        }

        // forge.trit[M, N] — dimensions given as raw integer args, no type arg.
        // Returns a zero-initialised DType::Trit tensor clamped to {-1, 0, +1}.
        if method == "trit" {
            let trit_dims: Vec<usize> = pos_vec.iter().filter_map(|e| match e {
                Expr::Literal(Literal::Int(k, _), _) if *k >= 0 => Some(*k as usize),
                _ => None,
            }).collect();
            let trit_elems = checked_shape_elems(&trit_dims).ok_or_else(|| RuntimeError::msg(
                format!("trit: shape {:?} is too large — its element count overflows the address space",
                        trit_dims)))?;
            // #781: declared width, not f64 storage width (`DType::Trit` is
            // one byte per element, matching an unpacked trit lane).
            // #674 slice 3d (§3.5): the constructor's OWN bytes — charged
            // to Vault's flat meter and stamped ROOT, whatever budget is
            // entered. Opened here, after the shape arguments above were
            // evaluated in the entered budget.
            let _vault_tag_guard = vault_stamps_root.then(VaultTagGuard::new);
            self.charge_ctor(
                charge_to,
                (trit_elems as u64).saturating_mul(DType::Trit.byte_width()),
                Some(&expr.span_of()))?;
            let arr = ArrayD::from_elem(IxDyn(&trit_dims), 0.0_f64);
            self.prof_alloc();
            return Ok(Some(Value::tensor_dt(arr, DType::Trit)));
        }

        // Find a shape literal among the positional args — typically the second
        // arg, after the element type.
        let is_kv = method == "kv";
        let shape_dims: Option<Vec<usize>> = pos_vec.iter().filter_map(|e| match e {
            Expr::TensorLit(dims, _) => Some(dims),
            _ => None,
        }).next().map(|dims| dims.iter().map(|e| {
            // For KV constructors, the streaming axis (`~`) starts at 0 (empty);
            // capacity is tracked separately and grows via `<-` append.
            if is_kv {
                if let Expr::Ident(name, _) = e {
                    if name == "~" { return Ok(0); }
                }
                // Also handle UnOp::BitNot used as ~ in expressions
            }
            self.eval_dim(e)
        }).collect::<EvalResult<Vec<usize>>>()).transpose()?;
        let dims = shape_dims.unwrap_or_default();
        let shape_span = pos_vec.iter().find_map(|e| match e {
            Expr::TensorLit(_, sp) => Some(sp.clone()),
            _ => None,
        });
        let located = |msg: String| match &shape_span {
            Some(sp) => RuntimeError::at(msg, sp.clone()),
            None => RuntimeError::msg(msg),
        };
        // Guard oversized shapes. An element count that overflows the address
        // space would panic deep in ndarray; one that merely exceeds RAM would
        // abort or trip the OS OOM-killer with no output. Both become a located
        // diagnostic here (#368 follow-up): first the overflow check, then a
        // fallible reserve so a too-large-but-valid shape fails cleanly.
        let nelems = checked_shape_elems(&dims).ok_or_else(|| located(format!(
            "{}: tensor shape {:?} is too large — its element count overflows the address space",
            method, dims)))?;
        // Reject a shape that can't possibly fit in RAM before touching it (the
        // interpreter backs every element with an f64). Overcommit lets the
        // allocation itself succeed, so this pre-check is what averts a silent
        // OOM-kill; the `try_reserve` below is the remaining backstop. This is
        // the REAL host allocation size, always f64 per element regardless of
        // `dtype` — kept separate from `nbytes` below (#781), the DECLARED
        // width charged against the arena budget.
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        let real_alloc_bytes = (nelems as u64).saturating_mul(8);
        // #781: the arena budget is requested bytes at the declared element
        // width (`DType::byte_width`), not the interpreter's f64 storage
        // width — an f32 tensor bills half what an f64 one does, matching
        // the JIT.
        let nbytes = (nelems as u64).saturating_mul(dtype.byte_width());
        // #400: the declared arena budget is checked before the machine's, so a
        // program run under `--forge=2G` hears about its own limit rather than
        // about the host's RAM.
        // #674 slice 3d (§3.5): the constructor's OWN bytes — charged
        // to Vault's flat meter and stamped ROOT, whatever budget is
        // entered. Opened here, after the shape arguments above were
        // evaluated in the entered budget.
        let _vault_tag_guard = vault_stamps_root.then(VaultTagGuard::new);
        self.charge_ctor(charge_to, nbytes, shape_span.as_ref())?;
        if let Some(total) = total_system_memory_bytes() {
            if real_alloc_bytes > total {
                return Err(located(format!(
                    "{}: tensor shape {:?} needs {:.1} GiB but the system has {:.1} GiB of RAM",
                    method, dims, real_alloc_bytes as f64 / GIB, total as f64 / GIB)));
            }
        }
        let mut data: Vec<f64> = Vec::new();
        if data.try_reserve_exact(nelems).is_err() {
            let gib = (nelems as f64) * 8.0 / (1024.0 * 1024.0 * 1024.0);
            return Err(located(format!(
                "{}: tensor shape {:?} needs {:.1} GiB and could not be allocated",
                method, dims, gib)));
        }
        let arr = if is_rng_ctor {
            // Real PRNG draw — Box-Muller for normal, xorshift uniform for
            // uniform. Reproducible from Rng.seed(N) earlier in the program.
            data.resize(nelems, 0.0);
            for slot in data.iter_mut() {
                *slot = if method == "normal" { self.rand_normal() } else { self.rand_uniform() };
            }
            ArrayD::from_shape_vec(IxDyn(&dims), data)
                .map_err(|e| RuntimeError::msg(format!("{}: {}", method, e)))?
        } else {
            let fill = if method == "ones" { 1.0 } else { 0.0 };
            data.resize(nelems, fill);
            ArrayD::from_shape_vec(IxDyn(&dims), data)
                .map_err(|e| RuntimeError::msg(format!("{}: {}", method, e)))?
        };
        self.prof_alloc();
        if is_rng_ctor {
            let rng_state = self.eval_expr(inner)?;
            Ok(Some(Value::Tuple(vec![rng_state, Value::tensor_dt(arr, dtype)])))
        } else {
            Ok(Some(Value::tensor_dt(arr, dtype)))
        }
    }
}

/// Element dtype of an arena constructor from its bracket type argument
/// (`forge.zeros[T, ..]`). Integer scalar types tag the tensor `Int` so element
/// reads round-trip as integers (#125); `f64` tags `F64` (full width, like the
/// JIT's f64 tensors, #241); other float/unknown types are `F32`, the default
/// f32-rounded float (f16/bf16/tf32/fp8 compute as f32, the #179 convention).
/// Product of tensor dims, guarding against `isize` overflow — an element count
/// past `isize::MAX` would panic deep inside `ndarray` with a raw backtrace.
/// Returns `None` when the shape is too large to allocate; the caller turns that
/// into a located diagnostic (#368 follow-up).
fn checked_shape_elems(dims: &[usize]) -> Option<usize> {
    dims.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .filter(|&n| n <= isize::MAX as usize)
}

/// Total physical RAM in bytes (queried once, cached), via Unix `sysconf`.
/// `None` if unavailable (non-Unix, or the OS doesn't report it). Used to reject
/// a single tensor allocation that can't possibly fit *before* it OOM-kills the
/// process — memory overcommit means the allocation itself succeeds and only the
/// first page-touch triggers the kill, so `try_reserve` alone doesn't catch it.
pub(crate) fn total_system_memory_bytes() -> Option<u64> {
    static MEM: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MEM.get_or_init(|| {
        #[cfg(unix)]
        {
            // SAFETY: `sysconf` is a pure query with no preconditions; it
            // returns -1 for an unknown/unsupported name, handled below.
            let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) };
            if pages > 0 && page_size > 0 {
                Some((pages as u64).saturating_mul(page_size as u64))
            } else {
                None
            }
        }
        #[cfg(not(unix))]
        { None }
    })
}

/// #266: base model name of a constructor element-type arg — bare (`Foo`) or
/// parameterized (`Bar[4]`, which parses as an Index/BracketArgs postfix over
/// the model ident). Mirrors the JIT's `model_ctor_base_name`.
fn model_ctor_base_name(e: &Expr) -> Option<&str> {
    match e {
        Expr::Ident(n, _) => Some(n.as_str()),
        Expr::Postfix { expr, op: PostfixOp::Index(_), .. }
        | Expr::Postfix { expr, op: PostfixOp::BracketArgs(_), .. } => match expr.as_ref() {
            Expr::Ident(n, _) => Some(n.as_str()),
            _ => None,
        },
        _ => None,
    }
}

/// #571: the constructor names the element type, so it names the element
/// WIDTH — `forge.zeros[i32, [N]]` is 32 bits wide at every element read,
/// write and elementwise op over it.
///
/// #834: this used to be its own ad hoc lexical table (an `int_type_name_
/// width`/`is_trit_type_name`/`"f64"`/`"bool"` chain checked in a loop over
/// EVERY positional arg) that fell back to `DType::F32` for anything outside
/// it — silently mistagging `fp8_e4m3`/`fp8_e5m2` (not in the old table at
/// all) at four bytes instead of one, and doing the same for a genuinely
/// bogus name like `qqq_not_a_type` that the checker should have refused
/// before `dmc run` ever got here. Now it anchors on `pos[0]` specifically —
/// exactly the position `check.rs`'s own `ctor_tensor_ty` requires the
/// element-type name to be at for every ctor form that reaches here with a
/// dtype that matters (`zeros`/`ones`/`uninit`/`identity`/`kv`) — and
/// resolves through `ScalarType::from_ctor_name`, the SAME table the checker
/// resolves the same name through. A name that table recognizes but
/// `dtype_for_scalar_elem` cannot represent is an internal error naming the
/// type (unreachable today); a name the table does not recognize at all —
/// a generic dim like `R`, or `str`/`nil`, which the checker admits here
/// pending #817 — keeps the pre-#834 `DType::F32` fallback.
///
/// `pos[0]` not being a RECOGNIZED type name — not an `Ident` at all
/// (`forge.trit[4, 3]`'s bare integer dims; an `rng.*` draw, already routed
/// around this function by its own `is_rng_ctor` check before it is ever
/// called; a generic dim ident like `R` in `forge.trit[R, C]`), or an
/// `Ident` `ScalarType::from_ctor_name` does not recognize at all (`str`/
/// `nil` — #817 is an OPEN owner ruling on whether either has a tensor-
/// element representation, which `check.rs`'s `ctor_tensor_ty` today still
/// ADMITS in constructor position, so this function must not preempt that
/// ruling; also a genuinely bogus name like `qqq_not_a_type`) — is not a
/// name to blame. `DType::F32` there is base's pre-existing default for all
/// of these, unchanged by this fix, and callers that do not use it (`trit`
/// hardcodes its own `DType::Trit`) simply ignore it.
///
/// The internal-error arm below is reserved for a name `from_ctor_name`
/// *does* recognize but `dtype_for_scalar_elem` cannot represent. Per the
/// #834 rework review this is unreachable today: `from_ctor_name`'s 19
/// spellings map onto every `ScalarType` variant `dtype_for_scalar_elem`
/// returns `Some` for (everything except `Str`/`Nil`, which `from_ctor_name`
/// does not recognize as ctor spellings in the first place) — so the arm
/// only guards against the two tables drifting apart in the future.
fn arena_ctor_dtype(pos: &[&Expr], span: &Span) -> EvalResult<DType> {
    match pos.first() {
        Some(Expr::Ident(name, _)) => match ScalarType::from_ctor_name(name) {
            Some(st) => dtype_for_scalar_elem(&st).ok_or_else(|| RuntimeError::at(
                format!(
                    "internal error: `{name}` is a recognized constructor \
                     element-type name with no `DType` representation — \
                     `ScalarType::from_ctor_name` and `dtype_for_scalar_elem` \
                     have fallen out of sync (#834)"
                ),
                span.clone(),
            )),
            None => Ok(DType::F32),
        },
        _ => Ok(DType::F32),
    }
}

// ─── Literal → Value ─────────────────────────────────────────────────────────

fn lit_value(lit: &Literal) -> Value {
    lit_value_hinted(lit, None)
}

/// #478: does this value position hold an *unsuffixed* float literal — the
/// only thing the type-direction hint can change? Follows exactly the
/// value-transparent forms that `eval_expr` re-arms the hint through, so a
/// `false` answer means the hint provably cannot matter and the sibling-branch
/// type peek can be skipped outright.
fn has_untyped_float_lit(e: &Expr) -> bool {
    match e {
        Expr::Literal(Literal::Float(_, None), _) => true,
        Expr::UnOp { op: UnOp::Neg, operand, .. } => has_untyped_float_lit(operand),
        Expr::Tuple(els, _) if els.len() == 1 => has_untyped_float_lit(&els[0]),
        Expr::Block(b) => block_has_untyped_float_lit(b),
        Expr::DirectiveBlock { body, .. } => block_has_untyped_float_lit(body),
        Expr::ArenaBlock(ab) => block_has_untyped_float_lit(&ab.body),
        // #674 slice 3a: `forge.enter(b) { … }`'s tail value is a value
        // position exactly like `ArenaBlock`'s, above — the hint must reach
        // through it the same way.
        Expr::BudgetBlock(bb) => block_has_untyped_float_lit(&bb.body),
        // #674 slice 3a (PR #814 review): `AllocIn` (`<ctor> in b`) gained no
        // arm here when `BudgetBlock` did, above — walk BOTH of its own
        // operands, `expr` (the ctor spine) and `budget` (the `in` target),
        // the same way `BudgetBlock` walks its operand and body.
        Expr::AllocIn { expr, budget, .. } =>
            has_untyped_float_lit(expr) || has_untyped_float_lit(budget),
        Expr::If(ie) => if_has_untyped_float_lit(ie),
        Expr::Match(me) => me.arms.iter().any(|a| has_untyped_float_lit(&a.body)),
        _ => false,
    }
}

fn block_has_untyped_float_lit(b: &Block) -> bool {
    if let Some(tail) = &b.tail_expr { return has_untyped_float_lit(tail); }
    match b.stmts.last() {
        Some(Stmt::If(ie)) => if_has_untyped_float_lit(ie),
        Some(Stmt::Match(me)) => me.arms.iter().any(|a| has_untyped_float_lit(&a.body)),
        Some(Stmt::Expr { lhs, assign: None, .. }) => has_untyped_float_lit(lhs),
        _ => false,
    }
}

fn if_has_untyped_float_lit(ie: &IfExpr) -> bool {
    block_has_untyped_float_lit(&ie.then_branch) || else_has_untyped_float_lit(&ie.else_branch)
}

fn else_has_untyped_float_lit(e: &Option<ElseBranch>) -> bool {
    match e {
        Some(ElseBranch::Block(b)) => block_has_untyped_float_lit(b),
        Some(ElseBranch::If(nested)) => if_has_untyped_float_lit(nested),
        None => false,
    }
}

/// `lit_value`, plus the #478 context hint an *unsuffixed* float literal
/// adopts. `want` is `None` everywhere except a branch-join value position
/// (see `Interpreter::float_lit_hint`), and a suffix always outranks it.
fn lit_value_hinted(lit: &Literal, want: Option<FW>) -> Value {
    match lit {
        // #545: an integer literal's SUFFIX is a width origin, exactly as §3.1
        // says ("an explicit type suffix … types the literal concretely"), so
        // `2147483647i32 + 1i32` wraps. `jit.rs`'s `lower_literal` was fixed in
        // the same change to lower the suffix at its own kind, which is what
        // makes honoring it here safe: #540 backed this line out only because
        // the JIT dropped the suffix and the two backends would have parted
        // company on it. An UNSUFFIXED literal is untyped and stays `I64`: it
        // adopts a narrow operand's width at the operator (`IW::join`), which
        // is what the JIT's `adopt_int_literal_kind` does.
        Literal::Int(n, sfx) => match sfx.as_ref().and_then(scalar_int_width) {
            Some(w) => Value::Int(w.wrap(*n), w),
            None => Value::Int(*n, IW::I64),
        },
        // #473: an f32-family suffix produces a true f32 value. Rounding here
        // matters on its own: `0.1f32` is 0.100000001490116119384765625, not
        // the f64 0.1, and the JIT has always lowered it that way.
        Literal::Float(x, sfx) => match sfx {
            Some(t) if scalar_is_f32_family(t) => Value::Float(quantize_f32(*x), FW::F32),
            // #478: no suffix — the literal is untyped and adopts its context.
            None if want == Some(FW::F32) => Value::Float(quantize_f32(*x), FW::F32),
            _ => Value::Float(*x, FW::F64),
        },
        Literal::Str(s)   => Value::Str(s.clone()),
        Literal::Char(c)  => Value::Int(*c as i64, IW::I64),
        Literal::Bool(b)  => Value::Bool(*b),
        Literal::Nil      => Value::Nil,
    }
}

// ─── Operators ───────────────────────────────────────────────────────────────

/// Find the streaming axis for a `<-` append: the single axis on which the
/// existing and incoming shapes differ (e.g. the `~` axis of a KV cache);
/// falls back to axis 0 when that's ambiguous.
fn stream_axis_for(existing: &ArrayD<f64>, new_data: &ArrayD<f64>) -> usize {
    let ex = existing.shape();
    let nw = new_data.shape();
    if ex.len() == nw.len() {
        let differing: Vec<usize> = ex.iter().zip(nw.iter()).enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| i)
            .collect();
        if differing.len() == 1 { return differing[0]; }
    }
    0
}

fn apply_binop(op: BinOp, l: &Value, r: &Value) -> EvalResult<Value> {
    use BinOp::*;
    // String concatenation: str + anything or anything + str
    if matches!(op, Add) {
        match (l, r) {
            (Value::Str(a), Value::Str(b)) => return Ok(Value::Str(format!("{}{}", a, b))),
            (Value::Str(a), other) => return Ok(Value::Str(format!("{}{}", a, other))),
            (other, Value::Str(b)) => return Ok(Value::Str(format!("{}{}", other, b))),
            _ => {}
        }
    }
    match op {
        Add | Sub | Mul | Div | Mod | Pow | StarStar => scalar_arith(op, l, r),
        DotAdd | DotSub | DotMul | DotDiv | DotPow => {
            // #275: a dotted arithmetic op on two scalars is the scalar op — the
            // checker types `5.0 ./ 2.0` as a scalar f32, and the JIT computes it
            // as such; don't promote to a rank-0 tensor here. Only when an operand
            // is an actual tensor does this become an elementwise tensor op.
            let scalar_pair = !matches!(l, Value::Tensor(_)) && !matches!(r, Value::Tensor(_))
                && l.as_float().is_some() && r.as_float().is_some();
            if scalar_pair {
                // #501 S16: `Pow` here is a transient dispatch tag handed
                // straight to `scalar_arith` — it is never stored in an AST
                // node and must never reach `fmt`, which has no source
                // spelling for it (see `fmt::reserved_binop`).
                let base = match op {
                    DotAdd => Add, DotSub => Sub, DotMul => Mul, DotDiv => Div,
                    _ => Pow, // DotPow
                };
                return scalar_arith(base, l, r);
            }
            tensor_elementwise(op, l, r)
        }
        DotGt | DotLt | DotGe | DotLe => tensor_elementwise(op, l, r),
        Matmul => tensor_matmul(l, r),
        Eq | NotEq | Lt | Gt | LtEq | GtEq => scalar_compare(op, l, r),
        And => Ok(Value::Bool(l.as_bool().unwrap_or(false) && r.as_bool().unwrap_or(false))),
        Or  => Ok(Value::Bool(l.as_bool().unwrap_or(false) || r.as_bool().unwrap_or(false))),
        // Pipe is handled in eval_expr (needs call_value); shouldn't reach apply_binop.
        // RShift is the pipe-era variant for `>>`; that spelling parses to
        // `BitShr` since #530, so nothing constructs `RShift` at all.
        Pipe | RShift => Ok(r.clone()),
        // #540: the result carries the operands' width. `& | ^` on two values
        // already inside the width cannot leave it, so the wrap is a no-op on
        // the value — but the TAG has to propagate, or the next `+` widens.
        BitAnd => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("& requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("& requires int, got {}", r.type_name())))?;
            let w = IW::join(int_width(l), int_width(r));
            Ok(Value::Int(w.wrap(a & b), w))
        }
        BitOr => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("| requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("| requires int, got {}", r.type_name())))?;
            let w = IW::join(int_width(l), int_width(r));
            Ok(Value::Int(w.wrap(a | b), w))
        }
        BitXor => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("^ requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("^ requires int, got {}", r.type_name())))?;
            let w = IW::join(int_width(l), int_width(r));
            Ok(Value::Int(w.wrap(a ^ b), w))
        }
        BitShl => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("<< requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("<< requires int, got {}", r.type_name())))?;
            // #215: the shift amount was cast `as u32`, so a negative or >=64 RHS
            // silently wrapped/masked (and panicked in debug builds). Validate it.
            // #540: the legal range is the SHIFTED VALUE's width — an i32 shift
            // tops out at 31 — and the result wraps at that width. Both match
            // the JIT's `shift_range_check`, which sizes its guard the same way.
            let w = IW::join(int_width(l), int_width(r));
            if !(0..w.bits() as i64).contains(&b) {
                Err(RuntimeError::msg(format!(
                    "<< shift amount {} out of range (expected 0..={})", b, w.bits() - 1)))
            } else {
                Ok(Value::Int(w.wrap(a.wrapping_shl(b as u32)), w))
            }
        }
        BitShr => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!(">> requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!(">> requires int, got {}", r.type_name())))?;
            // Arithmetic (sign-propagating) shift for a SIGNED width, matching
            // the JIT's `sshr`: `-8 >> 1` is -4 and `-1 >> 63` is -1, not 0.
            // #597: an UNSIGNED width shifts logically (zero-fill) instead —
            // `OPERATORS.md §8a` names arithmetic `>>` for signed operands and
            // logical for unsigned. `U8`/`U16`/`U32` never told the two apart
            // because their zero-extended pattern is already non-negative, so
            // Rust's sign-propagating `>>` on it happened to zero-fill anyway;
            // `U64` is the first width where a negative i64 pattern (the top
            // bit set) can occur, so it needs the explicit unsigned shift.
            // #215: the shift amount is range-guarded exactly as `<<` is —
            // Rust's `>>` panics in debug and masks mod-width in release
            // outside the range. #540 made that range the operands' joined
            // width, so an i32 shift is judged against 0..=31.
            let w = IW::join(int_width(l), int_width(r));
            if !(0..w.bits() as i64).contains(&b) {
                Err(RuntimeError::msg(format!(
                    ">> shift amount {} out of range (expected 0..={})", b, w.bits() - 1)))
            } else if w.is_signed() {
                Ok(Value::Int(w.wrap(w.wrap(a) >> b), w))
            } else {
                Ok(Value::Int(w.wrap(((w.wrap(a) as u64) >> b) as i64), w))
            }
        }
    }
}

/// `a OP b` at integer width `w`, for `+ - * / % **` — the one implementation
/// of SPEC §3.1's fixed-width integer arithmetic.
///
/// #540 wrote it for scalars; #571 made the elementwise path call the same
/// function, because a tensor element's width is part of its type exactly as a
/// local's is, and two wrap implementations would drift. Everything computes in
/// i64 and wraps once at the end. For `+ - * **` that is exactly arithmetic at
/// the narrow width, because reduction mod 2^w is a ring homomorphism, so
/// wrapping late equals wrapping early — as long as the i64 computation itself
/// does not overflow, which `**` reports rather than wrapping (the JIT does not
/// lower integer `**` at all, so nothing diverges). `/` and `%` cannot leave the
/// range their operands are already in, except `MIN / -1`, guarded at the width.
///
/// `pub(crate)` (#718 rework, R1): `jit.rs`'s local-`let` shape-dimension fold
/// (`fold_shape_expr_at_width`) calls this directly for a binding whose width
/// isn't `i64`, rather than re-deriving the wrap rule — a local literal `let`
/// used as a shape dimension must fold in the SAME width `dmc run` computes
/// in, and the one implementation this comment already insists on is the way
/// to guarantee that instead of hoping a second one stays in sync.
pub(crate) fn int_arith(op: BinOp, a: i64, b: i64, w: IW) -> EvalResult<i64> {
    use BinOp::*;
    let (a, b) = (w.wrap(a), w.wrap(b));
    // Integer power needs fallible handling (#215): the exponent was cast via
    // `as u32`, silently truncating exponents > u32::MAX and wrapping on
    // i64 overflow. Validate the exponent range and use checked_pow.
    if matches!(op, Pow | StarStar) {
        if b < 0 || b > u32::MAX as i64 {
            return Err(RuntimeError::msg(format!(
                "integer exponent {} out of range (expected 0..={})", b, u32::MAX)));
        }
        return match a.checked_pow(b as u32) {
            Some(v) => Ok(w.wrap(v)),
            None => Err(RuntimeError::msg(format!(
                "integer overflow: {} ** {} exceeds the i64 range", a, b))),
        };
    }
    let v = match op {
        // 2's-complement wrap, matching the JIT's iadd/isub/imul (#300 ruling:
        // overflow wraps everywhere — systems-language norm, no per-op cost).
        // div-by-zero stays 0 (#208) and INT_MIN/-1 stays a trap (below), the
        // two cases that have no defined wrapped value.
        Add => a.wrapping_add(b),
        Sub => a.wrapping_sub(b),
        Mul => a.wrapping_mul(b),
        // #544: `MIN / -1` is a SIGNED-only overflow. At an unsigned width
        // narrower than 64 bits both operands are zero-extended and
        // non-negative, so `-1` can never be the divisor there and the signed
        // `wrapping_div` on two non-negative i64s IS the unsigned division.
        // #597: `U64` breaks that equivalence — its pattern can be a negative
        // i64 (the top bit set), so an unsigned width now divides/rems by
        // re-interpreting both operands as `u64` explicitly rather than
        // reusing the signed op. `u64::wrapping_div`/`wrapping_rem` have no
        // `MIN / -1` case (unsigned has no most-negative value to overflow),
        // matching `is_signed`'s guard staying off for every unsigned width.
        Div => {
            if b == 0 { 0 }
            else if w.is_signed() {
                if a == w.min() && b == -1 {
                    return Err(RuntimeError::msg(format!(
                        "integer overflow: {} / {} exceeds the {} range", a, b, w.name())));
                }
                a.wrapping_div(b)
            }
            else { (a as u64).wrapping_div(b as u64) as i64 }
        },
        Mod => {
            if b == 0 { 0 }
            else if w.is_signed() {
                if a == w.min() && b == -1 {
                    return Err(RuntimeError::msg(format!(
                        "integer overflow: {} % {} exceeds the {} range", a, b, w.name())));
                }
                a.wrapping_rem(b)
            }
            else { (a as u64).wrapping_rem(b as u64) as i64 }
        },
        _ => unreachable!(),
    };
    Ok(w.wrap(v))
}

fn scalar_arith(op: BinOp, l: &Value, r: &Value) -> EvalResult<Value> {
    use BinOp::*;
    // int + int = int; otherwise float-promote
    if let (Some(a), Some(b)) = (l.as_int(), r.as_int()) {
        // #540: the result's width is the operands' — SPEC §3.1, arithmetic on
        // a fixed-width integer type wraps at that width.
        let w = IW::join(int_width(l), int_width(r));
        return Ok(Value::Int(int_arith(op, a, b, w)?, w));
    }
    if let (Some(a), Some(b)) = (l.as_float(), r.as_float()) {
        let v = match op {
            Add => a + b, Sub => a - b, Mul => a * b,
            Div => a / b, Mod => a % b,
            Pow | StarStar => a.powf(b),
            _ => unreachable!(),
        };
        // #473: f32 operands give an f32 result, rounded — which is bit-exact
        // with native f32 for `+ - * /` (see `FW`). `Pow` is the documented
        // exception carried over from #241: `powf` in f64 then rounded can
        // differ from the JIT's f32 `powf` by up to 1 ulp.
        let w = FW::join(float_width(l), float_width(r));
        return Ok(Value::Float(w.round(v), w));
    }
    Err(RuntimeError::msg(format!(
        "{:?} requires numeric operands, got {} and {}", op, l.type_name(), r.type_name()
    )))
}

fn scalar_compare(op: BinOp, l: &Value, r: &Value) -> EvalResult<Value> {
    use BinOp::*;
    // Compare two integers exactly as i64. Demoting both to f64 (the path
    // below) loses precision above 2^53 and disagrees with the JIT's integer
    // compare (#296): e.g. 9007199254740993 == 9007199254740992 wrongly
    // returned true. Mixed int/float still falls through to the f64 path.
    //
    // #597: equality is bit-for-bit either way, but ORDERING is not — a `u64`
    // at or above 2^63 is a negative `i64` pattern, so a signed `<`/`>` gets it
    // backwards (`(-1 as u64) > (1 as u64)` must be true: u64::MAX > 1). Narrower
    // unsigned widths (`u8`/`u16`/`u32`) never showed this because their
    // zero-extended pattern is already non-negative, so comparing the pattern
    // as `u64` here changes nothing for them and fixes it for `u64`.
    if let (Value::Int(a, wa), Value::Int(b, wb)) = (l, r) {
        // #597 (review): key the unsigned reading on the operands' OWN tags,
        // not on `IW::join`. `join` resolves an untyped literal's `I64` against
        // a narrow operand by taking the NARROWER width, so a genuine `i64` met
        // by a `u8` would join to `U8` and compare here as unsigned — on the
        // RAW patterns, since (unlike `int_arith`) this path does not `wrap`
        // its operands first, which would make `-1 > 200` true. §3.1's ban on
        // mixed widths means the checker rejects that pairing before it ever
        // gets here, so it is latent rather than live; keying on `U64` closes
        // it anyway. Nothing else changes: `u8`/`u16`/`u32` are stored
        // zero-extended, so the signed comparison of their pattern already IS
        // the unsigned one, and equality is bit-for-bit at every width.
        let unsigned = *wa == IW::U64 || *wb == IW::U64;
        return Ok(Value::Bool(if !unsigned {
            match op {
                Eq    => a == b, NotEq => a != b,
                Lt    => a < b,  Gt    => a > b,
                LtEq  => a <= b, GtEq  => a >= b,
                _ => unreachable!(),
            }
        } else {
            let (ua, ub) = (*a as u64, *b as u64);
            match op {
                Eq    => ua == ub, NotEq => ua != ub,
                Lt    => ua < ub,  Gt    => ua > ub,
                LtEq  => ua <= ub, GtEq  => ua >= ub,
                _ => unreachable!(),
            }
        }));
    }
    if let (Some(a), Some(b)) = (l.as_float(), r.as_float()) {
        return Ok(Value::Bool(match op {
            Eq    => a == b, NotEq => a != b,
            Lt    => a < b,  Gt    => a > b,
            LtEq  => a <= b, GtEq  => a >= b,
            _ => unreachable!(),
        }));
    }
    // nil, string, bool: only == and != are defined; ordering operators are a type error
    match (l, r) {
        (Value::Nil, Value::Nil) => return match op {
            Eq    => Ok(Value::Bool(true)),
            NotEq => Ok(Value::Bool(false)),
            _ => Err(RuntimeError::msg("ordering operators not defined for nil")),
        },
        (Value::Nil, _) | (_, Value::Nil) => return match op {
            Eq    => Ok(Value::Bool(false)),
            NotEq => Ok(Value::Bool(true)),
            _ => Err(RuntimeError::msg("ordering operators not defined for nil")),
        },
        _ => {}
    }
    match (l, r) {
        (Value::Str(a), Value::Str(b)) => match op {
            Eq    => Ok(Value::Bool(a == b)),
            NotEq => Ok(Value::Bool(a != b)),
            _ => Err(RuntimeError::msg(format!(
                "operator {:?} is not defined for str", op
            ))),
        },
        (Value::Bool(a), Value::Bool(b)) => match op {
            Eq    => Ok(Value::Bool(a == b)),
            NotEq => Ok(Value::Bool(a != b)),
            _ => Err(RuntimeError::msg(format!(
                "operator {:?} is not defined for bool", op
            ))),
        },
        _ => Err(RuntimeError::msg(format!(
            "comparison requires numeric, str, or bool operands; got {} and {}",
            l.type_name(), r.type_name()
        ))),
    }
}

fn numpy_broadcast(a: ArrayD<f64>, b: ArrayD<f64>) -> EvalResult<(ArrayD<f64>, ArrayD<f64>)> {
    if a.shape() == b.shape() {
        return Ok((a, b));
    }
    let ndim = a.ndim().max(b.ndim());
    // Pad shorter shape with leading 1s.
    let a_shape: Vec<usize> = std::iter::repeat(1).take(ndim - a.ndim())
        .chain(a.shape().iter().copied()).collect();
    let b_shape: Vec<usize> = std::iter::repeat(1).take(ndim - b.ndim())
        .chain(b.shape().iter().copied()).collect();
    // Compute output shape; fail on incompatible dimensions.
    let mut out_shape = vec![0usize; ndim];
    for i in 0..ndim {
        match (a_shape[i], b_shape[i]) {
            (x, y) if x == y => out_shape[i] = x,
            (1, y) => out_shape[i] = y,
            (x, 1) => out_shape[i] = x,
            (x, y) => return Err(RuntimeError::msg(format!(
                "elementwise broadcast: axis {} has incompatible sizes {} vs {}", i, x, y,
            ))),
        }
    }
    let out_dim = ndarray::IxDyn(&out_shape);
    // Reshape to add leading 1-dims, then let ndarray broadcast to the output shape.
    let a = a.into_shape_with_order(ndarray::IxDyn(&a_shape))
        .map_err(|e| RuntimeError::msg(format!("broadcast reshape: {}", e)))?;
    let b = b.into_shape_with_order(ndarray::IxDyn(&b_shape))
        .map_err(|e| RuntimeError::msg(format!("broadcast reshape: {}", e)))?;
    let a_out = a.broadcast(out_dim.clone())
        .ok_or_else(|| RuntimeError::msg("broadcast failed for lhs"))?.to_owned();
    let b_out = b.broadcast(out_dim)
        .ok_or_else(|| RuntimeError::msg("broadcast failed for rhs"))?.to_owned();
    Ok((a_out, b_out))
}

// Raise a tensor to a uniform scalar power, skipping per-element `powf`
// (and the constant-array materialization) for the exponents real code
// actually uses — `.^0.5` is sqrt, `.^2.0` is a multiply, etc.
fn pow_by_scalar(a: &ArrayD<f64>, c: f64) -> ArrayD<f64> {
    if c == 0.5       { a.mapv(f64::sqrt) }
    else if c == 1.0  { a.clone() }
    else if c == 2.0  { a.mapv(|x| x * x) }
    else if c == 3.0  { a.mapv(|x| x * x * x) }
    else if c == -1.0 { a.mapv(|x| 1.0 / x) }
    else if c == -0.5 { a.mapv(|x| 1.0 / x.sqrt()) }
    else if c.fract() == 0.0 && c.abs() <= 64.0 { let n = c as i32; a.mapv(|x| x.powi(n)) }
    else { a.mapv(|x| x.powf(c)) }
}

// Apply a scalar `s` on the right of a tensor `a` in a single pass —
// `a .+ s`, `a .* s`, `a .< s`, … — without materializing a constant
// array. Renderer/sim code is dominated by tensor-vs-scalar ops, so this
// is the hot path.
fn scalar_rhs(op: BinOp, a: &ArrayD<f64>, s: f64) -> ArrayD<f64> {
    use BinOp::*;
    match op {
        DotAdd => a.mapv(|x| x + s),
        DotSub => a.mapv(|x| x - s),
        DotMul => a.mapv(|x| x * s),
        DotDiv => a.mapv(|x| x / s),
        DotPow => pow_by_scalar(a, s),
        DotGt => a.mapv(|x| (x >  s) as i64 as f64),
        DotLt => a.mapv(|x| (x <  s) as i64 as f64),
        DotGe => a.mapv(|x| (x >= s) as i64 as f64),
        DotLe => a.mapv(|x| (x <= s) as i64 as f64),
        _ => unreachable!(),
    }
}

// Scalar `s` on the left of a tensor `b`: `s .- b`, `s ./ b`, … — order
// matters for non-commutative ops and comparisons.
fn scalar_lhs(op: BinOp, s: f64, b: &ArrayD<f64>) -> ArrayD<f64> {
    use BinOp::*;
    match op {
        DotAdd => b.mapv(|x| s + x),
        DotSub => b.mapv(|x| s - x),
        DotMul => b.mapv(|x| s * x),
        DotDiv => b.mapv(|x| s / x),
        DotPow => b.mapv(|x| s.powf(x)),
        DotGt => b.mapv(|x| (s >  x) as i64 as f64),
        DotLt => b.mapv(|x| (s <  x) as i64 as f64),
        DotGe => b.mapv(|x| (s >= x) as i64 as f64),
        DotLe => b.mapv(|x| (s <= x) as i64 as f64),
        _ => unreachable!(),
    }
}

/// Float width of a tensor-op result (#241): `F32` (rounded through f32 at
/// construction, matching the JIT's f32 lanes) only when every tensor operand
/// is f32-family — `F32` or `Trit` (trit values −1/0/1 are exact in f32, and
/// the JIT computes trit math as f32). An `F64` operand keeps the result wide,
/// matching the JIT's f64 tensors; an `Int` operand also stays wide, so large
/// integer values (beyond f32's 2^24 exact range) are not corrupted. Scalar
/// (non-tensor) operands follow the tensor side.
fn float_result_dtype(operands: &[&Value]) -> DType {
    let narrow = |v: &&Value| match v {
        // #834: fp8 operands narrow to a plain `F32` result exactly like
        // `Trit` operands already did — the specific narrow tag never
        // survives a binop, only whether the operands were narrow at all.
        Value::Tensor(t) => matches!(t.dtype, DType::F32 | DType::Trit | DType::Fp8E4M3 | DType::Fp8E5M2),
        _ => true,
    };
    if operands.iter().all(narrow) { DType::F32 } else { DType::F64 }
}

/// The integer element WIDTH an elementwise operand contributes (#571): a
/// tensor's element width, or a scalar integer's own. `None` for anything
/// float-valued, which sends the op down the f64 path where it belongs.
#[inline]
fn int_operand_width(v: &Value) -> Option<IW> {
    match v {
        Value::Tensor(t) => match t.dtype { DType::Int(w) => Some(w), _ => None },
        Value::Int(_, w) => Some(*w),
        _ => None,
    }
}

/// `a .OP b` over integer elements (#571): integer arithmetic at the element
/// width, element by element, through the same `int_arith` the scalar `a[0] OP
/// b[0]` goes through.
///
/// Computing these in the f64 backing lanes — what the interpreter did before —
/// is wrong twice over: it does not wrap (`2147483647 .+ 1` on `Tensor[i32]`
/// answered 2147483648, which is not an i32), and it is not integer division
/// (`.\/` gave 0.5 where `/` gives 0, and ∞ where §3.1 defines 0). A product of
/// two large `i64` lanes also silently lost precision f64 cannot carry.
fn tensor_elementwise_int(op: BinOp, l: &Value, r: &Value, w: IW) -> EvalResult<Value> {
    use BinOp::*;
    // The scalar op the dotted spelling means. `Pow` is the transient dispatch
    // tag `apply_binop` already hands to `scalar_arith` for `.^` (#501 S16).
    let base = match op {
        DotAdd => Add, DotSub => Sub, DotMul => Mul, DotDiv => Div,
        _ => Pow, // DotPow
    };
    let a = as_tensor(l)?;
    let b = as_tensor(r)?;
    // A scalar operand is a rank-0 array here, which `numpy_broadcast` pads to
    // the other side's rank like any shorter shape — so `t .+ 1` needs no
    // separate path, and there is one loop to keep correct.
    let (a, b) = if a.shape() == b.shape() { (a, b) } else { numpy_broadcast(a, b)? };
    let mut out = a.clone();
    for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
        *o = int_arith(base.clone(), *x as i64, *y as i64, w)? as f64;
    }
    Ok(Value::tensor_dt(out, DType::Int(w)))
}

fn tensor_elementwise(op: BinOp, l: &Value, r: &Value) -> EvalResult<Value> {
    use BinOp::*;
    // #571: an elementwise op over integer elements is INTEGER arithmetic at
    // the element width, not f64 arithmetic over the f64-backed lanes. `a .+ b`
    // on two `Tensor[i32, …]` wraps at 32 bits exactly as `a[0] + b[0]` does,
    // and through the same function. Comparisons stay on the float path below:
    // they read the (already wrapped) lane values and produce a 0/1 mask, which
    // the element width does not change.
    if matches!(op, DotAdd | DotSub | DotMul | DotDiv | DotPow) {
        if let (Some(wl), Some(wr)) = (int_operand_width(l), int_operand_width(r)) {
            return tensor_elementwise_int(op, l, r, IW::join(wl, wr));
        }
    }
    let dt = float_result_dtype(&[l, r]);
    let a = as_tensor(l)?;
    let b = as_tensor(r)?;
    // Scalar-broadcast fast paths: one pass with the scalar folded in, no
    // constant-array materialization. Covers `a .op scalar` either way.
    // In an F32 op the scalar is demoted to f32 first, as the JIT does when
    // it splats a scalar into f32 lanes — without this, `a .* 0.1` would
    // multiply by the f64 0.1 and double-round 1 ulp away from the JIT.
    if b.ndim() == 0 {
        let mut s = b.iter().next().copied().unwrap_or(0.0);
        if dt == DType::F32 { s = quantize_f32(s); }
        return Ok(Value::tensor_dt(scalar_rhs(op, &a, s), dt));
    }
    if a.ndim() == 0 {
        let mut s = a.iter().next().copied().unwrap_or(0.0);
        if dt == DType::F32 { s = quantize_f32(s); }
        return Ok(Value::tensor_dt(scalar_lhs(op, s, &b), dt));
    }
    // Equal shapes: operate directly, skipping the broadcast clones.
    let (a, b) = if a.shape() == b.shape() {
        (a, b)
    } else {
        numpy_broadcast(a, b)?
    };
    let out = match op {
        DotAdd => &a + &b,
        DotSub => &a - &b,
        DotMul => &a * &b,
        DotDiv => &a / &b,
        DotPow => {
            let mut out = a.clone();
            for (o, e) in out.iter_mut().zip(b.iter()) { *o = o.powf(*e); }
            out
        }
        DotGt | DotLt | DotGe | DotLe => {
            let mut out = a.clone();
            for (o, (x, y)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
                *o = match op {
                    DotGt => (x >  y) as i64 as f64,
                    DotLt => (x <  y) as i64 as f64,
                    DotGe => (x >= y) as i64 as f64,
                    DotLe => (x <= y) as i64 as f64,
                    _ => unreachable!(),
                };
            }
            out
        }
        _ => unreachable!(),
    };
    Ok(Value::tensor_dt(out, dt))
}

/// `A(m,k) @ B(k,n)` accumulated with FMA at f32 width (#481).
///
/// `ndarray`'s `dot` accumulates in f64 (and blocks internally), which is a
/// different number from what the JIT computes. All three JIT matmul kernels
/// contract with FMA over k ascending — one rounding per multiply-add — so the
/// interpreter must too, and `f32::mul_add` is exactly that: a single correctly
/// rounded operation, with a software fallback where the host has no FMA unit,
/// so the answer does not depend on the machine.
///
/// Deliberately NOT `ndarray::dot`: this is the reference implementation, and
/// being the same number as the JIT matters more here than being fast.
fn gemm_f32_fma(
    a: &ndarray::ArrayView2<'_, f64>,
    b: &ndarray::ArrayView2<'_, f64>,
) -> ndarray::Array2<f64> {
    let (m, k) = (a.nrows(), a.ncols());
    let n = b.ncols();
    let mut out = ndarray::Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc = (a[[i, kk]] as f32).mul_add(b[[kk, j]] as f32, acc);
            }
            out[[i, j]] = acc as f64;
        }
    }
    out
}

fn tensor_matmul(l: &Value, r: &Value) -> EvalResult<Value> {
    // #241: width of the product follows the operands. #481: an f32 product
    // accumulates with FMA at f32 width, matching all three JIT kernels; other
    // dtypes keep `ndarray`'s f64 dot.
    let dt = float_result_dtype(&[l, r]);
    // Trit operands: unpack to plain f64 (values are already -1/0/1) so the
    // standard matmul path handles them without modification.
    let l_unpacked;  let r_unpacked;
    let l = if let Value::Tensor(t) = l {
        if t.dtype == DType::Trit {
            l_unpacked = Value::tensor_dt(t.data.clone(), DType::F32);
            &l_unpacked
        } else { l }
    } else { l };
    let r = if let Value::Tensor(t) = r {
        if t.dtype == DType::Trit {
            r_unpacked = Value::tensor_dt(t.data.clone(), DType::F32);
            &r_unpacked
        } else { r }
    } else { r };
    let a = as_tensor(l)?;
    let b = as_tensor(r)?;
    if a.ndim() < 2 || b.ndim() < 2 {
        return Err(RuntimeError::msg(format!(
            "matmul requires rank>=2; got ranks {} and {}", a.ndim(), b.ndim(),
        )));
    }
    // 2D fast path.
    if a.ndim() == 2 && b.ndim() == 2 {
        let a2 = a.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| RuntimeError::msg(format!("matmul: {}", e)))?;
        let b2 = b.view().into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| RuntimeError::msg(format!("matmul: {}", e)))?;
        if a2.ncols() != b2.nrows() {
            return Err(RuntimeError::msg(format!(
                "matmul inner dims: {} vs {}", a2.ncols(), b2.nrows(),
            )));
        }
        let prod = if dt == DType::F32 { gemm_f32_fma(&a2, &b2) } else { a2.dot(&b2) };
        return Ok(Value::tensor_dt(prod.into_dyn(), dt));
    }
    // Batched matmul. Leading dims are batch dims and broadcast like
    // NumPy/PyTorch; the trailing (M, K) × (K, N) pair is multiplied per
    // output batch coordinate.
    let a_shape = a.shape().to_vec();
    let b_shape = b.shape().to_vec();
    let (a_batch, a_mk) = a_shape.split_at(a.ndim() - 2);
    let (b_batch, b_kn) = b_shape.split_at(b.ndim() - 2);
    let out_batch = broadcast_batch_shape(a_batch, b_batch)?;
    let (m, k_a) = (a_mk[0], a_mk[1]);
    let (k_b, n) = (b_kn[0], b_kn[1]);
    if k_a != k_b {
        return Err(RuntimeError::msg(format!("matmul inner dims: {} vs {}", k_a, k_b)));
    }
    let a_batch_size: usize = a_batch.iter().product();
    let b_batch_size: usize = b_batch.iter().product();
    let out_batch_size: usize = out_batch.iter().product();
    // Force contiguous (standard row-major) layout — a transpose produces a
    // non-contiguous view that reshape rejects.
    let a_std = a.as_standard_layout().to_owned();
    let b_std = b.as_standard_layout().to_owned();
    let a_flat = a_std.into_shape_with_order((a_batch_size, m, k_a))
        .map_err(|e| RuntimeError::msg(format!("matmul reshape: {}", e)))?;
    let b_flat = b_std.into_shape_with_order((b_batch_size, k_b, n))
        .map_err(|e| RuntimeError::msg(format!("matmul reshape: {}", e)))?;
    let mut out_flat = ndarray::Array3::<f64>::zeros((out_batch_size, m, n));
    for i in 0..out_batch_size {
        let out_idx = unravel_index(i, &out_batch);
        let a_i = broadcast_source_index(&out_idx, &out_batch, a_batch);
        let b_i = broadcast_source_index(&out_idx, &out_batch, b_batch);
        let a_slice = a_flat.index_axis(Axis(0), a_i);
        let b_slice = b_flat.index_axis(Axis(0), b_i);
        let prod = if dt == DType::F32 {
            gemm_f32_fma(&a_slice, &b_slice)
        } else {
            a_slice.dot(&b_slice)
        };
        out_flat.index_axis_mut(Axis(0), i).assign(&prod);
    }
    let mut out_shape = out_batch;
    out_shape.push(m);
    out_shape.push(n);
    let out = out_flat.into_shape_with_order(IxDyn(&out_shape))
        .map_err(|e| RuntimeError::msg(format!("matmul reshape: {}", e)))?
        .into_dyn();
    Ok(Value::tensor_dt(out, dt))
}

fn broadcast_batch_shape(a: &[usize], b: &[usize]) -> EvalResult<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for i in 0..rank {
        let a_dim = dim_aligned_from_right(a, rank, i);
        let b_dim = dim_aligned_from_right(b, rank, i);
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            return Err(RuntimeError::msg(format!(
                "batched matmul: batch dims cannot broadcast: {:?} vs {:?}", a, b,
            )));
        }
        out.push(a_dim.max(b_dim));
    }
    Ok(out)
}

fn dim_aligned_from_right(shape: &[usize], rank: usize, out_axis: usize) -> usize {
    if out_axis < rank - shape.len() {
        1
    } else {
        shape[out_axis - (rank - shape.len())]
    }
}

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        out[i] = index % shape[i];
        index /= shape[i];
    }
    out
}

fn broadcast_source_index(out_idx: &[usize], out_shape: &[usize], source_shape: &[usize]) -> usize {
    if source_shape.is_empty() {
        return 0;
    }
    let offset = out_shape.len() - source_shape.len();
    let mut flat = 0;
    for (i, dim) in source_shape.iter().enumerate() {
        let coord = if *dim == 1 { 0 } else { out_idx[offset + i] };
        flat = flat * *dim + coord;
    }
    flat
}

/// #476: store `rval` into slot `i` of a model array (`Value::List`) held in a
/// struct field, given an already-resolved index vector. Mirrors the local
/// `arr[i] = instance` path — single scalar index, negatives wrap from the end.
///
/// The caller holds a `borrow_mut()` on the owning struct, so the write lands
/// in the field itself rather than in a copy: this is what separates the
/// working `h.cells[i] = ..` spelling from `let !cs = h.cells` (a model-array
/// field binds by value, so a fill through the alias never reaches the field).
fn assign_into_model_array(
    slot: &mut Value,
    idx_vals: &[IndexVal],
    rval: Value,
    span: &Span,
) -> EvalResult<()> {
    let Value::List(buf) = slot else {
        return Err(RuntimeError::at(
            format!("indexed assignment requires a tensor, got {}", slot.type_name()),
            span.clone(),
        ));
    };
    let raw = match idx_vals {
        [IndexVal::Index(i)] => *i,
        _ => return Err(RuntimeError::at(
            "model-array assignment requires a single scalar index", span.clone())),
    };
    let len = buf.len() as i64;
    let i = if raw < 0 { len + raw } else { raw };
    if i < 0 || i >= len {
        return Err(RuntimeError::at(
            format!("index {} out of bounds for axis 0 of size {}", raw, len), span.clone()));
    }
    Rc::make_mut(buf)[i as usize] = rval;
    Ok(())
}

/// Write back the final values of `!`-prefixed parameters to the caller's scope.
///
/// `pending` holds `(param_index, final_value)` pairs populated by
/// `call_fn_with_shapes` before it pops the callee's scope. For each entry,
/// if the corresponding call argument was a bare identifier, the identifier is
/// updated in the caller's current scope stack.
///
/// `self_params` is the number of leading callee parameters that have no
/// counterpart in `call_args`. It is 0 for a free function and 1 for a model
/// method, where the receiver is prepended to the argument vector but written
/// at the call site as `recv.m(args)` — so callee param `i` is call arg
/// `i - self_params`. The receiver itself never needs a writeback: a
/// `Value::Struct` is an `Rc<RefCell<..>>` the callee already shares.
///
/// #475: methods used to skip this step entirely. Tensors are value-copies in
/// the interpreter (see `Value::Tensor`), so a `!` tensor parameter only
/// aliases by virtue of this writeback. Without it a method's writes through
/// an out-parameter were silently dropped while the same body as a free
/// function worked — no error, correct return value, lost mutation.
fn apply_writebacks(
    scopes: &mut Vec<HashMap<String, Value>>,
    pending: &mut Vec<(usize, Value)>,
    call_args: &[CallArg],
    self_params: usize,
) {
    for (idx, val) in pending.drain(..) {
        let Some(arg_idx) = idx.checked_sub(self_params) else { continue };
        if let Some(CallArg::Positional(Expr::Ident(name, _))) = call_args.get(arg_idx) {
            for scope in scopes.iter_mut().rev() {
                if scope.contains_key(name.as_str()) {
                    scope.insert(name.clone(), val);
                    break;
                }
            }
        }
    }
}

/// Returns the ShapeSpec of a Tensor/View/KV type, or None for scalars.
fn type_shape_spec(ty: &Type) -> Option<&ShapeSpec> {
    match ty {
        Type::Tensor(_, s, _) | Type::View(_, s, _) | Type::KV(_, s, _) => Some(s),
        _ => None,
    }
}

fn streaming_axis_from_type(ty: &Type) -> Option<usize> {
    let spec = type_shape_spec(ty)?;
    spec.elems.iter().position(|elem| matches!(elem, ShapeElem::Streaming(_)))
}

/// Bring a `<-` appendee up to the stream's rank.
///
/// `SPEC.md §4.8` accepts two spellings of the appendee for a `KV[T, S]`: `S`
/// with the streaming axis carrying the appended extent, or `S` with that axis
/// **dropped**, which appends a single frame. Only the first ever worked —
/// `ndarray::concatenate` requires equal rank, so the dropped form died as
/// `ShapeError/IncompatibleShape` (probe p31). Appending one `[4, 8]` token
/// into a `KV[f32, [4, ~, 8]]` cache is the shape the spec leads with and the
/// shape a decode loop actually writes.
///
/// Promoting here rather than at the `concatenate` call also fixes the capacity
/// check, which reads the appended extent off `shape()[axis]`: for a dropped-form
/// `[4, 8]` that is `shape()[1]` — 8, the *feature* width — so a 64-frame cache
/// would have reported itself full after 8 appends.
fn promote_stream_appendee(existing: &ArrayD<f64>, new: &ArrayD<f64>, axis: usize)
    -> Result<ArrayD<f64>, String>
{
    if new.ndim() == existing.ndim() {
        return Ok(new.clone());
    }
    if new.ndim() + 1 != existing.ndim() {
        return Err(format!(
            "cannot append a {}-d value to a {}-d stream: expected {}-d (streaming axis \
             carrying the appended extent) or {}-d (streaming axis dropped, appending one frame)",
            new.ndim(), existing.ndim(), existing.ndim(), existing.ndim() - 1,
        ));
    }
    let mut want: Vec<usize> = existing.shape().to_vec();
    want.remove(axis);
    if new.shape() != want.as_slice() {
        return Err(format!(
            "cannot append shape {:?} to a {:?} stream: with the streaming axis (axis {}) \
             dropped the appendee must be {:?}",
            new.shape(), existing.shape(), axis, want,
        ));
    }
    Ok(new.clone().insert_axis(Axis(axis)))
}

/// Extract the `capacity = N` literal from a KV constructor expression like
/// `stream.kv[T, shape](capacity = N)`. Returns None if the pattern isn't matched.
/// The streaming-axis (`~`) index of a `x.kv[T, [.., ~, ..]](...)` constructor
/// value, or None if `e` is not such a constructor. Mirrors the `~`→empty-axis
/// handling in `try_arena_constructor` so `<-` can resolve the append axis even
/// when the binding carries no explicit `KV[..~..]` type annotation (#399) —
/// without it, the append falls back to a shape-differ heuristic that picks the
/// wrong axis once frames are equal-shaped (the normal per-step decode case).
fn streaming_axis_from_kv_value(e: &Expr) -> Option<usize> {
    match e {
        // Peel the `(capacity = N)` call, if present.
        Expr::Postfix { expr: inner, op: PostfixOp::Call(_), .. } => {
            streaming_axis_from_kv_value(inner)
        }
        // `x.kv[T, [.., ~, ..]]` — locate the shape TensorLit and the `~` in it.
        Expr::Postfix { expr: recv, op, .. }
            if matches!(recv.as_ref(),
                Expr::Postfix { op: PostfixOp::Field(f), .. } if f == "kv") =>
        {
            let dims = match op {
                PostfixOp::Index(idxs) => idxs.iter().find_map(|el| match el {
                    IndexElem::Expr(Expr::TensorLit(d, _)) => Some(d),
                    _ => None,
                }),
                PostfixOp::BracketArgs(args) => args.iter().find_map(|a| match a {
                    CallArg::Positional(Expr::TensorLit(d, _)) => Some(d),
                    _ => None,
                }),
                _ => None,
            }?;
            dims.iter().position(|d| matches!(d, Expr::Ident(n, _) if n == "~"))
        }
        _ => None,
    }
}

fn extract_kv_capacity(expr: &Expr) -> Option<usize> {
    if let Expr::Postfix { expr: inner, op: PostfixOp::Call(args), .. } = expr {
        for arg in args {
            if let CallArg::Named { name, value, .. } = arg {
                if name == "capacity" {
                    if let Expr::Literal(Literal::Int(n, _), _) = value {
                        if *n > 0 { return Some(*n as usize); }
                    }
                }
            }
        }
        // Recurse into the inner expression in case it's chained.
        return extract_kv_capacity(inner);
    }
    None
}

/// Collect identifier names that appear in a ShapeSpec's elements.
/// Used to find every shape variable referenced in a function's signature,
/// so we can verify they all get bound from argument inference.
fn collect_idents_in_spec(spec: &ShapeSpec, out: &mut std::collections::HashSet<String>) {
    for elem in &spec.elems {
        if let ShapeElem::Expr(e) = elem {
            collect_idents_in_expr(e, out);
        }
    }
}

fn collect_idents_in_expr(e: &Expr, out: &mut std::collections::HashSet<String>) {
    match e {
        Expr::Ident(name, _) => {
            // Skip scalar-type and arena keywords; only shape-variable-like
            // names. Heuristic: uppercase-first identifiers are shape vars
            // (matches the language convention: B, S, D, H, ...).
            if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                out.insert(name.clone());
            }
        }
        Expr::BinOp { lhs, rhs, .. } => {
            collect_idents_in_expr(lhs, out);
            collect_idents_in_expr(rhs, out);
        }
        Expr::UnOp { operand, .. } => collect_idents_in_expr(operand, out),
        _ => {}
    }
}

/// Walk a declared ShapeSpec against an actual tensor's shape, binding any
/// Ident shape variable found at a given position. Conflicts error.
fn infer_shape_from_arg(
    spec: &ShapeSpec,
    actual: &[usize],
    inferred: &mut std::collections::HashMap<String, i64>,
    fn_name: &str,
    sp: &Span,
) -> EvalResult<()> {
    // If the spec contains a Spread (`..`) the bind-from-end semantics apply.
    let has_spread = spec.elems.iter().any(|e| matches!(e, ShapeElem::Spread(_)));
    if has_spread {
        // Bind elements before and after `..` against the tensor's prefix/suffix.
        let split = spec.elems.iter().position(|e| matches!(e, ShapeElem::Spread(_))).unwrap();
        let head = &spec.elems[..split];
        let tail = &spec.elems[split + 1..];
        if actual.len() < head.len() + tail.len() {
            return Err(RuntimeError::at(format!(
                "fn `{}`: tensor of rank {} too small for declared shape ({} fixed dims around `..`)",
                fn_name, actual.len(), head.len() + tail.len()), sp.clone()));
        }
        for (i, e) in head.iter().enumerate() {
            bind_shape_elem(e, actual[i] as i64, inferred, fn_name, sp)?;
        }
        let off = actual.len() - tail.len();
        for (i, e) in tail.iter().enumerate() {
            bind_shape_elem(e, actual[off + i] as i64, inferred, fn_name, sp)?;
        }
    } else {
        if spec.elems.len() != actual.len() { return Ok(()); }  // checker's job to flag
        for (e, &dim) in spec.elems.iter().zip(actual.iter()) {
            bind_shape_elem(e, dim as i64, inferred, fn_name, sp)?;
        }
    }
    Ok(())
}

fn bind_shape_elem(
    elem: &ShapeElem,
    dim: i64,
    inferred: &mut std::collections::HashMap<String, i64>,
    fn_name: &str,
    sp: &Span,
) -> EvalResult<()> {
    let ShapeElem::Expr(e) = elem else { return Ok(()); };
    bind_shape_ident_expr(e, dim, inferred, fn_name, sp)
}

/// Bind `dim` to `e` when `e` is a bare shape variable. Anything else (a
/// literal, an arithmetic expression) is not a binding site — the checker owns
/// verifying those.
fn bind_shape_ident_expr(
    e: &Expr,
    dim: i64,
    inferred: &mut std::collections::HashMap<String, i64>,
    fn_name: &str,
    sp: &Span,
) -> EvalResult<()> {
    let Expr::Ident(name, _) = e else { return Ok(()); };
    bind_shape_name(name, dim, inferred, fn_name, sp)
}

/// The one place a harvested dim meets an already-harvested one. Every source
/// — tensor shape, model-array length, a model argument's own shape args —
/// goes through here, so a disagreement reads the same whichever two
/// arguments produced it.
fn bind_shape_name(
    name: &str,
    dim: i64,
    inferred: &mut std::collections::HashMap<String, i64>,
    fn_name: &str,
    sp: &Span,
) -> EvalResult<()> {
    // Only bind shape-variable-like names (uppercase-first).
    if !name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) { return Ok(()); }
    if let Some(&prev) = inferred.get(name) {
        if prev != dim {
            return Err(RuntimeError::at(format!(
                "fn `{}`: shape param `{}` would need to be both {} and {} \
                (inconsistent argument shapes)",
                fn_name, name, prev, dim), sp.clone()));
        }
    } else {
        inferred.insert(name.to_string(), dim);
    }
    Ok(())
}

/// A postfix bracket read as shape arguments: one `(name?, value, span)` per
/// argument. `Index` is the all-positional spelling (`f[2, 2]`), `BracketArgs`
/// the one with names (`f[SH=2]`); either may in fact be ordinary indexing, so
/// anything that cannot be a shape argument — a slice, `..`, a spread —
/// returns `None` and leaves the expression to the indexing path.
fn shape_bracket_args<'a>(
    op: &'a PostfixOp,
    span: &Span,
) -> Option<Vec<(Option<&'a str>, &'a Expr, Span)>> {
    match op {
        PostfixOp::Index(elems) => elems.iter().map(|e| match e {
            IndexElem::Expr(e) => Some((None, e, span.clone())),
            _ => None,
        }).collect(),
        PostfixOp::BracketArgs(args) => args.iter().map(|a| match a {
            CallArg::Positional(e) => Some((None, e, span.clone())),
            CallArg::Named { name, value, span: nspan } =>
                Some((Some(name.as_str()), value, nspan.clone())),
            CallArg::Spread(_) => None,
        }).collect(),
        _ => None,
    }
}

/// A model type argument written as a bare shape variable. `Box[H, W]` parses
/// `H` as a *type* (an ident can start one) while `Box[2, W]` parses `2` as an
/// expression, so both spellings have to be recognised here.
fn type_arg_shape_var(arg: &TypeArg) -> Option<&str> {
    match arg {
        TypeArg::Type(Type::Named { name, args, .. }) if args.is_empty() => Some(name),
        TypeArg::Expr(e) => match e.as_ref() {
            Expr::Ident(name, _) => Some(name),
            _ => None,
        },
        _ => None,
    }
}

/// The table key for `name` as file number `module` sees it. Every fn, model,
/// enum and `type` alias a file declares or imports is registered under the
/// file's number, so a name resolves only against the file that mentions it
/// (SPEC §6.7, §9). `#` opens a comment and so cannot occur in an identifier
/// or alias; the key never collides with a source name, and `bare_name`
/// recovers the name.
///
/// This allocates a `String` on every resolution — once per call, per model
/// literal, per enum-variant test. Measured against the flat-`HashMap` path it
/// costs nothing at the sizes the interpreter runs at (`fib(29)` 1.89 s →
/// 1.86 s, `tictactoe` 14.1 s → 13.5 s), so it is left plain rather than
/// traded for a `(usize, &str)` key or an interned symbol; the tree walk
/// itself dominates by orders of magnitude. Revisit only behind a measurement
/// that shows otherwise.
fn module_key(module: usize, name: &str) -> String {
    format!("{}#m{}", name, module)
}

/// The source name inside a `module_key`, for diagnostics and display. Only
/// the exact `#m<digits>` suffix is stripped: other runtime strings carry a
/// `#` of their own (a port handle prints as `port#1:python`) and pass
/// through unchanged.
fn bare_name(key: &str) -> &str {
    match key.rsplit_once("#m") {
        Some((name, digits)) if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()) => name,
        _ => key,
    }
}

/// The `__model__` tag on a struct value, if it carries one.
fn struct_model_name(fields: &Rc<RefCell<Vec<(String, Value)>>>) -> Option<String> {
    fields.borrow().iter()
        .find(|(k, _)| k == "__model__")
        .and_then(|(_, v)| match v { Value::Str(s) => Some(s.clone()), _ => None })
}

/// The shape args an instance was constructed with, recovered from the hidden
/// `__shape_P__` fields a model literal writes.
fn model_shape_bindings(fields: &Rc<RefCell<Vec<(String, Value)>>>) -> Vec<(String, i64)> {
    fields.borrow().iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("__shape_")
                .and_then(|s| s.strip_suffix("__"))
                .and_then(|pname| v.as_int().map(|n| (pname.to_string(), n)))
        })
        .collect()
}

fn as_tensor(v: &Value) -> EvalResult<ArrayD<f64>> {
    match v {
        Value::Tensor(t) => Ok(t.data.clone()),
        Value::Int(n, _)   => Ok(ArrayD::from_elem(IxDyn(&[]), *n as f64)),
        Value::Float(x, _) => Ok(ArrayD::from_elem(IxDyn(&[]), *x)),
        _ => Err(RuntimeError::msg(format!("expected tensor or numeric scalar, got {}", v.type_name()))),
    }
}

/// Solve the dense square system `A x = b` (A is n×n, b is length n) and return x.
///
/// Implemented in-house with Gauss–Jordan elimination + partial pivoting (~O(n³))
/// rather than via a LAPACK binding: `ndarray` ships no solver, and pulling in
/// `ndarray-linalg`/BLAS just for the small systems demoniC programs use would be
/// a heavy, platform-fiddly dependency. Partial pivoting (always eliminate using
/// the largest-magnitude available pivot) keeps it numerically sane for the
/// well-conditioned, low-dimension matrices this targets, and makes a singular
/// matrix show up as a near-zero pivot — surfaced as a catchable error rather than
/// a NaN-laden result. Returns `Err("matrix is singular")` in that case.
fn gauss_jordan_solve(a: &ArrayD<f64>, b: &[f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    // Augmented working copy [A | b], row-major.
    let mut m: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row: Vec<f64> = (0..n).map(|j| a[IxDyn(&[i, j])]).collect();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivot: the row at/below `col` with the largest |entry| in this
        // column. Dividing by the biggest available magnitude limits error growth.
        let pivot = (col..n)
            .max_by(|&r1, &r2| m[r1][col].abs().total_cmp(&m[r2][col].abs()))
            .unwrap();
        if m[pivot][col].abs() < 1e-12 {
            return Err("matrix is singular".to_string());
        }
        m.swap(col, pivot);
        // Eliminate `col` from every other row, reducing A to diagonal form.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = m[row][col] / m[col][col];
            for k in col..=n {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    // A is now diagonal, so each unknown is just its augmented entry / its pivot.
    Ok((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

/// Invert a square n×n matrix by solving `A xᵢ = eᵢ` for each identity column eᵢ;
/// the solved columns are the columns of A⁻¹. Reuses [`gauss_jordan_solve`], so a
/// singular matrix propagates the same catchable error.
fn matrix_inverse(a: &ArrayD<f64>, n: usize) -> Result<ArrayD<f64>, String> {
    let mut out = ArrayD::zeros(IxDyn(&[n, n]));
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        let x = gauss_jordan_solve(a, &e)?;
        for (row, xi) in x.into_iter().enumerate() {
            out[IxDyn(&[row, col])] = xi;
        }
    }
    Ok(out)
}

fn apply_unop(op: UnOp, v: &Value) -> EvalResult<Value> {
    use UnOp::*;
    match op {
        Neg => match v {
            // #540: negation is `0 - n`, so it wraps at the operand's width —
            // `-i32::MIN` is `i32::MIN`, the same value the JIT's `ineg` gives.
            Value::Int(n, w)    => Ok(Value::Int(w.wrap(n.wrapping_neg()), *w)),
            Value::Float(x, w)  => Ok(Value::Float(w.round(-x), *w)),
            // Result width follows the operand (#241): an Int/F64 tensor must
            // not be rounded through f32 by the default-F32 construction.
            Value::Tensor(t) => Ok(Value::tensor_dt(t.map(|x| -x), float_result_dtype(&[v]))),
            _ => Err(RuntimeError::msg(format!("- requires numeric, got {}", v.type_name()))),
        }
        Not => Ok(Value::Bool(!v.as_bool().unwrap_or(false))),
        Deref => Ok(v.clone()),  // pre-alpha: no references
        ReLU => match v {
            Value::Tensor(t) => Ok(Value::tensor_dt(t.map(|x| x.max(0.0)), float_result_dtype(&[v]))),
            Value::Float(x, w)  => Ok(Value::Float(w.round(x.max(0.0)), *w)),
            Value::Int(n, w)    => Ok(Value::Int((*n).max(0), *w)),
            _ => Err(RuntimeError::msg(format!("\\> requires numeric/tensor, got {}", v.type_name()))),
        }
        GeLU => match v {
            // #567: `\<` IS the stdlib `gelu()` — OPERATORS.md says so, and the
            // JIT compiles both through one `emit_gelu_f32`. So both spellings
            // go through one kernel here too: `activation_f64`, then the f32
            // rounding `builtin_activation` applies. `\<` used to run its own
            // unrounded f64 kernel, which made `\<(x) == gelu(x)` false under
            // `dmc run` (off by ~1.6e-8) and true under `dmc jit`.
            // Tensors take the builtin's `Value::tensor` construction for the
            // same reason: it is where the per-element f32 rounding happens.
            Value::Tensor(t) => Ok(Value::tensor(t.data.mapv(|x| gelu(x)))),
            Value::Float(x, w) => Ok(Value::Float(w.round(gelu(*x)), *w)),
            _ => Err(RuntimeError::msg(format!("\\< requires numeric/tensor, got {}", v.type_name()))),
        }
        BitNot => match v {
            // #540: `~` complements the operand's width, not always 64 bits.
            Value::Int(n, w) => Ok(Value::Int(w.wrap(!w.wrap(*n)), *w)),
            _ => Err(RuntimeError::msg(format!("~ (bitwise NOT) requires int, got {}", v.type_name()))),
        }
    }
}

/// The one GeLU (#567): the tanh approximation of `activation_f64`, rounded
/// through f32 exactly as `builtin_activation` rounds it — which is what the
/// JIT computes, since `emit_gelu_f32` evaluates the whole expression in f32.
/// Both spellings of the operator call this, so there is one number to be
/// right about instead of two.
#[inline]
fn gelu(x: f64) -> f64 {
    quantize_f32(activation_f64("gelu", x))
}

/// Combine bytes into an i64, least significant byte first.
///
/// #606: little-endian, ratified — `SPEC.md §3.10`. The one byte order this
/// language fixes (`PORTS.md`), and the order every other byte path in
/// `compiler/src` already used; this was the sole big-endian decode of a
/// multi-byte scalar. Renamed from `be_bytes_to_i64`.
fn le_bytes_to_i64(slice: &[u8]) -> i64 {
    let mut acc: i64 = 0;
    for (i, &b) in slice.iter().enumerate() { acc |= (b as i64) << (8 * i); }
    acc
}

/// The `@cast(Model)` refusal, worded the same wherever it is raised (#598).
fn overlay_no_layout(model: &str, field: &str, reason: &str) -> RuntimeError {
    RuntimeError::msg(format!(
        "@cast({}): model field `{}` has no overlay layout — {} \
         (DIRECTIVES.md §4.1, SPEC.md §3.10)", model, field, reason))
}

/// IEEE 754 binary16 → f32. `mant * 2^-24` is the subnormal value exactly, and
/// the normal case rebiases the exponent (15 → 127) and left-shifts the
/// mantissa into binary32's 23 bits.
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x03ff) as u32;
    if exp == 0 {
        sign * (mant as f32) * f32::powi(2.0, -24)
    } else if exp == 0x1f {
        f32::from_bits(((h as u32 & 0x8000) << 16) | (0xff << 23) | (mant << 13))
    } else {
        f32::from_bits(((h as u32 & 0x8000) << 16) | ((exp + 112) << 23) | (mant << 13))
    }
}

/// Decode one `@cast(Model)` field's bytes into a value of its DECLARED type
/// (#598). `slice` is exactly `st.overlay_byte_width()` bytes long.
///
/// Every admitted kind lands in its own `Value` shape: an integer keeps its
/// declared WIDTH (`IW::wrap` sign-extends a narrow signed field, so an `i8`
/// byte `0xFF` is `-1`), a float is reinterpreted from its IEEE bit pattern at
/// its storage width, and a `bool` follows the interpreter's one bool rule —
/// `apply_cast`'s `Bool` arm, where any nonzero number is `true`. Before this,
/// every field came back `Value::Int(_, I64)` whatever it was declared, which
/// is #598's own signature: a `bool` or `f32` field read as an integer.
fn decode_overlay_scalar(st: &ScalarType, slice: &[u8]) -> Value {
    use ScalarType::*;
    match st {
        Bool => Value::Bool(slice[0] != 0),
        F64 => Value::Float(f64::from_bits(le_bytes_to_i64(slice) as u64), FW::F64),
        F32 => Value::Float(f32::from_bits(le_bytes_to_i64(slice) as u32) as f64, FW::F32),
        // bf16 is binary32's top 16 bits (PORTS.md's element table).
        Bf16 => Value::Float(
            f32::from_bits((le_bytes_to_i64(slice) as u32) << 16) as f64, FW::F32),
        F16 => Value::Float(f16_bits_to_f32(le_bytes_to_i64(slice) as u16) as f64, FW::F32),
        _ => {
            // Every remaining admitted kind is an integer: `scalar_int_width`
            // answers for all of them (`u64` shares `I64` — see its note).
            let w = scalar_int_width(st).unwrap_or(IW::I64);
            Value::Int(w.wrap(le_bytes_to_i64(slice)), w)
        }
    }
}

/// The same decode, as the f64 a tensor element is stored in.
fn overlay_scalar_as_f64(st: &ScalarType, slice: &[u8]) -> f64 {
    match decode_overlay_scalar(st, slice) {
        Value::Bool(b) => if b { 1.0 } else { 0.0 },
        Value::Float(x, _) => x,
        Value::Int(n, _) => n as f64,
        _ => 0.0,
    }
}

/// The tensor dtype a `@cast(Model)` tensor field carries — its DECLARED
/// element type, so a `Tensor[u8,[24]]` field is a u8 tensor rather than the
/// i64 one every overlay used to produce (#598).
///
/// Exhaustive over `ScalarType` (#852 — no trailing `_ =>`), so a future
/// variant is a compile error here too. Its ONLY caller (the tensor-field
/// arm of `@cast(Model)`'s decode loop, above) already checked
/// `est.overlay_byte_width()` (`ast.rs`) is `Some` before calling this —
/// that table answers `None`, and the field is refused with
/// `overlay_no_layout` before this function is ever reached, for exactly
/// seven kinds: `str`/`nil` (dynamic/zero-sized — #817 is an open ruling on
/// whether either has a tensor-ELEMENT representation elsewhere in the
/// interpreter, e.g. `dtype_for_scalar_elem`, but their exclusion from the
/// BYTE overlay specifically is not that open question — it is unconditional,
/// stated in `DIRECTIVES.md §4.1`), `trit`/`int4` (packed formats with no
/// stated byte encoding), and `tf32`/`fp8_e4m3`/`fp8_e5m2` (f32-backed
/// NOMINAL tags per `SPEC.md §3.1` and the #179 convention — computed in f32
/// and retagged, with no distinct byte storage of their own to decode out
/// of; `DIRECTIVES.md §4.1` names this exclusion explicitly: "`@cast(Model)`
/// is defined for a model whose fields are scalars with byte storage (not
/// `str`, `nil`, `trit`, `int4`, or the f32-backed `tf32`/`fp8` kinds)").
///
/// #852 found the old `_ => DType::Int(scalar_int_width(st).unwrap_or(IW::
/// I64))` fallback would have decoded any of those seven as an 8-byte
/// integer had it ever run — but with `overlay_byte_width` gating every
/// call site, it never could: a non-exhaustive table with no live input,
/// not a reachable mis-decode. Kept as a catchable internal error rather
/// than `unreachable!()` because the invariant crosses this file and
/// `ast.rs`'s table, the same shape of guard `arena_ctor_dtype` keeps
/// against `dtype_for_scalar_elem` drifting from `ScalarType::from_ctor_name`
/// (#834) — if the two ever fall out of sync, this names the type instead
/// of silently mis-decoding it.
fn overlay_elem_dtype(st: &ScalarType) -> EvalResult<DType> {
    use ScalarType::*;
    Ok(match st {
        Bool => DType::Bool,
        F64 => DType::F64,
        F32 | F16 | Bf16 => DType::F32,
        I8 | Int8 => DType::Int(IW::I8),
        I16 => DType::Int(IW::I16),
        I32 => DType::Int(IW::I32),
        I64 => DType::Int(IW::I64),
        U8 => DType::Int(IW::U8),
        U16 => DType::Int(IW::U16),
        U32 => DType::Int(IW::U32),
        U64 => DType::Int(IW::U64),
        Str | Nil | Trit | Int4 | Tf32 | Fp8E4M3 | Fp8E5M2 => return Err(RuntimeError::msg(format!(
            "internal error: overlay_elem_dtype called with `{:?}`, which \
             `ScalarType::overlay_byte_width` excludes from `@cast(Model)` \
             admission (DIRECTIVES.md §4.1) — the two tables have fallen out \
             of sync (#852)", st))),
    })
}

/// TEST-ONLY: `overlay_elem_dtype` for `interp_tests.rs` — a sibling module
/// of `interp` (not a descendant), so plain fn privacy would not reach it,
/// the same reason the `#674` harness above (`eval_expr_for_test` etc.) is
/// `pub(crate)` under `#[cfg(test)]` rather than `pub`.
#[cfg(test)]
pub(crate) fn overlay_elem_dtype_for_test(st: &ScalarType) -> EvalResult<DType> {
    overlay_elem_dtype(st)
}

/// Bounds-checked slice of `sz` bytes at `offset` for `@cast(Model)` overlay.
fn read_overlay_bytes<'a>(bytes: &'a [u8], offset: usize, sz: usize, model: &str, field: &str)
    -> EvalResult<&'a [u8]>
{
    if offset + sz > bytes.len() {
        return Err(RuntimeError::msg(format!(
            "@cast({}): field `{}` needs {} byte(s) at offset {}, but only {} byte(s) available",
            model, field, sz, offset, bytes.len())));
    }
    Ok(&bytes[offset..offset + sz])
}

fn apply_cast_by_name(v: &Value, ty_name: &str) -> Value {
    // Tensor casts: integer targets truncate toward zero and tag the result
    // Int (so element reads round-trip as integers, #125); float targets keep
    // values and tag Float. Other targets (bool/str) fall through unchanged.
    if let Value::Tensor(t) = v {
        // #571: the cast names the element type, so the result carries that
        // element WIDTH, and every element is narrowed into it — the same
        // table `apply_cast` uses for the `as` spelling, which until now
        // narrowed while this one only truncated.
        if let Some(w) = int_type_name_width(ty_name) {
            return Value::tensor_dt(cast_tensor_to_int_width(&t.data, w), DType::Int(w));
        }
        if ty_name == "f64" {
            // Widening retag — no value change (f32-rounded data is exact in f64).
            return Value::tensor_dt(t.data.clone(), DType::F64);
        }
        // #834: fp8 targets get their own one-byte-metered `DType` — same
        // arithmetic rounding as the rest of the f32-family below
        // (`DType::rounds_as_f32`), but a distinct declared width.
        if matches!(ty_name, "fp8_e4m3") {
            return Value::tensor_dt(t.data.clone(), DType::Fp8E4M3);
        }
        if matches!(ty_name, "fp8_e5m2") {
            return Value::tensor_dt(t.data.clone(), DType::Fp8E5M2);
        }
        if matches!(ty_name, "f16" | "bf16" | "tf32" | "f32") {
            // Trit→float: values are already stored as -1.0/0.0/1.0 f64, just
            // retag. F64→f32-family narrows via the F32 construction rounding.
            return Value::tensor_dt(t.data.clone(), DType::F32);
        }
        return v.clone();
    }
    match ty_name {
        // String → byte tensor: @cast(u8) { "text" } yields Tensor[u8, [N]] of UTF-8 bytes.
        "u8" => {
            if let Value::Str(s) = v {
                let bytes: Vec<f64> = s.bytes().map(|b| b as f64).collect();
                let arr = ndarray::Array1::from_vec(bytes).into_dyn();
                // The cast names `u8`, so the bytes are u8-wide elements (#571).
                return Value::tensor_dt(arr, DType::Int(IW::U8));
            }
            if let Some(n) = v.as_int()   { return Value::Int((n as u8)  as i64, IW::U8); }
            if let Some(n) = v.as_float() { return Value::Int(n as u8 as i64, IW::U8); }
        }
        // #544: the narrow targets carry their width the way `i32` has since
        // #540, and the float source is narrowed through that width for the
        // same reason — a value tagged `I8` has to be inside the i8 range.
        "i8" | "int8" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I8); }
                   if let Some(n) = v.as_int() { return Value::Int((n as i8)  as i64, IW::I8); }
                   if let Some(n) = v.as_float() { return Value::Int(n as i8 as i64, IW::I8); } }
        "i16" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I16); }
                   if let Some(n) = v.as_int() { return Value::Int((n as i16) as i64, IW::I16); }
                   if let Some(n) = v.as_float() { return Value::Int(n as i16 as i64, IW::I16); } }
        "int4" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I4); }
                   if let Some(n) = v.as_int() { return Value::Int(IW::I4.wrap(n), IW::I4); }
                   if let Some(n) = v.as_float() { return Value::Int(IW::I4.wrap(n as i64), IW::I4); } }
        // #540: an i32 target carries the width, exactly as in `apply_cast`.
        "i32" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I32); }
                   if let Some(n) = v.as_int() { return Value::Int((n as i32) as i64, IW::I32); }
                   // Narrow the float source too: a value tagged `I32` must be
                   // inside the i32 range, and this is what `x as i32` on the
                   // same float already produced (`apply_cast`).
                   if let Some(n) = v.as_float() { return Value::Int(n as i32 as i64, IW::I32); } }
        "i64" => {
            if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I64); }
            // #597: an INTEGER source reinterprets its 64-bit pattern
            // (`u64::MAX as i64` is -1), so read the pattern directly instead
            // of routing through `as_float`, which now reports a `u64`'s
            // unsigned magnitude. This also stops a large `i64` from
            // round-tripping through f64 and losing bits above 2^53.
            if let Some(n) = v.as_int() { return Value::Int(n, IW::I64); }
            if let Some(n) = v.as_float() { return Value::Int(n as i64, IW::I64); }
        }
        "u16" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::U16); }
                   if let Some(n) = v.as_int() { return Value::Int((n as u16) as i64, IW::U16); }
                   if let Some(n) = v.as_float() { return Value::Int(n as u16 as i64, IW::U16); } }
        "u32" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::U32); }
                   if let Some(n) = v.as_int() { return Value::Int((n as u32) as i64, IW::U32); }
                   if let Some(n) = v.as_float() { return Value::Int(n as u32 as i64, IW::U32); } }
        // #597: tagged `IW::U64`, not `IW::I64` — see `apply_cast`'s `U64` arm.
        "u64" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::U64); }
                   if let Some(n) = v.as_int() { return Value::Int((n as u64) as i64, IW::U64); }
                   if let Some(n) = v.as_float() { return Value::Int(n as u64 as i64, IW::U64); } }
        "f64" => {
            if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }, FW::F64); }
            if let Some(n) = v.as_float() { return Value::Float(n, FW::F64); }
        }
        // f32-family scalar casts round through f32 (#300 ruling) — see apply_cast.
        // #473: the result is also TAGGED f32, so the width propagates into
        // whatever the cast feeds. Rounding once and then computing in f64
        // would still diverge from the JIT on the very next operation.
        "f16" | "bf16" | "tf32" | "f32" | "fp8_e4m3" | "fp8_e5m2" => {
            if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }, FW::F32); }
            if let Some(n) = v.as_float() { return Value::Float(quantize_f32(n), FW::F32); }
        }
        "bool" => {
            if let Value::Bool(b) = v { return Value::Bool(*b); }
            if let Some(n) = v.as_int()   { return Value::Bool(n != 0); }
            if let Some(x) = v.as_float() { return Value::Bool(x != 0.0); }
            return Value::Bool(false);
        }
        "str"  => { return Value::Str(format!("{:?}", v)); }
        _ => {}
    }
    v.clone()
}

fn apply_cast(v: &Value, ty: &Type) -> Value {
    if let Type::Scalar(s, _) = ty {
        use ScalarType::*;
        // Tensor casts mirror apply_cast_by_name: integer targets truncate and
        // tag Int (#125); f64 retags wide; the f32-family retags F32, which
        // narrows through the construction rounding (#241). Others clone.
        if let Value::Tensor(t) = v {
            return match s {
                // Narrow each element through the concrete target width, matching
                // the scalar cast path (#298 / #291.1), and TAG that width so the
                // element keeps it (#571) — `int4`/`int8` narrow here too, where
                // before they only truncated and left an element that did not fit.
                I8 | I16 | I32 | U8 | U16 | U32 | I64 | U64 | Int4 | Int8 => {
                    let w = scalar_int_width(s).unwrap_or(IW::I64);
                    Value::tensor_dt(cast_tensor_to_int_width(&t.data, w), DType::Int(w))
                }
                F64 => Value::tensor_dt(t.data.clone(), DType::F64),
                // #834: fp8 targets keep their own one-byte-metered `DType`,
                // mirroring `apply_cast_by_name` — same f32 rounding as the
                // rest of the f32-family arm below, distinct declared width.
                Fp8E4M3 => Value::tensor_dt(t.data.clone(), DType::Fp8E4M3),
                Fp8E5M2 => Value::tensor_dt(t.data.clone(), DType::Fp8E5M2),
                F16 | Bf16 | Tf32 | F32 =>
                    Value::tensor_dt(t.data.clone(), DType::F32),
                _ => v.clone(),
            };
        }
        match s {
            // Narrowing integer casts: use Rust's `as` for proper 2's-complement wrap/truncation.
            // bool→int: true=1, false=0 (C/Rust/numpy convention).
            // Float→signed-narrow wraps through the target width to match the
            // int→signed-narrow path above and the unsigned-float path below
            // (#291.1): `300.0 as i8` == `300 as i8` == 44, not 300.
            // #544: every narrow target carries its width, exactly as `i32` has
            // since #540. The narrowing of the VALUE is unchanged — `300 as i8`
            // was 44 before and still is; what is new is that the result stays
            // an `i8`, so the arithmetic that follows wraps at 8 bits.
            I8 | Int8 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I8); }
                     if let Some(n) = v.as_int() { return Value::Int((n as i8)  as i64, IW::I8); }
                     if let Some(n) = v.as_float() { return Value::Int(n as i8 as i64, IW::I8); } }
            I16 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I16); }
                     if let Some(n) = v.as_int() { return Value::Int((n as i16) as i64, IW::I16); }
                     if let Some(n) = v.as_float() { return Value::Int(n as i16 as i64, IW::I16); } }
            Int4 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I4); }
                     if let Some(n) = v.as_int() { return Value::Int(IW::I4.wrap(n), IW::I4); }
                     if let Some(n) = v.as_float() { return Value::Int(IW::I4.wrap(n as i64), IW::I4); } }
            // #540: `as i32` already narrowed the VALUE; it now also carries the
            // width, so the arithmetic that follows wraps like the JIT's
            // `cl::I32`. The narrowing itself is unchanged — `5000000000 as i32`
            // was 705032704 on both backends before this and still is.
            I32 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I32); }
                     if let Some(n) = v.as_int() { return Value::Int((n as i32) as i64, IW::I32); }
                     if let Some(n) = v.as_float() { return Value::Int(n as i32 as i64, IW::I32); } }
            I64 | Trit => {
                if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::I64); }
                // #597: see `apply_cast_by_name`'s `"i64"` arm — an integer
                // source reinterprets its pattern, it does not go through f64.
                if let Some(n) = v.as_int() { return Value::Int(n, IW::I64); }
                if let Some(n) = v.as_float() { return Value::Int(n as i64, IW::I64); }
            }
            U8  => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::U8); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u8)  as i64, IW::U8); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u8 as i64, IW::U8); } }
            U16 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::U16); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u16) as i64, IW::U16); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u16 as i64, IW::U16); } }
            U32 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::U32); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u32) as i64, IW::U32); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u32 as i64, IW::U32); } }
            // #597: tagged `IW::U64` so the value's `/ % >>` and ordering
            // comparisons read unsigned from here on (see `IW`'s doc comment).
            U64 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }, IW::U64); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u64) as i64, IW::U64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u64 as i64, IW::U64); } }
            F64 => {
                if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }, FW::F64); }
                if let Some(n) = v.as_float() { return Value::Float(n, FW::F64); }
            }
            // f32-family scalar casts ROUND through f32 (#300 ruling): an explicit
            // `as f32` loses precision, matching the JIT and the tensor cast path
            // which retags these as f32. `0.1 as f32` → 0.10000000149… not 0.1.
            F16 | Bf16 | Tf32 | F32 | Fp8E4M3 | Fp8E5M2 => {
                // #473: round AND tag — see apply_cast_by_name.
                if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }, FW::F32); }
                if let Some(n) = v.as_float() { return Value::Float(quantize_f32(n), FW::F32); }
            }
            // int/float→bool: 0/0.0 → false, anything else → true.
            Bool => {
                if let Value::Bool(b) = v { return Value::Bool(*b); }
                if let Some(n) = v.as_int()   { return Value::Bool(n != 0); }
                if let Some(x) = v.as_float() { return Value::Bool(x != 0.0); }
                return Value::Bool(false);
            }
            Str  => { return Value::Str(format!("{}", v)); }
            Nil  => { return Value::Nil; }
        }
    }
    v.clone()
}

// ─── Host feature detection (@host match) ────────────────────────────────────

fn detect_host_features() -> std::collections::HashSet<String> {
    let mut f: std::collections::HashSet<String> = std::collections::HashSet::new();

    // AArch64: NEON (Advanced SIMD) is mandatory in the AArch64 baseline ISA.
    #[cfg(target_arch = "aarch64")]
    {
        f.insert("neon".into());
        f.insert("fp16".into());
    }

    // x86_64: runtime CPUID detection via std::arch macros.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2")    { f.insert("sse2".into()); }
        if is_x86_feature_detected!("sse4.1")  { f.insert("sse4_1".into()); f.insert("sse4.1".into()); }
        if is_x86_feature_detected!("avx")     { f.insert("avx".into()); }
        if is_x86_feature_detected!("avx2")    { f.insert("avx2".into()); }
        if is_x86_feature_detected!("avx512f") { f.insert("avx512f".into()); f.insert("avx512".into()); }
        if is_x86_feature_detected!("avx512bw") { f.insert("avx512bw".into()); }
        if is_x86_feature_detected!("avx512cd") { f.insert("avx512cd".into()); }
    }

    // #578: `accelerate` — a BLAS `cblas_sgemm` is in this process's image, so
    // a large f32 matmul can reach the platform's matrix unit (the AMX
    // coprocessor, on Apple silicon). Unlike the ISA flags above this is not a
    // property of the CPU but of what is linked, which is exactly what a
    // program dispatching on it needs to know: `@host match { .accelerate =>
    // … }` asks the same question the JIT's own kernel selection asks, from
    // the same detection, so the two cannot disagree.
    if crate::jit::blas_gemm_available() {
        f.insert("accelerate".into());
    }

    f
}

// ─── Iteration support ───────────────────────────────────────────────────────

fn expand_iter(v: &Value) -> EvalResult<Vec<Value>> {
    match v {
        Value::Range { start, end, inclusive } => {
            let end = if *inclusive { *end + 1 } else { *end };
            Ok((*start..end).map(|n| Value::Int(n, IW::I64)).collect())
        }
        Value::Tuple(vs) => Ok(vs.clone()),
        Value::List(vs) => Ok((**vs).clone()),
        Value::Tensor(t) => {
            // Iterate along the first axis as nested tensors / scalars,
            // preserving the dtype so iterating an integer tensor yields Ints
            // (and a bool tensor yields Bools).
            let dtype = t.dtype;
            let scalar = move |x: f64| match dtype {
                // #571: iterating a narrow integer tensor yields elements at
                // the element width, exactly as indexing one does.
                DType::Int(w) => Value::Int(w.wrap(x as i64), w),
                DType::Bool => Value::Bool(x != 0.0),
                _ => Value::Float(x, FW::F64),
            };
            let mut out = Vec::new();
            if t.ndim() == 0 { return Ok(vec![scalar(t[IxDyn(&[])])]); }
            for slice in t.axis_iter(Axis(0)) {
                if slice.ndim() == 0 {
                    out.push(scalar(slice[IxDyn(&[])]));
                } else {
                    out.push(Value::tensor_dt(slice.to_owned().into_dyn(), t.dtype));
                }
            }
            Ok(out)
        }
        _ => Err(RuntimeError::msg(format!("cannot iterate over {}", v.type_name()))),
    }
}

// ─── String method dispatch ──────────────────────────────────────────────────

fn call_str_method(s: &str, method: &str, args: Vec<Value>, sp: Span) -> EvalResult<Value> {
    match method {
        "split" => {
            let delim = match args.first() {
                Some(Value::Str(d)) => d.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("str.split: delimiter must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("str.split: needs delimiter arg", sp)),
            };
            let parts: Vec<Value> = s.split(delim.as_str())
                .map(|p| Value::Str(p.to_string()))
                .collect();
            Ok(Value::list(parts))
        }
        "trim" | "strip" => Ok(Value::Str(s.trim().to_string())),
        "upper" => Ok(Value::Str(s.to_uppercase())),
        "lower" => Ok(Value::Str(s.to_lowercase())),
        "starts_with" => {
            let prefix = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("str.starts_with: needs str arg", sp)),
            };
            Ok(Value::Bool(s.starts_with(prefix.as_str())))
        }
        "ends_with" => {
            let suffix = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("str.ends_with: needs str arg", sp)),
            };
            Ok(Value::Bool(s.ends_with(suffix.as_str())))
        }
        "contains" => {
            let needle = match args.first() {
                Some(Value::Str(n)) => n.clone(),
                _ => return Err(RuntimeError::at("str.contains: needs str arg", sp)),
            };
            Ok(Value::Bool(s.contains(needle.as_str())))
        }
        "replace" => {
            let old = match args.first() {
                Some(Value::Str(o)) => o.clone(),
                _ => return Err(RuntimeError::at("str.replace: needs old str arg", sp)),
            };
            let new = match args.get(1) {
                Some(Value::Str(n)) => n.clone(),
                _ => return Err(RuntimeError::at("str.replace: needs new str arg", sp)),
            };
            Ok(Value::Str(s.replace(old.as_str(), new.as_str())))
        }
        "find" => {
            let needle = match args.first() {
                Some(Value::Str(n)) => n.clone(),
                _ => return Err(RuntimeError::at("str.find: needs str arg", sp)),
            };
            let idx = s.find(needle.as_str()).map(|i| i as i64).unwrap_or(-1);
            Ok(Value::Int(idx, IW::I64))
        }
        "index" => {
            let needle = match args.first() {
                Some(Value::Str(n)) => n.clone(),
                _ => return Err(RuntimeError::at("str.index: needs str arg", sp)),
            };
            s.find(needle.as_str()).map(|i| Value::Int(i as i64, IW::I64))
                .ok_or_else(|| RuntimeError::at(
                    format!("str.index: {:?} not found in {:?}", needle, s), sp))
        }
        "count" => {
            let needle = match args.first() {
                Some(Value::Str(n)) => n.clone(),
                _ => return Err(RuntimeError::at("str.count: needs str arg", sp)),
            };
            if needle.is_empty() {
                return Ok(Value::Int(s.chars().count() as i64 + 1, IW::I64));
            }
            let count = s.matches(needle.as_str()).count();
            Ok(Value::Int(count as i64, IW::I64))
        }
        "lines" | "split_lines" => {
            let parts: Vec<Value> = s.lines()
                .map(|l| Value::Str(l.to_string()))
                .collect();
            Ok(Value::list(parts))
        }
        "len" => Ok(Value::Int(s.len() as i64, IW::I64)),
        _ => {
            // Unknown method — fall through to opaque (don't error, for forward compat)
            Ok(Value::Opaque(format!("str.{}", method)))
        }
    }
}

// ─── Builtins ────────────────────────────────────────────────────────────────

/// #402: recover the registry id from a `Port` handle value. The Value side
/// carries only the opaque `port#<id>:<lang>` text minted by `crate::ports`.
fn port_handle_id(v: Option<&Value>) -> Option<i64> {
    if let Some(Value::Opaque(s)) = v {
        crate::ports::handle_id(s)
    } else {
        None
    }
}

/// The `PORTS.md §3.2` wire dtype an interpreter tensor crosses as.
///
/// The interpreter has one integer tensor dtype and the JIT has two, so the
/// mapping deliberately lands every integer tensor on `i64` — see
/// `ports::wire_dtype_for`. `Trit` has no wire dtype at all: a packed ternary
/// weight is a demoniC storage format, not a portable element type, so it is
/// refused rather than silently widened to bytes no foreign runtime can read.
///
/// `Fp8E4M3`/`Fp8E5M2` (#834) cross the wire as `F32`, exactly as they did
/// before this fix (when the interpreter mistagged them as plain `DType::F32`
/// outright): both are nominal f32-backed tags with no distinct byte storage
/// of their own (`ScalarType::overlay_byte_width`'s reasoning, same as
/// `tf32`) — unlike `Trit`'s packed-ternary format, an f32-backed value is
/// unambiguously portable as f32 bytes, so there is no reason to newly refuse it.
fn wire_dtype_of(dt: DType) -> Option<crate::ports::WireDType> {
    use crate::ports::WireDType as W;
    Some(match dt {
        // The wire protocol carries integers as 64-bit lanes at every element
        // width; the narrow value already fits (#571).
        DType::Int(_) => W::I64,
        DType::F32 | DType::Fp8E4M3 | DType::Fp8E5M2 => W::F32,
        DType::F64 => W::F64,
        DType::Bool => W::Bool,
        DType::Trit => return None,
    })
}

pub(crate) fn is_builtin(name: &str) -> bool {
    matches!(name,
        "print" | "print_err" | "panic" |
        "assert" | "assert_eq" | "assert_ne" |
        "sum" | "mean" | "max" | "min" | "argmax" | "argmin" | "trace" | "diag" |
        "f32_to_bits" | "f32_from_bits" |
        // Variance-trait + per-axis reductions (parity with the JIT).
        "variance" | "pull_to_mean" | "sum_along" | "mean_along" | "max_along" | "min_along" |
        "variance_along" | "pull_to_mean_along" |
        // JIT typed-print shims (parity with the JIT print family).
        "print_tensor" | "print_i64" | "print_f64" | "print_bool" | "print_nil" |
        "softmax" | "sqrt" | "exp" | "log" | "abs" | "sin" | "cos" |
        // Elementwise activations (parity with the JIT; scalar or tensor).
        "relu" | "sigmoid" | "tanh" | "gelu" | "silu" | "elu" | "mish" |
        "floor" | "ceil" | "round" | "trunc" | "sign" |
        "chr" | "ord" | "len" |
        "to_str" | "to_string" | "to_int" | "to_float" |
        "to_hex" | "to_bin" | "to_binary" | "to_oct" |
        "str_repeat" | "clamp" |
        "tan" | "asin" | "acos" | "atan" | "atan2" | "hypot" |
        "sort" | "gcd" | "median" |   // stdlib under its Rust/Python spelling
        "log2" | "log10" | "isclose" |
        "solve" | "inv" | "lstsq" |
        "rms_norm" | "layer_norm" | "attn" | "attn_gqa" | "rope" | "embed" |
        "allreduce" | "load_batch" | "data_iter" |
        "read_file" | "read_bytes" | "write_file" | "append_file" | "file_exists" |
        "join" |
        // Dynamic collections
        "list" | "list_push" | "list_pop" | "list_get" | "list_set" |
        "list_len" | "list_concat" | "list_slice" | "list_contains" | "list_rev" |
        "map" | "map_set" | "map_get" | "map_has" | "map_del" |
        "map_keys" | "map_vals" | "map_len" |
        // JIT HashMap builtins (parity with jit::JIT_BUILTINS).
        "map_new" | "map_contains" |
        // Process / environment
        "env_var" | "argv" | "exit" |
        // Time
        "time_ms" | "sleep_ms" | "flush" |
        "set_raw_mode" | "read_char_nb" | "read_char" |
        // Extended RNG
        "rand_float" | "rand_int" | "rand_normal" | "rand_seed" | "rand_choice" |
        // JSON
        "json_encode" | "json_decode" |
        // Typed decode — one primitive per demoniC type (PORTS.md §6)
        "json_decode_i64" | "json_decode_f64" | "json_decode_str" |
        "json_decode_bool" | "json_decode_list" |
        // List functional combinators
        "list_map" | "list_filter" | "list_reduce" | "list_sort" | "list_sort_by" |
        "list_zip" | "list_enumerate" | "list_flatten" | "list_uniq" |
        "list_sum" | "list_min" | "list_max" |
        "list_head" | "list_last" | "list_take" | "list_drop" |
        "list_find" | "list_count" | "list_any" | "list_all" |
        "list_flat_map" | "list_partition" |
        "str_pad_left" | "str_pad_right" |
        "map_merge" |
        // String formatting
        "format" |
        // Hashing
        "hash_fnv" | "hash_crc32" |
        // Filesystem operations
        "get_cwd" | "list_dir" | "make_dir" | "delete_file" | "delete_dir" |
        "rename_file" | "file_size" | "path_join" | "path_dirname" | "path_basename" |
        "path_exists" | "path_is_dir" | "path_is_file" |
        // Process execution
        "exec_cmd" |
        // Ports (#402, PORTS.md §2) + tensor copy mode (PORTS.md §3.2)
        "port_open" | "port_call" | "port_close" |
        "port_tensor_encode" | "port_tensor_decode" |
        // CLI argument parsing
        "cli_arg" | "cli_flag" | "cli_positional" | "cli_positional_count" |
        // Regex
        "regex_match" | "regex_find" | "regex_find_all" |
        "regex_replace" | "regex_replace_all" | "regex_split" |
        // Compression
        "gzip_compress" | "gzip_decompress" |
        "zlib_compress" | "zlib_decompress" |
        // HTTP networking
        "http_get" | "http_post" | "http_post_json" |
        // Date/time
        "date_now_ms" | "date_now_s" | "date_format" | "date_parse" |
        "date_add_ms" | "date_diff_ms" |
        // Trit tensor operations.
        "trit_quantize" | "trit_quantize_soft" |
        "trit_neg" | "trit_sparsity" | "trit_pack" |
        // Runtime type introspection (#184) + safe numeric parse (#185).
        "typeof" |
        "is_int" | "is_float" | "is_str" | "is_bool" |
        "is_list" | "is_map" | "is_nil" | "is_fn" | "is_tensor" |
        "is_numeric" | "try_to_int" | "try_to_float"
    )
}

impl Interpreter {
    fn call_builtin(&mut self, name: &str, mut args: Vec<Value>, sp: Span) -> EvalResult<Value> {
    // #781 (reviewer rework): the ONE choke point for every builtin
    // dispatched through this match — softmax, sum_along, argmax,
    // rms_norm, embed, diag, solve, and the rest of the free-function
    // builtin set (arena constructors like `vault.zeros[..]` are a
    // separate dispatch and already charge at their own call sites; see
    // `charge_arena`'s other caller). Held here, cloned, rather than
    // reused from `sp` after the match: many arms below consume `sp` by
    // value on their own success/error paths, and a partial move inside
    // one arm would make the ORIGINAL `sp` unusable after the match ends
    // even on paths that never touched it.
    let charge_span = sp.clone();
    let result = match name {
        "print" => {
            let parts: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
            // #211: print is line-oriented — it appends a trailing newline (like
            // Python's print and the JIT's scalar print helpers). Both backends
            // now agree. Build a line piecewise with string concatenation, not
            // multiple print() calls.
            println!("{}", parts.join(" "));
            Ok(Value::Nil)
        }
        "panic" => {
            let msg = args.first().map(|v| format!("{}", v)).unwrap_or_default();
            Err(RuntimeError::at(format!("panic: {}", msg), sp))
        }
        "print_err" => {
            let parts: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
            eprint!("{}", parts.join(" "));
            Ok(Value::Nil)
        }
        "assert" => {
            let cond = args.first().and_then(|v| if let Value::Bool(b) = v { Some(*b) } else { None })
                .ok_or_else(|| RuntimeError::at("assert: bool arg required", sp.clone()))?;
            if !cond {
                let msg = args.get(1).map(|v| format!("{}", v))
                    .unwrap_or_else(|| "assertion failed".to_string());
                return Err(RuntimeError::at(msg, sp));
            }
            Ok(Value::Nil)
        }
        "assert_eq" => {
            if args.len() < 2 {
                return Err(RuntimeError::at("assert_eq: requires 2 args", sp));
            }
            if !values_equal(&args[0], &args[1]) {
                let msg = if args.len() > 2 {
                    format!("{}", args[2])
                } else {
                    format!("assertion failed: {} != {}", args[0], args[1])
                };
                return Err(RuntimeError::at(msg, sp));
            }
            Ok(Value::Nil)
        }
        "assert_ne" => {
            if args.len() < 2 {
                return Err(RuntimeError::at("assert_ne: requires 2 args", sp));
            }
            if values_equal(&args[0], &args[1]) {
                let msg = if args.len() > 2 {
                    format!("{}", args[2])
                } else {
                    format!("assertion failed: {} == {}", args[0], args[1])
                };
                return Err(RuntimeError::at(msg, sp));
            }
            Ok(Value::Nil)
        }
        // ── Linear algebra ────────────────────────────────────────────────
        // solve/inv/lstsq are declared in the typechecker's builtin set but had no
        // runtime impl (they errored as "unknown builtin"). Implemented here on
        // plain ArrayD via Gauss–Jordan elimination — see `gauss_jordan_solve`.
        "solve" => {
            // solve(A, b): exact solution of the square system A x = b.
            let a = as_tensor(args.first().ok_or_else(|| RuntimeError::at("solve: needs matrix A", sp.clone()))?)?;
            let b = as_tensor(args.get(1).ok_or_else(|| RuntimeError::at("solve: needs vector b", sp.clone()))?)?;
            if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
                return Err(RuntimeError::at("solve: A must be a square 2-D matrix", sp));
            }
            let n = a.shape()[0];
            let bv: Vec<f64> = b.iter().copied().collect();
            if bv.len() != n {
                return Err(RuntimeError::at(format!("solve: b has length {} but A is {n}×{n}", bv.len()), sp));
            }
            let x = gauss_jordan_solve(&a, &bv).map_err(|e| RuntimeError::at(format!("solve: {e}"), sp.clone()))?;
            Ok(Value::tensor(ArrayD::from_shape_vec(IxDyn(&[n]), x).unwrap()))
        }
        "inv" => {
            // inv(A): inverse of a square matrix.
            let a = as_tensor(args.first().ok_or_else(|| RuntimeError::at("inv: needs matrix A", sp.clone()))?)?;
            if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
                return Err(RuntimeError::at("inv: A must be a square 2-D matrix", sp));
            }
            let n = a.shape()[0];
            let m = matrix_inverse(&a, n).map_err(|e| RuntimeError::at(format!("inv: {e}"), sp.clone()))?;
            Ok(Value::tensor(m))
        }
        "lstsq" => {
            // lstsq(A, b): least-squares solution of an overdetermined A x ≈ b via
            // the normal equations AᵀA x = Aᵀb. Adequate (and dependency-free) for
            // the well-conditioned small fits demoniC uses; a QR/SVD solver would be
            // more numerically stable for ill-conditioned A, but isn't worth a LAPACK
            // dependency at this stage.
            let a = as_tensor(args.first().ok_or_else(|| RuntimeError::at("lstsq: needs matrix A", sp.clone()))?)?;
            let b = as_tensor(args.get(1).ok_or_else(|| RuntimeError::at("lstsq: needs vector b", sp.clone()))?)?;
            if a.ndim() != 2 {
                return Err(RuntimeError::at("lstsq: A must be a 2-D matrix", sp));
            }
            let (rows, cols) = (a.shape()[0], a.shape()[1]);
            let bv: Vec<f64> = b.iter().copied().collect();
            if bv.len() != rows {
                return Err(RuntimeError::at(format!("lstsq: b has length {} but A has {rows} rows", bv.len()), sp));
            }
            // AᵀA (cols×cols) and Aᵀb (length cols).
            let mut ata = ArrayD::zeros(IxDyn(&[cols, cols]));
            for i in 0..cols {
                for j in 0..cols {
                    let mut s = 0.0;
                    for r in 0..rows { s += a[IxDyn(&[r, i])] * a[IxDyn(&[r, j])]; }
                    ata[IxDyn(&[i, j])] = s;
                }
            }
            let mut atb = vec![0.0; cols];
            for (i, slot) in atb.iter_mut().enumerate() {
                let mut s = 0.0;
                for r in 0..rows { s += a[IxDyn(&[r, i])] * bv[r]; }
                *slot = s;
            }
            let x = gauss_jordan_solve(&ata, &atb).map_err(|e| RuntimeError::at(format!("lstsq: {e}"), sp.clone()))?;
            Ok(Value::tensor(ArrayD::from_shape_vec(IxDyn(&[cols]), x).unwrap()))
        }
        "sqrt" => {
            let x = args.first().and_then(|v| v.as_float()).ok_or_else(||
                RuntimeError::msg("sqrt: numeric arg required"))?;
            Ok(Value::Float(x.sqrt(), FW::F64))
        }
        "exp" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("exp: numeric"))?
            .exp(), FW::F64)),
        "log" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("log: numeric"))?
            .ln(), FW::F64)),
        "abs" => match args.first() {
            // #597: `u64` has no negative values — the magnitude IS the
            // pattern, so `abs` is the identity at this width. Taking
            // `i64::abs` of the pattern turned `abs(u64::MAX)` (pattern -1)
            // into 1. The result keeps its `U64` tag so what follows still
            // divides, shifts and compares unsigned.
            Some(Value::Int(n, IW::U64)) => Ok(Value::Int(*n, IW::U64)),
            Some(Value::Int(n, _)) => Ok(Value::Int(n.abs(), IW::I64)),
            Some(v) => Ok(Value::Float(v.as_float().unwrap_or(0.0).abs(), FW::F64)),
            None => Err(RuntimeError::msg("abs: needs arg")),
        }
        "sin" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .unwrap_or(0.0).sin(), FW::F64)),
        "cos" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .unwrap_or(0.0).cos(), FW::F64)),
        "floor" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("floor: numeric arg required"))?
            .floor(), FW::F64)),
        "ceil" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("ceil: numeric arg required"))?
            .ceil(), FW::F64)),
        "tan" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("tan: numeric arg required"))?
            .tan(), FW::F64)),
        "asin" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("asin: numeric arg required"))?
            .asin(), FW::F64)),
        "acos" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("acos: numeric arg required"))?
            .acos(), FW::F64)),
        "atan" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("atan: numeric arg required"))?
            .atan(), FW::F64)),
        "atan2" => {
            let y = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("atan2: numeric y arg required"))?;
            let x = args.get(1).and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("atan2: numeric x arg required"))?;
            Ok(Value::Float(y.atan2(x), FW::F64))
        }
        "hypot" => {
            let x = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("hypot: numeric x arg required"))?;
            let y = args.get(1).and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("hypot: numeric y arg required"))?;
            Ok(Value::Float(x.hypot(y), FW::F64))
        }
        // #335: greatest common divisor (Euclid), |a|,|b|; gcd(0,0)=0.
        "gcd" => {
            let mut a = args.first().and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("gcd: integer args required"))?.abs();
            let mut b = args.get(1).and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("gcd: integer args required"))?.abs();
            while b != 0 { let t = b; b = a % t; a = t; }
            Ok(Value::Int(a, IW::I64))
        }
        "log2" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("log2: numeric arg required"))?
            .log2(), FW::F64)),
        "log10" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("log10: numeric arg required"))?
            .log10(), FW::F64)),
        "isclose" => {
            let a = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("isclose: numeric a arg required"))?;
            let b = args.get(1).and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("isclose: numeric b arg required"))?;
            Ok(Value::Bool((a - b).abs() < 1e-9))
        }
        "chr" => {
            let n = args.first().and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("chr: integer arg required"))?;
            let c = char::from_u32(n as u32)
                .ok_or_else(|| RuntimeError::msg(format!("chr: {} is not a valid Unicode codepoint", n)))?;
            Ok(Value::Str(c.to_string()))
        }
        "round" => {
            let x = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("round: numeric arg required"))?;
            let ndigits = args.get(1).and_then(|v| v.as_int()).unwrap_or(0);
            if ndigits == 0 {
                Ok(Value::Float(x.round(), FW::F64))
            } else {
                let factor = 10f64.powi(ndigits as i32);
                Ok(Value::Float((x * factor).round() / factor, FW::F64))
            }
        }
        "trunc" => {
            let x = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("trunc: numeric arg required"))?;
            Ok(Value::Float(x.trunc(), FW::F64))
        }
        "sign" => {
            match args.first() {
                Some(Value::Int(n, _))   => Ok(Value::Int(n.signum(), IW::I64)),
                Some(Value::Float(f, w)) => Ok(Value::Float(f.signum(), *w)),
                _ => Err(RuntimeError::msg("sign: numeric arg required")),
            }
        }
        "to_hex" => {
            let n = args.first().and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("to_hex: integer arg required"))?;
            Ok(Value::Str(format!("{:x}", n)))
        }
        "to_bin" | "to_binary" => {
            let n = args.first().and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("to_bin: integer arg required"))?;
            Ok(Value::Str(format!("{:b}", n)))
        }
        "to_oct" => {
            let n = args.first().and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("to_oct: integer arg required"))?;
            Ok(Value::Str(format!("{:o}", n)))
        }
        "ord" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(Value::Int(n, _)) => return Ok(Value::Int(*n, IW::I64)), // already a codepoint
                _ => return Err(RuntimeError::at("ord: str arg required", sp)),
            };
            let ch = s.chars().next()
                .ok_or_else(|| RuntimeError::at("ord: empty string", sp.clone()))?;
            Ok(Value::Int(ch as i64, IW::I64))
        }
        "to_str" | "to_string" => {
            let v = args.into_iter().next().unwrap_or(Value::Nil);
            Ok(Value::Str(format!("{}", v)))
        }
        // ── Runtime type introspection (#184) ──────────────────────────────
        "typeof" => match args.first() {
            Some(v) => Ok(Value::Str(v.type_name().to_string())),
            None => Err(RuntimeError::at("typeof: requires 1 argument", sp)),
        },
        "is_int" => Ok(Value::Bool(matches!(args.first(), Some(Value::Int(_, _))))),
        "is_float" => Ok(Value::Bool(matches!(args.first(), Some(Value::Float(_, _))))),
        "is_str" => Ok(Value::Bool(matches!(args.first(), Some(Value::Str(_))))),
        "is_bool" => Ok(Value::Bool(matches!(args.first(), Some(Value::Bool(_))))),
        "is_list" => Ok(Value::Bool(matches!(args.first(), Some(Value::List(_))))),
        "is_map" => Ok(Value::Bool(matches!(args.first(), Some(Value::Map(_))))),
        "is_nil" => Ok(Value::Bool(matches!(args.first(), Some(Value::Nil) | None))),
        "is_fn" => Ok(Value::Bool(matches!(args.first(),
            Some(Value::Fn(_)) | Some(Value::BoundFn { .. })
            | Some(Value::Lambda { .. }) | Some(Value::Builtin(_))))),
        "is_tensor" => Ok(Value::Bool(matches!(args.first(), Some(Value::Tensor(_))))),

        // ── Safe string→number parse (#185) ────────────────────────────────
        // `is_numeric(s)` — true iff s parses as an i64 or f64 (after trim).
        "is_numeric" => match args.first() {
            Some(Value::Str(s)) => {
                let t = s.trim();
                Ok(Value::Bool(t.parse::<i64>().is_ok() || t.parse::<f64>().is_ok()))
            }
            // Numbers are trivially numeric; everything else is not.
            Some(Value::Int(_, _)) | Some(Value::Float(_, _)) => Ok(Value::Bool(true)),
            _ => Ok(Value::Bool(false)),
        },
        // `try_to_int(s)` / `try_to_float(s)` — return (value, Err) per Spec §3.9:
        // Err is nil on success, a str message on failure. Never aborts.
        "try_to_int" => {
            let (v, err) = match args.first() {
                Some(Value::Int(n, _))   => (Value::Int(*n, IW::I64), Value::Nil),
                Some(Value::Float(f, _)) => (Value::Int(*f as i64, IW::I64), Value::Nil),
                Some(Value::Bool(b))  => (Value::Int(*b as i64, IW::I64), Value::Nil),
                Some(Value::Str(s)) => {
                    let t = s.trim();
                    if let Ok(n) = t.parse::<i64>() {
                        (Value::Int(n, IW::I64), Value::Nil)
                    } else if let Ok(f) = t.parse::<f64>() {
                        (Value::Int(f as i64, IW::I64), Value::Nil)
                    } else {
                        (Value::Int(0, IW::I64), Value::Str(format!("try_to_int: not a number: {:?}", s)))
                    }
                }
                other => (Value::Int(0, IW::I64), Value::Str(format!(
                    "try_to_int: cannot convert {}",
                    other.map(|v| v.type_name()).unwrap_or("nil")))),
            };
            Ok(Value::Tuple(vec![v, err]))
        }
        "try_to_float" => {
            let (v, err) = match args.first() {
                Some(Value::Float(f, w)) => (Value::Float(*f, *w), Value::Nil),
                Some(Value::Int(n, _))   => (Value::Float(*n as f64, FW::F64), Value::Nil),
                Some(Value::Bool(b))  => (Value::Float(*b as i64 as f64, FW::F64), Value::Nil),
                Some(Value::Str(s)) => match s.trim().parse::<f64>() {
                    Ok(f) => (Value::Float(f, FW::F64), Value::Nil),
                    Err(_) => (Value::Float(0.0, FW::F64), Value::Str(format!("try_to_float: not a number: {:?}", s))),
                },
                other => (Value::Float(0.0, FW::F64), Value::Str(format!(
                    "try_to_float: cannot convert {}",
                    other.map(|v| v.type_name()).unwrap_or("nil")))),
            };
            Ok(Value::Tuple(vec![v, err]))
        }

        "to_int" => {
            match args.first() {
                Some(Value::Int(n, _))   => Ok(Value::Int(*n, IW::I64)),
                Some(Value::Float(f, _)) => Ok(Value::Int(*f as i64, IW::I64)),
                Some(Value::Bool(b))  => Ok(Value::Int(*b as i64, IW::I64)),
                Some(Value::Str(s))   => {
                    let trimmed = s.trim();
                    if let Ok(n) = trimmed.parse::<i64>() {
                        Ok(Value::Int(n, IW::I64))
                    } else if let Ok(f) = trimmed.parse::<f64>() {
                        Ok(Value::Int(f as i64, IW::I64))
                    } else {
                        Err(RuntimeError::at(format!("to_int: cannot convert {:?} to int", s), sp))
                    }
                }
                other => Err(RuntimeError::at(
                    format!("to_int: cannot convert {} to int",
                        other.map(|v| v.type_name()).unwrap_or("nil")), sp)),
            }
        }
        "to_float" => {
            match args.first() {
                Some(Value::Float(f, w)) => Ok(Value::Float(*f, *w)),
                Some(Value::Int(n, _))   => Ok(Value::Float(*n as f64, FW::F64)),
                Some(Value::Bool(b))  => Ok(Value::Float(*b as i64 as f64, FW::F64)),
                Some(Value::Str(s))   => {
                    s.trim().parse::<f64>().map(|v| Value::Float(v, FW::F64))
                        .map_err(|_| RuntimeError::at(
                            format!("to_float: cannot parse {:?}", s), sp))
                }
                other => Err(RuntimeError::at(
                    format!("to_float: cannot convert {} to float",
                        other.map(|v| v.type_name()).unwrap_or("nil")), sp)),
            }
        }
        // IEEE-754 bit reinterpret (#189). demoniC floats are f64 internally;
        // `f32_to_bits` narrows to f32 first, then reinterprets the 32 bits.
        "f32_to_bits" => {
            let f = args.first().and_then(|v| v.as_float()).ok_or_else(|| {
                RuntimeError::at("f32_to_bits: numeric argument required".to_string(), sp.clone())
            })?;
            Ok(Value::Int((f as f32).to_bits() as i64, IW::I64))
        }
        "f32_from_bits" => {
            let n = args.first().and_then(|v| v.as_int()).ok_or_else(|| {
                RuntimeError::at("f32_from_bits: integer argument required".to_string(), sp.clone())
            })?;
            // #473: the result IS an f32 by construction, so it carries f32
            // width — matching the JIT, which returns a true `ScalarKind::F32`.
            Ok(Value::Float(f32::from_bits(n as u32) as f64, FW::F32))
        }
        "str_repeat" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("str_repeat: str arg required", sp)),
            };
            let n = match args.get(1) {
                Some(v) => v.as_int()
                    .ok_or_else(|| RuntimeError::at("str_repeat: integer count required", sp.clone()))?,
                None => return Err(RuntimeError::at("str_repeat: count arg required", sp)),
            };
            Ok(Value::Str(s.repeat(n.max(0) as usize)))
        }
        "clamp" => {
            if args.len() < 3 {
                return Err(RuntimeError::at("clamp: requires (value, lo, hi)", sp));
            }
            // #597: an all-integer clamp involving a `u64` compares in the
            // unsigned domain and never goes through f64 — u64::MAX's pattern
            // reads as -1 signed (so `clamp(u64::MAX, 0, 10)` returned 0
            // instead of 10) and is not exactly f64-representable either. The
            // `lo > hi` guard is the same check the f64 path below does, done
            // on the same unsigned reading.
            if let (Value::Int(x, wx), Value::Int(lo, wlo), Value::Int(hi, whi)) =
                (&args[0], &args[1], &args[2])
            {
                if *wx == IW::U64 || *wlo == IW::U64 || *whi == IW::U64 {
                    let (ux, ulo, uhi) = (*x as u64, *lo as u64, *hi as u64);
                    if ulo > uhi {
                        return Err(RuntimeError::at(
                            format!("clamp: lo ({}) must be <= hi ({})", ulo, uhi), sp.clone()));
                    }
                    return Ok(Value::Int(ux.clamp(ulo, uhi) as i64, IW::U64));
                }
            }
            // #210: std `clamp` panics (aborting the interpreter) if lo > hi or on
            // a NaN bound. Validate via an f64 view of the bounds first and return
            // a clean RuntimeError. Non-numeric bounds fall through to the match's
            // "numeric args required" arm.
            if let (Some(lo), Some(hi)) = (args[1].as_float(), args[2].as_float()) {
                if lo.is_nan() || hi.is_nan() {
                    return Err(RuntimeError::at("clamp: bound must not be NaN", sp.clone()));
                }
                if lo > hi {
                    return Err(RuntimeError::at(
                        format!("clamp: lo ({}) must be <= hi ({})", lo, hi), sp.clone()));
                }
            }
            match (&args[0], &args[1], &args[2]) {
                (Value::Int(x, _), Value::Int(lo, _), Value::Int(hi, _))     => Ok(Value::Int((*x).clamp(*lo, *hi), IW::I64)),
                (Value::Float(x, _), Value::Float(lo, _), Value::Float(hi, _)) => Ok(Value::Float(x.clamp(*lo, *hi), FW::F64)),
                (Value::Int(x, _), Value::Float(lo, _), Value::Float(hi, _))  => Ok(Value::Float((*x as f64).clamp(*lo, *hi), FW::F64)),
                (Value::Float(x, _), Value::Int(lo, _), Value::Int(hi, _))    => Ok(Value::Float(x.clamp(*lo as f64, *hi as f64), FW::F64)),
                _ => Err(RuntimeError::at("clamp: numeric args required", sp)),
            }
        }
        "len" => {
            match args.first() {
                Some(Value::Tensor(t)) => Ok(Value::Int(t.shape().first().copied().unwrap_or(0) as i64, IW::I64)),
                Some(Value::Tuple(vs)) => Ok(Value::Int(vs.len() as i64, IW::I64)),
                Some(Value::Str(s))    => Ok(Value::Int(s.len() as i64, IW::I64)),
                Some(Value::List(vs))  => Ok(Value::Int(vs.len() as i64, IW::I64)),
                Some(Value::Map(m))    => Ok(Value::Int(m.borrow().len() as i64, IW::I64)),
                Some(Value::Range { start, end, inclusive }) => {
                    let e = if *inclusive { *end + 1 } else { *end };
                    Ok(Value::Int((e - *start).max(0), IW::I64))
                }
                // Located, like every other runtime error the two backends
                // are compared on: the JIT's `dmc_str_len` names the span, and
                // an unlocated twin here is a presentation divergence on the
                // same program.
                _ => Err(RuntimeError::at("len: requires tensor, tuple, str, list, or map", sp)),
            }
        }
        "sum" => {
            if let Some(Value::Tensor(t)) = args.first() {
                // #481: accumulate at the tensor's own width and carry that
                // width out. The JIT's reduction is native f32 and returns an
                // f32 scalar, so both halves matter — accumulating in f64
                // disagreed with the equivalent loop, and tagging the result
                // f64 made the very next `+ 0.1f32` disagree as well.
                Ok(Value::Float(tensor_sum(t), t.float_width()))
            } else if let Some(v) = args.first() {
                Ok(Value::Float(v.as_float().unwrap_or(0.0), float_width(v)))
            } else { Ok(Value::Float(0.0, FW::F64)) }
        }
        // trace(M) — sum of the diagonal elements of a square 2D tensor.
        // Requires a rank-2 tensor with equal dimensions (M[0] == M[1]).
        // A cold ChatGPT instance wrote `trace()` expecting it to exist; now it does.
        "trace" => {
            match args.first() {
                Some(Value::Tensor(t)) => {
                    let shape = t.shape();
                    if shape.len() != 2 || shape[0] != shape[1] {
                        return Err(RuntimeError::at(
                            format!("trace: expected a square 2D tensor, got shape {:?}", shape),
                            sp,
                        ));
                    }
                    let n = shape[0];
                    // #481: `trace` is `sum(diag(t))`, so it reduces at the
                    // tensor's width like every other reduction. (interp-only —
                    // the JIT does not lower `trace` — but an f32 trace that
                    // disagreed with an f32 sum would be the same defect.)
                    let diag = (0..n).map(|i| t[[i, i]]);
                    let diag_sum = if t.dtype.rounds_as_f32() { sum_f32(diag) } else { diag.sum() };
                    Ok(Value::Float(diag_sum, t.float_width()))
                }
                _ => Err(RuntimeError::at("trace: expected a 2D tensor argument", sp)),
            }
        }
        // `diag(m)` — extract the diagonal of a square 2D tensor: [N,N] -> [N] (#191).
        // The complement to `trace` (which sums what `diag` returns). For the
        // transpose of a matrix use the postfix operator `m'`.
        "diag" => {
            match args.first() {
                Some(Value::Tensor(t)) => {
                    let shape = t.shape();
                    if shape.len() != 2 || shape[0] != shape[1] {
                        return Err(RuntimeError::at(
                            format!("diag: expected a square 2D tensor, got shape {:?}", shape),
                            sp,
                        ));
                    }
                    let n = shape[0];
                    let out: Vec<f64> = (0..n).map(|i| t[[i, i]]).collect();
                    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), out)
                        .map_err(|e| RuntimeError::at(format!("diag: {}", e), sp.clone()))?;
                    // Elements are extracted verbatim — keep the source dtype.
                    Ok(Value::tensor_dt(arr, t.dtype))
                }
                _ => Err(RuntimeError::at("diag: expected a 2D tensor argument", sp)),
            }
        }
        "mean" => {
            if let Some(Value::Tensor(t)) = args.first() {
                // #258: mean of an empty tensor is 0/0 = NaN — the honest
                // "undefined" answer, matching the JIT (which the previous
                // special-case-to-0 diverged from). #481: reduce and divide at
                // the tensor's width.
                Ok(Value::Float(tensor_mean(t), t.float_width()))
            } else { Ok(Value::Float(0.0, FW::F64)) }
        }
        "max" | "min" => {
            // Variadic scalar min/max. For tensor input, reduce over all elements.
            // NaN PROPAGATES (#300 ruling): `f64::max`/`f64::min` skip NaN
            // (np.nanmax), but the JIT propagates it (np.max); a single NaN in the
            // input makes the whole reduction NaN, matching the JIT and sum/mean.
            let reduce = |it: &mut dyn Iterator<Item = f64>| -> f64 {
                let mut acc = if name == "max" { f64::NEG_INFINITY } else { f64::INFINITY };
                for x in it {
                    if x.is_nan() { return f64::NAN; }
                    acc = if name == "max" { acc.max(x) } else { acc.min(x) };
                }
                acc
            };
            // #597: an all-integer variadic `min`/`max` with a `u64` operand
            // must select in the UNSIGNED domain and must not touch f64 at
            // all: u64::MAX's pattern reads as -1 when taken signed (so
            // `min(u64::MAX, 1)` returned -1 and `max` returned 1, both
            // backwards), and u64::MAX is not exactly representable in an f64
            // anyway. Only this width takes the integer path — `u8`/`u16`/
            // `u32` are stored zero-extended and exactly f64-representable, so
            // their long-standing float-tagged result is left alone.
            if args.len() >= 2
                && args.iter().all(|v| matches!(v, Value::Int(..)))
                && args.iter().any(|v| matches!(v, Value::Int(_, IW::U64)))
            {
                let mut it = args.iter().filter_map(|v| v.as_int()).map(|n| n as u64);
                let first = it.next().unwrap_or(0);
                let acc = it.fold(first, |a, b| if name == "max" { a.max(b) } else { a.min(b) });
                return Ok(Value::Int(acc as i64, IW::U64));
            }
            if let Some(Value::Tensor(t)) = args.first() {
                if args.len() == 1 {
                    // #481: no accumulation to round — max/min just select an
                    // element — but the SELECTED element is an f32 when the
                    // tensor is, and the JIT types the result that way. Tagging
                    // it f64 made `max(t) + 0.1f32` diverge even though `max(t)`
                    // alone agreed.
                    return Ok(Value::Float(reduce(&mut t.iter().copied()), t.float_width()));
                }
            }
            // Variadic scalar form: f32 only when every operand is (FW::join).
            let w = args.iter().filter(|v| v.as_float().is_some())
                .fold(None::<FW>, |acc, v| Some(match acc {
                    None => float_width(v),
                    Some(a) => FW::join(a, float_width(v)),
                })).unwrap_or(FW::F64);
            let vals: Vec<f64> = args.iter().filter_map(|v| v.as_float()).collect();
            Ok(Value::Float(reduce(&mut vals.into_iter()), w))
        }
        "softmax"    => builtin_softmax(&args, sp),
        // #335: sort a tensor ascending along its LAST axis (numpy default).
        // 1-D → fully sorted; N-D → each row sorted. interp-only.
        "sort" => {
            let x = required_tensor(&args, 0, "sort", &sp)?;
            let mut out = x.clone();
            if out.ndim() >= 1 {
                let last = Axis(out.ndim() - 1);
                for mut lane in out.lanes_mut(last) {
                    let mut v: Vec<f64> = lane.iter().copied().collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    for (slot, val) in lane.iter_mut().zip(v) { *slot = val; }
                }
            }
            Ok(Value::tensor(out))
        }
        // #335: median of all elements (full tensor capacity, like sum/mean).
        // Even count -> mean of the two central values; returns an f64 scalar.
        "median" => {
            let x = required_tensor(&args, 0, "median", &sp)?;
            let mut v: Vec<f64> = x.iter().copied().collect();
            if v.is_empty() {
                return Err(RuntimeError::msg("median: empty tensor"));
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = v.len();
            let m = if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 };
            Ok(Value::Float(m, FW::F64))
        }
        // Elementwise activations: accept an f32 scalar (-> f32 scalar) or an
        // f32 tensor of any shape (-> same-shape f32 tensor). Numeric formulas
        // match the JIT exactly (interp computes in f64, then f32-rounds on the
        // tensor path via Value::tensor / on the scalar path the result is f32-
        // representable). See jit.rs::emit_gelu_f32 and the builtin_*_f32 externs.
        "relu" | "sigmoid" | "tanh" | "gelu" | "silu" | "elu" | "mish"
                     => builtin_activation(name, &args, sp),
        "argmax"     => builtin_arg_reduce(&args, sp, /*max=*/true),
        "argmin"     => builtin_arg_reduce(&args, sp, /*max=*/false),
        "rms_norm"   => builtin_rms_norm(&args, sp),
        "layer_norm" => builtin_layer_norm(&args, sp),
        "rope"       => builtin_rope(&args, sp),
        "attn"       => builtin_attn(&args, sp),
        "attn_gqa"   => builtin_attn_gqa(&args, sp),
        // Variance-trait + per-axis reductions — kept at parity with the JIT.
        "variance"            => builtin_variance(&args, sp),
        "pull_to_mean"        => builtin_pull_to_mean(&args, sp),
        "sum_along"           => builtin_reduce_along(&args, sp, "sum_along", false),
        "mean_along"          => builtin_reduce_along(&args, sp, "mean_along", true),
        "max_along"           => builtin_max_along(&args, sp),
        "min_along"           => builtin_min_along(&args, sp),
        "variance_along"      => builtin_variance_along(&args, sp),
        "pull_to_mean_along"  => builtin_pull_to_mean_along(&args, sp),
        // JIT typed-print shims (the interpreter's polymorphic `print` covers
        // these, but they exist so the JIT's print family has parity). These
        // newline-terminate, matching the JIT printers.
        "print_tensor" | "print_i64" | "print_f64" | "print_bool" | "print_nil" => {
            match args.first() {
                Some(v) => println!("{}", v),
                None => println!(),
            }
            Ok(Value::Nil)
        }

        "embed" => builtin_embed(&args, sp),
        // join(sep, parts...) or join(sep, tuple_of_strs)
        "join" => {
            let sep = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("join: first arg (separator) must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("join: needs at least a separator arg", sp)),
            };
            let parts: Vec<String> = if args.len() == 2 {
                // join(sep, collection) form
                match &args[1] {
                    Value::Tuple(vs) => vs.iter().map(|v| format!("{}", v)).collect(),
                    Value::List(vs) => vs.iter().map(|v| format!("{}", v)).collect(),
                    Value::Str(s) => vec![s.clone()],
                    other => return Err(RuntimeError::at(
                        format!("join: second arg must be a tuple, list, or str, got {}", other.type_name()), sp)),
                }
            } else {
                // join(sep, s1, s2, ...) variadic form
                args[1..].iter().map(|v| format!("{}", v)).collect()
            };
            Ok(Value::Str(parts.join(&sep)))
        }
        // File I/O
        "read_file" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("read_file: path must be str", sp)),
            };
            match std::fs::read_to_string(&path) {
                Ok(content) => Ok(Value::Tuple(vec![Value::Str(content), Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(e.to_string())])),
            }
        }
        // read_bytes(path) -> (Tensor[i64,[N]], err): exact binary read, one i64 per
        // byte (0-255). Unlike read_file it never decodes UTF-8, so weight files
        // (safetensors/gguf, raw little-endian f32) round-trip losslessly. The
        // resulting tensor feeds @cast(Model){bytes} and f32_from_bits directly.
        "read_bytes" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("read_bytes: path must be str", sp)),
            };
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let n = bytes.len();
                    let data: Vec<f64> = bytes.iter().map(|&b| b as f64).collect();
                    let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data)
                        .map_err(|e| RuntimeError::at(format!("read_bytes: {}", e), sp))?;
                    // Raw bytes with no declared element type: i64-wide, so
                    // byte arithmetic downstream is not silently taken mod 256.
                    Ok(Value::Tuple(vec![Value::tensor_dt(arr, DType::Int(IW::I64)), Value::Nil]))
                }
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(e.to_string())])),
            }
        }
        "write_file" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("write_file: path must be str", sp)),
            };
            let content = match args.get(1) {
                Some(Value::Str(c)) => c.clone(),
                _ => return Err(RuntimeError::at("write_file: content must be str", sp)),
            };
            match std::fs::write(&path, content) {
                Ok(()) => Ok(Value::Tuple(vec![Value::Nil, Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(e.to_string())])),
            }
        }
        "append_file" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("append_file: path must be str", sp)),
            };
            let content = match args.get(1) {
                Some(Value::Str(c)) => c.clone(),
                _ => return Err(RuntimeError::at("append_file: content must be str", sp)),
            };
            use std::io::Write;
            match std::fs::OpenOptions::new().append(true).create(true).open(&path) {
                Ok(mut f) => match f.write_all(content.as_bytes()) {
                    Ok(()) => Ok(Value::Tuple(vec![Value::Nil, Value::Nil])),
                    Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(e.to_string())])),
                },
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(e.to_string())])),
            }
        }
        "file_exists" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("file_exists: path must be str", sp)),
            };
            Ok(Value::Bool(std::path::Path::new(&path).exists()))
        }
        "allreduce" => Err(RuntimeError::at(
            "`allreduce` is a distributed collective with no single-machine semantics; \
            the interpreter cannot execute it. Use --check for type-only validation."
            .to_string(), sp,
        )),
        "load_batch" | "data_iter" => Err(RuntimeError::at(
            format!("`{}` requires a data-loading runtime the interpreter doesn't have. \
            Provide the tensor data directly in your program, or run under a backend that implements I/O.",
            name), sp,
        )),

        // ── Dynamic collections: list ─────────────────────────────────────────
        "list" => Ok(Value::list(args)), // list() → empty; list(1,2,3) → [1,2,3]
        "list_push" => {
            let mut lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_push: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_push: requires list and item args", sp)),
            };
            // Release the argument vector's handle on the buffer before writing:
            // `make_mut` copies whenever a second holder exists, and `args[0]`
            // would otherwise be one (#452).
            args[0] = Value::Nil;
            let item = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_push: requires item arg", sp))?;
            Rc::make_mut(&mut lst).push(item);
            Ok(Value::List(lst))
        }
        "list_pop" => {
            let mut lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_pop: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_pop: requires list arg", sp)),
            };
            if lst.is_empty() {
                return Err(RuntimeError::at("list_pop: list is empty", sp));
            }
            args[0] = Value::Nil;
            let last = Rc::make_mut(&mut lst).pop().unwrap();
            Ok(Value::Tuple(vec![Value::List(lst), last]))
        }
        "list_get" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_get: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_get: requires list and index args", sp)),
            };
            let idx_raw = args.get(1).and_then(|v| v.as_int()).ok_or_else(||
                RuntimeError::at("list_get: index must be an integer", sp.clone()))?;
            let len = lst.len() as i64;
            let idx = if idx_raw < 0 { (len + idx_raw) as usize } else { idx_raw as usize };
            lst.get(idx).cloned().ok_or_else(||
                RuntimeError::at(format!("list_get: index {} out of range for list of length {}", idx_raw, len), sp))
        }
        "list_set" => {
            let mut lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_set: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_set: requires list, index, and value args", sp)),
            };
            let idx_raw = args.get(1).and_then(|v| v.as_int()).ok_or_else(||
                RuntimeError::at("list_set: index must be an integer", sp.clone()))?;
            let len = lst.len() as i64;
            let idx = if idx_raw < 0 { (len + idx_raw) as usize } else { idx_raw as usize };
            if idx >= lst.len() {
                return Err(RuntimeError::at(
                    format!("list_set: index {} out of range for list of length {}", idx_raw, len), sp));
            }
            args[0] = Value::Nil;
            let val = args.into_iter().nth(2).ok_or_else(||
                RuntimeError::at("list_set: requires value arg", sp.clone()))?;
            Rc::make_mut(&mut lst)[idx] = val;
            Ok(Value::List(lst))
        }
        "list_len" => {
            match args.first() {
                Some(Value::List(vs)) => Ok(Value::Int(vs.len() as i64, IW::I64)),
                Some(other) => Err(RuntimeError::at(
                    format!("list_len: arg must be list, got {}", other.type_name()), sp)),
                None => Err(RuntimeError::at("list_len: requires list arg", sp)),
            }
        }
        "list_concat" => {
            let a = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_concat: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_concat: requires two list args", sp)),
            };
            let b = match args.get(1) {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_concat: second arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_concat: requires second list arg", sp)),
            };
            let mut result = a;
            args[0] = Value::Nil;
            Rc::make_mut(&mut result).extend(b.iter().cloned());
            Ok(Value::List(result))
        }
        "list_slice" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_slice: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_slice: requires list, start, end args", sp)),
            };
            let len = lst.len() as i64;
            let start_raw = args.get(1).and_then(|v| v.as_int()).ok_or_else(||
                RuntimeError::at("list_slice: start must be an integer", sp.clone()))?;
            let end_raw = args.get(2).and_then(|v| v.as_int()).ok_or_else(||
                RuntimeError::at("list_slice: end must be an integer", sp.clone()))?;
            let start = (if start_raw < 0 { (len + start_raw).max(0) } else { start_raw.min(len) }) as usize;
            let end = (if end_raw < 0 { (len + end_raw).max(0) } else { end_raw.min(len) }) as usize;
            let end = end.max(start);
            Ok(Value::list(lst[start..end].to_vec()))
        }
        "list_contains" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_contains: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_contains: requires list and item args", sp)),
            };
            let item = args.get(1).ok_or_else(||
                RuntimeError::at("list_contains: requires item arg", sp.clone()))?;
            let found = lst.iter().any(|v| match (v, item) {
                (Value::Int(a, _), Value::Int(b, _)) => a == b,
                (Value::Float(a, _), Value::Float(b, _)) => a == b,
                (Value::Int(a, _), Value::Float(b, _)) => (*a as f64) == *b,
                (Value::Float(a, _), Value::Int(b, _)) => *a == (*b as f64),
                (Value::Str(a), Value::Str(b)) => a == b,
                (Value::Bool(a), Value::Bool(b)) => a == b,
                (Value::Nil, Value::Nil) => true,
                _ => false,
            });
            Ok(Value::Bool(found))
        }
        "list_rev" => {
            let mut lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_rev: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_rev: requires list arg", sp)),
            };
            args[0] = Value::Nil;
            Rc::make_mut(&mut lst).reverse();
            Ok(Value::List(lst))
        }

        // ── Dynamic collections: map ──────────────────────────────────────────
        "map" | "map_new" => Ok(Value::Map(Rc::new(RefCell::new(HashMap::new())))),
        "map_set" => {
            let m = match args.first() {
                Some(Value::Map(m)) => m.clone(),  // cheap Rc clone — same backing map
                Some(other) => return Err(RuntimeError::at(
                    format!("map_set: first arg must be map, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_set: requires map, key, value args", sp)),
            };
            let key = match args.get(1) {
                Some(Value::Str(k)) => k.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_set: key must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_set: requires key arg", sp)),
            };
            let val = args.into_iter().nth(2).ok_or_else(||
                RuntimeError::at("map_set: requires value arg", sp))?;
            m.borrow_mut().insert(key, val);
            Ok(Value::Map(m))  // return same Rc — mutation already visible everywhere
        }
        "map_get" => {
            let m = match args.first() {
                Some(Value::Map(m)) => m.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_get: first arg must be map, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_get: requires map and key args", sp)),
            };
            let key = match args.get(1) {
                Some(Value::Str(k)) => k.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_get: key must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_get: requires key arg", sp)),
            };
            let result = m.borrow().get(&key).cloned().unwrap_or(Value::Nil);
            // #540: a map is `str → any` (§3.1) and the JIT backs it with a raw
            // i64 store, so an integer LOSES its declared width by going
            // through one — `map_get(m, k)` is typed i64 on that side. Drop the
            // width here too: keeping it would make `map_get(m,"k") + 1` wrap
            // under `dmc run` and not under `dmc jit`, which is the divergence
            // this whole change removes. Crossing `any` forgoes static
            // guarantees, and the width is one of them.
            Ok(match result { Value::Int(n, _) => Value::Int(n, IW::I64), other => other })
        }
        "map_has" | "map_contains" => {
            let builtin_name = name;
            let m = match args.first() {
                Some(Value::Map(m)) => m.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("{}: first arg must be map, got {}", builtin_name, other.type_name()), sp)),
                None => return Err(RuntimeError::at(format!("{}: requires map and key args", builtin_name), sp)),
            };
            let key = match args.get(1) {
                Some(Value::Str(k)) => k.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("{}: key must be str, got {}", builtin_name, other.type_name()), sp)),
                None => return Err(RuntimeError::at(format!("{}: requires key arg", builtin_name), sp)),
            };
            let has = m.borrow().contains_key(&key);
            Ok(Value::Bool(has))
        }
        "map_del" => {
            let m = match args.first() {
                Some(Value::Map(m)) => m.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_del: first arg must be map, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_del: requires map and key args", sp)),
            };
            let key = match args.get(1) {
                Some(Value::Str(k)) => k.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_del: key must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_del: requires key arg", sp)),
            };
            m.borrow_mut().remove(&key);
            Ok(Value::Map(m))
        }
        "map_keys" => {
            let m = match args.first() {
                Some(Value::Map(m)) => m.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_keys: arg must be map, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_keys: requires map arg", sp)),
            };
            let keys: Vec<Value> = m.borrow().keys().map(|k| Value::Str(k.clone())).collect();
            Ok(Value::list(keys))
        }
        "map_vals" => {
            let m = match args.first() {
                Some(Value::Map(m)) => m.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_vals: arg must be map, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_vals: requires map arg", sp)),
            };
            let vals: Vec<Value> = m.borrow().values().cloned().collect();
            Ok(Value::list(vals))
        }
        "map_len" => {
            match args.first() {
                Some(Value::Map(m)) => Ok(Value::Int(m.borrow().len() as i64, IW::I64)),
                Some(other) => Err(RuntimeError::at(
                    format!("map_len: arg must be map, got {}", other.type_name()), sp)),
                None => Err(RuntimeError::at("map_len: requires map arg", sp)),
            }
        }

        // ── Process / environment ─────────────────────────────────────────────
        "env_var" => {
            let name = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("env_var: name must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("env_var: requires name arg", sp)),
            };
            match std::env::var(&name) {
                Ok(val) => Ok(Value::Tuple(vec![Value::Str(val), Value::Nil])),
                Err(_) => Ok(Value::Tuple(vec![Value::Str(String::new()), Value::Str("not found".to_string())])),
            }
        }
        "argv" => {
            let argv: Vec<Value> = self.argv.iter().map(|s| Value::Str(s.clone())).collect();
            Ok(Value::list(argv))
        }
        "exit" => {
            let code = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            std::process::exit(code as i32);
        }

        // ── Time ──────────────────────────────────────────────────────────────
        "time_ms" => {
            let elapsed = self.start_time.elapsed();
            Ok(Value::Float(elapsed.as_secs_f64() * 1000.0, FW::F64))
        }
        "sleep_ms" => {
            let ms = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
            if ms > 0.0 {
                std::thread::sleep(std::time::Duration::from_secs_f64(ms / 1000.0));
            }
            Ok(Value::Nil)
        }

        "flush" => {
            use std::io::Write;
            let _ = std::io::stdout().flush();
            Ok(Value::Nil)
        }

        "set_raw_mode" => {
            let enable = args.first().and_then(|v| match v {
                Value::Bool(b) => Some(*b),
                Value::Int(n, _) => Some(*n != 0),
                _ => None,
            }).unwrap_or(false);
            // Silently ignore "not a tty" errors — game degrades gracefully
            if enable {
                let _ = crossterm::terminal::enable_raw_mode();
            } else {
                let _ = crossterm::terminal::disable_raw_mode();
            }
            Ok(Value::Nil)
        }

        "read_char_nb" => {
            use crossterm::event::{poll, read, Event, KeyCode, KeyModifiers};
            use std::time::Duration;
            match poll(Duration::from_millis(0)) {
                Ok(true) => match read() {
                    Ok(Event::Key(ke)) => {
                        if ke.modifiers.contains(KeyModifiers::CONTROL) {
                            if ke.code == KeyCode::Char('c') {
                                crossterm::terminal::disable_raw_mode().ok();
                                std::process::exit(0);
                            }
                        }
                        let s = match ke.code {
                            KeyCode::Char(c) => c.to_string(),
                            KeyCode::Enter    => "\n".to_string(),
                            KeyCode::Up       => "up".to_string(),
                            KeyCode::Down     => "down".to_string(),
                            KeyCode::Left     => "left".to_string(),
                            KeyCode::Right    => "right".to_string(),
                            KeyCode::Esc      => "esc".to_string(),
                            _                 => "".to_string(),
                        };
                        Ok(Value::Str(s))
                    }
                    _ => Ok(Value::Str(String::new())),
                },
                _ => Ok(Value::Str(String::new())),
            }
        }

        "read_char" => {
            use crossterm::event::{read, Event, KeyCode, KeyModifiers};
            loop {
                match read() {
                    Ok(Event::Key(ke)) => {
                        if ke.modifiers.contains(KeyModifiers::CONTROL) {
                            if ke.code == KeyCode::Char('c') {
                                crossterm::terminal::disable_raw_mode().ok();
                                std::process::exit(0);
                            }
                        }
                        let s = match ke.code {
                            KeyCode::Char(c) => c.to_string(),
                            KeyCode::Enter    => "\n".to_string(),
                            KeyCode::Up       => "up".to_string(),
                            KeyCode::Down     => "down".to_string(),
                            KeyCode::Left     => "left".to_string(),
                            KeyCode::Right    => "right".to_string(),
                            KeyCode::Esc      => "esc".to_string(),
                            _                 => continue,
                        };
                        return Ok(Value::Str(s));
                    }
                    Ok(_) => continue,
                    Err(e) => return Err(RuntimeError::msg(format!("read_char: {}", e))),
                }
            }
        }

        // ── Extended RNG builtins ─────────────────────────────────────────────
        "rand_seed" => {
            let seed = args.first().and_then(|v| v.as_int()).unwrap_or(0) as u64;
            // SplitMix64 step on the seed so seed=0 still gives a non-zero xorshift state.
            let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            self.rng_state = z ^ (z >> 31);
            Ok(Value::Nil)
        }
        "rand_float" => {
            Ok(Value::Float(self.rand_uniform(), FW::F64))
        }
        "rand_int" => {
            let lo = args.first().and_then(|v| v.as_int()).ok_or_else(||
                RuntimeError::at("rand_int: lo argument required (integer)", sp.clone()))?;
            let hi = args.get(1).and_then(|v| v.as_int()).ok_or_else(||
                RuntimeError::at("rand_int: hi argument required (integer)", sp.clone()))?;
            if hi <= lo {
                return Err(RuntimeError::at(
                    format!("rand_int: hi ({}) must be greater than lo ({})", hi, lo), sp));
            }
            let range = (hi - lo) as u64;
            let n = self.rand_u64() % range;
            Ok(Value::Int(lo + n as i64, IW::I64))
        }
        "rand_normal" => {
            let mean = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
            let std  = args.get(1).and_then(|v| v.as_float()).unwrap_or(1.0);
            let u1 = self.rand_uniform().max(f64::MIN_POSITIVE);
            let u2 = self.rand_uniform();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            Ok(Value::Float(mean + std * z, FW::F64))
        }
        "rand_choice" => {
            let lst: &[Value] = match args.first() {
                Some(Value::List(vs)) => vs.as_slice(),
                Some(Value::Tuple(vs)) => vs.as_slice(),
                Some(other) => return Err(RuntimeError::at(
                    format!("rand_choice: arg must be list or tuple, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("rand_choice: requires list arg", sp)),
            };
            if lst.is_empty() {
                return Err(RuntimeError::at("rand_choice: list is empty", sp));
            }
            let idx = self.rand_u64() as usize % lst.len();
            Ok(lst[idx].clone())
        }

        // ── JSON encode/decode ────────────────────────────────────────────────
        "json_encode" => {
            let val = args.into_iter().next().unwrap_or(Value::Nil);
            Ok(Value::Str(json_encode_value(&val)))
        }
        "json_decode" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("json_decode: arg must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("json_decode: requires str arg", sp)),
            };
            match json_parse(&s) {
                Ok(v) => Ok(Value::Tuple(vec![v, Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(format!("parse error: {}", e))])),
            }
        }

        // ── Typed JSON decode (PORTS.md §6, ASSIMILATE.md §5.1) ───────────────
        // `json_decode` hands back a dynamic value, so anything built on it is
        // untyped all the way out. The typed family commits to one demoniC type
        // per primitive and returns `(T, Err)`: the decoded value, or T's zero
        // and a `decode-*` tag. A value of the wrong JSON type is an error, not
        // a coercion — that is the whole point (an assimilated wrapper's
        // declared return type is only a contract if the boundary enforces it).
        "json_decode_i64" | "json_decode_f64" | "json_decode_str"
        | "json_decode_bool" | "json_decode_list" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("{}: arg must be str, got {}", name, other.type_name()), sp)),
                None => return Err(RuntimeError::at(
                    format!("{}: requires str arg", name), sp)),
            };
            Ok(json_decode_typed(&name["json_decode_".len()..], &s))
        }

        // ── List functional combinators ───────────────────────────────────────
        "list_map" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_map: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_map: requires list and fn args", sp)),
            };
            args[0] = Value::Nil;
            let fn_val = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_map: requires fn arg", sp.clone()))?;
            let mut result = Vec::with_capacity(lst.len());
            for item in into_list_vec(lst) {
                let out = self.call_value(fn_val.clone(), vec![item], sp.clone())?;
                result.push(out);
            }
            Ok(Value::list(result))
        }
        "list_filter" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_filter: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_filter: requires list and fn args", sp)),
            };
            args[0] = Value::Nil;
            let fn_val = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_filter: requires fn arg", sp.clone()))?;
            let mut result = Vec::new();
            for item in into_list_vec(lst) {
                let out = self.call_value(fn_val.clone(), vec![item.clone()], sp.clone())?;
                if out.as_bool().unwrap_or(match &out {
                    Value::Int(n, _) => *n != 0,
                    Value::Float(x, _) => *x != 0.0,
                    Value::Nil => false,
                    _ => true,
                }) {
                    result.push(item);
                }
            }
            Ok(Value::list(result))
        }
        "list_reduce" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_reduce: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_reduce: requires list, fn, and init args", sp)),
            };
            let fn_val = args.get(1).ok_or_else(||
                RuntimeError::at("list_reduce: requires fn arg", sp.clone()))?.clone();
            args[0] = Value::Nil;
            let init = args.into_iter().nth(2).ok_or_else(||
                RuntimeError::at("list_reduce: requires init arg", sp.clone()))?;
            let mut acc = init;
            for item in into_list_vec(lst) {
                acc = self.call_value(fn_val.clone(), vec![acc, item], sp.clone())?;
            }
            Ok(acc)
        }
        "list_sort" => {
            let mut lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_sort: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_sort: requires list arg", sp)),
            };
            args[0] = Value::Nil;
            Rc::make_mut(&mut lst).sort_by(cmp_value);
            Ok(Value::List(lst))
        }
        "list_sort_by" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_sort_by: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_sort_by: requires list and fn args", sp)),
            };
            let fn_val = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_sort_by: requires fn arg", sp.clone()))?;
            // Compute keys first (avoids re-calling during sort which would need &mut self)
            let mut keys: Vec<Value> = Vec::with_capacity(lst.len());
            for item in lst.iter() {
                let k = self.call_value(fn_val.clone(), vec![item.clone()], sp.clone())?;
                keys.push(k);
            }
            // Sort by keys using indices
            let mut indices: Vec<usize> = (0..lst.len()).collect();
            indices.sort_by(|&a, &b| cmp_value(&keys[a], &keys[b]));
            let sorted: Vec<Value> = indices.into_iter().map(|i| lst[i].clone()).collect();
            Ok(Value::list(sorted))
        }
        "list_zip" => {
            let a = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_zip: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_zip: requires two list args", sp)),
            };
            let b = match args.get(1) {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_zip: second arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_zip: requires second list arg", sp)),
            };
            let pairs: Vec<Value> = a.iter().zip(b.iter())
                .map(|(x, y)| Value::Tuple(vec![x.clone(), y.clone()]))
                .collect();
            Ok(Value::list(pairs))
        }
        "list_enumerate" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_enumerate: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_enumerate: requires list arg", sp)),
            };
            args[0] = Value::Nil;
            let pairs: Vec<Value> = into_list_vec(lst).into_iter().enumerate()
                .map(|(i, v)| Value::Tuple(vec![Value::Int(i as i64, IW::I64), v]))
                .collect();
            Ok(Value::list(pairs))
        }
        "list_flatten" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_flatten: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_flatten: requires list arg", sp)),
            };
            args[0] = Value::Nil;
            let mut result = Vec::new();
            for item in into_list_vec(lst) {
                match item {
                    Value::List(inner) => result.extend(into_list_vec(inner)),
                    other => result.push(other),
                }
            }
            Ok(Value::list(result))
        }
        "list_uniq" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_uniq: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_uniq: requires list arg", sp)),
            };
            args[0] = Value::Nil;
            let mut result: Vec<Value> = Vec::new();
            for item in into_list_vec(lst) {
                let already_seen = result.iter().any(|prev| values_equal(prev, &item));
                if !already_seen { result.push(item); }
            }
            Ok(Value::list(result))
        }
        "list_sum" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_sum: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_sum: requires list arg", sp)),
            };
            // Preserve integer type if all elements are integers
            if lst.iter().all(|v| matches!(v, Value::Int(_, _))) {
                let total: i64 = lst.iter().filter_map(|v| v.as_int()).sum();
                Ok(Value::Int(total, IW::I64))
            } else {
                let total: f64 = lst.iter().filter_map(|v| v.as_float()).sum();
                Ok(Value::Float(total, FW::F64))
            }
        }
        "list_min" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_min: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_min: requires list arg", sp)),
            };
            if lst.is_empty() {
                return Err(RuntimeError::at("list_min: list is empty", sp));
            }
            let min = lst.iter().cloned().reduce(|a, b| {
                if cmp_value(&a, &b) != std::cmp::Ordering::Greater { a } else { b }
            }).unwrap();
            Ok(min)
        }
        "list_max" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_max: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_max: requires list arg", sp)),
            };
            if lst.is_empty() {
                return Err(RuntimeError::at("list_max: list is empty", sp));
            }
            let max = lst.iter().cloned().reduce(|a, b| {
                if cmp_value(&a, &b) != std::cmp::Ordering::Less { a } else { b }
            }).unwrap();
            Ok(max)
        }

        "list_flat_map" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_flat_map: first arg must be list", sp)),
            };
            let f = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_flat_map: requires function arg", sp.clone()))?;
            args[0] = Value::Nil;
            let mut out = Vec::new();
            for item in into_list_vec(lst) {
                let result = self.call_value(f.clone(), vec![item], sp.clone())?;
                match result {
                    Value::List(vs) => out.extend(into_list_vec(vs)),
                    Value::Tuple(vs) => out.extend(vs),
                    other => out.push(other),
                }
            }
            Ok(Value::list(out))
        }
        "list_partition" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_partition: first arg must be list", sp)),
            };
            let f = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_partition: requires predicate arg", sp.clone()))?;
            args[0] = Value::Nil;
            let mut yes = Vec::new();
            let mut no = Vec::new();
            for item in into_list_vec(lst) {
                let r = self.call_value(f.clone(), vec![item.clone()], sp.clone())?;
                if matches!(r, Value::Bool(true)) { yes.push(item); } else { no.push(item); }
            }
            Ok(Value::Tuple(vec![Value::list(yes), Value::list(no)]))
        }
        "list_head" => {
            match args.first() {
                Some(Value::List(vs)) if !vs.is_empty() => Ok(vs[0].clone()),
                Some(Value::List(_)) => Err(RuntimeError::at("list_head: list is empty", sp)),
                _ => Err(RuntimeError::at("list_head: list arg required", sp)),
            }
        }
        "list_last" => {
            match args.first() {
                Some(Value::List(vs)) if !vs.is_empty() => Ok(vs[vs.len()-1].clone()),
                Some(Value::List(_)) => Err(RuntimeError::at("list_last: list is empty", sp)),
                _ => Err(RuntimeError::at("list_last: list arg required", sp)),
            }
        }
        "list_take" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_take: list arg required", sp)),
            };
            let n = match args.get(1) {
                Some(v) => v.as_int()
                    .ok_or_else(|| RuntimeError::at("list_take: integer count required", sp.clone()))?,
                None => return Err(RuntimeError::at("list_take: count arg required", sp)),
            };
            let take = (n.max(0) as usize).min(lst.len());
            Ok(Value::list(lst[..take].to_vec()))
        }
        "list_drop" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_drop: list arg required", sp)),
            };
            let n = match args.get(1) {
                Some(v) => v.as_int()
                    .ok_or_else(|| RuntimeError::at("list_drop: integer count required", sp.clone()))?,
                None => return Err(RuntimeError::at("list_drop: count arg required", sp)),
            };
            let skip = (n.max(0) as usize).min(lst.len());
            Ok(Value::list(lst[skip..].to_vec()))
        }
        "map_merge" => {
            let base_rc = match args.first() {
                Some(Value::Map(m)) => m.clone(),
                _ => return Err(RuntimeError::at("map_merge: first arg must be map", sp)),
            };
            let overlay_rc = match args.get(1) {
                Some(Value::Map(m)) => m.clone(),
                _ => return Err(RuntimeError::at("map_merge: second arg must be map", sp)),
            };
            // Build a fresh merged map (map_merge produces a new independent map)
            let mut merged: HashMap<String, Value> = base_rc.borrow().clone();
            for (k, v) in overlay_rc.borrow().iter() {
                merged.insert(k.clone(), v.clone());
            }
            Ok(Value::Map(Rc::new(RefCell::new(merged))))
        }
        "list_find" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_find: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_find: requires list and item args", sp)),
            };
            let needle = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_find: requires item arg", sp.clone()))?;
            let idx = lst.iter().position(|v| values_equal(v, &needle));
            Ok(Value::Int(idx.map(|i| i as i64).unwrap_or(-1), IW::I64))
        }
        "list_count" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_count: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_count: requires list and item args", sp)),
            };
            let needle = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_count: requires item arg", sp.clone()))?;
            let count = lst.iter().filter(|v| values_equal(v, &needle)).count();
            Ok(Value::Int(count as i64, IW::I64))
        }
        "list_any" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_any: first arg must be list", sp)),
            };
            let f = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_any: requires predicate arg", sp.clone()))?;
            args[0] = Value::Nil;
            for item in into_list_vec(lst) {
                let r = self.call_value(f.clone(), vec![item], sp.clone())?;
                if matches!(r, Value::Bool(true)) { return Ok(Value::Bool(true)); }
            }
            Ok(Value::Bool(false))
        }
        "list_all" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_all: first arg must be list", sp)),
            };
            let f = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_all: requires predicate arg", sp.clone()))?;
            args[0] = Value::Nil;
            for item in into_list_vec(lst) {
                let r = self.call_value(f.clone(), vec![item], sp.clone())?;
                if !matches!(r, Value::Bool(true)) { return Ok(Value::Bool(false)); }
            }
            Ok(Value::Bool(true))
        }
        "str_pad_left" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("str_pad_left: str arg required", sp)),
            };
            let width = match args.get(1) {
                Some(v) => v.as_int()
                    .ok_or_else(|| RuntimeError::at("str_pad_left: integer width required", sp.clone()))?,
                None => return Err(RuntimeError::at("str_pad_left: width arg required", sp)),
            };
            let pad_ch = match args.get(2) {
                Some(Value::Str(p)) => p.chars().next().unwrap_or(' '),
                None => ' ',
                _ => return Err(RuntimeError::at("str_pad_left: pad arg must be str", sp)),
            };
            let width = width.max(0) as usize;
            let len = s.chars().count();
            let result = if len >= width { s }
                else { pad_ch.to_string().repeat(width - len) + &s };
            Ok(Value::Str(result))
        }
        "str_pad_right" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("str_pad_right: str arg required", sp)),
            };
            let width = match args.get(1) {
                Some(v) => v.as_int()
                    .ok_or_else(|| RuntimeError::at("str_pad_right: integer width required", sp.clone()))?,
                None => return Err(RuntimeError::at("str_pad_right: width arg required", sp)),
            };
            let pad_ch = match args.get(2) {
                Some(Value::Str(p)) => p.chars().next().unwrap_or(' '),
                None => ' ',
                _ => return Err(RuntimeError::at("str_pad_right: pad arg must be str", sp)),
            };
            let width = width.max(0) as usize;
            let len = s.chars().count();
            let result = if len >= width { s }
                else { s + &pad_ch.to_string().repeat(width - len) };
            Ok(Value::Str(result))
        }

        // ── String formatting ─────────────────────────────────────────────────
        "format" => {
            let template = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("format: first arg must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("format: requires template arg", sp)),
            };
            let formatted = apply_format_template(&template, &args[1..])
                .map_err(|e| RuntimeError::at(e, sp))?;
            Ok(Value::Str(formatted))
        }

        // ── Hashing ───────────────────────────────────────────────────────────
        "hash_fnv" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("hash_fnv: arg must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("hash_fnv: requires str arg", sp)),
            };
            Ok(Value::Int(fnv1a_64(&s), IW::I64))
        }
        "hash_crc32" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("hash_crc32: arg must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("hash_crc32: requires str arg", sp)),
            };
            Ok(Value::Int(crc32_ieee(&s), IW::I64))
        }

        // ── Filesystem operations ─────────────────────────────────────────────
        "get_cwd" => {
            match std::env::current_dir() {
                Ok(p) => Ok(Value::Str(p.to_string_lossy().into_owned())),
                Err(e) => Err(RuntimeError::at(format!("get_cwd: {}", e), sp)),
            }
        }
        "list_dir" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("list_dir: path must be str", sp)),
            };
            match std::fs::read_dir(&path) {
                Ok(entries) => {
                    let mut names: Vec<Value> = Vec::new();
                    for entry in entries {
                        match entry {
                            Ok(e) => names.push(Value::Str(e.file_name().to_string_lossy().into_owned())),
                            Err(e) => return Ok(Value::Tuple(vec![
                                Value::list(Vec::new()),
                                Value::Str(format!("error: {}", e)),
                            ])),
                        }
                    }
                    Ok(Value::Tuple(vec![Value::list(names), Value::Nil]))
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::list(Vec::new()),
                    Value::Str(format!("error: {}", e)),
                ])),
            }
        }
        "make_dir" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("make_dir: path must be str", sp)),
            };
            match std::fs::create_dir_all(&path) {
                Ok(()) => Ok(Value::Tuple(vec![Value::Nil, Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(format!("error: {}", e))])),
            }
        }
        "delete_file" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("delete_file: path must be str", sp)),
            };
            match std::fs::remove_file(&path) {
                Ok(()) => Ok(Value::Tuple(vec![Value::Nil, Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(format!("error: {}", e))])),
            }
        }
        "delete_dir" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("delete_dir: path must be str", sp)),
            };
            match std::fs::remove_dir(&path) {
                Ok(()) => Ok(Value::Tuple(vec![Value::Nil, Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(format!("error: {}", e))])),
            }
        }
        "rename_file" => {
            let from = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("rename_file: from must be str", sp)),
            };
            let to = match args.get(1) {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("rename_file: to must be str", sp)),
            };
            match std::fs::rename(&from, &to) {
                Ok(()) => Ok(Value::Tuple(vec![Value::Nil, Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(format!("error: {}", e))])),
            }
        }
        "file_size" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("file_size: path must be str", sp)),
            };
            match std::fs::metadata(&path) {
                Ok(meta) => Ok(Value::Tuple(vec![Value::Int(meta.len() as i64, IW::I64), Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Int(0, IW::I64), Value::Str(format!("error: {}", e))])),
            }
        }
        "path_join" => {
            let a = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("path_join: a must be str", sp)),
            };
            let b = match args.get(1) {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("path_join: b must be str", sp)),
            };
            let joined = std::path::Path::new(&a).join(&b);
            Ok(Value::Str(joined.to_string_lossy().into_owned()))
        }
        "path_dirname" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("path_dirname: path must be str", sp)),
            };
            let parent = std::path::Path::new(&path)
                .parent()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            Ok(Value::Str(parent))
        }
        "path_basename" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("path_basename: path must be str", sp)),
            };
            let base = std::path::Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            Ok(Value::Str(base))
        }
        "path_exists" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("path_exists: path must be str", sp)),
            };
            Ok(Value::Bool(std::path::Path::new(&path).exists()))
        }
        "path_is_dir" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("path_is_dir: path must be str", sp)),
            };
            Ok(Value::Bool(std::path::Path::new(&path).is_dir()))
        }
        "path_is_file" => {
            let path = match args.first() {
                Some(Value::Str(p)) => p.clone(),
                _ => return Err(RuntimeError::at("path_is_file: path must be str", sp)),
            };
            Ok(Value::Bool(std::path::Path::new(&path).is_file()))
        }

        // ── Process execution ─────────────────────────────────────────────────
        "exec_cmd" => {
            let cmd = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("exec_cmd: cmd must be str", sp)),
            };
            let arg_strings: Vec<String> = match args.get(1) {
                Some(Value::List(vs)) => vs.iter().map(|v| format!("{}", v)).collect(),
                Some(Value::Nil) | None => Vec::new(),
                Some(other) => return Err(RuntimeError::at(
                    format!("exec_cmd: args must be list, got {}", other.type_name()), sp)),
            };
            match std::process::Command::new(&cmd).args(&arg_strings).output() {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                    let code = out.status.code().unwrap_or(-1) as i64;
                    Ok(Value::Tuple(vec![Value::Str(stdout), Value::Str(stderr), Value::Int(code, IW::I64)]))
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::Str(String::new()),
                    Value::Str(format!("failed to spawn: {}", e)),
                    Value::Int(-1, IW::I64),
                ])),
            }
        }

        // ── Ports (#402, PORTS.md §2/§6/§7.1) ────────────────────────────────
        // Process-port floor: a child runtime speaking line-oriented JSON
        // over stdin/stdout. All three builtins return `(_, Err)` where Err
        // is nil or a str beginning with a PORTS.md §6 tag; code must match
        // only the tag prefix. The protocol itself lives in `crate::ports`,
        // shared with the JIT; these arms only shape the result into Values.
        "port_open" => {
            let lang = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("port_open: lang must be str", sp)),
            };
            Ok(match self.ports.open(&lang) {
                Ok(handle) => Value::Tuple(vec![Value::Opaque(handle), Value::Nil]),
                Err(tag) => Value::Tuple(vec![Value::Nil, Value::Str(tag)]),
            })
        }
        "port_call" => {
            let id = match port_handle_id(args.first()) {
                Some(id) => id,
                None => return Err(RuntimeError::at(
                    "port_call: first arg must be a Port handle from port_open", sp)),
            };
            let name = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("port_call: name must be str", sp)),
            };
            let payload = match args.get(2) {
                Some(Value::Str(s)) => s.clone(),
                Some(Value::Nil) | None => String::new(),
                _ => return Err(RuntimeError::at("port_call: payload must be a JSON str", sp)),
            };
            Ok(match self.ports.call(id, &name, &payload) {
                Ok(out) => Value::Tuple(vec![Value::Str(out), Value::Nil]),
                Err(tag) => Value::Tuple(vec![Value::Str(String::new()), Value::Str(tag)]),
            })
        }
        "port_close" => {
            let id = match port_handle_id(args.first()) {
                Some(id) => id,
                None => return Err(RuntimeError::at(
                    "port_close: arg must be a Port handle from port_open", sp)),
            };
            Ok(match self.ports.close(id) {
                Ok(()) => Value::Tuple(vec![Value::Nil, Value::Nil]),
                Err(tag) => Value::Tuple(vec![Value::Nil, Value::Str(tag)]),
            })
        }

        // ── Tensor copy mode (PORTS.md §3.2) ─────────────────────────────────
        // A tensor crosses the port boundary as one canonical-JSON envelope
        // carrying `{dtype, shape, layout}` plus a base64 payload buffer.
        // `crate::ports` owns every byte of that text; these arms only read
        // the interpreter's f64-backed storage into it and back out.
        "port_tensor_encode" => {
            let t = match args.first() {
                Some(Value::Tensor(t)) => t,
                Some(other) => return Err(RuntimeError::at(format!(
                    "port_tensor_encode: arg must be a tensor, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at(
                    "port_tensor_encode: arg must be a tensor", sp)),
            };
            let store = wire_dtype_of(t.dtype).ok_or_else(|| RuntimeError::at(
                "port_tensor_encode: a `trit` tensor has no copy-mode wire dtype \
                 (PORTS.md §3.2)", sp.clone()))?;
            let shape: Vec<i64> = t.shape().iter().map(|&d| d as i64).collect();
            let vals: Vec<f64> = t.iter().copied().collect();
            match crate::ports::pack_from_f64(store, &shape, &vals) {
                Ok(text) => Ok(Value::Str(text)),
                Err(msg) => Err(RuntimeError::at(format!("port_tensor_encode: {}", msg), sp)),
            }
        }
        "port_tensor_decode" => {
            let text = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at(
                    "port_tensor_decode: first arg must be the envelope str", sp)),
            };
            let like = match args.get(1) {
                Some(Value::Tensor(t)) => t,
                _ => return Err(RuntimeError::at(
                    "port_tensor_decode: second arg must be a tensor declaring the \
                     expected dtype and shape", sp)),
            };
            let want = wire_dtype_of(like.dtype).ok_or_else(|| RuntimeError::at(
                "port_tensor_decode: a `trit` tensor has no copy-mode wire dtype \
                 (PORTS.md §3.2)", sp.clone()))?;
            let dims: Vec<usize> = like.shape().to_vec();
            let shape: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
            let (dtype, numel) = (like.dtype, dims.iter().product::<usize>());
            Ok(match crate::ports::unpack_raw(&text, want, &shape) {
                Ok(bytes) => {
                    let vals = crate::ports::raw_to_f64(want, &bytes);
                    let arr = ArrayD::from_shape_vec(IxDyn(&dims), vals)
                        .map_err(|e| RuntimeError::at(
                            format!("port_tensor_decode: {}", e), sp.clone()))?;
                    Value::Tuple(vec![Value::tensor_dt(arr, dtype), Value::Nil])
                }
                // The value half is the declared tensor's zero, the §3.1
                // discipline: `(T, Err)` stays well-typed on the failure path.
                Err(tag) => Value::Tuple(vec![
                    Value::tensor_dt(ArrayD::from_shape_vec(
                        IxDyn(&dims), vec![0.0; numel]).unwrap(), dtype),
                    Value::Str(tag),
                ]),
            })
        }

        // ── CLI argument builtins ─────────────────────────────────────────────
        "cli_arg" => {
            let name = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("cli_arg: first arg must be str name", sp)),
            };
            let default = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => String::new(),
            };
            let flag = format!("--{}", name);
            for i in 0..self.argv.len() {
                if self.argv[i] == flag {
                    if i + 1 < self.argv.len() {
                        return Ok(Value::Str(self.argv[i + 1].clone()));
                    }
                }
                let prefix = format!("--{}=", name);
                if self.argv[i].starts_with(&prefix) {
                    return Ok(Value::Str(self.argv[i][prefix.len()..].to_string()));
                }
            }
            Ok(Value::Str(default))
        }
        "cli_flag" => {
            let name = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("cli_flag: first arg must be str name", sp)),
            };
            let flag = format!("--{}", name);
            let found = self.argv.iter().any(|a| a == &flag);
            Ok(Value::Bool(found))
        }
        "cli_positional" => {
            let n = match args.first() {
                Some(Value::Int(n, _)) => *n,
                _ => return Err(RuntimeError::at("cli_positional: first arg must be i64", sp)),
            };
            // Skip over values that follow --flag entries
            let mut pos_args: Vec<String> = Vec::new();
            let mut i = 0;
            while i < self.argv.len() {
                let arg = &self.argv[i];
                if arg.starts_with("--") && !arg.contains('=') {
                    // Skip this flag and the next value (if it doesn't start with --)
                    i += 1;
                    if i < self.argv.len() && !self.argv[i].starts_with("--") {
                        i += 1;
                    }
                } else if arg.starts_with("--") {
                    // --key=value form, skip just this
                    i += 1;
                } else {
                    pos_args.push(arg.clone());
                    i += 1;
                }
            }
            let idx = if n < 0 { return Ok(Value::Str(String::new())); } else { n as usize };
            Ok(Value::Str(pos_args.get(idx).cloned().unwrap_or_default()))
        }
        "cli_positional_count" => {
            let mut count = 0usize;
            let mut i = 0;
            while i < self.argv.len() {
                let arg = &self.argv[i];
                if arg.starts_with("--") && !arg.contains('=') {
                    i += 1;
                    if i < self.argv.len() && !self.argv[i].starts_with("--") {
                        i += 1;
                    }
                } else if arg.starts_with("--") {
                    i += 1;
                } else {
                    count += 1;
                    i += 1;
                }
            }
            Ok(Value::Int(count as i64, IW::I64))
        }

        // ── Regex ─────────────────────────────────────────────────────────────
        "regex_match" => {
            let pattern = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_match: pattern must be str", sp)),
            };
            let text = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_match: text must be str", sp)),
            };
            match regex::Regex::new(&pattern) {
                Ok(re) => Ok(Value::Bool(re.is_match(&text))),
                Err(e) => Err(RuntimeError::at(
                    format!("regex_match: invalid pattern `{}`: {}", pattern, e), sp)),
            }
        }
        "regex_find" => {
            let pattern = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_find: pattern must be str", sp)),
            };
            let text = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_find: text must be str", sp)),
            };
            match regex::Regex::new(&pattern) {
                Ok(re) => {
                    match re.find(&text) {
                        Some(m) => Ok(Value::Tuple(vec![
                            Value::Str(m.as_str().to_string()),
                            Value::Int(m.start() as i64, IW::I64),
                        ])),
                        None => Ok(Value::Tuple(vec![
                            Value::Str(String::new()),
                            Value::Int(-1, IW::I64),
                        ])),
                    }
                }
                Err(e) => Err(RuntimeError::at(
                    format!("regex_find: invalid pattern `{}`: {}", pattern, e), sp)),
            }
        }
        "regex_find_all" => {
            let pattern = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_find_all: pattern must be str", sp)),
            };
            let text = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_find_all: text must be str", sp)),
            };
            match regex::Regex::new(&pattern) {
                Ok(re) => {
                    let matches: Vec<Value> = re.find_iter(&text)
                        .map(|m| Value::Str(m.as_str().to_string()))
                        .collect();
                    Ok(Value::list(matches))
                }
                Err(e) => Err(RuntimeError::at(
                    format!("regex_find_all: invalid pattern `{}`: {}", pattern, e), sp)),
            }
        }
        "regex_replace" => {
            let pattern = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_replace: pattern must be str", sp)),
            };
            let text = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_replace: text must be str", sp)),
            };
            let replacement = match args.get(2) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_replace: replacement must be str", sp)),
            };
            match regex::Regex::new(&pattern) {
                Ok(re) => Ok(Value::Str(re.replacen(&text, 1, replacement.as_str()).into_owned())),
                Err(e) => Err(RuntimeError::at(
                    format!("regex_replace: invalid pattern `{}`: {}", pattern, e), sp)),
            }
        }
        "regex_replace_all" => {
            let pattern = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_replace_all: pattern must be str", sp)),
            };
            let text = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_replace_all: text must be str", sp)),
            };
            let replacement = match args.get(2) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_replace_all: replacement must be str", sp)),
            };
            match regex::Regex::new(&pattern) {
                Ok(re) => Ok(Value::Str(re.replace_all(&text, replacement.as_str()).into_owned())),
                Err(e) => Err(RuntimeError::at(
                    format!("regex_replace_all: invalid pattern `{}`: {}", pattern, e), sp)),
            }
        }
        "regex_split" => {
            let pattern = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_split: pattern must be str", sp)),
            };
            let text = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("regex_split: text must be str", sp)),
            };
            match regex::Regex::new(&pattern) {
                Ok(re) => {
                    let parts: Vec<Value> = re.split(&text)
                        .map(|p| Value::Str(p.to_string()))
                        .collect();
                    Ok(Value::list(parts))
                }
                Err(e) => Err(RuntimeError::at(
                    format!("regex_split: invalid pattern `{}`: {}", pattern, e), sp)),
            }
        }

        // ── Compression ───────────────────────────────────────────────────────
        "gzip_compress" => {
            let data = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("gzip_compress: data must be str", sp)),
            };
            use std::io::Write;
            use flate2::write::GzEncoder;
            use flate2::Compression;
            let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
            match encoder.write_all(data.as_bytes()).and_then(|_| encoder.finish()) {
                Ok(compressed) => {
                    let hex = bytes_to_hex(&compressed);
                    Ok(Value::Tuple(vec![Value::Str(hex), Value::Nil]))
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::Str(String::new()),
                    Value::Str(format!("error: {}", e)),
                ])),
            }
        }
        "gzip_decompress" => {
            let hex_str = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("gzip_decompress: hex_str must be str", sp)),
            };
            match hex_to_bytes(&hex_str) {
                None => Ok(Value::Tuple(vec![
                    Value::Str(String::new()),
                    Value::Str("error: invalid hex string".to_string()),
                ])),
                Some(bytes) => {
                    use std::io::Read;
                    use flate2::read::GzDecoder;
                    let mut decoder = GzDecoder::new(&bytes[..]);
                    let mut decompressed = String::new();
                    match decoder.read_to_string(&mut decompressed) {
                        Ok(_) => Ok(Value::Tuple(vec![Value::Str(decompressed), Value::Nil])),
                        Err(e) => Ok(Value::Tuple(vec![
                            Value::Str(String::new()),
                            Value::Str(format!("error: {}", e)),
                        ])),
                    }
                }
            }
        }

        "zlib_compress" => {
            let data = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("zlib_compress: data must be str", sp)),
            };
            use std::io::Write;
            use flate2::write::ZlibEncoder;
            use flate2::Compression;
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
            match encoder.write_all(data.as_bytes()).and_then(|_| encoder.finish()) {
                Ok(compressed) => Ok(Value::Tuple(vec![Value::Str(bytes_to_hex(&compressed)), Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Str(String::new()), Value::Str(format!("error: {}", e))])),
            }
        }
        "zlib_decompress" => {
            let hex_str = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("zlib_decompress: hex_str must be str", sp)),
            };
            match hex_to_bytes(&hex_str) {
                None => Ok(Value::Tuple(vec![Value::Str(String::new()), Value::Str("error: invalid hex string".to_string())])),
                Some(bytes) => {
                    use std::io::Read;
                    use flate2::read::ZlibDecoder;
                    let mut decoder = ZlibDecoder::new(&bytes[..]);
                    let mut decompressed = String::new();
                    match decoder.read_to_string(&mut decompressed) {
                        Ok(_) => Ok(Value::Tuple(vec![Value::Str(decompressed), Value::Nil])),
                        Err(e) => Ok(Value::Tuple(vec![Value::Str(String::new()), Value::Str(format!("error: {}", e))])),
                    }
                }
            }
        }

        // ── HTTP networking ───────────────────────────────────────────────────
        "http_get" => {
            let url = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("http_get: url must be str", sp)),
            };
            match ureq::get(&url)
                .config()
                .timeout_global(Some(std::time::Duration::from_secs(30)))
                .build()
                .call()
            {
                Ok(resp) => {
                    match resp.into_body().read_to_string() {
                        Ok(body) => Ok(Value::Tuple(vec![Value::Str(body), Value::Nil])),
                        Err(e) => Ok(Value::Tuple(vec![
                            Value::Str(String::new()),
                            Value::Str(format!("error: {}", e)),
                        ])),
                    }
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::Str(String::new()),
                    Value::Str(format!("error: {}", e)),
                ])),
            }
        }
        "http_post" => {
            let url = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("http_post: url must be str", sp)),
            };
            let body = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("http_post: body must be str", sp)),
            };
            let content_type = match args.get(2) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("http_post: content_type must be str", sp)),
            };
            match ureq::post(&url)
                .config()
                .timeout_global(Some(std::time::Duration::from_secs(30)))
                .build()
                .header("Content-Type", &content_type)
                .send(&body)
            {
                Ok(resp) => {
                    match resp.into_body().read_to_string() {
                        Ok(body) => Ok(Value::Tuple(vec![Value::Str(body), Value::Nil])),
                        Err(e) => Ok(Value::Tuple(vec![
                            Value::Str(String::new()),
                            Value::Str(format!("error: {}", e)),
                        ])),
                    }
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::Str(String::new()),
                    Value::Str(format!("error: {}", e)),
                ])),
            }
        }
        "http_post_json" => {
            let url = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("http_post_json: url must be str", sp)),
            };
            let json_body = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("http_post_json: json_body must be str", sp)),
            };
            match ureq::post(&url)
                .config()
                .timeout_global(Some(std::time::Duration::from_secs(30)))
                .build()
                .header("Content-Type", "application/json")
                .send(&json_body)
            {
                Ok(resp) => {
                    match resp.into_body().read_to_string() {
                        Ok(body) => Ok(Value::Tuple(vec![Value::Str(body), Value::Nil])),
                        Err(e) => Ok(Value::Tuple(vec![
                            Value::Str(String::new()),
                            Value::Str(format!("error: {}", e)),
                        ])),
                    }
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::Str(String::new()),
                    Value::Str(format!("error: {}", e)),
                ])),
            }
        }

        // ── Date/time ─────────────────────────────────────────────────────────
        "date_now_ms" => {
            Ok(Value::Int(chrono::Utc::now().timestamp_millis(), IW::I64))
        }
        "date_now_s" => {
            Ok(Value::Int(chrono::Utc::now().timestamp(), IW::I64))
        }
        "date_format" => {
            let timestamp_ms = match args.first() {
                Some(Value::Int(n, _)) => *n,
                Some(Value::Float(x, _)) => *x as i64,
                _ => return Err(RuntimeError::at("date_format: timestamp_ms must be integer", sp)),
            };
            let format_str = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("date_format: format_str must be str", sp)),
            };
            match chrono::DateTime::from_timestamp_millis(timestamp_ms) {
                Some(dt) => Ok(Value::Str(dt.format(&format_str).to_string())),
                None => Ok(Value::Str(String::new())),
            }
        }
        "date_parse" => {
            let date_str = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("date_parse: date_str must be str", sp)),
            };
            let format_str = match args.get(1) {
                Some(Value::Str(s)) => s.clone(),
                _ => return Err(RuntimeError::at("date_parse: format_str must be str", sp)),
            };
            match chrono::NaiveDateTime::parse_from_str(&date_str, &format_str) {
                Ok(dt) => {
                    let ts_ms = dt.and_utc().timestamp_millis();
                    Ok(Value::Tuple(vec![Value::Int(ts_ms, IW::I64), Value::Nil]))
                }
                Err(_) => {
                    // Try parsing as a date only (NaiveDate)
                    match chrono::NaiveDate::parse_from_str(&date_str, &format_str) {
                        Ok(d) => {
                            let dt = d.and_hms_opt(0, 0, 0).unwrap();
                            let ts_ms = dt.and_utc().timestamp_millis();
                            Ok(Value::Tuple(vec![Value::Int(ts_ms, IW::I64), Value::Nil]))
                        }
                        Err(e) => Ok(Value::Tuple(vec![
                            Value::Int(0, IW::I64),
                            Value::Str(format!("parse error: {}", e)),
                        ])),
                    }
                }
            }
        }
        "date_add_ms" => {
            let timestamp_ms = match args.first() {
                Some(Value::Int(n, _)) => *n,
                Some(Value::Float(x, _)) => *x as i64,
                _ => return Err(RuntimeError::at("date_add_ms: timestamp_ms must be integer", sp)),
            };
            let delta_ms = match args.get(1) {
                Some(Value::Int(n, _)) => *n,
                Some(Value::Float(x, _)) => *x as i64,
                _ => return Err(RuntimeError::at("date_add_ms: delta_ms must be integer", sp)),
            };
            Ok(Value::Int(timestamp_ms + delta_ms, IW::I64))
        }
        "date_diff_ms" => {
            let ts_a = match args.first() {
                Some(Value::Int(n, _)) => *n,
                Some(Value::Float(x, _)) => *x as i64,
                _ => return Err(RuntimeError::at("date_diff_ms: ts_a must be integer", sp)),
            };
            let ts_b = match args.get(1) {
                Some(Value::Int(n, _)) => *n,
                Some(Value::Float(x, _)) => *x as i64,
                _ => return Err(RuntimeError::at("date_diff_ms: ts_b must be integer", sp)),
            };
            Ok(Value::Int(ts_a - ts_b, IW::I64))
        }

        // ── Trit tensor operations ────────────────────────────────────────
        "trit_quantize" => {
            // trit_quantize(x) — round a float tensor to {-1, 0, +1}.
            // x > 0.5 → 1.0, x < -0.5 → -1.0, else 0.0.
            let t = as_tensor(args.first().ok_or_else(|| RuntimeError::at("trit_quantize: needs tensor arg", sp.clone()))?)?;
            let data = t.mapv(|x| if x > 0.5 { 1.0 } else if x < -0.5 { -1.0 } else { 0.0 });
            Ok(Value::tensor_dt(data, DType::Trit))
        }
        "trit_quantize_soft" => {
            // trit_quantize_soft(x, tau) — soft ternary relaxation for training:
            //   tanh((x - 0.5) / tau) + tanh((x + 0.5) / tau)
            // Returns DType::F32 (not actual trit; training approximation only).
            let t = as_tensor(args.first().ok_or_else(|| RuntimeError::at("trit_quantize_soft: needs tensor arg", sp.clone()))?)?;
            let tau = args.get(1).and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::at("trit_quantize_soft: tau must be a float scalar", sp.clone()))?;
            if tau == 0.0 {
                return Err(RuntimeError::at("trit_quantize_soft: tau must be non-zero", sp));
            }
            let data = t.mapv(|x| ((x - 0.5) / tau).tanh() + ((x + 0.5) / tau).tanh());
            Ok(Value::tensor_dt(data, DType::F32))
        }
        "trit_neg" => {
            // trit_neg(w) — flip 1→-1, -1→1, 0→0.  Returns DType::Trit.
            let v = args.into_iter().next()
                .ok_or_else(|| RuntimeError::at("trit_neg: needs trit tensor", sp.clone()))?;
            let t = as_tensor(&v)?;
            let data = t.mapv(|x| -x);
            Ok(Value::tensor_dt(data, DType::Trit))
        }
        "trit_sparsity" => {
            // trit_sparsity(w) — fraction of 0-valued elements.  Returns f64 scalar.
            let v = args.into_iter().next()
                .ok_or_else(|| RuntimeError::at("trit_sparsity: needs trit tensor", sp.clone()))?;
            let t = as_tensor(&v)?;
            let total = t.len();
            if total == 0 { return Ok(Value::Float(0.0, FW::F64)); }
            let zeros = t.iter().filter(|&&x| x == 0.0).count();
            Ok(Value::Float(zeros as f64 / total as f64, FW::F64))
        }
        "trit_pack" => {
            // trit_pack(w) — return (pos_mask, neg_mask) as a tuple of float tensors.
            // pos_mask[i] = 1.0 iff w[i] == 1.0; neg_mask[i] = 1.0 iff w[i] == -1.0.
            // Interpreter version; Phase 2 will pack into u64 bitmaps.
            let v = args.into_iter().next()
                .ok_or_else(|| RuntimeError::at("trit_pack: needs trit tensor", sp.clone()))?;
            let t = as_tensor(&v)?;
            let pos = t.mapv(|x| if x == 1.0  { 1.0 } else { 0.0 });
            let neg = t.mapv(|x| if x == -1.0 { 1.0 } else { 0.0 });
            Ok(Value::Tuple(vec![
                Value::tensor_dt(pos, DType::F32),
                Value::tensor_dt(neg, DType::F32),
            ]))
        }

        // Unrecognised builtin name — refuse rather than silently emit Opaque.
        other => Err(RuntimeError::at(
            format!("unknown builtin `{}`; no interpreter implementation. \
            If this is a stdlib function, add a real implementation in call_builtin or remove it from the builtin set.",
            other), sp,
        )),
    };
    // #781: charge every tensor this builtin call materialized, at the ONE
    // choke point instead of a per-builtin patch — see the comment above
    // the match. Recurses into a `Tuple`/`List` result (e.g. `trit_pack`
    // returns two tensors) so a builtin that hands back more than one
    // fresh tensor is not undercharged.
    if let Ok(v) = &result {
        self.charge_builtin_result(v, &charge_span)?;
    }
    result
}

/// #781/#802: walk a `call_builtin` result and charge every `Value::Tensor`
/// and `Value::Str` found in it (including nested inside a `Tuple`/`List`).
/// Under `dmc jit`, a builtin's fresh string result (`read_file`'s content,
/// for instance) is bump-allocated in the Forge arena by
/// `forge_str_from_rust` and billed there; before #802, the interpreter never
/// charged that string against anything, so the same program needed a
/// `--forge` budget under `dmc jit` it did not need under `dmc run`.
/// Recurses into a `Tuple`/`List` result so a builtin returning more than one
/// fresh string or tensor (a `(content, err)` pair included) is not
/// undercharged.
///
/// The `Value::Tensor` arm routes through `charge_intermediate` — this IS
/// that choke point for a builtin's tensor result. The `Value::Str` arm does
/// NOT: a `call_builtin` result is never itself a literal element (a tensor
/// literal's elements are numeric or nested-tensor, never a bare builtin
/// call folded into the literal's own suppression), but it can be REACHED
/// from inside one — a tensor literal element that calls a fn which calls
/// `to_str` 1000 times, say — and `charge_intermediate`'s
/// `lit_charge_suppressed` early return exists to stop a literal's OWN rows
/// from double-billing, not to swallow an unrelated builtin's string result
/// that happens to be materialized while a literal element is being
/// evaluated (PR #815 review: routing this arm through `charge_intermediate`
/// made exactly that string vanish — 2898 B collapsed to 8 B on a tensor
/// literal whose single element called such a fn). So this arm calls
/// `charge_arena` directly, unconditionally, the same way `charge_ctor` does
/// for an explicit constructor — a builtin's fresh string is never a
/// literal's row, so it is never a candidate for that suppression at all.
fn charge_builtin_result(&mut self, v: &Value, span: &Span) -> EvalResult<()> {
    match v {
        Value::Tensor(_) => self.charge_intermediate(v, Some(span)),
        Value::Str(s) => {
            let nbytes = s.len() as u64;
            self.charge_arena(Some(crate::arena::Arena::Forge), nbytes, Some(span))
        }
        Value::Tuple(vs) => {
            for item in vs {
                self.charge_builtin_result(item, span)?;
            }
            Ok(())
        }
        Value::List(vs) => {
            for item in vs.iter() {
                self.charge_builtin_result(item, span)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
}

// ─── Stdlib builtin implementations ──────────────────────────────────────────
//
// These honour the semantic contract from STDLIB.md. They do NOT honour the
// JIT fusion contract from STDLIB.md §2 — that's the JIT's job; the
// interpreter is the reference for "what the right answer is," not "how
// fast." Each output is correct, not single-pass.

/// Normalize an axis index: negative axes count from the end.
fn normalize_axis(axis: i64, ndim: usize, sp: &Span) -> EvalResult<usize> {
    let n = ndim as i64;
    let a = if axis < 0 { axis + n } else { axis };
    if a < 0 || a >= n {
        return Err(RuntimeError::at(
            format!("axis {} out of range for rank-{} tensor", axis, ndim),
            sp.clone(),
        ));
    }
    Ok(a as usize)
}

/// Extract a required tensor arg by position with a helpful error.
/// Like `required_tensor`, but keeps the `TensorVal` wrapper so the caller can
/// see the DTYPE (#481). `required_tensor` derefs to the raw `ArrayD<f64>`,
/// which silently drops the f32/f64 tag every reduction now needs.
fn required_tensor_val<'a>(args: &'a [Value], pos: usize, fn_name: &str, sp: &Span)
    -> EvalResult<&'a TensorVal>
{
    match args.get(pos) {
        Some(Value::Tensor(t)) => Ok(t),
        Some(other) => Err(RuntimeError::at(
            format!("`{}`: arg {} must be a tensor, got {}", fn_name, pos, other.type_name()),
            sp.clone(),
        )),
        None => Err(RuntimeError::at(
            format!("`{}`: missing tensor arg at position {}", fn_name, pos),
            sp.clone(),
        )),
    }
}

fn required_tensor<'a>(args: &'a [Value], pos: usize, fn_name: &str, sp: &Span)
    -> EvalResult<&'a ArrayD<f64>>
{
    match args.get(pos) {
        Some(Value::Tensor(t)) => Ok(t),
        Some(other) => Err(RuntimeError::at(
            format!("`{}`: arg {} must be a tensor, got {}", fn_name, pos, other.type_name()),
            sp.clone(),
        )),
        None => Err(RuntimeError::at(
            format!("`{}`: missing tensor arg at position {}", fn_name, pos),
            sp.clone(),
        )),
    }
}

/// The optional trailing mask arg of `attn`/`attn_gqa`: absent or nil means
/// unmasked, a tensor is the mask, and anything else is an error — silently
/// dropping a non-tensor here would compute unmasked attention with no
/// diagnostic (e.g. the opaque stand-in the interpreter yields for
/// `vault.load_npz`).
fn optional_mask_tensor<'a>(args: &'a [Value], fn_name: &str, sp: &Span)
    -> EvalResult<Option<&'a ArrayD<f64>>>
{
    match args.get(3) {
        None | Some(Value::Nil) => Ok(None),
        Some(Value::Tensor(t)) => Ok(Some(t)),
        Some(other) => Err(RuntimeError::at(
            format!("`{}`: mask (arg 4) must be a tensor or nil, got {}",
                fn_name, other.type_name()),
            sp.clone(),
        )),
    }
}

/// Extract an optional integer arg, defaulting if missing.
fn opt_int(args: &[Value], pos: usize, default: i64) -> i64 {
    args.get(pos).and_then(|v| v.as_int()).unwrap_or(default)
}

// ─── Variance-trait + per-axis reductions ────────────────────────────────
// These mirror the JIT builtins of the same names (compiler/src/jit.rs) so
// the two backends stay at parity. A parity test asserts every JIT-dispatched
// builtin is recognized here; keep these in lockstep with the JIT.

/// `variance(t)` — population variance (1/N) over all elements.
fn builtin_variance(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor_val(args, 0, "variance", &sp)?;
    let n = t.len() as f64;
    if n == 0.0 { return Ok(Value::Float(0.0, t.float_width())); }
    // #481: two passes, both at the tensor's width — mean, then the mean of the
    // squared deviations. Same shape as the JIT's `lower_builtin_variance`.
    let mean = tensor_mean(t);
    let devs = t.iter().map(|v| { let d = v - mean; d * d });
    let var = if t.dtype.rounds_as_f32() {
        quantize_f32(sum_f32(devs.map(quantize_f32)) / quantize_f32(n))
    } else {
        devs.sum::<f64>() / n
    };
    Ok(Value::Float(var, t.float_width()))
}

/// `pull_to_mean(t, alpha)` — one variance-minimizing pass over all elements:
/// out[i] = t[i] + alpha * (mean(t) - t[i]).
fn builtin_pull_to_mean(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor_val(args, 0, "pull_to_mean", &sp)?;
    let alpha = args.get(1).and_then(|v| v.as_float()).ok_or_else(|| RuntimeError::at(
        "pull_to_mean: missing alpha arg".to_string(), sp.clone()))?;
    // #481: the mean this pulls toward is computed at the tensor's width.
    let mean = if t.len() > 0 { tensor_mean(t) } else { 0.0 };
    Ok(Value::tensor(t.mapv(|v| v + alpha * (mean - v))))
}

/// Resolve a per-axis reduction's axis arg (literal int) and validate it.
fn reduce_axis(args: &[Value], rank: usize, name: &str, sp: &Span) -> EvalResult<usize> {
    let axis = args.get(1).and_then(|v| v.as_int()).ok_or_else(|| RuntimeError::at(
        format!("{}: missing axis arg", name), sp.clone()))?;
    let axis = if axis < 0 { axis + rank as i64 } else { axis };
    if axis < 0 || axis as usize >= rank {
        return Err(RuntimeError::at(
            format!("{}: axis {} out of range for rank-{} tensor", name, axis, rank), sp.clone()));
    }
    Ok(axis as usize)
}

/// Reduce each lane along `axis` with `f`, preserving `ndarray`'s output shape.
///
/// `sum_axis` accumulates in f64, which is the #481 defect one axis down: an f32
/// tensor must round every add inside a lane, exactly as the whole-tensor
/// `sum` does. `map_axis` hands us each lane so the fold is ours to choose.
#[inline]
fn fold_lanes(t: &TensorVal, axis: usize, f32_width: bool) -> ArrayD<f64> {
    t.map_axis(Axis(axis), |lane| {
        if f32_width { sum_f32(lane.iter().copied()) } else { lane.iter().sum() }
    }).into_dyn()
}

/// `sum_along(t, axis)` / `mean_along(t, axis)`.
fn builtin_reduce_along(args: &[Value], sp: Span, name: &str, mean: bool) -> EvalResult<Value> {
    let t = required_tensor_val(args, 0, name, &sp)?;
    let axis = reduce_axis(args, t.ndim(), name, &sp)?;
    // #481: accumulate per lane at the source tensor's width, and keep that
    // dtype on the output so a later reduction over the result reduces at the
    // same width too. `Value::tensor` would have retagged it F32 regardless,
    // but only AFTER an f64 accumulation — rounding the answer, not producing
    // the f32 one.
    let f32_width = t.dtype.rounds_as_f32();
    let summed = fold_lanes(&t, axis, f32_width);
    let out = if mean {
        let n = t.shape()[axis] as f64;
        if n <= 0.0 {
            summed.mapv(|_| 0.0)
        } else if f32_width {
            let nq = quantize_f32(n);
            summed.mapv(|v| quantize_f32(v / nq))
        } else {
            summed.mapv(|v| v / n)
        }
    } else {
        summed
    };
    Ok(Value::tensor_dt(out, t.dtype))
}

/// `max_along(t, axis)` — maximum reduced along one axis (axis dropped).
fn builtin_max_along(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor_val(args, 0, "max_along", &sp)?;
    let axis = reduce_axis(args, t.ndim(), "max_along", &sp)?;
    if t.shape()[axis] == 0 {
        return Err(RuntimeError::at("max_along: empty axis".to_string(), sp));
    }
    // #481: selection, so nothing to round — but keep the source dtype so the
    // result reduces at the right width downstream.
    let out = t.fold_axis(Axis(axis), f64::NEG_INFINITY, |&acc, &v| acc.max(v));
    Ok(Value::tensor_dt(out.into_dyn(), t.dtype))
}

/// `min_along(t, axis)` — minimum reduced along one axis (axis dropped).
fn builtin_min_along(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor_val(args, 0, "min_along", &sp)?;
    let axis = reduce_axis(args, t.ndim(), "min_along", &sp)?;
    if t.shape()[axis] == 0 {
        return Err(RuntimeError::at("min_along: empty axis".to_string(), sp));
    }
    // #481: selection, so nothing to round — but keep the source dtype so the
    // result reduces at the right width downstream.
    let out = t.fold_axis(Axis(axis), f64::INFINITY, |&acc, &v| acc.min(v));
    Ok(Value::tensor_dt(out.into_dyn(), t.dtype))
}

/// `variance_along(t, axis)` — population variance reduced along one axis.
fn builtin_variance_along(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor_val(args, 0, "variance_along", &sp)?;
    let axis = reduce_axis(args, t.ndim(), "variance_along", &sp)?;
    let n = t.shape()[axis] as f64;
    if n <= 0.0 {
        return Err(RuntimeError::at("variance_along: empty axis".to_string(), sp));
    }
    // #481: TWO passes — mean, then the mean of the squared deviations —
    // matching the JIT's `lower_builtin_variance_along`. The old form was the
    // algebraically equivalent `mean(x²) - mean(x)²`, which is a different
    // computation in floating point (catastrophic cancellation when the mean
    // dominates the spread) and diverged from the JIT independently of width.
    let f32_width = t.dtype.rounds_as_f32();
    let nq = if f32_width { quantize_f32(n) } else { n };
    let means = {
        let summed = fold_lanes(&t, axis, f32_width);
        if f32_width { summed.mapv(|v| quantize_f32(v / nq)) } else { summed.mapv(|v| v / nq) }
    };
    // Broadcast the per-lane mean back over the axis, then reduce the squared
    // deviations the same way.
    let devs = {
        let mut d = t.data.clone();
        for (mut lane, m) in d.lanes_mut(Axis(axis)).into_iter().zip(means.iter()) {
            for slot in lane.iter_mut() {
                let dev = *slot - *m;
                *slot = if f32_width { quantize_f32(dev * dev) } else { dev * dev };
            }
        }
        TensorVal::new(d.into_dyn(), t.dtype)
    };
    let summed = fold_lanes(&devs, axis, f32_width);
    let var = if f32_width { summed.mapv(|v| quantize_f32(v / nq)) } else { summed.mapv(|v| v / nq) };
    Ok(Value::tensor_dt(var, t.dtype))
}

/// `pull_to_mean_along(t, axis, alpha)` — per-axis variance-minimizing pass.
fn builtin_pull_to_mean_along(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor(args, 0, "pull_to_mean_along", &sp)?;
    let axis = reduce_axis(args, t.ndim(), "pull_to_mean_along", &sp)?;
    let alpha = args.get(2).and_then(|v| v.as_float()).ok_or_else(|| RuntimeError::at(
        "pull_to_mean_along: missing alpha arg".to_string(), sp.clone()))?;
    let mean = t.mean_axis(Axis(axis)).ok_or_else(|| RuntimeError::at(
        "pull_to_mean_along: empty axis".to_string(), sp.clone()))?;
    let mut out = t.clone();
    for (idx_owned, slot) in out.indexed_iter_mut() {
        let idx: Vec<usize> = idx_owned.as_array_view().iter().copied().collect();
        let mut mi = idx.clone();
        mi.remove(axis);
        let m = mean[IxDyn(&mi)];
        *slot += alpha * (m - *slot);
    }
    Ok(Value::tensor(out))
}

/// Elementwise activation, applied to an f32 scalar or an f32 tensor.
/// The per-element function matches the JIT bit-for-bit in intent:
///   relu(x)    = max(0, x)
///   sigmoid(x) = 1 / (1 + exp(-x))
///   tanh(x)    = tanh(x)
///   gelu(x)    = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))   (tanh approx)
///   silu(x)    = x * sigmoid(x) = x / (1 + exp(-x))
/// The smooth activation builtins that trace through `@grad` (#306). `relu`
/// stays a `TapeOp::ReLU`; these get `TapeOp::Activation(kind)`.
#[derive(Clone, Copy, PartialEq, Debug)]
enum ActKind { Sigmoid, Tanh, Gelu, Silu, Elu, Mish }

impl ActKind {
    fn from_name(name: &str) -> Option<ActKind> {
        match name {
            "sigmoid" => Some(ActKind::Sigmoid),
            "tanh"    => Some(ActKind::Tanh),
            "gelu"    => Some(ActKind::Gelu),
            "silu"    => Some(ActKind::Silu),
            "elu"     => Some(ActKind::Elu),
            "mish"    => Some(ActKind::Mish),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            ActKind::Sigmoid => "sigmoid",
            ActKind::Tanh    => "tanh",
            ActKind::Gelu    => "gelu",
            ActKind::Silu    => "silu",
            ActKind::Elu     => "elu",
            ActKind::Mish    => "mish",
        }
    }
}

/// Numerically-stable softplus `ln(1 + e^x)` = `max(x,0) + ln(1 + e^-|x|)`.
/// Shared by mish's forward and derivatives so run/jit agree (mirrors the
/// `__dmc_softplus`-style form used in the JIT extern).
fn softplus_f64(x: f64) -> f64 {
    x.max(0.0) + (-(x.abs())).exp().ln_1p()
}

/// Derivative of an activation w.r.t. its input `x`, matching `activation_f64`
/// (note: gelu is the tanh approximation, so its derivative is too). Used by the
/// reverse-mode VJP for activation builtins.
fn activation_deriv_f64(name: &str, x: f64) -> f64 {
    match name {
        "relu"    => if x > 0.0 { 1.0 } else { 0.0 },
        "tanh"    => { let t = x.tanh(); 1.0 - t * t }
        "sigmoid" => { let s = 1.0 / (1.0 + (-x).exp()); s * (1.0 - s) }
        "silu"    => { let s = 1.0 / (1.0 + (-x).exp()); s * (1.0 + x * (1.0 - s)) }
        // elu (α=1): d/dx = 1 for x>0, e^x for x≤0.
        "elu"     => if x > 0.0 { 1.0 } else { x.exp() }
        // mish = x·tanh(sp), sp=softplus(x), sp'=σ(x); mish' = t + x(1-t²)σ.
        "mish"    => {
            let t = softplus_f64(x).tanh();
            let s = 1.0 / (1.0 + (-x).exp());
            t + x * (1.0 - t * t) * s
        }
        "gelu"    => {
            let c = (2.0f64 / std::f64::consts::PI).sqrt();
            let u = c * (x + 0.044715 * x * x * x);
            let t = u.tanh();
            let du = c * (1.0 + 3.0 * 0.044715 * x * x);
            0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du
        }
        _ => unreachable!("activation_deriv_f64: unknown activation {name}"),
    }
}

/// Second derivative of an activation w.r.t. its input `x`, matching
/// `activation_deriv_f64` (gelu is the tanh approximation, so `act''` is the
/// derivative of the tanh-approx `act'`). Powers the second-order VJP
/// (`@grad @grad`) for activation builtins (#306). `relu''` is 0 a.e. and never
/// reaches here — `backward_symbolic` keeps ReLU's const-mask path.
fn activation_second_deriv_f64(name: &str, x: f64) -> f64 {
    match name {
        // tanh'  = 1 - t²        → tanh''  = -2 t (1 - t²)
        "tanh"    => { let t = x.tanh(); -2.0 * t * (1.0 - t * t) }
        // sigmoid' = s(1-s)      → sigmoid'' = s(1-s)(1-2s)
        "sigmoid" => { let s = 1.0 / (1.0 + (-x).exp()); s * (1.0 - s) * (1.0 - 2.0 * s) }
        // silu = x·s; silu' = s + x·s' → silu'' = 2 s' + x s'' = s(1-s)[2 + x(1-2s)]
        "silu"    => {
            let s = 1.0 / (1.0 + (-x).exp());
            s * (1.0 - s) * (2.0 + x * (1.0 - 2.0 * s))
        }
        // elu (α=1): elu'' = 0 for x>0, e^x for x≤0 (elu' is 1 vs e^x).
        "elu"     => if x > 0.0 { 0.0 } else { x.exp() }
        // mish'' = (1-t²)σ·[ 2 + x((1-σ) - 2tσ) ], t=tanh(softplus(x)), σ=sigmoid(x).
        "mish"    => {
            let t = softplus_f64(x).tanh();
            let s = 1.0 / (1.0 + (-x).exp());
            (1.0 - t * t) * s * (2.0 + x * ((1.0 - s) - 2.0 * t * s))
        }
        // gelu (tanh approx), u = c(x + a x³), t = tanh(u), u' = c(1 + 3a x²):
        //   gelu'  = 0.5(1+t) + 0.5 x (1-t²) u'
        //   gelu'' = (1-t²)[ u' - x t u'² + 3 a c x² ]
        "gelu"    => {
            let c = (2.0f64 / std::f64::consts::PI).sqrt();
            let a = 0.044715;
            let u = c * (x + a * x * x * x);
            let t = u.tanh();
            let up = c * (1.0 + 3.0 * a * x * x);
            (1.0 - t * t) * (up - x * t * up * up + 3.0 * a * c * x * x)
        }
        _ => unreachable!("activation_second_deriv_f64: unknown activation {name}"),
    }
}

fn activation_f64(name: &str, x: f64) -> f64 {
    match name {
        "relu"    => if x > 0.0 { x } else { 0.0 },
        "tanh"    => x.tanh(),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "silu"    => x / (1.0 + (-x).exp()),
        // elu (α=1): x for x>0, e^x - 1 for x≤0.
        "elu"     => if x > 0.0 { x } else { x.exp() - 1.0 },
        // mish(x) = x · tanh(softplus(x)).
        "mish"    => x * softplus_f64(x).tanh(),
        "gelu"    => {
            // sqrt(2/pi) as the f32 the JIT uses, promoted to f64 for the math.
            let c = (2.0f64 / std::f64::consts::PI).sqrt();
            0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        }
        _ => unreachable!("activation_f64: unknown activation {name}"),
    }
}

fn builtin_activation(name: &str, args: &[Value], sp: Span) -> EvalResult<Value> {
    match args.first() {
        Some(Value::Tensor(t)) => {
            let data = t.data.mapv(|x| activation_f64(name, x));
            // Same shape, f32 tensor (Value::tensor re-quantizes per element).
            Ok(Value::tensor(data))
        }
        Some(v) => {
            let x = v.as_float().ok_or_else(|| RuntimeError::at(
                format!("{name}: requires an f32 scalar or f32 tensor"), sp.clone()))?;
            // f32-round the scalar result so run/jit agree at f32 precision.
            // #567: `\<` rounds here too, through `gelu` — the two spellings of
            // GeLU must not differ by the rounding alone.
            Ok(Value::Float(quantize_f32(activation_f64(name, x)), FW::F64))
        }
        None => Err(RuntimeError::at(format!("{name}: needs 1 argument"), sp)),
    }
}

fn builtin_softmax(args: &[Value], sp: Span) -> EvalResult<Value> {
    let x = required_tensor(args, 0, "softmax", &sp)?;
    let axis = normalize_axis(opt_int(args, 1, -1), x.ndim(), &sp)?;
    let axis_obj = Axis(axis);
    // y[i] = exp(x[i] - max) / sum_along_axis(exp(x - max))
    // Compute per-slice along `axis`.
    let max_along = x.fold_axis(axis_obj, f64::NEG_INFINITY, |a, b| a.max(*b));
    // Broadcast max back: shape with axis=1.
    let mut shape_with_1 = x.shape().to_vec();
    shape_with_1[axis] = 1;
    let max_b = max_along.into_shape_with_order(IxDyn(&shape_with_1))
        .map_err(|e| RuntimeError::at(format!("softmax max reshape: {}", e), sp.clone()))?;
    let mut shifted = x.clone();
    // Per-element weight via numerically-stable, ±inf-aware rule (#258):
    //   max == -inf  → 0           (fully-masked row; spec §3.2)
    //   x_i == max   → 1 (= exp 0) (the argmax, incl. +inf — avoids +inf-+inf=NaN)
    //   else         → exp(x_i-max) (below max, incl. -inf → 0)
    // For finite in-range inputs this is exactly standard stable softmax.
    for (idx, slot) in shifted.indexed_iter_mut() {
        let mut max_idx: Vec<usize> = idx.as_array_view().iter().copied().collect();
        max_idx[axis] = 0;
        let m = max_b[IxDyn(&max_idx)];
        let xi = *slot;
        *slot = if m == f64::NEG_INFINITY { 0.0 }
                else if xi == m { 1.0 }
                else { (xi - m).exp() };
    }
    let sum_along = shifted.sum_axis(axis_obj);
    let sum_b = sum_along.into_shape_with_order(IxDyn(&shape_with_1))
        .map_err(|e| RuntimeError::at(format!("softmax sum reshape: {}", e), sp.clone()))?;
    for (idx, slot) in shifted.indexed_iter_mut() {
        let mut sum_idx: Vec<usize> = idx.as_array_view().iter().copied().collect();
        sum_idx[axis] = 0;
        let denom = sum_b[IxDyn(&sum_idx)];
        // denom == 0 when entire row was masked; output 0 per spec §3.2.
        *slot = if denom == 0.0 { 0.0 } else { *slot / denom };
    }
    Ok(Value::tensor(shifted))
}

// embed(vocab, ids) — canonical 2-arg form per STDLIB.md §3.5 (issue #113/2.3).
// vocab: Tensor[f32, V, D]  ids: Tensor[i64, B...]
// Returns Tensor[f32, B..., D].
fn builtin_embed(args: &[Value], sp: Span) -> EvalResult<Value> {
    let vocab = required_tensor(args, 0, "embed", &sp)?;
    if vocab.ndim() != 2 {
        return Err(RuntimeError::at(
            format!("embed: first arg (vocab) must be 2-D [V, D], got {} dims", vocab.ndim()), sp));
    }
    let v_size = vocab.shape()[0];
    let d_size = vocab.shape()[1];
    let ids = required_tensor(args, 1, "embed", &sp)?;
    let n = ids.len();
    let mut flat_out = Vec::with_capacity(n * d_size);
    for &raw_id in ids.iter() {
        let idx = (raw_id as i64).clamp(0, v_size as i64 - 1) as usize;
        for d in 0..d_size {
            flat_out.push(vocab[IxDyn(&[idx, d])]);
        }
    }
    let mut out_shape: Vec<usize> = ids.shape().iter().copied().collect();
    out_shape.push(d_size);
    let out = ArrayD::from_shape_vec(IxDyn(&out_shape), flat_out)
        .map_err(|e| RuntimeError::at(format!("embed shape error: {}", e), sp))?;
    Ok(Value::tensor(out))
}

fn builtin_arg_reduce(args: &[Value], sp: Span, want_max: bool) -> EvalResult<Value> {
    let fn_name = if want_max { "argmax" } else { "argmin" };
    let x = required_tensor(args, 0, fn_name, &sp)?;
    let axis = normalize_axis(opt_int(args, 1, -1), x.ndim(), &sp)?;
    // Output shape: input shape with `axis` dropped.
    let out_shape: Vec<usize> = x.shape().iter().enumerate()
        .filter_map(|(i, &d)| if i == axis { None } else { Some(d) }).collect();
    // #272: when the reduction collapses to a scalar (a rank-1 input, or any
    // input reduced on its only remaining axis), return a scalar `Int` — a usable
    // index/token, and at parity with the JIT (which yields i64). A rank-0 tensor
    // can't even be used as an index. Multi-axis results stay a tensor.
    if out_shape.is_empty() {
        let axis_len = x.shape()[axis];
        let mut best_i: usize = 0;
        let mut best_v: f64 = if want_max { f64::NEG_INFINITY } else { f64::INFINITY };
        for i in 0..axis_len {
            let v = x[IxDyn(&[i])];
            let pick = if want_max { v > best_v } else { v < best_v };
            if pick { best_v = v; best_i = i; }
        }
        return Ok(Value::Int(best_i as i64, IW::I64));
    }
    let mut out = ArrayD::<f64>::zeros(IxDyn(&out_shape));
    // For each output index, scan along `axis` and pick argmax/argmin.
    for (out_idx_owned, slot) in out.indexed_iter_mut() {
        let out_idx: Vec<usize> = out_idx_owned.as_array_view().iter().copied().collect();
        let axis_len = x.shape()[axis];
        let mut best_i: usize = 0;
        let mut best_v: f64 = if want_max { f64::NEG_INFINITY } else { f64::INFINITY };
        for i in 0..axis_len {
            let mut full = out_idx.clone();
            full.insert(axis, i);
            let v = x[IxDyn(&full)];
            let pick = if want_max { v > best_v } else { v < best_v };
            if pick { best_v = v; best_i = i; }
        }
        *slot = best_i as f64;
    }
    Ok(Value::tensor(out))
}

fn builtin_rms_norm(args: &[Value], sp: Span) -> EvalResult<Value> {
    // rms_norm(x, g, eps) — per STDLIB.md §3.3: x * g / sqrt(mean(x^2) + eps)
    // along the last axis.
    let x = required_tensor(args, 0, "rms_norm", &sp)?;
    let g = required_tensor(args, 1, "rms_norm", &sp)?;
    let eps = args.get(2).and_then(|v| v.as_float()).unwrap_or(1e-6);
    let last = x.ndim() - 1;
    let axis_obj = Axis(last);
    if g.ndim() != 1 || g.shape()[0] != x.shape()[last] {
        return Err(RuntimeError::at(format!(
            "rms_norm: gain must be rank-1 of length {}, got shape {:?}",
            x.shape()[last], g.shape()), sp));
    }
    // mean(x^2) along last axis.
    let sq = x.mapv(|v| v * v);
    let mean_sq = sq.mean_axis(axis_obj).ok_or_else(|| RuntimeError::at(
        "rms_norm: empty axis", sp.clone()))?;
    let mut shape_with_1 = x.shape().to_vec();
    shape_with_1[last] = 1;
    let mean_b = mean_sq.into_shape_with_order(IxDyn(&shape_with_1))
        .map_err(|e| RuntimeError::at(format!("rms_norm reshape: {}", e), sp.clone()))?;
    let mut out = x.clone();
    for (idx_owned, slot) in out.indexed_iter_mut() {
        let idx: Vec<usize> = idx_owned.as_array_view().iter().copied().collect();
        let mut mean_idx = idx.clone();
        mean_idx[last] = 0;
        let denom = (mean_b[IxDyn(&mean_idx)] + eps).sqrt();
        let gain = g[IxDyn(&[idx[last]])];
        *slot = (*slot / denom) * gain;
    }
    Ok(Value::tensor(out))
}

fn builtin_layer_norm(args: &[Value], sp: Span) -> EvalResult<Value> {
    // layer_norm(x, g, b, eps) — mean+variance normalize on last axis, affine.
    let x = required_tensor(args, 0, "layer_norm", &sp)?;
    let g = required_tensor(args, 1, "layer_norm", &sp)?;
    let b = required_tensor(args, 2, "layer_norm", &sp)?;
    let eps = args.get(3).and_then(|v| v.as_float()).unwrap_or(1e-5);
    let last = x.ndim() - 1;
    let axis_obj = Axis(last);
    let d = x.shape()[last];
    if g.ndim() != 1 || g.shape()[0] != d || b.ndim() != 1 || b.shape()[0] != d {
        return Err(RuntimeError::at(format!(
            "layer_norm: gain and bias must both be rank-1 of length {}", d), sp));
    }
    let mean = x.mean_axis(axis_obj).ok_or_else(|| RuntimeError::at(
        "layer_norm: empty axis", sp.clone()))?;
    let mut shape_with_1 = x.shape().to_vec();
    shape_with_1[last] = 1;
    let mean_b = mean.into_shape_with_order(IxDyn(&shape_with_1))
        .map_err(|e| RuntimeError::at(format!("layer_norm reshape: {}", e), sp.clone()))?;
    // Compute variance manually (mapv + mean_axis avoids a second reshape juggle).
    let mut diff = x.clone();
    for (idx_owned, slot) in diff.indexed_iter_mut() {
        let idx: Vec<usize> = idx_owned.as_array_view().iter().copied().collect();
        let mut mi = idx.clone();
        mi[last] = 0;
        *slot -= mean_b[IxDyn(&mi)];
    }
    let var = diff.mapv(|v| v * v).mean_axis(axis_obj).ok_or_else(||
        RuntimeError::at("layer_norm: empty axis", sp.clone()))?
        .into_shape_with_order(IxDyn(&shape_with_1))
        .map_err(|e| RuntimeError::at(format!("layer_norm var reshape: {}", e), sp.clone()))?;
    let mut out = diff;
    for (idx_owned, slot) in out.indexed_iter_mut() {
        let idx: Vec<usize> = idx_owned.as_array_view().iter().copied().collect();
        let mut vi = idx.clone();
        vi[last] = 0;
        let denom = (var[IxDyn(&vi)] + eps).sqrt();
        let gain = g[IxDyn(&[idx[last]])];
        let bias = b[IxDyn(&[idx[last]])];
        *slot = (*slot / denom) * gain + bias;
    }
    Ok(Value::tensor(out))
}

fn builtin_rope(args: &[Value], sp: Span) -> EvalResult<Value> {
    // rope(x, cos, sin) — per STDLIB.md §3.5: paired-coordinate rotation on
    // the last axis. D must be even.
    //   new[..., 2i  ] = x[..., 2i] * cos[s, i] - x[..., 2i+1] * sin[s, i]
    //   new[..., 2i+1] = x[..., 2i] * sin[s, i] + x[..., 2i+1] * cos[s, i]
    // x has shape [..., S, D]; cos/sin have shape [S, D/2].
    let x   = required_tensor(args, 0, "rope", &sp)?;
    let cos = required_tensor(args, 1, "rope", &sp)?;
    let sin = required_tensor(args, 2, "rope", &sp)?;
    if x.ndim() < 2 {
        return Err(RuntimeError::at("rope: x must be rank >= 2 ([..., S, D])", sp));
    }
    let d_axis = x.ndim() - 1;
    let s_axis = x.ndim() - 2;
    let d = x.shape()[d_axis];
    let s = x.shape()[s_axis];
    if d % 2 != 0 {
        return Err(RuntimeError::at(format!("rope: D must be even, got {}", d), sp));
    }
    if cos.ndim() != 2 || cos.shape() != [s, d / 2]
        || sin.ndim() != 2 || sin.shape() != [s, d / 2]
    {
        return Err(RuntimeError::at(format!(
            "rope: cos and sin must both be [S={}, D/2={}], got cos {:?} sin {:?}",
            s, d / 2, cos.shape(), sin.shape()), sp));
    }
    let mut out = x.clone();
    for (idx_owned, _) in x.indexed_iter() {
        let idx: Vec<usize> = idx_owned.as_array_view().iter().copied().collect();
        let dim_i = idx[d_axis];
        if dim_i % 2 != 0 { continue; }   // we handle pairs by even index
        let pair_i = dim_i / 2;
        let mut idx_b = idx.clone();
        idx_b[d_axis] = dim_i + 1;
        let xa = x[IxDyn(&idx)];
        let xb = x[IxDyn(&idx_b)];
        let c = cos[IxDyn(&[idx[s_axis], pair_i])];
        let si = sin[IxDyn(&[idx[s_axis], pair_i])];
        out[IxDyn(&idx)]   = xa * c - xb * si;
        out[IxDyn(&idx_b)] = xa * si + xb * c;
    }
    Ok(Value::tensor(out))
}

fn builtin_attn(args: &[Value], sp: Span) -> EvalResult<Value> {
    // attn(q, k, v, mask?) — per STDLIB.md §3.1: softmax((q @ k') / sqrt(D)) @ v.
    // q, k, v all shape [B, H, S, D]; mask optional bool-mask [S, S].
    // The interpreter materializes the [B, H, S, S] score matrix — that
    // violates the FUSION contract but not the SEMANTIC contract. The JIT
    // is responsible for fusion; the interpreter is the reference output.
    let q = required_tensor(args, 0, "attn", &sp)?;
    let k = required_tensor(args, 1, "attn", &sp)?;
    let v = required_tensor(args, 2, "attn", &sp)?;
    if q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 {
        return Err(RuntimeError::at(format!(
            "attn: q/k/v must all be rank-4 [B,H,S,D]; got {:?} / {:?} / {:?}",
            q.shape(), k.shape(), v.shape()), sp));
    }
    let d = q.shape()[3] as f64;
    // k' transposes the last two axes.
    let mut kt_axes: Vec<usize> = (0..k.ndim()).collect();
    let n = kt_axes.len();
    kt_axes.swap(n - 1, n - 2);
    let kt = k.view().permuted_axes(IxDyn(&kt_axes)).as_standard_layout().to_owned();
    // q @ kt  → [B, H, S, S]
    let scores = match tensor_matmul(&Value::tensor(q.clone()), &Value::tensor(kt))? {
        Value::Tensor(t) => t,
        _ => unreachable!(),
    };
    // scale + optional mask
    let mask = optional_mask_tensor(args, "attn", &sp)?;
    let scale = 1.0 / d.sqrt();
    let mut scaled = scores.mapv(|x| x * scale);
    if let Some(mask) = mask {
        // mask is bool tensor [S, S]; we broadcast by adding -inf where mask
        // is 0/false. mask elements are 0.0 or 1.0 in our f64-tensor world.
        let s = scaled.shape()[scaled.ndim() - 1];
        if mask.ndim() != 2 || mask.shape() != [s, s] {
            return Err(RuntimeError::at(format!(
                "attn mask: expected [S,S]=[{},{}], got {:?}", s, s, mask.shape()), sp));
        }
        for (idx_owned, slot) in scaled.indexed_iter_mut() {
            let idx: Vec<usize> = idx_owned.as_array_view().iter().copied().collect();
            let i = idx[idx.len() - 2];
            let j = idx[idx.len() - 1];
            if mask[IxDyn(&[i, j])] == 0.0 { *slot = f64::NEG_INFINITY; }
        }
    }
    let weights = builtin_softmax(&[Value::tensor(scaled), Value::Int(-1, IW::I64)], sp.clone())?;
    let weights_t = match weights { Value::Tensor(t) => t, _ => unreachable!() };
    // weights @ v
    tensor_matmul(&Value::Tensor(weights_t), &Value::tensor(v.clone()))
}

fn builtin_attn_gqa(args: &[Value], sp: Span) -> EvalResult<Value> {
    // attn_gqa(q, k, v, mask?) — GQA: q [B,H_q,S,D], k/v [B,H_kv,S,D], H_q%H_kv==0
    let q = required_tensor(args, 0, "attn_gqa", &sp)?;
    let k = required_tensor(args, 1, "attn_gqa", &sp)?;
    let v = required_tensor(args, 2, "attn_gqa", &sp)?;
    if q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4 {
        return Err(RuntimeError::at(format!(
            "attn_gqa: q/k/v must be rank-4; got {:?}/{:?}/{:?}",
            q.shape(), k.shape(), v.shape()), sp));
    }
    let b   = q.shape()[0];
    let h_q = q.shape()[1];
    let s   = q.shape()[2];
    let d   = q.shape()[3];
    let h_kv = k.shape()[1];
    if k.shape() != &[b, h_kv, s, d] || v.shape() != &[b, h_kv, s, d] {
        return Err(RuntimeError::at("attn_gqa: k/v must be [B,H_kv,S,D]", sp));
    }
    if h_q % h_kv != 0 {
        return Err(RuntimeError::at(format!(
            "attn_gqa: H_q ({}) must be divisible by H_kv ({})", h_q, h_kv), sp));
    }
    let g = h_q / h_kv;
    let d_f = d as f64;
    let scale = 1.0 / d_f.sqrt();

    // Optional causal mask [S, S] (bool, 0.0/1.0): masked positions (j > i for
    // a standard causal mask) are forced to -inf before softmax. Validated once
    // here; applied per head below. Matches `builtin_attn` and the reference.
    let mask = optional_mask_tensor(args, "attn_gqa", &sp)?;
    if let Some(mask) = mask {
        if mask.ndim() != 2 || mask.shape() != [s, s] {
            return Err(RuntimeError::at(format!(
                "attn_gqa mask: expected [S,S]=[{},{}], got {:?}", s, s, mask.shape()), sp));
        }
    }

    // Build output [B, H_q, S, D] by attending each Q head against its KV head.
    let out_shape = IxDyn(&[b, h_q, s, d]);
    let mut out = ndarray::ArrayD::<f64>::zeros(out_shape);

    for bi in 0..b {
        for hqi in 0..h_q {
            let hkvi = hqi / g;
            // q_head: [S, D]
            let q_head = q.slice(ndarray::s![bi, hqi, .., ..]).to_owned();
            // k_head: [S, D] -> k_head_t: [D, S]
            let k_head = k.slice(ndarray::s![bi, hkvi, .., ..]).to_owned();
            let k_head_t = k_head.view().permuted_axes([1usize, 0usize])
                .as_standard_layout().to_owned();
            let v_head = v.slice(ndarray::s![bi, hkvi, .., ..]).to_owned();

            // scores = (q_head @ k_head_t) * scale  [S, S]
            let scores_raw = match tensor_matmul(
                &Value::tensor(q_head.into_dyn()),
                &Value::tensor(k_head_t.into_dyn()),
            )? {
                Value::Tensor(t) => t,
                _ => unreachable!(),
            };
            let mut scaled = scores_raw.mapv(|x| x * scale);

            // Apply the optional causal mask: scaled[i,j] = -inf where mask==0.
            if let Some(mask) = mask {
                for i in 0..s {
                    for j in 0..s {
                        if mask[IxDyn(&[i, j])] == 0.0 {
                            scaled[IxDyn(&[i, j])] = f64::NEG_INFINITY;
                        }
                    }
                }
            }

            // softmax along last axis (rows of S)
            let weights_v = builtin_softmax(
                &[Value::tensor(scaled), Value::Int(-1, IW::I64)], sp.clone())?;
            let weights = match weights_v { Value::Tensor(t) => t, _ => unreachable!() };

            // head_out = weights @ v_head  [S, D]
            let head_out = match tensor_matmul(
                &Value::Tensor(weights),
                &Value::tensor(v_head.into_dyn()),
            )? {
                Value::Tensor(t) => t,
                _ => unreachable!(),
            };

            out.slice_mut(ndarray::s![bi, hqi, .., ..]).assign(&head_out);
        }
    }
    Ok(Value::tensor(out))
}

// ─── Reverse-mode autodiff tape ──────────────────────────────────────────────

/// One node in the @grad tape: an op, its input node ids (in the order the
/// op expects), and the forward value the op produced. Backward uses both
/// the value and the input node values to compute VJPs.
#[derive(Debug, Clone)]
struct TapeNode {
    op: TapeOp,
    inputs: Vec<usize>,
    value: Value,
}

#[derive(Debug, Clone)]
enum TapeOp {
    /// A function parameter. `mutating` says whether we need to return its
    /// gradient at the end. `param_idx` is preserved for debugging.
    Input {
        #[allow(dead_code)] param_idx: usize,
        #[allow(dead_code)] mutating: bool,
    },
    /// A captured `mut` binding read by the differentiated body (AUTODIFF.md
    /// §2, #398). Differentiated exactly like a `!` parameter: it seeds the
    /// tape with the binding's value at call time and its adjoint comes back in
    /// `Grads` under `name` — which is also the binding's name, so
    /// `g.<name>` reads it.
    Captured {
        #[allow(dead_code)] name: String,
    },
    /// A constant value that didn't come from the gradient graph — its
    /// gradient is discarded.
    Const,
    /// Tensor ops with established VJPs.
    Matmul,
    DotAdd, DotSub, DotMul, DotDiv,
    Sum,
    ReLU,
    /// Elementwise activation builtin (sigmoid/tanh/gelu/silu). `\>`/`\<` ReLU
    /// stay as `ReLU`; this covers the smooth activation *builtins* (#306).
    Activation(ActKind),
    /// The first derivative `act'(x)` of a smooth activation, as a node over the
    /// activation's input. Only emitted by `backward_symbolic` (the first-order
    /// VJP `dx = g .* act'(x)`); its own VJP is `g .* act''(x)`, which is what
    /// lets second-order gradients (`@grad @grad`) flow through the activation
    /// (#306). The forward recorder never emits it.
    ActivationGrad(ActKind),
    /// `softmax(x, axis)` — the (normalized) reduction axis is carried so the
    /// VJP `dx = y .* (g - rowsum(g .* y))` knows which axis to reduce (#307).
    Softmax(usize),
    /// `rms_norm(x, gain, eps)` over the last axis. inputs = [x, gain]; eps is
    /// carried for the normalization-Jacobian VJP (#307).
    RmsNorm(f64),
    /// `layer_norm(x, gain, bias, eps)` over the last axis. inputs = [x, gain,
    /// bias]; eps carried for the mean+var normalization VJP (#307).
    LayerNorm(f64),
    /// `rope(x, cos, sin)` over the last two axes `[..., S, D]`. inputs = [x,
    /// cos, sin]; cos/sin are read-only position tables (const nodes) so the VJP
    /// flows only to `x`. RoPE is a per-pair orthogonal rotation, so the
    /// backward is the inverse (transpose) rotation `dx = rope(g, cos, -sin)`
    /// (#368).
    Rope,
    /// `attn(q, k, v, mask?)` / `attn_gqa(q, k, v, mask?)` — fused scaled-dot-
    /// product attention `O = softmax(mask((Q Kᵀ)/√D)) V` per (batch, query
    /// head). inputs = [q, k, v] (+ mask as a const node when present). The
    /// backward recomputes the row-softmax weights P exactly as the forward
    /// did, then per head: dV += Pᵀ·dO, dP = dO·Vᵀ,
    /// dS = P ∘ (dP − rowsum(dP ∘ P))/√D, dQ = dS·K, dK += dSᵀ·Q. GQA: each KV
    /// head serves H_q/H_kv query heads, so dK/dV accumulate across the group.
    /// The mask gets no gradient — masked positions have P = 0 (#368).
    Attn,
    /// `sum_along(x, axis)` (axis dropped) — VJP broadcasts the adjoint back
    /// along the inserted axis (#307).
    SumAlong(usize),
    /// `mean_along(x, axis)` — like SumAlong but the adjoint is divided by the
    /// reduced axis length (#307).
    MeanAlong(usize),
    /// `max_along(x, axis)` / `min_along(x, axis)` (axis dropped) — the
    /// per-lane counterpart of `MaxReduce`/`MinReduce`. Each lane's adjoint
    /// goes to that lane's single extreme element (first occurrence on a tie,
    /// matching `argmax`/`argmin`), 0 elsewhere (#434).
    MaxAlong(usize),
    MinAlong(usize),
    /// `x[i, j, ...]` — a fully-qualified element read of a traced tensor. The
    /// index is fixed at trace time, so the VJP scatters the scalar adjoint
    /// into that one slot of a zero tensor shaped like `x` (#420).
    IndexRead(Vec<usize>),
    /// `x.reshape[[..]]` — VJP reshapes the adjoint back to the input's shape (#307).
    Reshape,
    /// `variance(x)` — population variance over all elements. VJP
    /// `dx = g·(2/N)(x - mean(x))` (#307, Tier C).
    Variance,
    /// global `max(x)` / `min(x)` reduction. Subgradient routes the (scalar)
    /// adjoint to the single extreme element, 0 elsewhere; ties break toward
    /// the first occurrence, matching argmax/argmin (#307, Tier C).
    MaxReduce,
    MinReduce,
    Negate,
    Transpose,
    /// scalar/scalar or tensor/scalar division — the second input is treated
    /// as the divisor (no gradient flows back into it for pre-alpha).
    ScalarDiv,
    /// scalar*tensor or tensor*scalar — scaling.
    ScalarMul,
    /// Scalar add/sub — the carrier op tag is preserved so backward knows
    /// the sign.
    ScalarAddSub(BinOp),
    /// A scalar-math builtin applied to a traced scalar (#420):
    /// sqrt/exp/log/sin/cos/tan. One input (the argument); the VJP multiplies
    /// the adjoint by the elementary derivative. These builtins are
    /// scalar-only (they do not broadcast over tensors), so the node value
    /// and adjoint are always scalars.
    ScalarMath(ScalarMathKind),
    /// Scalar broadcast to a tensor shape (the VJP of `Sum`). Only created by
    /// `backward_symbolic` — the forward recorder never emits it. Its own VJP
    /// is `Sum` of the incoming adjoint, which is what lets second-order
    /// gradients flow back through a first-order `sum()` reduction.
    Broadcast,
}

/// The scalar-math builtins with tape support (#420). Derivatives are
/// elementary; each is computable from the input `x` and/or the already-
/// computed output `y = f(x)`.
#[derive(Debug, Clone, Copy)]
enum ScalarMathKind { Sqrt, Exp, Log, Sin, Cos, Tan }

impl ScalarMathKind {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "sqrt" => Some(Self::Sqrt),
            "exp"  => Some(Self::Exp),
            "log"  => Some(Self::Log),
            "sin"  => Some(Self::Sin),
            "cos"  => Some(Self::Cos),
            "tan"  => Some(Self::Tan),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Sqrt => "sqrt", Self::Exp => "exp", Self::Log => "log",
            Self::Sin => "sin", Self::Cos => "cos", Self::Tan => "tan",
        }
    }

    /// dy/dx at input `x`, given the forward output `y = f(x)`. IEEE
    /// semantics at the domain edges (sqrt'(0) = +inf, log'(0) = +inf),
    /// matching what the forward itself produces there.
    fn derivative(self, x: f64, y: f64) -> f64 {
        match self {
            Self::Sqrt => 0.5 / y,
            Self::Exp  => y,
            Self::Log  => 1.0 / x,
            Self::Sin  => x.cos(),
            Self::Cos  => -x.sin(),
            Self::Tan  => 1.0 + y * y,
        }
    }
}

#[derive(Debug, Default)]
struct Tape {
    nodes: Vec<TapeNode>,
}

impl Tape {
    fn new() -> Self { Self::default() }

    fn push(&mut self, node: TapeNode) -> usize {
        let id = self.nodes.len();
        self.nodes.push(node);
        id
    }

    fn push_const(&mut self, v: Value) -> usize {
        self.push(TapeNode { op: TapeOp::Const, inputs: Vec::new(), value: v })
    }

    /// #674: one pass over
    /// every node's forward value before the backward walk touches any of
    /// them. `backward`/`backward_symbolic` take `&self`/no interpreter
    /// access and re-enter no interpreter method (#791's own doc on
    /// `backward`, just below), so no reap can interleave with the walk once
    /// this prepass has passed — the whole point of checking liveness here.
    /// Located at the
    /// `@grad` call site (`call_grad_in_module`'s span): "`forge.reap` inside
    /// `@grad` is legal, and the doc owes no fourth refusal" — a budget
    /// reaped mid-pass fails the BACKWARD pass here, not the carve/reap
    /// itself.
    fn check_live(&self, span: &Span) -> EvalResult<()> {
        for n in &self.nodes {
            if let Value::Tensor(t) = &n.value {
                if tensor_is_stale(t) {
                    return Err(RuntimeError::at(
                        "budget-stale: this tensor was allocated in a budget \
                         that has already been reaped".to_string(),
                        span.clone(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Walk the tape in reverse, propagating gradients via per-op VJP rules.
    /// `grads[id]` holds the accumulated dL/d(node id) value, or None if no
    /// gradient has reached the node yet.
    fn backward(&self, grads: &mut Vec<Option<Value>>, span: &Span) -> EvalResult<()> {
        self.check_live(span)?;
        for i in (0..self.nodes.len()).rev() {
            let n = &self.nodes[i];
            let Some(g_out) = grads[i].clone() else { continue; };
            match &n.op {
                TapeOp::Input { .. } | TapeOp::Captured { .. } | TapeOp::Const => {
                    // Leaves of the graph: no upstream to propagate to. Their
                    // accumulated adjoint *is* the answer `Grads` hands back.
                }
                TapeOp::Matmul => {
                    // c = a @ b. Each is a tensor.
                    //   dL/da = dL/dc @ b'
                    //   dL/db = a' @ dL/dc
                    let a = &self.nodes[n.inputs[0]].value;
                    let b = &self.nodes[n.inputs[1]].value;
                    let g = as_tensor(&g_out)?;
                    let bt = transpose_last_two(as_tensor(b)?);
                    let at = transpose_last_two(as_tensor(a)?);
                    let da = tensor_matmul(&Value::tensor(g.clone()), &Value::tensor(bt))?;
                    let db = tensor_matmul(&Value::tensor(at), &Value::tensor(g))?;
                    accumulate(grads, n.inputs[0], da);
                    accumulate(grads, n.inputs[1], db);
                }
                TapeOp::DotAdd => {
                    // c = a .+ b →  dL/da = dL/dc; dL/db = dL/dc
                    // (each reduced to its operand's shape so a scalar operand
                    // gets the summed contribution, #252).
                    let a = self.nodes[n.inputs[0]].value.clone();
                    let b = self.nodes[n.inputs[1]].value.clone();
                    accumulate(grads, n.inputs[0], grad_reduce_to(g_out.clone(), &a));
                    accumulate(grads, n.inputs[1], grad_reduce_to(g_out, &b));
                }
                TapeOp::DotSub => {
                    // c = a .- b →  dL/da = dL/dc; dL/db = -dL/dc
                    let a = self.nodes[n.inputs[0]].value.clone();
                    let b = self.nodes[n.inputs[1]].value.clone();
                    accumulate(grads, n.inputs[0], grad_reduce_to(g_out.clone(), &a));
                    accumulate(grads, n.inputs[1], grad_reduce_to(negate_value(&g_out)?, &b));
                }
                // Product rule, both operands (#252). Dotted `.*` and scalar `*`
                // share the math; `grad_mul`/`grad_reduce_to` handle every
                // scalar/tensor mix, so a traced scalar or reduction operand no
                // longer drops its gradient.
                TapeOp::DotMul | TapeOp::ScalarMul => {
                    let a = self.nodes[n.inputs[0]].value.clone();
                    let b = self.nodes[n.inputs[1]].value.clone();
                    let da = grad_reduce_to(grad_mul(&g_out, &b)?, &a);
                    let db = grad_reduce_to(grad_mul(&g_out, &a)?, &b);
                    accumulate(grads, n.inputs[0], da);
                    accumulate(grads, n.inputs[1], db);
                }
                // Quotient rule, both operands (#252): da = g/b; db = -g·a/b².
                TapeOp::DotDiv | TapeOp::ScalarDiv => {
                    let a = self.nodes[n.inputs[0]].value.clone();
                    let b = self.nodes[n.inputs[1]].value.clone();
                    let da = grad_reduce_to(grad_div(&g_out, &b)?, &a);
                    let ga = grad_mul(&g_out, &a)?;
                    let b2 = grad_mul(&b, &b)?;
                    let db = grad_reduce_to(negate_value(&grad_div(&ga, &b2)?)?, &b);
                    accumulate(grads, n.inputs[0], da);
                    accumulate(grads, n.inputs[1], db);
                }
                TapeOp::Sum => {
                    // c = sum(a) — scalar.  dL/da = dL/dc * ones_like(a)
                    let a = &self.nodes[n.inputs[0]].value;
                    let at = as_tensor(a)?;
                    let g_scalar = g_out.as_float().unwrap_or(0.0);
                    let da = ArrayD::from_elem(at.raw_dim(), g_scalar);
                    accumulate(grads, n.inputs[0], Value::tensor(da));
                }
                TapeOp::ReLU => {
                    // c = relu(a) →  dL/da = dL/dc * (a > 0)
                    let a = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = as_tensor(&g_out)?;
                    let mut out = g.clone();
                    for (slot, &av) in out.iter_mut().zip(a.iter()) {
                        if av <= 0.0 { *slot = 0.0; }
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(out));
                }
                TapeOp::Activation(kind) => {
                    // c = act(a) →  dL/da = dL/dc * act'(a) (#306). Derivative
                    // taken w.r.t. the input, matching the forward activation.
                    let a = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = as_tensor(&g_out)?;
                    let name = kind.name();
                    let mut out = g.clone();
                    for (slot, &av) in out.iter_mut().zip(a.iter()) {
                        *slot *= activation_deriv_f64(name, av);
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(out));
                }
                TapeOp::ActivationGrad(kind) => {
                    // d = act'(x) →  dL/dx = dL/dd * act''(x) (#306, second-order
                    // term). Emitted only by backward_symbolic; differentiating it
                    // here is what gives `@grad @grad` the activation curvature.
                    let a = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = as_tensor(&g_out)?;
                    let name = kind.name();
                    let mut out = g.clone();
                    for (slot, &av) in out.iter_mut().zip(a.iter()) {
                        *slot *= activation_second_deriv_f64(name, av);
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(out));
                }
                TapeOp::Softmax(axis) => {
                    // y = softmax(x, axis) → dx = y .* (g - rowsum(g .* y, axis))
                    // (#307). The forward output y is on the node.
                    let y = as_tensor(&n.value)?.clone();
                    let g = as_tensor(&g_out)?.clone();
                    let ax = Axis(*axis);
                    let gy = &g * &y;
                    let s = gy.sum_axis(ax).insert_axis(ax);
                    let s_b = s.broadcast(y.raw_dim()).expect("softmax grad broadcast").to_owned();
                    let dx = &y * &(&g - &s_b);
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::RmsNorm(eps) => {
                    // y = gain * x / r,  r = sqrt(mean(x^2)+eps)  over the last axis (#307).
                    //   dL/dx_k    = g_k gain_k / r  -  x_k S / (D r^3),  S = Σ_i g_i gain_i x_i
                    //   dL/dgain_j = Σ_rows g_j x_j / r
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let gain = as_tensor(&self.nodes[n.inputs[1]].value)?;
                    let g = as_tensor(&g_out)?;
                    let last = x.ndim() - 1;
                    let dlen = x.shape()[last] as f64;
                    let mut red = x.shape().to_vec();
                    red[last] = 1;
                    let r = x.mapv(|v| v * v).mean_axis(Axis(last))
                        .expect("rms_norm grad: empty axis")
                        .into_shape_with_order(IxDyn(&red)).unwrap()
                        .mapv(|m| (m + *eps).sqrt());
                    // S per row (keepdim).
                    let mut s = ArrayD::<f64>::zeros(IxDyn(&red));
                    for (ido, &xv) in x.indexed_iter() {
                        let idx: Vec<usize> = ido.as_array_view().iter().copied().collect();
                        let j = idx[last];
                        let mut ri = idx.clone(); ri[last] = 0;
                        s[IxDyn(&ri)] += g[IxDyn(&idx)] * gain[IxDyn(&[j])] * xv;
                    }
                    let mut dx = ArrayD::<f64>::zeros(x.raw_dim());
                    let mut dgain = ArrayD::<f64>::zeros(gain.raw_dim());
                    for (ido, &xv) in x.indexed_iter() {
                        let idx: Vec<usize> = ido.as_array_view().iter().copied().collect();
                        let j = idx[last];
                        let mut ri = idx.clone(); ri[last] = 0;
                        let rr = r[IxDyn(&ri)];
                        let ss = s[IxDyn(&ri)];
                        let gv = g[IxDyn(&idx)];
                        dx[IxDyn(&idx)] = gv * gain[IxDyn(&[j])] / rr - xv * ss / (dlen * rr * rr * rr);
                        dgain[IxDyn(&[j])] += gv * xv / rr;
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                    accumulate(grads, n.inputs[1], Value::tensor(dgain));
                }
                TapeOp::LayerNorm(eps) => {
                    // y = gain * (x-μ)/std + bias,  std = sqrt(var+eps), last axis (#307).
                    //   x̂_i = (x_i-μ)/std ; dx̂_i = g_i gain_i
                    //   dx_k = (dx̂_k - mean(dx̂) - x̂_k·mean(dx̂·x̂)) / std
                    //   dgain_j = Σ_rows g_j x̂_j ;  dbias_j = Σ_rows g_j
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let gain = as_tensor(&self.nodes[n.inputs[1]].value)?;
                    let g = as_tensor(&g_out)?;
                    let last = x.ndim() - 1;
                    let dlen = x.shape()[last] as f64;
                    let mut red = x.shape().to_vec();
                    red[last] = 1;
                    let mean = x.mean_axis(Axis(last)).expect("layer_norm grad: empty axis")
                        .into_shape_with_order(IxDyn(&red)).unwrap();
                    let mut var = ArrayD::<f64>::zeros(IxDyn(&red));
                    for (ido, &xv) in x.indexed_iter() {
                        let idx: Vec<usize> = ido.as_array_view().iter().copied().collect();
                        let mut ri = idx.clone(); ri[last] = 0;
                        let d = xv - mean[IxDyn(&ri)];
                        var[IxDyn(&ri)] += d * d;
                    }
                    let std = var.mapv(|v| (v / dlen + *eps).sqrt());
                    // Per-row reductions of dx̂ and dx̂·x̂.
                    let mut a = ArrayD::<f64>::zeros(IxDyn(&red));
                    let mut bsum = ArrayD::<f64>::zeros(IxDyn(&red));
                    for (ido, &xv) in x.indexed_iter() {
                        let idx: Vec<usize> = ido.as_array_view().iter().copied().collect();
                        let j = idx[last];
                        let mut ri = idx.clone(); ri[last] = 0;
                        let st = std[IxDyn(&ri)];
                        let xhat = (xv - mean[IxDyn(&ri)]) / st;
                        let dxhat = g[IxDyn(&idx)] * gain[IxDyn(&[j])];
                        a[IxDyn(&ri)] += dxhat;
                        bsum[IxDyn(&ri)] += dxhat * xhat;
                    }
                    let mut dx = ArrayD::<f64>::zeros(x.raw_dim());
                    let mut dgain = ArrayD::<f64>::zeros(gain.raw_dim());
                    let mut dbias = ArrayD::<f64>::zeros(gain.raw_dim());
                    for (ido, &xv) in x.indexed_iter() {
                        let idx: Vec<usize> = ido.as_array_view().iter().copied().collect();
                        let j = idx[last];
                        let mut ri = idx.clone(); ri[last] = 0;
                        let st = std[IxDyn(&ri)];
                        let xhat = (xv - mean[IxDyn(&ri)]) / st;
                        let gv = g[IxDyn(&idx)];
                        let dxhat = gv * gain[IxDyn(&[j])];
                        dx[IxDyn(&idx)] = (dxhat - a[IxDyn(&ri)] / dlen - xhat * bsum[IxDyn(&ri)] / dlen) / st;
                        dgain[IxDyn(&[j])] += gv * xhat;
                        dbias[IxDyn(&[j])] += gv;
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                    accumulate(grads, n.inputs[1], Value::tensor(dgain));
                    accumulate(grads, n.inputs[2], Value::tensor(dbias));
                }
                TapeOp::Rope => {
                    // rope is a per-pair orthogonal rotation; its VJP is the
                    // inverse (transpose) rotation of the adjoint, i.e.
                    // `dx = rope(g, cos, -sin)`. Mirrors builtin_rope's forward
                    // loop with the rotation transposed. cos/sin (inputs 1,2)
                    // are read-only tables — no gradient flows to them (#368).
                    let cos = as_tensor(&self.nodes[n.inputs[1]].value)?;
                    let sin = as_tensor(&self.nodes[n.inputs[2]].value)?;
                    let g = as_tensor(&g_out)?;
                    let d_axis = g.ndim() - 1;
                    let s_axis = g.ndim() - 2;
                    let mut dx = g.clone();
                    for (idx_owned, _) in g.indexed_iter() {
                        let idx: Vec<usize> = idx_owned.as_array_view().iter().copied().collect();
                        let dim_i = idx[d_axis];
                        if dim_i % 2 != 0 { continue; }   // pairs keyed by even index
                        let pair_i = dim_i / 2;
                        let mut idx_b = idx.clone();
                        idx_b[d_axis] = dim_i + 1;
                        let ge = g[IxDyn(&idx)];
                        let go = g[IxDyn(&idx_b)];
                        let c  = cos[IxDyn(&[idx[s_axis], pair_i])];
                        let si = sin[IxDyn(&[idx[s_axis], pair_i])];
                        // J^T: [[c, s], [-s, c]] applied to (ge, go).
                        dx[IxDyn(&idx)]   = ge * c + go * si;
                        dx[IxDyn(&idx_b)] = go * c - ge * si;
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::Attn => {
                    // Fused attention VJP (#368). Recompute the row-softmax
                    // weights P per (batch, query head) exactly as the forward
                    // did (scores → scale → mask → softmax), then:
                    //   dV += Pᵀ·dO          dP = dO·Vᵀ
                    //   dS = P ∘ (dP − rowsum(dP ∘ P)) / √D
                    //   dQ = dS·K            dK += dSᵀ·Q
                    // GQA accumulates dK/dV across the query heads sharing a KV
                    // head. Masked positions have P = 0, so dS is 0 there and
                    // no gradient flows to (or through) the mask.
                    let q = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let k = as_tensor(&self.nodes[n.inputs[1]].value)?;
                    let v = as_tensor(&self.nodes[n.inputs[2]].value)?;
                    let mask = if n.inputs.len() == 4 {
                        Some(as_tensor(&self.nodes[n.inputs[3]].value)?)
                    } else { None };
                    let g = as_tensor(&g_out)?;
                    let b = q.shape()[0];
                    let h_q = q.shape()[1];
                    let s = q.shape()[2];
                    let d = q.shape()[3];
                    let h_kv = k.shape()[1];
                    let grp = h_q / h_kv;
                    let scale = 1.0 / (d as f64).sqrt();
                    let mm = |a: &ArrayD<f64>, bb: &ArrayD<f64>| -> EvalResult<ArrayD<f64>> {
                        match tensor_matmul(&Value::tensor(a.clone()), &Value::tensor(bb.clone()))? {
                            Value::Tensor(t) => Ok(t.data),
                            _ => unreachable!(),
                        }
                    };
                    let t2 = |a: &ArrayD<f64>| -> ArrayD<f64> {
                        a.view().permuted_axes(IxDyn(&[1, 0]))
                            .as_standard_layout().to_owned()
                    };
                    let mut dq = ArrayD::<f64>::zeros(q.raw_dim());
                    let mut dk = ArrayD::<f64>::zeros(k.raw_dim());
                    let mut dv = ArrayD::<f64>::zeros(v.raw_dim());
                    for bi in 0..b {
                        for hqi in 0..h_q {
                            let hkvi = hqi / grp;
                            let q_head = q.slice(ndarray::s![bi, hqi, .., ..]).to_owned().into_dyn();
                            let k_head = k.slice(ndarray::s![bi, hkvi, .., ..]).to_owned().into_dyn();
                            let v_head = v.slice(ndarray::s![bi, hkvi, .., ..]).to_owned().into_dyn();
                            let g_head = g.slice(ndarray::s![bi, hqi, .., ..]).to_owned().into_dyn();
                            // P: scores → scale → mask → row softmax (as forward).
                            let mut scaled = mm(&q_head, &t2(&k_head))?.mapv(|x| x * scale);
                            if let Some(m) = &mask {
                                for i in 0..s {
                                    for j in 0..s {
                                        if m[IxDyn(&[i, j])] == 0.0 {
                                            scaled[IxDyn(&[i, j])] = f64::NEG_INFINITY;
                                        }
                                    }
                                }
                            }
                            let p = match builtin_softmax(
                                &[Value::tensor(scaled), Value::Int(-1, IW::I64)],
                                Span { start: 0, end: 0, line: 0, col: 0 })? {
                                Value::Tensor(t) => t,
                                _ => unreachable!(),
                            };
                            // dV += Pᵀ·dO
                            let dv_head = mm(&t2(&p), &g_head)?;
                            dv.slice_mut(ndarray::s![bi, hkvi, .., ..])
                                .zip_mut_with(&dv_head, |a, bb| *a += *bb);
                            // dP = dO·Vᵀ; dS = P ∘ (dP − rowsum(dP ∘ P)) / √D
                            let dp = mm(&g_head, &t2(&v_head))?;
                            let mut ds = ArrayD::<f64>::zeros(p.raw_dim());
                            for i in 0..s {
                                let mut c = 0.0;
                                for j in 0..s {
                                    c += dp[IxDyn(&[i, j])] * p[IxDyn(&[i, j])];
                                }
                                for j in 0..s {
                                    ds[IxDyn(&[i, j])] =
                                        p[IxDyn(&[i, j])] * (dp[IxDyn(&[i, j])] - c) * scale;
                                }
                            }
                            // dQ = dS·K (own slice, written once); dK += dSᵀ·Q
                            let dq_head = mm(&ds, &k_head)?;
                            dq.slice_mut(ndarray::s![bi, hqi, .., ..]).assign(&dq_head);
                            let dk_head = mm(&t2(&ds), &q_head)?;
                            dk.slice_mut(ndarray::s![bi, hkvi, .., ..])
                                .zip_mut_with(&dk_head, |a, bb| *a += *bb);
                        }
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(dq));
                    accumulate(grads, n.inputs[1], Value::tensor(dk));
                    accumulate(grads, n.inputs[2], Value::tensor(dv));
                }
                TapeOp::SumAlong(axis) | TapeOp::MeanAlong(axis) => {
                    // y = sum/mean over `axis` (axis dropped). dx broadcasts the
                    // adjoint back along that axis (÷ len for mean) (#307).
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = as_tensor(&g_out)?;
                    let dlen = x.shape()[*axis] as f64;
                    // g has x's shape with `axis` removed → insert it back, broadcast.
                    let g_keep = g.clone().insert_axis(Axis(*axis));
                    let mut dx = g_keep.broadcast(x.raw_dim())
                        .expect("reduce_along grad broadcast").to_owned();
                    if matches!(n.op, TapeOp::MeanAlong(_)) {
                        dx.mapv_inplace(|v| v / dlen);
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::MaxAlong(axis) | TapeOp::MinAlong(axis) => {
                    // y = max/min over `axis` (axis dropped). Subgradient: each
                    // lane's adjoint goes to that lane's extreme element, 0
                    // elsewhere — the per-axis form of MaxReduce/MinReduce
                    // (#434). Ties break toward the first occurrence, matching
                    // argmax/argmin and the whole-tensor rule.
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = as_tensor(&g_out)?;
                    let want_max = matches!(n.op, TapeOp::MaxAlong(_));
                    let mut best = ArrayD::from_elem(
                        g.raw_dim(),
                        if want_max { f64::NEG_INFINITY } else { f64::INFINITY });
                    let mut arg = ArrayD::<usize>::zeros(g.raw_dim());
                    for (ido, &xv) in x.indexed_iter() {
                        let full: Vec<usize> = ido.as_array_view().iter().copied().collect();
                        let k = full[*axis];
                        let mut red = full;
                        red.remove(*axis);
                        let ri = IxDyn(&red);
                        let better = if want_max { xv > best[ri.clone()] } else { xv < best[ri.clone()] };
                        if better {
                            best[ri.clone()] = xv;
                            arg[ri] = k;
                        }
                    }
                    let mut dx = ArrayD::<f64>::zeros(x.raw_dim());
                    for (ido, &gv) in g.indexed_iter() {
                        let red: Vec<usize> = ido.as_array_view().iter().copied().collect();
                        let mut full = red.clone();
                        full.insert(*axis, arg[IxDyn(&red)]);
                        dx[IxDyn(&full)] = gv;
                    }
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::IndexRead(idx) => {
                    // y = x[i, j, ...] → dx is zero everywhere but that slot,
                    // which receives the whole scalar adjoint (#420).
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let mut dx = ArrayD::<f64>::zeros(x.raw_dim());
                    dx[IxDyn(idx)] = g_out.as_float().unwrap_or(0.0);
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::Reshape => {
                    // y = reshape(x). dx = reshape(g, shape(x)) (row-major order
                    // preserved, so element-for-element) (#307).
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = as_tensor(&g_out)?;
                    let dx = g.clone().into_shape_with_order(x.raw_dim())
                        .expect("reshape grad: element count mismatch");
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::Variance => {
                    // var = (1/N) Σ (x_i - μ)²  →  dx_i = g·(2/N)(x_i - μ) (#307).
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = g_out.as_float().unwrap_or(0.0);
                    let nlen = x.len() as f64;
                    let mean = if nlen > 0.0 { x.sum() / nlen } else { 0.0 };
                    let k = if nlen > 0.0 { g * 2.0 / nlen } else { 0.0 };
                    let dx = x.mapv(|v| k * (v - mean));
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::MaxReduce | TapeOp::MinReduce => {
                    // global max/min: the scalar adjoint flows to the single
                    // extreme element (first occurrence on a tie), 0 elsewhere
                    // (#307). Flat iteration order matches `x.iter()` in
                    // call_builtin's reduction.
                    let x = as_tensor(&self.nodes[n.inputs[0]].value)?;
                    let g = g_out.as_float().unwrap_or(0.0);
                    let want_max = matches!(n.op, TapeOp::MaxReduce);
                    let mut best = if want_max { f64::NEG_INFINITY } else { f64::INFINITY };
                    let mut best_i = 0usize;
                    for (i, &v) in x.iter().enumerate() {
                        let better = if want_max { v > best } else { v < best };
                        if better { best = v; best_i = i; }
                    }
                    let mut dx = ArrayD::<f64>::zeros(x.raw_dim());
                    if let Some(slot) = dx.iter_mut().nth(best_i) { *slot = g; }
                    accumulate(grads, n.inputs[0], Value::tensor(dx));
                }
                TapeOp::Negate => {
                    let neg = negate_value(&g_out)?;
                    accumulate(grads, n.inputs[0], neg);
                }
                TapeOp::Transpose => {
                    // c = a' (swap last two axes) →  dL/da = dL/dc'
                    let g = as_tensor(&g_out)?;
                    let t = transpose_last_two(g);
                    accumulate(grads, n.inputs[0], Value::tensor(t));
                }
                TapeOp::ScalarAddSub(op) => {
                    // a + b / a - b for scalar operands. Reduce to each
                    // operand's shape so a tensor adjoint into a scalar operand
                    // sums rather than zeroing (#252).
                    let a = self.nodes[n.inputs[0]].value.clone();
                    let b = self.nodes[n.inputs[1]].value.clone();
                    let sign = matches!(op, BinOp::Add);
                    accumulate(grads, n.inputs[0], grad_reduce_to(g_out.clone(), &a));
                    let g_right = if sign { g_out } else { negate_value(&g_out)? };
                    accumulate(grads, n.inputs[1], grad_reduce_to(g_right, &b));
                }
                TapeOp::ScalarMath(kind) => {
                    // y = f(x), scalar → dL/dx = dL/dy · f'(x)
                    let x = self.nodes[n.inputs[0]].value.as_float().unwrap_or(0.0);
                    let y = n.value.as_float().unwrap_or(0.0);
                    let g = g_out.as_float().unwrap_or(0.0);
                    accumulate(grads, n.inputs[0], Value::Float(g * kind.derivative(x, y), FW::F64));
                }
                TapeOp::Broadcast => {
                    // c = broadcast(s) (scalar → tensor) → dL/ds = sum(dL/dc)
                    let g = as_tensor(&g_out)?;
                    accumulate(grads, n.inputs[0], Value::Float(g.iter().sum(), FW::F64));
                }
            }
        }
        Ok(())
    }

    // ── Second-order: symbolic taping of the reverse pass ──────────────────
    //
    // `fwd_bwd_bwd` needs to differentiate the backward pass itself. These
    // helpers replay each VJP rule as *real tape nodes* (with eagerly computed
    // values), so a subsequent ordinary `backward()` over the extended tape
    // yields second derivatives. Mirrors the JIT's `tape_backward` (jit.rs).

    /// Push a `DotMul` node over two existing nodes, computing its value.
    fn push_dotmul(&mut self, a: usize, b: usize) -> EvalResult<usize> {
        let val = apply_binop(BinOp::DotMul,
            &self.nodes[a].value.clone(), &self.nodes[b].value.clone())?;
        Ok(self.push(TapeNode { op: TapeOp::DotMul, inputs: vec![a, b], value: val }))
    }

    /// Push a `Negate` node over an existing node, computing its value.
    fn push_negate(&mut self, a: usize) -> EvalResult<usize> {
        let val = negate_value(&self.nodes[a].value)?;
        Ok(self.push(TapeNode { op: TapeOp::Negate, inputs: vec![a], value: val }))
    }

    /// Symbolic adjoint accumulation: where `accumulate` adds gradient
    /// *values*, this records the addition as a tape node so the second
    /// backward pass can differentiate through it.
    fn accumulate_sym(
        &mut self,
        adj: &mut [Option<usize>],
        target: usize,
        contrib: usize,
    ) -> EvalResult<()> {
        match adj[target] {
            None => { adj[target] = Some(contrib); }
            Some(existing) => {
                let a = self.nodes[existing].value.clone();
                let b = self.nodes[contrib].value.clone();
                let id = if matches!(a, Value::Tensor(_)) {
                    let val = apply_binop(BinOp::DotAdd, &a, &b)?;
                    self.push(TapeNode {
                        op: TapeOp::DotAdd, inputs: vec![existing, contrib], value: val,
                    })
                } else {
                    let val = apply_binop(BinOp::Add, &a, &b)?;
                    self.push(TapeNode {
                        op: TapeOp::ScalarAddSub(BinOp::Add),
                        inputs: vec![existing, contrib], value: val,
                    })
                };
                adj[target] = Some(id);
            }
        }
        Ok(())
    }

    /// Replay the backward pass from `loss_node` symbolically onto the tape.
    /// Returns, for every *original* node, the tape node id holding its
    /// adjoint dL/d(node) — `None` where no gradient flows. The returned ids
    /// (and the nodes appended along the way) form a differentiable extension
    /// of the tape: seed an ordinary `backward()` at one of them to get the
    /// second derivative.
    fn backward_symbolic(&mut self, loss_node: usize, span: &Span) -> EvalResult<Vec<Option<usize>>> {
        // #674: see
        // `backward`'s doc.
        self.check_live(span)?;
        let n = self.nodes.len();
        let mut adj: Vec<Option<usize>> = vec![None; n];
        let seed = self.push_const(Value::Float(1.0, FW::F64));
        adj[loss_node] = Some(seed);
        for i in (0..n).rev() {
            let Some(g) = adj[i] else { continue };
            let op = self.nodes[i].op.clone();
            let inputs = self.nodes[i].inputs.clone();
            let g_val = self.nodes[g].value.clone();
            match op {
                TapeOp::Input { .. } | TapeOp::Captured { .. } | TapeOp::Const => {}
                TapeOp::Matmul => {
                    // da = g @ b';  db = a' @ g
                    let b_val = self.nodes[inputs[1]].value.clone();
                    let bt_val = Value::tensor(transpose_last_two(as_tensor(&b_val)?));
                    let da_val = tensor_matmul(&g_val, &bt_val)?;
                    let bt = self.push(TapeNode {
                        op: TapeOp::Transpose, inputs: vec![inputs[1]], value: bt_val,
                    });
                    let da = self.push(TapeNode {
                        op: TapeOp::Matmul, inputs: vec![g, bt], value: da_val,
                    });
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                    let a_val = self.nodes[inputs[0]].value.clone();
                    let at_val = Value::tensor(transpose_last_two(as_tensor(&a_val)?));
                    let db_val = tensor_matmul(&at_val, &g_val)?;
                    let at = self.push(TapeNode {
                        op: TapeOp::Transpose, inputs: vec![inputs[0]], value: at_val,
                    });
                    let db = self.push(TapeNode {
                        op: TapeOp::Matmul, inputs: vec![at, g], value: db_val,
                    });
                    self.accumulate_sym(&mut adj, inputs[1], db)?;
                }
                TapeOp::DotAdd => {
                    self.accumulate_sym(&mut adj, inputs[0], g)?;
                    self.accumulate_sym(&mut adj, inputs[1], g)?;
                }
                TapeOp::DotSub => {
                    self.accumulate_sym(&mut adj, inputs[0], g)?;
                    let db = self.push_negate(g)?;
                    self.accumulate_sym(&mut adj, inputs[1], db)?;
                }
                TapeOp::DotMul => {
                    let da = self.push_dotmul(g, inputs[1])?;
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                    let db = self.push_dotmul(g, inputs[0])?;
                    self.accumulate_sym(&mut adj, inputs[1], db)?;
                }
                TapeOp::DotDiv => {
                    // da = g ./ b;   db = -(g .* a ./ (b .* b))
                    let da_val = apply_binop(BinOp::DotDiv,
                        &g_val, &self.nodes[inputs[1]].value.clone())?;
                    let da = self.push(TapeNode {
                        op: TapeOp::DotDiv, inputs: vec![g, inputs[1]], value: da_val,
                    });
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                    let num = self.push_dotmul(g, inputs[0])?;
                    let b2 = self.push_dotmul(inputs[1], inputs[1])?;
                    let q_val = apply_binop(BinOp::DotDiv,
                        &self.nodes[num].value.clone(), &self.nodes[b2].value.clone())?;
                    let q = self.push(TapeNode {
                        op: TapeOp::DotDiv, inputs: vec![num, b2], value: q_val,
                    });
                    let db = self.push_negate(q)?;
                    self.accumulate_sym(&mut adj, inputs[1], db)?;
                }
                TapeOp::Sum => {
                    // da = broadcast(g, shape(a)) — recorded as a Broadcast
                    // node so the second pass can flow back into `g`.
                    let at = as_tensor(&self.nodes[inputs[0]].value)?;
                    let gs = g_val.as_float().unwrap_or(0.0);
                    let da_val = Value::tensor(ArrayD::from_elem(at.raw_dim(), gs));
                    let da = self.push(TapeNode {
                        op: TapeOp::Broadcast, inputs: vec![g], value: da_val,
                    });
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                }
                TapeOp::ReLU => {
                    // The 0/1 mask is constant wrt the input (a.e.), so it
                    // enters as a Const — second derivative through it is 0.
                    let a = as_tensor(&self.nodes[inputs[0]].value)?;
                    let mask = a.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                    let mask_id = self.push_const(Value::tensor(mask));
                    let da = self.push_dotmul(g, mask_id)?;
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                }
                TapeOp::Activation(kind) => {
                    // dx = g .* act'(x). Record act'(x) as an `ActivationGrad`
                    // node over the *input* so the second backward differentiates
                    // it (→ g .* act''(x) into x). Unlike ReLU's a.e.-constant
                    // mask, smooth activations carry curvature, so act'(x) must
                    // stay a live function of x rather than a Const (#306).
                    let name = kind.name();
                    let xval = self.nodes[inputs[0]].value.clone();
                    let d1_val = Value::tensor(as_tensor(&xval)?.mapv(|v| activation_deriv_f64(name, v)));
                    let d1 = self.push(TapeNode {
                        op: TapeOp::ActivationGrad(kind), inputs: vec![inputs[0]], value: d1_val,
                    });
                    let da = self.push_dotmul(g, d1)?;
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                }
                TapeOp::ActivationGrad(_) => {
                    // Would only be reached at third order (`@grad @grad @grad`):
                    // backward_symbolic appends ActivationGrad nodes but never
                    // revisits them in this same pass. Third-order is out of scope.
                    return Err(RuntimeError::msg(
                        "@grad: third-order gradient through an activation is not \
                         supported (first- and second-order work)".to_string()));
                }
                TapeOp::Softmax(_) => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `softmax` is not \
                         supported yet (first-order works)".to_string()));
                }
                TapeOp::RmsNorm(_) => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `rms_norm` is not \
                         supported yet (first-order works)".to_string()));
                }
                TapeOp::LayerNorm(_) => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `layer_norm` is not \
                         supported yet (first-order works)".to_string()));
                }
                TapeOp::Rope => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `rope` is not \
                         supported yet (first-order works)".to_string()));
                }
                TapeOp::Attn => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `attn`/`attn_gqa` \
                         is not supported yet (first-order works)".to_string()));
                }
                TapeOp::SumAlong(_) | TapeOp::MeanAlong(_) | TapeOp::Reshape => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `sum_along`/`mean_along`/\
                         `reshape` is not supported yet (first-order works)".to_string()));
                }
                TapeOp::Variance | TapeOp::MaxReduce | TapeOp::MinReduce => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `variance`/`max`/`min` \
                         is not supported yet (first-order works)".to_string()));
                }
                TapeOp::MaxAlong(_) | TapeOp::MinAlong(_) => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through `max_along`/`min_along` \
                         is not supported yet (first-order works)".to_string()));
                }
                TapeOp::IndexRead(_) => {
                    return Err(RuntimeError::msg(
                        "@grad @grad: second-order gradient through an indexed read `x[i]` \
                         is not supported yet (first-order works)".to_string()));
                }
                TapeOp::ScalarMath(kind) => {
                    return Err(RuntimeError::msg(format!(
                        "@grad @grad: second-order gradient through scalar `{}` is not \
                         supported yet (first-order works)", kind.name())));
                }
                TapeOp::Negate => {
                    let da = self.push_negate(g)?;
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                }
                TapeOp::Transpose => {
                    let da_val = Value::tensor(transpose_last_two(as_tensor(&g_val)?));
                    let da = self.push(TapeNode {
                        op: TapeOp::Transpose, inputs: vec![g], value: da_val,
                    });
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                }
                TapeOp::ScalarDiv => {
                    let s = self.nodes[inputs[1]].value.as_float().unwrap_or(1.0);
                    if s == 0.0 {
                        return Err(RuntimeError::msg("@grad: divide by zero in backward"));
                    }
                    let da_val = scale_value(&g_val, 1.0 / s)?;
                    let da = self.push(TapeNode {
                        op: TapeOp::ScalarDiv, inputs: vec![g, inputs[1]], value: da_val,
                    });
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                }
                TapeOp::ScalarMul => {
                    let s = self.nodes[inputs[1]].value.as_float().unwrap_or(1.0);
                    let da_val = scale_value(&g_val, s)?;
                    let da = self.push(TapeNode {
                        op: TapeOp::ScalarMul, inputs: vec![g, inputs[1]], value: da_val,
                    });
                    self.accumulate_sym(&mut adj, inputs[0], da)?;
                }
                TapeOp::ScalarAddSub(op) => {
                    self.accumulate_sym(&mut adj, inputs[0], g)?;
                    let db = if matches!(op, BinOp::Add) { g } else { self.push_negate(g)? };
                    self.accumulate_sym(&mut adj, inputs[1], db)?;
                }
                TapeOp::Broadcast =>
                    unreachable!("Broadcast nodes are only created by backward_symbolic"),
            }
        }
        Ok(adj)
    }
}

/// Can this runtime value carry a gradient (AUTODIFF.md §2)? Float tensors and
/// float scalars can; integer/bool/trit tensors, integers, strings, structs and
/// the rest cannot — taping them would hand back a meaningless adjoint. Used to
/// decide which captured `mut` bindings become tape inputs (#398); the checker's
/// `capture_type_is_differentiable` is the static counterpart.
fn value_is_differentiable(v: &Value) -> bool {
    match v {
        // #834: fp8 tensors are nominal f32-backed tags, differentiable
        // exactly like plain `F32` (unchanged from before this fix, when the
        // interpreter mistagged them AS `DType::F32` outright).
        Value::Tensor(t) => matches!(t.dtype, DType::F32 | DType::F64 | DType::Fp8E4M3 | DType::Fp8E5M2),
        Value::Float(..) => true,
        _ => false,
    }
}

fn accumulate(grads: &mut Vec<Option<Value>>, idx: usize, add: Value) {
    grads[idx] = Some(match grads[idx].take() {
        None => add,
        Some(prev) => grad_add(&prev, &add).unwrap_or(add),
    });
}

/// Add two gradient values, tolerating any scalar/tensor mix. `apply_binop`'s
/// `DotAdd` only accepts two tensors, so two scalar adjoints (which arise once
/// gradients are reduced to a scalar reduction node) would otherwise fall
/// through `accumulate`'s `unwrap_or` and *overwrite* instead of summing.
fn grad_add(a: &Value, b: &Value) -> EvalResult<Value> {
    match (a, b) {
        (Value::Tensor(_), Value::Tensor(_)) => apply_binop(BinOp::DotAdd, a, b),
        (Value::Tensor(t), s) | (s, Value::Tensor(t)) => {
            let sf = s.as_float().unwrap_or(0.0);
            Ok(Value::tensor(t.mapv(|x| x + sf)))
        }
        _ => Ok(Value::Float(a.as_float().unwrap_or(0.0) + b.as_float().unwrap_or(0.0), FW::F64)),
    }
}

/// Elementwise multiply for VJP combination. Handles tensor⊙tensor,
/// tensor⊙scalar (broadcast scaling), and scalar·scalar — the operand mixes
/// that arise differentiating `*` / `.*` when one side is a scalar or a
/// reduction result (#252).
fn grad_mul(x: &Value, y: &Value) -> EvalResult<Value> {
    match (x, y) {
        (Value::Tensor(_), Value::Tensor(_)) => apply_binop(BinOp::DotMul, x, y),
        (Value::Tensor(t), s) | (s, Value::Tensor(t)) => {
            let sf = s.as_float().unwrap_or(0.0);
            Ok(Value::tensor(t.mapv(|v| v * sf)))
        }
        _ => Ok(Value::Float(x.as_float().unwrap_or(0.0) * y.as_float().unwrap_or(0.0), FW::F64)),
    }
}

/// Elementwise `x / y` for VJP combination, mirroring `grad_mul`'s type mixes
/// (including `scalar / tensor`, which arises in the quotient rule's `g / b`).
fn grad_div(x: &Value, y: &Value) -> EvalResult<Value> {
    match (x, y) {
        (Value::Tensor(_), Value::Tensor(_)) => apply_binop(BinOp::DotDiv, x, y),
        (Value::Tensor(t), s) => {
            let sf = s.as_float().unwrap_or(1.0);
            Ok(Value::tensor(t.mapv(|v| v / sf)))
        }
        (s, Value::Tensor(t)) => {
            let sf = s.as_float().unwrap_or(0.0);
            Ok(Value::tensor(t.mapv(|v| sf / v)))
        }
        _ => Ok(Value::Float(x.as_float().unwrap_or(0.0) / y.as_float().unwrap_or(1.0), FW::F64)),
    }
}

/// Reduce a gradient contribution to the shape of the operand it flows into.
/// When the forward op broadcast a scalar operand over a tensor, that operand
/// receives the **sum** of the broadcast contributions. Without this, a tensor
/// adjoint reaching a scalar node (e.g. a reduction like `sum(w)` used as a
/// multiplier) is silently zeroed by the scalar VJPs (`as_float()` on a tensor
/// → 0) — the core of #252's wrong/zero gradients.
fn grad_reduce_to(contrib: Value, operand: &Value) -> Value {
    match operand {
        // Tensor operand: reduce the contribution back to this operand's shape.
        // When the operand was broadcast in the forward op (e.g. a bias `[H]`
        // added to `[B, H]`), the upstream cotangent arrives at the broadcast
        // (larger) shape and must be summed over the broadcast axes (#299). When
        // the shapes already match this is a no-op, so the common case (the
        // other side was a scalar) is unaffected.
        Value::Tensor(op) => match contrib {
            Value::Tensor(c) => Value::tensor(reduce_grad_to_shape(c.data.clone(), op.data.shape())),
            other => other,
        },
        // Scalar operand: collapse any tensor contribution to its scalar sum.
        _ => match contrib {
            Value::Tensor(t) => Value::Float(t.data.iter().sum(), FW::F64),
            other => other,
        },
    }
}

/// Reduce a gradient tensor to `target` shape using the standard reverse-mode
/// broadcasting rule: sum away leading axes the operand didn't have, then sum
/// (keeping the axis) over any axis the operand had as length 1. A no-op when
/// `c` already has the target shape.
fn reduce_grad_to_shape(mut c: ArrayD<f64>, target: &[usize]) -> ArrayD<f64> {
    if c.shape() == target {
        return c;
    }
    // 1. Collapse leading axes that the operand does not have (e.g. `[B, H]` → `[H]`).
    while c.ndim() > target.len() {
        c = c.sum_axis(Axis(0));
    }
    // 2. Sum (keepdim) over axes where the operand had extent 1 but the
    //    contribution does not (e.g. `[1, H]` broadcast across rows).
    for ax in 0..target.len() {
        if target[ax] == 1 && c.shape()[ax] != 1 {
            c = c.sum_axis(Axis(ax)).insert_axis(Axis(ax));
        }
    }
    c
}

fn negate_value(v: &Value) -> EvalResult<Value> {
    match v {
        Value::Tensor(t) => Ok(Value::tensor(t.mapv(|x| -x))),
        Value::Float(x, w)  => Ok(Value::Float(w.round(-x), *w)),
        Value::Int(n, _)    => Ok(Value::Int(-n, IW::I64)),
        other => Err(RuntimeError::msg(format!("cannot negate {}", other.type_name()))),
    }
}

fn scale_value(v: &Value, s: f64) -> EvalResult<Value> {
    match v {
        Value::Tensor(t) => Ok(Value::tensor(t.mapv(|x| x * s))),
        Value::Float(x, _)  => Ok(Value::Float(x * s, FW::F64)),
        Value::Int(n, _)    => Ok(Value::Float(*n as f64 * s, FW::F64)),
        other => Err(RuntimeError::msg(format!("cannot scale {}", other.type_name()))),
    }
}

fn transpose_last_two(t: ArrayD<f64>) -> ArrayD<f64> {
    let n = t.ndim();
    if n < 2 { return t; }
    let mut axes: Vec<usize> = (0..n).collect();
    axes.swap(n - 1, n - 2);
    t.permuted_axes(IxDyn(&axes)).as_standard_layout().to_owned()
}

// ─── JSON encode/decode ───────────────────────────────────────────────────────

pub(crate) fn json_encode_value(v: &Value) -> String {
    match v {
        Value::Nil       => "null".to_string(),
        Value::Bool(b)   => if *b { "true".to_string() } else { "false".to_string() },
        // #597: encode a `u64`'s unsigned magnitude, not its signed pattern —
        // see the identical carve-out in `Value`'s `Debug` impl.
        Value::Int(n, IW::U64) => (*n as u64).to_string(),
        Value::Int(n, _)    => n.to_string(),
        Value::Float(x, _)  => {
            // Produce a compact representation; avoid trailing .0 for whole numbers
            // that the JSON spec allows, but be explicit enough to round-trip.
            if x.fract() == 0.0 && x.abs() < 1e15 {
                format!("{}", *x as i64)
            } else {
                format!("{}", x)
            }
        }
        Value::Str(s)    => json_encode_str(s),
        Value::List(xs)  => {
            let parts: Vec<String> = xs.iter().map(json_encode_value).collect();
            format!("[{}]", parts.join(","))
        }
        Value::Map(m)    => {
            let borrowed = m.borrow();
            let mut pairs: Vec<(&String, &Value)> = borrowed.iter().collect();
            pairs.sort_by_key(|(k, _)| k.as_str());
            let parts: Vec<String> = pairs.iter()
                .map(|(k, v)| format!("{}:{}", json_encode_str(k), json_encode_value(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Tuple(vs) => {
            let parts: Vec<String> = vs.iter().map(json_encode_value).collect();
            format!("[{}]", parts.join(","))
        }
        // #674: no error
        // channel here either — a stale tensor falls to the SAME "null" the
        // fallback arm already uses for every unencodable value, rather than
        // encoding stale bytes as a number.
        Value::Tensor(t) if tensor_is_stale(t) => "null".to_string(),
        Value::Tensor(t) => {
            // Serialize as a flat JSON array of its elements (row-major). Reuses
            // the scalar float formatting so 1.5 -> "1.5", 2.0 -> "2" (#190).
            let parts: Vec<String> = t.iter().map(|x| json_encode_value(&Value::Float(*x, FW::F64))).collect();
            format!("[{}]", parts.join(","))
        }
        // Fns, structs, ranges, etc. — fall back to null per spec.
        _ => "null".to_string(),
    }
}

pub(crate) fn json_encode_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ── Minimal recursive-descent JSON parser ────────────────────────────────────

struct JsonParser<'a> {
    src: &'a [u8],
    pos: usize,
    /// Current nesting depth, bounded by `MAX_JSON_DEPTH` so adversarial input
    /// can't overflow the native stack (the call-depth guard only counts user
    /// functions, not these native parser frames) — #213.
    depth: usize,
}

/// Maximum JSON nesting depth before `json_decode` returns a recoverable parse
/// error instead of overflowing the stack. Generous vs real data (serde_json
/// defaults to 128) yet far below the native-stack overflow threshold.
const MAX_JSON_DEPTH: usize = 512;

/// How a diagnostic names a byte it found in the input.
///
/// An ASCII byte prints as itself, which is what every existing parser message
/// says. A byte at or above 0x80 does not: on its own it is a fragment of a
/// UTF-8 sequence, not a character, and `b as char` printed it as the latin-1
/// letter it is not — the same misreading as #509, one layer up in the
/// diagnostics. Show the byte.
fn json_byte_desc(b: u8) -> String {
    if b < 0x80 { format!("'{}'", b as char) } else { format!("0x{:02X}", b) }
}

impl<'a> JsonParser<'a> {
    fn new(src: &'a [u8]) -> Self { Self { src, pos: 0, depth: 0 } }

    fn skip_ws(&mut self) {
        while self.pos < self.src.len() && matches!(self.src[self.pos], b' ' | b'\t' | b'\n' | b'\r') {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> { self.src.get(self.pos).copied() }

    fn consume(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied();
        if b.is_some() { self.pos += 1; }
        b
    }

    fn expect(&mut self, b: u8) -> Result<(), String> {
        match self.consume() {
            Some(got) if got == b => Ok(()),
            Some(got) => Err(format!("expected '{}', got {}", b as char, json_byte_desc(got))),
            None => Err(format!("expected '{}', got EOF", b as char)),
        }
    }

    fn parse_value(&mut self) -> Result<Value, String> {
        // Depth guard (#213): arrays/objects recurse back into parse_value per
        // element, so bound it here to return a recoverable parse error instead
        // of overflowing the native stack on deeply nested input.
        self.depth += 1;
        if self.depth > MAX_JSON_DEPTH {
            self.depth -= 1;
            return Err(format!("nesting too deep (max {})", MAX_JSON_DEPTH));
        }
        let r = self.parse_value_inner();
        self.depth -= 1;
        r
    }

    fn parse_value_inner(&mut self) -> Result<Value, String> {
        self.skip_ws();
        match self.peek() {
            Some(b'"')  => self.parse_string().map(Value::Str),
            Some(b'{')  => self.parse_object(),
            Some(b'[')  => self.parse_array(),
            Some(b't')  => { self.consume_literal(b"true")?;  Ok(Value::Bool(true))  }
            Some(b'f')  => { self.consume_literal(b"false")?; Ok(Value::Bool(false)) }
            Some(b'n')  => { self.consume_literal(b"null")?;  Ok(Value::Nil) }
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(c) => Err(format!("unexpected byte {} at position {}",
                json_byte_desc(c), self.pos)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn consume_literal(&mut self, expected: &[u8]) -> Result<(), String> {
        for &b in expected {
            match self.consume() {
                Some(got) if got == b => {}
                Some(got) => return Err(format!("expected '{}', got {}",
                    b as char, json_byte_desc(got))),
                None => return Err("unexpected EOF in literal".to_string()),
            }
        }
        Ok(())
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            match self.consume() {
                Some(b'"')  => break,
                Some(b'\\') => {
                    match self.consume() {
                        Some(b'"')  => s.push('"'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'/')  => s.push('/'),
                        Some(b'n')  => s.push('\n'),
                        Some(b'r')  => s.push('\r'),
                        Some(b't')  => s.push('\t'),
                        Some(b'b')  => s.push('\x08'),
                        Some(b'f')  => s.push('\x0C'),
                        Some(b'u')  => {
                            let cp = self.read_hex4()?;
                            s.push(self.escaped_scalar(cp)?);
                        }
                        Some(c) if c < 0x80 =>
                            return Err(format!("unknown escape \\{}", c as char)),
                        Some(c) => return Err(format!("unknown escape \\{}", json_byte_desc(c))),
                        None => return Err("truncated escape".to_string()),
                    }
                }
                Some(b) if b < 0x80 => s.push(b as char),
                // A raw byte at or above 0x80 opens a multi-byte sequence:
                // JSON text is UTF-8 by definition, so decode the sequence
                // instead of pushing the byte as a char. Pushing it read the
                // text as latin-1 and turned every unescaped non-ASCII scalar
                // into mojibake — an em dash arrived as `â\u{80}\u{94}` (#509).
                // Same shape as the lexer's string-literal path.
                Some(b) => {
                    let mut buf = vec![b];
                    while let Some(next) = self.peek() {
                        if next & 0xC0 == 0x80 { buf.push(next); self.pos += 1; } else { break; }
                    }
                    match std::str::from_utf8(&buf) {
                        Ok(text) => s.push_str(text),
                        Err(_) => return Err("invalid UTF-8 in string".to_string()),
                    }
                }
                None => return Err("unterminated string".to_string()),
            }
        }
        Ok(s)
    }

    /// The four hex digits of a `\uXXXX` escape, `\u` already consumed. Shared
    /// by a lone escape and by each half of a surrogate pair.
    fn read_hex4(&mut self) -> Result<u32, String> {
        let mut hex = String::new();
        for _ in 0..4 {
            match self.consume() {
                Some(b) if b < 0x80 => hex.push(b as char),
                // Not a hex digit, and not a character either — naming it as
                // one would echo the same latin-1 misreading back at the user.
                Some(b) => return Err(format!(
                    "invalid \\u escape: {} is not a hex digit", json_byte_desc(b))),
                None => return Err("truncated \\uXXXX".to_string()),
            }
        }
        u32::from_str_radix(&hex, 16).map_err(|_| format!("invalid \\u{}", hex))
    }

    /// The scalar value a `\uXXXX` escape denotes, joining a surrogate pair
    /// when it opens one.
    ///
    /// A code point past the BMP has no four-digit escape; JSON spells it as a
    /// UTF-16 surrogate pair, and that is what every ASCII-only writer emits —
    /// including python's `json.dumps`, the producer on the other side of the
    /// process port. Reading the halves separately rejects valid JSON, since
    /// neither half is a Unicode scalar value on its own. A surrogate that is
    /// not part of a pair stays an error for the same reason (#509).
    fn escaped_scalar(&mut self, cp: u32) -> Result<char, String> {
        let scalar = match cp {
            0xD800..=0xDBFF => {
                if self.peek() != Some(b'\\')
                    || self.src.get(self.pos + 1).copied() != Some(b'u')
                {
                    return Err(format!("unpaired high surrogate U+{:04X}", cp));
                }
                self.pos += 2;
                let lo = self.read_hex4()?;
                if !(0xDC00..=0xDFFF).contains(&lo) {
                    return Err(format!(
                        "expected low surrogate after U+{:04X}, got U+{:04X}", cp, lo));
                }
                0x1_0000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
            }
            0xDC00..=0xDFFF => return Err(format!("unpaired low surrogate U+{:04X}", cp)),
            _ => cp,
        };
        char::from_u32(scalar).ok_or_else(|| format!("invalid codepoint U+{:04X}", scalar))
    }

    fn parse_number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        let mut is_float = false;
        // optional leading minus
        if self.peek() == Some(b'-') { self.pos += 1; }
        // integer part
        while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        // fractional part
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        }
        // exponent part
        if matches!(self.peek(), Some(b'e') | Some(b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+') | Some(b'-')) { self.pos += 1; }
            while matches!(self.peek(), Some(b'0'..=b'9')) { self.pos += 1; }
        }
        let s = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| "invalid UTF-8 in number".to_string())?;
        if is_float {
            let x: f64 = s.parse().map_err(|_| format!("invalid float: {}", s))?;
            Ok(Value::Float(x, FW::F64))
        } else {
            // A fraction-less token is a JSON integer, so it lands in `i64`
            // whenever it fits. When it does not, it is still a well-formed
            // JSON number and must not become a parse error — the canonical
            // writer prints a whole float without its fraction (PORTS.md §2),
            // so every whole-valued float past 2^63 arrives here looking like
            // a very long integer. Widen those to f64: `json_decode_f64` then
            // succeeds, and `json_decode_i64` reports an honest `decode-type:
            // expected i64, got f64` instead of claiming valid JSON is not
            // JSON (PORTS.md §3.1, §6).
            match s.parse::<i64>() {
                Ok(n) => Ok(Value::Int(n, IW::I64)),
                Err(_) => {
                    let x: f64 = s.parse()
                        .map_err(|_| format!("invalid integer: {}", s))?;
                    Ok(Value::Float(x, FW::F64))
                }
            }
        }
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.expect(b'[')?;
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') { self.consume(); return Ok(Value::list(items)); }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.consume() {
                Some(b']') => break,
                Some(b',') => {}
                Some(c) => return Err(format!("expected ',' or ']', got {}", json_byte_desc(c))),
                None => return Err("unterminated array".to_string()),
            }
        }
        Ok(Value::list(items))
    }

    fn parse_object(&mut self) -> Result<Value, String> {
        self.expect(b'{')?;
        self.skip_ws();
        let mut m = HashMap::new();
        if self.peek() == Some(b'}') { self.consume(); return Ok(Value::Map(Rc::new(RefCell::new(m)))); }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let val = self.parse_value()?;
            m.insert(key, val);
            self.skip_ws();
            match self.consume() {
                Some(b'}') => break,
                Some(b',') => {}
                Some(c) => return Err(format!("expected ',' or '}}', got {}", json_byte_desc(c))),
                None => return Err("unterminated object".to_string()),
            }
        }
        Ok(Value::Map(Rc::new(RefCell::new(m))))
    }
}

pub(crate) fn json_parse(s: &str) -> Result<Value, String> {
    json_parse_bytes(s.as_bytes())
}

/// The parser's real entry point: JSON is a byte format, and `parse_string`
/// decodes UTF-8 sequences out of these bytes itself.
///
/// Every caller in the language reaches it through `json_parse`, whose `&str`
/// is already valid UTF-8 by construction — a demoniC `str` cannot hold an
/// ill-formed sequence, and the process port's `read_line` rejects one before
/// the text gets here. So the invalid-UTF-8 branch is unreachable from the
/// language surface and is a guard, not a live path; this entry point is what
/// lets a test aim at it directly.
pub(crate) fn json_parse_bytes(src: &[u8]) -> Result<Value, String> {
    let mut parser = JsonParser::new(src);
    let v = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos < parser.src.len() {
        return Err(format!("trailing content at position {}", parser.pos));
    }
    Ok(v)
}

// ─── Typed JSON decode ────────────────────────────────────────────────────────

/// The demoniC types the typed-decode family targets. Kept next to the
/// primitives so `assimilate` and the interpreter cannot drift on the set:
/// `assimilate` reads it to decide whether a descriptor's `ret` is decodable.
pub(crate) const DECODE_TYPES: &[&str] = &["i64", "f64", "str", "bool", "list"];

/// The decode-target name for a decoded JSON value, so a `decode-type` tag
/// speaks one vocabulary on both sides of "expected X, got Y".
pub(crate) fn decode_kind(v: &Value) -> &'static str {
    match v {
        Value::Int(_, _)      => "i64",
        Value::Float(_, _) => "f64",
        Value::Str(_)      => "str",
        Value::Bool(_)     => "bool",
        Value::List(_)     => "list",
        Value::Map(_)      => "map",
        Value::Nil         => "nil",
        other              => other.type_name(),
    }
}

/// `T`'s zero, returned beside a non-nil Err so `(T, Err)` stays well-typed on
/// the failure path — the same discipline a hand-written wrapper follows.
fn decode_zero(want: &str) -> Value {
    match want {
        "i64"  => Value::Int(0, IW::I64),
        "f64"  => Value::Float(0.0, FW::F64),
        "bool" => Value::Bool(false),
        "list" => Value::List(Rc::new(Vec::new())),
        _      => Value::Str(String::new()),
    }
}

/// Decode `src` as JSON and require it to be `want`. Returns the `(T, Err)`
/// tuple the `json_decode_<T>` builtins hand back.
///
/// Two tags, both documented in PORTS.md §6 and distinct from the `port-*`
/// family so a descriptor that lies about a type is never confused with a
/// runtime that failed:
///
/// - `decode-parse` — the text is not JSON at all.
/// - `decode-type`  — the text is JSON of another kind. Never coerced.
///
/// Numbers are the one place the check cannot be exact, and the reason is
/// JSON's: it has a single number type, and the canonical writer prints a
/// whole float without its fraction (`2.0` → `2`, PORTS.md §2). A whole-valued
/// float and an integer are the same bytes here, so the widening runs both ways
/// *for whole values* — an integer always satisfies `f64` (refusing it would
/// fail on every whole float that crosses a port), and a whole float always
/// satisfies `i64` because the difference was erased upstream of this function.
/// What the `i64` direction does catch is every token that stayed distinct: a
/// str, a bool, a null, a list, a map, and every non-whole number. `2.5` into
/// `i64` is a `decode-type` — truncation is exactly the silent coercion this
/// family exists to prevent. PORTS.md §3.1 documents the gap in full.
pub(crate) fn json_decode_typed(want: &str, src: &str) -> Value {
    let parsed = match json_parse(src) {
        Ok(v) => v,
        Err(e) => return Value::Tuple(vec![
            decode_zero(want), Value::Str(format!("decode-parse: {}", e))]),
    };
    let got = decode_kind(&parsed);
    let value = match (want, parsed) {
        ("f64", Value::Int(n, _)) => Value::Float(n as f64, FW::F64),
        (w, v) if w == got => v,
        _ => return Value::Tuple(vec![decode_zero(want), Value::Str(
            format!("decode-type: expected {}, got {}", want, got))]),
    };
    Value::Tuple(vec![value, Value::Nil])
}

// ─── List combinator helpers ──────────────────────────────────────────────────

/// Total order over Values for list_sort / list_min / list_max / list_sort_by.
///
/// Type rank (for mixed-type lists): Bool < Int < Float < Str < everything else.
/// Within a type:
///   - Bool:  false < true
///   - Int:   exact i64 comparison
///   - Float: total order via f64::total_cmp (NaN sorts after all finite values)
///   - Str:   full lexicographic byte comparison
///   - other: treated as equal (stable relative order preserved by sort_by)
fn cmp_value(a: &Value, b: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    fn type_rank(v: &Value) -> u8 {
        match v {
            Value::Bool(_)  => 0,
            Value::Int(_, _)   => 1,
            Value::Float(_, _) => 2,
            Value::Str(_)   => 3,
            _               => 4,
        }
    }
    let ra = type_rank(a);
    let rb = type_rank(b);
    if ra != rb {
        return ra.cmp(&rb);
    }
    match (a, b) {
        (Value::Bool(x),  Value::Bool(y))  => x.cmp(y),
        (Value::Int(x, _),   Value::Int(y, _))   => x.cmp(y),
        (Value::Float(x, _), Value::Float(y, _)) => x.total_cmp(y),
        (Value::Str(x),   Value::Str(y))   => x.cmp(y),
        _                                  => Ordering::Equal,
    }
}

/// Compare two Values for equality (for list_uniq).
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x, _), Value::Int(y, _)) => x == y,
        (Value::Float(x, _), Value::Float(y, _)) => x == y,
        (Value::Int(x, _), Value::Float(y, _)) => (*x as f64) == *y,
        (Value::Float(x, _), Value::Int(y, _)) => *x == (*y as f64),
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Nil, Value::Nil) => true,
        _ => false,
    }
}

/// Convert a Value to a string for the `format` builtin (same as print does).
/// Format template: `{}` plain substitution; `{:spec}` with printf-like spec.
/// Supported specs: `.Nf` (float precision), `d` (integer), `s` (string),
/// `Nd` / `Ns` (width right-align), `0Nd` (zero-pad), `x`/`b`/`o` (int bases),
/// `e`/`E` (scientific), `+d` / `+f` (force sign).
fn apply_format_template(template: &str, args: &[Value]) -> Result<String, String> {
    let mut result = String::new();
    let mut arg_idx = 0usize;
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '}' {
            // `}}` → literal `}`
            if chars.peek() == Some(&'}') { chars.next(); }
            result.push('}');
            continue;
        }
        if ch != '{' {
            result.push(ch);
            continue;
        }
        // Peek: could be `{{` (escaped brace)
        if chars.peek() == Some(&'{') {
            chars.next();
            result.push('{');
            continue;
        }
        // Collect everything up to `}`
        let mut spec = String::new();
        let mut closed = false;
        for sc in chars.by_ref() {
            if sc == '}' { closed = true; break; }
            spec.push(sc);
        }
        if !closed {
            return Err("format: unclosed '{'".to_string());
        }
        let val = args.get(arg_idx).cloned().unwrap_or(Value::Nil);
        arg_idx += 1;

        // spec may be empty (`{}`) or have a format after `:`
        let fmt_spec = if spec.starts_with(':') { &spec[1..] } else { "" };
        if fmt_spec.is_empty() {
            result.push_str(&format!("{}", val));
            continue;
        }

        let formatted = format_value_with_spec(&val, fmt_spec)?;
        result.push_str(&formatted);
    }
    Ok(result)
}

fn format_value_with_spec(val: &Value, spec: &str) -> Result<String, String> {
    // Detect sign prefix
    let (force_sign, spec) = if spec.starts_with('+') { (true, &spec[1..]) } else { (false, spec) };

    // Detect fill/align: `<`, `>`, `^` with optional fill char (default space)
    // Also detect zero-pad: starts with `0` digit followed by digits and type
    let zero_pad = spec.starts_with('0') && spec.len() > 1 && spec.chars().nth(1).map(|c| c.is_ascii_digit()).unwrap_or(false);

    // Extract optional width and precision
    // Patterns: `Nd`, `Nf`, `.Nf`, `N.Mf`, `x`, `b`, `o`, `e`, `E`, `s`
    let type_char = spec.chars().last().unwrap_or('s');
    // #212: slice off the type char by its byte offset (via char_indices), not
    // `spec.len()-1`. The old byte index underflowed on an empty spec (e.g. the
    // `{:+}` sign-only case, where the `+` is already stripped above) and split a
    // UTF-8 sequence when the last char was multibyte — both panicked the process.
    let mid = match spec.char_indices().last() {
        Some((byte_off, _)) => &spec[..byte_off], // everything before the type char
        None => "",                               // empty spec → no width/precision
    };

    let (width_str, prec_str) = if let Some(dot) = mid.find('.') {
        (&mid[..dot], Some(&mid[dot+1..]))
    } else {
        (mid, None)
    };

    let width_str = if zero_pad { &width_str[..] } else { width_str.trim_start_matches('0') };
    let width: Option<usize> = if width_str.is_empty() { None }
        else { width_str.trim_start_matches('0').parse().ok().or_else(|| width_str.parse().ok()) };
    let prec: Option<usize> = prec_str.and_then(|p| p.parse().ok());

    let x = val.as_float().unwrap_or(0.0);
    let n = val.as_int().unwrap_or(x as i64);
    let sign = if force_sign && x >= 0.0 { "+" } else { "" };

    let raw = match type_char {
        'f' => {
            let p = prec.unwrap_or(6);
            format!("{}{:.prec$}", sign, x, prec = p)
        }
        'd' | 'i' => {
            format!("{}{}", sign, n)
        }
        's' => format!("{}", val),
        'x' => format!("{:x}", n),
        'X' => format!("{:X}", n),
        'b' => format!("{:b}", n),
        'o' => format!("{:o}", n),
        'e' => {
            let p = prec.unwrap_or(6);
            format!("{}{:.prec$e}", sign, x, prec = p)
        }
        'E' => {
            let p = prec.unwrap_or(6);
            format!("{}{:.prec$E}", sign, x, prec = p)
        }
        _ => format!("{}", val),
    };

    // Apply width padding
    if let Some(w) = width {
        if zero_pad && matches!(type_char, 'd'|'i'|'f'|'x'|'X'|'b'|'o') {
            // Zero-pad: insert zeros after sign
            let pad_needed = w.saturating_sub(raw.len());
            let zeros = "0".repeat(pad_needed);
            let result = if raw.starts_with('-') || raw.starts_with('+') {
                format!("{}{}{}", &raw[..1], zeros, &raw[1..])
            } else {
                format!("{}{}", zeros, raw)
            };
            Ok(result)
        } else if raw.len() < w {
            Ok(format!("{:>width$}", raw, width = w))
        } else {
            Ok(raw)
        }
    } else {
        Ok(raw)
    }
}

// ─── Hash functions ───────────────────────────────────────────────────────────

fn fnv1a_64(s: &str) -> i64 {
    let mut hash: u64 = 14695981039346656037;
    for byte in s.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash as i64
}

// CRC32 IEEE polynomial table
static CRC32_TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();

fn get_crc32_table() -> &'static [u32; 256] {
    CRC32_TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for n in 0..256u32 {
            let mut c = n;
            for _ in 0..8 {
                if c & 1 != 0 {
                    c = 0xEDB88320 ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            table[n as usize] = c;
        }
        table
    })
}

fn crc32_ieee(s: &str) -> i64 {
    let table = get_crc32_table();
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in s.bytes() {
        let idx = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    (!crc) as i64
}

// ─── Hex helpers for gzip compression ────────────────────────────────────────

fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

fn hex_to_bytes(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 { return None; }
    (0..s.len()).step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i+2], 16).ok())
        .collect()
}
