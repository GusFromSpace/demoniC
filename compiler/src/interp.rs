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
    /// error — it carries the `(T, Err)` tuple that the enclosing function
    /// should return, and is intercepted at the nearest function-call
    /// boundary (Rust-`?`-style early return). `None` for ordinary errors.
    pub propagate: Option<Box<Value>>,
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
    pub fn msg(s: impl Into<String>) -> Self { Self { msg: s.into(), span: None, propagate: None } }
    pub fn at(s: impl Into<String>, sp: Span) -> Self { Self { msg: s.into(), span: Some(sp), propagate: None } }
    /// `?`-operator early-return signal carrying the `(T, Err)` tuple to return.
    fn propagate(value: Value) -> Self {
        Self { msg: "`?` propagated past a non-(T, Err) function".into(), span: None, propagate: Some(Box::new(value)) }
    }
}

pub type EvalResult<T> = Result<T, RuntimeError>;

// ─── Value ───────────────────────────────────────────────────────────────────

/// Coarse runtime dtype tag for tensors. The interpreter stores all tensor
/// data as `f64` (pre-alpha pragma), but it must remember whether a tensor
/// holds *integers* so that a scalar element read returns `Value::Int` rather
/// than `Value::Float`. Without this, integer division, `%`, bitwise ops and
/// range/loop bounds sourced from a tensor element silently misbehave (#125).
///
/// `Int` covers the whole signed/unsigned integer family — that's all the
/// element read needs, since `Value::Int` is `i64`.
///
/// #241: floats track their *width*, because the two widths have different
/// JIT-parity semantics. `F32` covers f32 and the narrower float-likes
/// (f16/bf16/tf32/fp8) that the JIT computes as f32 (the cast-no-op
/// convention, #179): the data stays f64-backed but is rounded through f32
/// whenever a tensor is produced or an element is stored, so `dmc run`
/// matches `dmc jit` numerics by default. `F64` keeps full f64, which is what
/// the JIT does for declared-f64 tensors. `F32` is the default float tag —
/// tensor literals, op results, and rng constructors are f32 on the JIT too.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType { Int, F32, F64, Trit }

/// Round a scalar through f32 (the F32-tensor store semantics).
#[inline]
fn quantize_f32(x: f64) -> f64 {
    x as f32 as f64
}

/// Round every element of a tensor's data through f32.
#[inline]
fn quantize_f32_arr(data: &mut ArrayD<f64>) {
    data.mapv_inplace(|x| x as f32 as f64);
}

/// A tensor value: f64-backed storage plus a coarse dtype tag. Derefs to the
/// underlying `ArrayD<f64>` so the bulk of the interpreter (shape queries,
/// iteration, elementwise math) operates on it unchanged.
#[derive(Clone)]
pub struct TensorVal {
    pub data: ArrayD<f64>,
    pub dtype: DType,
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
        if matches!(dtype, DType::F32) {
            quantize_f32_arr(&mut data);
        }
        TensorVal { data, dtype }
    }
    pub fn is_int(&self) -> bool { self.dtype == DType::Int }
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
    Int(i64),
    Float(f64),
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
    List(Vec<Value>),
    /// Dynamic map — string-keyed heterogeneous collection.
    /// Wrapped in Rc<RefCell<...>> so all clones share the same backing
    /// HashMap — mutations (map_set, map_del) are visible through every copy,
    /// matching the JIT's pointer/reference semantics.
    Map(Rc<RefCell<HashMap<String, Value>>>),
    /// Module reference
    Module { alias: String, path: std::path::PathBuf },
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n)   => write!(f, "{}", n),
            Value::Float(x) => write!(f, "{}", x),
            Value::Bool(b)  => write!(f, "{}", b),
            Value::Str(s)   => write!(f, "{:?}", s),
            Value::Nil      => write!(f, "nil"),
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
                    write!(f, "{}: {:?}", k, v)?;
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
            Value::Fn(n)      => write!(f, "<fn {}>", n),
            Value::BoundFn { name, .. } => write!(f, "<fn {}>", name),
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
            Value::Opaque(n)  => write!(f, "<opaque {}>", n),
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

impl Value {
    /// Construct a `Float`-tagged tensor value (the common case).
    fn tensor(data: ArrayD<f64>) -> Value { Value::Tensor(TensorVal::new(data, DType::F32)) }
    /// Construct a tensor value with an explicit dtype tag.
    fn tensor_dt(data: ArrayD<f64>, dtype: DType) -> Value { Value::Tensor(TensorVal::new(data, dtype)) }
    fn as_int(&self) -> Option<i64> {
        match self { Value::Int(n) => Some(*n), _ => None }
    }
    fn as_float(&self) -> Option<f64> {
        match self { Value::Int(n) => Some(*n as f64), Value::Float(x) => Some(*x), _ => None }
    }
    fn as_bool(&self) -> Option<bool> {
        match self { Value::Bool(b) => Some(*b), _ => None }
    }
    fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_)     => "int",
            Value::Float(_)   => "float",
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
        }
    }
}

// ─── Control-flow signals ────────────────────────────────────────────────────

/// Statements can produce control-flow signals — `break`, `continue`,
/// `return`. We thread them up through eval_block.
enum Flow {
    Normal(Value),
    Break,
    Continue,
    Return(Value),
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
    pub models: std::collections::HashSet<String>,
    pub model_methods: HashMap<String, HashMap<String, FnDecl>>,
    pub public_items: std::collections::HashSet<String>,
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

pub struct Interpreter {
    /// Lexical scope stack: each scope a name→value map.
    scopes: Vec<HashMap<String, Value>>,
    /// Lexical scope stack for declared streaming axes on KV-like bindings.
    stream_axes: Vec<HashMap<String, usize>>,
    /// Declared capacity (max streaming-axis length) for each KV binding.
    /// Populated from the `capacity = N` arg in `stream.kv[...](capacity = N)`.
    kv_capacities: HashMap<String, usize>,
    /// Top-level fn declarations (collected from program before any eval).
    fns: HashMap<String, FnDecl>,
    /// Top-level model declarations — referenced as opaque values at
    /// runtime (e.g. `Transformer[L=24, ...]`), so we just need the name set.
    models: std::collections::HashSet<String>,
    /// Shape parameter names for each model, used to bind them into method scopes.
    /// e.g. `model S[P]` → `model_shape_params["S"] = ["P"]`
    model_shape_params: HashMap<String, Vec<String>>,
    /// Methods for each model, keyed by model name then method name.
    model_methods: HashMap<String, HashMap<String, FnDecl>>,
    /// Field declarations (name + type, in declaration order) for each model.
    /// Used by `@cast(Model){bytes}` to overlay a raw byte tensor onto a
    /// struct: it needs the field types to compute byte offsets and sizes.
    model_fields: HashMap<String, Vec<(String, crate::ast::Type)>>,
    /// #336: top-level enum declarations — name → ordered variant names. A
    /// variant's value is its index here (the i64 ordinal).
    enums: HashMap<String, Vec<String>>,
    /// xorshift64 RNG state, advanced by every `rng.normal[...]` /
    /// `rng.uniform[...]`. Seeded from `Rng.seed(N)` at the call site; this
    /// global is the fallback so programs that never call seed still get a
    /// reproducible (per-process) random stream rather than zeros.
    rng_state: u64,
    /// Monotonic start time for `time_ms()`.
    start_time: std::time::Instant,
    /// Command-line arguments passed to the program (for `argv()`).
    argv: Vec<String>,
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
    /// Current recursive-call depth, bumped via `CallDepthGuard` on every user-fn /
    /// lambda entry. Trips `MAX_CALL_DEPTH` with a catchable error before the native
    /// stack is exhausted. `Rc<Cell<…>>` so the guard can decrement on drop without
    /// holding a borrow of the interpreter across the call body.
    call_depth: Rc<Cell<usize>>,
}

struct AxisSelection {
    indices: Vec<usize>,
    reduce_rank: bool,
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
    fn resolve_index_selection(&mut self, arr_shape: &[usize], elems: &[IndexElem], span: Span) -> EvalResult<Vec<AxisSelection>> {
        let ndim = arr_shape.len();
        let mut selections = Vec::new();

        let mut padded_elems = elems.to_vec();
        while padded_elems.len() < ndim {
            padded_elems.push(IndexElem::FullSlice(span.clone()));
        }

        if padded_elems.len() > ndim {
            return Err(RuntimeError::at(
                format!("index has {} dims but tensor has {}", padded_elems.len(), ndim),
                span,
            ));
        }

        for (axis, elem) in padded_elems.iter().enumerate() {
            let dim = arr_shape[axis];
            match elem {
                IndexElem::Expr(Expr::Range { start, end, inclusive, .. }) => {
                    let start_val = if let Some(start_expr) = start {
                        let val = self.eval_expr(start_expr)?;
                        Some(val.as_int().ok_or_else(|| {
                            RuntimeError::at(format!("range start must be integer, got {}", val.type_name()), span.clone())
                        })?)
                    } else {
                        None
                    };

                    let end_val = if let Some(end_expr) = end {
                        let val = self.eval_expr(end_expr)?;
                        Some(val.as_int().ok_or_else(|| {
                            RuntimeError::at(format!("range end must be integer, got {}", val.type_name()), span.clone())
                        })?)
                    } else {
                        None
                    };

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
                IndexElem::Expr(e) => {
                    let val = self.eval_expr(e)?;
                    let n = val.as_int().ok_or_else(|| {
                        RuntimeError::at(format!("tensor index must be integer, got {}", val.type_name()), span.clone())
                    })?;
                    let i = if n < 0 { (dim as i64 + n) as usize } else { n as usize };
                    if i >= dim {
                        return Err(RuntimeError::at(format!("index {} out of bounds for axis {} of size {}", i, axis, dim), span.clone()));
                    }
                    let is_slice = |el: &IndexElem| -> bool {
                        matches!(el, IndexElem::FullSlice(_) | IndexElem::Slice { .. })
                            || matches!(el, IndexElem::Expr(Expr::Range { .. }))
                    };
                    let has_before = padded_elems[..axis].iter().any(is_slice);
                    let has_after = padded_elems[axis + 1..].iter().any(is_slice);
                    let keepdims = has_before && has_after;

                    selections.push(AxisSelection {
                        indices: vec![i],
                        reduce_rank: !keepdims,
                    });
                }
                IndexElem::FullSlice(_) => {
                    selections.push(AxisSelection {
                        indices: (0..dim).collect(),
                        reduce_rank: false,
                    });
                }
                IndexElem::Slice { start, end, step, .. } => {
                    let step_val = if let Some(step_expr) = step {
                        let val = self.eval_expr(step_expr)?;
                        val.as_int().ok_or_else(|| {
                            RuntimeError::at(format!("slice step must be integer, got {}", val.type_name()), span.clone())
                        })?
                    } else {
                        1
                    };
                    if step_val == 0 {
                        return Err(RuntimeError::at("slice step cannot be zero".to_string(), span.clone()));
                    }

                    let start_val = if let Some(start_expr) = start {
                        let val = self.eval_expr(start_expr)?;
                        Some(val.as_int().ok_or_else(|| {
                            RuntimeError::at(format!("slice start must be integer, got {}", val.type_name()), span.clone())
                        })?)
                    } else {
                        None
                    };

                    let end_val = if let Some(end_expr) = end {
                        let val = self.eval_expr(end_expr)?;
                        Some(val.as_int().ok_or_else(|| {
                            RuntimeError::at(format!("slice end must be integer, got {}", val.type_name()), span.clone())
                        })?)
                    } else {
                        None
                    };

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

    fn assign_to_tensor(&mut self, arr: &mut ArrayD<f64>, dtype: DType, idx_elems: &[IndexElem], rval: Value, span: Span) -> EvalResult<()> {
        // #226/#241: element writes mutate the backing array directly
        // (bypassing TensorVal::new), so round the written value through f32
        // here when the target is an F32 tensor. F64/Int/Trit pass through.
        let rval = if matches!(dtype, DType::F32) {
            match rval {
                Value::Float(x) => Value::Float(quantize_f32(x)),
                Value::Tensor(mut t) => { quantize_f32_arr(&mut t.data); Value::Tensor(t) }
                other => other,
            }
        } else { rval };
        let selections = self.resolve_index_selection(arr.shape(), idx_elems, span.clone())?;

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
            Value::Int(n) => {
                assign_scalar_to_selection(arr, &selections, n as f64);
            }
            Value::Float(val) => {
                assign_scalar_to_selection(arr, &selections, val);
            }
            other => return Err(RuntimeError::at(format!(
                "cannot assign value of type {} to tensor slice",
                other.type_name()
            ), span)),
        }
        Ok(())
    }

    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            stream_axes: vec![HashMap::new()],
            kv_capacities: HashMap::new(),
            model_shape_params: HashMap::new(),
            fns: HashMap::new(),
            models: std::collections::HashSet::new(),
            model_methods: HashMap::new(),
            model_fields: HashMap::new(),
            enums: HashMap::new(),
            rng_state: 0x9E3779B97F4A7C15,  // SplitMix64 golden-ratio seed
            start_time: std::time::Instant::now(),
            argv: Vec::new(),
            profile: None,
            interp_modules: HashMap::new(),
            public_items: std::collections::HashSet::new(),
            host_features: detect_host_features(),
            extern_fns: std::collections::HashSet::new(),
            pending_writebacks: Vec::new(),
            call_depth: Rc::new(Cell::new(0)),
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
        // Pass 3: if main() exists, call it; otherwise return Nil
        if self.fns.contains_key("main") {
            let main = self.fns.get("main").unwrap().clone();
            self.call_fn(&main, Vec::new(), main.span.clone())
        } else {
            Ok(Value::Nil)
        }
    }

    pub fn load_program(&mut self, program: &Program, path: Option<&std::path::Path>) -> EvalResult<()> {
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
            model_methods: self.model_methods.clone(),
            public_items: self.public_items.clone(),
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
                        if let Some(alias) = &us.alias {
                            self.bind(alias.clone(), Value::Module { alias: alias.clone(), path: canonical_path });
                            // Qualified imports: register functions, models, methods, variables under alias prefix
                            for (name, f) in imported_env.fns {
                                if !imported_env.public_items.contains(&name) {
                                    continue;
                                }
                                self.fns.insert(format!("{}.{}", alias, name), f);
                            }
                            for name in imported_env.models {
                                if !imported_env.public_items.contains(&name) {
                                    continue;
                                }
                                self.models.insert(format!("{}.{}", alias, name));
                            }
                            for (mname, methods) in imported_env.model_methods {
                                if !imported_env.public_items.contains(&mname) {
                                    continue;
                                }
                                self.model_methods.insert(format!("{}.{}", alias, mname), methods);
                            }
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
                            // Unqualified imports: merge directly
                            for (name, f) in imported_env.fns {
                                if !imported_env.public_items.contains(&name) {
                                    continue;
                                }
                                self.fns.insert(name, f);
                            }
                            for name in imported_env.models {
                                if !imported_env.public_items.contains(&name) {
                                    continue;
                                }
                                self.models.insert(name);
                            }
                            for (mname, methods) in imported_env.model_methods {
                                if !imported_env.public_items.contains(&mname) {
                                    continue;
                                }
                                self.model_methods.insert(mname, methods);
                            }
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

    pub fn call_named_fn(&mut self, name: &str, args: Vec<Value>) -> EvalResult<Value> {
        let f = self.fns.get(name).cloned().ok_or_else(|| {
            RuntimeError::msg(format!("undefined fn `{}`", name))
        })?;
        self.call_fn(&f, args, f.span.clone())
    }

    fn collect_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f)       => { self.fns.insert(f.name.clone(), f.clone()); }
            Item::ExternFn(e) => { self.extern_fns.insert(e.name.clone()); }
            Item::Model(m) => {
                self.models.insert(m.name.clone());
                let shape_param_names: Vec<String> =
                    m.shape_params.iter().map(|sp| sp.name.clone()).collect();
                self.model_shape_params.insert(m.name.clone(), shape_param_names);
                let mut methods: HashMap<String, FnDecl> = HashMap::new();
                let mut fields: Vec<(String, crate::ast::Type)> = Vec::new();
                for member in &m.members {
                    match member {
                        ModelMember::Method(f) => { methods.insert(f.name.clone(), f.clone()); }
                        ModelMember::Field { name, ty, .. } => { fields.push((name.clone(), ty.clone())); }
                    }
                }
                self.model_methods.insert(m.name.clone(), methods);
                self.model_fields.insert(m.name.clone(), fields);
            }
            Item::Enum(e) => {
                let variants: Vec<String> = e.variants.iter().map(|v| v.name.clone()).collect();
                self.enums.insert(e.name.clone(), variants);
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
                    return Err(RuntimeError::at(format!("extern fn `{}` cannot be called without a JIT backend", name), sp));
                }
                let f = self.fns.get(&name).cloned().ok_or_else(|| {
                    RuntimeError::at(format!("undefined fn `{}`", name), sp.clone())
                })?;
                self.call_fn(&f, args, sp)
            }
            Value::BoundFn { ref name, ref shape_bindings } => {
                let f = self.fns.get(name).cloned().ok_or_else(|| {
                    RuntimeError::at(format!("undefined fn `{}`", name), sp.clone())
                })?;
                let bindings = shape_bindings.clone();
                self.call_fn_with_shapes(&f, args, &bindings, sp)
            }
            Value::Lambda { lit, captured_env } => self.call_lambda(&lit.clone(), captured_env, args, sp),
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

    fn call_lambda(&mut self, lit: &crate::ast::FnLit, captured_env: HashMap<String, Value>, args: Vec<Value>, sp: Span) -> EvalResult<Value> {
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
            self.bind(&p.name, v);
        }
        let result = self.eval_block(&lit.body);
        self.pop_scope();
        match result {
            Ok(Flow::Normal(v)) | Ok(Flow::Return(v)) => Ok(v),
            Ok(_) => Ok(Value::Nil),
            // `?` early-return: this lambda returns the propagated tuple.
            Err(e) => match e.propagate { Some(v) => Ok(*v), None => Err(e) },
        }
    }

    // ── Scoping ───────────────────────────────────────────────────────────

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.stream_axes.push(HashMap::new());
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
        self.stream_axes.pop();
    }

    fn bind(&mut self, name: impl Into<String>, v: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), v);
        }
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

    fn lookup_stream_axis(&self, name: &str) -> Option<usize> {
        for scope in self.stream_axes.iter().rev() {
            if let Some(axis) = scope.get(name) { return Some(*axis); }
        }
        None
    }

    fn assign(&mut self, name: &str, v: Value) -> EvalResult<()> {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), v);
                return Ok(());
            }
        }
        Err(RuntimeError::msg(format!("cannot assign to undefined `{}`", name)))
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
    fn call_grad(
        &mut self,
        fn_name: &str,
        method: &str,
        args: Vec<Value>,
        span: Span,
    ) -> EvalResult<Value> {
        let f = self.fns.get(fn_name).cloned().ok_or_else(||
            RuntimeError::at(format!("@grad: undefined fn `{}`", fn_name), span.clone()))?;
        if args.len() != f.params.len() {
            return Err(RuntimeError::at(
                format!("@grad `{}` expects {} args, got {}", fn_name, f.params.len(), args.len()),
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
                Some(&dim) => self.bind(name, Value::Int(dim)),
                None => {
                    self.pop_scope();
                    return Err(RuntimeError::at(format!(
                        "@grad `{}`: shape param `{}` cannot be inferred from args",
                        f.name, name), span));
                }
            }
        }

        let mut tape = Tape::new();
        let mut var_node: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

        // Bind params: tensor params become Input nodes; the `mutating` flag
        // tells the backward pass which ones to return gradients for.
        for (i, (p, v)) in f.params.iter().zip(&args).enumerate() {
            self.bind(&p.name, v.clone());
            if let Value::Tensor(_) = v {
                let node_id = tape.push(TapeNode {
                    op: TapeOp::Input { param_idx: i, mutating: p.mutating },
                    inputs: Vec::new(),
                    value: v.clone(),
                });
                var_node.insert(p.name.clone(), node_id);
            }
        }

        // Forward pass: evaluate the body, recording tensor ops to the tape.
        let result = self.eval_block_with_tape(&f.body, &mut tape, &mut var_node, &span);
        self.pop_scope();
        let (loss_value, loss_node) = result?;

        // The function's tail must produce a scalar (real-valued loss). The
        // tape-recording layer returns the node id only when the tail is a
        // tensor-tracked computation. For scalar losses, find the most recent
        // tracked tensor node (the one whose `Sum`/`ScalarDiv` produced the
        // scalar); the tape's `loss_node` field holds it.
        let loss_node = loss_node.ok_or_else(|| RuntimeError::at(format!(
            "@grad `{}`: function body must produce a scalar derived from a tensor computation \
            (got a value that doesn't participate in the gradient graph).\n\
            hint: indexed reads `x[i]` exit the graph — reduce with a traced reduction:\n\
            \t`sum(x)` or `mean(x)` (e.g. `sum(x .* x)`).\n\
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
            let adj = tape.backward_symbolic(loss_node)?;
            let p0_node = f.params.iter()
                .find(|p| p.mutating)
                .and_then(|p| var_node.get(&p.name).copied())
                .ok_or_else(|| RuntimeError::at(format!(
                    "@grad `{}`: fwd_bwd_bwd needs a `!` tensor parameter",
                    f.name), span.clone()))?;
            let g1 = adj[p0_node].ok_or_else(|| RuntimeError::at(format!(
                "@grad `{}`: first `!` param has no gradient \
                 (loss independent of it?)", f.name), span.clone()))?;
            let loss2 = match &tape.nodes[g1].value {
                Value::Tensor(t) => {
                    let total: f64 = t.iter().sum();
                    tape.push(TapeNode {
                        op: TapeOp::Sum, inputs: vec![g1], value: Value::Float(total),
                    })
                }
                _ => g1,
            };
            grads = vec![None; tape.nodes.len()];
            grads[loss2] = Some(Value::Float(1.0));
        } else {
            grads = vec![None; tape.nodes.len()];
            grads[loss_node] = Some(Value::Float(1.0));
        }
        tape.backward(&mut grads)?;

        // Collect gradients for each `!`-marked tensor parameter.
        let mut grad_struct: Vec<(String, Value)> = Vec::new();
        for (i, p) in f.params.iter().enumerate() {
            if !p.mutating { continue; }
            let Some(node_id) = var_node.get(&p.name) else { continue; };
            let g = grads[*node_id].clone().unwrap_or_else(|| {
                // No gradient flowed back — this parameter doesn't affect
                // the loss. Surface it as zeros of the param shape, since
                // that IS the mathematically correct answer (∂L/∂x=0 when
                // L doesn't depend on x), not a placeholder.
                match &args[i] {
                    Value::Tensor(t) => Value::tensor(ArrayD::zeros(t.raw_dim())),
                    other => other.clone(),
                }
            });
            grad_struct.push((p.name.clone(), g));
        }
        let grad_value = Value::Struct(Rc::new(RefCell::new(grad_struct)));
        match method {
            "grad" => Ok(grad_value),
            "fwd_bwd" | "fwd_bwd_bwd" => Ok(Value::Tuple(vec![loss_value, grad_value])),
            _ => unreachable!(),
        }
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

    /// Push a binary-op node onto the tape, reusing one op→VJP mapping for both
    /// the `BinOp` expression arm and compound reassignment. Returns the forward
    /// value and its node id (None when neither operand is tracked).
    fn tape_binop(
        &self,
        tape: &mut Tape,
        op: &BinOp,
        lv: &Value, ln: Option<usize>,
        rv: &Value, rn: Option<usize>,
        span: &Span,
    ) -> EvalResult<(Value, Option<usize>)> {
        let out = apply_binop(op.clone(), lv, rv)?;
        if ln.is_none() && rn.is_none() {
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
                                    let sum_val = Value::Float(as_tensor(&v)?.iter().sum());
                                    let sum_id = tape.push(TapeNode {
                                        op: TapeOp::Sum, inputs: vec![nid],
                                        value: if fname == "sum" { out.clone() } else { sum_val },
                                    });
                                    if fname == "mean" {
                                        let count = as_tensor(&v)?.len() as f64;
                                        let cnode = tape.push_const(Value::Float(count));
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
                    // sum_along / mean_along (axis dropped) differentiate through
                    // the tape (#307): the adjoint broadcasts back along the axis.
                    if let Expr::Ident(fname, _) = expr.as_ref() {
                        if (fname == "sum_along" || fname == "mean_along") && args.len() == 2 {
                            if let (CallArg::Positional(xe), CallArg::Positional(ae)) = (&args[0], &args[1]) {
                                let (v, n) = self.eval_expr_with_tape(xe, tape, var_node, span)?;
                                let axis_lit = self.eval_expr(ae)?.as_int().unwrap_or(0);
                                let out = self.call_builtin(fname, vec![v.clone(), Value::Int(axis_lit)], psp.clone())?;
                                if let Some(nid) = n {
                                    let ndim = as_tensor(&v)?.ndim();
                                    let axis = normalize_axis(axis_lit, ndim, span)?;
                                    let top = if fname == "sum_along" { TapeOp::SumAlong(axis) } else { TapeOp::MeanAlong(axis) };
                                    let id = tape.push(TapeNode { op: top, inputs: vec![nid], value: out.clone() });
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
                                let out = self.call_builtin(fname, vec![v.clone(), Value::Int(axis_lit)], psp.clone())?;
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
                                let out = self.call_builtin(fname, vec![xv.clone(), gv.clone(), Value::Float(eps)], psp.clone())?;
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
                                let out = self.call_builtin(fname, vec![xv.clone(), gv.clone(), bv.clone(), Value::Float(eps)], psp.clone())?;
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
                PostfixOp::Index(_elems) => {
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
                    (Value::Int(a),   Value::Int(b))   => a == b,
                    (Value::Float(a), Value::Float(b)) => a == b,
                    // Cross-match Int/Float like `==` (scalar_compare) and
                    // `list_contains` already do (#291.2): a `0` literal pattern
                    // matches a `0.0` scrutinee, so match agrees with equality.
                    (Value::Int(a),   Value::Float(b)) => (*a as f64) == *b,
                    (Value::Float(a), Value::Int(b))   => *a == (*b as f64),
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

    /// #336: the i64 ordinal of `variant` within `enum_name`, or None if either
    /// is unknown.
    fn enum_variant_ordinal(&self, enum_name: &str, variant: &str) -> Option<i64> {
        self.enums.get(enum_name)
            .and_then(|vs| vs.iter().position(|v| v == variant))
            .map(|p| p as i64)
    }

    fn eval_match(&mut self, me: &MatchExpr) -> EvalResult<Value> {
        let scrutinee = self.eval_expr(&me.scrutinee)?;
        for arm in &me.arms {
            if self.pattern_matches(&arm.pattern, &scrutinee) {
                self.push_scope();
                self.bind_pattern(&arm.pattern, scrutinee.clone());
                let guard_ok = if let Some(guard) = &arm.guard {
                    self.eval_expr(guard)?.as_bool().unwrap_or(false)
                } else {
                    true
                };
                if guard_ok {
                    let val = self.eval_expr(&arm.body)?;
                    self.pop_scope();
                    return Ok(val);
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

    fn call_fn_with_shapes(&mut self, f: &FnDecl, args: Vec<Value>, explicit_shapes: &[(String, i64)], sp: Span) -> EvalResult<Value> {
        if args.len() != f.params.len() {
            return Err(RuntimeError::at(
                format!("fn `{}` expects {} args, got {}", f.name, f.params.len(), args.len()),
                sp,
            ));
        }
        let _depth = self.enter_call(&sp)?;
        self.prof_fn_call();
        self.push_scope();
        // Shape params: bind from explicit args first, then infer from tensor shapes.
        // Bind all explicit shape params upfront so they are visible in the method body
        // even when not referenced in any parameter type annotation (e.g. model shape params
        // that only appear in the body as bare identifiers like `D`).
        for (name, dim) in explicit_shapes {
            self.bind(name, Value::Int(*dim));
        }
        let mut inferred: std::collections::HashMap<String, i64> =
            explicit_shapes.iter().cloned().collect();
        let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (p, v) in f.params.iter().zip(&args) {
            let Some(ty) = &p.ty else { continue; };
            let Some(spec) = type_shape_spec(ty) else { continue; };
            collect_idents_in_spec(spec, &mut referenced);
            let Value::Tensor(t) = v else { continue; };
            infer_shape_from_arg(spec, t.shape(), &mut inferred, &f.name, &sp)?;
        }
        for sp_decl in &f.shape_params {
            referenced.insert(sp_decl.name.clone());
        }
        for name in &referenced {
            match inferred.get(name) {
                Some(&dim) => self.bind(name, Value::Int(dim)),
                None => {
                    self.pop_scope();
                    return Err(RuntimeError::at(format!(
                        "fn `{}`: shape param `{}` cannot be inferred from any tensor argument's shape. \
                        Either pass a tensor whose declared type uses `{}`, or make the value explicit in the program.",
                        f.name, name, name), sp));
                }
            }
        }
        for (p, v) in f.params.iter().zip(args) {
            self.bind(&p.name, v);
            if let Some(axis) = p.ty.as_ref().and_then(streaming_axis_from_type) {
                self.bind_stream_axis(&p.name, axis);
            }
        }
        // @pp(stages=N): evaluate each `stage K: expr` in sequence, binding `_`
        // to the previous stage's output so the next stage can reference it.
        let is_pp = f.directives.iter().any(|d| d.name == "pp");
        if is_pp {
            let mut prev = Value::Nil;
            for stmt in &f.body.stmts {
                if let Stmt::Stage { body, .. } = stmt {
                    self.bind("_", prev.clone());
                    prev = self.eval_expr(body)?;
                }
            }
            self.pop_scope();
            return Ok(prev);
        }
        let result = self.eval_block(&f.body);
        // Before popping the fn scope, capture the final values of all `!` params.
        // The call site uses these to write back to the caller's bindings.
        self.pending_writebacks = f.params.iter().enumerate()
            .filter(|(_, p)| p.mutating)
            .filter_map(|(i, p)| {
                self.scopes.last()
                    .and_then(|s| s.get(&p.name))
                    .map(|v| (i, v.clone()))
            })
            .collect();
        self.pop_scope();
        match result {
            Ok(Flow::Normal(v)) | Ok(Flow::Return(v)) => Ok(v),
            Ok(Flow::Break) | Ok(Flow::Continue) => Err(RuntimeError::msg("break/continue outside loop")),
            // `?` early-return: this function returns the propagated (T, Err) tuple.
            Err(e) => match e.propagate { Some(v) => Ok(*v), None => Err(e) },
        }
    }

    // ── Block ─────────────────────────────────────────────────────────────

    fn eval_block(&mut self, block: &Block) -> EvalResult<Flow> {
        self.push_scope();
        let result = (|| -> EvalResult<Flow> {
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
                Ok(Flow::Normal(self.eval_expr(tail)?))
            } else if deferred_last {
                match &block.stmts[n - 1] {
                    Stmt::DirectiveBlock { directives, body, .. } => {
                        for s in &body.stmts {
                            match self.eval_stmt(s)? {
                                Flow::Normal(_) => {}
                                other => return Ok(other),
                            }
                        }
                        let raw = if let Some(t) = &body.tail_expr {
                            self.eval_expr(t)?
                        } else { Value::Nil };
                        // Apply @cast directive if present (same logic as Expr::DirectiveBlock).
                        let v = if let Some(cast_dir) = directives.iter().find(|d| d.name == "cast") {
                            if let Some(crate::ast::DArg::Positional(type_expr)) = cast_dir.args.first() {
                                if let Expr::Ident(ty_name, _) = type_expr {
                                    if self.model_fields.contains_key(ty_name) {
                                        self.cast_to_struct(&raw, ty_name)?
                                    } else {
                                        apply_cast_by_name(&raw, ty_name)
                                    }
                                } else { raw }
                            } else { raw }
                        } else { raw };
                        Ok(Flow::Normal(v))
                    }
                    Stmt::If(ie) => self.eval_if_flow(ie),
                    Stmt::Expr { lhs, assign: None, .. } => {
                        Ok(Flow::Normal(self.eval_expr(lhs)?))
                    }
                    Stmt::Stage { body, .. } => Ok(Flow::Normal(self.eval_expr(body)?)),
                    Stmt::Match(me) => Ok(Flow::Normal(self.eval_match(me)?)),
                    _ => unreachable!(),
                }
            } else {
                Ok(Flow::Normal(Value::Nil))
            }
        })();
        self.pop_scope();
        result
    }

    // ── Stmt ──────────────────────────────────────────────────────────────

    fn eval_stmt(&mut self, stmt: &Stmt) -> EvalResult<Flow> {
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
                                    self.bind(alias, Value::FieldRef { rc, field: fname.clone() });
                                    return Ok(Flow::Normal(Value::Nil));
                                }
                            }
                        }
                    }
                }
                let v = self.eval_expr(&l.value)?;
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
                    let rval = self.eval_expr(rhs)?;
                    match (op, lhs) {
                        // Simple assignment to ident
                        (AssignOp::Eq, Expr::Ident(name, _)) => {
                            // Try to assign to existing binding; create in current scope if new.
                            if self.assign(name, rval.clone()).is_err() {
                                self.bind(name, rval);
                            }
                        }
                        (AssignOp::PlusEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
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
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
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
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
                            let op = if matches!(cur, Value::Tensor(_)) || matches!(rval, Value::Tensor(_)) {
                                BinOp::DotMul
                            } else { BinOp::Mul };
                            let new = apply_binop(op, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::SlashEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
                            let op = if matches!(cur, Value::Tensor(_)) || matches!(rval, Value::Tensor(_)) {
                                BinOp::DotDiv
                            } else { BinOp::Div };
                            let new = apply_binop(op, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        // Bitwise compound-assign (int only; no tensor variant).
                        (AssignOp::AmpEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
                            let new = apply_binop(BinOp::BitAnd, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::BarEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
                            let new = apply_binop(BinOp::BitOr, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::CaretEq, Expr::Ident(name, _)) => {
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
                            let new = apply_binop(BinOp::BitXor, &cur, &rval)?;
                            self.assign(name, new).ok();
                        }
                        (AssignOp::StreamArrow, Expr::Ident(name, _)) => {
                            let cur = self.lookup(name).unwrap_or(Value::Nil);
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
                                    // Enforce declared capacity (Spec §3.6: panics on overflow).
                                    if let Some(&cap) = self.kv_capacities.get(name.as_str()) {
                                        let current_len = existing.shape()[axis];
                                        let append_len = new_data.shape()[axis];
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
                                        &[existing.view(), new_data.view()],
                                    ).map_err(|e| RuntimeError::msg(
                                        format!("<- stream append failed: {}", e)
                                    ))?;
                                    self.assign(name, Value::tensor_dt(cat, existing.dtype)).ok();
                                }
                                // `let !q = self.field; q <- x` — q is a FieldRef
                                // alias. Append through it back into the field.
                                (Value::FieldRef { rc, field }, Value::Tensor(new_data)) => {
                                    let mut borrowed = rc.borrow_mut();
                                    let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == &field) else {
                                        return Err(RuntimeError::msg(format!("unknown field `{}` in <- append", field)));
                                    };
                                    let (ex_data, dtype, ndim) = match slot {
                                        Value::Tensor(t) => (t.data.clone(), t.dtype, t.ndim()),
                                        other => return Err(RuntimeError::msg(
                                            format!("<- stream-arrow requires a tensor field, got {}", other.type_name()))),
                                    };
                                    let axis = stream_axis_for(&ex_data, &new_data.data);
                                    if axis >= ndim {
                                        return Err(RuntimeError::msg(format!(
                                            "<- stream append axis {} out of bounds for {}-d tensor", axis, ndim)));
                                    }
                                    let cat = ndarray::concatenate(Axis(axis), &[ex_data.view(), new_data.data.view()])
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
                                Value::Tensor(t) => (t.data.clone(), t.dtype, t.ndim()),
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
                                let cur = self.lookup(name).unwrap_or(Value::Nil);
                                match cur {
                                    Value::Tensor(mut arr) => {
                                        let dt = arr.dtype;
                                        self.assign_to_tensor(&mut arr, dt, idx_elems, rval, span.clone())?;
                                        self.assign(name, Value::Tensor(arr)).ok();
                                    }
                                    // #266: model-array (List) write — `arr[i] = instance`
                                    // stores the value into slot i (parity with the JIT's
                                    // ModelArray store). Single scalar index; negatives wrap.
                                    Value::List(mut items) => {
                                        let raw = match idx_elems.as_slice() {
                                            [IndexElem::Expr(e)] => self.eval_expr(e)?.as_int().ok_or_else(|| {
                                                RuntimeError::at("model-array index must be an integer", span.clone())
                                            })?,
                                            _ => return Err(RuntimeError::at(
                                                "model-array assignment requires a single scalar index", span.clone())),
                                        };
                                        let len = items.len() as i64;
                                        let i = if raw < 0 { len + raw } else { raw };
                                        if i < 0 || i >= len {
                                            return Err(RuntimeError::at(
                                                format!("index {} out of bounds for axis 0 of size {}", raw, len), span.clone()));
                                        }
                                        items[i as usize] = rval;
                                        self.assign(name, Value::List(items)).ok();
                                    }
                                    Value::FieldRef { rc, field } => {
                                        let mut borrowed = rc.borrow_mut();
                                        if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == &field) {
                                            if let Value::Tensor(ref mut arr) = slot {
                                                let dt = arr.dtype;
                                                self.assign_to_tensor(arr, dt, idx_elems, rval, span.clone())?;
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
                                    let mut borrowed = fields.borrow_mut();
                                    if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == field_name) {
                                        if let Value::Tensor(ref mut arr) = slot {
                                            let dt = arr.dtype;
                                            self.assign_to_tensor(arr, dt, idx_elems, rval, span.clone())?;
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
                        // Direct field write: m.field = expr  (including self.field = expr in methods)
                        (AssignOp::Eq, Expr::Postfix { expr: base_expr, op: PostfixOp::Field(field_name), .. }) => {
                            let recv = self.eval_expr(base_expr)?;
                            if let Value::Struct(fields) = recv {
                                let mut borrowed = fields.borrow_mut();
                                if let Some((_, slot)) = borrowed.iter_mut().find(|(k, _)| k == field_name) {
                                    *slot = rval;
                                }
                            }
                        }
                        // `:=` — shadow/rebind: create a new binding in current scope.
                        (AssignOp::ColonEq, Expr::Ident(name, _)) => {
                            self.bind(name, rval);
                        }
                        // Field / index assignments on non-ident bases — no-op
                        _ => {
                            // Evaluate lhs for side-effects only
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
                        Flow::Break => break,
                        Flow::Continue | Flow::Normal(_) => continue,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                    }
                }
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::While { cond, body, .. } => {
                loop {
                    let c = self.eval_expr(cond)?;
                    if !c.as_bool().unwrap_or(false) { break; }
                    match self.eval_block(body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::Loop { body, .. } => {
                loop {
                    match self.eval_block(body)? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
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
                let _ = self.eval_block(body)?;
                Ok(Flow::Normal(Value::Nil))
            }
            Stmt::Break(_)    => Ok(Flow::Break),
            Stmt::Continue(_) => Ok(Flow::Continue),
            Stmt::Return { value, .. } => {
                let v = if let Some(e) = value { self.eval_expr(e)? } else { Value::Nil };
                Ok(Flow::Return(v))
            }
        }
    }

    /// Evaluate an if-expression in *flow-aware* form. break/continue/return
    /// inside an if-body propagate out via the returned Flow. Callers that
    /// only want a Value should unwrap (or treat non-Normal as an error in
    /// expression context).
    fn eval_if_flow(&mut self, ie: &IfExpr) -> EvalResult<Flow> {
        let c = self.eval_expr(&ie.cond)?;
        if c.as_bool().unwrap_or(false) {
            self.eval_block(&ie.then_branch)
        } else {
            match &ie.else_branch {
                Some(ElseBranch::Block(b)) => self.eval_block(b),
                Some(ElseBranch::If(nested)) => self.eval_if_flow(nested),
                None => Ok(Flow::Normal(Value::Nil)),
            }
        }
    }

    /// Value-returning if for expression contexts. Non-Normal flow becomes
    /// an error here because break/continue/return don't make sense as values.
    fn eval_if(&mut self, ie: &IfExpr) -> EvalResult<Value> {
        match self.eval_if_flow(ie)? {
            Flow::Normal(v) | Flow::Return(v) => Ok(v),
            Flow::Break    => Err(RuntimeError::at("`break` outside loop".to_string(), ie.span.clone())),
            Flow::Continue => Err(RuntimeError::at("`continue` outside loop".to_string(), ie.span.clone())),
        }
    }

    // ── Expr ──────────────────────────────────────────────────────────────

    fn eval_expr(&mut self, expr: &Expr) -> EvalResult<Value> {
        match expr {
            Expr::Literal(lit, _) => Ok(lit_value(lit)),
            Expr::Nil(_)          => Ok(Value::Nil),
            Expr::Underscore(_)   => {
                // In @pp stage bodies `_` refers to the previous stage's output.
                // Look it up in scope; fall back to Nil for other uses (pipe placeholders).
                Ok(self.lookup("_").unwrap_or(Value::Nil))
            }
            Expr::Spread(_)       => Ok(Value::Nil),
            Expr::Ident(name, span) => self.eval_ident(name, span.clone()),
            Expr::Tuple(elems, _) => {
                if elems.len() == 1 { return self.eval_expr(&elems[0]); }
                let vs: Result<Vec<_>, _> = elems.iter().map(|e| self.eval_expr(e)).collect();
                Ok(Value::Tuple(vs?))
            }
            Expr::TensorLit(elems, lit_span) => {
                if elems.is_empty() {
                    self.prof_alloc();
                    return Ok(Value::tensor(ArrayD::zeros(IxDyn(&[0]))));
                }
                let first = self.eval_expr(&elems[0])?;
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
                            let v = self.eval_expr(e)?;
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
                    Value::Int(f0) => {
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
                        Ok(Value::tensor_dt(arr, DType::Int))
                    }
                    Value::Float(f0) => {
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
                if result.is_ok() { self.prof_alloc(); }
                result
            }
            Expr::Block(b) => match self.eval_block(b)? {
                Flow::Normal(v) | Flow::Return(v) => Ok(v),
                _ => Ok(Value::Nil),
            },
            Expr::ArenaBlock(ab) => match self.eval_block(&ab.body)? {
                Flow::Normal(v) | Flow::Return(v) => Ok(v),
                _ => Ok(Value::Nil),
            },
            Expr::If(ie) => self.eval_if(ie),
            Expr::Match(me) => self.eval_match(me),
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
                })
            }
            Expr::StructLit { name, type_args, fields, .. } => {
                let mut field_vals: Vec<(String, Value)> = Vec::with_capacity(fields.len() + 1);
                // Store model name as a hidden tag so method dispatch can find it.
                field_vals.push(("__model__".to_string(), Value::Str(name.clone())));
                // Store shape param bindings (e.g. P=5) so methods can use them at runtime.
                if let Some(param_names) = self.model_shape_params.get(name).cloned() {
                    for (pname, arg_expr) in param_names.iter().zip(type_args.iter()) {
                        if let Ok(v) = self.eval_expr(arg_expr) {
                            field_vals.push((format!("__shape_{}__", pname), v));
                        }
                    }
                }
                for (fname, fexpr) in fields {
                    field_vals.push((fname.clone(), self.eval_expr(fexpr)?));
                }
                Ok(Value::Struct(Rc::new(RefCell::new(field_vals))))
            }
            Expr::DirectiveBlock { directives, body, .. } => {
                let v = match self.eval_block(body)? {
                    Flow::Normal(v) | Flow::Return(v) => v,
                    _ => Value::Nil,
                };
                // Apply @cast directive to the block result if present.
                if let Some(cast_dir) = directives.iter().find(|d| d.name == "cast") {
                    if let Some(crate::ast::DArg::Positional(type_expr)) = cast_dir.args.first() {
                        let target = match type_expr {
                            Expr::Ident(name, _) => Some(name.as_str()),
                            _ => None,
                        };
                        if let Some(ty_name) = target {
                            if self.model_fields.contains_key(ty_name) {
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
                // Profile: count tensor ops vs scalar ops.
                match op {
                    BinOp::Matmul | BinOp::DotAdd | BinOp::DotSub | BinOp::DotMul | BinOp::DotDiv
                    | BinOp::DotPow | BinOp::DotPow2 | BinOp::DotGt | BinOp::DotLt
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
                            (Value::Int(_), _) | (Value::Float(_), _)
                            | (_, Value::Int(_)) | (_, Value::Float(_)) => {
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
                let v = self.eval_expr(operand)?;
                apply_unop(op.clone(), &v)
            }
            Expr::Postfix { expr, op, span } => self.eval_postfix(expr, op, span.clone()),
            Expr::Cast { expr, ty, .. } => {
                let v = self.eval_expr(expr)?;
                // #336: `Color.Red as i64` (or any numeric) → the variant ordinal.
                // Resolve to an Int here, where the enum registry is in scope,
                // then let the generic cast widen/narrow it.
                if let Value::EnumVal { enum_name, variant, .. } = &v {
                    let ord = self.enum_variant_ordinal(enum_name, variant).unwrap_or(0);
                    return Ok(apply_cast(&Value::Int(ord), ty));
                }
                Ok(apply_cast(&v, ty))
            }
            Expr::Range { start, end, inclusive, .. } => {
                let s = start.as_ref().and_then(|e| self.eval_expr(e).ok())
                    .and_then(|v| v.as_int()).unwrap_or(0);
                let e = end.as_ref().and_then(|e| self.eval_expr(e).ok())
                    .and_then(|v| v.as_int()).unwrap_or(0);
                Ok(Value::Range { start: s, end: e, inclusive: *inclusive })
            }
        }
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
        if self.fns.contains_key(name) { return Ok(Value::Fn(name.to_string())); }
        if self.extern_fns.contains(name) { return Ok(Value::Fn(name.to_string())); }
        // `pi` is a well-known math constant
        if name == "pi"  { return Ok(Value::Float(std::f64::consts::PI)); }
        if name == "tau" { return Ok(Value::Float(std::f64::consts::TAU)); }
        if name == "e"   { return Ok(Value::Float(std::f64::consts::E)); }
        if name == "inf" { return Ok(Value::Float(f64::INFINITY)); }
        if name == "nan" { return Ok(Value::Float(f64::NAN)); }
        if self.models.contains(name) {
            // Model names appear in `Transformer[L=24, ...].load(...)` patterns;
            // the interpreter doesn't realize methods, so it's opaque-but-known.
            return Ok(Value::Opaque(name.to_string()));
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
                // String method calls: s.split(","), s.trim(), s.upper(), etc.
                // Also handles model instance method calls: m.method(args).
                if let Expr::Postfix { expr: inner_expr, op: PostfixOp::Field(method), .. } = expr {
                    let recv = self.eval_expr(inner_expr)?;
                    if let Value::Str(s) = &recv {
                        return call_str_method(s, method, arg_vals, span);
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
                                return self.call_fn_with_shapes(&f, call_args, &shape_bindings, span);
                            }
                            // #394: model serialization `instance.save(...)` /
                            // `.load(...)` is specified (SPEC §3.10) but not
                            // implemented. With no user method of that name it
                            // used to fall through to a silent opaque (no file
                            // written, no error). Error loudly instead.
                            if matches!(method.as_str(), "save" | "load") {
                                return Err(RuntimeError::at(format!(
                                    "model serialization `{}.{}` is specified (SPEC §3.10) \
                                     but not yet implemented (#394)", mname, method), span));
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
                            if self.models.contains(name) {
                                return Err(RuntimeError::at(format!(
                                    "model serialization `{}.{}` is specified (SPEC §3.10) \
                                     but not yet implemented (#394)", name, method), span));
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
                                "distributed collective `{}.{}` is not executable here (#396); \
                                 collectives are unimplemented on a single node", coll, op), span));
                        }
                    }
                }
                // @grad call patterns: `<fn>.fwd_bwd(args)` or `<fn>.grad(args)`.
                // Intercept before generic field-access, since `.fwd_bwd` on a
                // Value::Fn would otherwise just yield Opaque.
                if let Expr::Postfix { expr: inner, op: PostfixOp::Field(method), .. } = expr {
                    if let Expr::Ident(fn_name, _) = inner.as_ref() {
                        if matches!(method.as_str(), "fwd_bwd" | "grad" | "fwd" | "fwd_bwd_bwd")
                            && self.fns.contains_key(fn_name)
                        {
                            return self.call_grad(fn_name, method, arg_vals, span);
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
                            return Err(RuntimeError::at(format!("extern fn `{}` cannot be called without a JIT backend", name), span));
                        }
                        let f = self.fns.get(&name).cloned().ok_or_else(|| {
                            RuntimeError::at(format!("undefined fn `{}`", name), span.clone())
                        })?;
                        let result = self.call_fn(&f, arg_vals, span)?;
                        apply_writebacks(&mut self.scopes, &mut self.pending_writebacks, args);
                        Ok(result)
                    }
                    Value::BoundFn { ref name, ref shape_bindings } => {
                        let f = self.fns.get(name).cloned().ok_or_else(|| {
                            RuntimeError::at(format!("undefined fn `{}`", name), span.clone())
                        })?;
                        let bindings = shape_bindings.clone();
                        let result = self.call_fn_with_shapes(&f, arg_vals, &bindings, span)?;
                        apply_writebacks(&mut self.scopes, &mut self.pending_writebacks, args);
                        Ok(result)
                    }
                    Value::Lambda { lit, captured_env } => self.call_lambda(&lit.clone(), captured_env, arg_vals, span),
                    Value::Builtin(name) => self.call_builtin(&name, arg_vals, span),
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
                                return self.call_fn_with_shapes(&f, call_args, &shape_bindings, span);
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
                let recv = self.eval_expr(expr)?;
                match (&recv, name.as_str()) {
                    (Value::Module { alias, .. }, _) => {
                        let qual_name = format!("{}.{}", alias, name);
                        if self.fns.contains_key(&qual_name) {
                            return Ok(Value::Fn(qual_name));
                        }
                        if let Some(v) = self.lookup(&qual_name) {
                            return Ok(v.clone());
                        }
                        if self.models.contains(&qual_name) {
                            return Ok(Value::Opaque(qual_name));
                        }
                        Err(RuntimeError::at(format!("no member `{}` in module `{}`", name, alias), span))
                    }
                    (Value::Tensor(t), "shape") => {
                        let dims: Vec<Value> = t.shape().iter()
                            .map(|&d| Value::Int(d as i64)).collect();
                        Ok(Value::Tuple(dims))
                    }
                    (Value::Tensor(_), "rank" | "ndim") => Ok(Value::Int(
                        if let Value::Tensor(t) = &recv { t.ndim() as i64 } else { 0 }
                    )),
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
                        return Ok(Value::tensor_dt(t.data.clone().permuted_axes(IxDyn(&axes)), t.dtype));
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
                            // with this whole (T, Err) tuple (Rust-`?` style).
                            // The signal is caught at the function-call boundary.
                            _ => Err(RuntimeError::propagate(val)),
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
                // `f[N]` with user fn: bind explicit shape params
                if let Expr::Ident(fn_name, _) = expr {
                    if let Some(f) = self.fns.get(fn_name).cloned() {
                        if !f.shape_params.is_empty() {
                            let mut shape_bindings = Vec::new();
                            for (sp, arg) in f.shape_params.iter().zip(args.iter().filter_map(|a| match a {
                                CallArg::Positional(e) => Some(e),
                                _ => None,
                            })) {
                                if let Ok(v) = self.eval_expr(arg) {
                                    if let Some(n) = v.as_int() {
                                        shape_bindings.push((sp.name.clone(), n));
                                    }
                                }
                            }
                            if !shape_bindings.is_empty() {
                                return Ok(Value::BoundFn { name: fn_name.clone(), shape_bindings });
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
                            let shape_val = match elems.first() {
                                Some(IndexElem::Expr(e)) => self.eval_expr(e)?,
                                _ => return Err(RuntimeError::at(
                                    "reshape: expected shape list in brackets", span)),
                            };
                            let new_shape: Vec<usize> = match shape_val {
                                Value::List(vs) | Value::Tuple(vs) => vs.iter()
                                    .filter_map(|v| v.as_int().map(|n| n as usize))
                                    .collect(),
                                // [2, 3, 4] in demoniC evaluates as a Tensor literal
                                Value::Tensor(t) => t.iter().map(|&x| x as usize).collect(),
                                Value::Int(n) => vec![n as usize],
                                _ => return Err(RuntimeError::at(
                                    "reshape: shape must be a list of integers", span)),
                            };
                            let dtype = t.dtype;
                            return t.data.clone().into_shape_with_order(ndarray::IxDyn(&new_shape))
                                .map(|a| Value::tensor_dt(a, dtype))
                                .map_err(|e| RuntimeError::at(
                                    format!("reshape failed: {}", e), span));
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
                                Value::Int(n)   => raw_idx.push(n),
                                Value::Float(x) => raw_idx.push(x as i64),
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
                            if arr.is_int() { Ok(Value::Int(x as i64)) } else { Ok(Value::Float(x)) }
                        } else if idx.len() < arr.ndim() {
                            // Partial index → sub-tensor slice along first axes;
                            // a sub-tensor keeps its parent's dtype.
                            let mut view = arr.view();
                            for &i in &idx {
                                view = view.index_axis_move(Axis(0), i);
                            }
                            Ok(Value::tensor_dt(view.to_owned().into_dyn(), arr.dtype))
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
                        vs.into_iter().nth(resolved as usize).ok_or_else(|| RuntimeError::at(
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
                        Ok(Value::tensor_dt(result, arr.dtype))
                    }
                    // `s[i]` on a string — return the byte value at index i.
                    // Matches JIT behaviour: returns an i64 ASCII/UTF-8 byte.
                    Value::Str(ref s) if all_scalar && raw_idx.len() == 1 => {
                        let len = s.len() as i64;
                        let n = raw_idx[0];
                        let i = if n < 0 { (len + n) as usize } else { n as usize };
                        match s.as_bytes().get(i) {
                            Some(&b) => Ok(Value::Int(b as i64)),
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
                    _ => Ok(Value::Opaque("index".into())),
                }
            }
        }
    }

    /// Evaluate an expression as a dimension count. Falls back to a small
    /// default for symbols that can't be resolved (e.g. streaming-axis `~`,
    /// unbound shape params), so we always get a concrete shape.
    fn eval_dim(&mut self, e: &Expr) -> usize {
        match self.eval_expr(e) {
            Ok(Value::Int(n))   if n >= 0 => n as usize,
            Ok(Value::Float(x)) if x >= 0.0 => x as usize,
            _ => 4,
        }
    }

    /// Byte size of a field type for `@cast(Model)` overlay. Scalars use their
    /// declared width; tensors are element-size × element-count.
    fn type_byte_size(&mut self, ty: &crate::ast::Type) -> EvalResult<usize> {
        use crate::ast::Type;
        match ty {
            Type::Scalar(st, _) => Ok(scalar_byte_size(st)),
            Type::Tensor(elem, shape, _) => {
                let elem_sz = self.type_byte_size(elem)?;
                let count: usize = self.shape_dims(shape)?.iter().product();
                Ok(elem_sz * count)
            }
            other => Err(RuntimeError::msg(format!("@cast: unsupported field type {:?}", other))),
        }
    }

    /// Concrete dimensions of a shape spec (each axis must evaluate to an int).
    fn shape_dims(&mut self, shape: &crate::ast::ShapeSpec) -> EvalResult<Vec<usize>> {
        use crate::ast::ShapeElem;
        let mut dims = Vec::with_capacity(shape.elems.len());
        for e in &shape.elems {
            match e {
                ShapeElem::Expr(ex) => dims.push(self.eval_dim(ex)),
                _ => return Err(RuntimeError::msg("@cast: tensor field shape must be concrete")),
            }
        }
        Ok(dims)
    }

    /// Overlay a raw byte tensor onto a model's fields (`@cast(Model){bytes}`).
    /// Fields are read in declaration order at successive byte offsets. Scalars
    /// are decoded big-endian (network byte order — `@cast(Struct)` parses wire
    /// formats; the only corpus user is a packet header). Returns a struct value
    /// whose fields can then be read like any model instance.
    fn cast_to_struct(&mut self, raw: &Value, model_name: &str) -> EvalResult<Value> {
        use crate::ast::Type;
        let fields_def = match self.model_fields.get(model_name) {
            Some(f) => f.clone(),
            None => return Err(RuntimeError::msg(format!("@cast: `{}` is not a known model", model_name))),
        };
        let bytes: Vec<u8> = match raw {
            Value::Tensor(t) => t.data.iter().map(|&x| x as i64 as u8).collect(),
            other => return Err(RuntimeError::msg(format!(
                "@cast({}) expects a byte tensor to overlay, got {}", model_name, other.type_name()))),
        };
        let mut offset = 0usize;
        let mut out: Vec<(String, Value)> =
            vec![("__model__".to_string(), Value::Str(model_name.to_string()))];
        for (fname, fty) in &fields_def {
            let val = match fty {
                Type::Scalar(_, _) => {
                    let sz = self.type_byte_size(fty)?;
                    let slice = read_overlay_bytes(&bytes, offset, sz, model_name, fname)?;
                    offset += sz;
                    Value::Int(be_bytes_to_i64(slice))
                }
                Type::Tensor(elem, shape, _) => {
                    let elem_sz = self.type_byte_size(elem)?;
                    let dims = self.shape_dims(shape)?;
                    let count: usize = dims.iter().product();
                    let mut data: Vec<f64> = Vec::with_capacity(count);
                    for _ in 0..count {
                        let slice = read_overlay_bytes(&bytes, offset, elem_sz, model_name, fname)?;
                        offset += elem_sz;
                        data.push(be_bytes_to_i64(slice) as f64);
                    }
                    let arr = ArrayD::from_shape_vec(IxDyn(&dims), data)
                        .map_err(|e| RuntimeError::msg(format!("@cast({}): {}", model_name, e)))?;
                    Value::tensor_dt(arr, DType::Int)
                }
                other => return Err(RuntimeError::msg(format!(
                    "@cast({}): field `{}` has unsupported type {:?}", model_name, fname, other))),
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
        let n = positional.first()
            .and_then(|e| self.eval_expr(e).ok())
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
            pieces.push(Value::Tensor(TensorVal::new(slice, dtype)));
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
        let dims: Vec<usize> = bracket_args.iter().find_map(|e| match e {
            Expr::TensorLit(d, _) => Some(d.iter().map(|x| self.eval_dim(x)).collect()),
            _ => None,
        }).unwrap_or_default();
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
        Ok(Value::Tuple(vec![rng_state, Value::tensor_dt(arr, DType::Int)]))
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
                 `rng.uniform_int[T, [shape]](low, high)` (SPEC §3.8)".to_string(),
                expr.span_of(),
            ));
        }
        if !is_tensor_ctor && !is_rng_ctor { return Ok(None); }

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
                if self.models.contains(base) {
                    let n = pos_vec.iter().find_map(|e| match e {
                        Expr::TensorLit(dims, _) => dims.first().map(|d| self.eval_dim(d)),
                        _ => None,
                    }).ok_or_else(|| RuntimeError::msg(format!(
                        "`{}[{}, [N]]` requires a 1D size", method, base)))?;
                    self.prof_alloc();
                    return Ok(Some(Value::List(vec![Value::Nil; n])));
                }
            }
        }

        // Random draws are float-valued by definition regardless of the
        // declared type; everything else inherits the declared element type.
        let dtype = if is_rng_ctor { DType::F32 } else { arena_ctor_dtype(&pos_vec) };

        // `identity` builds an N×N matrix with 1.0 on the diagonal. It accepts
        // either a scalar dimension (`vault.identity[T, N]`) or a square shape
        // literal (`vault.identity[T, [N, N]]`). Completes the zeros/ones/
        // identity additive-and-multiplicative-identity triple.
        if method == "identity" {
            let n = pos_vec.iter().find_map(|e| match e {
                Expr::TensorLit(dims, _) => dims.first().map(|d| self.eval_dim(d)),
                Expr::Literal(Literal::Int(k), _) if *k >= 0 => Some(*k as usize),
                _ => None,
            }).unwrap_or(0);
            if checked_shape_elems(&[n, n]).is_none() {
                return Err(RuntimeError::msg(format!(
                    "identity: size {} is too large — the {}×{} element count overflows the address space",
                    n, n, n)));
            }
            let mut a = ArrayD::<f64>::zeros(IxDyn(&[n, n]));
            for i in 0..n { a[[i, i]] = 1.0; }
            self.prof_alloc();
            return Ok(Some(Value::tensor_dt(a, dtype)));
        }

        // forge.trit[M, N] — dimensions given as raw integer args, no type arg.
        // Returns a zero-initialised DType::Trit tensor clamped to {-1, 0, +1}.
        if method == "trit" {
            let trit_dims: Vec<usize> = pos_vec.iter().filter_map(|e| match e {
                Expr::Literal(Literal::Int(k), _) if *k >= 0 => Some(*k as usize),
                _ => None,
            }).collect();
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
                    if name == "~" { return 0; }
                }
                // Also handle UnOp::BitNot used as ~ in expressions
            }
            self.eval_dim(e)
        }).collect());
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
        // OOM-kill; the `try_reserve` below is the remaining backstop.
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        let nbytes = (nelems as u64).saturating_mul(8);
        if let Some(total) = total_system_memory_bytes() {
            if nbytes > total {
                return Err(located(format!(
                    "{}: tensor shape {:?} needs {:.1} GiB but the system has {:.1} GiB of RAM",
                    method, dims, nbytes as f64 / GIB, total as f64 / GIB)));
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

fn arena_ctor_dtype(pos: &[&Expr]) -> DType {
    for e in pos {
        if let Expr::Ident(name, _) = e {
            if is_int_type_name(name)  { return DType::Int; }
            if is_trit_type_name(name) { return DType::Trit; }
            if name == "f64"           { return DType::F64; }
        }
    }
    DType::F32
}

fn is_int_type_name(name: &str) -> bool {
    matches!(name,
        "i8" | "i16" | "i32" | "i64" |
        "u8" | "u16" | "u32" | "u64" |
        "int4" | "int8")
}

fn is_trit_type_name(name: &str) -> bool {
    name == "trit"
}

// ─── Literal → Value ─────────────────────────────────────────────────────────

fn lit_value(lit: &Literal) -> Value {
    match lit {
        Literal::Int(n)   => Value::Int(*n),
        Literal::Float(x, _) => Value::Float(*x),
        Literal::Str(s)   => Value::Str(s.clone()),
        Literal::Char(c)  => Value::Int(*c as i64),
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
        DotAdd | DotSub | DotMul | DotDiv | DotPow | DotPow2 => {
            // #275: a dotted arithmetic op on two scalars is the scalar op — the
            // checker types `5.0 ./ 2.0` as a scalar f32, and the JIT computes it
            // as such; don't promote to a rank-0 tensor here. Only when an operand
            // is an actual tensor does this become an elementwise tensor op.
            let scalar_pair = !matches!(l, Value::Tensor(_)) && !matches!(r, Value::Tensor(_))
                && l.as_float().is_some() && r.as_float().is_some();
            if scalar_pair {
                let base = match op {
                    DotAdd => Add, DotSub => Sub, DotMul => Mul, DotDiv => Div,
                    _ => Pow, // DotPow | DotPow2
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
        // RShift now parses as Pipe so it also never reaches here.
        Pipe | RShift => Ok(r.clone()),
        BitAnd => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("& requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("& requires int, got {}", r.type_name())))?;
            Ok(Value::Int(a & b))
        }
        BitOr => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("| requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("| requires int, got {}", r.type_name())))?;
            Ok(Value::Int(a | b))
        }
        BitXor => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("^ requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("^ requires int, got {}", r.type_name())))?;
            Ok(Value::Int(a ^ b))
        }
        BitShl => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!("<< requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!("<< requires int, got {}", r.type_name())))?;
            // #215: the shift amount was cast `as u32`, so a negative or >=64 RHS
            // silently wrapped/masked (and panicked in debug builds). Validate it.
            if !(0..64).contains(&b) {
                Err(RuntimeError::msg(format!("<< shift amount {} out of range (expected 0..=63)", b)))
            } else {
                Ok(Value::Int(a << b))
            }
        }
        BitShr => {
            let a = l.as_int().ok_or_else(|| RuntimeError::msg(format!(">> requires int, got {}", l.type_name())))?;
            let b = r.as_int().ok_or_else(|| RuntimeError::msg(format!(">> requires int, got {}", r.type_name())))?;
            // (`>>` parses as the pipe operator, so this arm is currently dead, but
            // guard the shift amount too for consistency — #215.)
            if !(0..64).contains(&b) {
                Err(RuntimeError::msg(format!(">> shift amount {} out of range (expected 0..=63)", b)))
            } else {
                Ok(Value::Int(a >> b))
            }
        }
    }
}

fn scalar_arith(op: BinOp, l: &Value, r: &Value) -> EvalResult<Value> {
    use BinOp::*;
    // int + int = int; otherwise float-promote
    if let (Some(a), Some(b)) = (l.as_int(), r.as_int()) {
        // Integer power needs fallible handling (#215): the exponent was cast via
        // `as u32`, silently truncating exponents > u32::MAX and wrapping on
        // i64 overflow. Validate the exponent range and use checked_pow.
        if matches!(op, Pow | StarStar) {
            if b < 0 || b > u32::MAX as i64 {
                return Err(RuntimeError::msg(format!(
                    "integer exponent {} out of range (expected 0..={})", b, u32::MAX)));
            }
            return match a.checked_pow(b as u32) {
                Some(v) => Ok(Value::Int(v)),
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
            Div => {
                if b == 0 { 0 }
                else {
                    a.checked_div(b).ok_or_else(|| RuntimeError::msg(
                        format!("integer overflow: {} / {} exceeds the i64 range", a, b)))?
                }
            },
            Mod => {
                if b == 0 { 0 }
                else {
                    a.checked_rem(b).ok_or_else(|| RuntimeError::msg(
                        format!("integer overflow: {} % {} exceeds the i64 range", a, b)))?
                }
            },
            _ => unreachable!(),
        };
        return Ok(Value::Int(v));
    }
    if let (Some(a), Some(b)) = (l.as_float(), r.as_float()) {
        let v = match op {
            Add => a + b, Sub => a - b, Mul => a * b,
            Div => a / b, Mod => a % b,
            Pow | StarStar => a.powf(b),
            _ => unreachable!(),
        };
        return Ok(Value::Float(v));
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
    if let (Value::Int(a), Value::Int(b)) = (l, r) {
        return Ok(Value::Bool(match op {
            Eq    => a == b, NotEq => a != b,
            Lt    => a < b,  Gt    => a > b,
            LtEq  => a <= b, GtEq  => a >= b,
            _ => unreachable!(),
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
        DotPow | DotPow2 => pow_by_scalar(a, s),
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
        DotPow | DotPow2 => b.mapv(|x| s.powf(x)),
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
        Value::Tensor(t) => matches!(t.dtype, DType::F32 | DType::Trit),
        _ => true,
    };
    if operands.iter().all(narrow) { DType::F32 } else { DType::F64 }
}

fn tensor_elementwise(op: BinOp, l: &Value, r: &Value) -> EvalResult<Value> {
    use BinOp::*;
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
        DotPow | DotPow2 => {
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

fn tensor_matmul(l: &Value, r: &Value) -> EvalResult<Value> {
    // #241: width of the product follows the operands. Note the interpreter
    // still *accumulates* dot products in f64 and rounds once at the end,
    // where the JIT accumulates in f32 per step — a documented residual
    // divergence (the interpreter is the more accurate of the two here).
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
        return Ok(Value::tensor_dt(a2.dot(&b2).into_dyn(), dt));
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
        let prod = a_slice.dot(&b_slice);
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

/// Write back the final values of `!`-prefixed parameters to the caller's scope.
///
/// `pending` holds `(param_index, final_value)` pairs populated by
/// `call_fn_with_shapes` before it pops the callee's scope. For each entry,
/// if the corresponding call argument was a bare identifier, the identifier is
/// updated in the caller's current scope stack.
fn apply_writebacks(
    scopes: &mut Vec<HashMap<String, Value>>,
    pending: &mut Vec<(usize, Value)>,
    call_args: &[CallArg],
) {
    for (idx, val) in pending.drain(..) {
        if let Some(CallArg::Positional(Expr::Ident(name, _))) = call_args.get(idx) {
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
                    if let Expr::Literal(Literal::Int(n), _) = value {
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
    let Expr::Ident(name, _) = e.as_ref() else { return Ok(()); };
    // Only bind shape-variable-like names (uppercase-first).
    if !name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) { return Ok(()); }
    if let Some(&prev) = inferred.get(name) {
        if prev != dim {
            return Err(RuntimeError::at(format!(
                "fn `{}`: shape param `{}` would need to be both {} and {} \
                (inconsistent tensor shapes across arguments)",
                fn_name, name, prev, dim), sp.clone()));
        }
    } else {
        inferred.insert(name.clone(), dim);
    }
    Ok(())
}

fn as_tensor(v: &Value) -> EvalResult<ArrayD<f64>> {
    match v {
        Value::Tensor(t) => Ok(t.data.clone()),
        Value::Int(n)   => Ok(ArrayD::from_elem(IxDyn(&[]), *n as f64)),
        Value::Float(x) => Ok(ArrayD::from_elem(IxDyn(&[]), *x)),
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
            Value::Int(n)    => Ok(Value::Int(-n)),
            Value::Float(x)  => Ok(Value::Float(-x)),
            // Result width follows the operand (#241): an Int/F64 tensor must
            // not be rounded through f32 by the default-F32 construction.
            Value::Tensor(t) => Ok(Value::tensor_dt(t.map(|x| -x), float_result_dtype(&[v]))),
            _ => Err(RuntimeError::msg(format!("- requires numeric, got {}", v.type_name()))),
        }
        Not => Ok(Value::Bool(!v.as_bool().unwrap_or(false))),
        Deref => Ok(v.clone()),  // pre-alpha: no references
        ReLU => match v {
            Value::Tensor(t) => Ok(Value::tensor_dt(t.map(|x| x.max(0.0)), float_result_dtype(&[v]))),
            Value::Float(x)  => Ok(Value::Float(x.max(0.0))),
            Value::Int(n)    => Ok(Value::Int((*n).max(0))),
            _ => Err(RuntimeError::msg(format!("\\> requires numeric/tensor, got {}", v.type_name()))),
        }
        GeLU => match v {
            // GeLU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
            Value::Tensor(t) => Ok(Value::tensor_dt(t.map(|x| gelu(*x)), float_result_dtype(&[v]))),
            Value::Float(x) => Ok(Value::Float(gelu(*x))),
            _ => Err(RuntimeError::msg(format!("\\< requires numeric/tensor, got {}", v.type_name()))),
        }
        BitNot => match v {
            Value::Int(n) => Ok(Value::Int(!n)),
            _ => Err(RuntimeError::msg(format!("~ (bitwise NOT) requires int, got {}", v.type_name()))),
        }
    }
}

fn gelu(x: f64) -> f64 {
    let c = (2.0f64 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
}

/// Cast by type name string — used when the directive carries the type as an identifier.
/// Declared byte width of a scalar type, for `@cast(Model)` overlay.
fn scalar_byte_size(st: &ScalarType) -> usize {
    use ScalarType::*;
    match st {
        I8 | U8 | Int4 | Int8 | Trit | Bool | Fp8E4M3 | Fp8E5M2 => 1,
        I16 | U16 | F16 | Bf16 => 2,
        I32 | U32 | F32 | Tf32 => 4,
        I64 | U64 | F64 => 8,
        Str | Nil => 0,
    }
}

/// Combine bytes big-endian (network byte order) into an i64.
fn be_bytes_to_i64(slice: &[u8]) -> i64 {
    let mut acc: i64 = 0;
    for &b in slice { acc = (acc << 8) | (b as i64); }
    acc
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
        if is_int_type_name(ty_name) {
            return Value::tensor_dt(t.data.mapv(|x| x.trunc()), DType::Int);
        }
        if ty_name == "f64" {
            // Widening retag — no value change (f32-rounded data is exact in f64).
            return Value::tensor_dt(t.data.clone(), DType::F64);
        }
        if matches!(ty_name, "f16" | "bf16" | "tf32" | "f32" | "fp8_e4m3" | "fp8_e5m2") {
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
                return Value::tensor_dt(arr, DType::Int);
            }
            if let Some(n) = v.as_int()   { return Value::Int((n as u8)  as i64); }
            if let Some(n) = v.as_float() { return Value::Int(n as u8 as i64); }
        }
        "i8"  => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                   if let Some(n) = v.as_int() { return Value::Int((n as i8)  as i64); }
                   if let Some(n) = v.as_float() { return Value::Int(n as i64); } }
        "i16" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                   if let Some(n) = v.as_int() { return Value::Int((n as i16) as i64); }
                   if let Some(n) = v.as_float() { return Value::Int(n as i64); } }
        "i32" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                   if let Some(n) = v.as_int() { return Value::Int((n as i32) as i64); }
                   if let Some(n) = v.as_float() { return Value::Int(n as i64); } }
        "i64" | "int4" | "int8" => {
            if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
            if let Some(n) = v.as_float() { return Value::Int(n as i64); }
        }
        "u16" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                   if let Some(n) = v.as_int() { return Value::Int((n as u16) as i64); }
                   if let Some(n) = v.as_float() { return Value::Int(n as u16 as i64); } }
        "u32" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                   if let Some(n) = v.as_int() { return Value::Int((n as u32) as i64); }
                   if let Some(n) = v.as_float() { return Value::Int(n as u32 as i64); } }
        "u64" => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                   if let Some(n) = v.as_int() { return Value::Int((n as u64) as i64); }
                   if let Some(n) = v.as_float() { return Value::Int(n as u64 as i64); } }
        "f64" => {
            if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }); }
            if let Some(n) = v.as_float() { return Value::Float(n); }
        }
        // f32-family scalar casts round through f32 (#300 ruling) — see apply_cast.
        "f16" | "bf16" | "tf32" | "f32" | "fp8_e4m3" | "fp8_e5m2" => {
            if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }); }
            if let Some(n) = v.as_float() { return Value::Float(n as f32 as f64); }
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
                // the scalar cast path (#298 / #291.1). Narrow targets fit f64
                // exactly; I64/U64/Int4/Int8 stay wide (f64 can't hold the full
                // range, same as the prior behavior).
                I8  => Value::tensor_dt(t.data.mapv(|x| x as i8  as f64), DType::Int),
                I16 => Value::tensor_dt(t.data.mapv(|x| x as i16 as f64), DType::Int),
                I32 => Value::tensor_dt(t.data.mapv(|x| x as i32 as f64), DType::Int),
                U8  => Value::tensor_dt(t.data.mapv(|x| x as u8  as f64), DType::Int),
                U16 => Value::tensor_dt(t.data.mapv(|x| x as u16 as f64), DType::Int),
                U32 => Value::tensor_dt(t.data.mapv(|x| x as u32 as f64), DType::Int),
                I64 | U64 | Int4 | Int8 =>
                    Value::tensor_dt(t.data.mapv(|x| x.trunc()), DType::Int),
                F64 => Value::tensor_dt(t.data.clone(), DType::F64),
                F16 | Bf16 | Tf32 | F32 | Fp8E4M3 | Fp8E5M2 =>
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
            I8  => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                     if let Some(n) = v.as_int() { return Value::Int((n as i8)  as i64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as i8 as i64); } }
            I16 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                     if let Some(n) = v.as_int() { return Value::Int((n as i16) as i64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as i16 as i64); } }
            I32 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                     if let Some(n) = v.as_int() { return Value::Int((n as i32) as i64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as i32 as i64); } }
            I64 | Int4 | Int8 | Trit => {
                if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                if let Some(n) = v.as_float() { return Value::Int(n as i64); }
            }
            U8  => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u8)  as i64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u8 as i64); } }
            U16 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u16) as i64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u16 as i64); } }
            U32 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u32) as i64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u32 as i64); } }
            U64 => { if let Value::Bool(b) = v { return Value::Int(if *b { 1 } else { 0 }); }
                     if let Some(n) = v.as_int() { return Value::Int((n as u64) as i64); }
                     if let Some(n) = v.as_float() { return Value::Int(n as u64 as i64); } }
            F64 => {
                if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }); }
                if let Some(n) = v.as_float() { return Value::Float(n); }
            }
            // f32-family scalar casts ROUND through f32 (#300 ruling): an explicit
            // `as f32` loses precision, matching the JIT and the tensor cast path
            // which retags these as f32. `0.1 as f32` → 0.10000000149… not 0.1.
            F16 | Bf16 | Tf32 | F32 | Fp8E4M3 | Fp8E5M2 => {
                if let Value::Bool(b) = v { return Value::Float(if *b { 1.0 } else { 0.0 }); }
                if let Some(n) = v.as_float() { return Value::Float(n as f32 as f64); }
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

    f
}

// ─── Iteration support ───────────────────────────────────────────────────────

fn expand_iter(v: &Value) -> EvalResult<Vec<Value>> {
    match v {
        Value::Range { start, end, inclusive } => {
            let end = if *inclusive { *end + 1 } else { *end };
            Ok((*start..end).map(Value::Int).collect())
        }
        Value::Tuple(vs) => Ok(vs.clone()),
        Value::List(vs) => Ok(vs.clone()),
        Value::Tensor(t) => {
            // Iterate along the first axis as nested tensors / scalars,
            // preserving the dtype so iterating an integer tensor yields Ints.
            let is_int = t.is_int();
            let scalar = |x: f64| if is_int { Value::Int(x as i64) } else { Value::Float(x) };
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
            Ok(Value::List(parts))
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
            Ok(Value::Int(idx))
        }
        "index" => {
            let needle = match args.first() {
                Some(Value::Str(n)) => n.clone(),
                _ => return Err(RuntimeError::at("str.index: needs str arg", sp)),
            };
            s.find(needle.as_str()).map(|i| Value::Int(i as i64))
                .ok_or_else(|| RuntimeError::at(
                    format!("str.index: {:?} not found in {:?}", needle, s), sp))
        }
        "count" => {
            let needle = match args.first() {
                Some(Value::Str(n)) => n.clone(),
                _ => return Err(RuntimeError::at("str.count: needs str arg", sp)),
            };
            if needle.is_empty() {
                return Ok(Value::Int(s.chars().count() as i64 + 1));
            }
            let count = s.matches(needle.as_str()).count();
            Ok(Value::Int(count as i64))
        }
        "lines" | "split_lines" => {
            let parts: Vec<Value> = s.lines()
                .map(|l| Value::Str(l.to_string()))
                .collect();
            Ok(Value::List(parts))
        }
        "len" => Ok(Value::Int(s.len() as i64)),
        _ => {
            // Unknown method — fall through to opaque (don't error, for forward compat)
            Ok(Value::Opaque(format!("str.{}", method)))
        }
    }
}

// ─── Builtins ────────────────────────────────────────────────────────────────

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
        "sort" | "gcd" | "median" |   // #335: stdlib the harvest showed models reach for
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
    fn call_builtin(&mut self, name: &str, args: Vec<Value>, sp: Span) -> EvalResult<Value> {
    match name {
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
            Ok(Value::Float(x.sqrt()))
        }
        "exp" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("exp: numeric"))?
            .exp())),
        "log" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("log: numeric"))?
            .ln())),
        "abs" => match args.first() {
            Some(Value::Int(n)) => Ok(Value::Int(n.abs())),
            Some(v) => Ok(Value::Float(v.as_float().unwrap_or(0.0).abs())),
            None => Err(RuntimeError::msg("abs: needs arg")),
        }
        "sin" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .unwrap_or(0.0).sin())),
        "cos" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .unwrap_or(0.0).cos())),
        "floor" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("floor: numeric arg required"))?
            .floor())),
        "ceil" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("ceil: numeric arg required"))?
            .ceil())),
        "tan" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("tan: numeric arg required"))?
            .tan())),
        "asin" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("asin: numeric arg required"))?
            .asin())),
        "acos" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("acos: numeric arg required"))?
            .acos())),
        "atan" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("atan: numeric arg required"))?
            .atan())),
        "atan2" => {
            let y = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("atan2: numeric y arg required"))?;
            let x = args.get(1).and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("atan2: numeric x arg required"))?;
            Ok(Value::Float(y.atan2(x)))
        }
        "hypot" => {
            let x = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("hypot: numeric x arg required"))?;
            let y = args.get(1).and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("hypot: numeric y arg required"))?;
            Ok(Value::Float(x.hypot(y)))
        }
        // #335: greatest common divisor (Euclid), |a|,|b|; gcd(0,0)=0.
        "gcd" => {
            let mut a = args.first().and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("gcd: integer args required"))?.abs();
            let mut b = args.get(1).and_then(|v| v.as_int())
                .ok_or_else(|| RuntimeError::msg("gcd: integer args required"))?.abs();
            while b != 0 { let t = b; b = a % t; a = t; }
            Ok(Value::Int(a))
        }
        "log2" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("log2: numeric arg required"))?
            .log2())),
        "log10" => Ok(Value::Float(args.first().and_then(|v| v.as_float())
            .ok_or_else(|| RuntimeError::msg("log10: numeric arg required"))?
            .log10())),
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
                Ok(Value::Float(x.round()))
            } else {
                let factor = 10f64.powi(ndigits as i32);
                Ok(Value::Float((x * factor).round() / factor))
            }
        }
        "trunc" => {
            let x = args.first().and_then(|v| v.as_float())
                .ok_or_else(|| RuntimeError::msg("trunc: numeric arg required"))?;
            Ok(Value::Float(x.trunc()))
        }
        "sign" => {
            match args.first() {
                Some(Value::Int(n))   => Ok(Value::Int(n.signum())),
                Some(Value::Float(f)) => Ok(Value::Float(f.signum())),
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
                Some(Value::Int(n)) => return Ok(Value::Int(*n)), // already a codepoint
                _ => return Err(RuntimeError::at("ord: str arg required", sp)),
            };
            let ch = s.chars().next()
                .ok_or_else(|| RuntimeError::at("ord: empty string", sp.clone()))?;
            Ok(Value::Int(ch as i64))
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
        "is_int" => Ok(Value::Bool(matches!(args.first(), Some(Value::Int(_))))),
        "is_float" => Ok(Value::Bool(matches!(args.first(), Some(Value::Float(_))))),
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
            Some(Value::Int(_)) | Some(Value::Float(_)) => Ok(Value::Bool(true)),
            _ => Ok(Value::Bool(false)),
        },
        // `try_to_int(s)` / `try_to_float(s)` — return (value, Err) per Spec §3.9:
        // Err is nil on success, a str message on failure. Never aborts.
        "try_to_int" => {
            let (v, err) = match args.first() {
                Some(Value::Int(n))   => (Value::Int(*n), Value::Nil),
                Some(Value::Float(f)) => (Value::Int(*f as i64), Value::Nil),
                Some(Value::Bool(b))  => (Value::Int(*b as i64), Value::Nil),
                Some(Value::Str(s)) => {
                    let t = s.trim();
                    if let Ok(n) = t.parse::<i64>() {
                        (Value::Int(n), Value::Nil)
                    } else if let Ok(f) = t.parse::<f64>() {
                        (Value::Int(f as i64), Value::Nil)
                    } else {
                        (Value::Int(0), Value::Str(format!("try_to_int: not a number: {:?}", s)))
                    }
                }
                other => (Value::Int(0), Value::Str(format!(
                    "try_to_int: cannot convert {}",
                    other.map(|v| v.type_name()).unwrap_or("nil")))),
            };
            Ok(Value::Tuple(vec![v, err]))
        }
        "try_to_float" => {
            let (v, err) = match args.first() {
                Some(Value::Float(f)) => (Value::Float(*f), Value::Nil),
                Some(Value::Int(n))   => (Value::Float(*n as f64), Value::Nil),
                Some(Value::Bool(b))  => (Value::Float(*b as i64 as f64), Value::Nil),
                Some(Value::Str(s)) => match s.trim().parse::<f64>() {
                    Ok(f) => (Value::Float(f), Value::Nil),
                    Err(_) => (Value::Float(0.0), Value::Str(format!("try_to_float: not a number: {:?}", s))),
                },
                other => (Value::Float(0.0), Value::Str(format!(
                    "try_to_float: cannot convert {}",
                    other.map(|v| v.type_name()).unwrap_or("nil")))),
            };
            Ok(Value::Tuple(vec![v, err]))
        }

        "to_int" => {
            match args.first() {
                Some(Value::Int(n))   => Ok(Value::Int(*n)),
                Some(Value::Float(f)) => Ok(Value::Int(*f as i64)),
                Some(Value::Bool(b))  => Ok(Value::Int(*b as i64)),
                Some(Value::Str(s))   => {
                    let trimmed = s.trim();
                    if let Ok(n) = trimmed.parse::<i64>() {
                        Ok(Value::Int(n))
                    } else if let Ok(f) = trimmed.parse::<f64>() {
                        Ok(Value::Int(f as i64))
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
                Some(Value::Float(f)) => Ok(Value::Float(*f)),
                Some(Value::Int(n))   => Ok(Value::Float(*n as f64)),
                Some(Value::Bool(b))  => Ok(Value::Float(*b as i64 as f64)),
                Some(Value::Str(s))   => {
                    s.trim().parse::<f64>().map(Value::Float)
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
            Ok(Value::Int((f as f32).to_bits() as i64))
        }
        "f32_from_bits" => {
            let n = args.first().and_then(|v| v.as_int()).ok_or_else(|| {
                RuntimeError::at("f32_from_bits: integer argument required".to_string(), sp.clone())
            })?;
            Ok(Value::Float(f32::from_bits(n as u32) as f64))
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
                (Value::Int(x), Value::Int(lo), Value::Int(hi))     => Ok(Value::Int((*x).clamp(*lo, *hi))),
                (Value::Float(x), Value::Float(lo), Value::Float(hi)) => Ok(Value::Float(x.clamp(*lo, *hi))),
                (Value::Int(x), Value::Float(lo), Value::Float(hi))  => Ok(Value::Float((*x as f64).clamp(*lo, *hi))),
                (Value::Float(x), Value::Int(lo), Value::Int(hi))    => Ok(Value::Float(x.clamp(*lo as f64, *hi as f64))),
                _ => Err(RuntimeError::at("clamp: numeric args required", sp)),
            }
        }
        "len" => {
            match args.first() {
                Some(Value::Tensor(t)) => Ok(Value::Int(t.shape().first().copied().unwrap_or(0) as i64)),
                Some(Value::Tuple(vs)) => Ok(Value::Int(vs.len() as i64)),
                Some(Value::Str(s))    => Ok(Value::Int(s.len() as i64)),
                Some(Value::List(vs))  => Ok(Value::Int(vs.len() as i64)),
                Some(Value::Map(m))    => Ok(Value::Int(m.borrow().len() as i64)),
                Some(Value::Range { start, end, inclusive }) => {
                    let e = if *inclusive { *end + 1 } else { *end };
                    Ok(Value::Int((e - *start).max(0)))
                }
                _ => Err(RuntimeError::msg("len: requires tensor, tuple, str, list, or map")),
            }
        }
        "sum" => {
            if let Some(Value::Tensor(t)) = args.first() {
                Ok(Value::Float(t.sum()))
            } else if let Some(v) = args.first() {
                Ok(Value::Float(v.as_float().unwrap_or(0.0)))
            } else { Ok(Value::Float(0.0)) }
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
                    let diag_sum: f64 = (0..n).map(|i| t[[i, i]]).sum();
                    Ok(Value::Float(diag_sum))
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
                // special-case-to-0 diverged from).
                let n = t.len() as f64;
                Ok(Value::Float(t.sum() / n))
            } else { Ok(Value::Float(0.0)) }
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
            if let Some(Value::Tensor(t)) = args.first() {
                if args.len() == 1 {
                    return Ok(Value::Float(reduce(&mut t.iter().copied())));
                }
            }
            let vals: Vec<f64> = args.iter().filter_map(|v| v.as_float()).collect();
            Ok(Value::Float(reduce(&mut vals.into_iter())))
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
            Ok(Value::Float(m))
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
                    Value::Tuple(vs) | Value::List(vs) => vs.iter().map(|v| format!("{}", v)).collect(),
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
                    Ok(Value::Tuple(vec![Value::tensor_dt(arr, DType::Int), Value::Nil]))
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
        "list" => Ok(Value::List(args)), // list() → empty; list(1,2,3) → [1,2,3]
        "list_push" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_push: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_push: requires list and item args", sp)),
            };
            let item = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_push: requires item arg", sp))?;
            let mut new_lst = lst;
            new_lst.push(item);
            Ok(Value::List(new_lst))
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
            let last = lst.pop().unwrap();
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
            lst.into_iter().nth(idx).ok_or_else(||
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
            let val = args.into_iter().nth(2).ok_or_else(||
                RuntimeError::at("list_set: requires value arg", sp.clone()))?;
            let len = lst.len() as i64;
            let idx = if idx_raw < 0 { (len + idx_raw) as usize } else { idx_raw as usize };
            if idx >= lst.len() {
                return Err(RuntimeError::at(
                    format!("list_set: index {} out of range for list of length {}", idx_raw, len), sp));
            }
            lst[idx] = val;
            Ok(Value::List(lst))
        }
        "list_len" => {
            match args.first() {
                Some(Value::List(vs)) => Ok(Value::Int(vs.len() as i64)),
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
            result.extend(b);
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
            Ok(Value::List(lst[start..end].to_vec()))
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
                (Value::Int(a), Value::Int(b)) => a == b,
                (Value::Float(a), Value::Float(b)) => a == b,
                (Value::Int(a), Value::Float(b)) => (*a as f64) == *b,
                (Value::Float(a), Value::Int(b)) => *a == (*b as f64),
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
            lst.reverse();
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
            Ok(result)
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
            Ok(Value::List(keys))
        }
        "map_vals" => {
            let m = match args.first() {
                Some(Value::Map(m)) => m.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("map_vals: arg must be map, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("map_vals: requires map arg", sp)),
            };
            let vals: Vec<Value> = m.borrow().values().cloned().collect();
            Ok(Value::List(vals))
        }
        "map_len" => {
            match args.first() {
                Some(Value::Map(m)) => Ok(Value::Int(m.borrow().len() as i64)),
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
            Ok(Value::List(argv))
        }
        "exit" => {
            let code = args.first().and_then(|v| v.as_int()).unwrap_or(0);
            std::process::exit(code as i32);
        }

        // ── Time ──────────────────────────────────────────────────────────────
        "time_ms" => {
            let elapsed = self.start_time.elapsed();
            Ok(Value::Float(elapsed.as_secs_f64() * 1000.0))
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
                Value::Int(n) => Some(*n != 0),
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
            Ok(Value::Float(self.rand_uniform()))
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
            Ok(Value::Int(lo + n as i64))
        }
        "rand_normal" => {
            let mean = args.first().and_then(|v| v.as_float()).unwrap_or(0.0);
            let std  = args.get(1).and_then(|v| v.as_float()).unwrap_or(1.0);
            let u1 = self.rand_uniform().max(f64::MIN_POSITIVE);
            let u2 = self.rand_uniform();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            Ok(Value::Float(mean + std * z))
        }
        "rand_choice" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(Value::Tuple(vs)) => vs.clone(),
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
            match json_decode_str(&s) {
                Ok(v) => Ok(Value::Tuple(vec![v, Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Nil, Value::Str(format!("parse error: {}", e))])),
            }
        }

        // ── List functional combinators ───────────────────────────────────────
        "list_map" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_map: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_map: requires list and fn args", sp)),
            };
            let fn_val = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_map: requires fn arg", sp.clone()))?;
            let mut result = Vec::with_capacity(lst.len());
            for item in lst {
                let out = self.call_value(fn_val.clone(), vec![item], sp.clone())?;
                result.push(out);
            }
            Ok(Value::List(result))
        }
        "list_filter" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_filter: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_filter: requires list and fn args", sp)),
            };
            let fn_val = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_filter: requires fn arg", sp.clone()))?;
            let mut result = Vec::new();
            for item in lst {
                let out = self.call_value(fn_val.clone(), vec![item.clone()], sp.clone())?;
                if out.as_bool().unwrap_or(match &out {
                    Value::Int(n) => *n != 0,
                    Value::Float(x) => *x != 0.0,
                    Value::Nil => false,
                    _ => true,
                }) {
                    result.push(item);
                }
            }
            Ok(Value::List(result))
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
            let init = args.into_iter().nth(2).ok_or_else(||
                RuntimeError::at("list_reduce: requires init arg", sp.clone()))?;
            let mut acc = init;
            for item in lst {
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
            lst.sort_by(|a, b| cmp_value(a, b));
            Ok(Value::List(lst))
        }
        "list_sort_by" => {
            let mut lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_sort_by: first arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_sort_by: requires list and fn args", sp)),
            };
            let fn_val = args.into_iter().nth(1).ok_or_else(||
                RuntimeError::at("list_sort_by: requires fn arg", sp.clone()))?;
            // Compute keys first (avoids re-calling during sort which would need &mut self)
            let mut keys: Vec<Value> = Vec::with_capacity(lst.len());
            for item in &lst {
                let k = self.call_value(fn_val.clone(), vec![item.clone()], sp.clone())?;
                keys.push(k);
            }
            // Sort by keys using indices
            let mut indices: Vec<usize> = (0..lst.len()).collect();
            indices.sort_by(|&a, &b| cmp_value(&keys[a], &keys[b]));
            let sorted: Vec<Value> = indices.into_iter().map(|i| lst[i].clone()).collect();
            lst = sorted;
            Ok(Value::List(lst))
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
            let pairs: Vec<Value> = a.into_iter().zip(b.into_iter())
                .map(|(x, y)| Value::Tuple(vec![x, y]))
                .collect();
            Ok(Value::List(pairs))
        }
        "list_enumerate" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_enumerate: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_enumerate: requires list arg", sp)),
            };
            let pairs: Vec<Value> = lst.into_iter().enumerate()
                .map(|(i, v)| Value::Tuple(vec![Value::Int(i as i64), v]))
                .collect();
            Ok(Value::List(pairs))
        }
        "list_flatten" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_flatten: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_flatten: requires list arg", sp)),
            };
            let mut result = Vec::new();
            for item in lst {
                match item {
                    Value::List(inner) => result.extend(inner),
                    other => result.push(other),
                }
            }
            Ok(Value::List(result))
        }
        "list_uniq" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_uniq: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_uniq: requires list arg", sp)),
            };
            let mut result: Vec<Value> = Vec::new();
            for item in lst {
                let already_seen = result.iter().any(|prev| values_equal(prev, &item));
                if !already_seen { result.push(item); }
            }
            Ok(Value::List(result))
        }
        "list_sum" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("list_sum: arg must be list, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("list_sum: requires list arg", sp)),
            };
            // Preserve integer type if all elements are integers
            if lst.iter().all(|v| matches!(v, Value::Int(_))) {
                let total: i64 = lst.iter().filter_map(|v| v.as_int()).sum();
                Ok(Value::Int(total))
            } else {
                let total: f64 = lst.iter().filter_map(|v| v.as_float()).sum();
                Ok(Value::Float(total))
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
            let min = lst.into_iter().reduce(|a, b| {
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
            let max = lst.into_iter().reduce(|a, b| {
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
            let mut out = Vec::new();
            for item in lst {
                let result = self.call_value(f.clone(), vec![item], sp.clone())?;
                match result {
                    Value::List(vs) => out.extend(vs),
                    Value::Tuple(vs) => out.extend(vs),
                    other => out.push(other),
                }
            }
            Ok(Value::List(out))
        }
        "list_partition" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_partition: first arg must be list", sp)),
            };
            let f = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_partition: requires predicate arg", sp.clone()))?;
            let mut yes = Vec::new();
            let mut no = Vec::new();
            for item in lst {
                let r = self.call_value(f.clone(), vec![item.clone()], sp.clone())?;
                if matches!(r, Value::Bool(true)) { yes.push(item); } else { no.push(item); }
            }
            Ok(Value::Tuple(vec![Value::List(yes), Value::List(no)]))
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
            Ok(Value::List(lst.into_iter().take(take).collect()))
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
            Ok(Value::List(lst.into_iter().skip(skip).collect()))
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
            Ok(Value::Int(idx.map(|i| i as i64).unwrap_or(-1)))
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
            Ok(Value::Int(count as i64))
        }
        "list_any" => {
            let lst = match args.first() {
                Some(Value::List(vs)) => vs.clone(),
                _ => return Err(RuntimeError::at("list_any: first arg must be list", sp)),
            };
            let f = args.get(1).cloned()
                .ok_or_else(|| RuntimeError::at("list_any: requires predicate arg", sp.clone()))?;
            for item in lst {
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
            for item in lst {
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
            Ok(Value::Int(fnv1a_64(&s)))
        }
        "hash_crc32" => {
            let s = match args.first() {
                Some(Value::Str(s)) => s.clone(),
                Some(other) => return Err(RuntimeError::at(
                    format!("hash_crc32: arg must be str, got {}", other.type_name()), sp)),
                None => return Err(RuntimeError::at("hash_crc32: requires str arg", sp)),
            };
            Ok(Value::Int(crc32_ieee(&s)))
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
                                Value::List(Vec::new()),
                                Value::Str(format!("error: {}", e)),
                            ])),
                        }
                    }
                    Ok(Value::Tuple(vec![Value::List(names), Value::Nil]))
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::List(Vec::new()),
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
                Ok(meta) => Ok(Value::Tuple(vec![Value::Int(meta.len() as i64), Value::Nil])),
                Err(e) => Ok(Value::Tuple(vec![Value::Int(0), Value::Str(format!("error: {}", e))])),
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
                    Ok(Value::Tuple(vec![Value::Str(stdout), Value::Str(stderr), Value::Int(code)]))
                }
                Err(e) => Ok(Value::Tuple(vec![
                    Value::Str(String::new()),
                    Value::Str(format!("failed to spawn: {}", e)),
                    Value::Int(-1),
                ])),
            }
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
                Some(Value::Int(n)) => *n,
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
            Ok(Value::Int(count as i64))
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
                            Value::Int(m.start() as i64),
                        ])),
                        None => Ok(Value::Tuple(vec![
                            Value::Str(String::new()),
                            Value::Int(-1),
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
                    Ok(Value::List(matches))
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
                    Ok(Value::List(parts))
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
            Ok(Value::Int(chrono::Utc::now().timestamp_millis()))
        }
        "date_now_s" => {
            Ok(Value::Int(chrono::Utc::now().timestamp()))
        }
        "date_format" => {
            let timestamp_ms = match args.first() {
                Some(Value::Int(n)) => *n,
                Some(Value::Float(x)) => *x as i64,
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
                    Ok(Value::Tuple(vec![Value::Int(ts_ms), Value::Nil]))
                }
                Err(_) => {
                    // Try parsing as a date only (NaiveDate)
                    match chrono::NaiveDate::parse_from_str(&date_str, &format_str) {
                        Ok(d) => {
                            let dt = d.and_hms_opt(0, 0, 0).unwrap();
                            let ts_ms = dt.and_utc().timestamp_millis();
                            Ok(Value::Tuple(vec![Value::Int(ts_ms), Value::Nil]))
                        }
                        Err(e) => Ok(Value::Tuple(vec![
                            Value::Int(0),
                            Value::Str(format!("parse error: {}", e)),
                        ])),
                    }
                }
            }
        }
        "date_add_ms" => {
            let timestamp_ms = match args.first() {
                Some(Value::Int(n)) => *n,
                Some(Value::Float(x)) => *x as i64,
                _ => return Err(RuntimeError::at("date_add_ms: timestamp_ms must be integer", sp)),
            };
            let delta_ms = match args.get(1) {
                Some(Value::Int(n)) => *n,
                Some(Value::Float(x)) => *x as i64,
                _ => return Err(RuntimeError::at("date_add_ms: delta_ms must be integer", sp)),
            };
            Ok(Value::Int(timestamp_ms + delta_ms))
        }
        "date_diff_ms" => {
            let ts_a = match args.first() {
                Some(Value::Int(n)) => *n,
                Some(Value::Float(x)) => *x as i64,
                _ => return Err(RuntimeError::at("date_diff_ms: ts_a must be integer", sp)),
            };
            let ts_b = match args.get(1) {
                Some(Value::Int(n)) => *n,
                Some(Value::Float(x)) => *x as i64,
                _ => return Err(RuntimeError::at("date_diff_ms: ts_b must be integer", sp)),
            };
            Ok(Value::Int(ts_a - ts_b))
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
            if total == 0 { return Ok(Value::Float(0.0)); }
            let zeros = t.iter().filter(|&&x| x == 0.0).count();
            Ok(Value::Float(zeros as f64 / total as f64))
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
    let t = required_tensor(args, 0, "variance", &sp)?;
    let n = t.len() as f64;
    if n == 0.0 { return Ok(Value::Float(0.0)); }
    let mean = t.sum() / n;
    let var = t.iter().map(|v| { let d = v - mean; d * d }).sum::<f64>() / n;
    Ok(Value::Float(var))
}

/// `pull_to_mean(t, alpha)` — one variance-minimizing pass over all elements:
/// out[i] = t[i] + alpha * (mean(t) - t[i]).
fn builtin_pull_to_mean(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor(args, 0, "pull_to_mean", &sp)?;
    let alpha = args.get(1).and_then(|v| v.as_float()).ok_or_else(|| RuntimeError::at(
        "pull_to_mean: missing alpha arg".to_string(), sp.clone()))?;
    let n = t.len() as f64;
    let mean = if n > 0.0 { t.sum() / n } else { 0.0 };
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

/// `sum_along(t, axis)` / `mean_along(t, axis)`.
fn builtin_reduce_along(args: &[Value], sp: Span, name: &str, mean: bool) -> EvalResult<Value> {
    let t = required_tensor(args, 0, name, &sp)?;
    let axis = reduce_axis(args, t.ndim(), name, &sp)?;
    let summed = t.sum_axis(Axis(axis));
    let out = if mean {
        let n = t.shape()[axis] as f64;
        summed.mapv(|v| if n > 0.0 { v / n } else { 0.0 })
    } else {
        summed
    };
    Ok(Value::tensor(out.into_dyn()))
}

/// `max_along(t, axis)` — maximum reduced along one axis (axis dropped).
fn builtin_max_along(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor(args, 0, "max_along", &sp)?;
    let axis = reduce_axis(args, t.ndim(), "max_along", &sp)?;
    if t.shape()[axis] == 0 {
        return Err(RuntimeError::at("max_along: empty axis".to_string(), sp));
    }
    let out = t.fold_axis(Axis(axis), f64::NEG_INFINITY, |&acc, &v| acc.max(v));
    Ok(Value::tensor(out.into_dyn()))
}

/// `min_along(t, axis)` — minimum reduced along one axis (axis dropped).
fn builtin_min_along(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor(args, 0, "min_along", &sp)?;
    let axis = reduce_axis(args, t.ndim(), "min_along", &sp)?;
    if t.shape()[axis] == 0 {
        return Err(RuntimeError::at("min_along: empty axis".to_string(), sp));
    }
    let out = t.fold_axis(Axis(axis), f64::INFINITY, |&acc, &v| acc.min(v));
    Ok(Value::tensor(out.into_dyn()))
}

/// `variance_along(t, axis)` — population variance reduced along one axis.
fn builtin_variance_along(args: &[Value], sp: Span) -> EvalResult<Value> {
    let t = required_tensor(args, 0, "variance_along", &sp)?;
    let axis = reduce_axis(args, t.ndim(), "variance_along", &sp)?;
    let n = t.shape()[axis] as f64;
    let mean = t.mean_axis(Axis(axis)).ok_or_else(|| RuntimeError::at(
        "variance_along: empty axis".to_string(), sp.clone()))?;
    // var = mean(x^2) - mean(x)^2 reduced along the axis.
    let mean_sq = t.mapv(|v| v * v).mean_axis(Axis(axis)).ok_or_else(|| RuntimeError::at(
        "variance_along: empty axis".to_string(), sp.clone()))?;
    let _ = n;
    let var = &mean_sq - &mean.mapv(|m| m * m);
    Ok(Value::tensor(var.into_dyn()))
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
            Ok(Value::Float(activation_f64(name, x) as f32 as f64))
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
        return Ok(Value::Int(best_i as i64));
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
    let weights = builtin_softmax(&[Value::tensor(scaled), Value::Int(-1)], sp.clone())?;
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
                &[Value::tensor(scaled), Value::Int(-1)], sp.clone())?;
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

    /// Walk the tape in reverse, propagating gradients via per-op VJP rules.
    /// `grads[id]` holds the accumulated dL/d(node id) value, or None if no
    /// gradient has reached the node yet.
    fn backward(&self, grads: &mut Vec<Option<Value>>) -> EvalResult<()> {
        for i in (0..self.nodes.len()).rev() {
            let n = &self.nodes[i];
            let Some(g_out) = grads[i].clone() else { continue; };
            match &n.op {
                TapeOp::Input { .. } | TapeOp::Const => {
                    // No upstream to propagate to.
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
                                &[Value::tensor(scaled), Value::Int(-1)],
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
                    accumulate(grads, n.inputs[0], Value::Float(g * kind.derivative(x, y)));
                }
                TapeOp::Broadcast => {
                    // c = broadcast(s) (scalar → tensor) → dL/ds = sum(dL/dc)
                    let g = as_tensor(&g_out)?;
                    accumulate(grads, n.inputs[0], Value::Float(g.iter().sum()));
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
    fn backward_symbolic(&mut self, loss_node: usize) -> EvalResult<Vec<Option<usize>>> {
        let n = self.nodes.len();
        let mut adj: Vec<Option<usize>> = vec![None; n];
        let seed = self.push_const(Value::Float(1.0));
        adj[loss_node] = Some(seed);
        for i in (0..n).rev() {
            let Some(g) = adj[i] else { continue };
            let op = self.nodes[i].op.clone();
            let inputs = self.nodes[i].inputs.clone();
            let g_val = self.nodes[g].value.clone();
            match op {
                TapeOp::Input { .. } | TapeOp::Const => {}
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
        _ => Ok(Value::Float(a.as_float().unwrap_or(0.0) + b.as_float().unwrap_or(0.0))),
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
        _ => Ok(Value::Float(x.as_float().unwrap_or(0.0) * y.as_float().unwrap_or(0.0))),
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
        _ => Ok(Value::Float(x.as_float().unwrap_or(0.0) / y.as_float().unwrap_or(1.0))),
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
            Value::Tensor(t) => Value::Float(t.data.iter().sum()),
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
        Value::Float(x)  => Ok(Value::Float(-x)),
        Value::Int(n)    => Ok(Value::Int(-n)),
        other => Err(RuntimeError::msg(format!("cannot negate {}", other.type_name()))),
    }
}

fn scale_value(v: &Value, s: f64) -> EvalResult<Value> {
    match v {
        Value::Tensor(t) => Ok(Value::tensor(t.mapv(|x| x * s))),
        Value::Float(x)  => Ok(Value::Float(x * s)),
        Value::Int(n)    => Ok(Value::Float(*n as f64 * s)),
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

fn json_encode_value(v: &Value) -> String {
    match v {
        Value::Nil       => "null".to_string(),
        Value::Bool(b)   => if *b { "true".to_string() } else { "false".to_string() },
        Value::Int(n)    => n.to_string(),
        Value::Float(x)  => {
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
        Value::Tensor(t) => {
            // Serialize as a flat JSON array of its elements (row-major). Reuses
            // the scalar float formatting so 1.5 -> "1.5", 2.0 -> "2" (#190).
            let parts: Vec<String> = t.iter().map(|x| json_encode_value(&Value::Float(*x))).collect();
            format!("[{}]", parts.join(","))
        }
        // Fns, structs, ranges, etc. — fall back to null per spec.
        _ => "null".to_string(),
    }
}

fn json_encode_str(s: &str) -> String {
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

impl<'a> JsonParser<'a> {
    fn new(s: &'a str) -> Self { Self { src: s.as_bytes(), pos: 0, depth: 0 } }

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
            Some(got) => Err(format!("expected '{}', got '{}'", b as char, got as char)),
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
            Some(c) => Err(format!("unexpected byte '{}' at position {}", c as char, self.pos)),
            None => Err("unexpected end of input".to_string()),
        }
    }

    fn consume_literal(&mut self, expected: &[u8]) -> Result<(), String> {
        for &b in expected {
            match self.consume() {
                Some(got) if got == b => {}
                Some(got) => return Err(format!("expected '{}', got '{}'", b as char, got as char)),
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
                            // 4 hex digits
                            let mut hex = String::new();
                            for _ in 0..4 {
                                match self.consume() {
                                    Some(b) => hex.push(b as char),
                                    None => return Err("truncated \\uXXXX".to_string()),
                                }
                            }
                            let cp = u32::from_str_radix(&hex, 16)
                                .map_err(|_| format!("invalid \\u{}", hex))?;
                            let c = char::from_u32(cp)
                                .ok_or_else(|| format!("invalid codepoint U+{:04X}", cp))?;
                            s.push(c);
                        }
                        Some(c) => return Err(format!("unknown escape \\{}", c as char)),
                        None => return Err("truncated escape".to_string()),
                    }
                }
                Some(b) => s.push(b as char),
                None => return Err("unterminated string".to_string()),
            }
        }
        Ok(s)
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
            Ok(Value::Float(x))
        } else {
            let n: i64 = s.parse().map_err(|_| format!("invalid integer: {}", s))?;
            Ok(Value::Int(n))
        }
    }

    fn parse_array(&mut self) -> Result<Value, String> {
        self.expect(b'[')?;
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') { self.consume(); return Ok(Value::List(items)); }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.consume() {
                Some(b']') => break,
                Some(b',') => {}
                Some(c) => return Err(format!("expected ',' or ']', got '{}'", c as char)),
                None => return Err("unterminated array".to_string()),
            }
        }
        Ok(Value::List(items))
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
                Some(c) => return Err(format!("expected ',' or '}}', got '{}'", c as char)),
                None => return Err("unterminated object".to_string()),
            }
        }
        Ok(Value::Map(Rc::new(RefCell::new(m))))
    }
}

fn json_decode_str(s: &str) -> Result<Value, String> {
    let mut parser = JsonParser::new(s);
    let v = parser.parse_value()?;
    parser.skip_ws();
    if parser.pos < parser.src.len() {
        return Err(format!("trailing content at position {}", parser.pos));
    }
    Ok(v)
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
            Value::Int(_)   => 1,
            Value::Float(_) => 2,
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
        (Value::Int(x),   Value::Int(y))   => x.cmp(y),
        (Value::Float(x), Value::Float(y)) => x.total_cmp(y),
        (Value::Str(x),   Value::Str(y))   => x.cmp(y),
        _                                  => Ordering::Equal,
    }
}

/// Compare two Values for equality (for list_uniq).
fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Int(x), Value::Float(y)) => (*x as f64) == *y,
        (Value::Float(x), Value::Int(y)) => *x == (*y as f64),
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
