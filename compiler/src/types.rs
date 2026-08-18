/// demoniC resolved types — Phase 3 typechecker.
///
/// `TyType` is the type-checker's view of a type: like `ast::Type` but with
/// shape expressions converted to `SymDim`, generics resolved against the
/// surrounding scope, and known stdlib-shape constraints inlined.
///
/// `Env` is the symbol table. It's lexically scoped — push/pop on block
/// entry, lookup walks scopes inside-out. Models and top-level functions
/// live in dedicated namespaces, not the lexical scope chain.

use std::collections::HashMap;

use crate::ast::ScalarType;
use crate::shape::{Shape, SymDim};

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TyType {
    Scalar(ScalarType),
    /// An untyped integer literal (`5`) before it adopts a context type (#295).
    /// Compatible with any *integral* scalar so `let x: i32 = 5` checks, but
    /// distinct from a concrete `i64` value so strict numeric typing (#284) can
    /// still reject `let y: i32 = some_i64`. Carries the literal value for
    /// range diagnostics (`let x: i8 = 300`). Concretizes to `i64` when bound
    /// without an integral context (`let x = 5`).
    IntLit(i64),
    /// An untyped float literal (`5.0`) before it adopts a context type
    /// (#284, the float analog of [`IntLit`]). Compatible with any *float*
    /// scalar so `fn f() -> f64 { 5.0 }` and `let x: f32 = 5.0` both check,
    /// but distinct from a concrete `f32`/`f64` so strict numeric typing
    /// (#284) still rejects `let y: f32 = some_f64`. Concretizes to `f32`
    /// (the historical default) when bound without a float context. The value
    /// is stored as `f64::to_bits()` so `TyType` can keep deriving `Eq`/`Hash`.
    FloatLit(u64),
    Tensor(Box<TyType>, Shape),
    View(Box<TyType>, Shape),
    KV(Box<TyType>, Shape),
    Mesh(Vec<(String, SymDim)>),
    Fn { params: Vec<TyType>, ret: Box<TyType> },
    Tuple(Vec<TyType>),
    Array(Box<TyType>, SymDim),

    /// `*T` — raw pointer, extern fn boundary only (§3.12)
    RawPtr(Box<TyType>),

    /// Named (model) type instantiated with concrete generic args.
    /// We store the model name; method resolution looks up in Env.models.
    Named { name: String, args: Vec<TyType> },

    /// A C-like enum type (#336). Carries the enum's name; variant values are
    /// i64 ordinals. Distinct from a plain `i64` so `match` can do real
    /// (closed-set) exhaustiveness and a stray int can't silently flow in.
    Enum(String),

    /// The `nil` unit type.
    Unit,

    /// Dynamic string-keyed hash map (`map_new()`, or a `m: map` annotation).
    /// Reference-semantic and heterogeneous-valued, so it stays `Unknown`-
    /// compatible everywhere (see `compatible_with`); the distinct variant
    /// exists only so the checker can flag `for … in <map>` — maps are not
    /// iterable (#204).
    Map,

    /// Type couldn't be determined; conservative ⊥. Suppresses cascade errors.
    Unknown,

    /// Module namespace type
    Module { alias: String, path: std::path::PathBuf },
}

impl TyType {
    #[allow(dead_code)]
    pub fn scalar(s: ScalarType) -> Self { TyType::Scalar(s) }
    #[allow(dead_code)]
    pub fn unknown() -> Self { TyType::Unknown }
    #[allow(dead_code)]
    pub fn unit() -> Self { TyType::Unit }

    /// Is this a tensor-like type (Tensor / View / KV)?
    pub fn as_tensor_like(&self) -> Option<(&TyType, &Shape)> {
        match self {
            TyType::Tensor(t, s) | TyType::View(t, s) | TyType::KV(t, s) => Some((t, s)),
            _ => None,
        }
    }

    /// Is this an integral type (signed/unsigned int family, or an untyped int
    /// literal)? Floats and trit are excluded. Used by literal inference (#295):
    /// an int literal may adopt an integral context but not a float one.
    #[allow(dead_code)]
    pub fn is_integral(&self) -> bool {
        matches!(self, TyType::IntLit(_)) || matches!(self, TyType::Scalar(s) if matches!(s,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 |
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64 |
            ScalarType::Int4 | ScalarType::Int8
        ))
    }

    /// Is this a float-family scalar (or an untyped float literal)? Used by
    /// literal inference (#284): a float literal may adopt a float context but
    /// not an integral one.
    #[allow(dead_code)]
    pub fn is_float(&self) -> bool {
        matches!(self, TyType::FloatLit(_)) || matches!(self, TyType::Scalar(s) if matches!(s,
            ScalarType::F16 | ScalarType::Bf16 | ScalarType::Tf32 |
            ScalarType::F32 | ScalarType::F64 |
            ScalarType::Fp8E4M3 | ScalarType::Fp8E5M2
        ))
    }

    /// Is this a numeric scalar (int or float family)?
    pub fn is_numeric(&self) -> bool {
        // An untyped int/float literal is numeric (it adopts a numeric context).
        matches!(self, TyType::IntLit(_) | TyType::FloatLit(_)) || matches!(self, TyType::Scalar(s) if matches!(s,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 |
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64 |
            ScalarType::Int4 | ScalarType::Int8 |
            ScalarType::F16 | ScalarType::Bf16 | ScalarType::Tf32 |
            ScalarType::F32 | ScalarType::F64 |
            ScalarType::Fp8E4M3 | ScalarType::Fp8E5M2 |
            ScalarType::Trit
        ))
    }

    /// Compatibility for assignment / passing-as-arg. Accepts Unknown as
    /// pessimistic-match-anything. Otherwise structural equality.
    pub fn compatible_with(&self, other: &TyType) -> bool {
        use TyType::*;
        if matches!(self, Unknown) || matches!(other, Unknown) { return true; }
        // Maps are heterogeneous-valued and were `Unknown` before #204 gave them
        // a distinct variant; keep them match-anything so no prior map code newly
        // errors. The variant exists for the `for … in <map>` lint, not stricter
        // map type-checking.
        if matches!(self, Map) || matches!(other, Map) { return true; }
        // `nil` the type and Unit are the same thing.
        // `nil` is also compatible with `str` in error position: the convention
        // is Err=nil (success) | str (failure), so `(T, nil)` satisfies `(T, str)`.
        if matches!((self, other),
            (Unit, Scalar(crate::ast::ScalarType::Nil))
            | (Scalar(crate::ast::ScalarType::Nil), Unit)
            | (Unit, Scalar(crate::ast::ScalarType::Str))
            | (Scalar(crate::ast::ScalarType::Str), Unit)) {
            return true;
        }
        // Err is the nil-or-str error alias (Spec §3.9): compatible with nil, str, and Err itself.
        let is_err_named = |t: &TyType| matches!(t, Named { name, .. } if name == "Err");
        if is_err_named(self) {
            if matches!(other, Unit | Scalar(crate::ast::ScalarType::Nil) | Scalar(crate::ast::ScalarType::Str))
                || is_err_named(other) {
                return true;
            }
        }
        if is_err_named(other) {
            if matches!(self, Unit | Scalar(crate::ast::ScalarType::Nil) | Scalar(crate::ast::ScalarType::Str))
                || is_err_named(self) {
                return true;
            }
        }
        match (self, other) {
            // An untyped integer literal (#295/#284). A literal may adopt an
            // integral context (i8..u64, int4/int8) but NOT a float one: per
            // SPEC §149 there are no implicit numeric conversions, so `let
            // y: f32 = 5` must be written `5.0` (or `5 as f32`). Two untyped
            // int literals are mutually compatible.
            (IntLit(_), IntLit(_)) => true,
            (IntLit(_), Scalar(b)) => TyType::Scalar(b.clone()).is_integral(),
            (Scalar(a), IntLit(_)) => TyType::Scalar(a.clone()).is_integral(),
            // A float literal adopts a float context but not an integral one
            // (#284). Mixed untyped literals (`1` vs `2.0`) stay compatible —
            // the int literal can promote to the float one.
            (FloatLit(_), FloatLit(_))
            | (IntLit(_), FloatLit(_))
            | (FloatLit(_), IntLit(_)) => true,
            (FloatLit(_), Scalar(b)) => TyType::Scalar(b.clone()).is_float(),
            (Scalar(a), FloatLit(_)) => TyType::Scalar(a.clone()).is_float(),
            // #284: strict numeric typing — no implicit widening/narrowing
            // between distinct scalar types (SPEC §149). Conversions are
            // explicit via `x as T`.
            (Scalar(a), Scalar(b)) => a == b,
            (Tensor(at, ash), Tensor(bt, bsh))
            | (View(at, ash), View(bt, bsh))
            | (KV(at, ash), KV(bt, bsh)) => at.compatible_with(bt) && ash.same(bsh),
            // #234: fixed-size arrays (`[Model; N]`, `[KV[..]; N]`) — compatible
            // when the element types match and the sizes are provably equal.
            // Without this arm two identical `[Expr; 8]` fell through to `false`,
            // so a function taking a model-array param (the core of an AST
            // walker) couldn't be called — even recursively with its own param.
            (Array(at, an), Array(bt, bn)) =>
                at.compatible_with(bt) && matches!(an.equivalent(bn), crate::shape::Equiv::Equal),
            (Unit, Unit) => true,
            (Tuple(a), Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.compatible_with(y))
            }
            (Named { name: a, args: aa }, Named { name: b, args: ba }) => {
                a == b && aa.len() == ba.len() && aa.iter().zip(ba).all(|(x, y)| x.compatible_with(y))
            }
            // #336: enums are nominal — same name only. An int does NOT implicitly
            // flow into an enum (or vice versa); use `as i64` for the ordinal.
            (Enum(a), Enum(b)) => a == b,
            (Fn { params: ap, ret: ar }, Fn { params: bp, ret: br }) => {
                ap.len() == bp.len()
                    && ap.iter().zip(bp).all(|(x, y)| x.compatible_with(y))
                    && ar.compatible_with(br)
            }
            (Mesh(a), Mesh(b)) => a.len() == b.len()
                && a.iter().zip(b).all(|((n1, _), (n2, _))| n1 == n2),
            _ => false,
        }
    }
}

impl std::fmt::Display for TyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use TyType::*;
        match self {
            Scalar(s) => write!(f, "{:?}", s),
            IntLit(_) => write!(f, "{{integer}}"),
            FloatLit(_) => write!(f, "{{float}}"),
            Enum(name) => write!(f, "{}", name),
            Tensor(t, sh) => write!(f, "Tensor[{}, {}]", t, sh),
            View(t, sh)   => write!(f, "View[{}, {}]", t, sh),
            KV(t, sh)     => write!(f, "KV[{}, {}]", t, sh),
            Mesh(axes) => {
                write!(f, "Mesh[")?;
                for (i, (n, sz)) in axes.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}={}", n, sz)?;
                }
                write!(f, "]")
            }
            Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
            Array(t, n) => write!(f, "[{}; {}]", t, n),
            RawPtr(t) => write!(f, "*{}", t),
            Named { name, args } => {
                write!(f, "{}", name)?;
                if !args.is_empty() {
                    write!(f, "[")?;
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", a)?;
                    }
                    write!(f, "]")?;
                }
                Ok(())
            }
            Unit => write!(f, "nil"),
            Map => write!(f, "map"),
            Unknown => write!(f, "?"),
            Module { alias, path } => write!(f, "module({} {:?})", alias, path),
        }
    }
}

// ─── Function / Model signatures ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FnSig {
    #[allow(dead_code)]
    pub shape_params: Vec<String>,
    pub params: Vec<(String, TyType)>,
    pub ret: TyType,
}

/// Returns the declared `FnSig` for a fixed-arity stdlib builtin, or `None`
/// for truly variadic builtins (`print`, `panic`, `argmax`, `allreduce`, etc.).
///
/// Parameter types are `Unknown` throughout because the type system does not
/// yet support generic type/shape variables.  Arity is always exact.
pub fn builtin_sig(name: &str) -> Option<FnSig> {
    let mk = |n: usize, ret: TyType| FnSig {
        shape_params: Vec::new(),
        params: (0..n).map(|i| (format!("_{}", i), TyType::Unknown)).collect(),
        ret,
    };
    let u = TyType::Unknown;
    match name {
        // Elementwise math — 1 arg, return mirrors input type (generic over T).
        "sqrt" | "exp" | "log" | "abs" | "sin" | "cos" | "floor" | "ceil"
        | "tan" | "asin" | "acos" | "atan" | "log2" | "log10" => Some(mk(1, u)),
        // Elementwise activations — 1 arg, f32 scalar or f32 tensor (any shape).
        // Return mirrors the input (scalar->scalar, tensor->same-shape tensor),
        // modelled as `Unknown` like the other elementwise math builtins.
        "relu" | "sigmoid" | "tanh" | "gelu" | "silu" | "elu" | "mish" => Some(mk(1, u)),
        // 2-arg math
        "atan2" | "hypot" => Some(mk(2, u)),
        "gcd" => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::I64))),  // #335
        "sort" => Some(mk(1, u)),   // #335: tensor -> same-shape sorted tensor
        "median" => Some(mk(1, u)), // #335: tensor -> scalar (element type)
        // isclose(a, b) -> bool
        "isclose" => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Bool))),
        // File I/O: (content, err) or (nil, err) tuples
        "read_file" => Some(mk(1, TyType::Unknown)),
        // read_bytes(path) -> (Tensor[i64,[N]], err): binary read, each byte 0-255 as an i64.
        "read_bytes" => Some(mk(1, TyType::Unknown)),
        "write_file" | "append_file" => Some(mk(2, TyType::Unknown)),
        "file_exists" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Bool))),
        // chr(n: int) -> str
        "chr" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Str))),
        // len(x) -> i64
        "len" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::I64))),
        // Reductions — 1 tensor arg; return is the element type (generic over T).
        // `trace` additionally requires a square 2D tensor (checked at runtime).
        "sum" | "mean" | "trace" => Some(mk(1, u)),
        // `diag` extracts the diagonal of a square 2D tensor: [N,N] -> [N].
        // Return shape is input-dependent (no generic-var support yet, #175),
        // so it's `Unknown` and the square-2D constraint is checked at runtime.
        "diag" => Some(mk(1, u)),
        // softmax(x, axis=-1) — axis is optional; variadic bypass.
        // attn(q,k,v,mask?) / attn_gqa(q,k,v,mask?) — mask optional; variadic bypass.
        // rope(x,cos,sin) — fixed 3 args.
        "rope" => Some(mk(3, u)),
        // rms_norm(x, g, eps) -> Tensor same shape as x.
        "rms_norm" => Some(mk(3, u)),
        // layer_norm(x, g, b, eps) -> Tensor same shape as x.
        "layer_norm" => Some(mk(4, u)),
        // embed(vocab, token_ids) -> Tensor[T, [batch..., D]] per §2.3.
        "embed" => Some(mk(2, u)),
        // Linear algebra: solve(A, b) -> x, inv(A) -> A⁻¹, lstsq(A, b) -> x
        "solve" | "lstsq" => Some(mk(2, u.clone())),
        "inv"             => Some(mk(1, u.clone())),
        // Dynamic collections: list
        "list" => Some(mk(0, u.clone())),
        "list_push" => Some(mk(2, u.clone())),
        "list_pop" => Some(mk(1, u.clone())),
        "list_get" => Some(mk(2, u.clone())),
        "list_set" => Some(mk(3, u.clone())),
        "list_len" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::I64))),
        "list_concat" => Some(mk(2, u.clone())),
        "list_slice" => Some(mk(3, u.clone())),
        "list_contains" => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Bool))),
        "list_rev" => Some(mk(1, u.clone())),
        // Dynamic collections: map
        // The map creators/mutators return the map itself (Value::Map at runtime),
        // so they carry the Map type — this is what lets the #204 for-loop lint
        // see `for … in m` where `m = map_new()`.
        "map"          => Some(mk(0, TyType::Map)),
        "map_new"      => Some(mk(0, TyType::Map)),    // alias for map()
        "map_set"      => Some(mk(3, TyType::Map)),
        "map_get"      => Some(mk(2, u.clone())),
        "map_has"      => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Bool))),
        "map_contains" => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Bool))),
        "map_del"      => Some(mk(2, TyType::Map)),
        "map_keys" => Some(mk(1, u.clone())),
        "map_vals" => Some(mk(1, u.clone())),
        "map_len" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::I64))),
        // Process / environment
        "env_var" => Some(mk(1, u.clone())),
        "argv" => Some(mk(0, u.clone())),
        "exit" => Some(mk(1, TyType::Unit)),
        // CLI argument parsing
        "cli_arg"              => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Str))),
        "cli_flag"             => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Bool))),
        "cli_positional"       => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Str))),
        "cli_positional_count" => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::I64))),
        // Time
        "time_ms" => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::F64))),
        "sleep_ms" => Some(mk(1, TyType::Unit)),
        // Terminal I/O
        "flush"        => Some(mk(0, TyType::Unit)),
        "set_raw_mode" => Some(mk(1, TyType::Unit)),
        "read_char_nb" => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::Str))),
        "read_char"    => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::Str))),
        // Extended RNG
        "rand_seed"   => Some(mk(1, TyType::Unit)),
        "rand_float"  => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::F64))),
        "rand_int"    => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::I64))),
        "rand_normal" => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::F64))),
        "rand_choice" => Some(mk(1, u)),
        // JSON
        "json_encode" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Str))),
        "json_decode" => Some(mk(1, TyType::Unknown)),
        // List functional combinators
        "list_map" => Some(mk(2, u.clone())),
        "list_filter" => Some(mk(2, u.clone())),
        "list_reduce" => Some(mk(3, u.clone())),
        "list_sort" => Some(mk(1, u.clone())),
        "list_sort_by" => Some(mk(2, u.clone())),
        "list_zip" => Some(mk(2, u.clone())),
        "list_enumerate" => Some(mk(1, u.clone())),
        "list_flatten" => Some(mk(1, u.clone())),
        "list_uniq" => Some(mk(1, u.clone())),
        "list_sum" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::F64))),
        "list_min" => Some(mk(1, u.clone())),
        "list_max" => Some(mk(1, u.clone())),
        // Hashing
        "hash_fnv"  => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::I64))),
        "hash_crc32" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::I64))),
        // Filesystem operations
        "get_cwd" => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::Str))),
        "list_dir" => Some(mk(1, TyType::Unknown)),
        "make_dir" => Some(mk(1, TyType::Unknown)),
        "delete_file" => Some(mk(1, TyType::Unknown)),
        "delete_dir" => Some(mk(1, TyType::Unknown)),
        "rename_file" => Some(mk(2, TyType::Unknown)),
        "file_size" => Some(mk(1, TyType::Unknown)),
        "path_join" => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Str))),
        "path_dirname" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Str))),
        "path_basename" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Str))),
        "path_exists" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Bool))),
        "path_is_dir" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Bool))),
        "path_is_file" => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Bool))),
        // Process execution
        "exec_cmd" => Some(mk(2, TyType::Unknown)),
        // Ports (#402, PORTS.md §2) — each returns `(_, Err)`
        "port_open"  => Some(mk(1, TyType::Unknown)),
        "port_call"  => Some(mk(3, TyType::Unknown)),
        "port_close" => Some(mk(1, TyType::Unknown)),
        // Regex
        "regex_match"       => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Bool))),
        "regex_find"        => Some(mk(2, TyType::Unknown)),
        "regex_find_all"    => Some(mk(2, TyType::Unknown)),
        "regex_replace"     => Some(mk(3, TyType::Scalar(crate::ast::ScalarType::Str))),
        "regex_replace_all" => Some(mk(3, TyType::Scalar(crate::ast::ScalarType::Str))),
        "regex_split"       => Some(mk(2, TyType::Unknown)),
        // Compression
        "gzip_compress"   => Some(mk(1, TyType::Unknown)),
        "gzip_decompress" => Some(mk(1, TyType::Unknown)),
        "zlib_compress"   => Some(mk(1, TyType::Unknown)),
        "zlib_decompress" => Some(mk(1, TyType::Unknown)),
        // HTTP networking
        "http_get"       => Some(mk(1, TyType::Unknown)),
        "http_post"      => Some(mk(3, TyType::Unknown)),
        "http_post_json" => Some(mk(2, TyType::Unknown)),
        // Date/time
        "date_now_ms"  => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::I64))),
        "date_now_s"   => Some(mk(0, TyType::Scalar(crate::ast::ScalarType::I64))),
        "date_format"  => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::Str))),
        "date_parse"   => Some(mk(2, TyType::Unknown)),
        "date_add_ms"  => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::I64))),
        "date_diff_ms" => Some(mk(2, TyType::Scalar(crate::ast::ScalarType::I64))),
        // Typed print variants
        "print_i64"    => Some(mk(1, TyType::Unit)),
        "print_f64"    => Some(mk(1, TyType::Unit)),
        "print_bool"   => Some(mk(1, TyType::Unit)),
        "print_nil"    => Some(mk(0, TyType::Unit)),
        "print_tensor" => Some(mk(1, TyType::Unit)),
        // Variance-trait primitives
        "variance"           => Some(mk(1, TyType::Unknown)),
        "pull_to_mean"       => Some(mk(2, TyType::Unknown)),
        "sum_along"          => Some(mk(2, TyType::Unknown)),
        "mean_along"         => Some(mk(2, TyType::Unknown)),
        "max_along"          => Some(mk(2, TyType::Unknown)),
        "min_along"          => Some(mk(2, TyType::Unknown)),
        "variance_along"     => Some(mk(2, TyType::Unknown)),
        "pull_to_mean_along" => Some(mk(3, TyType::Unknown)),
        // Type conversions
        "to_str"   => Some(mk(1, TyType::Scalar(ScalarType::Str))),
        // aliases under the equivalent Rust/Python names.
        "to_string" => Some(mk(1, TyType::Scalar(ScalarType::Str))),
        "to_int"   => Some(mk(1, TyType::Scalar(ScalarType::I64))),
        "to_float" => Some(mk(1, TyType::Scalar(ScalarType::F64))),
        // IEEE-754 bit reinterpret (#189): no arithmetic conversion, just a
        // reinterpret of the 32-bit pattern. Unblocks fast-inverse-sqrt and
        // exact f32 weight decoding. The bits are returned zero-extended into
        // i64 (demoniC's native integer); `f32_from_bits` reads the low 32 bits.
        "f32_to_bits"   => Some(mk(1, TyType::Scalar(ScalarType::I64))),
        "f32_from_bits" => Some(mk(1, TyType::Scalar(ScalarType::F32))),
        // Runtime type introspection (#184)
        "typeof"    => Some(mk(1, TyType::Scalar(ScalarType::Str))),
        "is_int"    => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_float"  => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_str"    => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_bool"   => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_list"   => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_map"    => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_nil"    => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_fn"     => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "is_tensor" => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        // Safe numeric parse (#185)
        "is_numeric"   => Some(mk(1, TyType::Scalar(ScalarType::Bool))),
        "try_to_int"   => Some(mk(1, TyType::Unknown)),  // returns (i64, Err)
        "try_to_float" => Some(mk(1, TyType::Unknown)),  // returns (f64, Err)
        "to_hex"   => Some(mk(1, TyType::Scalar(ScalarType::Str))),
        "to_bin"   => Some(mk(1, TyType::Scalar(ScalarType::Str))),
        "to_binary" => Some(mk(1, TyType::Scalar(ScalarType::Str))),  // #335 alias of to_bin
        "to_oct"   => Some(mk(1, TyType::Scalar(ScalarType::Str))),
        "ord"      => Some(mk(1, TyType::Scalar(ScalarType::I64))),
        // Numeric utilities
        "trunc" => Some(mk(1, TyType::Scalar(ScalarType::F64))),
        "sign"  => Some(mk(1, TyType::Unknown)),
        "clamp" => Some(mk(3, TyType::Unknown)),
        // String utilities
        "str_repeat" => Some(mk(2, TyType::Scalar(ScalarType::Str))),
        // List utilities (fixed arity)
        "list_head"      => Some(mk(1, TyType::Unknown)),
        "list_last"      => Some(mk(1, TyType::Unknown)),
        "list_take"      => Some(mk(2, TyType::Unknown)),
        "list_drop"      => Some(mk(2, TyType::Unknown)),
        "list_find"      => Some(mk(2, TyType::Scalar(ScalarType::I64))),
        "list_count"     => Some(mk(2, TyType::Scalar(ScalarType::I64))),
        "list_any"       => Some(mk(2, TyType::Scalar(ScalarType::Bool))),
        "list_all"       => Some(mk(2, TyType::Scalar(ScalarType::Bool))),
        "list_flat_map"  => Some(mk(2, TyType::Unknown)),
        "list_partition" => Some(mk(2, TyType::Unknown)),
        // Map utilities
        "map_merge" => Some(mk(2, TyType::Unknown)),
        // Trit (ternary-weight) builtins
        "trit_quantize"      => Some(mk(1, TyType::Unknown)),
        "trit_quantize_soft" => Some(mk(2, TyType::Unknown)),
        "trit_neg"           => Some(mk(1, TyType::Unknown)),
        "trit_sparsity"      => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::F64))),
        "trit_pack"          => Some(mk(1, TyType::Unknown)),
        "is_trit"            => Some(mk(1, TyType::Scalar(crate::ast::ScalarType::Bool))),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    #[allow(dead_code)]
    pub shape_params: Vec<String>,
    pub fields: HashMap<String, TyType>,
    pub methods: HashMap<String, FnSig>,
}

// ─── Environment ─────────────────────────────────────────────────────────────

/// Lexically-scoped symbol table.
#[derive(Debug, Clone, Default)]
pub struct Env {
    /// Each entry is one lexical scope (block, fn body).
    scopes: Vec<HashMap<String, TyType>>,
    /// Tracks which binding names were declared with `let !` at each scope level.
    /// Parallel to `scopes`; used to warn when `:=` shadows an outer mutable binding.
    mutable_idents: Vec<std::collections::HashSet<String>>,
    /// #248: static tensor shapes for bindings whose RHS is a literal arena
    /// constructor (`forge.zeros[f32, [..]]`). Parallel to `scopes`. Kept
    /// SEPARATE from the binding's `TyType` (which stays `Unknown`) so it is
    /// never consulted by `compatible_with` — it exists only to let the indexer
    /// flag static out-of-bounds through a binding, without the shape leaking
    /// into assignment / arg-passing compatibility (where a concrete shape would
    /// wrongly clash with symbolic model fields or `View` params).
    ctor_shapes: Vec<HashMap<String, Shape>>,
    /// #403 (MEMORY §2): bindings allocated by `forge.uninit`/`vault.uninit`
    /// that no write has landed on yet. Parallel to `scopes`, masked by
    /// shadowing bindings exactly like `ctor_shapes`.
    uninit_bindings: Vec<std::collections::HashSet<String>>,
    /// #442 (MEMORY §3.1): bindings whose value lives in the Vault — the RHS
    /// bottomed out in a `vault.*` constructor or a `vault { … }` block.
    /// Mutating one outside a `vault { … }` context is a cross-arena write.
    /// Same scoping discipline as `uninit_bindings`.
    vault_bindings: Vec<std::collections::HashSet<String>>,
    /// Symbolic shape parameters in scope, mapped to their declared bounds
    /// (None if unbounded, Some(c) if `= c` was provided).
    pub shape_params: Vec<HashMap<String, Option<SymDim>>>,
    /// Top-level model declarations.
    pub models: HashMap<String, ModelInfo>,
    /// Top-level function declarations.
    pub functions: HashMap<String, FnSig>,
    /// Top-level enum declarations (#336): name → ordered variant names. A
    /// variant's value is its index here (the i64 ordinal).
    pub enums: HashMap<String, Vec<String>>,
    /// Payload-carrying variants (#350 Part 2): enum name → variant name →
    /// ordered positional field types (raw AST, resolved at use sites once all
    /// types are registered). Absent / empty = a tag-only C-like variant.
    pub enum_payloads: HashMap<String, HashMap<String, Vec<crate::ast::Type>>>,
}

impl Env {
    pub fn new() -> Self {
        let mut env = Self::default();
        env.push_scope();
        env.push_shape_scope();
        env.install_builtins();
        env
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
        self.mutable_idents.push(std::collections::HashSet::new());
        self.ctor_shapes.push(HashMap::new());
        self.uninit_bindings.push(std::collections::HashSet::new());
        self.vault_bindings.push(std::collections::HashSet::new());
    }
    pub fn pop_scope(&mut self)  {
        self.scopes.pop();
        self.mutable_idents.pop();
        self.ctor_shapes.pop();
        self.uninit_bindings.pop();
        self.vault_bindings.pop();
    }

    /// #442 (MEMORY §3.1): record that a binding's value lives in the Vault.
    pub fn mark_vault_origin(&mut self, name: impl Into<String>) {
        if let Some(scope) = self.vault_bindings.last_mut() {
            scope.insert(name.into());
        }
    }

    /// #442: a same-scope rebind to a non-Vault value masks the tag here only.
    pub fn unmark_vault_origin_here(&mut self, name: &str) {
        if let Some(scope) = self.vault_bindings.last_mut() {
            scope.remove(name);
        }
    }

    /// #442: the binding was rebound to a non-Vault value — drop the tag
    /// wherever it lives (same shadow-masked walk as `clear_uninit`).
    pub fn clear_vault_origin(&mut self, name: &str) {
        for (vals, marks) in self.scopes.iter().rev().zip(self.vault_bindings.iter_mut().rev()) {
            if marks.remove(name) { return; }
            if vals.contains_key(name) { return; }
        }
    }

    /// #442: does the binding's value live in the Vault?
    pub fn is_vault_origin(&self, name: &str) -> bool {
        for (vals, marks) in self.scopes.iter().rev().zip(self.vault_bindings.iter().rev()) {
            if marks.contains(name) { return true; }
            if vals.contains_key(name) { return false; }
        }
        false
    }

    /// #403 (MEMORY §2): record a fresh uninit allocation for a binding in the
    /// current scope (where the `let` also binds the value).
    pub fn mark_uninit(&mut self, name: impl Into<String>) {
        if let Some(scope) = self.uninit_bindings.last_mut() {
            scope.insert(name.into());
        }
    }

    /// #403: a same-scope rebind of the name to a non-uninit value masks the
    /// flag in this scope only — outer-scope flags survive the shadow.
    pub fn unmark_uninit_here(&mut self, name: &str) {
        if let Some(scope) = self.uninit_bindings.last_mut() {
            scope.remove(name);
        }
    }

    /// #403: the first write landed (or the read was reported — one bug, one
    /// report). Walks inside-out with the same shadow mask as
    /// `lookup_ctor_shape`: stop at the first scope binding the name at all.
    pub fn clear_uninit(&mut self, name: &str) {
        for (vals, marks) in self.scopes.iter().rev().zip(self.uninit_bindings.iter_mut().rev()) {
            if marks.remove(name) { return; }
            if vals.contains_key(name) { return; }
        }
    }

    /// #403: is the binding still an unwritten uninit allocation?
    pub fn is_uninit(&self, name: &str) -> bool {
        for (vals, marks) in self.scopes.iter().rev().zip(self.uninit_bindings.iter().rev()) {
            if marks.contains(name) { return true; }
            if vals.contains_key(name) { return false; }
        }
        false
    }
    pub fn push_shape_scope(&mut self) { self.shape_params.push(HashMap::new()); }
    pub fn pop_shape_scope(&mut self)  { self.shape_params.pop(); }

    pub fn bind(&mut self, name: impl Into<String>, ty: TyType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), ty);
        }
    }

    /// Bind a `let !name` — same as `bind` but also records the name as mutable.
    pub fn bind_mutable(&mut self, name: impl Into<String>, ty: TyType) {
        let n: String = name.into();
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(n.clone(), ty);
        }
        if let Some(mut_scope) = self.mutable_idents.last_mut() {
            mut_scope.insert(n);
        }
    }

    /// Returns true if `name` is bound as a `!` (mutable) binding in any outer scope
    /// (i.e., any scope level except the current innermost one).
    pub fn outer_scope_has_mutable(&self, name: &str) -> bool {
        let n = self.scopes.len();
        if n < 2 { return false; }
        self.mutable_idents[..n - 1].iter().any(|s| s.contains(name))
    }

    /// Returns true if `name` is a `let !` or `let mut` binding in any scope.
    pub fn is_mutable_binding(&self, name: &str) -> bool {
        self.mutable_idents.iter().any(|s| s.contains(name))
    }

    pub fn bind_shape_param(&mut self, name: impl Into<String>, default: Option<SymDim>) {
        if let Some(scope) = self.shape_params.last_mut() {
            scope.insert(name.into(), default);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&TyType> {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(name) { return Some(t); }
        }
        None
    }

    /// #248: record (or clear) the static constructor shape for `name` in the
    /// current scope. Call on every `let` so a rebind to a non-constructor
    /// value (`Some` → `None`) drops the stale shape in this scope. Insert into
    /// the innermost scope — exactly where the matching `TyType` binding lands —
    /// so shadowing and block scoping stay consistent with `lookup`.
    pub fn set_ctor_shape(&mut self, name: impl Into<String>, shape: Option<Shape>) {
        if let Some(scope) = self.ctor_shapes.last_mut() {
            let n: String = name.into();
            match shape {
                Some(s) => { scope.insert(n, s); }
                None => { scope.remove(&n); }
            }
        }
    }

    /// #248: look up a binding's static constructor shape, inside-out, but stop
    /// at the first scope that *binds the name at all* — a shadowing binding
    /// without a recorded shape must mask an outer one, never see through it.
    pub fn lookup_ctor_shape(&self, name: &str) -> Option<&Shape> {
        for (vals, shapes) in self.scopes.iter().rev().zip(self.ctor_shapes.iter().rev()) {
            if let Some(s) = shapes.get(name) { return Some(s); }
            if vals.contains_key(name) { return None; }
        }
        None
    }

    pub fn shape_param_in_scope(&self, name: &str) -> bool {
        self.shape_params.iter().rev().any(|s| s.contains_key(name))
    }

    /// Seed the env with built-in stdlib functions.
    ///
    /// Fixed-arity builtins get their signatures from `builtin_sig()`.
    /// Truly variadic builtins (print, panic, axis-polymorphic reductions,
    /// namespace-like collectives) remain permissive stubs until the type
    /// system gains generic variables (#175).
    fn install_builtins(&mut self) {
        let variadic = |ret: TyType| FnSig {
            shape_params: Vec::new(),
            params: Vec::new(),  // empty = variadic; arity bypass in check.rs
            ret,
        };
        for (name, ret) in [
            // Truly variadic (format string or unbounded args)
            ("print",     TyType::Unit),
            ("print_err", TyType::Unit),
            ("panic",     TyType::Unit),
            ("format",    TyType::Scalar(ScalarType::Str)),
            ("join",      TyType::Scalar(ScalarType::Str)),
            // Optional-arg ML primitives
            ("softmax",  TyType::Unknown), // (x, axis=-1)
            ("attn",     TyType::Unknown), // (q, k, v, mask?)
            ("attn_gqa", TyType::Unknown), // (q, k, v, mask?)
            // Optional 2nd arg
            ("assert",    TyType::Unit),    // (bool, str?)
            ("assert_eq", TyType::Unit),    // (any, any, str?)
            ("assert_ne", TyType::Unit),    // (any, any, str?)
            ("round",     TyType::Scalar(ScalarType::F64)), // (f64, i64?)
            ("str_pad_left",  TyType::Scalar(ScalarType::Str)), // (str, i64, str?)
            ("str_pad_right", TyType::Scalar(ScalarType::Str)), // (str, i64, str?)
            // Axis-polymorphic or arity-polymorphic
            ("argmax",    TyType::Unknown), // (Tensor, i64?)
            ("argmin",    TyType::Unknown), // (Tensor, i64?)
            ("max",       TyType::Unknown), // max(a,b) or max(tensor)
            ("min",       TyType::Unknown), // min(a,b) or min(tensor)
            // Namespace / variadic config
            ("allreduce",  TyType::Unknown),
            ("load_batch", TyType::Unknown),
            ("data_iter",  TyType::Unknown),
            // Variadic constructors
            ("list", TyType::Unknown),
            ("map",  TyType::Unknown),
        ] {
            self.functions.insert(name.to_string(), variadic(ret));
        }
        for name in [
            "sum", "mean", "trace", "diag",
            "f32_to_bits", "f32_from_bits",
            "sqrt", "exp", "log", "abs", "sin", "cos", "floor", "ceil",
            // Elementwise activations (scalar or same-shape tensor).
            "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "mish",
            "tan", "asin", "acos", "atan", "atan2", "hypot",
            "gcd", "sort", "median",   // #335
            "log2", "log10", "isclose",
            "chr", "len",
            "rms_norm", "layer_norm", "rope", "embed",
            "solve", "inv", "lstsq",
            "read_file", "read_bytes", "write_file", "append_file", "file_exists",
            // Dynamic collections
            "list", "list_push", "list_pop", "list_get", "list_set",
            "list_len", "list_concat", "list_slice", "list_contains", "list_rev",
            "map", "map_new", "map_set", "map_get", "map_has", "map_contains", "map_del",
            "map_keys", "map_vals", "map_len",
            // Process / environment
            "env_var", "argv", "exit",
            // CLI argument parsing
            "cli_arg", "cli_flag", "cli_positional", "cli_positional_count",
            // Time
            "time_ms", "sleep_ms",
            // Terminal I/O
            "flush", "set_raw_mode", "read_char_nb", "read_char",
            // Extended RNG
            "rand_seed", "rand_float", "rand_int", "rand_normal", "rand_choice",
            // JSON
            "json_encode", "json_decode",
            // List functional combinators
            "list_map", "list_filter", "list_reduce", "list_sort", "list_sort_by",
            "list_zip", "list_enumerate", "list_flatten", "list_uniq",
            "list_sum", "list_min", "list_max",
            // Hashing
            "hash_fnv", "hash_crc32",
            // Filesystem operations
            "get_cwd", "list_dir", "make_dir", "delete_file", "delete_dir",
            "rename_file", "file_size", "path_join", "path_dirname", "path_basename",
            "path_exists", "path_is_dir", "path_is_file",
            // Process execution
            "exec_cmd",
            // Ports (#402, PORTS.md §2)
            "port_open", "port_call", "port_close",
            // Regex
            "regex_match", "regex_find", "regex_find_all",
            "regex_replace", "regex_replace_all", "regex_split",
            // Compression
            "gzip_compress", "gzip_decompress",
            "zlib_compress", "zlib_decompress",
            // HTTP networking
            "http_get", "http_post", "http_post_json",
            // Date/time
            "date_now_ms", "date_now_s", "date_format", "date_parse",
            "date_add_ms", "date_diff_ms",
            // Typed print variants
            "print_i64", "print_f64", "print_bool", "print_nil", "print_tensor",
            // Variance-trait primitives
            "variance", "pull_to_mean",
            "sum_along", "mean_along", "max_along", "min_along", "variance_along", "pull_to_mean_along",
            // Type conversions
            "to_str", "to_string", "to_int", "to_float",
            "to_hex", "to_bin", "to_binary", "to_oct", "ord",
            // Numeric utilities
            "trunc", "sign", "clamp", "str_repeat",
            // List utilities
            "list_head", "list_last", "list_take", "list_drop",
            "list_find", "list_count", "list_any", "list_all",
            "list_flat_map", "list_partition",
            // Map utilities
            "map_merge",
            // Trit (ternary-weight) builtins
            "trit_quantize", "trit_quantize_soft", "trit_neg",
            "trit_sparsity", "trit_pack", "is_trit",
            // Runtime type introspection (#184) + safe numeric parse (#185)
            "typeof", "is_int", "is_float", "is_str", "is_bool",
            "is_list", "is_map", "is_nil", "is_fn", "is_tensor",
            "is_numeric", "try_to_int", "try_to_float",
        ] {
            if let Some(sig) = builtin_sig(name) {
                self.functions.insert(name.to_string(), sig);
            }
        }
    }

    /// Is `name` a truly variadic builtin?  Used by check.rs to bypass arity
    /// validation for builtins that accept an optional axis argument or are
    /// called as namespaces (e.g. `allreduce.sum(...)`).
    pub fn is_builtin(&self, name: &str) -> bool {
        matches!(name,
            // Truly variadic (unbounded args)
            "print" | "print_err" | "panic" | "format" | "join" |
            // Optional-arg ML primitives
            "softmax" | "attn" | "attn_gqa" |
            // Optional 2nd/3rd arg
            "assert" | "assert_eq" | "assert_ne" |
            "round" | "str_pad_left" | "str_pad_right" |
            // Axis-polymorphic or arity-polymorphic
            "argmax" | "argmin" | "max" | "min" |
            // Namespace / variadic constructors
            "allreduce" | "load_batch" | "data_iter" | "list" | "map"
        )
    }
}
