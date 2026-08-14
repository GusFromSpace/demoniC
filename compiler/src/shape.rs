/// demoniC symbolic shape arithmetic — Phase 3 typechecker foundation.
///
/// SymDim represents a symbolic dimension that may be a constant (`32`), a
/// shape parameter (`B`, `D`, `H`), or an arithmetic combination thereof
/// (`B/dp`, `H*D`, `4*D/tp`). The typechecker reasons over SymDim instead
/// of concrete integers so it can prove shape facts about programs that
/// haven't been instantiated to concrete dimensions yet.
///
/// Equivalence checking is structural after normalization. Divisibility
/// uses constant-folding + simple algebraic identities — enough to handle
/// the cases that appear in the example corpus. Future work: integrate a
/// real Presburger / SMT solver for the cases this can't decide.

use std::fmt;

use crate::ast::{Expr, Literal, BinOp, UnOp};

// ─── SymDim ──────────────────────────────────────────────────────────────────

/// Symbolic dimension. Supports a small algebra: constants, variables,
/// add/sub/mul/div/mod. `Streaming` (`~`) and `Wildcard` (`_`) are
/// special markers; `Unknown` is the bottom for cases we can't analyze.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymDim {
    Const(i64),
    Var(String),
    Add(Box<SymDim>, Box<SymDim>),
    Sub(Box<SymDim>, Box<SymDim>),
    Mul(Box<SymDim>, Box<SymDim>),
    Div(Box<SymDim>, Box<SymDim>),
    Mod(Box<SymDim>, Box<SymDim>),
    Neg(Box<SymDim>),
    Streaming,   // `~` — growing axis (per-step extent)
    Wildcard,    // `_` — matches anything in shape patterns
    Unknown,     // analysis failure; conservative ⊥
}

impl SymDim {
    /// Try to convert a parsed `Expr` into a SymDim. Returns Unknown for
    /// any expression we can't analyze (function calls, indexing, etc.).
    pub fn from_expr(expr: &Expr) -> SymDim {
        match expr {
            Expr::Literal(Literal::Int(n, _), _) => SymDim::Const(*n),
            Expr::Ident(name, _) => {
                if name == "_" { SymDim::Wildcard }
                else if name == "~" { SymDim::Streaming }
                else { SymDim::Var(name.clone()) }
            }
            Expr::UnOp { op: UnOp::Neg, operand, .. } => {
                SymDim::Neg(Box::new(SymDim::from_expr(operand)))
            }
            Expr::BinOp { op, lhs, rhs, .. } => {
                let l = SymDim::from_expr(lhs);
                let r = SymDim::from_expr(rhs);
                match op {
                    BinOp::Add => SymDim::Add(Box::new(l), Box::new(r)),
                    BinOp::Sub => SymDim::Sub(Box::new(l), Box::new(r)),
                    BinOp::Mul => SymDim::Mul(Box::new(l), Box::new(r)),
                    BinOp::Div => SymDim::Div(Box::new(l), Box::new(r)),
                    BinOp::Mod => SymDim::Mod(Box::new(l), Box::new(r)),
                    _ => SymDim::Unknown,
                }
            }
            Expr::Tuple(elems, _) if elems.len() == 1 => SymDim::from_expr(&elems[0]),
            _ => SymDim::Unknown,
        }
    }

    /// Constant-fold + apply simple algebraic identities. Idempotent.
    /// Identities applied:
    ///   x + 0 = x,  0 + x = x
    ///   x - 0 = x,  x - x = 0
    ///   x * 1 = x,  1 * x = x,  x * 0 = 0,  0 * x = 0
    ///   x / 1 = x,  x / x = 1 (when x is not zero/unknown)
    ///   x % 1 = 0,  x % x = 0
    pub fn simplify(&self) -> SymDim {
        use SymDim::*;
        match self {
            Const(_) | Var(_) | Streaming | Wildcard | Unknown => self.clone(),
            Neg(x) => match x.simplify() {
                Const(n) => Const(-n),
                Neg(y) => *y,
                s => Neg(Box::new(s)),
            },
            Add(l, r) => match (l.simplify(), r.simplify()) {
                (Const(a), Const(b)) => Const(a + b),
                (Const(0), x) | (x, Const(0)) => x,
                (l, r) => Add(Box::new(l), Box::new(r)),
            },
            Sub(l, r) => match (l.simplify(), r.simplify()) {
                (Const(a), Const(b)) => Const(a - b),
                (x, Const(0)) => x,
                (l, r) if l == r => Const(0),
                (l, r) => Sub(Box::new(l), Box::new(r)),
            },
            Mul(l, r) => match (l.simplify(), r.simplify()) {
                (Const(a), Const(b)) => Const(a * b),
                (Const(0), _) | (_, Const(0)) => Const(0),
                (Const(1), x) | (x, Const(1)) => x,
                (l, r) => Mul(Box::new(l), Box::new(r)),
            },
            Div(l, r) => match (l.simplify(), r.simplify()) {
                (Const(a), Const(b)) if b != 0 && a % b == 0 => Const(a / b),
                (x, Const(1)) => x,
                (l, r) if l == r && !matches!(l, Const(0) | Unknown) => Const(1),
                (l, r) => Div(Box::new(l), Box::new(r)),
            },
            Mod(l, r) => match (l.simplify(), r.simplify()) {
                (Const(a), Const(b)) if b != 0 => Const(a % b),
                (_, Const(1)) => Const(0),
                (l, r) if l == r => Const(0),
                (l, r) => Mod(Box::new(l), Box::new(r)),
            },
        }
    }

    /// Check if two SymDims are provably equal. Uses structural equality
    /// after simplification. For more sophisticated cases (commutativity
    /// across non-equal sub-terms, distributivity), returns Unknown.
    pub fn equivalent(&self, other: &SymDim) -> Equiv {
        use SymDim::*;
        let a = self.simplify();
        let b = other.simplify();
        if a == b { return Equiv::Equal; }
        // Constants that differ — definitely not equal.
        if let (Const(_), Const(_)) = (&a, &b) { return Equiv::NotEqual; }
        // Wildcard matches anything (used in shape patterns).
        if matches!(a, Wildcard) || matches!(b, Wildcard) { return Equiv::Equal; }
        // Streaming is opaque — only equal to itself.
        if matches!(a, Streaming) || matches!(b, Streaming) {
            return if a == b { Equiv::Equal } else { Equiv::Unknown };
        }
        // Otherwise: don't know.
        Equiv::Unknown
    }

    /// Does `divisor` divide `self`? Used to check shard feasibility
    /// (e.g., is `B` divisible by `dp` for `@shard(axis=0, mesh=mesh.dp)`?).
    /// Returns Yes/No/Unknown.
    #[allow(dead_code)]
    pub fn divisible_by(&self, divisor: &SymDim) -> Tristate {
        use SymDim::*;
        let n = self.simplify();
        let d = divisor.simplify();
        match (&n, &d) {
            (_, Const(0)) => Tristate::No,  // div by zero is never well-defined
            (_, Const(1)) => Tristate::Yes,
            (Const(a), Const(b)) => if a % b == 0 { Tristate::Yes } else { Tristate::No },
            // `x` divides itself
            _ if n == d => Tristate::Yes,
            // `(x * d) / d` is exact
            (Mul(l, r), _) if **l == d || **r == d => Tristate::Yes,
            _ => Tristate::Unknown,
        }
    }

    /// True iff this is a concrete (variable-free) integer.
    #[allow(dead_code)]
    pub fn is_const(&self) -> bool { matches!(self.simplify(), SymDim::Const(_)) }
}

impl fmt::Display for SymDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SymDim::*;
        match self {
            Const(n) => write!(f, "{}", n),
            Var(s) => write!(f, "{}", s),
            Add(l, r) => write!(f, "({}+{})", l, r),
            Sub(l, r) => write!(f, "({}-{})", l, r),
            Mul(l, r) => write!(f, "({}*{})", l, r),
            Div(l, r) => write!(f, "({}/{})", l, r),
            Mod(l, r) => write!(f, "({}%{})", l, r),
            Neg(x) => write!(f, "-{}", x),
            Streaming => write!(f, "~"),
            Wildcard => write!(f, "_"),
            Unknown => write!(f, "?"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Equiv { Equal, NotEqual, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum Tristate { Yes, No, Unknown }

// ─── Shape ───────────────────────────────────────────────────────────────────

/// An ordered sequence of SymDims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<SymDim>,
}

impl Shape {
    pub fn new(dims: Vec<SymDim>) -> Self { Self { dims } }
    pub fn rank(&self) -> usize { self.dims.len() }

    /// Matmul shape check. Returns the resulting shape if compatible.
    /// Rules (broadcasting outer dims, contracting last/second-to-last):
    ///   [..., M, K] @ [..., K, N] → [..., M, N]
    /// Outer dims are broadcast-merged (one must be 1 or both must match).
    pub fn matmul(&self, other: &Shape) -> Result<Shape, ShapeError> {
        if self.rank() < 2 || other.rank() < 2 {
            return Err(ShapeError::msg(format!(
                "matmul requires rank>=2 on both sides; got {} and {}",
                self.rank(), other.rank()
            )));
        }
        let m = &self.dims[self.rank() - 2];
        let k1 = &self.dims[self.rank() - 1];
        let k2 = &other.dims[other.rank() - 2];
        let n = &other.dims[other.rank() - 1];
        match k1.equivalent(k2) {
            Equiv::Equal | Equiv::Unknown => {}  // accept Unknown conservatively
            Equiv::NotEqual => return Err(ShapeError::msg(format!(
                "matmul inner dims don't match: {} vs {}", k1, k2
            ))),
        }
        // Build outer shape via broadcasting
        let lhs_outer = &self.dims[..self.rank() - 2];
        let rhs_outer = &other.dims[..other.rank() - 2];
        let outer = broadcast(lhs_outer, rhs_outer)?;
        let mut result = outer;
        result.push(m.clone());
        result.push(n.clone());
        Ok(Shape::new(result))
    }

    /// Elementwise broadcast: shape of `self .op other`.
    pub fn broadcast(&self, other: &Shape) -> Result<Shape, ShapeError> {
        let dims = broadcast(&self.dims, &other.dims)?;
        Ok(Shape::new(dims))
    }

    /// True iff both shapes are provably the same length and dim-equal.
    pub fn same(&self, other: &Shape) -> bool {
        if self.rank() != other.rank() { return false; }
        self.dims.iter().zip(other.dims.iter())
            .all(|(a, b)| matches!(a.equivalent(b), Equiv::Equal))
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

/// NumPy/PyTorch-style broadcasting. Right-aligned. A dim of 1 broadcasts
/// to anything; otherwise dims must be provably equal.
fn broadcast(a: &[SymDim], b: &[SymDim]) -> Result<Vec<SymDim>, ShapeError> {
    let max_rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(max_rank);
    for i in 0..max_rank {
        let ai = a.len().checked_sub(max_rank - i).and_then(|j| a.get(j));
        let bi = b.len().checked_sub(max_rank - i).and_then(|j| b.get(j));
        let dim = match (ai, bi) {
            (Some(x), Some(y)) => {
                let xs = x.simplify();
                let ys = y.simplify();
                if matches!(xs, SymDim::Const(1)) { ys }
                else if matches!(ys, SymDim::Const(1)) { xs }
                else {
                    match xs.equivalent(&ys) {
                        Equiv::Equal | Equiv::Unknown => xs,
                        Equiv::NotEqual => return Err(ShapeError::msg(format!(
                            "incompatible broadcast dims at axis {}: {} vs {}", i, x, y
                        ))),
                    }
                }
            }
            (Some(x), None) | (None, Some(x)) => x.clone(),
            (None, None) => unreachable!(),
        };
        out.push(dim);
    }
    Ok(out)
}

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ShapeError { pub msg: String }
impl ShapeError {
    pub fn msg(s: impl Into<String>) -> Self { Self { msg: s.into() } }
}
impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.msg) }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn c(n: i64) -> SymDim { SymDim::Const(n) }
    fn v(s: &str) -> SymDim { SymDim::Var(s.to_string()) }
    fn mul(a: SymDim, b: SymDim) -> SymDim { SymDim::Mul(Box::new(a), Box::new(b)) }
    fn div(a: SymDim, b: SymDim) -> SymDim { SymDim::Div(Box::new(a), Box::new(b)) }
    fn add(a: SymDim, b: SymDim) -> SymDim { SymDim::Add(Box::new(a), Box::new(b)) }

    #[test]
    fn simplify_constants() {
        assert_eq!(add(c(2), c(3)).simplify(), c(5));
        assert_eq!(mul(c(4), c(5)).simplify(), c(20));
        assert_eq!(div(c(20), c(4)).simplify(), c(5));
    }

    #[test]
    fn simplify_identities() {
        assert_eq!(add(v("B"), c(0)).simplify(), v("B"));
        assert_eq!(mul(v("D"), c(1)).simplify(), v("D"));
        assert_eq!(mul(v("D"), c(0)).simplify(), c(0));
        assert_eq!(div(v("D"), v("D")).simplify(), c(1));
    }

    #[test]
    fn equivalent_structural() {
        assert_eq!(v("B").equivalent(&v("B")), Equiv::Equal);
        assert_eq!(v("B").equivalent(&v("D")), Equiv::Unknown);
        assert_eq!(c(8).equivalent(&c(8)), Equiv::Equal);
        assert_eq!(c(8).equivalent(&c(16)), Equiv::NotEqual);
    }

    #[test]
    fn divisible_constants() {
        assert_eq!(c(32).divisible_by(&c(8)), Tristate::Yes);
        assert_eq!(c(33).divisible_by(&c(8)), Tristate::No);
        assert_eq!(c(8).divisible_by(&c(0)), Tristate::No);
    }

    #[test]
    fn divisible_self() {
        assert_eq!(v("B").divisible_by(&v("B")), Tristate::Yes);
        assert_eq!(mul(v("B"), v("dp")).divisible_by(&v("dp")), Tristate::Yes);
    }

    #[test]
    fn matmul_compatible() {
        let lhs = Shape::new(vec![v("B"), v("S"), v("D")]);
        let _rhs = Shape::new(vec![v("D"), v("H")]);
        // [B, S, D] @ [D, H] → broadcast: shape doesn't allow rank<2 outer broadcasting easily
        // here we test the simpler case [B, S, D] @ [B, D, H] → [B, S, H]
        let rhs2 = Shape::new(vec![v("B"), v("D"), v("H")]);
        let result = lhs.matmul(&rhs2).unwrap();
        assert_eq!(result.dims, vec![v("B"), v("S"), v("H")]);
    }

    #[test]
    fn matmul_inner_mismatch() {
        let lhs = Shape::new(vec![v("B"), v("M"), c(8)]);
        let rhs = Shape::new(vec![v("B"), c(16), v("N")]);
        assert!(lhs.matmul(&rhs).is_err());
    }

    #[test]
    fn broadcast_basic() {
        let a = Shape::new(vec![v("B"), v("S"), c(1)]);
        let b = Shape::new(vec![c(1), v("S"), v("D")]);
        let out = a.broadcast(&b).unwrap();
        // Both have same simplified shape after broadcast
        assert_eq!(out.dims.len(), 3);
    }
}
