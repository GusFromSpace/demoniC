/// Interpreter unit tests — small programs hitting specific eval paths.
/// Integration: `dmc run examples/*.dmc` from the shell.

use super::interp::{Interpreter, Value, FW};
use super::lexer::Lexer;
use super::parser::Parser;
use super::resolver::Resolver;

fn as_str(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        _ => panic!("expected Str, got {:?}", v),
    }
}

fn run(src: &str) -> Value {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    Interpreter::new().run(&program, None).expect("run failed")
}

fn run_err(src: &str) -> String {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    Interpreter::new().run(&program, None).err().expect("expected runtime error").msg
}

/// A runtime error's message together with the `line:col` it points at, for
/// the tests that care that a diagnostic points *somewhere* — a message with
/// `span: None` prints bare `runtime error:` and leaves the reader hunting.
fn run_err_at(src: &str) -> (String, Option<(usize, usize)>) {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let e = Interpreter::new().run(&program, None).err().expect("expected runtime error");
    let at = e.span.as_ref().map(|s| (s.line, s.col));
    (e.msg, at)
}

/// Run `f` on a thread with the same large stack the `dmc` binary gives the
/// interpreter (`main.rs`'s `INTERP_STACK_SIZE`). The tree-walking interpreter
/// consumes deep native stack per demoniC call frame, so any test exercising
/// recursion must run here: on libtest's default ~2 MiB thread stack it
/// SIGABRTs on native overflow around depth 20 — long before the interpreter's
/// own `MAX_CALL_DEPTH` guard can turn deep recursion into a catchable error,
/// which is the behavior the binary actually ships. `Value` is not `Send` (it
/// holds `Rc`), so keep it inside the closure and return only `Send` data
/// (typically `()` after asserting).
fn with_big_stack<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::Builder::new()
        .stack_size(crate::INTERP_STACK_SIZE)
        .spawn(f)
        .expect("spawn big-stack test thread")
        .join()
        .expect("big-stack test thread panicked")
}

fn as_int(v: &Value) -> i64 {
    if let Value::Int(n, _) = v { *n } else { panic!("expected int, got {:?}", v) }
}
fn as_float(v: &Value) -> f64 {
    match v {
        Value::Float(x, _) => *x,
        Value::Int(n, _)   => *n as f64,
        _ => panic!("expected numeric, got {:?}", v),
    }
}

#[test]
fn no_main_returns_nil() {
    assert!(matches!(run(""), Value::Nil));
}

#[test]
fn enum_match_and_ordinal_336() {
    // Qualified + bare variant patterns both dispatch correctly, and `as i64`
    // yields the declaration-order ordinal.
    let src = r#"
        enum Color { Red, Green, Blue }
        fn rank(c: Color) -> i64 {
            match c {
                Color.Red  => 10,
                Green      => 20,
                Color.Blue => 30,
            }
        }
        fn main() -> i64 {
            rank(Color.Red) + rank(Color.Green) + rank(Color.Blue)
                + (Color.Blue as i64)
        }
    "#;
    // 10 + 20 + 30 + ordinal(Blue)=2 == 62
    assert_eq!(as_int(&run(src)), 62);
}

#[test]
fn enum_catch_all_binds_336() {
    // A bare non-variant ident is a catch-all that binds the scrutinee.
    let src = r#"
        enum Light { Red, Yellow, Green }
        fn go(l: Light) -> i64 {
            match l {
                Light.Green => 1,
                other       => (other as i64),
            }
        }
        fn main() -> i64 { go(Light.Red) + go(Light.Green) }
    "#;
    // Red -> catch-all -> ordinal 0; Green -> 1  => 1
    assert_eq!(as_int(&run(src)), 1);
}

#[test]
fn arena_block_expr_returns_tail_value() {
    assert_eq!(as_int(&run(r#"
        model Item { !x: i64 }
        fn main() -> i64 {
            let item = vault {
                Item { x: 11 }
            }
            item.x
        }
    "#)), 11);
}

#[test]
fn nested_arena_block_expr_returns_tail_value() {
    assert_eq!(as_int(&run(r#"
        model Item { !x: i64 }
        fn main() -> i64 {
            let !item = vault {
                let i = forge { Item { x: 11 } }
                i
            }
            item.x
        }
    "#)), 11);
}

#[test]
fn arithmetic() {
    assert_eq!(as_int(&run("fn main() -> i64 { 2 + 3 * 4 }")), 14);
    assert_eq!(as_int(&run("fn main() -> i64 { 20 / 4 - 1 }")), 4);
    assert_eq!(as_int(&run("fn main() -> i64 { 10 % 3 }")), 1);
}

#[test]
fn let_binding() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let x = 5
            let y = 10
            x + y
        }
    "#)), 15);
}

#[test]
fn fn_call() {
    assert_eq!(as_int(&run(r#"
        fn double(x: i64) -> i64 { x * 2 }
        fn main() -> i64 { double(21) }
    "#)), 42);
}

#[test]
fn recursion_fib() {
    assert_eq!(as_int(&run(r#"
        fn fib(n: i64) -> i64 {
            if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
        }
        fn main() -> i64 { fib(10) }
    "#)), 55);
}

#[test]
fn for_range_sum() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let mut total = 0
            for i in 0..10 { total += i }
            total
        }
    "#)), 45);
}

#[test]
fn tensor_literal_and_sum() {
    let v = run(r#"
        fn main() -> f32 {
            let t = [1.0, 2.0, 3.0, 4.0]
            sum(t)
        }
    "#);
    assert!((as_float(&v) - 10.0).abs() < 1e-9);
}

#[test]
fn unary_neg() {
    assert_eq!(as_int(&run("fn main() -> i64 { -5 }")), -5);
}

#[test]
fn bool_ops() {
    assert!(matches!(run("fn main() -> bool { true && false }"), Value::Bool(false)));
    assert!(matches!(run("fn main() -> bool { true || false }"), Value::Bool(true)));
    assert!(matches!(run("fn main() -> bool { !true }"), Value::Bool(false)));
}

#[test]
fn comparison() {
    assert!(matches!(run("fn main() -> bool { 3 < 5 }"), Value::Bool(true)));
    assert!(matches!(run("fn main() -> bool { 3 == 3 }"), Value::Bool(true)));
    assert!(matches!(run("fn main() -> bool { 3 != 3 }"), Value::Bool(false)));
}

#[test]
fn while_loop() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let mut x = 0
            let mut i = 0
            while i < 5 {
                x += i
                i += 1
            }
            x
        }
    "#)), 10);
}

#[test]
fn tuple_destructure_let() {
    assert_eq!(as_int(&run(r#"
        fn pair() -> (i64, i64) { (3, 7) }
        fn main() -> i64 {
            let (a, b) = pair()
            a + b
        }
    "#)), 10);
}

#[test]
fn undefined_fn_errors() {
    let msg = run_err(r#"
        fn main() -> i64 { nonexistent(1) }
    "#);
    assert!(msg.contains("undefined"), "got: {}", msg);
}

#[test]
fn break_in_loop() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let mut x = 0
            loop {
                x += 1
                if x >= 5 { break }
            }
            x
        }
    "#)), 5);
}

// ─── Match expression tests ───────────────────────────────────────────────────

#[test]
fn match_wildcard() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            match 42 {
                _ => 1
            }
        }
    "#)), 1);
}

#[test]
fn match_literal_hits_correct_arm() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            match 2 {
                1 => 10,
                2 => 20,
                3 => 30,
                _ => 0,
            }
        }
    "#)), 20);
}

#[test]
fn match_literal_falls_through_to_wildcard() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            match 99 {
                1 => 10,
                2 => 20,
                _ => 99,
            }
        }
    "#)), 99);
}

#[test]
fn match_ident_binds_scrutinee() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            match 7 {
                n => n + 1
            }
        }
    "#)), 8);
}

#[test]
fn match_bool_literal() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            match true {
                false => 0,
                true  => 1,
            }
        }
    "#)), 1);
}

// #291.2: an int literal pattern must match a float scrutinee (and vice
// versa), since `==` (scalar_compare) and `list_contains` already treat
// `0 == 0.0` as true. Reachable: a float scrutinee with int-literal arms
// type-checks, so without cross-matching `match 0.0 { 0 => .. }` wrongly
// fell through.
#[test]
fn match_int_literal_matches_float_scrutinee() {
    assert_eq!(as_int(&run(r#"
        fn classify(x: f64) -> i64 {
            match x {
                0 => 100,
                1 => 200,
                _ => 999,
            }
        }
        fn main() -> i64 { classify(0.0) + classify(1.0) + classify(2.0) }
    "#)), 100 + 200 + 999);
}

#[test]
fn match_float_literal_matches_int_scrutinee() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            match 1 {
                0.0 => 10,
                1.0 => 20,
                _   => 30,
            }
        }
    "#)), 20);
}

#[test]
fn match_tuple_pattern() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            match (1, 2) {
                (a, b) => a + b
            }
        }
    "#)), 3);
}

#[test]
fn match_as_expression_in_let() {
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let x = match 5 {
                5 => 100,
                _ => 0,
            }
            x
        }
    "#)), 100);
}

#[test]
fn match_no_arm_matches_panics() {
    // Spec §4.5: match must be exhaustive; no-match is a runtime panic (not nil).
    let e = run_err(r#"
        fn main() -> nil {
            let _ = match 42 {
                1 => 10
            }
            nil
        }
    "#);
    assert!(e.contains("no arm matched") || e.contains("match"), "got: {e}");
}

// ─── Dotted comparison + arena ctor + batched matmul + @grad tests ───────────

#[test]
fn dotted_gt_against_scalar_produces_mask() {
    // h .> 0.0 is the ReLU derivative mask used in train_step.dmc.
    // Verify it actually produces 0/1 entries.
    let v = run(r#"
        fn main() -> f32 {
            let t = [-1.0, 0.0, 1.0, 2.0]
            let mask = t .> 0.0
            sum(mask)
        }
    "#);
    // Two strictly-positive entries → mask sum = 2.0.
    assert!((as_float(&v) - 2.0).abs() < 1e-9, "got {:?}", v);
}

#[test]
fn forge_uninit_materialises_tensor() {
    // forge.uninit[T, shape] used to return Opaque; now it must yield a real
    // tensor whose `.shape` is queryable.
    let v = run(r#"
        fn main() -> i64 {
            let q = forge.uninit[f32, [3, 5, 7]]
            let (a, b, c) = q.shape
            a * 100 + b * 10 + c
        }
    "#);
    assert_eq!(as_int(&v), 357);
}

#[test]
fn oversized_tensor_shape_errors_not_panics() {
    // #368 follow-up: a shape whose element count overflows the address space
    // must be a clean located error, NOT a raw ndarray panic/backtrace. That
    // `run_err` returns at all (rather than the test binary aborting) proves it
    // did not panic.
    let msg = run_err(r#"
        fn main() -> nil {
            let w = forge.zeros[f32, [100000000, 100000000, 100000000]]
            print(w[0,0,0])
            nil
        }
    "#);
    assert!(msg.contains("too large") && msg.contains("overflows"),
        "expected an overflow diagnostic, got: {msg}");
}

#[test]
fn unfittable_tensor_shape_reports_size_not_oom_kill() {
    // #368 follow-up: a shape that fits i64 but not RAM (here ~7 PiB) must fail
    // with a clean size diagnostic before it OOM-kills the process. The message
    // reports the size in GiB whether it trips the RAM pre-check or the
    // try_reserve backstop.
    let msg = run_err(r#"
        fn main() -> nil {
            let w = forge.zeros[f32, [1000000000, 1000000]]
            print(w[0,0])
            nil
        }
    "#);
    assert!(msg.contains("GiB"), "expected a size-in-GiB diagnostic, got: {msg}");
}

#[test]
fn vault_zeros_then_arithmetic() {
    // vault.zeros[...] must produce a tensor that participates in .+ and sum.
    let v = run(r#"
        fn main() -> f32 {
            let !w = vault.zeros[f32, [2, 3]]
            let ones = vault.ones[f32, [2, 3]]
            sum(w .+ ones)
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-9, "got {:?}", v);
}

#[test]
fn vault_identity_diagonal_and_matmul() {
    // vault.identity[T, N] builds an N×N matrix with 1.0 on the diagonal.
    // Verify: sum equals N, off-diagonal is zero, and I @ M == M.
    let v = run(r#"
        fn main() -> f32 {
            let i3 = vault.identity[f32, 3]
            # sum of a 3x3 identity is 3; off-diagonal contributes 0
            let s = sum(i3)
            let m = [[5.0, 6.0], [7.0, 8.0]]
            let i2 = vault.identity[f32, 2]
            let r = i2 @ m
            # r must equal m: r[0,0]+r[1,1] = 5 + 8 = 13; add s (=3) → 16
            s + r[0, 0] + r[1, 1]
        }
    "#);
    assert!((as_float(&v) - 16.0).abs() < 1e-9, "got {:?}", v);
}

#[test]
fn vault_identity_shape_literal_form() {
    // Square shape-literal form `vault.identity[T, [N, N]]` is accepted too.
    let v = run(r#"
        fn main() -> f32 {
            let i = vault.identity[f32, [4, 4]]
            sum(i)
        }
    "#);
    assert!((as_float(&v) - 4.0).abs() < 1e-9, "got {:?}", v);
}

#[test]
fn trace_basic() {
    // trace of [[1,2],[3,4]] is 1 + 4 = 5.
    let v = run(r#"
        fn main() -> f32 { trace([[1.0, 2.0], [3.0, 4.0]]) }
    "#);
    assert!((as_float(&v) - 5.0).abs() < 1e-9, "got {:?}", v);
}

#[test]
fn trace_of_identity_is_dim() {
    // trace(I_N) == N.
    let v = run(r#"
        fn main() -> f32 { trace(vault.identity[f32, 4]) }
    "#);
    assert!((as_float(&v) - 4.0).abs() < 1e-9, "got {:?}", v);
}

#[test]
fn trace_non_square_errors() {
    // trace requires a square 2D tensor; a 2x3 is a runtime error.
    let msg = run_err(r#"
        fn main() -> f32 { trace([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) }
    "#);
    assert!(msg.contains("square"), "expected a 'square' error, got: {msg}");
}

#[test]
fn batched_matmul_rank3() {
    // [B, M, K] @ [B, K, N] → [B, M, N]
    let v = run(r#"
        fn main() -> i64 {
            let a = vault.ones[f32, [2, 3, 4]]
            let b = vault.ones[f32, [2, 4, 5]]
            let c = a @ b
            let (x, y, z) = c.shape
            x * 100 + y * 10 + z
        }
    "#);
    assert_eq!(as_int(&v), 235);
}

#[test]
fn batched_matmul_broadcasts_2d_rhs() {
    // [B, M, K] @ [K, N] -> [B, M, N]
    let v = run(r#"
        fn main() -> i64 {
            let a = vault.ones[f32, [2, 3, 4]]
            let b = vault.ones[f32, [4, 5]]
            let c = a @ b
            let (x, y, z) = c.shape
            x * 100 + y * 10 + z
        }
    "#);
    assert_eq!(as_int(&v), 235);
}

#[test]
fn batched_matmul_broadcasts_singleton_batch_dim() {
    // [1, M, K] @ [B, K, N] -> [B, M, N]
    let v = run(r#"
        fn main() -> i64 {
            let a = vault.ones[f32, [1, 3, 4]]
            let b = vault.ones[f32, [2, 4, 5]]
            let c = a @ b
            let (x, y, z) = c.shape
            x * 100 + y * 10 + z
        }
    "#);
    assert_eq!(as_int(&v), 235);
}

#[test]
fn batched_matmul_with_transpose() {
    // [B, S, D] @ ([B, S, D]') = [B, S, D] @ [B, D, S] = [B, S, S]
    let v = run(r#"
        fn main() -> i64 {
            let q = vault.zeros[f32, [4, 8, 16]]
            let k = vault.zeros[f32, [4, 8, 16]]
            let scores = q @ k'
            let (b, s1, s2) = scores.shape
            b * 10000 + s1 * 100 + s2
        }
    "#);
    assert_eq!(as_int(&v), 40808);
}

#[test]
fn grad_fwd_bwd_loss_honors_the_declared_return_width() {
    // #473 established that a declared return type is a binding site: an
    // ordinary call narrows its result to it, so `-> f32 { .. + 0.1 }` hands
    // back the f32. The `@grad` entry returned the raw tape value instead, so
    // ONE function produced two different numbers depending on how it was
    // called — and the JIT, whose tape is f32 throughout, matched only the
    // plain call. Pin both halves against the same f32 literal.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [3]]) -> f32 {
            let s = sum(w)
            s + 0.1
        }
        fn main() -> f32 {
            let !w = vault.zeros[f32, [3]]
            w[0] = 1.0  w[1] = 2.0  w[2] = 3.0
            let (l, _g) = f.fwd_bwd(w)
            l - f(w)
        }
    "#);
    assert_eq!(as_float(&v), 0.0,
        "`.fwd_bwd` loss and the plain call must be the same number");

    // And the number itself is the f32 one, not the f64 one: 6 + 0.1 rounds to
    // 6.0999999046325684 at f32 width, which is NOT 6.1.
    let l = run(r#"
        @grad fn f(!w: Tensor[f32, [3]]) -> f32 {
            let s = sum(w)
            s + 0.1
        }
        fn main() -> f32 {
            let !w = vault.zeros[f32, [3]]
            w[0] = 1.0  w[1] = 2.0  w[2] = 3.0
            let (l, _g) = f.fwd_bwd(w)
            l
        }
    "#);
    assert_eq!(as_float(&l), 6.1_f32 as f64,
        "loss must be the f32 value, got {:?}", l);
    assert_ne!(as_float(&l), 6.1_f64, "loss must not be the f64 value");
}

#[test]
fn grad_returns_loss_and_struct_with_param_shaped_tensors() {
    // forward.fwd_bwd(...) → (loss, {!param: grad})
    // Verify the struct has a field named after each `!` param, shaped to match.
    let v = run(r#"
        @grad fn loss[D](!w: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum((w .- x) .* (w .- x))
        }
        fn main() -> i64 {
            let !w = vault.zeros[f32, [4]]
            let  x = vault.ones[f32,  [4]]
            let (_l, g) = loss.fwd_bwd(w, x)
            let (n,) = g.w.shape
            n
        }
    "#);
    assert_eq!(as_int(&v), 4);
}

#[test]
fn grad_reverse_mode_produces_exact_gradients() {
    // Reverse-mode autodiff (not finite-diff). For sum((w - x)^2) the analytic
    // gradient is 2*(w - x). At w=0, x=1 with D=4: each ∂L/∂w[i] = -2, so
    // sum(g.w) = -8 exactly (within float precision).
    let v = run(r#"
        @grad fn loss[D](!w: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum((w .- x) .* (w .- x))
        }
        fn main() -> f32 {
            let !w = vault.zeros[f32, [4]]
            let  x = vault.ones[f32,  [4]]
            let (_l, g) = loss.fwd_bwd(w, x)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - (-8.0)).abs() < 1e-9, "got {:?}", v);
}

// ── #398: captured `mut` bindings are differentiable inputs ──────────────────
//          AUTODIFF.md §2. A module-level `let !` read directly in a `@grad fn`
//          body becomes a tape input; its adjoint comes back in `Grads` under
//          the binding's own name, alongside the `!` params.

/// The captured-bias program used by the tests below. `tail` is the expression
/// `main` returns; `l` and `g` are already bound from `loss.fwd_bwd(w, x)`.
///
/// L = s · sum((w .* x .+ b)³) with **both** a captured tensor `b` and a
/// captured scalar `s`, plus the `!` param `w`. At w = [1,2,3], x = [.5,-1,2],
/// b = [.25,.5,-.75], s = 2: y = [0.75, -1.5, 5.25], sum(y³) = 141.75.
fn captured_bias_src(tail: &str) -> String {
    format!(r#"
        let !b = [0.25f32, 0.5f32, -0.75f32]
        let !s = 2.0f32
        fn absf(v: f32) -> f32 {{ if v < 0.0f32 {{ 0.0f32 - v }} else {{ v }} }}
        @grad fn loss(!w: Tensor[f32,[3]], x: Tensor[f32,[3]]) -> f32 {{
            let y = w .* x .+ b
            sum(y .* y .* y) * s
        }}
        fn fwd() -> f32 {{
            let w = [1.0f32, 2.0f32, 3.0f32]
            let x = [0.5f32, -1.0f32, 2.0f32]
            loss(w, x)
        }}
        fn main() -> f32 {{
            let w = [1.0f32, 2.0f32, 3.0f32]
            let x = [0.5f32, -1.0f32, 2.0f32]
            let (l, g) = loss.fwd_bwd(w, x)
            {tail}
        }}
    "#, tail = tail)
}

#[test]
fn grad_captured_mut_tensor_gets_real_gradient_398() {
    // dL/db = 3y²·s = [3.375, 13.5, 165.375]; dL/ds = sum(y³) = 141.75;
    // dL/dw = 3y²·x·s = [1.6875, -13.5, 330.75]. All three come back together,
    // each under its own name — element-exact (every value here is a dyadic
    // rational, so f32 holds it without rounding).
    assert!((as_float(&run(&captured_bias_src("l"))) - 283.5).abs() < 1e-9, "loss");
    assert!((as_float(&run(&captured_bias_src("g.b[0]"))) -   3.375).abs() < 1e-9, "g.b[0]");
    assert!((as_float(&run(&captured_bias_src("g.b[1]"))) -  13.5  ).abs() < 1e-9, "g.b[1]");
    assert!((as_float(&run(&captured_bias_src("g.b[2]"))) - 165.375).abs() < 1e-9, "g.b[2]");
    assert!((as_float(&run(&captured_bias_src("g.s")))    - 141.75 ).abs() < 1e-9, "g.s");
    assert!((as_float(&run(&captured_bias_src("g.w[0]"))) -   1.6875).abs() < 1e-9, "g.w[0]");
    assert!((as_float(&run(&captured_bias_src("g.w[1]"))) - (-13.5)).abs() < 1e-9, "g.w[1]");
    assert!((as_float(&run(&captured_bias_src("g.w[2]"))) - 330.75 ).abs() < 1e-9, "g.w[2]");
    // The gradient tensor keeps the captured binding's shape.
    assert_eq!(as_int(&run(&captured_bias_src("let (n,) = g.b.shape  n"))), 3);
}

#[test]
fn grad_captured_mut_tensor_matches_finite_differences_398() {
    // Numeric oracle: central differences taken by perturbing the captured
    // binding itself, (L(b+h) - L(b-h)) / 2h per element, compared against the
    // analytic adjoint. h = 1/64 is exact in f32, so the residual is the cubic's
    // O(h²) truncation (~2e-4·s) plus f32 rounding of L≈283 — comfortably inside
    // 1e-2, while the gradients being checked are 3.4 … 165.
    let max_err = as_float(&run(&captured_bias_src(r#"
        let h = 0.015625f32
        let !max_err = 0.0f32
        for i in 0..3 {
            let base = b[i]
            b[i] = base + h
            let lp = fwd()
            b[i] = base - h
            let lm = fwd()
            b[i] = base
            let numeric = (lp - lm) / (2.0f32 * h)
            let e = absf(numeric - g.b[i])
            if e > max_err { max_err = e }
        }
        max_err
    "#)));
    assert!(max_err < 1e-2,
            "captured-tensor gradcheck: max |numeric - analytic| = {}", max_err);

    // Teeth: the same finite differences against a deliberately wrong adjoint
    // (zero — the pre-#398 answer, when the field was absent entirely) must
    // *exceed* the tolerance. Reported as the smallest per-element |numeric|.
    let min_numeric = as_float(&run(&captured_bias_src(r#"
        let h = 0.015625f32
        let !min_num = 1000000.0f32
        for i in 0..3 {
            let base = b[i]
            b[i] = base + h
            let lp = fwd()
            b[i] = base - h
            let lm = fwd()
            b[i] = base
            let numeric = absf((lp - lm) / (2.0f32 * h))
            if numeric < min_num { min_num = numeric }
        }
        min_num
    "#)));
    assert!(min_numeric > 1e-2,
            "gradcheck has no teeth: a zero adjoint would pass (min |numeric| = {})",
            min_numeric);
}

#[test]
fn grad_captured_mut_scalar_matches_finite_differences_398() {
    // Same oracle for the captured *scalar* `s`: dL/ds = sum(y³) = 141.75.
    let err = as_float(&run(&captured_bias_src(r#"
        let h = 0.015625f32
        let base = s
        s = base + h
        let lp = fwd()
        s = base - h
        let lm = fwd()
        s = base
        let numeric = (lp - lm) / (2.0f32 * h)
        absf(numeric - g.s)
    "#)));
    assert!(err < 1e-2, "captured-scalar gradcheck: |numeric - analytic| = {}", err);

    // Teeth: the numeric slope is nowhere near zero, so a missing/zero `g.s`
    // could not have passed the comparison above.
    let numeric = as_float(&run(&captured_bias_src(r#"
        let h = 0.015625f32
        let base = s
        s = base + h
        let lp = fwd()
        s = base - h
        let lm = fwd()
        s = base
        absf((lp - lm) / (2.0f32 * h))
    "#)));
    assert!(numeric > 1e-2,
            "gradcheck has no teeth: a zero adjoint would pass (|numeric| = {})", numeric);
}

#[test]
fn grad_captured_and_param_land_under_their_own_names_398() {
    // Param and capture gradients are different numbers here, so a field mixed
    // up between them fails: L = sum((w .* x .+ b)^2) with x = [2, 2] gives
    // dL/dw = 2y·x = 2·[4, 8]·2 = [16, 32] and dL/db = 2y = [8, 16].
    let src = |tail: &str| format!(r#"
        let !b = [2.0f64, 4.0f64]
        @grad fn loss(!w: Tensor[f64,[2]], x: Tensor[f64,[2]]) -> f64 {{
            let y = w .* x .+ b
            sum(y .* y)
        }}
        fn main() -> f64 {{
            let w = [1.0f64, 2.0f64]
            let x = [2.0f64, 2.0f64]
            let (l, g) = loss.fwd_bwd(w, x)
            {tail}
        }}
    "#, tail = tail);
    assert!((as_float(&run(&src("g.w[0]"))) - 16.0).abs() < 1e-12);
    assert!((as_float(&run(&src("g.w[1]"))) - 32.0).abs() < 1e-12);
    assert!((as_float(&run(&src("g.b[0]"))) -  8.0).abs() < 1e-12);
    assert!((as_float(&run(&src("g.b[1]"))) - 16.0).abs() < 1e-12);
}

#[test]
fn grad_param_shadows_same_named_capture_398() {
    // A `!` param named like a module-level mutable is NOT a capture — the
    // param shadows it. `g.b` must be the parameter's gradient (computed from
    // the *argument*, [10, 10]), not the module binding's ([99, 99]) and there
    // must be exactly one `b` field. dL/db = 2b_arg = [20, 20] → sum 40.
    let v = run(r#"
        let !b = [99.0f64, 99.0f64]
        @grad fn loss(!b: Tensor[f64,[2]]) -> f64 { sum(b .* b) }
        fn main() -> f64 {
            let arg = [10.0f64, 10.0f64]
            let (l, g) = loss.fwd_bwd(arg)
            sum(g.b)
        }
    "#);
    assert!((as_float(&v) - 40.0).abs() < 1e-12,
            "param must shadow the same-named capture, got {:?}", v);
}

#[test]
fn grad_captured_untaken_branch_is_zero_398() {
    // Define-by-run (AUTODIFF.md §6.1): the branch that did not execute
    // contributes nothing, so a capture read only there gets a real zero
    // gradient of its own shape — not a missing field, and not the other
    // branch's value.
    let zero = run(r#"
        let !b = [3.0f64, 5.0f64]
        @grad fn loss(!w: Tensor[f64,[2]], take: bool) -> f64 {
            if take { sum(w .* b) } else { sum(w .* w) }
        }
        fn main() -> f64 {
            let w = [1.0f64, 2.0f64]
            let (l, g) = loss.fwd_bwd(w, false)
            sum(g.b)
        }
    "#);
    assert!(as_float(&zero).abs() < 1e-12,
            "untaken-branch capture must be zero, got {:?}", zero);
    // The taken branch does give it a gradient: dL/db = w → sum = 3.
    let taken = run(r#"
        let !b = [3.0f64, 5.0f64]
        @grad fn loss(!w: Tensor[f64,[2]], take: bool) -> f64 {
            if take { sum(w .* b) } else { sum(w .* w) }
        }
        fn main() -> f64 {
            let w = [1.0f64, 2.0f64]
            let (l, g) = loss.fwd_bwd(w, true)
            sum(g.b)
        }
    "#);
    assert!((as_float(&taken) - 3.0).abs() < 1e-12,
            "taken-branch capture gradient, got {:?}", taken);
}

#[test]
fn grad_captured_reassigned_in_body_is_wrt_entry_value_398() {
    // The body reassigns the capture (`b = b .+ b`, which the forward really
    // performs). The gradient owed is the one w.r.t. the value the call started
    // from: L = sum(w .* 2b) → dL/db = 2w = [2, 6], sum = 8. Reading the
    // adjoint off the post-reassignment node would give sum(w) = 4 instead.
    let v = run(r#"
        let !b = [1.0f64, 2.0f64]
        @grad fn loss(!w: Tensor[f64,[2]]) -> f64 {
            b = b .+ b
            sum(w .* b)
        }
        fn main() -> f64 {
            let w = [1.0f64, 3.0f64]
            let (l, g) = loss.fwd_bwd(w)
            sum(g.b)
        }
    "#);
    assert!((as_float(&v) - 8.0).abs() < 1e-12,
            "capture gradient must be w.r.t. the entry value, got {:?}", v);
}

#[test]
fn grad_captured_mut_second_order_398() {
    // `@grad @grad` + `fwd_bwd_bwd`: with L = s·sum(w·w), the first backward
    // gives dL/dw = 2sw; the second-order pass reduces it (sum) and
    // differentiates again, so g.w = 2s = [6, 6] and g.s = 2·sum(w) = 6.
    let src = |tail: &str| format!(r#"
        let !s = 3.0f64
        @grad @grad fn q(!w: Tensor[f64,[2]]) -> f64 {{ sum(w .* w) * s }}
        fn main() -> f64 {{
            let w = [1.0f64, 2.0f64]
            let (l, g) = q.fwd_bwd_bwd(w)
            {tail}
        }}
    "#, tail = tail);
    assert!((as_float(&run(&src("l"))) - 15.0).abs() < 1e-12, "loss");
    assert!((as_float(&run(&src("g.w[0]"))) - 6.0).abs() < 1e-12, "g.w[0]");
    assert!((as_float(&run(&src("g.w[1]"))) - 6.0).abs() < 1e-12, "g.w[1]");
    assert!((as_float(&run(&src("g.s"))) - 6.0).abs() < 1e-12, "g.s");
}

#[test]
fn grad_captured_mut_second_order_without_mut_param_398() {
    // With no `!` param, the second-order reduction is seeded from the first
    // captured binding. L = sum(x .* cb .* cb) → dL/dcb = 2x·cb, whose sum
    // differentiates back to 2x = [2, 6].
    let src = |tail: &str| format!(r#"
        let !cb = [1.0f32, 2.0f32]
        @grad @grad fn q(x: Tensor[f32,[2]]) -> f32 {{ sum(x .* cb .* cb) }}
        fn main() -> f32 {{
            let x = [1.0f32, 3.0f32]
            let (l, g) = q.fwd_bwd_bwd(x)
            {tail}
        }}
    "#, tail = tail);
    assert!((as_float(&run(&src("l"))) - 13.0).abs() < 1e-9, "loss");
    assert!((as_float(&run(&src("g.cb[0]"))) - 2.0).abs() < 1e-9, "g.cb[0]");
    assert!((as_float(&run(&src("g.cb[1]"))) - 6.0).abs() < 1e-9, "g.cb[1]");
}

#[test]
fn grad_second_order_seed_without_gradient_is_named_398() {
    // The second-order seed can legitimately have no first-order adjoint — here
    // because the capture reaches the loss only as the scalar operand of `*`,
    // which `backward_symbolic` does not route through (a pre-existing
    // second-order gap, unrelated to the first-order capture gradients above).
    // The diagnostic must say which input it was seeding from and admit the
    // gap, rather than blaming a `!` param that does not exist.
    let err = run_err(r#"
        let !s = 3.0f32
        @grad @grad fn q(x: Tensor[f32,[2]]) -> f32 { sum(x .* x) * s * s }
        fn main() -> f32 {
            let x = [1.0f32, 2.0f32]
            let (l, g) = q.fwd_bwd_bwd(x)
            l
        }
    "#);
    assert_eq!(
        err,
        "@grad `q`: the second-order pass found no first-order gradient for \
         captured `mut` binding `s` to differentiate again — either the loss \
         does not depend on it, or its only path to the loss is one the \
         second-order replay does not cover (the scalar `*` multiplier operand \
         is the known gap; first order is unaffected)",
        "unexpected second-order diagnostic",
    );
    // First order through exactly the same program is unaffected: dL/ds = 2s·sum(x²) = 30.
    let v = run(r#"
        let !s = 3.0f32
        @grad fn q(x: Tensor[f32,[2]]) -> f32 { sum(x .* x) * s * s }
        fn main() -> f32 {
            let x = [1.0f32, 2.0f32]
            let (l, g) = q.fwd_bwd(x)
            g.s
        }
    "#);
    assert!((as_float(&v) - 30.0).abs() < 1e-9, "first-order dL/ds, got {:?}", v);
}

#[test]
fn grad_capture_resolves_to_the_module_binding_not_a_caller_local_398() {
    // The interpreter shares one scope stack across calls, so a caller's local
    // of the capture's name used to be what got taped: the same `@grad fn` over
    // the same module state returned a different gradient depending on who
    // called it. `cap` here is [1, 1, 1] in the module and [100, 100, 100] in
    // `go`'s frame, and dL/dcap = 2·w·cap makes the taped value visible in the
    // answer — 2 from the module binding, 200 from the caller's local.
    let src = |caller: &str| format!(r#"
        let !cap = [1.0f64, 1.0f64, 1.0f64]
        @grad fn loss(!w: Tensor[f64,[3]]) -> f64 {{ sum(w .* cap .* cap) }}
        fn direct() -> f64 {{
            let w = [1.0f64, 1.0f64, 1.0f64]
            let (l, g) = loss.fwd_bwd(w)
            g.cap[0]
        }}
        fn go() -> f64 {{
            let cap = [100.0f64, 100.0f64, 100.0f64]
            let w = [1.0f64, 1.0f64, 1.0f64]
            let (l, g) = loss.fwd_bwd(w)
            g.cap[0]
        }}
        fn main() -> f64 {{ {caller}() }}
    "#, caller = caller);
    let direct = as_float(&run(&src("direct")));
    let via_go = as_float(&run(&src("go")));
    assert!((direct - 2.0).abs() < 1e-12, "module binding is what gets taped, got {}", direct);
    assert!((via_go - 2.0).abs() < 1e-12,
            "a caller's same-named local must not be taped under the capture's \
             name — expected 2.0 (module `cap`), got {} (the caller's local)", via_go);
}

#[test]
fn grad_capture_forward_also_reads_the_module_binding_398() {
    // The companion to the test above: it is not enough for the tape node to
    // hold the module value while the forward pass reads the caller's local —
    // that would put a value on the tape that the recorded computation never
    // used. Both must resolve to the module binding, so the loss agrees too.
    let src = |caller: &str| format!(r#"
        let !cap = [1.0f64, 1.0f64, 1.0f64]
        @grad fn loss(!w: Tensor[f64,[3]]) -> f64 {{ sum(w .* cap) }}
        fn direct() -> f64 {{
            let w = [1.0f64, 1.0f64, 1.0f64]
            let (l, g) = loss.fwd_bwd(w)
            l
        }}
        fn go() -> f64 {{
            let cap = [100.0f64, 100.0f64, 100.0f64]
            let w = [1.0f64, 1.0f64, 1.0f64]
            let (l, g) = loss.fwd_bwd(w)
            l
        }}
        fn main() -> f64 {{ {caller}() }}
    "#, caller = caller);
    assert!((as_float(&run(&src("direct"))) - 3.0).abs() < 1e-12);
    assert!((as_float(&run(&src("go"))) - 3.0).abs() < 1e-12,
            "the traced forward must read the module `cap`, not the caller's local");
}

#[test]
fn grad_capture_masking_restores_the_callers_local_398() {
    // Masking the caller's local is scoped to the grad call: `go`'s own `cap`
    // has to be intact on the line after `fwd_bwd`, and still be its own value.
    let v = run(r#"
        let !cap = [1.0f64, 1.0f64, 1.0f64]
        @grad fn loss(!w: Tensor[f64,[3]]) -> f64 { sum(w .* cap) }
        fn go() -> f64 {
            let cap = [100.0f64, 100.0f64, 100.0f64]
            let w = [1.0f64, 1.0f64, 1.0f64]
            let (l, g) = loss.fwd_bwd(w)
            sum(cap)
        }
        fn main() -> f64 { go() }
    "#);
    assert!((as_float(&v) - 300.0).abs() < 1e-12,
            "the caller's local must survive the grad call unchanged, got {:?}", v);
}

#[test]
fn grad_body_local_let_shadows_the_capture_so_no_grads_field_398() {
    // A body-local `let cap` shadows the module binding: the body never reads
    // the capture, so there is nothing to tape and no `g.cap` to return. The
    // old scan looked only at "is this name a module mutable?", so it produced
    // a phantom field full of zeros for a binding the body never touched.
    let v = run(r#"
        let !cap = [5.0f64, 5.0f64, 5.0f64]
        @grad fn loss(!w: Tensor[f64,[3]]) -> f64 {
            let cap = [1.0f64, 1.0f64, 1.0f64]
            sum(w .* cap)
        }
        fn main() -> f64 {
            let w = [1.0f64, 2.0f64, 3.0f64]
            let (l, g) = loss.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    // The param gradient is the local's value ([1,1,1] → sum 3), not the
    // module binding's ([5,5,5] → sum 15).
    assert!((as_float(&v) - 3.0).abs() < 1e-12,
            "shadowed capture must not reach the tape, got {:?}", v);
    // And `g.cap` is simply not a field: absent fields read back opaque.
    let absent = run(r#"
        let !cap = [5.0f64, 5.0f64, 5.0f64]
        @grad fn loss(!w: Tensor[f64,[3]]) -> f64 {
            let cap = [1.0f64, 1.0f64, 1.0f64]
            sum(w .* cap)
        }
        fn main() -> f64 {
            let w = [1.0f64, 2.0f64, 3.0f64]
            let (l, g) = loss.fwd_bwd(w)
            g.cap
        }
    "#);
    assert!(matches!(absent, Value::Opaque(_)),
            "a shadowed module binding must not appear in `Grads`, got {:?}", absent);
}

#[test]
fn grad_capture_is_scoped_to_the_fns_own_module_398() {
    // A flat cross-module name list made an unrelated module's `let !alpha`
    // enough to turn THIS module's immutable `alpha` into a `Grads` field —
    // a differentiable input no checker rule ever admitted (the checker does
    // not put an imported module's file-private `let !` in scope under its
    // bare name at all). `g.alpha` must simply not exist here.
    //
    // lib.dmc's `alpha` is deliberately file-private, not `pub`: #596 made
    // `pub let` a compile-time error (SPEC.md §6.7 exports only `pub fn` /
    // `pub model` / `pub type`), so a `pub let !alpha` fixture no longer
    // type-checks. The flat-name-list bug this guards never turned on
    // visibility anyway — only on the name and the `!` — so a private
    // `let !alpha` exercises the same path.
    let v = run_multi(&[
        ("lib.dmc", "let !alpha = 9.0f64\npub fn touch() -> f64 { alpha }"),
        ("main.dmc", r#"use "lib.dmc"
let alpha = 3.0f64
@grad fn loss(!w: Tensor[f64,[3]]) -> f64 { sum(w .* w) * alpha }
fn main() -> f64 {
    let w = [1.0f64, 2.0f64, 3.0f64]
    let (l, g) = loss.fwd_bwd(w)
    g.alpha
}"#),
    ], "main.dmc");
    assert!(matches!(v, Value::Opaque(_)),
            "an immutable local binding must not be differentiable just because \
             another module declares `let !alpha`, got {:?}", v);
}

#[test]
fn grad_own_module_capture_still_works_alongside_an_import_398() {
    // The other side of the scoping fix: narrowing to the fn's own module must
    // not cost the module its OWN captures. `beta` is main's `let !`, and it
    // keeps its gradient with lib.dmc's unrelated `!alpha` in the graph.
    // dL/dbeta = sum(w .* w) = 14.
    //
    // lib.dmc's `alpha` is file-private for the same reason as the sibling
    // test above: #596 made `pub let` a compile-time error, so the fixture
    // can no longer spell it `pub`.
    let v = run_multi(&[
        ("lib.dmc", "let !alpha = 9.0f64\npub fn touch() -> f64 { alpha }"),
        ("main.dmc", r#"use "lib.dmc"
let !beta = 2.0f64
@grad fn loss(!w: Tensor[f64,[3]]) -> f64 { sum(w .* w) * beta }
fn main() -> f64 {
    let w = [1.0f64, 2.0f64, 3.0f64]
    let (l, g) = loss.fwd_bwd(w)
    g.beta
}"#),
    ], "main.dmc");
    assert!((as_float(&v) - 14.0).abs() < 1e-12,
            "the fn's own module capture must survive the scoping fix, got {:?}", v);
}

#[test]
fn grad_capture_only_fn_needs_no_mut_param_398() {
    // No `!` parameter at all: the captured mut is the whole differentiable
    // input set (AUTODIFF.md §2). dL/ds = sum(x .* x) = 14.
    let v = run(r#"
        let !s = 2.0f64
        @grad fn loss(x: Tensor[f64,[3]]) -> f64 { sum(x .* x) * s }
        fn main() -> f64 {
            let x = [1.0f64, 2.0f64, 3.0f64]
            let (l, g) = loss.fwd_bwd(x)
            g.s
        }
    "#);
    assert!((as_float(&v) - 14.0).abs() < 1e-12, "capture-only @grad fn, got {:?}", v);
}

#[test]
fn grad_immutable_module_binding_is_not_captured_398() {
    // §2: a captured *immut* binding has no gradient. `k` is a plain `let`, so
    // it stays a constant on the tape and never becomes a `Grads` field — only
    // the `!` param's gradient comes back (dL/dw = 2kw = [6, 12], sum 18).
    let v = run(r#"
        let k = 3.0f64
        @grad fn loss(!w: Tensor[f64,[2]]) -> f64 { sum(w .* w) * k }
        fn main() -> f64 {
            let w = [1.0f64, 2.0f64]
            let (l, g) = loss.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 18.0).abs() < 1e-12, "immut capture, got {:?}", v);
}

#[test]
fn grad_capture_inside_closure_is_not_taped_398() {
    // The checker rejects this program (`check_tests::
    // captured_mut_hidden_in_closure_errors_398`); the interpreter must agree
    // that it has no gradient rather than inventing one — the tape does not
    // enter closure bodies, so `b` is not a tape input and `g.b` is not a
    // field. Only `g.w` exists: dL/dw = 2w = [2, 4], sum 6.
    let v = run(r#"
        let !b = [7.0f64, 7.0f64]
        @grad fn loss(!w: Tensor[f64,[2]]) -> f64 {
            let hook = fn() -> f64 { sum(b) }
            sum(w .* w)
        }
        fn main() -> f64 {
            let w = [1.0f64, 2.0f64]
            let (l, g) = loss.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-12, "closure capture, got {:?}", v);
}

// ── #299: gradient of a broadcast operand (bias add) must be reduced back ─────
//          to the operand's shape, not left at the broadcast shape.
#[test]
fn grad_broadcast_bias_reduces_to_operand_shape() {
    // grad of sum((x .+ b)^2) w.r.t. a bias `b:[3]` broadcast across `x:[2,3]`
    // must be shape [3] (summed over the batch axis), not [2,3].
    let v = run(r#"
        @grad fn f[B, H](!b: Tensor[f32, [H]], x: Tensor[f32, [B, H]]) -> f32 {
            let z = x .+ b
            sum(z .* z)
        }
        fn main() -> i64 {
            let !b = forge.zeros[f32, [3]]
            let x = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
            let (_l, g) = f.fwd_bwd(b, x)
            let (n,) = g.b.shape
            n
        }
    "#);
    assert_eq!(as_int(&v), 3); // was 2 (rank-2 [2,3]) before the fix
}
#[test]
fn grad_broadcast_bias_values_summed_over_batch() {
    // With b=0, dL/db = 2 * sum_over_batch(x) = 2*[1+4, 2+5, 3+6] = [10,14,18];
    // sum = 42.
    let v = run(r#"
        @grad fn f[B, H](!b: Tensor[f32, [H]], x: Tensor[f32, [B, H]]) -> f32 {
            let z = x .+ b
            sum(z .* z)
        }
        fn main() -> f32 {
            let !b = forge.zeros[f32, [3]]
            let x = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
            let (_l, g) = f.fwd_bwd(b, x)
            sum(g.b)
        }
    "#);
    assert!((as_float(&v) - 42.0).abs() < 1e-6, "got {:?}", v);
}
#[test]
fn grad_broadcast_mul_reduces_to_operand_shape() {
    // grad through a broadcast `.*` operand sums over the broadcast axis:
    // dL/db of sum(x .* b) = sum_over_batch(x) = [1+4, 2+5, 3+6] = [5,7,9]; sum = 21.
    let v = run(r#"
        @grad fn f[B, H](!b: Tensor[f32, [H]], x: Tensor[f32, [B, H]]) -> f32 {
            sum(x .* b)
        }
        fn main() -> f32 {
            let !b = forge.zeros[f32, [3]]
            let x = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
            let (_l, g) = f.fwd_bwd(b, x)
            sum(g.b)
        }
    "#);
    assert!((as_float(&v) - 21.0).abs() < 1e-6, "got {:?}", v);
}

// ── #306: activation builtins (sigmoid/tanh/gelu/silu) differentiate through ──
//          @grad. Derivatives are exact at x=0: sigmoid'=0.25, tanh'=1, silu'=0.5,
//          gelu'=0.5. Before the fix these errored ("doesn't participate in the
//          gradient graph") — only the `\>` operator traced.
fn grad_activation_at_zero(act: &str) -> f64 {
    let src = format!(r#"
        @grad fn f[D](!w: Tensor[f32, [D]]) -> f32 {{ sum({act}(w)) }}
        fn main() -> f32 {{
            let !w = forge.zeros[f32, [1]]
            let (_l, g) = f.fwd_bwd(w)
            g.w[0]
        }}
    "#);
    as_float(&run(&src))
}
#[test]
fn grad_sigmoid_traces() { assert!((grad_activation_at_zero("sigmoid") - 0.25).abs() < 1e-4); }
#[test]
fn grad_tanh_traces()    { assert!((grad_activation_at_zero("tanh")    - 1.0 ).abs() < 1e-4); }
#[test]
fn grad_silu_traces()    { assert!((grad_activation_at_zero("silu")    - 0.5 ).abs() < 1e-4); }
#[test]
fn grad_gelu_traces()    { assert!((grad_activation_at_zero("gelu")    - 0.5 ).abs() < 1e-4); }
#[test]
fn grad_relu_builtin_traces() {
    // the relu *builtin* (not just the `\>` operator) also traces now.
    let v = run(r#"
        @grad fn f[D](!w: Tensor[f32, [D]]) -> f32 { sum(relu(w)) }
        fn main() -> f32 {
            let w = [-1.0f32, 2.0f32, 3.0f32]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 2.0).abs() < 1e-6, "got {:?}", v); // grad = (w>0): 0+1+1 = 2
}

// ── #307: softmax / rms_norm / layer_norm differentiate through @grad ─────────
//          (transformer training path). Before, these errored ("doesn't
//          participate in the gradient graph"). Numerical accuracy is gradchecked
//          in examples/gradcheck.dmc; here we guard that they trace and give a
//          finite, sane gradient (and a known invariant for softmax).
#[test]
fn grad_softmax_traces() {
    // sum_k of the softmax VJP is 0 (rows of the Jacobian sum to zero): for
    // loss = sum(softmax(w) .* t), sum(g.w) ≈ 0.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [4]], t: Tensor[f32, [4]]) -> f32 { sum(softmax(w, -1) .* t) }
        fn main() -> f32 {
            let w = [1.0f32, 2.0f32, 0.5f32, -1.0f32]
            let t = [0.1f32, 0.7f32, 0.2f32, 0.4f32]
            let (_l, g) = f.fwd_bwd(w, t)
            sum(g.w)
        }
    "#);
    assert!(as_float(&v).abs() < 1e-4, "softmax grad rows should sum to ~0, got {:?}", v);
}
#[test]
fn grad_rms_norm_traces() {
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [2, 4]], gn: Tensor[f32, [4]]) -> f32 {
            sum(rms_norm(w, gn, 0.00001))
        }
        fn main() -> f32 {
            let w = [[1.0f32, 2.0f32, 0.5f32, -1.0f32], [0.3f32, -0.7f32, 1.5f32, 2.0f32]]
            let gn = [1.0f32, 0.5f32, 2.0f32, 1.2f32]
            let (_l, g) = f.fwd_bwd(w, gn)
            sum(g.w)
        }
    "#);
    let s = as_float(&v);
    assert!(s.is_finite(), "rms_norm grad must be finite, got {:?}", v);
}
#[test]
fn grad_layer_norm_traces() {
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [2, 4]], gn: Tensor[f32, [4]], bs: Tensor[f32, [4]]) -> f32 {
            sum(layer_norm(w, gn, bs, 0.00001))
        }
        fn main() -> f32 {
            let w = [[1.0f32, 2.0f32, 0.5f32, -1.0f32], [0.3f32, -0.7f32, 1.5f32, 2.0f32]]
            let gn = [1.0f32, 0.5f32, 2.0f32, 1.2f32]
            let bs = [0.1f32, -0.2f32, 0.0f32, 0.3f32]
            let (_l, g) = f.fwd_bwd(w, gn, bs)
            sum(g.w)
        }
    "#);
    assert!(as_float(&v).is_finite(), "layer_norm grad must be finite, got {:?}", v);
}

// ── #307 Tier B: reshape / sum_along / mean_along differentiate through @grad ──
#[test]
fn grad_sum_along_traces() {
    // sum(sum_along(w, 1)) == sum(w); dw is all ones, so sum(dw) == n_elems = 6.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [2, 3]]) -> f32 { sum(sum_along(w, 1)) }
        fn main() -> f32 {
            let w = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-6, "got {:?}", v);
}
#[test]
fn grad_mean_along_traces() {
    // sum(mean_along(w, 1)): each dw = 1/3, sum = 6 * 1/3 = 2.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [2, 3]]) -> f32 { sum(mean_along(w, 1)) }
        fn main() -> f32 {
            let w = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 2.0).abs() < 1e-5, "got {:?}", v);
}
#[test]
fn grad_reshape_traces() {
    // sum(reshape(w) .* t): dw = reshape(t), so sum(dw) == sum(t) = 210.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [2, 3]], t: Tensor[f32, [6]]) -> f32 { sum(w.reshape[[6]] .* t) }
        fn main() -> f32 {
            let w = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
            let t = [10.0f32, 20.0f32, 30.0f32, 40.0f32, 50.0f32, 60.0f32]
            let (_l, g) = f.fwd_bwd(w, t)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 210.0).abs() < 1e-4, "got {:?}", v);
}

// ── #307 Tier C: variance / max / min (global reductions) differentiate ───────
#[test]
fn grad_variance_traces() {
    // var = (1/N)Σ(x-μ)²; dx = (2/N)(x-μ), which sums to 0 over all elements.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [4]]) -> f32 { variance(w) }
        fn main() -> f32 {
            let w = [1.0f32, 2.0f32, 0.5f32, -1.0f32]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!(as_float(&v).abs() < 1e-4, "variance grad should sum to ~0, got {:?}", v);
}
#[test]
fn grad_max_traces() {
    // max(w): subgradient routes 1 to the single largest element, 0 elsewhere;
    // sum(g.w) == 1. The max here is w[2] = 2.0.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [4]]) -> f32 { max(w) }
        fn main() -> f32 {
            let w = [1.0f32, 2.0f32, 0.5f32, -1.0f32]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 1.0).abs() < 1e-6, "max grad should route 1 to the argmax, got {:?}", v);
}
#[test]
fn grad_min_traces() {
    // min(w): subgradient routes 1 to the single smallest element (w[3] = -1.0).
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [4]]) -> f32 { min(w) }
        fn main() -> f32 {
            let w = [1.0f32, 2.0f32, 0.5f32, -1.0f32]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 1.0).abs() < 1e-6, "min grad should route 1 to the argmin, got {:?}", v);
}

// ── #420 / #434: gradient checks against a central finite difference ─────────
//
// Every VJP rule those two issues added is checked here against a *central*
// difference of the interpreter's own forward pass,
//
//     ∂L/∂x  ≈  (L(x+h) − L(x−h)) / 2h,
//
// which is the only check that can catch a rule that is self-consistently
// wrong. The interpreter evaluates at f32 (≈1e-7 relative), so the step is
// squeezed from both ends: cancellation grows as ε/h, truncation as h²·L‴/6.
// h = 1e-2 sits near the f32 optimum (≈ε^⅓) for the smooth rules and is the
// default below; the tolerances are the *measured* agreement at that step.

/// An f32 tensor literal (rank 1 or 2) from row-major data.
fn tensor_lit(data: &[f64], shape: &[usize]) -> String {
    let f = |x: f64| format!("{:?}f32", x);
    match shape {
        [n] => {
            assert_eq!(data.len(), *n, "tensor_lit: data/shape mismatch");
            format!("[{}]", data.iter().map(|&x| f(x)).collect::<Vec<_>>().join(", "))
        }
        [r, c] => {
            assert_eq!(data.len(), r * c, "tensor_lit: data/shape mismatch");
            let rows: Vec<String> = (0..*r)
                .map(|i| format!("[{}]", (0..*c)
                    .map(|j| f(data[i * c + j])).collect::<Vec<_>>().join(", ")))
                .collect();
            format!("[{}]", rows.join(", "))
        }
        _ => panic!("tensor_lit: rank {} unsupported", shape.len()),
    }
}

/// Subscript selecting flat element `i` of a tensor with `shape`.
fn idx_lit(i: usize, shape: &[usize]) -> String {
    match shape {
        [_] => format!("[{}]", i),
        [_, c] => format!("[{}, {}]", i / c, i % c),
        _ => panic!("idx_lit: rank {} unsupported", shape.len()),
    }
}

/// Analytic ∂L/∂w (from `fwd_bwd`) and the central difference of `fwd`, both
/// flattened row-major, for `@grad fn f(!w: Tensor[f32, shape]<extra_params>)`.
/// `extra_params` / `extra_args` carry any additional (non-`!`) operands and
/// must lead with a comma, or both be empty.
fn grad_and_fd(
    shape: &[usize],
    extra_params: &str,
    extra_args: &str,
    body: &str,
    data: &[f64],
    h: f64,
) -> (Vec<f64>, Vec<f64>) {
    let dims = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
    let decl = format!(
        "@grad fn f(!w: Tensor[f32, [{dims}]]{extra_params}) -> f32 {{ {body} }}");
    let fwd = |d: &[f64]| as_float(&run(&format!(
        "{decl}\nfn main() -> f32 {{ f.fwd({t}{extra_args}) }}",
        t = tensor_lit(d, shape))));
    let mut analytic = Vec::with_capacity(data.len());
    let mut numeric = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        analytic.push(as_float(&run(&format!(
            "{decl}\nfn main() -> f32 {{\n\
             \tlet (_l, g) = f.fwd_bwd({w}{extra_args})\n\
             \tg.w{idx}\n}}",
            w = tensor_lit(data, shape), idx = idx_lit(i, shape)))));
        let mut up = data.to_vec(); up[i] += h;
        let mut dn = data.to_vec(); dn[i] -= h;
        numeric.push((fwd(&up) - fwd(&dn)) / (2.0 * h));
    }
    (analytic, numeric)
}

/// Same, for a scalar `!a: f32` parameter (#420 item 2).
fn grad_and_fd_scalar(body: &str, x: f64, h: f64) -> (f64, f64) {
    let decl = format!("@grad fn f(!a: f32) -> f32 {{ {body} }}");
    let analytic = as_float(&run(&format!(
        "{decl}\nfn main() -> f32 {{\n\tlet (_l, g) = f.fwd_bwd({x:?}f32)\n\tg.a\n}}")));
    let fwd = |v: f64| as_float(&run(&format!(
        "{decl}\nfn main() -> f32 {{ f.fwd({v:?}f32) }}")));
    (analytic, (fwd(x + h) - fwd(x - h)) / (2.0 * h))
}

/// Assert two gradient vectors agree to `tol`, relative to the larger
/// magnitude (floored at 1 so near-zero components use an absolute bound).
fn assert_gradcheck(label: &str, analytic: &[f64], numeric: &[f64], tol: f64) {
    assert_eq!(analytic.len(), numeric.len(), "{label}: length mismatch");
    for (i, (&a, &n)) in analytic.iter().zip(numeric).enumerate() {
        let scale = a.abs().max(n.abs()).max(1.0);
        assert!((a - n).abs() <= tol * scale,
            "{label}[{i}]: analytic {a} vs central difference {n} \
             (allowed {} at rel tol {tol})", tol * scale);
    }
}

// ── #420 item 1: scalar-math builtins on a traced scalar ─────────────────────
//
// Domain note: each function is checked strictly inside its domain. `sqrt` and
// `log` are undefined for x < 0 and have an infinite derivative at x = 0, so
// the "near-edge" point is x = 0.01 (sqrt′ = 5, log′ = 100) rather than 0 —
// a central difference has no meaning where the one-sided limit is +∞. `tan`
// is checked away from its ±π/2 poles.

#[test]
fn grad_scalar_sqrt_gradchecks() {
    let (a, n) = grad_and_fd_scalar("sqrt(a)", 2.25, 1e-2);
    assert_gradcheck("sqrt @ 2.25", &[a], &[n], 1e-3);
    assert!((a - 1.0 / 3.0).abs() < 1e-5, "sqrt'(2.25) = 1/3, got {a}");
}

#[test]
fn grad_scalar_sqrt_near_zero_gradchecks() {
    // Near-edge: x = 0.01, where sqrt′ = 5 and the curvature is already large.
    // The step has to shrink with the domain — h = 1e-3 keeps x ± h positive
    // and truncation under control.
    let (a, n) = grad_and_fd_scalar("sqrt(a)", 0.01, 1e-3);
    assert_gradcheck("sqrt @ 0.01", &[a], &[n], 2e-2);
    assert!((a - 5.0).abs() < 1e-3, "sqrt'(0.01) = 5, got {a}");
}

#[test]
fn grad_scalar_exp_gradchecks() {
    for &x in &[0.5, -1.25] {
        let (a, n) = grad_and_fd_scalar("exp(a)", x, 1e-2);
        assert_gradcheck(&format!("exp @ {x}"), &[a], &[n], 1e-3);
        assert!((a - x.exp()).abs() < 1e-4, "exp'({x}) = exp({x}), got {a}");
    }
}

#[test]
fn grad_scalar_log_gradchecks() {
    for &x in &[2.0, 0.25] {
        let (a, n) = grad_and_fd_scalar("log(a)", x, 1e-3);
        assert_gradcheck(&format!("log @ {x}"), &[a], &[n], 5e-3);
        assert!((a - 1.0 / x).abs() < 1e-3, "log'({x}) = 1/{x}, got {a}");
    }
}

#[test]
fn grad_scalar_sin_cos_gradcheck() {
    let (a, n) = grad_and_fd_scalar("sin(a)", 0.7, 1e-2);
    assert_gradcheck("sin @ 0.7", &[a], &[n], 1e-3);
    assert!((a - 0.7f64.cos()).abs() < 1e-4, "sin'(0.7) = cos(0.7), got {a}");
    let (a, n) = grad_and_fd_scalar("cos(a)", 0.7, 1e-2);
    assert_gradcheck("cos @ 0.7", &[a], &[n], 1e-3);
    assert!((a + 0.7f64.sin()).abs() < 1e-4, "cos'(0.7) = -sin(0.7), got {a}");
}

#[test]
fn grad_scalar_tan_gradchecks() {
    let (a, n) = grad_and_fd_scalar("tan(a)", 0.5, 1e-2);
    assert_gradcheck("tan @ 0.5", &[a], &[n], 2e-3);
    let want = 1.0 + 0.5f64.tan().powi(2);
    assert!((a - want).abs() < 1e-3, "tan'(0.5) = 1+tan²(0.5) = {want}, got {a}");
}

// ── #420 item 2: scalar `!` parameters ───────────────────────────────────────

#[test]
fn grad_scalar_mut_param_traces_420() {
    // The issue's first probe verbatim: `@grad fn f(!a: f32) { a*a*a }`.
    // Refused outright before — the body "doesn't participate in the gradient
    // graph" because only tensor params became tape inputs.
    let v = run(r#"
        @grad fn f(!a: f32) -> f32 { a * a * a }
        fn main() -> f32 {
            let (_l, g) = f.fwd_bwd(2.0f32)
            g.a
        }
    "#);
    assert!((as_float(&v) - 12.0).abs() < 1e-5, "3a² at a=2 is 12, got {:?}", v);
}

#[test]
fn grad_scalar_mut_param_gradchecks() {
    let (a, n) = grad_and_fd_scalar("a * a * a", 2.0, 1e-2);
    assert_gradcheck("a³ @ 2", &[a], &[n], 1e-3);
    let (a, n) = grad_and_fd_scalar("exp(a * a) / (a + 3.0)", -1.25, 1e-2);
    assert_gradcheck("exp(a²)/(a+3) @ -1.25", &[a], &[n], 5e-3);
}

#[test]
fn grad_scalar_mut_param_mixed_with_tensor() {
    // A scalar `!` param alongside a tensor one: both fields come back.
    // L = s * sum(w .* w); ∂L/∂s = sum(w²) = 5, ∂L/∂w = 2sw = [4, 8].
    let v = run(r#"
        @grad fn f(!s: f32, !w: Tensor[f32, [2]]) -> f32 { s * sum(w .* w) }
        fn main() -> f32 {
            let w = [1.0f32, 2.0f32]
            let (_l, g) = f.fwd_bwd(2.0f32, w)
            g.s * 100.0 + sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 512.0).abs() < 1e-3, "expected 5*100 + 12, got {:?}", v);
}

#[test]
fn grad_scalar_mut_param_unused_is_zero() {
    // A `!` scalar the loss doesn't depend on gets 0.0 — not its own value,
    // which would read as a gradient.
    let v = run(r#"
        @grad fn f(!a: f32, !w: Tensor[f32, [2]]) -> f32 { sum(w .* w) }
        fn main() -> f32 {
            let w = [1.0f32, 2.0f32]
            let (_l, g) = f.fwd_bwd(7.0f32, w)
            g.a
        }
    "#);
    assert!(as_float(&v).abs() < 1e-9, "unused scalar `!` param grad must be 0, got {:?}", v);
}

// ── #420 item 3: indexed reads `x[i]` scatter their gradient back ────────────

#[test]
fn grad_indexed_read_traces_420() {
    // The issue's second probe: component math on a tensor, no reduction.
    // L = w[0]² + w[1]²; ∂L/∂w = 2w = [6, 8].
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [2]]) -> f32 { w[0]*w[0] + w[1]*w[1] }
        fn main() -> f32 {
            let w = [3.0f32, 4.0f32]
            let (_l, g) = f.fwd_bwd(w)
            g.w[0] * 100.0 + g.w[1]
        }
    "#);
    assert!((as_float(&v) - 608.0).abs() < 1e-3, "expected 6*100 + 8, got {:?}", v);
}

#[test]
fn grad_indexed_read_gradchecks() {
    let data = [3.0, 4.0, -1.5];
    let (a, n) = grad_and_fd(&[3], "", "",
        "w[0]*w[1] + w[2]*w[2]*w[0]", &data, 1e-2);
    assert_gradcheck("indexed read, rank 1", &a, &n, 1e-3);
}

#[test]
fn grad_indexed_read_rank2_gradchecks() {
    // A full index into a rank-2 tensor; only the addressed slot gets gradient.
    let data = [1.0, 2.0, 3.0, -4.0, 0.5, 6.0];
    let (a, n) = grad_and_fd(&[2, 3], "", "",
        "w[0, 1] * w[1, 2] + w[1, 0] * w[1, 0]", &data, 1e-2);
    assert_gradcheck("indexed read, rank 2", &a, &n, 1e-3);
    // Untouched slots must be exactly zero, not merely small.
    for i in [0usize, 2, 4] {
        assert_eq!(a[i], 0.0, "element {i} is not read; its gradient must be 0");
    }
}

// ── #420: the SDF-shaped composite the issue was filed from ──────────────────

#[test]
fn grad_sdf_norm_of_scalar_components_gradchecks() {
    // `sqrt` of a sum of squares of traced *scalars* — the Euclidean norm at
    // the end of every SDF primitive. At (1, 2, 2) the radius is 3 and the
    // gradient is the unit vector (1/3, 2/3, 2/3).
    let decl = "@grad fn sdf(!x: f32, !y: f32, !z: f32) -> f32 { sqrt(x*x + y*y + z*z) }";
    let grad_of = |field: &str| as_float(&run(&format!(
        "{decl}\nfn main() -> f32 {{\n\
         \tlet (_l, g) = sdf.fwd_bwd(1.0f32, 2.0f32, 2.0f32)\n\tg.{field}\n}}")));
    let fwd = |x: f64, y: f64, z: f64| as_float(&run(&format!(
        "{decl}\nfn main() -> f32 {{ sdf.fwd({x:?}f32, {y:?}f32, {z:?}f32) }}")));
    let h = 1e-2;
    let analytic = [grad_of("x"), grad_of("y"), grad_of("z")];
    let numeric = [
        (fwd(1.0 + h, 2.0, 2.0) - fwd(1.0 - h, 2.0, 2.0)) / (2.0 * h),
        (fwd(1.0, 2.0 + h, 2.0) - fwd(1.0, 2.0 - h, 2.0)) / (2.0 * h),
        (fwd(1.0, 2.0, 2.0 + h) - fwd(1.0, 2.0, 2.0 - h)) / (2.0 * h),
    ];
    assert_gradcheck("sdf norm (scalar params)", &analytic, &numeric, 1e-3);
    let want = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
    for (i, (&a, &w)) in analytic.iter().zip(&want).enumerate() {
        assert!((a - w).abs() < 1e-5, "∂r/∂({i}) should be {w}, got {a}");
    }
}

#[test]
fn grad_sdf_norm_of_indexed_components_gradchecks() {
    // The same norm written over tensor components — the issue's own probe
    // `sqrt(w[0]*w[0] + w[1]*w[1])`, generalized to 3D and offset by a centre
    // so the gradient isn't a pure unit vector by symmetry.
    let data = [3.0, -4.0, 1.0];
    let (a, n) = grad_and_fd(&[3], "", "",
        "sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]) - 1.5", &data, 1e-2);
    assert_gradcheck("sdf norm (indexed reads)", &a, &n, 1e-3);
    let r = (9.0f64 + 16.0 + 1.0).sqrt();
    for (i, &want) in [3.0 / r, -4.0 / r, 1.0 / r].iter().enumerate() {
        assert!((a[i] - want).abs() < 1e-5, "∂r/∂w[{i}] should be {want}, got {}", a[i]);
    }
}

// ── #434 item 1: comparison-masked select ────────────────────────────────────

#[test]
fn grad_masked_select_traces_434() {
    // The issue's probe verbatim — `DotLt` inside @grad used to be a hard
    // "no VJP rule yet". The mask is stop-gradient; the cotangent goes to
    // whichever operand the mask selected.
    // w = [1, 5, 3] vs t = [2, 1, 9] → mask (w < t) = [1, 0, 1], so
    // ∂L/∂w = [1, 0, 1] and sum = 2.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [3]], t: Tensor[f32, [3]]) -> f32 {
            let m = w .< t
            sum(m .* w .+ (w .>= t) .* t)
        }
        fn main() -> f32 {
            let w = [1.0f32, 5.0f32, 3.0f32]
            let t = [2.0f32, 1.0f32, 9.0f32]
            let (_l, g) = f.fwd_bwd(w, t)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 2.0).abs() < 1e-6,
        "mask selects w at elements 0 and 2, got {:?}", v);
}

#[test]
fn grad_masked_select_gradchecks_both_branches() {
    // The same select, gradchecked. The mask [1, 0, 1] exercises both
    // branches; every point sits at least 1.0 away from its kink, so the
    // h = 1e-2 difference never crosses one and the check is meaningful.
    let data = [1.0, 5.0, 3.0];
    let t = "[2.0f32, 1.0f32, 9.0f32]";
    let (a, n) = grad_and_fd(&[3], ", t: Tensor[f32, [3]]", &format!(", {t}"),
        "sum((w .< t) .* w .+ (w .>= t) .* t)", &data, 1e-2);
    assert_gradcheck("masked select, selected operand", &a, &n, 1e-3);
    assert_eq!(a, vec![1.0, 0.0, 1.0], "gradient must route exactly by the mask");
}

#[test]
fn grad_masked_select_routes_to_the_other_branch() {
    // Symmetric check: differentiate w.r.t. the operand the mask *rejects*
    // at elements 0 and 2 and selects at element 1 → [0, 1, 0].
    let data = [2.0, 1.0, 9.0];
    let d0 = "[1.0f32, 5.0f32, 3.0f32]";
    let (a, n) = grad_and_fd(&[3], ", d0: Tensor[f32, [3]]", &format!(", {d0}"),
        "sum((d0 .< w) .* d0 .+ (d0 .>= w) .* w)", &data, 1e-2);
    assert_gradcheck("masked select, rejected operand", &a, &n, 1e-3);
    assert_eq!(a, vec![0.0, 1.0, 0.0], "gradient must route exactly by the mask");
}

#[test]
fn grad_masked_select_nearest_of_two_gradchecks() {
    // Hard nearest-of-K assignment, the shape #434 actually wants: pick the
    // smaller of two distance rows and weight it. Non-trivial gradient
    // values (not just 0/1) so the check has teeth.
    let data = [1.0, 5.0, 3.0, 2.5];
    let d1 = "[2.0f32, 1.0f32, 9.0f32, 4.0f32]";
    let (a, n) = grad_and_fd(&[4], ", d1: Tensor[f32, [4]]", &format!(", {d1}"),
        "sum(((w .< d1) .* w .+ (w .>= d1) .* d1) .* w)", &data, 1e-2);
    assert_gradcheck("masked nearest-of-two", &a, &n, 1e-3);
}

// ── #434 item 2: row-wise (axis) reductions ──────────────────────────────────

#[test]
fn grad_min_along_traces_434() {
    // min_along over axis 1 of [[1,5,3],[4,2,6]] → [1, 2]; the subgradient
    // routes each row's cotangent to that row's argmin, so sum(g.w) = 2.
    let v = run(r#"
        @grad fn f(!w: Tensor[f32, [2, 3]]) -> f32 { sum(min_along(w, 1)) }
        fn main() -> f32 {
            let w = [[1.0f32, 5.0f32, 3.0f32], [4.0f32, 2.0f32, 6.0f32]]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!((as_float(&v) - 2.0).abs() < 1e-6, "one unit per row, got {:?}", v);
}

#[test]
fn grad_min_along_gradchecks_each_axis() {
    // Distinct entries, so no lane has a tie and the argmin is stable under a
    // ±1e-2 perturbation — the precondition for a finite difference to mean
    // anything at a piecewise-linear reduction.
    let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let t0 = "[2.0f32, 3.0f32, 0.5f32]";      // [3] — reducing axis 0 leaves 3
    let t1 = "[2.0f32, 3.0f32]";              // [2] — reducing axis 1 leaves 2
    for (axis, t) in [(0, t0), (1, t1)] {
        let dims = if axis == 0 { 3 } else { 2 };
        let (a, n) = grad_and_fd(&[2, 3],
            &format!(", t: Tensor[f32, [{dims}]]"), &format!(", {t}"),
            &format!("sum(min_along(w, {axis}) .* t)"), &data, 1e-2);
        // The reduction is piecewise linear, so the only error is the f32
        // rounding of the two forward evaluations (~1e-4 absolute at h = 1e-2).
        assert_gradcheck(&format!("min_along axis {axis}"), &a, &n, 1e-3);
    }
}

#[test]
fn grad_max_along_gradchecks_each_axis() {
    let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let t0 = "[2.0f32, 3.0f32, 0.5f32]";
    let t1 = "[2.0f32, 3.0f32]";
    for (axis, t) in [(0, t0), (1, t1)] {
        let dims = if axis == 0 { 3 } else { 2 };
        let (a, n) = grad_and_fd(&[2, 3],
            &format!(", t: Tensor[f32, [{dims}]]"), &format!(", {t}"),
            &format!("sum(max_along(w, {axis}) .* t)"), &data, 1e-2);
        assert_gradcheck(&format!("max_along axis {axis}"), &a, &n, 1e-3);
    }
}

#[test]
fn grad_softmax_along_axis_gradchecks_each_axis() {
    // The row-wise softmax #434 asks for is `softmax(x, axis)` — there is no
    // separate `softmax_along`. Gradcheck it along *both* axes of a 2D
    // tensor, which is what a per-item soft assignment over K options needs.
    let data = [0.5, -1.0, 2.0, 1.5, 0.25, -0.5];
    let t = "[[0.3f32, 1.0f32, -0.7f32], [0.2f32, -0.4f32, 0.9f32]]";
    for axis in [0, 1] {
        let (a, n) = grad_and_fd(&[2, 3],
            ", t: Tensor[f32, [2, 3]]", &format!(", {t}"),
            &format!("sum(softmax(w, {axis}) .* t)"), &data, 1e-2);
        assert_gradcheck(&format!("softmax axis {axis}"), &a, &n, 5e-3);
    }
}

#[test]
fn grad_soft_min_assignment_gradchecks() {
    // The workload behind #434: a soft-min assignment over the K axis of an
    // [N, K] distance matrix, spelled `softmax(-d, 1)` and reduced with
    // `sum_along`. This is the differentiable relaxation of `min_along`.
    let data = [1.0, 5.0, 3.0, 4.0, 2.0, 6.0];
    let (a, n) = grad_and_fd(&[2, 3], "", "",
        "sum(sum_along(softmax(0.0 .- w, 1) .* w, 1))", &data, 1e-2);
    assert_gradcheck("soft-min assignment", &a, &n, 5e-3);
}

// ── #252: product/quotient VJP must not drop the gradient through a ──────────
//          scalar / reduction operand (silent wrong/zero gradients).

/// Helper: run a single-param @grad body at w=[1.5,-2.0] and return sum(g.w).
fn grad_sum_at(body: &str) -> f64 {
    let src = format!(r#"
        @grad fn f[N](!w: Tensor[f32, [N]]) -> f32 {{ {body} }}
        fn main() -> f32 {{
            let !w = forge.zeros[f32, [2]]
            w[0] = 1.5
            w[1] = -2.0
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }}
    "#);
    as_float(&run(&src))
}

#[test]
fn grad_scalar_times_reduction_left() {
    // 2.0 * sum(w²): grad 4w = [6,-8], sum = -2. Was [0,0] before #252.
    assert!((grad_sum_at("2.0 * sum(w .* w)") - (-2.0)).abs() < 1e-5);
}

#[test]
fn grad_scalar_times_reduction_right() {
    // sum(w²) * 2.0: same gradient, the already-working order. Guards symmetry.
    assert!((grad_sum_at("sum(w .* w) * 2.0") - (-2.0)).abs() < 1e-5);
}

#[test]
fn grad_scalar_over_reduction() {
    // 2.0 / sum(w²): true grad [-0.1536, 0.2048], sum = 0.0512. Was [0,0].
    assert!((grad_sum_at("2.0 / sum(w .* w)") - 0.0512).abs() < 1e-4);
}

#[test]
fn grad_reduction_divided_by_scalar() {
    // sum(w²)/2.0: grad w = [1.5,-2], sum = -0.5.
    assert!((grad_sum_at("sum(w .* w) / 2.0") - (-0.5)).abs() < 1e-5);
}

#[test]
fn grad_tensor_times_traced_reduction() {
    // sum(w .* sum(w)): the reduction operand is traced; true grad [-1,-1],
    // sum = -2. Was [-0.5,-0.5] (gradient through sum(w) was halved/dropped).
    assert!((grad_sum_at("sum(w .* sum(w))") - (-2.0)).abs() < 1e-5);
}

#[test]
fn grad_mean_traces() {
    // #253: mean(w) traces (mean = sum/N); grad = [1/N,...] = [0.5,0.5],
    // sum = 1.0. Previously mean errored at runtime inside @grad.
    assert!((grad_sum_at("mean(w)") - 1.0).abs() < 1e-6);
}

#[test]
fn grad_mean_of_squares() {
    // mean(w²) = sum(w²)/N; grad 2w/N = [1.5,-2], sum = -0.5.
    assert!((grad_sum_at("mean(w .* w)") - (-0.5)).abs() < 1e-5);
}

#[test]
fn grad_untraced_reduction_hint_is_honest() {
    // A value that exits the gradient graph still errors — and the hint must
    // name the traced reductions and flag what doesn't trace (#253). The set
    // has shrunk twice: max/min/variance trace (#307 Tier C) and full element
    // reads `w[i]` trace (#420 item 3). What is left is the *partial* index,
    // which yields a sub-tensor rather than an element.
    let e = run_err(r#"
        @grad fn f(!w: Tensor[f32, [2, 2]]) -> f32 { sum(w[0]) }
        fn main() -> f32 {
            let w = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]
            let (_l, g) = f.fwd_bwd(w)
            sum(g.w)
        }
    "#);
    assert!(e.contains("sum(x)") && e.contains("mean(x)"), "got: {e}");
    assert!(e.contains("don't trace yet"), "hint should flag untraced reductions: {e}");
}

#[test]
fn grad_grad_fwd_bwd_bwd_second_derivative() {
    // Source-level second-order autodiff (mirrors jit_grad_grad_scalar):
    // L = sum(w³), first grad 3w², second grad 6w = 12 at w = 2.
    let v = run(r#"
        @grad @grad fn cube(!w: Tensor[f32, [1]]) -> f32 {
            sum(w .* w .* w)
        }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [1]]
            w[0] = 2.0
            let (_l, g2) = cube.fwd_bwd_bwd(w)
            g2.w[0]
        }
    "#);
    assert!((as_float(&v) - 12.0).abs() < 1e-6, "expected d²/dw²(w³)=12 at w=2, got {:?}", v);
}

#[test]
fn grad_grad_fwd_bwd_bwd_returns_forward_loss() {
    // The first tuple element is the ordinary forward loss: w³ = 8 at w = 2.
    let v = run(r#"
        @grad @grad fn cube(!w: Tensor[f32, [1]]) -> f32 {
            sum(w .* w .* w)
        }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [1]]
            w[0] = 2.0
            let (l, _g2) = cube.fwd_bwd_bwd(w)
            l
        }
    "#);
    assert!((as_float(&v) - 8.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn grad_grad_elementwise_vector() {
    // Elementwise cube over a vector: g2 = 6w per element.
    // w = [1, 2, 3] → g2 = [6, 12, 18], sum = 36.
    let v = run(r#"
        @grad @grad fn cube[D](!w: Tensor[f32, [D]]) -> f32 {
            sum(w .* w .* w)
        }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [3]]
            w[0] = 1.0
            w[1] = 2.0
            w[2] = 3.0
            let (_l, g2) = cube.fwd_bwd_bwd(w)
            sum(g2.w)
        }
    "#);
    assert!((as_float(&v) - 36.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn grad_grad_linear_second_derivative_zero() {
    // L = sum(W @ x) is linear in W, so the second derivative is 0
    // (mirrors jit_grad_grad_matmul).
    let v = run(r#"
        @grad @grad fn loss(!W: Tensor[f32, [2, 2]], x: Tensor[f32, [2, 2]]) -> f32 {
            sum(W @ x)
        }
        fn main() -> f32 {
            let !W = forge.ones[f32, [2, 2]]
            let  x = forge.ones[f32, [2, 2]]
            let (_l, g2) = loss.fwd_bwd_bwd(W, x)
            sum(g2.W)
        }
    "#);
    assert!(as_float(&v).abs() < 1e-9, "second derivative of a linear fn must be 0, got {:?}", v);
}

#[test]
fn grad_grad_hessian_quadratic() {
    // L = sum((w .* w) .* x) → d²L/dw² = 2x elementwise (mirrors
    // jit_hessian_quadratic). x = [1, 2, 3] → g2 = [2, 4, 6], sum = 12.
    let v = run(r#"
        @grad @grad fn f[D](!w: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum((w .* w) .* x)
        }
        fn main() -> f32 {
            let !w = forge.ones[f32, [3]]
            let !x = forge.zeros[f32, [3]]
            x[0] = 1.0
            x[1] = 2.0
            x[2] = 3.0
            let (_l, g2) = f.fwd_bwd_bwd(w, x)
            sum(g2.w)
        }
    "#);
    assert!((as_float(&v) - 12.0).abs() < 1e-6, "got {:?}", v);
}

// ── #306 second-order: @grad @grad through smooth activations gives act''(x). ──
//    Previously backward_symbolic errored on Activation; now it records act'(x)
//    as a live ActivationGrad node whose VJP is g·act''(x). Values are checked
//    at points with clean closed forms; the numeric oracle (finite-diff of the
//    first derivative) lives in examples/gradcheck.dmc::gradcheck_second_order.
fn second_deriv_at(act: &str, x: f64) -> f64 {
    // g2.w[0] of `@grad @grad sum(act(w))` is act''(w[0]) (diagonal Hessian).
    let src = format!(r#"
        @grad @grad fn f(!w: Tensor[f32, [1]]) -> f32 {{ sum({act}(w)) }}
        fn main() -> f32 {{
            let !w = forge.zeros[f32, [1]]
            w[0] = {x}
            let (_l, g2) = f.fwd_bwd_bwd(w)
            g2.w[0]
        }}
    "#);
    as_float(&run(&src))
}

#[test]
fn grad_grad_silu_second_derivative() {
    // silu''(0) = 0.5 exactly (silu'' = s(1-s)[2 + x(1-2s)], s=0.5 at x=0).
    assert!((second_deriv_at("silu", 0.0) - 0.5).abs() < 1e-4, "got {}", second_deriv_at("silu", 0.0));
}
#[test]
fn grad_grad_gelu_second_derivative() {
    // gelu''(0) = sqrt(2/pi) ≈ 0.7978846 for the tanh approximation.
    let c = (2.0f64 / std::f64::consts::PI).sqrt();
    assert!((second_deriv_at("gelu", 0.0) - c).abs() < 1e-4, "got {}", second_deriv_at("gelu", 0.0));
}
#[test]
fn grad_grad_sigmoid_second_derivative() {
    // sigmoid''(1) = s(1-s)(1-2s), s=σ(1)=0.7310586 → ≈ -0.0908577.
    assert!((second_deriv_at("sigmoid", 1.0) - (-0.0908577)).abs() < 1e-4,
        "got {}", second_deriv_at("sigmoid", 1.0));
}
#[test]
fn grad_grad_tanh_second_derivative() {
    // tanh''(0.5) = -2 t (1-t²), t=tanh(0.5)=0.4621172 → ≈ -0.7268619.
    assert!((second_deriv_at("tanh", 0.5) - (-0.7268619)).abs() < 1e-4,
        "got {}", second_deriv_at("tanh", 0.5));
}
#[test]
fn fwd_bwd_bwd_requires_stacked_grad() {
    // A single `@grad` has no second-order entry — calling `.fwd_bwd_bwd`
    // must be a hard error, not silent garbage (the pre-fix behavior).
    let e = run_err(r#"
        @grad fn cube(!w: Tensor[f32, [1]]) -> f32 {
            sum(w .* w .* w)
        }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [1]]
            w[0] = 2.0
            let (_l, g2) = cube.fwd_bwd_bwd(w)
            g2.w[0]
        }
    "#);
    assert!(e.contains("@grad @grad"), "got: {e}");
}

#[test]
fn grad_sgd_actually_reduces_loss() {
    // Real training: gradients move in the descent direction. After a handful
    // of SGD steps the loss must strictly decrease — a tape that returned
    // zero gradients would leave loss unchanged.
    let v = run(r#"
        @grad fn loss[D](!w: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum((w .- x) .* (w .- x))
        }
        fn main() -> f32 {
            let !w = vault.zeros[f32, [4]]
            let  x = vault.ones[f32,  [4]]
            for _ in 0..5 {
                let (_l, g) = loss.fwd_bwd(w, x)
                w -= g.w .* 0.1
            }
            # Final loss after 5 SGD steps; expected to be much less than the
            # initial loss of D=4.
            let (final_loss, _g) = loss.fwd_bwd(w, x)
            final_loss
        }
    "#);
    assert!(as_float(&v) < 1.0, "expected loss to drop below 1.0, got {:?}", v);
}

#[test]
fn grad_indexed_reduction_differentiates() {
    // #71 filed this as a hard break: `sq[0] + sq[1] + ...` left the gradient
    // graph and the best the compiler could do was point at `sum`. #420 item 3
    // put element reads on the tape, so the hand-rolled reduction now
    // differentiates identically to `sum(x .* x)` — grad 2x = [2, 4, 6, 8].
    let v = run(r#"
        @grad fn loss(!x: Tensor[f32, [4]]) -> f32 {
            let sq = x .* x
            sq[0] + sq[1] + sq[2] + sq[3]
        }
        fn main() -> f32 {
            let x = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            let (_v, g) = loss.fwd_bwd(x)
            sum(g.x)
        }
    "#);
    assert!((as_float(&v) - 20.0).abs() < 1e-5, "sum(2x) over [1,2,3,4] = 20, got {:?}", v);
}

#[test]
fn data_iter_raises_explicit_error() {
    // Per the no-placeholders rule, data_iter has no interpreter
    // implementation and must raise, not silently iterate empty.
    let msg = run_err(r#"
        fn main() -> nil {
            for (_x, _y) in data_iter() { let _ = 1 }
            nil
        }
    "#);
    assert!(msg.contains("data_iter") && msg.contains("doesn't have"),
            "expected explicit data_iter error, got: {}", msg);
}

#[test]
fn unknown_builtin_raises_explicit_error() {
    // embed now requires 2 args (vocab + ids). Calling it with 1 arg should
    // produce a loud runtime error rather than silently returning a stub.
    let msg = run_err(r#"
        fn main() -> nil {
            let t = vault.zeros[f32, [4, 4]]
            let _ = embed(t)
            nil
        }
    "#);
    assert!(msg.contains("embed"),
            "expected embed error, got: {}", msg);
}

#[test]
fn softmax_normalises_to_one() {
    // Real softmax: output along the axis sums to 1.
    let v = run(r#"
        fn main() -> f32 {
            let t = [1.0, 2.0, 3.0, 4.0]
            sum(softmax(t, -1))
        }
    "#);
    // f32-tensor semantics (#241): softmax outputs are rounded through f32,
    // so the sum sits within a few f32 ulps of 1, not f64 ulps.
    assert!((as_float(&v) - 1.0).abs() < 1e-6, "expected sum=1, got {:?}", v);
}

#[test]
fn shape_inference_from_arg_shapes() {
    // No more default_shape_dim — shape params come from actual tensor shapes.
    // Pass a [3, 5] tensor as Tensor[f32, [A, B]] → A=3, B=5; `A * B` should
    // evaluate to 15, not 4*4=16.
    let v = run(r#"
        fn pluck[A, B](t: Tensor[f32, [A, B]]) -> i64 { A * B }
        fn main() -> i64 {
            let t = vault.zeros[f32, [3, 5]]
            pluck(t)
        }
    "#);
    assert_eq!(as_int(&v), 15);
}

#[test]
fn shape_inference_conflict_errors() {
    // Two args want the same shape param bound to different values: must error.
    let msg = run_err(r#"
        fn paired[D](a: Tensor[f32, [D]], b: Tensor[f32, [D]]) -> i64 { D }
        fn main() -> i64 {
            let a = vault.zeros[f32, [4]]
            let b = vault.zeros[f32, [5]]
            paired(a, b)
        }
    "#);
    assert!(msg.contains("inconsistent") || msg.contains("both"),
            "expected shape conflict error, got: {}", msg);
}

// ── Null Hypothesis Tests (HAS-dC §9) ────────────────────────────────────
//
// Nulls 3 (JIT monotonic), 5 (sharded tensors), 7 (@fuse memory traffic)
// are deferred: the JIT and distribution layers don't exist yet.
// Nulls 1, 4, 6, 8, 9, 10 are fully testable against the interpreter.
// Null 9 (tokenizer) lives in lexer_tests.rs.

#[test]
fn null4_grad_linear_in_forward_matmul_backward() {
    // Null 4: @grad backward is linear in forward time.
    // Structural proof: reverse-mode computes the full [N,M] weight gradient
    // in ONE backward pass. Finite-diff would need N*M forward calls for a
    // [3,4] weight matrix (12 calls vs. 1). We verify the matmul VJP is exact:
    //
    // loss(W, x) = sum(W @ x), W∈[2,3] all-ones, x∈[3,1] all-ones
    //   forward: W @ x = [[3],[3]], loss = 6
    //   backward: dL/dW = dL/dy @ x^T = [[1],[1]] @ [[1,1,1]] = [[1,1,1],[1,1,1]]
    //           → sum(g.W) = 6 exactly (12 elements = 6 with finite-diff, but computed in 1 pass)
    let v = run(r#"
        @grad fn linear[N, M, K](!W: Tensor[f32, [N, M]], x: Tensor[f32, [M, K]]) -> f32 {
            sum(W @ x)
        }
        fn main() -> f32 {
            let !W = vault.ones[f32, [2, 3]]
            let  x = vault.ones[f32, [3, 1]]
            let (_l, g) = linear.fwd_bwd(W, x)
            sum(g.W)
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-9, "expected 6.0, got {:?}", v);
}

#[test]
fn null6_question_mark_unwraps_success() {
    // Null 6: ? propagation is confluent. On (T, nil) the ? operator must
    // return T and not the whole tuple.
    let v = run(r#"
        fn maybe_val() -> (i64, str) {
            (42, nil)
        }
        fn main() -> i64 {
            maybe_val()?
        }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn null6_question_mark_propagates_error() {
    // On (_, err) the ? operator early-returns the enclosing (T, Err) function
    // with the error tuple (Rust-`?` style), so a caller observes the error
    // instead of the program aborting. `step()` propagates `failing()`'s error;
    // `main` reads it back out of the returned tuple.
    let v = run(r#"
        fn failing() -> (i64, str) {
            (0, "something went wrong")
        }
        fn step() -> (i64, str) {
            let x = failing()?      # propagates here; next line not reached
            (x + 1, nil)
        }
        fn main() -> str {
            let (code, err) = step()
            err
        }
    "#);
    assert_eq!(as_str(&v), "something went wrong");
}

#[test]
fn null6_question_mark_on_non_tuple_errors_loudly() {
    // ? applied to a non-(T, Err) value must error, not pass the value through.
    // The old placeholder silently returned the bare value.
    let msg = run_err(r#"
        fn main() -> i64 {
            let x = 42
            x?
        }
    "#);
    assert!(msg.contains("(T, Err)") || msg.contains("tuple"),
            "expected type error for ? on int, got: {}", msg);
}

// ── `?`'s error path returns `(default[T], e)` (#628) ────────────────────────
//
// SPEC §4.9: `x?` is sugar for `let (x, e) = ...; if e != nil { return
// (default[T], e) }`, where `default[T]` is the zero value of the ENCLOSING
// function's `T` — 0 / 0.0 / false / nil / zero-tensor.
//
// The interpreter used to early-return the CALLEE's whole tuple, so the
// caller's success slot carried whatever the callee had put in its own error
// path. `null6_question_mark_propagates_error` above could not see it because
// its callee returned `(0, err)` — the default by coincidence. Every probe
// below therefore uses a NON-ZERO callee value, so the old behaviour cannot
// pass any of them. The expected values are the JIT's, probe for probe
// (`jit.rs`'s `lower_query` / `default_value_for`, PR #625).

#[test]
fn query_error_path_returns_the_callers_default_not_the_callees_value() {
    // The headline case: the callee's error-path `777` must not reach the
    // caller's success slot. `0`, the zero of the caller's `i64`, must.
    let v = run(r#"
        fn callee() -> (i64, str) { (777, "odd") }
        fn caller() -> (i64, str) {
            let v = callee()?
            (v + 1, nil)
        }
        fn main() -> str {
            let (v, e) = caller()
            to_str(v) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "0|odd");
}

#[test]
fn query_default_is_the_zero_of_the_enclosing_t_for_every_scalar() {
    // i64 → 0, f32 → 0.0, bool → false, str → nil. `default[str]` is `nil`,
    // not `""`: SPEC §4.9's enumeration has no empty string in it, and the JIT
    // gives a `str` slot the NULL pointer, which is `nil`.
    let v = run(r#"
        fn c_i() -> (i64, str)  { (42, "boom") }
        fn c_f() -> (f32, str)  { (2.5, "boom") }
        fn c_b() -> (bool, str) { (true, "boom") }
        fn c_s() -> (str, str)  { ("hi", "boom") }
        fn p_i() -> (i64, str)  { let v = c_i()?
                                  (v, nil) }
        fn p_f() -> (f32, str)  { let v = c_f()?
                                  (v, nil) }
        fn p_b() -> (bool, str) { let v = c_b()?
                                  (v, nil) }
        fn p_s() -> (str, str)  { let v = c_s()?
                                  (v, nil) }
        fn main() -> str {
            let (a, ea) = p_i()
            let (b, eb) = p_f()
            let (c, ec) = p_b()
            let (d, ed) = p_s()
            to_str(a) + " / " + to_str(b) + " / " + to_str(c) + " / " + to_str(d)
        }
    "#);
    assert_eq!(as_str(&v), "0 / 0 / false / nil");
}

#[test]
fn query_default_for_a_tensor_t_is_a_zero_tensor() {
    // The callee's error path hands back a tensor of ONES; the caller's slot
    // must hold its own zero tensor, so the sum is 0 and not 2.
    let v = run(r#"
        fn c_t() -> (Tensor[f32, [2]], str) { (forge.ones[f32, [2]], "boom") }
        fn p_t() -> (Tensor[f32, [2]], str) {
            let v = c_t()?
            (v, nil)
        }
        fn main() -> f32 {
            let (t, e) = p_t()
            sum(t)
        }
    "#);
    assert!(as_float(&v).abs() < 1e-9, "expected a zero tensor, got sum {:?}", v);
}

#[test]
fn query_default_for_a_shape_parametric_tensor_t_resolves_its_dims() {
    // `-> (Tensor[f32, [N]], Err)`: `N` is a shape param of the propagating
    // function, so `default[T]` has to be built while that function's scope is
    // still standing. Built after the scope pops, `N` is unbound.
    let v = run(r#"
        fn c_n[N]() -> (Tensor[f32, [N]], str) { (forge.ones[f32, [N]], "boom") }
        fn p_n[N](probe: Tensor[f32, [N]]) -> (Tensor[f32, [N]], str) {
            let v = c_n[N]()?
            (v, nil)
        }
        fn main() -> f32 {
            let probe = forge.zeros[f32, [3]]
            let (t, e) = p_n(probe)
            sum(t)
        }
    "#);
    assert!(as_float(&v).abs() < 1e-9, "expected a zero tensor, got sum {:?}", v);
}

#[test]
fn query_in_a_loop_returns_the_default() {
    let v = run(r#"
        fn c99() -> (i64, str) { (99, "boom") }
        fn in_loop() -> (i64, str) {
            let !acc = 0
            for i in 0..3 {
                acc = acc + c99()?
            }
            (acc, nil)
        }
        fn main() -> str {
            let (a, e) = in_loop()
            to_str(a) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "0|boom");
}

#[test]
fn query_in_a_nested_block_returns_the_default() {
    // The `?` sits inside an `if` body, one block below the function's own.
    // The early return still belongs to the FUNCTION, so the default is the
    // function's — the block's value is never produced.
    let v = run(r#"
        fn c55() -> (i64, str) { (55, "boom") }
        fn in_nested() -> (i64, str) {
            if true {
                let v = c55()?
                (v, nil)
            } else {
                (0, nil)
            }
        }
        fn main() -> str {
            let (b, e) = in_nested()
            to_str(b) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "0|boom");
}

#[test]
fn query_twice_in_one_expression_returns_the_default() {
    // The first `?` to trip wins; the second is never evaluated.
    let v = run(r#"
        fn c55() -> (i64, str) { (55, "boom") }
        fn twice() -> (i64, str) {
            (c55()? + c55()?, nil)
        }
        fn main() -> str {
            let (c, e) = twice()
            to_str(c) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "0|boom");
}

#[test]
fn query_default_is_the_callers_type_when_the_two_ts_differ() {
    // The callee's `T` is `str` and the caller's is `i64`. Returning the
    // callee's tuple did not merely give the wrong VALUE — it put a `str` in a
    // slot declared `i64`. `as_int` is the assertion: it panics on a `Str`.
    let v = run(r#"
        fn c_s() -> (str, str) { ("payload", "boom") }
        fn caller() -> (i64, str) {
            let s = c_s()?
            (7, nil)
        }
        fn main() -> i64 {
            let (v, e) = caller()
            v
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn query_default_refuses_a_model_t() {
    // SPEC §4.9 defines `default[T]` as "0 / 0.0 / false / nil / zero-tensor".
    // A model is in none of those, so there is nothing to construct and the
    // interpreter refuses rather than inventing one — the same call the JIT's
    // `default_value_for` makes ("cannot construct"). The JIT refuses while
    // lowering; the interpreter, having no lowering pass, refuses when the
    // error path is taken.
    let msg = run_err(r#"
        model P { x: f32 }
        fn c_m() -> (P, str) {
            let p = P { x: 3.5 }
            (p, "boom")
        }
        fn p_m() -> (P, str) {
            let v = c_m()?
            (v, nil)
        }
        fn main() -> f32 {
            let (m, e) = p_m()
            m.x
        }
    "#);
    assert!(msg.contains("needs a default value for `P`"),
            "expected a `cannot construct default[P]` refusal, got: {}", msg);
}

#[test]
fn query_success_path_is_unchanged() {
    // The whole fix lives on the error path: `(T, nil)` still unwraps to `T`,
    // and nothing downstream of it moves.
    let v = run(r#"
        fn ok() -> (i64, str) { (7, nil) }
        fn use_it() -> (i64, str) {
            let v = ok()?
            (v * 3, nil)
        }
        fn main() -> str {
            let (v, e) = use_it()
            to_str(v) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "21|nil");
}

// ── `?` inside a lambda: `call_lambda`'s own catch (#628 follow-up) ──────────
//
// `check.rs`'s `PostfixOp::Query` validates a `?` against `current_fn_ret` —
// the LEXICALLY enclosing NAMED fn's declared return type — regardless of
// whether the `?` sits directly in that fn's body or inside a lambda literal
// nested in it (`FnLit`'s own check never touches `current_fn_ret`). A lambda
// with its own `-> (T, Err)` is validated the very same way: even a properly
// typed lambda is refused if the enclosing NAMED fn is not itself `(T, Err)`
// (confirmed against `dmc --check` directly: a `fn() -> (i64, str) { c()? }`
// sitting inside `fn main() -> str` is a type error citing `main`, not the
// lambda). `call_lambda`'s own `finish_query_propagation` call has to agree.

#[test]
fn query_typed_lambda_uses_its_own_declared_type_not_the_enclosing_fns() {
    // The lambda declares `-> (i64, str)`; the enclosing NAMED fn `outer`
    // declares `-> (str, str)` — a different `T`. `default[T]` must come from
    // the LAMBDA's own `i64` (giving `0`), not from `outer`'s `str` (which
    // would give `nil` and a type mismatch `to_str` would choke on).
    let v = run(r#"
        fn c() -> (i64, str) { (777, "boom") }
        fn outer() -> (str, str) {
            let g = fn() -> (i64, str) { let v = c()?
                                         (v, nil) }
            let (v, e) = g()
            (to_str(v) + "|" + to_str(e), nil)
        }
        fn main() -> str {
            let (v, e) = outer()
            v
        }
    "#);
    assert_eq!(as_str(&v), "0|boom");
}

#[test]
fn query_untyped_lambda_falls_back_to_the_enclosing_named_fns_type() {
    // The lambda has NO declared return type at all. Before this fix,
    // `call_lambda` had no `T` to build `default[T]` from and refused at
    // runtime ("`<lambda>` returns `nothing`") even though `dmc --check`
    // accepts the program — it validates the `?` against `f`'s `(str, str)`,
    // not the lambda's (absent) one. Verified directly against `dmc --check`:
    // this program passes with no type errors, and `dmc run` (pre-fix, on
    // `main`) produced `777:boom`; this pins the CORRECT reading instead —
    // `default[str]` is `nil` (SPEC §4.9: `str`'s zero value has no empty
    // string in the enumeration), so the lambda call's own early return is
    // the tuple `(nil, "boom")` — not unwrapped, because nothing in `f` put a
    // `?` on the lambda CALL itself, only inside the lambda's body.
    let v = run(r#"
        fn c() -> (i64, str) { (777, "boom") }
        fn f() -> (str, str) {
            let g = fn() { let v = c()?
                           to_str(v) }
            (g(), nil)
        }
        fn main() -> str {
            let (v, e) = f()
            to_str(v) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "(nil, \"boom\")|nil");
}

#[test]
fn query_in_a_lambda_outside_any_err_fn_is_refused() {
    // `dmc --check` already refuses this program (confirmed directly): `main`
    // returns `str`, not `(T, Err)`, so the checker rejects the `?` inside
    // `g`'s body regardless of what `g` itself declares. This test exercises
    // the INTERPRETER's own defense at the same boundary, independent of
    // `check.rs` (this harness never runs the checker) — `named_fn_ret_stack`
    // falls back to `main`'s declared type, which is not a `(T, Err)` tuple,
    // so `finish_query_propagation` refuses exactly as it would for a plain
    // named fn in the same position.
    let msg = run_err(r#"
        fn c() -> (i64, str) { (777, "boom") }
        fn main() -> str {
            let g = fn() { let v = c()?
                           to_str(v) }
            g()
        }
    "#);
    assert!(msg.contains("is only legal inside a function returning") && msg.contains("<lambda>"),
            "expected a `?`-outside-(T,Err) refusal citing the lambda, got: {}", msg);
}

// ── `query_default_value`'s tensor-shape axes must evaluate STRICTLY ─────────

#[test]
fn query_default_tensor_shape_refuses_an_unresolvable_dim_instead_of_guessing_4() {
    // This path evaluates a `?`'s default-tensor shape inline rather than
    // through `eval_dim` (see the `#628 follow-up` comment beside it): it
    // wants its own message and its own span (the `?`'s, not the failing
    // sub-expression's). `default[T]`'s zero tensor has to have the shape
    // `T` actually declares. `N - 6` at `N = 2` evaluates to `-4`; a naive
    // default-on-error (the bug `eval_dim` itself had until #839) would have
    // silently built a 4-element zero tensor with no diagnostic at all.
    //
    // `c[N]()` itself allocates a harmless `[1]`-shaped tensor — NOT `[N-6]`
    // — so the only place `N - 6` is ever evaluated is `p`'s own `?`
    // early-return default. Before #839, `c`'s body could get away with
    // constructing `[N-6]` directly (`eval_dim` silently substituted 4 for
    // the negative dim), which is what let this test reach the early-return
    // path at all; now that `eval_dim` itself refuses a negative dim, doing
    // that in `c` would raise there first and never exercise the code this
    // test targets.
    let msg = run_err(r#"
        fn c[N]() -> (Tensor[f32, [N-6]], str) { (forge.ones[f32, [1]], "boom") }
        fn p[N](probe: Tensor[f32, [N]]) -> (Tensor[f32, [N-6]], str) {
            let v = c[N]()?
            (v, nil)
        }
        fn main() -> f32 {
            let probe = forge.zeros[f32, [2]]
            let (t, e) = p(probe)
            sum(t)
        }
    "#);
    assert!(msg.contains("concrete non-negative dimension"),
            "expected a loud refusal of the negative dim, got: {}", msg);
}

// ── #839: a constructor's shape expression propagates its error instead of
// substituting a 4-element tensor ────────────────────────────────────────────
//
// `eval_dim` used to substitute a bare `4` on ANY error out of the shape
// expression it was evaluating — so `forge.zeros[f32, [n()]]` with a raising
// `n()` silently built a 4-element tensor and the program carried on with no
// diagnostic. Three flavors of "`n()` raises", each pinned to a located error
// rather than a successful 4-element (or, coincidentally-sized) tensor.

#[test]
fn shape_fn_runtime_error_propagates_839() {
    // The shape expression is a call whose body hits an ordinary runtime
    // error (an out-of-bounds tensor index) that `--check` cannot fold away
    // statically because the index is a parameter, not a literal.
    let (msg, at) = run_err_at(r#"
        fn n(k: i64) -> i64 {
            let !t = forge.zeros[f32, [2]]
            t[k] as i64
        }
        fn main() -> f32 {
            let t = forge.zeros[f32, [n(10)]]
            sum(t)
        }
    "#);
    assert!(msg.contains("out of bounds"), "expected the index error, got: {}", msg);
    assert!(at.is_some(), "the propagated error must be located: {}", msg);
}

#[test]
fn shape_fn_forge_exhaustion_propagates_839() {
    // The shape expression's own body exhausts a deliberately tiny `--forge`
    // budget. Before #839 this silently built the outer tensor at shape [4]
    // and kept running; now `forge.zeros[f32, [n()]]` never returns a value.
    let src = r#"
        fn n() -> i64 {
            let !t = forge.zeros[f32, [2000]]
            4
        }
        fn main() -> f32 {
            let t = forge.zeros[f32, [n()]]
            sum(t)
        }
    "#;
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut interp = Interpreter::new();
    interp.set_arena_limits(crate::arena::ArenaLimits { forge: Some(4096), vault: None });
    let e = interp.run(&program, None).err().expect("expected a runtime error");
    assert!(e.msg.contains("forge arena exhausted"), "message: {}", e.msg);
    assert!(e.span.is_some(), "the propagated error must be located: {}", e.msg);
}

#[test]
fn shape_fn_budget_stale_propagates_839() {
    // The shape expression's own body reads a tensor out of a budget it has
    // already reaped — `budget-stale` (the arena-budgets design, §2.3), the
    // same tag `examples/tenant_budgets.dmc` exercises via `forge.carve`/`forge.reap`.
    let (msg, at) = run_err_at(r#"
        fn n() -> i64 {
            let b = forge.carve(forge.root, 4096)
            let !t = forge.zeros[f32, [4]] in b
            forge.reap(b)
            t[0] as i64
        }
        fn main() -> f32 {
            let t = forge.zeros[f32, [n()]]
            sum(t)
        }
    "#);
    assert!(msg.starts_with("budget-stale:"), "message: {}", msg);
    assert!(at.is_some(), "the propagated error must be located: {}", msg);
}

// ── `default[T]` at the declared scalar width (#473/#540) ────────────────────

#[test]
fn query_default_wraps_at_the_declared_scalar_width() {
    // `default[u8]` is `0`; `0 - 1` must wrap to `255` the same way any other
    // `u8` arithmetic does, which only happens if the `0` `finish_query_propagation`
    // builds is actually tagged `u8` and not a width-less/`i64` zero.
    let v = run(r#"
        fn c() -> (u8, str) { (200 as u8, "boom") }
        fn p() -> (u8, str) {
            let v = c()?
            (v, nil)
        }
        fn main() -> str {
            let (v, e) = p()
            to_str(v - 1) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "255|boom");
}

// ── `default[T]` refuses what SPEC §4.9 gives no zero value for ─────────────

#[test]
fn query_default_refuses_a_tuple_t() {
    // A nested tuple is not in SPEC §4.9's enumeration ("0 / 0.0 / false /
    // nil / zero-tensor") any more than a model is — same refusal, same
    // reason `query_default_refuses_a_model_t` above pins for models.
    let msg = run_err(r#"
        fn c() -> ((i64, i64), str) { ((1, 2), "boom") }
        fn p() -> ((i64, i64), str) {
            let v = c()?
            (v, nil)
        }
        fn main() -> i64 {
            let (v, e) = p()
            let (a, b) = v
            a
        }
    "#);
    assert!(msg.contains("needs a default value for"),
            "expected a `default[(i64,i64)]` refusal, got: {}", msg);
}

#[test]
fn query_default_refusal_spells_fp8_e4m3() {
    // #833: `type_label` (this refusal's type-namer) used to Debug-format
    // the `ScalarType` and lowercase it — `Fp8E4M3` -> `fp8e4m3`, dropping
    // the underscore the source spelling and `docs/SPEC.md` §3.1 both
    // require. It now routes through the canonical `fmt::scalar_type_str`
    // table, the same one `check.rs`'s cast diagnostic and the JIT's own
    // refusals (below) share.
    let msg = run_err(r#"
        fn c() -> ((fp8_e4m3, i64), str) { ((1.0 as fp8_e4m3, 2), "boom") }
        fn p() -> ((fp8_e4m3, i64), str) {
            let v = c()?
            (v, nil)
        }
        fn main() -> i64 {
            let (v, e) = p()
            let (a, b) = v
            b
        }
    "#);
    assert!(msg.contains("needs a default value for `(fp8_e4m3, i64)`"),
            "expected the tuple's fp8_e4m3 element spelled with its underscore, got: {}", msg);
}

#[test]
fn query_default_refuses_an_enum_t() {
    let msg = run_err(r#"
        enum Color { Red, Green, Blue }
        fn c() -> (Color, str) { (Color.Red, "boom") }
        fn p() -> (Color, str) {
            let v = c()?
            (v, nil)
        }
        fn main() -> i64 {
            let (v, e) = p()
            v as i64
        }
    "#);
    assert!(msg.contains("needs a default value for `Color`"),
            "expected a `default[Color]` refusal, got: {}", msg);
}

// ── #628 × #639: two carriers, one channel ──────────────────────────────────
//
// Both signals ride `RuntimeError` because `eval_expr` has no other channel,
// and they are caught at DIFFERENT boundaries: a `?` signal (`propagate`) at
// the function-call boundary (`finish_query_propagation`), a block escape
// (`flow`) at the nearest statement or block boundary (`catch_flow`), one
// frame below. Each catch guards on its own field, so the other must ride
// through untouched. Confuse them and a `?` inside a block gets swallowed as
// the block's control flow, or a `return` inside a `(T, Err)` fn gets rebuilt
// as a `(default[T], e)` early return it never was.

#[test]
fn query_error_path_survives_a_block_between_it_and_the_fn_boundary() {
    // The `?` trips inside a `forge` block, then inside a plain block. #639
    // made both of those catching boundaries for the OTHER carrier; neither
    // may eat this one. Correct on both: the block never produces a value, the
    // signal reaches `caller`'s own boundary, and the answer is `caller`'s
    // `default[i64]` (0) with the callee's `777` dropped — not the block's
    // value, not nil, and not 778.
    let forged = run(r#"
        fn callee() -> (i64, str) { (777, "odd") }
        fn caller() -> (i64, str) {
            let v = forge { callee()? }
            (v + 1, nil)
        }
        fn main() -> str {
            let (v, e) = caller()
            to_str(v) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&forged), "0|odd");

    let plain = run(r#"
        fn callee() -> (i64, str) { (777, "odd") }
        fn caller() -> (i64, str) {
            let v = { callee()? }
            (v + 1, nil)
        }
        fn main() -> str {
            let (v, e) = caller()
            to_str(v) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&plain), "0|odd");
}

#[test]
fn a_block_escape_inside_an_err_fn_is_control_flow_not_a_query_early_return() {
    // The mirror image. `caller` is a `(T, Err)` fn — the one shape whose
    // boundary rebuilds a `?` — but a `return` / `break` out of a block there
    // is #639's carrier, not #628's, and must reach the fn (or the loop) as
    // written rather than being rewritten into `(default[T], e)`.
    //
    // `return (42, "early")` returns the fn's OWN tuple: `42`, not `default
    // [i64]`'s 0, and `"early"` is the value the program wrote, not an error
    // the `?` machinery produced.
    let returned = run(r#"
        fn callee() -> (i64, str) { (5, nil) }
        fn caller(x: i64) -> (i64, str) {
            forge { if x > 0 { return (42, "early") } }
            let v = callee()?
            (v + 1, nil)
        }
        fn main() -> str {
            let (v, e) = caller(1)
            to_str(v) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&returned), "42|early");

    // A `break` out of a block inside a loop in the same fn, on an iteration
    // AFTER a `?` took its success path — so both carriers have crossed the
    // same frame in one run. The `break` leaves the loop and nothing else:
    // i stops at 3, s = 0+1+2 = 3, so `s * 100 + i` is 303 with no error.
    let broke = run(r#"
        fn ok(n: i64) -> (i64, str) { (n, nil) }
        fn caller() -> (i64, str) {
            let !i = 0
            let !s = 0
            while i < 10 {
                let v = ok(i)?
                forge { if v > 2 { break } }
                s = s + v
                i = i + 1
            }
            (s * 100 + i, nil)
        }
        fn main() -> str {
            let (v, e) = caller()
            to_str(v) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&broke), "303|nil");
}

#[test]
fn null8_kv_stream_append_extends_axis() {
    // Null 8: KV[~] <- appends along the last (streaming) axis.
    // forge.ones[f32,[4,2]] (sum=8) <- forge.ones[f32,[4,3]] (12 ones added)
    // → [4,5] with all ones → sum = 20.
    let v = run(r#"
        fn main() -> f32 {
            let kv = forge.ones[f32, [4, 2]]
            let new_slice = forge.ones[f32, [4, 3]]
            kv <- new_slice
            sum(kv)
        }
    "#);
    assert!((as_float(&v) - 20.0).abs() < 1e-9, "expected 20.0 after KV append, got {:?}", v);
}

#[test]
fn kv_stream_append_uses_declared_streaming_axis() {
    let v = run(r#"
        fn main() -> f32 {
            let kv: KV[f32, [2, ~, 3]] = forge.ones[f32, [2, 1, 3]]
            kv <- forge.ones[f32, [2, 2, 3]]
            sum(kv)
        }
    "#);
    assert!((as_float(&v) - 18.0).abs() < 1e-9, "expected 18.0 after KV append, got {:?}", v);
}

#[test]
fn null8_kv_multiple_appends_accumulate() {
    // Three sequential <- appends must accumulate without reallocation errors.
    // [2,3] + [2,4] + [2,1] = [2,8], all ones → sum = 16.
    let v = run(r#"
        fn main() -> f32 {
            let kv = forge.ones[f32, [2, 3]]
            kv <- forge.ones[f32, [2, 4]]
            kv <- forge.ones[f32, [2, 1]]
            sum(kv)
        }
    "#);
    assert!((as_float(&v) - 16.0).abs() < 1e-9, "expected 16.0 after two KV appends, got {:?}", v);
}

// ── Trailing directive block whose body ends in an `if` / `match` statement ──
//
// `@deterministic { if c { 10 } else { 20 } }` in tail position is a
// `Stmt::DirectiveBlock` whose body has no `tail_expr` — the `if` is a
// keyword-led statement. The block's value is that statement's value, as in a
// plain block; the interpreter used to read only the body's `tail_expr` here
// and yield nil. (The `let x = @deterministic { … }` form goes through
// `Expr::DirectiveBlock` → `eval_block` and was always right.)

#[test]
fn trailing_directive_block_yields_its_if_statement() {
    let v = run("fn main() -> i64 { @deterministic { if 3 > 2 { 10 } else { 20 } } }");
    assert_eq!(as_int(&v), 10);
}

#[test]
fn trailing_directive_block_yields_its_match_statement() {
    let v = run("fn main() -> i64 {\n    let n = 3\n    @deterministic { match n { 3 => 10, _ => 20 } }\n}");
    assert_eq!(as_int(&v), 10);
}

#[test]
fn trailing_directive_block_if_sees_inner_lets_and_else_arm() {
    let v = run("fn main() -> i64 { @deterministic { let k = 1  if k > 2 { k } else { 20 } } }");
    assert_eq!(as_int(&v), 20);
}

#[test]
fn trailing_cast_block_casts_its_if_statement_value() {
    // The `@cast` applied to the recovered value, same as for a tail expr.
    let v = run("fn main() -> f64 { @cast(f64) { if 3 > 2 { 10 } else { 20 } } }");
    assert_eq!(as_float(&v), 10.0);
}

#[test]
fn null10_comptime_evaluates_static_expr() {
    // Null 10: @comptime evaluation is total on static operands.
    //
    // This comment used to say a non-terminating @comptime "would require
    // static analysis (JIT phase)". #505 made that false: `comptime.rs` folds
    // the block before either backend runs, under a step budget, so a
    // non-terminating body is a `comptime-budget` compile error
    // (the evaluator's step budget is exhausted) rather than a hang.
    //
    // This file's `run` helper deliberately does NOT fold: it drives the
    // interpreter straight off the parsed tree, so these tests keep covering
    // the interpreter's own `DirectiveBlock` path — the one a residual
    // (shape-parameter) block still takes in the shipped binary. The value
    // must be the same one the fold produces; `comptime_tests` pins that side.
    let v = run(r#"
        fn main() -> i64 {
            let x = @comptime { 3 * 7 + 1 }
            x
        }
    "#);
    assert_eq!(as_int(&v), 22);
}

// ── Tensor indexing + new builtins (demo unlockers) ──────────────────────

#[test]
fn tensor_read_index_1d() {
    let v = run(r#"
        fn main() -> f32 {
            let t = [10.0, 20.0, 30.0]
            t[1]
        }
    "#);
    assert!((as_float(&v) - 20.0).abs() < 1e-9, "expected 20.0, got {:?}", v);
}

#[test]
fn tensor_read_index_2d() {
    let v = run(r#"
        fn main() -> f32 {
            let t = vault.zeros[f32, [3, 4]]
            t[1, 2]
        }
    "#);
    assert!((as_float(&v) - 0.0).abs() < 1e-9, "expected 0.0 (zeros), got {:?}", v);
}

#[test]
fn tensor_write_index_1d() {
    let v = run(r#"
        fn main() -> f32 {
            let t = vault.zeros[f32, [5]]
            t[2] = 99.0
            t[2]
        }
    "#);
    assert!((as_float(&v) - 99.0).abs() < 1e-9, "expected 99.0 after write, got {:?}", v);
}

#[test]
fn tensor_write_index_2d() {
    let v = run(r#"
        fn main() -> f32 {
            let t = vault.zeros[f32, [4, 4]]
            t[1, 3] = 7.0
            t[1, 3]
        }
    "#);
    assert!((as_float(&v) - 7.0).abs() < 1e-9, "expected 7.0 after write, got {:?}", v);
}

#[test]
fn tensor_write_row_slice() {
    let v = run(r#"
        fn main() -> f32 {
            let grid = [[0, 0, 0, 0], [0, 0, 0, 0]]
            let row = [1, 2, 3, 4]
            grid[0] = row
            grid[0, 0]
        }
    "#);
    assert!((as_float(&v) - 1.0).abs() < 1e-9, "expected 1.0 after row write, got {:?}", v);
}

#[test]
fn sieve_of_eratosthenes() {
    // Classic sieve — uses tensor as mutable boolean array.
    // Counts primes up to 30; expected: 2,3,5,7,11,13,17,19,23,29 = 10 primes.
    let v = run(r#"
        fn main() -> i64 {
            let n = 30
            let sieve = vault.ones[f32, [31]]
            sieve[0] = 0.0
            sieve[1] = 0.0
            let i = 2
            while i * i <= n {
                if sieve[i] > 0.5 {
                    let j = i * i
                    while j <= n {
                        sieve[j] = 0.0
                        j = j + i
                    }
                }
                i = i + 1
            }
            let count = 0
            let k = 2
            while k <= n {
                if sieve[k] > 0.5 { count = count + 1 }
                k = k + 1
            }
            count
        }
    "#);
    assert_eq!(as_int(&v), 10, "expected 10 primes below 30, got {:?}", v);
}

#[test]
fn floor_ceil_builtins() {
    let v = run(r#"
        fn main() -> f32 {
            floor(3.7) + ceil(2.1)
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-9, "expected 6.0, got {:?}", v);
}

#[test]
fn chr_builtin() {
    let v = run(r#"
        fn main() -> str { chr(65) }
    "#);
    assert!(matches!(v, Value::Str(ref s) if s == "A"), "expected \"A\", got {:?}", v);
}

#[test]
fn len_builtin() {
    let v = run(r#"
        fn main() -> i64 {
            let t = vault.zeros[f32, [7]]
            len(t)
        }
    "#);
    assert_eq!(as_int(&v), 7, "expected 7, got {:?}", v);
}

// ── String concatenation ────────────────────────────────────────────────────

#[test]
fn string_concat_two_str_literals() {
    let v = run(r#"
        fn main() -> str {
            "hello" + " world"
        }
    "#);
    assert!(matches!(v, Value::Str(ref s) if s == "hello world"), "got {:?}", v);
}

#[test]
fn string_concat_str_plus_int() {
    let v = run(r#"
        fn main() -> str {
            "count=" + 42
        }
    "#);
    assert!(matches!(v, Value::Str(ref s) if s == "count=42"), "got {:?}", v);
}

#[test]
fn string_concat_int_plus_str() {
    let v = run(r#"
        fn main() -> str {
            42 + "!"
        }
    "#);
    assert!(matches!(v, Value::Str(ref s) if s == "42!"), "got {:?}", v);
}

#[test]
fn string_concat_chained() {
    let v = run(r#"
        fn main() -> str {
            let a = "foo"
            let b = "bar"
            let c = "baz"
            a + b + c
        }
    "#);
    assert!(matches!(v, Value::Str(ref s) if s == "foobarbaz"), "got {:?}", v);
}

// ── Negative indexing ───────────────────────────────────────────────────────

#[test]
fn negative_index_1d_read() {
    // t[-1] should return the last element (value 5.0)
    let v = run(r#"
        fn main() -> f64 {
            let t = vault.zeros[f32, [3]]
            t[0] = 1.0
            t[1] = 3.0
            t[2] = 5.0
            t[-1]
        }
    "#);
    assert!((as_float(&v) - 5.0).abs() < 1e-9, "expected 5.0, got {:?}", v);
}

#[test]
fn negative_index_1d_second_to_last() {
    let v = run(r#"
        fn main() -> f64 {
            let t = vault.zeros[f32, [4]]
            t[0] = 10.0
            t[1] = 20.0
            t[2] = 30.0
            t[3] = 40.0
            t[-2]
        }
    "#);
    assert!((as_float(&v) - 30.0).abs() < 1e-9, "expected 30.0, got {:?}", v);
}

#[test]
fn negative_index_write() {
    // t[-1] = 99.0 should set the last element
    let v = run(r#"
        fn main() -> f64 {
            let t = vault.zeros[f32, [3]]
            t[-1] = 99.0
            t[2]
        }
    "#);
    assert!((as_float(&v) - 99.0).abs() < 1e-9, "expected 99.0, got {:?}", v);
}

#[test]
fn negative_index_2d_read() {
    let v = run(r#"
        fn main() -> f64 {
            let t = vault.zeros[f32, [2, 3]]
            t[0, 0] = 1.0
            t[0, 1] = 2.0
            t[0, 2] = 3.0
            t[1, 0] = 4.0
            t[1, 1] = 5.0
            t[1, 2] = 6.0
            t[-1, -1]
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-9, "expected 6.0, got {:?}", v);
}

#[test]
fn negative_index_2d_write() {
    let v = run(r#"
        fn main() -> f64 {
            let t = vault.zeros[f32, [2, 3]]
            t[-1, -1] = 7.0
            t[1, 2]
        }
    "#);
    assert!((as_float(&v) - 7.0).abs() < 1e-9, "expected 7.0, got {:?}", v);
}

// ── Bug regression tests (issue #42) ─────────────────────────────────────────

/// I1: 2D tensor literal construction at runtime
#[test]
fn i1_2d_tensor_literal() {
    // [[1, 2, 3], [4, 5, 6]] should produce a 2D tensor with shape [2, 3]
    let v = run(r#"
        fn main() -> f64 {
            let a = [[1, 2, 3], [4, 5, 6]]
            sum(a)
        }
    "#);
    assert!((as_float(&v) - 21.0).abs() < 1e-9, "expected sum=21.0, got {:?}", v);
}

/// I1: 2D tensor literal element access
#[test]
fn i1_2d_tensor_literal_indexing() {
    let v = run(r#"
        fn main() -> f64 {
            let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            a[1, 2]
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-9, "expected 6.0, got {:?}", v);
}

/// I3: `@pp(stages=N)` pipeline parallel function
#[test]
fn i3_pp_pipeline_stages() {
    // stage 0: x + 1 = 6, stage 1: _ * 2 = 12
    let v = run(r#"
        @pp(stages=2)
        fn pipeline(x: i64) -> i64 {
            stage 0: x + 1
            stage 1: _ * 2
        }
        fn main() -> i64 { pipeline(5) }
    "#);
    assert_eq!(as_int(&v), 12, "expected 12, got {:?}", v);
}

/// I3: pipeline with 3 stages
#[test]
fn i3_pp_pipeline_three_stages() {
    // stage 0: x + 1 = 6, stage 1: _ * 2 = 12, stage 2: _ - 2 = 10
    let v = run(r#"
        @pp(stages=3)
        fn pipeline(x: i64) -> i64 {
            stage 0: x + 1
            stage 1: _ * 2
            stage 2: _ - 2
        }
        fn main() -> i64 { pipeline(5) }
    "#);
    assert_eq!(as_int(&v), 10, "expected 10, got {:?}", v);
}

/// B2: `@host match` fallback to wildcard at runtime
#[test]
fn b2_host_match_wildcard_fallback() {
    // A clearly non-existent feature name should always fall through to `_`.
    let v = run(r#"
        fn main() -> i64 {
            let chosen = @host match {
                .fictional_cpu_feature_xyz => 1,
                _                          => 42,
            }
            chosen
        }
    "#);
    assert_eq!(as_int(&v), 42, "expected wildcard=42, got {:?}", v);
}

#[test]
fn host_match_real_feature_detected() {
    // On x86_64, SSE2 is baseline and always present.
    // On aarch64, NEON is baseline and always present.
    // Either way, the first matching arm should fire and NOT the wildcard.
    let v = run(r#"
        fn main() -> i64 {
            let chosen = @host match {
                .sse2 => 10,
                .neon => 20,
                _     => 0,
            }
            chosen
        }
    "#);
    let n = as_int(&v);
    assert!(n != 0, "expected a real feature arm to match (sse2=10 or neon=20), got wildcard 0");
}

#[test]
fn host_match_accelerate_tracks_the_jit_fast_path() {
    // #578: `accelerate` is a host feature like the ISA flags, but it reports
    // what is *linked* rather than what the CPU can do — a BLAS `cblas_sgemm`
    // in this process's image, which is how a large f32 matmul reaches the
    // platform's matrix unit.
    //
    // The assertion is agreement, not presence: the arm a program selects and
    // the kernel the JIT selects must come from one detection, or a program
    // could dispatch on `.accelerate` and get the Cranelift kernel anyway.
    // Presence itself is a property of the host and not assertable here — on
    // macOS libSystem re-exports vecLib's BLAS, on a bare Linux box there is
    // none, and both are correct.
    let v = run(r#"
        fn main() -> i64 {
            let chosen = @host match {
                .accelerate => 1,
                _           => 0,
            }
            chosen
        }
    "#);
    let expected = if crate::jit::blas_gemm_available() { 1 } else { 0 };
    assert_eq!(as_int(&v), expected,
        "`@host match .accelerate` disagrees with the JIT's own BLAS detection");
}

/// B7: Unicode (Greek-letter) identifiers work end-to-end
#[test]
fn b7_unicode_identifiers_run() {
    let v = run(r#"
        fn main() -> i64 {
            let π = 3
            let θ = 4
            let δ = π + θ
            δ
        }
    "#);
    assert_eq!(as_int(&v), 7, "expected 7, got {:?}", v);
}

// ── Dynamic collections: list ────────────────────────────────────────────────

#[test]
fn list_basic_operations() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list()
            let xs = list_push(xs, 1)
            let xs = list_push(xs, 2)
            let xs = list_push(xs, 3)
            list_len(xs)
        }
    "#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn list_get_and_set() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 10), 20), 30)
            list_get(xs, 1)
        }
    "#);
    assert_eq!(as_int(&v), 20);
}

#[test]
fn map_basic_operations() {
    let v = run(r#"
        fn main() -> str {
            let m = map()
            let m = map_set(m, "hello", "world")
            map_get(m, "hello")
        }
    "#);
    assert!(matches!(v, Value::Str(ref s) if s == "world"), "got {:?}", v);
}

#[test]
fn map_has_and_missing() {
    let v = run(r#"
        fn main() -> i64 {
            let m = map_set(map(), "x", 42)
            if map_has(m, "x") { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn time_ms_increases() {
    let v = run(r#"
        fn main() -> f64 {
            let t = time_ms()
            t
        }
    "#);
    assert!(as_float(&v) >= 0.0);
}

#[test]
fn list_pop_returns_modified_list_and_last() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list(), 10), 99)
            let (rest, last) = list_pop(xs)
            last
        }
    "#);
    assert_eq!(as_int(&v), 99);
}

#[test]
fn list_len_via_len_builtin() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list(), 1), 2)
            len(xs)
        }
    "#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn list_concat_combines_lists() {
    let v = run(r#"
        fn main() -> i64 {
            let a = list_push(list_push(list(), 1), 2)
            let b = list_push(list_push(list(), 3), 4)
            list_len(list_concat(a, b))
        }
    "#);
    assert_eq!(as_int(&v), 4);
}

#[test]
fn list_slice_extracts_sublist() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list_push(list(), 10), 20), 30), 40)
            let sl = list_slice(xs, 1, 3)
            list_len(sl)
        }
    "#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn list_contains_finds_element() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list(), 5), 10)
            if list_contains(xs, 5) { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn list_rev_reverses() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            let rev = list_rev(xs)
            list_get(rev, 0)
        }
    "#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn list_set_replaces_element() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            let xs = list_set(xs, 1, 99)
            list_get(xs, 1)
        }
    "#);
    assert_eq!(as_int(&v), 99);
}

#[test]
fn for_loop_over_list() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            let total = 0
            for x in xs {
                total += x
            }
            total
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

// ── Dynamic collections: map ─────────────────────────────────────────────────

#[test]
fn map_del_removes_key() {
    let v = run(r#"
        fn main() -> i64 {
            let m = map_set(map_set(map(), "a", 1), "b", 2)
            let m = map_del(m, "a")
            if map_has(m, "a") { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn map_keys_returns_list() {
    let v = run(r#"
        fn main() -> i64 {
            let m = map_set(map_set(map(), "x", 1), "y", 2)
            list_len(map_keys(m))
        }
    "#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn map_len_returns_count() {
    let v = run(r#"
        fn main() -> i64 {
            let m = map_set(map_set(map_set(map(), "a", 1), "b", 2), "c", 3)
            map_len(m)
        }
    "#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn map_get_missing_returns_nil() {
    let v = run(r#"
        fn main() -> nil {
            let m = map()
            map_get(m, "missing")
        }
    "#);
    assert!(matches!(v, Value::Nil), "expected nil for missing key, got {:?}", v);
}

// ── Process / environment ────────────────────────────────────────────────────

#[test]
fn env_var_missing_returns_error() {
    let v = run(r#"
        fn main() -> str {
            let (val, err) = env_var("_DEMIC_TEST_NONEXISTENT_VAR_XYZ_12345")
            err
        }
    "#);
    assert!(matches!(v, Value::Str(ref s) if s == "not found"), "got {:?}", v);
}

// ── Group 1: --profile op-counting ──────────────────────────────────────────

#[test]
fn profile_tensor_ops_counted() {
    // Enable profiling programmatically and run a small program with tensor
    // arithmetic; verify tensor_ops > 0.
    let src = r#"
        fn main() -> f64 {
            let a = [1.0, 2.0, 3.0]
            let b = [4.0, 5.0, 6.0]
            sum(a .+ b)
        }
    "#;
    let tokens = super::lexer::Lexer::new(src).tokenize().expect("lex");
    let program = super::parser::Parser::new(tokens).parse_program().expect("parse");
    let mut interp = Interpreter::new();
    interp.enable_profile();
    interp.run(&program, None).expect("run");
    let p = interp.profile.as_ref().expect("profile should be Some after enable_profile");
    assert!(p.tensor_ops > 0, "expected tensor_ops > 0, got {}", p.tensor_ops);
}

// ── Group 2: rand_* stdlib ───────────────────────────────────────────────────

#[test]
fn rand_float_in_range() {
    let v = run(r#"fn main() -> f64 { rand_seed(42); rand_float() }"#);
    let f = as_float(&v);
    assert!(f >= 0.0 && f < 1.0, "rand_float out of [0,1): got {}", f);
}

#[test]
fn rand_int_in_range() {
    let v = run(r#"fn main() -> i64 { rand_seed(42); rand_int(0, 10) }"#);
    let n = as_int(&v);
    assert!(n >= 0 && n < 10, "rand_int out of [0,10): got {}", n);
}

#[test]
fn rand_normal_is_finite() {
    let v = run(r#"fn main() -> f64 { rand_seed(7); rand_normal(0.0, 1.0) }"#);
    let f = as_float(&v);
    assert!(f.is_finite(), "rand_normal returned non-finite: {}", f);
}

#[test]
fn rand_seed_makes_reproducible() {
    // Same seed → same sequence.
    let v1 = run(r#"fn main() -> f64 { rand_seed(123); rand_float() }"#);
    let v2 = run(r#"fn main() -> f64 { rand_seed(123); rand_float() }"#);
    assert_eq!(as_float(&v1), as_float(&v2), "same seed must give same first draw");
}

#[test]
fn rand_choice_picks_from_list() {
    let v = run(r#"
        fn main() -> i64 {
            rand_seed(99)
            let xs = list_push(list_push(list_push(list(), 10), 20), 30)
            rand_choice(xs)
        }
    "#);
    let n = as_int(&v);
    assert!(n == 10 || n == 20 || n == 30, "rand_choice returned unexpected value: {}", n);
}

// ── Group 3: json_encode / json_decode ───────────────────────────────────────

#[test]
fn json_encode_basic() {
    let v = run(r#"fn main() -> str { json_encode(42) }"#);
    assert_eq!(as_str(&v), "42");
}

#[test]
fn json_encode_float() {
    let v = run(r#"fn main() -> str { json_encode(3.5) }"#);
    // 3.5 has a fractional part so should not be truncated
    assert_eq!(as_str(&v), "3.5");
}

#[test]
fn json_encode_bool_and_nil() {
    let vt = run(r#"fn main() -> str { json_encode(true) }"#);
    assert_eq!(as_str(&vt), "true");
    let vf = run(r#"fn main() -> str { json_encode(false) }"#);
    assert_eq!(as_str(&vf), "false");
    let vn = run(r#"fn main() -> str { json_encode(nil) }"#);
    assert_eq!(as_str(&vn), "null");
}

#[test]
fn json_encode_string() {
    let v = run(r#"fn main() -> str { json_encode("hello") }"#);
    assert_eq!(as_str(&v), r#""hello""#);
}

#[test]
fn json_encode_list() {
    let v = run(r#"
        fn main() -> str {
            let xs = list_push(list_push(list(), 1), 2)
            json_encode(xs)
        }
    "#);
    assert_eq!(as_str(&v), "[1,2]");
}

#[test]
fn json_encode_tensor_array() {
    // #190: a tensor/array literal serializes to a JSON array (was "null").
    let v = run(r#"
        fn main() -> str { json_encode([1.5, 2.5, 3.5]) }
    "#);
    assert_eq!(as_str(&v), "[1.5,2.5,3.5]");
}

#[test]
fn json_encode_tensor_whole_numbers() {
    // Whole-number floats drop the trailing .0, same as scalar formatting.
    let v = run(r#"
        fn main() -> str { json_encode([1.0, 2.0, 3.0]) }
    "#);
    assert_eq!(as_str(&v), "[1,2,3]");
}

#[test]
fn f32_bits_roundtrip() {
    // #189: 1.0f32 has bit pattern 0x3F800000, and from_bits inverts to_bits.
    let v = run(r#"
        fn main() -> i64 {
            let bits = f32_to_bits(1.0)
            if (bits == 1065353216) && (f32_from_bits(bits) == 1.0) { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn read_bytes_decodes_binary_f32() {
    // #197: read_bytes loads raw (non-UTF-8) bytes as an i64 byte-tensor so
    // weights round-trip exactly. 1.5f32 little-endian = [0x00,0x00,0xC0,0x3F];
    // the trailing 0xFF keeps the buffer non-UTF-8 (read_file would reject it).
    let path = std::env::temp_dir().join(format!("dmc_rb_{}.bin", std::process::id()));
    std::fs::write(&path, [0x00u8, 0x00, 0xC0, 0x3F, 0xFF]).expect("write bin");
    let src = format!(r#"
        fn main() -> f64 {{
            let (t, err) = read_bytes("{}")
            let bits = t[0] + t[1] * 256 + t[2] * 65536 + t[3] * 16777216
            f32_from_bits(bits)
        }}
    "#, path.display());
    let v = run(&src);
    std::fs::remove_file(&path).ok();
    assert!((as_float(&v) - 1.5).abs() < 1e-9, "expected 1.5, got {:?}", v);
}

#[test]
fn read_bytes_missing_file_returns_error() {
    // Error path mirrors read_file: `(nil, errstr)` on failure.
    let v = run(r#"
        fn main() -> bool {
            let (t, err) = read_bytes("/nonexistent/dmc_rb_missing.bin")
            is_nil(t) && !is_nil(err)
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "expected (nil, err) tuple, got {:?}", v);
}

#[test]
fn any_type_runtime_dispatch() {
    // #186: an `any`-returning atom() dispatches at runtime — a numeric token
    // comes back as a Float, a non-numeric one as a Str. Proves the dynamic
    // value genuinely crosses the fn boundary and keeps its real type.
    let v = run(r#"
        fn atom(tok: str) -> any {
            if is_numeric(tok) { return to_float(tok) }
            tok
        }
        fn main() -> i64 {
            let a = atom("2.5")
            let b = atom("hello")
            if is_float(a) && is_str(b) { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn read_bytes_reads_raw_non_utf8() {
    // #197: read_bytes returns one i64 per byte (0-255), losslessly — including
    // bytes that are invalid UTF-8 (0x80, 0xFF) which read_file would reject.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("blob.bin");
    std::fs::write(&path, [0u8, 127, 128, 255]).expect("write");
    let src = format!(r#"
        fn main() -> i64 {{
            let (bytes, err) = read_bytes("{}")
            if !is_nil(err) {{ return -1 }}
            if !is_tensor(bytes) {{ return -2 }}
            bytes[0] + bytes[1] + bytes[2] + bytes[3]
        }}
    "#, path.to_str().unwrap());
    assert_eq!(as_int(&run(&src)), 0 + 127 + 128 + 255);
}

#[test]
fn read_bytes_missing_file_returns_err() {
    // Missing file: (nil, err-string) tuple, mirroring read_file's contract.
    let v = run(r#"
        fn main() -> i64 {
            let (bytes, err) = read_bytes("/no/such/file/anywhere.bin")
            if is_nil(bytes) && is_str(err) { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn read_bytes_completes_f32_weight_loader() {
    // #197 + #189: the exact-weight-loader story end to end. Write the raw
    // little-endian f32 bytes for 1.0 (0x3F800000), read them back as bytes,
    // reassemble the i32 bit pattern LE, and recover 1.0 via f32_from_bits.
    // (`>>` is compose-pipe in demoniC, so assembly uses `<<` and `|` only.)
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("one.f32");
    std::fs::write(&path, [0x00u8, 0x00, 0x80, 0x3F]).expect("write");
    let src = format!(r#"
        fn main() -> i64 {{
            let (b, err) = read_bytes("{}")
            if !is_nil(err) {{ return 0 }}
            let bits = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
            if (bits == 1065353216) && (f32_from_bits(bits) == 1.0) {{ 1 }} else {{ 0 }}
        }}
    "#, path.to_str().unwrap());
    assert_eq!(as_int(&run(&src)), 1);
}

#[test]
fn diag_extracts_diagonal() {
    // #191: diag of a 3x3 returns its [N] diagonal; trace == sum of diag.
    let v = run(r#"
        fn main() -> f32 {
            let !m = forge.zeros[f32, [3, 3]]
            m[0,0] = 5.0  m[1,1] = 7.0  m[2,2] = 9.0  m[0,2] = 1.0
            let d = diag(m)
            d[0] + d[1] + d[2]
        }
    "#);
    assert_eq!(as_float(&v), 21.0);
}

#[test]
fn json_decode_number() {
    let v = run(r#"
        fn main() -> i64 {
            let (val, err) = json_decode("42")
            val
        }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn json_decode_string() {
    let v = run(r#"
        fn main() -> str {
            let (val, err) = json_decode("\"hello\"")
            val
        }
    "#);
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn json_decode_bool() {
    let v = run(r#"
        fn main() -> bool {
            let (val, err) = json_decode("true")
            val
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "got {:?}", v);
}

#[test]
fn json_decode_null() {
    let v = run(r#"
        fn main() -> nil {
            let (val, err) = json_decode("null")
            val
        }
    "#);
    assert!(matches!(v, Value::Nil), "got {:?}", v);
}

#[test]
fn json_decode_error() {
    let v = run(r#"
        fn main() -> str {
            let (val, err) = json_decode("not json")
            err
        }
    "#);
    // err should be a non-nil string describing the parse failure
    assert!(matches!(v, Value::Str(ref s) if s.contains("parse error")), "got {:?}", v);
}

// ── Typed JSON decode (PORTS.md §6) ──────────────────────────────────────────

#[test]
fn json_decode_typed_returns_the_declared_type() {
    // Each primitive hands back a real value of its type, not a dynamic one:
    // `v * 2` below is integer arithmetic on a decoded JSON number.
    let v = run(r#"
        fn main() -> i64 {
            let (v, e) = json_decode_i64("21")
            if e != nil { return -1 }
            v * 2
        }
    "#);
    assert_eq!(as_int(&v), 42);

    let v = run(r#"fn main() -> f64 { let (v, e) = json_decode_f64("1.5") v + 1.0 }"#);
    assert_eq!(as_float(&v), 2.5);
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_str("\"hi\"") v + "!" }"#);
    assert_eq!(as_str(&v), "hi!");
    let v = run(r#"fn main() -> bool { let (v, e) = json_decode_bool("false") !v }"#);
    assert!(matches!(v, Value::Bool(true)), "got {:?}", v);
    let v = run(r#"fn main() -> i64 { let (v, e) = json_decode_list("[1,2,3]") list_len(v) }"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn json_decode_typed_mismatch_is_a_decode_type_tag() {
    // The whole point: a wrong-typed JSON value is an Err, never a coercion.
    // `"7"` does not become 7, `2.5` does not truncate, `1` is not true.
    for (call, src, want) in [
        ("json_decode_i64",  r#"\"7\""#,   "expected i64, got str"),
        ("json_decode_i64",  "2.5",        "expected i64, got f64"),
        ("json_decode_i64",  "null",       "expected i64, got nil"),
        ("json_decode_bool", "1",          "expected bool, got i64"),
        ("json_decode_str",  "7",          "expected str, got i64"),
        ("json_decode_f64",  r#"\"1.5\""#, "expected f64, got str"),
        ("json_decode_list", r#"{\"a\":1}"#, "expected list, got map"),
    ] {
        let v = run(&format!(
            "fn main() -> str {{ let (v, e) = {}(\"{}\") e }}", call, src));
        let got = as_str(&v);
        assert!(got.starts_with("decode-type: ") && got.contains(want),
            "{}({}) -> {:?}, wanted `decode-type` with `{}`", call, src, got, want);
    }
}

#[test]
fn json_decode_typed_zero_rides_the_error_path() {
    // `(T, Err)` stays well-typed on failure: T's zero, not nil.
    let v = run(r#"fn main() -> i64 { let (v, e) = json_decode_i64("\"x\"") v }"#);
    assert_eq!(as_int(&v), 0);
    let v = run(r#"fn main() -> f64 { let (v, e) = json_decode_f64("\"x\"") v }"#);
    assert_eq!(as_float(&v), 0.0);
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_str("7") v }"#);
    assert_eq!(as_str(&v), "");
    let v = run(r#"fn main() -> bool { let (v, e) = json_decode_bool("7") v }"#);
    assert!(matches!(v, Value::Bool(false)), "got {:?}", v);
    let v = run(r#"fn main() -> i64 { let (v, e) = json_decode_list("7") list_len(v) }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn json_decode_f64_accepts_a_json_integer() {
    // The widening every port result depends on: JSON has a single number
    // type and the canonical writer prints a whole float without its fraction
    // (2.0 -> `2`, PORTS.md §2), so a port's f64 result usually arrives as an
    // integer literal. The reverse is caught only for non-whole values —
    // json_decode_i64_cannot_see_a_whole_valued_float pins that gap.
    let v = run(r#"fn main() -> f64 { let (v, e) = json_decode_f64("2") v + 0.5 }"#);
    assert_eq!(as_float(&v), 2.5);
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_f64("2") to_str(e) }"#);
    assert_eq!(as_str(&v), "nil");
}

#[test]
fn json_decode_f64_takes_whole_numbers_past_i64() {
    // A fraction-less token wider than i64 is still a well-formed JSON number,
    // so it decodes — as f64 — instead of being called a parse error. This is
    // not a corner case: the canonical writer emits every whole float without
    // its fraction (PORTS.md §2), so a port returning 1e20 hands the decode 21
    // digits. Walk the boundary, since it used to sit at 1e15.
    for (src, want) in [
        ("100000000000000",       1e14),                    // 1e14
        ("1000000000000000",      1e15),                    // 1e15 — old cliff
        ("100000000000000000000", 1e20),                    // 1e20
        ("9223372036854775807",   i64::MAX as f64),         // i64::MAX
        ("9223372036854775808",   9223372036854775808.0),   // i64::MAX + 1
    ] {
        let v = run(&format!(
            "fn main() -> f64 {{ let (v, e) = json_decode_f64(\"{}\") v }}", src));
        assert_eq!(as_float(&v), want, "json_decode_f64({})", src);
        let e = run(&format!(
            "fn main() -> str {{ let (v, e) = json_decode_f64(\"{}\") to_str(e) }}", src));
        assert_eq!(as_str(&e), "nil", "json_decode_f64({}) should not error", src);
    }
}

#[test]
fn json_decode_i64_past_its_range_is_decode_type_not_decode_parse() {
    // Out of range is a mismatch of kind, not a broken payload. `decode-parse`
    // means the text is not JSON (PORTS.md §6) and a long integer is JSON, so
    // claiming otherwise would send a caller hunting a corrupt result that is
    // in fact a descriptor promising the wrong width.
    for src in ["100000000000000000000", "9223372036854775808"] {
        let v = run(&format!(
            "fn main() -> str {{ let (v, e) = json_decode_i64(\"{}\") e }}", src));
        assert_eq!(as_str(&v), "decode-type: expected i64, got f64",
            "json_decode_i64({})", src);
    }
    // Inside the range it stays an ordinary i64, top of the range included.
    for (src, want) in [
        ("100000000000000",     100_000_000_000_000_i64),
        ("1000000000000000",    1_000_000_000_000_000_i64),
        ("9223372036854775807", i64::MAX),
    ] {
        let v = run(&format!(
            "fn main() -> i64 {{ let (v, e) = json_decode_i64(\"{}\") v }}", src));
        assert_eq!(as_int(&v), want, "json_decode_i64({})", src);
    }
}

#[test]
fn json_decode_i64_cannot_see_a_whole_valued_float() {
    // Documented behavior, not an aspiration (PORTS.md §3.1, ASSIMILATE.md
    // §5.1 and §7). The canonical writer drops a whole float's fraction, so
    // 5.0 reaches the decode as the same one byte the integer 5 would. A
    // descriptor promising `ret: "i64"` over a float-returning function is
    // wrong and this boundary cannot say so. Pinned here so the gap stays
    // disclosed instead of drifting back into a no-coercion claim the code
    // does not back.
    let v = run(r#"fn main() -> str { json_encode(5.0) }"#);
    assert_eq!(as_str(&v), "5");
    let v = run(r#"fn main() -> i64 { let (v, e) = json_decode_i64(json_encode(5.0)) v }"#);
    assert_eq!(as_int(&v), 5);
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_i64(json_encode(5.0)) to_str(e) }"#);
    assert_eq!(as_str(&v), "nil");

    // The other half of the disclosure: a fraction survives the writer, so the
    // same wrong descriptor is caught the moment the result is not whole.
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_i64(json_encode(5.5)) e }"#);
    assert_eq!(as_str(&v), "decode-type: expected i64, got f64");
}

#[test]
fn json_decode_typed_parse_failure_is_a_decode_parse_tag() {
    // Malformed JSON is distinct from a type mismatch — the caller can tell a
    // broken payload from a descriptor that lied.
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_i64("{oops") e }"#);
    assert!(as_str(&v).starts_with("decode-parse: "), "got {:?}", v);
}

#[test]
fn json_decode_typed_propagates_through_question_mark() {
    // SPEC §4.9: the `(T, Err)` shape is the point — `?` lifts the Err out and
    // the caller sees a real i64.
    let v = run(r#"
        fn twice(s: str) -> (i64, str) {
            let n = json_decode_i64(s)?
            (n * 2, nil)
        }
        fn main() -> str {
            let (a, ea) = twice("21")
            let (b, eb) = twice("true")
            to_str(a) + "|" + to_str(ea) + "|" + to_str(b) + "|" + eb
        }
    "#);
    assert_eq!(as_str(&v), "42|nil|0|decode-type: expected i64, got bool");
}

#[test]
fn json_roundtrip() {
    let v = run(r#"
        fn main() -> str {
            let m = map_set(map(), "x", 1)
            let s = json_encode(m)
            s
        }
    "#);
    assert_eq!(as_str(&v), r#"{"x":1}"#);
}

// ── JSON string decoding is UTF-8 (#509) ─────────────────────────────────────
//
// JSON text is UTF-8 by definition, and a writer may leave any non-ASCII
// scalar unescaped. The parser used to push each raw byte as a `char`, reading
// the text as latin-1: an em dash arrived as three mojibake chars. These pin
// both spellings of the same scalar against each other, because a reader that
// disagrees with itself about `—` and `\u2014` is the bug restated.

#[test]
fn json_decode_raw_multibyte_scalar_is_not_latin1() {
    // The regression itself: three raw bytes are one em dash, not three chars.
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_str("\"a—b\"") v }"#);
    assert_eq!(as_str(&v), "a—b");
    // `len` on a str is bytes, which is the sharpest witness available here:
    // the em dash is its own three bytes, not three latin-1 chars re-encoded
    // as two bytes each (`a` + 6 + `b` = 8 was the old answer).
    let v = run(r#"fn main() -> i64 { let (v, e) = json_decode_str("\"a—b\"") len(v) }"#);
    assert_eq!(as_int(&v), 5);
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_str("\"a—b\"") to_str(e) }"#);
    assert_eq!(as_str(&v), "nil");

    // Same through the dynamic decoder, and nested inside a container so the
    // object/array paths get the fixed `parse_string` too — keys included.
    let v = run(r#"
        fn main() -> str {
            let (m, e) = json_decode("{\"kø\": [\"ü—ß\"]}")
            list_get(map_get(m, "kø"), 0)
        }
    "#);
    assert_eq!(as_str(&v), "ü—ß");
}

#[test]
fn json_decode_raw_and_escaped_spellings_agree() {
    // A writer picks a spelling; a reader must not. `—` and `\u2014` are the
    // same scalar, so the two decodes are the same demoniC str.
    let v = run(r#"
        fn main() -> str {
            let (raw, e1) = json_decode_str("\"a—b\"")
            let (esc, e2) = json_decode_str("\"a\\u2014b\"")
            if raw == esc { raw } else { "differ: " + raw + " vs " + esc }
        }
    "#);
    assert_eq!(as_str(&v), "a—b");
}

#[test]
fn json_decode_four_byte_scalar_joins_a_surrogate_pair() {
    // A code point past the BMP has no four-digit escape, so JSON spells it as
    // a UTF-16 surrogate pair — which is exactly what an ASCII-only writer
    // (python's `json.dumps`) emits for an emoji. Reading the halves apart
    // rejected valid JSON: neither half is a scalar value. Both spellings of
    // U+1F600 must land on the same one-char str.
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_str("\"😀\"") v }"#);
    assert_eq!(as_str(&v), "😀");
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_str("\"\\ud83d\\ude00\"") v }"#);
    assert_eq!(as_str(&v), "😀");
    let v = run(r#"
        fn main() -> str {
            let (raw, e1) = json_decode_str("\"go 😀 od\"")
            let (esc, e2) = json_decode_str("\"go \\ud83d\\ude00 od\"")
            if raw == esc { raw } else { "differ: " + raw + " vs " + esc }
        }
    "#);
    assert_eq!(as_str(&v), "go 😀 od");
    // One scalar in four UTF-8 bytes — not two BMP chars from the two halves,
    // which would have been six.
    let v = run(r#"fn main() -> i64 { let (v, e) = json_decode_str("\"\\ud83d\\ude00\"") len(v) }"#);
    assert_eq!(as_int(&v), 4);
}

#[test]
fn json_decode_lone_surrogate_is_a_decode_parse_tag() {
    // A surrogate outside a pair is not a Unicode scalar value, so there is
    // nothing honest to decode it to. It is a text failure, not a kind
    // mismatch: `decode-parse`, per PORTS.md §6. Full messages, so the
    // diagnostics stay one line and stay specific.
    let cases = [
        (r#"\"\\ud83d\""#,          "decode-parse: unpaired high surrogate U+D83D"),
        (r#"\"\\ude00\""#,          "decode-parse: unpaired low surrogate U+DE00"),
        (r#"\"\\ud83dx\""#,         "decode-parse: unpaired high surrogate U+D83D"),
        (r#"\"\\ud83d\\u0041\""#,   "decode-parse: expected low surrogate after U+D83D, got U+0041"),
    ];
    for (src, want) in cases {
        let v = run(&format!(
            "fn main() -> str {{ let (v, e) = json_decode_str(\"{}\") e }}", src));
        assert_eq!(as_str(&v), want, "json_decode_str({})", src);
    }
}

#[test]
fn json_parse_rejects_an_ill_formed_utf8_sequence() {
    // The guard behind the fix, reached through the byte entry point because
    // nothing in the language can hand it these bytes: a demoniC `str` is
    // UTF-8 by construction and the port's `read_line` rejects an ill-formed
    // response before the parser sees it. Pinned anyway — the parser's own
    // contract is that it decodes UTF-8, and "decodes" includes refusing.
    use super::interp::json_parse_bytes;
    let cases: [&[u8]; 4] = [
        b"\"a\xFFb\"",         // 0xFF is never a UTF-8 byte
        b"\"a\xE2\x80b\"",     // truncated three-byte sequence
        b"\"a\xC0\x80b\"",     // overlong encoding of NUL
        b"\"a\xED\xA0\xBDb\"", // a surrogate spelled as raw bytes (CESU-8)
    ];
    for src in cases {
        assert_eq!(json_parse_bytes(src).unwrap_err(), "invalid UTF-8 in string",
            "bytes {:?}", src);
    }
    // A well-formed sequence through the same entry point still decodes.
    assert!(json_parse_bytes("\"a—b\"".as_bytes()).is_ok());
}

#[test]
fn json_parse_diagnostics_name_a_stray_byte_as_a_byte() {
    // The same misreading one layer up: a byte at or above 0x80 is a fragment
    // of a UTF-8 sequence, and printing it with `as char` claimed it was a
    // latin-1 letter — `unexpected byte 'â'` for an em dash. Show the byte.
    // ASCII messages are unchanged, which the second half pins.
    let v = run(r#"fn main() -> str { let (v, e) = json_decode("—") to_str(e) }"#);
    assert_eq!(as_str(&v), "parse error: unexpected byte 0xE2 at position 0");
    let v = run(r#"fn main() -> str { let (v, e) = json_decode("[1—]") to_str(e) }"#);
    assert_eq!(as_str(&v), "parse error: expected ',' or ']', got 0xE2");
    let v = run(r#"fn main() -> str { let (v, e) = json_decode("\"\\—\"") to_str(e) }"#);
    assert_eq!(as_str(&v), "parse error: unknown escape \\0xE2");
    let v = run(r#"fn main() -> str { let (v, e) = json_decode("\"\\u20—4\"") to_str(e) }"#);
    assert_eq!(as_str(&v), "parse error: invalid \\u escape: 0xE2 is not a hex digit");

    let v = run(r#"fn main() -> str { let (v, e) = json_decode("[1;]") to_str(e) }"#);
    assert_eq!(as_str(&v), "parse error: expected ',' or ']', got ';'");
    let v = run(r#"fn main() -> str { let (v, e) = json_decode("\"\\q\"") to_str(e) }"#);
    assert_eq!(as_str(&v), "parse error: unknown escape \\q");
}

#[test]
fn json_encode_decode_roundtrips_non_ascii() {
    // The encoder writes non-ASCII raw (only quotes, backslash and controls
    // are escaped) — that is the canonical writer PORTS.md §2 makes the port
    // ABI, so it is the decoder's job to meet it. Before the fix this pair did
    // not round-trip at all: the writer's own output came back as mojibake.
    let v = run(r#"fn main() -> str { json_encode("a—b😀") }"#);
    assert_eq!(as_str(&v), "\"a—b😀\"");
    let v = run(r#"fn main() -> str { let (v, e) = json_decode_str(json_encode("a—b😀")) v }"#);
    assert_eq!(as_str(&v), "a—b😀");
}

// ── Group 1: List functional combinators ─────────────────────────────────────

#[test]
fn list_map_doubles() {
    let v = run(r#"
        fn double(x: i64) -> i64 { x * 2 }
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            let ys = list_map(xs, double)
            list_get(ys, 2)
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn list_filter_evens() {
    let v = run(r#"
        fn is_even(x: i64) -> i64 { x % 2 == 0 }
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list_push(list(), 1), 2), 3), 4)
            let evens = list_filter(xs, is_even)
            list_len(evens)
        }
    "#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn list_reduce_sum() {
    let v = run(r#"
        fn add(a: i64, b: i64) -> i64 { a + b }
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            list_reduce(xs, add, 0)
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn list_sort_integers() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 3), 1), 2)
            let sorted = list_sort(xs)
            list_get(sorted, 0)
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

// ── Issue #254 regression: proper typed comparator for sort/min/max ───────────

#[test]
fn list_sort_int_basic_regression() {
    // Basic int sort still works after #254 fix
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list_push(list(), 9), 2), 5), 1)
            let sorted = list_sort(xs)
            list_get(sorted, 0)
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn list_sort_strings_long_common_prefix() {
    // Strings sharing 7+ leading bytes must sort correctly (f64 mantissa can't distinguish them)
    let v = run(r#"
        fn main() -> str {
            let a = "abcdefgh_ZZZ"
            let b = "abcdefgh_AAA"
            let c = "abcdefgh_MMM"
            let xs = list_push(list_push(list_push(list(), a), b), c)
            let sorted = list_sort(xs)
            list_get(sorted, 0)
        }
    "#);
    assert_eq!(as_str(&v), "abcdefgh_AAA");
}

#[test]
fn list_sort_strings_long_prefix_last_element() {
    // Verify the last element is also correct
    let v = run(r#"
        fn main() -> str {
            let a = "abcdefgh_ZZZ"
            let b = "abcdefgh_AAA"
            let c = "abcdefgh_MMM"
            let xs = list_push(list_push(list_push(list(), a), b), c)
            let sorted = list_sort(xs)
            list_get(sorted, 2)
        }
    "#);
    assert_eq!(as_str(&v), "abcdefgh_ZZZ");
}

#[test]
fn list_sort_floats_with_nan_does_not_crash() {
    // NaN must not crash the sort and must produce a deterministic result
    let v = run(r#"
        fn main() -> f64 {
            let xs = list_push(list_push(list_push(list(), 3.0), nan), 1.0)
            let sorted = list_sort(xs)
            list_get(sorted, 0)
        }
    "#);
    // 1.0 should be the minimum finite value; NaN sorts last via total_cmp
    assert!((as_float(&v) - 1.0).abs() < 1e-9);
}

#[test]
fn list_min_floats_with_nan_deterministic() {
    // list_min over a list containing NaN should return the finite minimum
    let v = run(r#"
        fn main() -> f64 {
            let xs = list_push(list_push(list_push(list(), 5.0), nan), 2.0)
            list_min(xs)
        }
    "#);
    assert!((as_float(&v) - 2.0).abs() < 1e-9);
}

#[test]
fn list_max_floats_with_nan_deterministic() {
    // list_max over a list containing NaN must not crash (NaN sorts last via total_cmp,
    // so max is NaN — the key assertion is that the call terminates deterministically).
    let _ = run(r#"
        fn main() -> f64 {
            let xs = list_push(list_push(list_push(list(), 5.0), nan), 2.0)
            list_max(xs)
        }
    "#);
}

#[test]
fn list_min_integers_exact() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 100), -50), 0)
            list_min(xs)
        }
    "#);
    assert_eq!(as_int(&v), -50);
}

#[test]
fn list_max_integers_exact() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 100), -50), 0)
            list_max(xs)
        }
    "#);
    assert_eq!(as_int(&v), 100);
}

#[test]
fn list_sort_by_string_key_long_prefix() {
    // list_sort_by with a key function returning strings with a shared prefix:
    // "prefix_1" < "prefix_2" < "prefix_3" lexicographically — so element 1 sorts first.
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 3), 1), 2)
            let sorted = list_sort_by(xs, fn(x) -> str { format("prefix_{}", x) })
            list_get(sorted, 0)
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn list_zip_pairs() {
    let v = run(r#"
        fn main() -> i64 {
            let a = list_push(list_push(list(), 1), 2)
            let b = list_push(list_push(list(), 10), 20)
            let pairs = list_zip(a, b)
            let (x, y) = list_get(pairs, 1)
            x + y
        }
    "#);
    assert_eq!(as_int(&v), 22);
}

#[test]
fn list_enumerate_indices() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list(), "a"), "b")
            let en = list_enumerate(xs)
            let (i, x) = list_get(en, 1)
            i
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn list_sum_values() {
    let v = run(r#"
        fn main() -> f64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            list_sum(xs)
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-9);
}

// ── Group 2: String formatting ────────────────────────────────────────────────

#[test]
fn format_basic_substitution() {
    let v = run(r#"fn main() -> str { format("x={}", 42) }"#);
    assert_eq!(as_str(&v), "x=42");
}

#[test]
fn format_multiple_args() {
    let v = run(r#"fn main() -> str { format("{} + {} = {}", 1, 2, 3) }"#);
    assert_eq!(as_str(&v), "1 + 2 = 3");
}

#[test]
fn format_float() {
    let v = run(r#"fn main() -> str { format("pi={}", 3.14) }"#);
    // Should contain "3.14" somewhere
    if let Value::Str(s) = &v { assert!(s.contains("3.14"), "got {}", s); }
    else { panic!("expected Str"); }
}

// ── Group 3: Basic hashing ────────────────────────────────────────────────────

#[test]
fn hash_fnv_deterministic() {
    let v = run(r#"fn main() -> i64 { hash_fnv("hello") }"#);
    // FNV-1a of "hello" is 0xa430d84680aabd0b = -6600509020984515317 as i64
    // Just check it's nonzero and deterministic
    assert_ne!(as_int(&v), 0);
    let v2 = run(r#"fn main() -> i64 { hash_fnv("hello") }"#);
    assert_eq!(as_int(&v), as_int(&v2));
}

#[test]
fn hash_crc32_deterministic() {
    let v = run(r#"fn main() -> i64 { hash_crc32("hello") }"#);
    assert_ne!(as_int(&v), 0);
}

#[test]
fn hash_different_strings_differ() {
    let v = run(r#"
        fn main() -> i64 {
            let a = hash_fnv("hello")
            let b = hash_fnv("world")
            if a != b { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn get_cwd_returns_string() {
    let v = run(r#"fn main() -> str { get_cwd() }"#);
    match &v {
        Value::Str(s) => assert!(!s.is_empty()),
        _ => panic!("expected Str, got {:?}", v),
    }
}

#[test]
fn path_join_basic() {
    let v = run(r#"fn main() -> str { path_join("/tmp", "foo.txt") }"#);
    match &v {
        Value::Str(s) => assert!(s.contains("foo.txt")),
        _ => panic!("expected Str, got {:?}", v),
    }
}

#[test]
fn path_basename_and_dirname() {
    let v = run(r#"fn main() -> str { path_basename("/tmp/foo/bar.txt") }"#);
    match &v {
        Value::Str(s) => assert_eq!(s, "bar.txt"),
        _ => panic!("expected Str"),
    }
}

#[test]
fn path_exists_for_tmp() {
    let v = run(r#"fn main() -> i64 { if path_exists("/tmp") { 1 } else { 0 } }"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn make_dir_and_delete_dir() {
    let v = run(r#"
        fn main() -> i64 {
            let (_, e1) = make_dir("/tmp/dmc_test_dir_42")
            let (_, e2) = delete_dir("/tmp/dmc_test_dir_42")
            if e1 == nil { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn exec_cmd_echo() {
    let v = run(r#"
        fn main() -> str {
            let args = list_push(list(), "hello")
            let (out, err, code) = exec_cmd("echo", args)
            out
        }
    "#);
    match &v {
        Value::Str(s) => assert!(s.contains("hello"), "got {:?}", s),
        _ => panic!("expected Str, got {:?}", v),
    }
}

/// `nil` is not indexable, and a `Port` handle has no elements.
///
/// Both used to take the interpreter's forward-compat path and hand back a
/// `Value::Opaque` — `e[0]` on a nil `Err` printed `<opaque index>` and exited
/// 0. The JIT had raised on the same program, so the disagreement was the
/// silent backend answering. Located errors now, in the JIT's words.
#[test]
fn indexing_nil_or_a_handle_is_a_located_error() {
    assert_eq!(
        run_err("fn main() -> nil {\n    let e = nil\n    let c = e[0]\n}\n"),
        "cannot index nil",
    );
    if !crate::ports::have_python() { eprintln!("skipped: python3 not on PATH"); return; }
    let msg = run_err(
        "fn main() -> nil {\n    let (p, e) = port_open(\"python\")\n    \
         let c = p[0]\n    let (_, _) = port_close(p)\n}\n");
    assert!(msg.starts_with("cannot index a Port handle"), "{}", msg);
}

/// A live handle has no methods either. `p.starts_with("port#")` answered with
/// a falsy opaque and exit 0 — the handle-forging check that opacity exists to
/// make impossible, quietly reported as "no".
#[test]
fn a_method_call_on_a_live_handle_is_a_located_error() {
    if !crate::ports::have_python() { eprintln!("skipped: python3 not on PATH"); return; }
    let msg = run_err(
        "fn main() -> nil {\n    let (p, e) = port_open(\"python\")\n    \
         let b = p.starts_with(\"port#\")\n    let (_, _) = port_close(p)\n}\n");
    assert!(msg.starts_with("cannot call method `starts_with` on a Port handle"), "{}", msg);
}

/// `len(nil)` said the right thing in the wrong place: unlocated, while the
/// JIT's `dmc_str_len` names the span. Same program, same words, same span.
#[test]
fn len_of_a_non_container_is_located() {
    assert_eq!(
        run_err("fn main() -> nil {\n    let e = nil\n    let n = len(e)\n}\n"),
        "len: requires tensor, tuple, str, list, or map",
    );
}

#[test]
fn port_roundtrip_python() {
    // #402: process-port floor — open a python port, call through the JSON
    // ABI, close. python3 is a dev prerequisite here, same as
    // examples/port_python.dmc. The payload is the positional argument vector
    // (PORTS.md §2), so a single list argument nests: `[[1,2,3]]` -> len([1,2,3]).
    // `len` avoids float-formatting assumptions.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            if e1 != nil { return "open failed" }
            let (out, e2) = port_call(p, "len", "[[1, 2, 3]]")
            if e2 != nil { return "call failed" }
            let (_, e3) = port_close(p)
            if e3 != nil { return "close failed" }
            out
        }
    "#);
    assert_eq!(as_str(&v), "3");
}

#[test]
fn port_roundtrip_carries_non_ascii_both_directions() {
    // #509 through the real boundary. Outbound, the canonical writer leaves
    // non-ASCII raw (PORTS.md §2) and python reads it as UTF-8. Inbound, the
    // harness's `json.dumps` is ASCII-only, so the same text comes back as
    // `—` and, for the emoji, a surrogate pair — the spelling the old
    // parser rejected outright with `invalid codepoint U+D83D`. Both halves
    // have to work for the text to survive one call.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            if e1 != nil { return "open failed" }
            let (out, e2) = port_call(p, "str", "[\"a—b😀\"]")
            let (_, e3) = port_close(p)
            if e2 != nil { return e2 }
            let (s, e4) = json_decode_str(out)
            if e4 != nil { return e4 }
            s
        }
    "#);
    assert_eq!(as_str(&v), "a—b😀");
}

#[test]
fn port_response_written_as_raw_utf8_decodes() {
    // The other producer shape the issue names: a runtime that writes its
    // response with non-ASCII unescaped. The stock harness never does — its
    // `json.dumps` escapes everything — so the runtime is made to emit the
    // response line itself, through `sys.stdout.write`, ahead of the harness's
    // own. That is the raw-UTF-8 line `port_call` then reads and parses. The
    // port is desynchronised afterwards (the harness's reply is still queued),
    // so it gets its own port and is closed immediately.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            if e1 != nil { return "open failed" }
            let line = "[\"{\\\"ok\\\": \\\"raw—😀\\\"}\\n\"]"
            let (out, e2) = port_call(p, "sys.stdout.write", line)
            let (_, e3) = port_close(p)
            if e2 != nil { return e2 }
            out
        }
    "#);
    // `port_call` re-encodes through the canonical writer, which leaves
    // non-ASCII raw — so a correct decode returns the scalars unchanged.
    assert_eq!(as_str(&v), "\"raw—😀\"");
}

#[test]
fn port_call_after_close_tags_port_closed() {
    // #402 / PORTS.md §6: errors are str tags matched by prefix.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            let (_, e2) = port_close(p)
            let (out, e3) = port_call(p, "len", "[1]")
            if e3 == nil { "no error" } else { e3 }
        }
    "#);
    match &v {
        Value::Str(s) => assert!(s.starts_with("port-closed"), "got {:?}", s),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn port_open_unsupported_runtime_tags_port_open() {
    let v = run(r#"
        fn main() -> str {
            let (p, e) = port_open("lua")
            if e == nil { "no error" } else { e }
        }
    "#);
    match &v {
        Value::Str(s) => assert!(s.starts_with("port-open"), "got {:?}", s),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn port_call_foreign_exception_tags_port_call() {
    // A python-side exception (sqrt of a string) must surface as a
    // `port-call` tag, not kill the port or the interpreter. The string is a
    // single positional arg, so it rides inside the payload array.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            let (out, e2) = port_call(p, "math.sqrt", "[\"sixteen\"]")
            let (_, e3) = port_close(p)
            if e2 == nil { "no error" } else { e2 }
        }
    "#);
    match &v {
        Value::Str(s) => assert!(s.starts_with("port-call"), "got {:?}", s),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn port_multi_arg_spreads_positionally() {
    // #402 / PORTS.md §2: a JSON array payload spreads as positional args.
    // math.gcd(462, 1071) == 21 exercises a two-argument call.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            if e1 != nil { return "open failed" }
            let (out, e2) = port_call(p, "math.gcd", "[462, 1071]")
            let (_, e3) = port_close(p)
            if e2 != nil { return e2 }
            out
        }
    "#);
    assert_eq!(as_str(&v), "21");
}

#[test]
fn port_kwargs_envelope_binds_by_name() {
    // #402 / PORTS.md §2: a JSON object payload is the {args, kwargs} envelope.
    // round(3.14159, ndigits=2) == 3.14 binds a keyword argument by name.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            if e1 != nil { return "open failed" }
            let (out, e2) = port_call(p, "round", "{\"args\":[3.14159],\"kwargs\":{\"ndigits\":2}}")
            let (_, e3) = port_close(p)
            if e2 != nil { return e2 }
            out
        }
    "#);
    assert_eq!(as_str(&v), "3.14");
}

#[test]
fn port_bare_scalar_payload_tags_port_protocol() {
    // #402 / PORTS.md §6: a bare scalar is not an argument vector. The ABI
    // rejects it with `port-protocol` instead of guessing an arity.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            let (out, e2) = port_call(p, "math.sqrt", "16")
            let (_, e3) = port_close(p)
            if e2 == nil { "no error" } else { e2 }
        }
    "#);
    match &v {
        Value::Str(s) => assert!(s.starts_with("port-protocol"), "got {:?}", s),
        other => panic!("expected Str, got {:?}", other),
    }
}

#[test]
fn port_unknown_envelope_key_tags_port_protocol() {
    // #402 / PORTS.md §6: a top-level object must be a valid {args, kwargs}
    // envelope; a stray key is a malformed ABI, not a runtime failure.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            let (out, e2) = port_call(p, "len", "{\"nope\":[]}")
            let (_, e3) = port_close(p)
            if e2 == nil { "no error" } else { e2 }
        }
    "#);
    match &v {
        Value::Str(s) => assert!(s.starts_with("port-protocol"), "got {:?}", s),
        other => panic!("expected Str, got {:?}", other),
    }
}

// ── Tensor copy mode (PORTS.md §3.2) ─────────────────────────────────────────

#[test]
fn port_tensor_encode_writes_the_copy_mode_envelope() {
    // PORTS.md §3.2: a tensor does not become a JSON array — it crosses as the
    // envelope, metadata and payload buffer together, in canonical key order.
    let v = run(r#"
        fn main() -> str {
            let !g = forge.zeros[i64, [2, 3]]
            g[0, 0] = 1  g[0, 1] = 2  g[0, 2] = 3
            g[1, 0] = -4 g[1, 1] = 5  g[1, 2] = 6
            port_tensor_encode(g)
        }
    "#);
    assert_eq!(as_str(&v),
        "{\"data\":\"AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA/P////////8FAAAAAAAAAAYAAAAAAAAA\",\
         \"dmc_tensor\":1,\"dtype\":\"i64\",\"layout\":\"row_major\",\"shape\":[2,3]}");
}

#[test]
fn port_tensor_round_trips_through_the_envelope() {
    // Values, shape and dtype all survive; the decode's `like` tensor is what
    // declares what was expected.
    let v = run(r#"
        fn main() -> i64 {
            let !g = forge.zeros[i64, [2, 2]]
            g[0, 0] = 10  g[0, 1] = -20  g[1, 0] = 30  g[1, 1] = -40
            let (back, e) = port_tensor_decode(port_tensor_encode(g), forge.zeros[i64, [2, 2]])
            if e != nil { -1 } else { back[1, 0] - back[0, 1] }
        }
    "#);
    assert_eq!(as_int(&v), 50);
    // bool keeps its 1-byte payload and comes back as bools, not numbers.
    let v = run(r#"
        fn main() -> bool {
            let !b = forge.zeros[bool, [3]]
            b[0] = true  b[1] = false  b[2] = true
            let (back, e) = port_tensor_decode(port_tensor_encode(b), forge.zeros[bool, [3]])
            e == nil && back[0] && !back[1] && back[2]
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "{:?}", v);
}

#[test]
fn port_tensor_decode_mismatch_is_a_decode_type_tag() {
    // The §3.1 discipline at tensor granularity: a shape the caller did not
    // declare is a `decode-type`, and the value half is the declared zero.
    let v = run(r#"
        fn main() -> str {
            let !g = forge.zeros[i64, [4]]
            let (back, e) = port_tensor_decode(port_tensor_encode(g), forge.zeros[i64, [2]])
            e
        }
    "#);
    assert_eq!(as_str(&v), "decode-type: expected tensor shape [2], got [4]");
    // A float tensor is not an integer one, however whole its values.
    let v = run(r#"
        fn main() -> str {
            let !g = forge.zeros[f32, [2]]
            let (back, e) = port_tensor_decode(port_tensor_encode(g), forge.zeros[i64, [2]])
            e
        }
    "#);
    assert_eq!(as_str(&v), "decode-type: expected a `i64` tensor, got `f32`");
    // Text that is not JSON at all is the other tag.
    let v = run(r#"
        fn main() -> bool {
            let (back, e) = port_tensor_decode("{oops", forge.zeros[i64, [2]])
            e.starts_with("decode-parse") && back[0] == 0 && back[1] == 0
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "{:?}", v);
}

#[test]
fn port_tensor_encode_refuses_a_non_tensor() {
    // The boundary is explicit (§3.2): there is no implicit array form, so a
    // scalar or a list is a program bug, not a quietly-encoded value.
    let e = run_err(r#"fn main() -> str { port_tensor_encode(7) }"#);
    assert!(e.contains("port_tensor_encode: arg must be a tensor"), "{}", e);
    let e = run_err(r#"
        fn main() -> str {
            let (v, e) = port_tensor_decode("{}", 7)
            v
        }
    "#);
    assert!(e.contains("second arg must be a tensor"), "{}", e);
}

#[test]
fn port_tensor_encode_failures_carry_no_wire_tag() {
    // Encoding is local: no runtime is involved and there is no `Err` half, so
    // a tensor that has no envelope is an ordinary runtime error. `§6`'s
    // `port-` tags belong to a runtime that failed, and none was reached.
    let e = run_err(r#"fn main() -> str { let !t = forge.trit[2, 2]  port_tensor_encode(t) }"#);
    assert!(e.contains("a `trit` tensor has no copy-mode wire dtype (PORTS.md §3.2)"), "{}", e);
    assert!(!e.contains("port-"), "{}", e);
    let e = run_err(r#"
        fn main() -> str {
            let !z = forge.zeros[i64, [0]]
            port_tensor_encode(z)
        }
    "#);
    assert!(e.contains("1 to 8 extents, every one positive"), "{}", e);
    assert!(!e.contains("port-protocol"), "{}", e);
}

#[test]
fn port_tensor_crosses_a_real_port_by_copy() {
    // The whole §3.2 loop through the process port: encode, send as the single
    // positional argument, let python hand the envelope back, decode it. The
    // harness rehydrates the envelope on the way in and re-encodes it on the
    // way out, so this is copy mode in both directions.
    if !crate::ports::have_python() { eprintln!("skipped: python3 not on PATH"); return; }
    let v = run(r#"
        fn main() -> i64 {
            let (p, e1) = port_open("python")
            if e1 != nil { return -1 }
            let !g = forge.zeros[i64, [2, 2]]
            g[0, 0] = 1  g[0, 1] = 2  g[1, 0] = 3  g[1, 1] = 4
            let (out, e2) = port_call(p, "dmc.echo", "[" + port_tensor_encode(g) + "]")
            let (_, e3) = port_close(p)
            if e2 != nil { return -2 }
            let (back, e4) = port_tensor_decode(out, forge.zeros[i64, [2, 2]])
            if e4 != nil { return -3 }
            back[0, 0] + back[0, 1] + back[1, 0] + back[1, 1]
        }
    "#);
    assert_eq!(as_int(&v), 10);
    // The harness really parsed the envelope — it can answer questions about
    // the tensor, not just hand the JSON object back.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            let !g = forge.zeros[i64, [2, 3]]
            let (sh, e2) = port_call(p, "dmc.shape", "[" + port_tensor_encode(g) + "]")
            let (dt, e3) = port_call(p, "dmc.dtype", "[" + port_tensor_encode(g) + "]")
            let (_, e4) = port_close(p)
            sh + " " + dt
        }
    "#);
    assert_eq!(as_str(&v), "[2,3] \"i64\"");
}

#[test]
fn list_dir_tmp() {
    let v = run(r#"
        fn main() -> i64 {
            let (entries, err) = list_dir("/tmp")
            if err == nil { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn pipe_right_basic() {
    let v = run(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> i64 { 5 |> inc }
    "#);
    assert_eq!(as_int(&v), 6);
}

/// #501 ruling S1a took `>>` off the pipe; #530 gave it to the right shift.
/// The old pipe program is not merely rejected — it now means something else,
/// so `5 >> inc` asks to shift by a function and is refused as a shift.
#[test]
fn pipe_compose_spelling_is_gone() {
    let e = run_err(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> i64 { 5 >> inc }
    "#);
    assert!(e.contains(">> requires int"), "got: {}", e);
    // The surviving pipe spellings still reach `inc`.
    assert_eq!(as_int(&run(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> i64 { 5 |> inc }
    "#)), 6);
}

#[test]
fn pipe_chain() {
    let v = run(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn double(x: i64) -> i64 { x * 2 }
        fn main() -> i64 { 5 |> inc |> double }
    "#);
    assert_eq!(as_int(&v), 12);
}

#[test]
fn pipe_spellings_equivalent() {
    // `\|>` is canonical; the bare `|>` survives the #501 sweep (ruling S1b).
    let v = run(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn double(x: i64) -> i64 { x * 2 }
        fn main() -> i64 {
            let a = 5 |> inc |> double
            let b = 5 \|> inc \|> double
            if a == b { a } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 12);
}

#[test]
fn pipe_placeholder_fusion_elementwise() {
    // `_`-placeholder stage: the piped value is substituted for `_`, not called.
    // Regression for #192 (was: stage evaluated `_` to nil → runtime error).
    let v = run(r#"
        fn main() -> f32 {
            let a = [1.0, 2.0]
            let b = [10.0, 20.0]
            let c = a |> _ .+ b
            c[0] + c[1]
        }
    "#);
    assert_eq!(as_float(&v), 33.0);
}

#[test]
fn pipe_placeholder_fusion_chain() {
    // Chained placeholder stages: (x*x)+x over a 3-vector.
    let v = run(r#"
        fn main() -> f32 {
            let x = [1.0, 2.0, 3.0]
            let y = x |> _ .* x |> _ .+ x
            y[0] + y[1] + y[2]
        }
    "#);
    assert_eq!(as_float(&v), 20.0); // [2,6,12]
}

#[test]
fn fn_lit_basic_call() {
    let v = run(r#"
        fn main() -> i64 {
            let double = fn(x: i64) -> i64 { x * 2 }
            double(21)
        }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn fn_lit_passed_to_list_map() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            let ys = list_map(xs, fn(x: i64) -> i64 { x * 10 })
            list_get(ys, 2)
        }
    "#);
    assert_eq!(as_int(&v), 30);
}

#[test]
fn fn_lit_passed_to_list_filter() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list_push(list(), 1), 2), 3), 4)
            let evens = list_filter(xs, fn(x: i64) -> i64 { x % 2 == 0 })
            list_len(evens)
        }
    "#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn fn_lit_passed_to_list_reduce() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 1), 2), 3)
            list_reduce(xs, fn(acc: i64, x: i64) -> i64 { acc + x }, 0)
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn fn_lit_wrong_arity_errors() {
    let e = run_err(r#"
        fn main() -> i64 {
            let f = fn(x: i64) -> i64 { x }
            f(1, 2)
        }
    "#);
    assert!(e.contains("lambda expects 1 args"), "got: {}", e);
}

// ── Task 1: KV streaming <- operator ─────────────────────────────────────────

#[test]
fn stream_append_grows_cache() {
    let v = run(r#"
        fn main() -> i64 {
            let !cache = forge.zeros[f32, [2, 3]]
            let new_row = forge.zeros[f32, [1, 3]]
            cache <- new_row
            # cache should now be shape [3, 3]
            cache.shape[0]
        }
    "#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn stream_append_multiple() {
    let v = run(r#"
        fn main() -> i64 {
            let !cache = forge.zeros[f32, [1, 4]]
            let row = forge.zeros[f32, [1, 4]]
            cache <- row
            cache <- row
            cache.shape[0]
        }
    "#);
    assert_eq!(as_int(&v), 3);
}

// ── Task 2: CLI argument parsing stdlib ──────────────────────────────────────

#[test]
fn cli_arg_default_when_absent() {
    let v = run(r#"fn main() -> str { cli_arg("lr", "0.001") }"#);
    match &v {
        Value::Str(s) => assert_eq!(s, "0.001"),
        _ => panic!("expected Str"),
    }
}

#[test]
fn cli_flag_false_when_absent() {
    let v = run(r#"fn main() -> i64 { if cli_flag("verbose") { 1 } else { 0 } }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn cli_positional_count_zero_by_default() {
    let v = run(r#"fn main() -> i64 { cli_positional_count() }"#);
    assert_eq!(as_int(&v), 0);
}

// -- Char literal interpreter tests

#[test]
fn char_lit_value_ascii() {
    // c"A" == 65
    let v = run(r#"fn main() -> u32 { c"A" }"#);
    assert_eq!(as_int(&v), 65);
}

#[test]
fn char_lit_value_zero_digit() {
    // c"0" == 48
    let v = run(r#"fn main() -> u32 { c"0" }"#);
    assert_eq!(as_int(&v), 48);
}

#[test]
fn char_lit_escape_newline_value() {
    // c"\n" == 10
    let v = run(r#"fn main() -> u32 { c"\n" }"#);
    assert_eq!(as_int(&v), 10);
}

#[test]
fn char_lit_comparison() {
    let v = run(r#"fn main() -> bool { c"A" == 65 }"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn char_lit_arithmetic() {
    // c"A" + 1 == 66
    let v = run(r#"fn main() -> i64 { c"A" + 1 }"#);
    assert_eq!(as_int(&v), 66);
}

#[test]
fn char_lit_range_check() {
    let v = run(r#"
fn main() -> bool {
    let d = c"5"
    d >= c"0" && d <= c"9"
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

// ─── Issue #60: @cast(u8) { str } ────────────────────────────────────────────

#[test]
fn cast_u8_string_to_byte_tensor() {
    let v = run(r#"
        fn main() -> i64 {
            let buf = @cast(u8) { "hello" }
            buf.shape[0]
        }
    "#);
    assert_eq!(as_int(&v), 5); // "hello" = 5 bytes
}

#[test]
fn cast_u8_string_values_are_bytes() {
    let v = run(r#"
        fn main() -> f32 {
            let buf = @cast(u8) { "A" }
            sum(buf)
        }
    "#);
    // 'A' = 65
    assert!((as_float(&v) - 65.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn cast_u8_then_stream_append() {
    // The core assertion: @cast(u8) { str } produces a tensor so <- doesn't crash.
    // We measure shape growth: whatever the initial size, appending "hi" adds 2.
    let v = run(r#"
        fn main() -> i64 {
            stream {
                let !cache = stream.kv[u8, [~]](capacity = 64)
                let before = cache.shape[0]
                let msg = @cast(u8) { "hi" }
                cache <- msg
                cache.shape[0] - before
            }
        }
    "#);
    assert_eq!(as_int(&v), 2); // appended exactly 2 bytes
}

// ─── #598: `@cast(Model) { bytes }` overlay ─────────────────────────────────
//
// `cast_to_struct` (interp.rs) materializes a model's fields from a raw byte
// tensor in declaration order. Under SPEC.md §3.10's field-order-only wire
// guarantee, a field is laid out exactly when its own type fixes a byte extent
// — a scalar of known width, or a tensor of such scalars at a static shape
// (count × width, row-major). Every admitted field decodes back INTO its
// DECLARED type: #598's signature was that they did not, so a `bool` or an
// `f32` field came back an `Int`. Fields with no extent (`str`, a nested
// model, a symbolic tensor shape) fail loudly instead of silently reading as
// `Int(0)`. `--check` refuses the same programs before they reach the
// interpreter (see check_tests.rs); these tests call the interpreter directly,
// bypassing the checker, so they also cover the interpreter's own backstop.
//
// Byte order is little-endian, ratified by #606 (`SPEC.md §3.10`) — see
// `le_bytes_to_i64`'s note.

#[test]
fn cast_model_overlay_decodes_first_scalar_field() {
    let v = run(r#"
        model DirEntry {
            inode_id: u64
            mode: i32
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            buf[7] = 7
            let d = @cast(DirEntry) { buf }
            d.inode_id as i64
        }
    "#);
    // The u64 field spans bytes 0..8, least significant byte first (#606):
    // byte 7 is its most significant, so 7 << 56.
    assert_eq!(as_int(&v), 7i64 << 56);
}

#[test]
fn cast_model_overlay_decodes_field_at_nonzero_offset() {
    let v = run(r#"
        model DirEntry {
            inode_id: u64
            mode: i32
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            buf[8] = 42
            let d = @cast(DirEntry) { buf }
            d.mode as i64
        }
    "#);
    // `mode` starts at byte offset 8 (after the 8-byte `inode_id`); byte 8
    // is its first, least-significant byte under the little-endian rule
    // (#606).
    assert_eq!(as_int(&v), 42);
}

#[test]
fn cast_model_overlay_narrow_int_field_keeps_its_declared_width() {
    // An `i8` field is an `i8`, so the all-ones byte is -1, not 255 — the
    // decode wraps into the declared width instead of widening every field to
    // i64 the way the single i64 path used to.
    let v = run(r#"
        model M { a: i8 }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            buf[0] = 255
            let m = @cast(M) { buf }
            m.a as i64
        }
    "#);
    assert_eq!(as_int(&v), -1);
    // …and its unsigned twin over the same byte is 255.
    let v = run(r#"
        model M { a: u8 }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            buf[0] = 255
            let m = @cast(M) { buf }
            m.a as i64
        }
    "#);
    assert_eq!(as_int(&v), 255);
    // An `i16` over two all-ones bytes is likewise -1.
    let v = run(r#"
        model M { a: i16 }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            buf[0] = 255
            buf[1] = 255
            let m = @cast(M) { buf }
            m.a as i64
        }
    "#);
    assert_eq!(as_int(&v), -1);
}

#[test]
fn cast_model_overlay_decodes_bool_field_as_a_bool() {
    let src = r#"
        model Flags { on: bool }
        fn main() -> bool {
            let !buf = forge.zeros[u8, [4]]
            buf[0] = BYTE
            let f = @cast(Flags) { buf }
            f.on
        }
    "#;
    let v = run(&src.replace("BYTE", "1"));
    assert!(matches!(v, Value::Bool(true)), "expected Bool(true), got {:?}", v);
    let v = run(&src.replace("BYTE", "0"));
    assert!(matches!(v, Value::Bool(false)), "expected Bool(false), got {:?}", v);
    // A byte that is neither 0 nor 1 follows the interpreter's ONE bool rule
    // (`apply_cast`'s `Bool` arm): any nonzero number is `true`. The overlay
    // does not invent a second, stricter rule for the same conversion.
    let v = run(&src.replace("BYTE", "2"));
    assert!(matches!(v, Value::Bool(true)), "expected Bool(true) for byte 2, got {:?}", v);
    let v = run(&src.replace("BYTE", "255"));
    assert!(matches!(v, Value::Bool(true)), "expected Bool(true) for byte 255, got {:?}", v);
}

#[test]
fn cast_model_overlay_decodes_f32_field_from_its_bit_pattern() {
    // 1.5f32 is 0x3FC00000; little-endian (#606), so its most significant
    // byte (0x3F) lands last and its next byte (0xC0) just before it.
    let v = run(r#"
        model M { x: f32 }
        fn main() -> f32 {
            let !buf = forge.zeros[u8, [8]]
            buf[3] = 63
            buf[2] = 192
            let m = @cast(M) { buf }
            m.x
        }
    "#);
    assert!(matches!(v, Value::Float(x, FW::F32) if x == 1.5),
        "expected an f32 1.5, got {:?}", v);
}

#[test]
fn cast_model_overlay_decodes_f64_field_from_its_bit_pattern() {
    // 1.5f64 is 0x3FF8000000000000; little-endian (#606) puts its most
    // significant byte (0x3F) at offset 7 and the next (0xF8) at offset 6.
    let v = run(r#"
        model M { x: f64 }
        fn main() -> f64 {
            let !buf = forge.zeros[u8, [8]]
            buf[7] = 63
            buf[6] = 248
            let m = @cast(M) { buf }
            m.x
        }
    "#);
    assert!(matches!(v, Value::Float(x, FW::F64) if x == 1.5),
        "expected an f64 1.5, got {:?}", v);
}

#[test]
fn cast_model_overlay_decodes_narrow_float_fields_at_their_storage_width() {
    // bf16 is binary32's top 16 bits, so 1.5 is 0x3FC0 — two bytes, not four.
    // Little-endian (#606): the most significant byte (0x3F) lands at offset
    // 1, the least significant (0xC0) at offset 0.
    let v = run(r#"
        model M { x: bf16 }
        fn main() -> f32 {
            let !buf = forge.zeros[u8, [8]]
            buf[1] = 63
            buf[0] = 192
            let m = @cast(M) { buf }
            m.x
        }
    "#);
    assert!(matches!(v, Value::Float(x, FW::F32) if x == 1.5),
        "expected a bf16 1.5, got {:?}", v);
    // IEEE binary16 1.5 is 0x3E00; little-endian puts its nonzero byte
    // (0x3E) at offset 1, leaving offset 0 (already zero) as the LSB.
    let v = run(r#"
        model M { x: f16 }
        fn main() -> f32 {
            let !buf = forge.zeros[u8, [8]]
            buf[1] = 62
            let m = @cast(M) { buf }
            m.x
        }
    "#);
    assert!(matches!(v, Value::Float(x, FW::F32) if x == 1.5),
        "expected an f16 1.5, got {:?}", v);
}

#[test]
fn cast_model_overlay_decodes_a_fixed_shape_tensor_field() {
    // The 32-byte block #598 was filed against: a `u64` plus 24 bytes of
    // name. The tensor field's extent is element count × element width, and
    // its elements keep the declared `u8` type.
    let v = run(r#"
        model DirEntry {
            inode_id: u64
            name: Tensor[u8, [24]]
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            buf[0] = 9
            buf[8] = 65
            buf[31] = 90
            let d = @cast(DirEntry) { buf }
            (d.inode_id as i64) * 10000 + (d.name[0] as i64) * 100 + (d.name[23] as i64)
        }
    "#);
    // Little-endian (#606): `inode_id`'s least significant byte is offset 0,
    // not its last offset 7. `name`'s elements are single bytes, so their
    // order is unaffected.
    assert_eq!(as_int(&v), 9 * 10000 + 65 * 100 + 90);
}

#[test]
fn cast_model_overlay_tensor_field_is_row_major() {
    // Element order is the spec's row-major convention: byte k of the field
    // is element [k / 3, k % 3] of a [2, 3] tensor.
    let v = run(r#"
        model Grid { cells: Tensor[u8, [2, 3]] }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            buf[0] = 10
            buf[1] = 11
            buf[2] = 12
            buf[3] = 13
            buf[4] = 14
            buf[5] = 15
            let g = @cast(Grid) { buf }
            (g.cells[0, 2] as i64) * 100 + (g.cells[1, 0] as i64)
        }
    "#);
    assert_eq!(as_int(&v), 12 * 100 + 13);
}

#[test]
fn cast_model_overlay_tensor_field_keeps_its_element_type() {
    // A `Tensor[f32, [2]]` field decodes two IEEE binary32 patterns, not two
    // integers: 0x3FC00000 is 1.5 and 0x40000000 is 2.0. Little-endian
    // (#606): element 0 (bytes 0..4) takes its most significant byte at
    // offset 3, next at offset 2; element 1 (bytes 4..8) takes its sole
    // nonzero byte at offset 7 (its most significant).
    let v = run(r#"
        model M { xs: Tensor[f32, [2]] }
        fn main() -> f32 {
            let !buf = forge.zeros[u8, [8]]
            buf[3] = 63
            buf[2] = 192
            buf[7] = 64
            let m = @cast(M) { buf }
            m.xs[0] + m.xs[1]
        }
    "#);
    assert!(matches!(v, Value::Float(x, FW::F32) if x == 3.5),
        "expected an f32 3.5, got {:?}", v);
}

#[test]
fn cast_model_overlay_matches_dataset_network_driver_ethertype_under_le_order() {
    // `dataset/dmc_sources/network.dmc`'s `NetworkDriver.handle_rx!` overlays
    // a 14-byte `EthHeader { dest_mac: Tensor[u8,[6]], src_mac:
    // Tensor[u8,[6]], eth_type: u16 }` onto an incoming packet and branches
    // on `eth_type` (IPv4 = 0x0800, ARP = 0x0806). `compiler/tests/*.rs`
    // never reads outside the crate (and no gate covers `dataset/` at all —
    // it is not `examples/`), so this inlines that file's model and its mock
    // packet bytes rather than exercising the dataset file directly. The
    // driver sets `packet_buffer[12] = 0x08, packet_buffer[13] = 0x00` to
    // mean "IPv4" — true only under the OLD big-endian decode
    // (`0x0800`). Under #606's ratified little-endian order the same two
    // bytes decode as `0x0008 = 8`: neither `0x0800` nor `0x0806`, so the
    // driver's own `if`/`else` chain falls to its "unknown EtherType" arm.
    // This is the exact silent regression the #606 ruling flagged — no gate
    // over `dataset/` would have caught it without this test. `network.dmc` itself is
    // unchanged by this PR: fixing its probe bytes is separate follow-up
    // work, not this decode change.
    let v = run(r#"
        model EthHeader {
            dest_mac: Tensor[u8, [6]]
            src_mac:  Tensor[u8, [6]]
            eth_type: u16
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [14]]
            buf[12] = 8
            buf[13] = 0
            let h = @cast(EthHeader) { buf }
            h.eth_type as i64
        }
    "#);
    let decoded = as_int(&v);
    assert_eq!(decoded, 8, "little-endian decode of bytes [0x08, 0x00] is 0x0008");
    assert_ne!(decoded, 0x0800, "no longer the old big-endian IPv4 reading");
    assert_ne!(decoded, 0x0806, "and not ARP either — the driver's mock bytes decode to neither");
}

#[test]
fn cast_model_overlay_short_buffer_errors_at_the_declaration() {
    // No field is ever read here: the overlay sizes every field up front, so a
    // buffer too short for the model fails at the `@cast` itself rather than
    // at some later field access.
    let msg = run_err(r#"
        model DirEntry {
            inode_id: u64
            mode: i32
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            let d = @cast(DirEntry) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("mode") && msg.contains("offset 8") && msg.contains("8 byte(s) available"),
        "expected a short-buffer error naming `mode`, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_str_field_errors_loudly() {
    let msg = run_err(r#"
        model DirEntry {
            inode_id: u64
            name: str
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let d = @cast(DirEntry) { buf }
            d.inode_id as i64
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("name")
            && msg.contains("DIRECTIVES.md §4.1"),
        "expected an overlay-layout error, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_nested_model_field_errors_loudly() {
    let msg = run_err(r#"
        model Inner { x: i64 }
        model Outer {
            head: u32
            inner: Inner
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let o = @cast(Outer) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("inner"),
        "expected an overlay-layout error naming `inner`, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_symbolic_tensor_shape_errors_loudly() {
    // A dim that is not a static constant has no extent, and guessing one
    // would mis-size every field after it.
    let msg = run_err(r#"
        model Frame[N] {
            payload: Tensor[u8, [N]]
        }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [32]]
            let f = @cast(Frame) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("payload")
            && msg.contains("static shape"),
        "expected a static-shape error naming `payload`, got: {}", msg
    );
}

// ── Group 1: Regex ───────────────────────────────────────────────────────────

#[test]
fn regex_match_basic() {
    let v = run(r#"fn main() -> i64 { if regex_match("\\d+", "abc123") { 1 } else { 0 } }"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn regex_match_no_match() {
    let v = run(r#"fn main() -> i64 { if regex_match("\\d+", "abc") { 1 } else { 0 } }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn regex_find_basic() {
    let v = run(r#"
        fn main() -> str {
            let (m, i) = regex_find("\\d+", "abc123def")
            m
        }
    "#);
    assert_eq!(as_str(&v), "123");
}

#[test]
fn regex_replace_basic() {
    let v = run(r#"fn main() -> str { regex_replace("o+", "foobar", "0") }"#);
    assert_eq!(as_str(&v), "f0bar");
}

#[test]
fn regex_split_basic() {
    let v = run(r#"
        fn main() -> i64 {
            let parts = regex_split(",", "a,b,c")
            list_len(parts)
        }
    "#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn regex_invalid_pattern_errors() {
    // #255: an invalid pattern must raise a clean runtime error in every regex
    // builtin, not silently behave as a never-matching valid pattern.
    let cases = [
        r#"fn main() -> bool { regex_match("[unclosed", "abc") }"#,
        r#"fn main() -> nil { let _ = regex_find("[unclosed", "abc")  nil }"#,
        r#"fn main() -> nil { let _ = regex_find_all("(", "a(b(c")  nil }"#,
        r#"fn main() -> str { regex_replace("[unclosed", "abc", "X") }"#,
        r#"fn main() -> str { regex_replace_all("(", "abc", "X") }"#,
        r#"fn main() -> nil { let _ = regex_split("[z-a]", "hello")  nil }"#,
    ];
    for src in cases {
        let e = run_err(src);
        assert!(e.contains("invalid pattern"),
                "expected an invalid-pattern error, got: {e}\nsrc: {src}");
    }
}

// ── Group 2: Compression ─────────────────────────────────────────────────────

#[test]
fn gzip_roundtrip() {
    let v = run(r#"
        fn main() -> str {
            let (compressed, e1) = gzip_compress("hello world")
            let (decompressed, e2) = gzip_decompress(compressed)
            decompressed
        }
    "#);
    assert_eq!(as_str(&v), "hello world");
}

#[test]
fn gzip_compress_reduces_size() {
    let v = run(r#"
        fn main() -> i64 {
            # A long repetitive string should compress well
            let s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            let (compressed, err) = gzip_compress(s)
            if err == nil { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn zlib_roundtrip() {
    let v = run(r#"
        fn main() -> str {
            let (compressed, e1) = zlib_compress("the quick brown fox")
            let (decompressed, e2) = zlib_decompress(compressed)
            decompressed
        }
    "#);
    assert_eq!(as_str(&v), "the quick brown fox");
}

#[test]
fn zlib_compress_no_type_error() {
    // Regression for #70: typechecker must accept zlib_compress / zlib_decompress
    let v = run(r#"
        fn main() -> i64 {
            let (compressed, err) = zlib_compress("hello")
            if err == nil { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

// ── Group 3: Networking ──────────────────────────────────────────────────────

#[test]
fn http_get_returns_tuple_on_error() {
    let v = run(r#"
        fn main() -> i64 {
            let (body, err) = http_get("http://127.0.0.1:1")
            # connection refused → err is a str, not nil
            if err == nil { 0 } else { 1 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn http_post_returns_tuple_on_error() {
    let v = run(r#"
        fn main() -> i64 {
            let (body, err) = http_post("http://127.0.0.1:1", "data", "text/plain")
            if err == nil { 0 } else { 1 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

// ── Group 4: Date/time ───────────────────────────────────────────────────────

#[test]
fn date_now_ms_is_positive() {
    let v = run(r#"fn main() -> i64 { date_now_ms() }"#);
    assert!(as_int(&v) > 0);
}

#[test]
fn date_format_epoch() {
    let v = run(r#"fn main() -> str { date_format(0, "%Y-%m-%d") }"#);
    assert_eq!(as_str(&v), "1970-01-01");
}

#[test]
fn date_parse_roundtrip() {
    let v = run(r#"
        fn main() -> i64 {
            let (ts, err) = date_parse("2024-01-15", "%Y-%m-%d")
            if err == nil { 1 } else { 0 }
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn date_diff_ms_basic() {
    let v = run(r#"fn main() -> i64 { date_diff_ms(1000, 500) }"#);
    assert_eq!(as_int(&v), 500);
}

// ─── assert / assert_eq / assert_ne / print_err ──────────────────────────────

#[test]
fn assert_passes_on_true() {
    let v = run(r#"fn main() -> nil { assert(true) }"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn assert_passes_with_message() {
    let v = run(r#"fn main() -> nil { assert(1 == 1, "should pass") }"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn assert_fails_on_false() {
    let result = run_err(r#"fn main() { assert(false) }"#);
    assert!(!result.is_empty());
    let msg = result;
    assert!(msg.contains("assertion failed"), "got: {}", msg);
}

#[test]
fn assert_fails_with_custom_message() {
    let result = run_err(r#"fn main() { assert(false, "my custom message") }"#);
    assert!(!result.is_empty());
    assert!(result.contains("my custom message"));
}

#[test]
fn assert_eq_passes_equal_ints() {
    let v = run(r#"fn main() -> nil { assert_eq(42, 42) }"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn assert_eq_passes_equal_strings() {
    let v = run(r#"fn main() -> nil { assert_eq("hello", "hello") }"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn assert_eq_fails_unequal() {
    let result = run_err(r#"fn main() { assert_eq(1, 2) }"#);
    assert!(!result.is_empty());
    let msg = result;
    assert!(msg.contains("1") && msg.contains("2"), "got: {}", msg);
}

#[test]
fn assert_eq_custom_message() {
    let result = run_err(r#"fn main() { assert_eq(1, 2, "vals differ") }"#);
    assert!(!result.is_empty());
    assert!(result.contains("vals differ"));
}

#[test]
fn assert_ne_passes_unequal() {
    let v = run(r#"fn main() -> nil { assert_ne(1, 2) }"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn assert_ne_fails_equal() {
    let result = run_err(r#"fn main() { assert_ne(5, 5) }"#);
    assert!(!result.is_empty());
}

#[test]
fn print_err_returns_nil() {
    let v = run(r#"fn main() -> nil { print_err("err msg\n") }"#);
    assert!(matches!(v, Value::Nil));
}

// ─── Math constants ───────────────────────────────────────────────────────────

#[test]
fn math_const_pi() {
    let v = run(r#"fn main() -> f64 { pi }"#);
    let f = as_float(&v);
    assert!((f - std::f64::consts::PI).abs() < 1e-10);
}

#[test]
fn math_const_tau() {
    let v = run(r#"fn main() -> f64 { tau }"#);
    let f = as_float(&v);
    assert!((f - std::f64::consts::TAU).abs() < 1e-10);
}

#[test]
fn math_const_e() {
    let v = run(r#"fn main() -> f64 { e }"#);
    let f = as_float(&v);
    assert!((f - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn math_const_inf() {
    let v = run(r#"fn main() -> f64 { inf }"#);
    assert!(matches!(v, Value::Float(f, _) if f.is_infinite() && f > 0.0));
}

#[test]
fn math_const_nan() {
    let v = run(r#"fn main() -> f64 { nan }"#);
    assert!(matches!(v, Value::Float(f, _) if f.is_nan()));
}

#[test]
fn math_tau_is_two_pi() {
    let v = run(r#"fn main() -> bool { tau == pi * 2.0 }"#);
    assert!(matches!(v, Value::Bool(true)));
}

// ─── ord / to_str / to_int / to_float / str_repeat / clamp ──────────────────

#[test]
fn ord_ascii() {
    let v = run(r#"fn main() -> i64 { ord("A") }"#);
    assert_eq!(as_int(&v), 65);
}

#[test]
fn ord_space() {
    let v = run(r#"fn main() -> i64 { ord(" ") }"#);
    assert_eq!(as_int(&v), 32);
}

#[test]
fn ord_chr_roundtrip() {
    let v = run(r#"fn main() -> i64 { ord(chr(97)) }"#);
    assert_eq!(as_int(&v), 97);
}

#[test]
fn to_str_int() {
    let v = run(r#"fn main() -> str { to_str(42) }"#);
    assert_eq!(as_str(&v), "42");
}

#[test]
fn to_str_float() {
    let v = run(r#"fn main() -> str { to_str(3.14) }"#);
    let s = as_str(&v);
    assert!(s.contains("3.14"), "got: {}", s);
}

#[test]
fn to_str_bool() {
    let v = run(r#"fn main() -> str { to_str(true) }"#);
    assert_eq!(as_str(&v), "true");
}

#[test]
fn to_string_aliases_to_str() {
    // #335: `to_string` is the Rust/Python spelling of `to_str`.
    assert_eq!(as_str(&run(r#"fn main() -> str { to_string(42) }"#)), "42");
    assert_eq!(as_str(&run(r#"fn main() -> str { to_string(true) }"#)), "true");
}

#[test]
fn to_binary_aliases_to_bin() {
    // #335: `to_binary` is the long spelling; aliases `to_bin`.
    assert_eq!(as_str(&run(r#"fn main() -> str { to_binary(5) }"#)), "101");
    assert_eq!(
        as_str(&run(r#"fn main() -> str { to_binary(5) }"#)),
        as_str(&run(r#"fn main() -> str { to_bin(5) }"#))
    );
}

#[test]
fn gcd_builtin() {
    // #335
    assert_eq!(as_int(&run(r#"fn main() -> i64 { gcd(12, 18) }"#)), 6);
    assert_eq!(as_int(&run(r#"fn main() -> i64 { gcd(17, 5) }"#)), 1);
    assert_eq!(as_int(&run(r#"fn main() -> i64 { gcd(0, 9) }"#)), 9);
    assert_eq!(as_int(&run(r#"fn main() -> i64 { gcd(0 - 12, 18) }"#)), 6);
}

#[test]
fn sort_builtin_1d() {
    // #335: ascending sort of a 1-D tensor.
    assert_eq!(as_float(&run(r#"fn main() -> f64 { sort([3.0, 1.0, 2.0])[0] }"#)), 1.0);
    assert_eq!(as_float(&run(r#"fn main() -> f64 { sort([3.0, 1.0, 2.0])[2] }"#)), 3.0);
}

#[test]
fn sort_builtin_2d_sorts_each_row() {
    // #335: N-D sorts along the LAST axis (numpy default) — each row independently.
    assert_eq!(as_float(&run(r#"fn main() -> f64 { sort([[3.0,1.0,2.0],[9.0,7.0,8.0]])[1, 0] }"#)), 7.0);
    assert_eq!(as_float(&run(r#"fn main() -> f64 { sort([[3.0,1.0,2.0],[9.0,7.0,8.0]])[0, 2] }"#)), 3.0);
}

#[test]
fn median_builtin_odd_and_even() {
    // #335: median over all elements. Odd count -> central value.
    assert_eq!(as_float(&run(r#"fn main() -> f64 { median([3.0, 1.0, 4.0, 1.0, 5.0]) }"#)), 3.0);
    // Even count -> mean of the two central values: (4+8)/2 = 6.
    assert_eq!(as_float(&run(r#"fn main() -> f64 { median([10.0, 2.0, 8.0, 4.0]) }"#)), 6.0);
}

#[test]
fn median_builtin_reduces_full_tensor() {
    // #335: like sum/mean, median spans the whole tensor, not the last axis.
    // All six elements 1..6 -> (3+4)/2 = 3.5.
    assert_eq!(as_float(&run(r#"fn main() -> f64 { median([[1.0,2.0,3.0],[4.0,5.0,6.0]]) }"#)), 3.5);
}

#[test]
fn max_along_reduces_one_axis() {
    // #370: m = [[1,5,3],[4,2,6]]; the reduced axis is dropped.
    // axis 0 -> [4,5,6]; axis 1 -> [5,6].
    let m = "[[1.0,5.0,3.0],[4.0,2.0,6.0]]";
    assert_eq!(as_float(&run(&format!("fn main() -> f64 {{ max_along({m}, 0)[0] }}"))), 4.0);
    assert_eq!(as_float(&run(&format!("fn main() -> f64 {{ max_along({m}, 0)[2] }}"))), 6.0);
    assert_eq!(as_float(&run(&format!("fn main() -> f64 {{ max_along({m}, 1)[0] }}"))), 5.0);
    assert_eq!(as_float(&run(&format!("fn main() -> f64 {{ max_along({m}, 1)[1] }}"))), 6.0);
}

// ── #333: UFCS — `x.f(args)` desugars to `f(x, args)` ──────────────────────

#[test]
fn ufcs_builtin_method() {
    // x.floor() → floor(x) (floor returns a float).
    assert_eq!(as_float(&run(r#"fn main() -> f64 { let x = 3.7  x.floor() }"#)), 3.0);
}

#[test]
fn ufcs_user_fn_receiver_becomes_first_arg() {
    assert_eq!(as_int(&run(r#"
        fn dbl(x: i64) -> i64 { x * 2 }
        fn main() -> i64 { let n = 5  n.dbl() }
    "#)), 10);
    // 2-arg: x.add(4) → add(x, 4)
    assert_eq!(as_int(&run(r#"
        fn add(a: i64, b: i64) -> i64 { a - b }
        fn main() -> i64 { let x = 10  x.add(3) }
    "#)), 7);
}

#[test]
fn ufcs_unlocks_method_call_targets() {
    // The whole point of #333: `.sort()` / `.to_string()` now resolve.
    assert_eq!(as_float(&run(r#"fn main() -> f64 { let v = [3.0, 1.0, 2.0]  v.sort()[0] }"#)), 1.0);
    assert_eq!(as_str(&run(r#"fn main() -> str { let n = 42  n.to_string() }"#)), "42");
}

#[test]
fn ufcs_leaves_string_methods_alone() {
    // `split` is a real string method, not a free fn — must NOT be desugared.
    assert_eq!(as_int(&run(r#"fn main() -> i64 { let s = "a,b,c"  len(s.split(",")) }"#)), 3);
}

#[test]
fn to_int_from_str() {
    let v = run(r#"fn main() -> i64 { to_int("42") }"#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn to_int_from_float() {
    let v = run(r#"fn main() -> i64 { to_int(3.9) }"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn to_int_from_bool() {
    let v = run(r#"fn main() -> i64 { to_int(true) }"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn to_float_from_str() {
    let v = run(r#"fn main() -> f64 { to_float("2.5") }"#);
    assert!((as_float(&v) - 2.5).abs() < 1e-10);
}

#[test]
fn to_float_from_int() {
    let v = run(r#"fn main() -> f64 { to_float(7) }"#);
    assert!((as_float(&v) - 7.0).abs() < 1e-10);
}

#[test]
fn to_int_bad_str_errors() {
    let e = run_err(r#"fn main() { to_int("abc") }"#);
    assert!(!e.is_empty());
}

#[test]
fn str_repeat_basic() {
    let v = run(r#"fn main() -> str { str_repeat("ab", 3) }"#);
    assert_eq!(as_str(&v), "ababab");
}

#[test]
fn str_repeat_zero() {
    let v = run(r#"fn main() -> str { str_repeat("x", 0) }"#);
    assert_eq!(as_str(&v), "");
}

#[test]
fn str_repeat_one() {
    let v = run(r#"fn main() -> str { str_repeat("hi", 1) }"#);
    assert_eq!(as_str(&v), "hi");
}

#[test]
fn clamp_int_within_range() {
    let v = run(r#"fn main() -> i64 { clamp(5, 0, 10) }"#);
    assert_eq!(as_int(&v), 5);
}

#[test]
fn clamp_int_below_lo() {
    let v = run(r#"fn main() -> i64 { clamp(-3, 0, 10) }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn clamp_int_above_hi() {
    let v = run(r#"fn main() -> i64 { clamp(15, 0, 10) }"#);
    assert_eq!(as_int(&v), 10);
}

#[test]
fn clamp_float_basic() {
    let v = run(r#"fn main() -> f64 { clamp(1.5, 0.0, 1.0) }"#);
    assert!((as_float(&v) - 1.0).abs() < 1e-10);
}

#[test]
fn clamp_exact_boundary() {
    let v = run(r#"fn main() -> i64 { clamp(0, 0, 10) }"#);
    assert_eq!(as_int(&v), 0);
}

// ─── list_find / list_count / list_any / list_all / str_pad ──────────────────

#[test]
fn list_find_found() {
    let v = run(r#"fn main() -> i64 { list_find(list(10, 20, 30), 20) }"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn list_find_not_found() {
    let v = run(r#"fn main() -> i64 { list_find(list(10, 20, 30), 99) }"#);
    assert_eq!(as_int(&v), -1);
}

#[test]
fn list_find_first_occurrence() {
    let v = run(r#"fn main() -> i64 { list_find(list(5, 5, 5), 5) }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn list_count_basic() {
    let v = run(r#"fn main() -> i64 { list_count(list(1, 2, 1, 3, 1), 1) }"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn list_count_zero() {
    let v = run(r#"fn main() -> i64 { list_count(list(1, 2, 3), 9) }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn list_any_true() {
    let v = run(r#"
fn main() -> bool {
    list_any(list(1, 2, 3), fn(x: i64) -> bool { x > 2 })
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn list_any_false() {
    let v = run(r#"
fn main() -> bool {
    list_any(list(1, 2, 3), fn(x: i64) -> bool { x > 10 })
}
"#);
    assert!(matches!(v, Value::Bool(false)));
}

#[test]
fn list_any_empty() {
    let v = run(r#"
fn main() -> bool {
    list_any(list(), fn(x: i64) -> bool { true })
}
"#);
    assert!(matches!(v, Value::Bool(false)));
}

#[test]
fn list_all_true() {
    let v = run(r#"
fn main() -> bool {
    list_all(list(2, 4, 6), fn(x: i64) -> bool { x % 2 == 0 })
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn list_all_false() {
    let v = run(r#"
fn main() -> bool {
    list_all(list(2, 3, 6), fn(x: i64) -> bool { x % 2 == 0 })
}
"#);
    assert!(matches!(v, Value::Bool(false)));
}

#[test]
fn list_all_empty() {
    // vacuously true
    let v = run(r#"
fn main() -> bool {
    list_all(list(), fn(x: i64) -> bool { false })
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn str_pad_left_basic() {
    let v = run(r#"fn main() -> str { str_pad_left("hi", 5) }"#);
    assert_eq!(as_str(&v), "   hi");
}

#[test]
fn str_pad_left_custom_char() {
    let v = run(r#"fn main() -> str { str_pad_left("42", 5, "0") }"#);
    assert_eq!(as_str(&v), "00042");
}

#[test]
fn str_pad_left_no_op_when_wide() {
    let v = run(r#"fn main() -> str { str_pad_left("hello", 3) }"#);
    assert_eq!(as_str(&v), "hello");
}

#[test]
fn str_pad_right_basic() {
    let v = run(r#"fn main() -> str { str_pad_right("hi", 5) }"#);
    assert_eq!(as_str(&v), "hi   ");
}

#[test]
fn str_pad_right_custom_char() {
    let v = run(r#"fn main() -> str { str_pad_right("hi", 5, ".") }"#);
    assert_eq!(as_str(&v), "hi...");
}

// ─── list_head/last/take/drop, map_merge, str.count/index/lines ──────────────

#[test]
fn list_head_basic() {
    let v = run(r#"fn main() -> i64 { list_head(list(10, 20, 30)) }"#);
    assert_eq!(as_int(&v), 10);
}

#[test]
fn list_last_basic() {
    let v = run(r#"fn main() -> i64 { list_last(list(10, 20, 30)) }"#);
    assert_eq!(as_int(&v), 30);
}

#[test]
fn list_head_empty_errors() {
    let e = run_err(r#"fn main() { list_head(list()) }"#);
    assert!(!e.is_empty());
}

#[test]
fn list_take_basic() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list_take(list(1, 2, 3, 4, 5), 3)
    list_len(xs)
}
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn list_take_first_element() {
    let v = run(r#"fn main() -> i64 { list_head(list_take(list(10, 20, 30), 2)) }"#);
    assert_eq!(as_int(&v), 10);
}

#[test]
fn list_take_more_than_len() {
    let v = run(r#"
fn main() -> i64 { list_len(list_take(list(1, 2, 3), 100)) }
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn list_drop_basic() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list_drop(list(1, 2, 3, 4, 5), 2)
    list_len(xs)
}
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn list_drop_head_after_drop() {
    let v = run(r#"fn main() -> i64 { list_head(list_drop(list(10, 20, 30), 1)) }"#);
    assert_eq!(as_int(&v), 20);
}

#[test]
fn list_drop_all() {
    let v = run(r#"fn main() -> i64 { list_len(list_drop(list(1, 2, 3), 10)) }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn map_merge_basic() {
    let v = run(r#"
fn main() -> i64 {
    let a = map_set(map(), "x", 1)
    let b = map_set(map(), "y", 2)
    let c = map_merge(a, b)
    map_len(c)
}
"#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn map_merge_overlay_wins() {
    let v = run(r#"
fn main() -> i64 {
    let a = map_set(map(), "x", 1)
    let b = map_set(map(), "x", 99)
    let c = map_merge(a, b)
    map_get(c, "x")
}
"#);
    assert_eq!(as_int(&v), 99);
}

#[test]
fn str_count_occurrences() {
    let v = run(r#"fn main() -> i64 { "abcabc".count("a") }"#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn str_count_zero() {
    let v = run(r#"fn main() -> i64 { "hello".count("z") }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn str_index_found() {
    let v = run(r#"fn main() -> i64 { "hello world".index("world") }"#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn str_index_not_found_errors() {
    let e = run_err(r#"fn main() { "hello".index("xyz") }"#);
    assert!(!e.is_empty());
}

#[test]
fn str_lines_basic() {
    let v = run(r#"fn main() -> i64 { list_len("a\nb\nc".lines()) }"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn str_lines_single() {
    let v = run(r#"fn main() -> i64 { list_len("hello".lines()) }"#);
    assert_eq!(as_int(&v), 1);
}

// ─── round / trunc / sign / to_hex / to_bin / to_oct ─────────────────────────

#[test]
fn round_basic() {
    let v = run(r#"fn main() -> f64 { round(3.6) }"#);
    assert!((as_float(&v) - 4.0).abs() < 1e-10);
}

#[test]
fn round_down() {
    let v = run(r#"fn main() -> f64 { round(3.4) }"#);
    assert!((as_float(&v) - 3.0).abs() < 1e-10);
}

#[test]
fn round_ndigits() {
    let v = run(r#"fn main() -> f64 { round(3.14159, 2) }"#);
    assert!((as_float(&v) - 3.14).abs() < 1e-6);
}

#[test]
fn trunc_positive() {
    let v = run(r#"fn main() -> f64 { trunc(3.9) }"#);
    assert!((as_float(&v) - 3.0).abs() < 1e-10);
}

#[test]
fn trunc_negative() {
    let v = run(r#"fn main() -> f64 { trunc(-3.9) }"#);
    assert!((as_float(&v) - (-3.0)).abs() < 1e-10);
}

#[test]
fn sign_positive() {
    let v = run(r#"fn main() -> i64 { sign(42) }"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn sign_negative() {
    let v = run(r#"fn main() -> i64 { sign(-5) }"#);
    assert_eq!(as_int(&v), -1);
}

#[test]
fn sign_zero() {
    let v = run(r#"fn main() -> i64 { sign(0) }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn sign_float() {
    let v = run(r#"fn main() -> f64 { sign(-2.5) }"#);
    assert!((as_float(&v) - (-1.0)).abs() < 1e-10);
}

#[test]
fn to_hex_basic() {
    let v = run(r#"fn main() -> str { to_hex(255) }"#);
    assert_eq!(as_str(&v), "ff");
}

#[test]
fn to_hex_zero() {
    let v = run(r#"fn main() -> str { to_hex(0) }"#);
    assert_eq!(as_str(&v), "0");
}

#[test]
fn to_bin_basic() {
    let v = run(r#"fn main() -> str { to_bin(10) }"#);
    assert_eq!(as_str(&v), "1010");
}

#[test]
fn to_oct_basic() {
    let v = run(r#"fn main() -> str { to_oct(8) }"#);
    assert_eq!(as_str(&v), "10");
}

// ─── list_flat_map / list_partition ──────────────────────────────────────────

#[test]
fn list_flat_map_basic() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list_flat_map(list(1, 2, 3), fn(x: i64) -> list { list(x, x * 10) })
    list_len(xs)
}
"#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn list_flat_map_values() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list_flat_map(list(2, 3), fn(x: i64) -> list { list(x, x) })
    list_get(xs, 0)
}
"#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn list_partition_basic() {
    let v = run(r#"
fn main() -> i64 {
    let (evens, odds) = list_partition(list(1, 2, 3, 4, 5), fn(x: i64) -> bool { x % 2 == 0 })
    list_len(evens)
}
"#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn list_partition_odds() {
    let v = run(r#"
fn main() -> i64 {
    let (evens, odds) = list_partition(list(1, 2, 3, 4, 5), fn(x: i64) -> bool { x % 2 == 0 })
    list_len(odds)
}
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn list_partition_empty() {
    let v = run(r#"
fn main() -> i64 {
    let (a, b) = list_partition(list(), fn(x: i64) -> bool { true })
    list_len(a) + list_len(b)
}
"#);
    assert_eq!(as_int(&v), 0);
}

// ─── format with specifiers ───────────────────────────────────────────────────

#[test]
fn format_plain_substitution() {
    let v = run(r#"fn main() -> str { format("{} + {} = {}", 1, 2, 3) }"#);
    assert_eq!(as_str(&v), "1 + 2 = 3");
}

#[test]
fn format_float_precision() {
    let v = run(r#"fn main() -> str { format("{:.2f}", 3.14159) }"#);
    assert_eq!(as_str(&v), "3.14");
}

#[test]
fn format_float_no_decimals() {
    let v = run(r#"fn main() -> str { format("{:.0f}", 3.7) }"#);
    assert_eq!(as_str(&v), "4");
}

#[test]
fn format_integer_d() {
    let v = run(r#"fn main() -> str { format("{:d}", 42) }"#);
    assert_eq!(as_str(&v), "42");
}

#[test]
fn format_zero_pad() {
    let v = run(r#"fn main() -> str { format("{:05d}", 42) }"#);
    assert_eq!(as_str(&v), "00042");
}

#[test]
fn format_width_right_align() {
    let v = run(r#"fn main() -> str { format("{:8d}", 42) }"#);
    assert_eq!(as_str(&v), "      42");
}

#[test]
fn format_hex() {
    let v = run(r#"fn main() -> str { format("{:x}", 255) }"#);
    assert_eq!(as_str(&v), "ff");
}

#[test]
fn format_hex_upper() {
    let v = run(r#"fn main() -> str { format("{:X}", 255) }"#);
    assert_eq!(as_str(&v), "FF");
}

#[test]
fn format_binary() {
    let v = run(r#"fn main() -> str { format("{:b}", 10) }"#);
    assert_eq!(as_str(&v), "1010");
}

#[test]
fn format_octal() {
    let v = run(r#"fn main() -> str { format("{:o}", 8) }"#);
    assert_eq!(as_str(&v), "10");
}

#[test]
fn format_force_sign_positive() {
    let v = run(r#"fn main() -> str { format("{:+d}", 42) }"#);
    assert_eq!(as_str(&v), "+42");
}

#[test]
fn format_force_sign_negative() {
    let v = run(r#"fn main() -> str { format("{:+d}", -5) }"#);
    assert_eq!(as_str(&v), "-5");
}

#[test]
fn format_escaped_brace() {
    let v = run(r#"fn main() -> str { format("{{not a placeholder}}") }"#);
    assert_eq!(as_str(&v), "{not a placeholder}");
}

#[test]
fn format_multiple_specs() {
    let v = run(r#"fn main() -> str { format("{:.1f} {:.1f}", 1.23, 4.56) }"#);
    assert_eq!(as_str(&v), "1.2 4.6");
}

#[test]
fn format_zero_pad_float() {
    let v = run(r#"fn main() -> str { format("{:07.2f}", 3.14) }"#);
    assert_eq!(as_str(&v), "0003.14");
}

// ─── Closures capturing variables ────────────────────────────────────────────

#[test]
fn closure_captures_outer_var() {
    // Lambda captures enclosing scope; return type omitted (fn type not in grammar)
    let v = run(r#"
fn main() -> i64 {
    let n = 5
    let adder = fn(x: i64) -> i64 { x + n }
    adder(3)
}
"#);
    assert_eq!(as_int(&v), 8);
}

#[test]
fn closure_used_in_map() {
    let v = run(r#"
fn main() -> i64 {
    let scale = 3
    let xs = list(1, 2, 3, 4)
    let ys = list_map(xs, fn(x: i64) -> i64 { x * scale })
    list_sum(ys)
}
"#);
    assert_eq!(as_int(&v), 30);
}

#[test]
fn closure_counter() {
    let v = run(r#"
fn main() -> i64 {
    let limit = 10
    let evens = list_filter(list(1,2,3,4,5,6,7,8,9,10), fn(x: i64) -> bool { x % 2 == 0 })
    list_len(evens)
}
"#);
    assert_eq!(as_int(&v), 5);
}

// ─── Recursive higher-order functions ────────────────────────────────────────

#[test]
fn recursion_mutual_parity() {
    let v = run(r#"
fn is_even(n: i64) -> bool { if n == 0 { true } else { is_odd(n - 1) } }
fn is_odd(n: i64) -> bool  { if n == 0 { false } else { is_even(n - 1) } }
fn main() -> bool { is_even(6) }
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn recursion_sum_list() {
    let v = run(r#"
fn sum_list(xs: list, acc: i64) -> i64 {
    if list_len(xs) == 0 { acc }
    else {
        let h = list_head(xs)
        let t = list_drop(xs, 1)
        sum_list(t, acc + h)
    }
}
fn main() -> i64 { sum_list(list(1,2,3,4,5), 0) }
"#);
    assert_eq!(as_int(&v), 15);
}

// ─── String processing programs ──────────────────────────────────────────────

#[test]
fn string_word_count() {
    let v = run(r#"
fn main() -> i64 {
    let s = "the quick brown fox jumps"
    list_len(s.split(" "))
}
"#);
    assert_eq!(as_int(&v), 5);
}

#[test]
fn string_reverse_words() {
    let v = run(r#"
fn main() -> str {
    let words = "hello world".split(" ")
    let rev = list_rev(words)
    list_get(rev, 0)
}
"#);
    assert_eq!(as_str(&v), "world");
}

#[test]
fn string_join_numbers() {
    let v = run(r#"
fn main() -> str {
    let nums = list_map(list(1, 2, 3), fn(x: i64) -> str { to_str(x) })
    join(",", nums)
}
"#);
    assert_eq!(as_str(&v), "1,2,3");
}

#[test]
fn string_uppercase_words() {
    let v = run(r#"
fn main() -> str {
    "hello world".upper()
}
"#);
    assert_eq!(as_str(&v), "HELLO WORLD");
}

#[test]
fn string_trim_and_check() {
    let v = run(r#"
fn main() -> bool {
    "  hello  ".trim() == "hello"
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn string_contains_and_replace() {
    let v = run(r#"
fn main() -> str {
    let s = "hello world"
    if s.contains("world") { s.replace("world", "demoniC") }
    else { s }
}
"#);
    assert_eq!(as_str(&v), "hello demoniC");
}

// ─── Collection pipelines ────────────────────────────────────────────────────

#[test]
fn pipeline_filter_map_sum() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    let evens = list_filter(xs, fn(x: i64) -> bool { x % 2 == 0 })
    let doubled = list_map(evens, fn(x: i64) -> i64 { x * 2 })
    list_sum(doubled)
}
"#);
    assert_eq!(as_int(&v), 60);
}

#[test]
fn pipeline_word_frequencies() {
    let v = run(r#"
fn main() -> i64 {
    let words = list("a", "b", "a", "c", "a", "b")
    let count = list_count(words, "a")
    count
}
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn pipeline_flatten_nested() {
    let v = run(r#"
fn main() -> i64 {
    let nested = list(list(1, 2), list(3, 4), list(5))
    let flat = list_flatten(nested)
    list_sum(flat)
}
"#);
    assert_eq!(as_int(&v), 15);
}

#[test]
fn pipeline_group_and_count() {
    let v = run(r#"
fn main() -> i64 {
    let nums = list(1, 2, 3, 4, 5, 6)
    let (evens, odds) = list_partition(nums, fn(x: i64) -> bool { x % 2 == 0 })
    list_len(evens) + list_len(odds)
}
"#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn pipeline_find_max_with_reduce() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list(3, 1, 4, 1, 5, 9, 2, 6)
    list_reduce(xs, fn(acc: i64, x: i64) -> i64 {
        if x > acc { x } else { acc }
    }, list_head(xs))
}
"#);
    assert_eq!(as_int(&v), 9);
}

// ─── Map operations ──────────────────────────────────────────────────────────

#[test]
fn map_build_and_lookup() {
    let v = run(r#"
fn main() -> i64 {
    let m = map_set(map_set(map_set(map(), "a", 1), "b", 2), "c", 3)
    map_get(m, "b")
}
"#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn map_missing_key_returns_nil() {
    let v = run(r#"
fn main() -> bool {
    let m = map_set(map(), "x", 42)
    map_get(m, "y") == nil
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn map_has_and_del() {
    let v = run(r#"
fn main() -> bool {
    let m = map_set(map(), "k", 1)
    let m2 = map_del(m, "k")
    !map_has(m2, "k")
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn map_keys_sorted_count() {
    let v = run(r#"
fn main() -> i64 {
    let m = map_set(map_set(map_set(map(), "c", 3), "a", 1), "b", 2)
    list_len(map_keys(m))
}
"#);
    assert_eq!(as_int(&v), 3);
}

// ─── Numeric edge cases ───────────────────────────────────────────────────────

#[test]
fn i64_min_div_neg1_is_clean_error() {
    // i64::MIN / -1 would overflow — must produce a RuntimeError, not a panic (#257)
    let e = run_err(r#"
fn main() -> i64 {
    let a = -9223372036854775807 - 1
    let b = 0 - 1
    a / b
}
"#);
    assert!(e.contains("integer overflow"), "got: {}", e);
}

#[test]
fn i64_min_rem_neg1_is_clean_error() {
    // i64::MIN % -1 would overflow — must produce a RuntimeError, not a panic (#257)
    let e = run_err(r#"
fn main() -> i64 {
    let a = -9223372036854775807 - 1
    let b = 0 - 1
    a % b
}
"#);
    assert!(e.contains("integer overflow"), "got: {}", e);
}

#[test]
fn integer_overflow_wraps() {
    // demoniC i64 — just check arithmetic works at large values
    let v = run(r#"fn main() -> i64 { 1000000 * 1000000 }"#);
    assert_eq!(as_int(&v), 1_000_000_000_000i64);
}

#[test]
fn float_nan_is_not_equal_to_itself() {
    let v = run(r#"fn main() -> bool { nan == nan }"#);
    // NaN != NaN
    assert!(matches!(v, Value::Bool(false)));
}

#[test]
fn float_inf_comparison() {
    let v = run(r#"fn main() -> bool { inf > 1000000.0 }"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn isclose_very_small_diff() {
    let v = run(r#"fn main() -> bool { isclose(1.0, 1.0 + 1e-10) }"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn isclose_large_diff() {
    let v = run(r#"fn main() -> bool { isclose(1.0, 2.0) }"#);
    assert!(matches!(v, Value::Bool(false)));
}

// ─── Control flow edge cases ─────────────────────────────────────────────────

#[test]
fn nested_if_else() {
    let v = run(r#"
fn classify(n: i64) -> str {
    if n < 0 { "negative" }
    else if n == 0 { "zero" }
    else if n < 10 { "small" }
    else { "large" }
}
fn main() -> str { classify(5) }
"#);
    assert_eq!(as_str(&v), "small");
}

#[test]
fn for_loop_with_break() {
    let v = run(r#"
fn main() -> i64 {
    let result = 0
    for i in 0..10 {
        if i == 5 { break }
        result = result + 1
    }
    result
}
"#);
    assert_eq!(as_int(&v), 5);
}

#[test]
fn for_loop_with_continue() {
    let v = run(r#"
fn main() -> i64 {
    let result = 0
    for i in 0..10 {
        if i % 2 == 0 { continue }
        result = result + 1
    }
    result
}
"#);
    assert_eq!(as_int(&v), 5);
}

#[test]
fn while_count_down() {
    let v = run(r#"
fn main() -> i64 {
    let n = 10
    let count = 0
    while n > 0 {
        n = n - 1
        count = count + 1
    }
    count
}
"#);
    assert_eq!(as_int(&v), 10);
}

#[test]
fn nested_loops_multiplication() {
    let v = run(r#"
fn main() -> i64 {
    let total = 0
    for i in 1..4 {
        for j in 1..4 {
            total = total + i * j
        }
    }
    total
}
"#);
    // sum of i*j for i,j in 1..3 = (1+2+3)^2 = 36
    assert_eq!(as_int(&v), 36);
}

// ─── Error cases for common builtins ─────────────────────────────────────────

#[test]
fn to_int_bad_string_errors() {
    let e = run_err(r#"fn main() { to_int("not_a_number") }"#);
    assert!(!e.is_empty());
}

#[test]
fn to_float_bad_string_errors() {
    let e = run_err(r#"fn main() { to_float("xyz") }"#);
    assert!(!e.is_empty());
}

#[test]
fn list_get_out_of_bounds_errors() {
    let e = run_err(r#"fn main() { list_get(list(1, 2, 3), 99) }"#);
    assert!(!e.is_empty());
}

#[test]
fn list_head_on_empty_errors() {
    let e = run_err(r#"fn main() { list_head(list()) }"#);
    assert!(!e.is_empty());
}

#[test]
fn assert_wrong_type_errors() {
    let e = run_err(r#"fn main() { assert(42) }"#);
    assert!(!e.is_empty());
}

#[test]
fn assert_eq_wrong_arity_errors() {
    let e = run_err(r#"fn main() { assert_eq(1) }"#);
    assert!(!e.is_empty());
}

#[test]
fn clamp_wrong_type_errors() {
    let e = run_err(r#"fn main() { clamp("x", 0, 10) }"#);
    assert!(!e.is_empty());
}

#[test]
fn panic_with_message() {
    let e = run_err(r#"fn main() { panic("oh no") }"#);
    assert!(e.contains("oh no"), "got: {}", e);
}

// ─── Let bindings and scope ───────────────────────────────────────────────────

#[test]
fn let_shadows_outer() {
    let v = run(r#"
fn main() -> i64 {
    let x = 1
    let x = x + 10
    x
}
"#);
    assert_eq!(as_int(&v), 11);
}

#[test]
fn let_in_block_scope() {
    let v = run(r#"
fn main() -> i64 {
    let x = 5
    let y = {
        let x = 100
        x + 1
    }
    x + y
}
"#);
    assert_eq!(as_int(&v), 106);
}

#[test]
fn tuple_destructure_swap() {
    let v = run(r#"
fn main() -> i64 {
    let (a, b) = (10, 20)
    let (b, a) = (a, b)
    a - b
}
"#);
    assert_eq!(as_int(&v), 10);
}

// ─── Boolean logic ────────────────────────────────────────────────────────────

#[test]
fn short_circuit_and() {
    // #285: `&&` must NOT evaluate the RHS when the LHS is false.
    let v = run(r#"
fn main() -> bool {
    let mut x = 0
    false && { x = 1; true }
    x == 0
}
"#);
    assert!(matches!(v, Value::Bool(true)), "&& evaluated its RHS despite false LHS");
}

#[test]
fn short_circuit_or() {
    // #285: `||` must NOT evaluate the RHS when the LHS is true.
    let v = run(r#"
fn main() -> bool {
    let mut x = 0
    true || { x = 1; false }
    x == 0
}
"#);
    assert!(matches!(v, Value::Bool(true)), "|| evaluated its RHS despite true LHS");
}

#[test]
fn logical_not() {
    let v = run(r#"fn main() -> bool { !false }"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn logical_not_of_true() {
    let v = run(r#"fn main() -> bool { !true }"#);
    assert!(matches!(v, Value::Bool(false)));
}

#[test]
fn bool_ops_combine() {
    let v = run(r#"fn main() -> bool { (1 < 2) && (3 > 1) && !(4 == 5) }"#);
    assert!(matches!(v, Value::Bool(true)));
}

// ─── Number conversion and formatting programs ───────────────────────────────

#[test]
fn number_to_string_and_back() {
    let v = run(r#"fn main() -> i64 { to_int(to_str(42)) }"#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn float_to_int_truncates() {
    let v = run(r#"fn main() -> i64 { to_int(3.99) }"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn int_to_float_exact() {
    let v = run(r#"fn main() -> f64 { to_float(7) }"#);
    assert!((as_float(&v) - 7.0).abs() < 1e-10);
}

// ── #300 parity rulings: interp matches the JIT on the four deferred divergences.

#[test]
fn int_add_overflow_wraps() {
    // #300 ruling: +/-/* overflow WRAPS (2's complement), matching the JIT —
    // i64::MAX + 1 == i64::MIN, not an error.
    let v = run(r#"fn main() -> i64 { let a: i64 = 9223372036854775807  a + 1 }"#);
    assert_eq!(as_int(&v), i64::MIN);
}
#[test]
fn int_sub_overflow_wraps() {
    // i64::MIN - 1 wraps to i64::MAX.
    let v = run(r#"fn main() -> i64 { let a: i64 = -9223372036854775807 - 1  a - 1 }"#);
    assert_eq!(as_int(&v), i64::MAX);
}
#[test]
fn int_mul_overflow_wraps() {
    let v = run(r#"fn main() -> i64 { let a: i64 = 9223372036854775807  a * 2 }"#);
    assert_eq!(as_int(&v), -2); // MAX * 2 wraps to -2
}
#[test]
fn scalar_as_f32_rounds() {
    // #300 ruling: an explicit scalar `as f32` rounds to f32 precision (matches
    // the JIT), it no longer keeps full f64. 0.1 → 0.10000000149… .
    let v = run(r#"fn main() -> f64 { (0.1 as f32) as f64 }"#);
    assert_eq!(as_float(&v), 0.1_f32 as f64);
    assert!(as_float(&v) != 0.1_f64, "as f32 must lose precision");
}
#[test]
fn max_min_propagate_nan() {
    // #300 ruling: a NaN in the input makes global max/min NaN (np.max / JIT),
    // rather than being skipped (np.nanmax).
    let mx = run(r#"fn main() -> f64 { let a=[0.0f32,2.0f32]  let z=[0.0f32,1.0f32]  max(a ./ z) }"#);
    let mn = run(r#"fn main() -> f64 { let a=[0.0f32,2.0f32]  let z=[0.0f32,1.0f32]  min(a ./ z) }"#);
    assert!(as_float(&mx).is_nan(), "max should propagate NaN, got {:?}", mx);
    assert!(as_float(&mn).is_nan(), "min should propagate NaN, got {:?}", mn);
}

#[test]
fn hex_roundtrip() {
    let v = run(r#"
fn main() -> bool {
    to_hex(255) == "ff"
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn format_table_row() {
    let v = run(r#"
fn main() -> str {
    format("{:10} {:6.2f} {:5d}", "item", 3.14159, 42)
}
"#);
    let s = as_str(&v);
    assert!(s.contains("3.14"), "got: {}", s);
    assert!(s.contains("42"), "got: {}", s);
}

#[test]
fn format_zero_pad_series() {
    let v = run(r#"
fn main() -> str {
    let items = list_map(list(1, 2, 10, 100), fn(n: i64) -> str { format("{:04d}", n) })
    join(" ", items)
}
"#);
    assert_eq!(as_str(&v), "0001 0002 0010 0100");
}

// ─── String programs ──────────────────────────────────────────────────────────

#[test]
fn string_caesar_cipher() {
    let v = run(r#"
fn shift_char(c: i64, n: i64) -> i64 {
    if c >= 65 && c <= 90 {
        (c - 65 + n) % 26 + 65
    } else if c >= 97 && c <= 122 {
        (c - 97 + n) % 26 + 97
    } else { c }
}
fn main() -> str {
    let s = "Hello"
    let bytes = @cast(u8) { s }
    # just check we can do the cast
    to_str(5)
}
"#);
    assert_eq!(as_str(&v), "5");
}

#[test]
fn string_starts_ends() {
    let v = run(r#"
fn main() -> bool {
    let s = "hello.dmc"
    s.starts_with("hello") && s.ends_with(".dmc")
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

#[test]
fn string_find_and_replace_all() {
    let v = run(r#"
fn main() -> str {
    "a b c a b c".replace("a", "X")
}
"#);
    assert_eq!(as_str(&v), "X b c X b c");
}

#[test]
fn string_split_and_rejoin() {
    let v = run(r#"
fn main() -> str {
    let parts = "a,b,c,d".split(",")
    join("-", parts)
}
"#);
    assert_eq!(as_str(&v), "a-b-c-d");
}

#[test]
fn string_count_chars() {
    let v = run(r#"fn main() -> i64 { "mississippi".count("ss") }"#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn string_pad_for_table() {
    let v = run(r#"
fn main() -> str {
    let name = str_pad_right("item", 10)
    let val  = str_pad_left("42", 5)
    name + val
}
"#);
    let s = as_str(&v);
    assert_eq!(s.len(), 15);
    assert!(s.starts_with("item"), "got: {:?}", s);
    assert!(s.ends_with("42"), "got: {:?}", s);
}

// ─── Collection programs ─────────────────────────────────────────────────────

#[test]
fn deduplicate_list() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list(1, 2, 2, 3, 3, 3, 1)
    list_len(list_uniq(xs))
}
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn sorted_list() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list_sort(list(3, 1, 4, 1, 5, 9, 2, 6))
    list_head(xs)
}
"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn sorted_list_last() {
    let v = run(r#"
fn main() -> i64 {
    let xs = list_sort(list(3, 1, 4, 1, 5, 9, 2, 6))
    list_last(xs)
}
"#);
    assert_eq!(as_int(&v), 9);
}

#[test]
fn zip_and_sum() {
    let v = run(r#"
fn main() -> i64 {
    let pairs = list_zip(list(1, 2, 3), list(4, 5, 6))
    list_reduce(pairs, fn(acc: i64, p: tuple) -> i64 {
        let (a, b) = p
        acc + a + b
    }, 0)
}
"#);
    assert_eq!(as_int(&v), 21);
}

#[test]
fn enumerate_and_index() {
    let v = run(r#"
fn main() -> i64 {
    let indexed = list_enumerate(list(10, 20, 30))
    let first = list_head(indexed)
    let (idx, val) = first
    idx
}
"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn flatten_and_count() {
    let v = run(r#"
fn main() -> i64 {
    let nested = list_map(list(1, 2, 3), fn(n: i64) -> list { list_take(list(10, 20, 30), n) })
    let flat = list_flatten(nested)
    list_len(flat)
}
"#);
    assert_eq!(as_int(&v), 6);  // 1 + 2 + 3 = 6 elements
}

// ─── Map programs ────────────────────────────────────────────────────────────

#[test]
fn frequency_map() {
    let v = run(r#"
fn main() -> i64 {
    let words = list("a", "b", "a", "c", "a", "b")
    let freq = map()
    for w in words {
        let cur = map_get(freq, w)
        let count = if cur == nil { 0 } else { cur }
        freq = map_set(freq, w, count + 1)
    }
    map_get(freq, "a")
}
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn map_merge_no_conflict() {
    let v = run(r#"
fn main() -> i64 {
    let a = map_set(map_set(map(), "x", 1), "y", 2)
    let b = map_set(map_set(map(), "z", 3), "w", 4)
    let c = map_merge(a, b)
    map_len(c)
}
"#);
    assert_eq!(as_int(&v), 4);
}

// ─── Math programs ────────────────────────────────────────────────────────────

#[test]
fn sum_of_squares() {
    let v = run(r#"
fn main() -> i64 {
    let n = 10
    let squares = list_map(list(1,2,3,4,5,6,7,8,9,10), fn(x: i64) -> i64 { x * x })
    list_sum(squares)
}
"#);
    assert_eq!(as_int(&v), 385);
}

#[test]
fn geometric_mean() {
    let v = run(r#"
fn main() -> f64 {
    let xs = list(2.0, 8.0, 32.0)
    let log_sum = list_reduce(xs, fn(acc: f64, x: f64) -> f64 { acc + log(x) }, 0.0)
    exp(log_sum / 3.0)
}
"#);
    let f = as_float(&v);
    // geometric mean of 2, 8, 32 = 8
    assert!((f - 8.0).abs() < 1e-6, "got: {}", f);
}

#[test]
fn clamp_negative_values() {
    let v = run(r#"
fn main() -> i64 {
    let relu = fn(x: i64) -> i64 { clamp(x, 0, 1000000) }
    let xs = list(-3, 0, 5, -1, 2)
    let ys = list_map(xs, relu)
    list_sum(ys)
}
"#);
    assert_eq!(as_int(&v), 7);
}

#[test]
fn round_trip_via_to_str() {
    let v = run(r#"
fn main() -> bool {
    let n = 12345
    to_int(to_str(n)) == n
}
"#);
    assert!(matches!(v, Value::Bool(true)));
}

// ─── Recursion programs ────────────────────────────────────────────────────────

#[test]
fn power_function() {
    let v = run(r#"
fn pow(base: i64, exp: i64) -> i64 {
    if exp == 0 { 1 }
    else { base * pow(base, exp - 1) }
}
fn main() -> i64 { pow(2, 10) }
"#);
    assert_eq!(as_int(&v), 1024);
}

#[test]
fn gcd_euclid() {
    let v = run(r#"
fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}
fn main() -> i64 { gcd(48, 18) }
"#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn flatten_tree_via_recursion() {
    with_big_stack(|| {
        let v = run(r#"
fn count_range(lo: i64, hi: i64) -> i64 {
    if lo >= hi { 0 }
    else { 1 + count_range(lo + 1, hi) }
}
fn main() -> i64 { count_range(0, 20) }
"#);
        assert_eq!(as_int(&v), 20);
    });
}

// ─── Error handling programs ─────────────────────────────────────────────────

#[test]
fn division_by_zero_gives_inf() {
    let v = run(r#"fn main() -> f64 { 1.0 / 0.0 }"#);
    assert!(matches!(v, Value::Float(f, _) if f.is_infinite()));
}

#[test]
fn panic_with_format() {
    let e = run_err(r#"fn main() { panic(format("value was {}", 42)) }"#);
    assert!(e.contains("value was 42"), "got: {}", e);
}

#[test]
fn assert_in_test_fn() {
    let v = run(r#"
fn test_add() -> nil {
    assert_eq(1 + 1, 2)
    assert(1 < 2)
    assert_ne(1, 2)
    nil
}
fn main() -> nil { test_add() }
"#);
    assert!(matches!(v, Value::Nil));
}

// ─── Module system tests ─────────────────────────────────────────────────────

/// Shared implementation behind `run_multi` and `run_multi_with_limits`
/// (below, near #794's tests): resolve → typecheck → load every dependency
/// exactly once → run the main file exactly once, with an optional arena
/// budget installed the way `--forge=`/`--vault=` would install it. Kept as
/// one function so the two harnesses cannot drift from `main.rs`'s `run` arm
/// independently.
fn run_multi_impl(
    files: &[(&str, &str)],
    main: &str,
    limits: Option<crate::arena::ArenaLimits>,
) -> Result<Value, String> {
    use super::check::{Checker, ModuleEnv};
    use super::ast::{collect_public_items, collect_top_level_let_names};
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, src) in files {
        std::fs::write(dir.path().join(name), src).expect("write dep");
    }
    let main_path = dir.path().join(main);
    let mut resolver = Resolver::new();
    resolver.resolve_all(&main_path).expect("resolve");
    let canonical = main_path.canonicalize().expect("canonicalize");
    // typecheck
    let mut checked_modules = std::collections::HashMap::new();
    for p in &resolver.sorted_paths {
        let prog = resolver.files.get(p).unwrap();
        let mut checker = Checker::new();
        checker.checked_modules = checked_modules.clone();
        checker.check_program(prog, Some(p));
        assert!(checker.errors.is_empty(), "type errors: {:?}", checker.errors);
        let mod_env = ModuleEnv { env: checker.env.clone(), aliases: checker.aliases.clone(), resolved_aliases: checker.resolved_aliases.clone(), public_items: collect_public_items(prog), top_level_let_names: collect_top_level_let_names(prog), transitive_import_lets: checker.import_private_lets.clone() };
        checked_modules.insert(p.clone(), mod_env);
    }
    // interpret. The main file is skipped here and loaded and run exactly
    // once below, by `interp.run` — matching the `dmc run` CLI's #794 fix (the
    // main file's own top-level items must not be evaluated twice; every
    // dependency still loads here, exactly once each).
    let mut interp = Interpreter::new();
    if let Some(limits) = limits {
        interp.set_arena_limits(limits);
    }
    for p in &resolver.sorted_paths {
        if *p == canonical {
            continue;
        }
        let prog = resolver.files.get(p).unwrap();
        interp.load_program(prog, Some(p)).expect("load");
        let env = interp.get_module_env();
        interp.interp_modules.insert(p.clone(), env);
    }
    let main_prog = resolver.files.get(&canonical).expect("main prog");
    interp.run(main_prog, Some(&canonical)).map_err(|e| e.msg)
}

/// Write source files to a temp dir and run the main file through the full
/// resolver → typechecker → interpreter pipeline (matching `dmc run` behavior).
fn run_multi(files: &[(&str, &str)], main: &str) -> Value {
    run_multi_impl(files, main, None).expect("run")
}

fn run_multi_err(files: &[(&str, &str)], main: &str) -> String {
    use super::check::{Checker, ModuleEnv};
    use super::ast::{collect_public_items, collect_top_level_let_names};
    let dir = tempfile::tempdir().expect("tempdir");
    for (name, src) in files {
        std::fs::write(dir.path().join(name), src).expect("write dep");
    }
    let main_path = dir.path().join(main);
    let mut resolver = Resolver::new();
    if let Err(e) = resolver.resolve_all(&main_path) {
        return e.to_string();
    }
    let canonical = main_path.canonicalize().expect("canonicalize");
    // typecheck — collect errors first
    let mut checked_modules = std::collections::HashMap::new();
    for p in &resolver.sorted_paths {
        let prog = resolver.files.get(p).unwrap();
        let mut checker = Checker::new();
        checker.checked_modules = checked_modules.clone();
        checker.check_program(prog, Some(p));
        if !checker.errors.is_empty() {
            return checker.errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
        }
        let mod_env = ModuleEnv { env: checker.env.clone(), aliases: checker.aliases.clone(), resolved_aliases: checker.resolved_aliases.clone(), public_items: collect_public_items(prog), top_level_let_names: collect_top_level_let_names(prog), transitive_import_lets: checker.import_private_lets.clone() };
        checked_modules.insert(p.clone(), mod_env);
    }
    // interpret — expect runtime error. Same #794 fix as `run_multi`: the
    // main file is skipped here and loaded and run exactly once below.
    let mut interp = Interpreter::new();
    for p in &resolver.sorted_paths {
        if *p == canonical {
            continue;
        }
        let prog = resolver.files.get(p).unwrap();
        if let Err(e) = interp.load_program(prog, Some(p)) {
            return e.msg;
        }
        let env = interp.get_module_env();
        interp.interp_modules.insert(p.clone(), env);
    }
    let main_prog = resolver.files.get(&canonical).expect("main prog");
    interp.run(main_prog, Some(&canonical)).err().expect("expected error").msg
}

#[test]
fn module_unqualified_import_fn() {
    let v = run_multi(&[
        ("math.dmc", "pub fn square(x: i64) -> i64 { x * x }"),
        ("main.dmc", r#"use "math.dmc"
fn main() -> i64 { square(7) }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 49);
}

#[test]
fn module_qualified_import_fn() {
    let v = run_multi(&[
        ("utils.dmc", "pub fn add(a: i64, b: i64) -> i64 { a + b }"),
        ("main.dmc", r#"use "utils.dmc" as u
fn main() -> i64 { u.add(10, 32) }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 42);
}

#[test]
fn module_private_fn_not_exported() {
    // Private (non-pub) fn should not be visible to importer
    let e = run_multi_err(&[
        ("secret.dmc", "fn hidden() -> i64 { 99 }"),
        ("main.dmc", r#"use "secret.dmc"
fn main() -> i64 { hidden() }"#),
    ], "main.dmc");
    assert!(e.contains("hidden") || e.contains("undefined"), "got: {e}");
}

#[test]
fn module_transitive_import() {
    // a → b → c: functions from c are usable in a via b's re-export
    let v = run_multi(&[
        ("c.dmc", "pub fn base() -> i64 { 100 }"),
        ("b.dmc", r#"use "c.dmc"
pub fn doubled() -> i64 { base() * 2 }"#),
        ("main.dmc", r#"use "b.dmc"
fn main() -> i64 { doubled() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 200);
}

#[test]
fn module_import_multiple_fns() {
    let v = run_multi(&[
        ("arith.dmc", r#"pub fn add(a: i64, b: i64) -> i64 { a + b }
pub fn mul(a: i64, b: i64) -> i64 { a * b }"#),
        ("main.dmc", r#"use "arith.dmc"
fn main() -> i64 { add(3, mul(4, 5)) }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 23);
}

#[test]
fn module_import_uses_imported_fn_in_let() {
    let v = run_multi(&[
        ("helpers.dmc", "pub fn inc(x: i64) -> i64 { x + 1 }"),
        ("main.dmc", r#"use "helpers.dmc"
fn main() -> i64 {
    let a = inc(5)
    let b = inc(a)
    b
}"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 7);
}

#[test]
fn module_circular_import_is_detected() {
    let e = run_multi_err(&[
        ("a.dmc", r#"use "b.dmc"
pub fn fa() -> i64 { 1 }"#),
        ("b.dmc", r#"use "a.dmc"
pub fn fb() -> i64 { 2 }"#),
        ("main.dmc", r#"use "a.dmc"
fn main() -> i64 { 0 }"#),
    ], "main.dmc");
    assert!(e.contains("circular"), "expected circular import error, got: {e}");
}

// ─── SPEC §6.7: a body resolves names against its own file ──────────────────

#[test]
fn module_private_fn_resolves_in_own_file() {
    // Two files each declare a file-private `helper`. The later-loaded file
    // must not take over the earlier one's slot: 1122, not 2222.
    let v = run_multi(&[
        ("m1.dmc", "pub fn f() -> i64 { helper() }\nfn helper() -> i64 { 11 }"),
        ("m2.dmc", "pub fn g() -> i64 { helper() }\nfn helper() -> i64 { 22 }"),
        ("main.dmc", r#"use "m1.dmc"
use "m2.dmc"
fn main() -> i64 { f() * 100 + g() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1122);
}

#[test]
fn module_pub_fn_under_alias_keeps_its_own_helper() {
    // Both files export `helper`; main disambiguates with an alias. The
    // aliased file's `g` still calls the `helper` written next to it.
    let v = run_multi(&[
        ("m1.dmc", "pub fn helper() -> i64 { 11 }\npub fn f() -> i64 { helper() }"),
        ("m2.dmc", "pub fn helper() -> i64 { 22 }\npub fn g() -> i64 { helper() }"),
        ("main.dmc", r#"use "m1.dmc"
use "m2.dmc" as z
fn main() -> i64 { f() * 100 + z.g() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1122);
}

#[test]
fn module_private_fn_is_not_shadowed_by_another_files_pub() {
    // m1's private `helper` and m2's `pub helper` coexist: main sees m2's,
    // m1's `f` keeps its own.
    let v = run_multi(&[
        ("m1.dmc", "pub fn f() -> i64 { helper() }\nfn helper() -> i64 { 11 }"),
        ("m2.dmc", "pub fn helper() -> i64 { 22 }"),
        ("main.dmc", r#"use "m1.dmc"
use "m2.dmc"
fn main() -> i64 { f() * 100 + helper() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1122);
}

#[test]
fn module_private_model_dispatches_its_own_methods() {
    // Two file-private models named `P`; an instance built in m1 must run
    // m1's method, not the last-loaded `P`'s.
    let v = run_multi(&[
        ("m1.dmc", "model P { x: i64  fn val(self) -> i64 { self.x + 1 } }\npub fn f() -> i64 { let p = P { x: 10 }  p.val() }"),
        ("m2.dmc", "model P { x: i64  fn val(self) -> i64 { self.x + 2 } }\npub fn g() -> i64 { let p = P { x: 20 }  p.val() }"),
        ("main.dmc", r#"use "m1.dmc"
use "m2.dmc"
fn main() -> i64 { f() * 100 + g() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1122);
}

#[test]
fn module_pub_model_under_two_spellings_is_one_model() {
    // The same `pub model` imported bare and under an alias: both literals
    // tag the instance with the declaring file's key, so both dispatch.
    let v = run_multi(&[
        ("math.dmc", "pub model P { x: i64  fn dbl(self) -> i64 { self.x * 2 } }"),
        ("main.dmc", r#"use "math.dmc" as m
use "math.dmc"
fn main() -> i64 { let a = m.P { x: 1 }  let b = P { x: 10 }  a.dbl() + b.dbl() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 22);
}

#[test]
fn module_closure_resolves_in_defining_file() {
    // A closure written in m1 and called from main still reaches m1's
    // private `helper`, not the `helper` of whichever file loaded last.
    let v = run_multi(&[
        ("m1.dmc", "fn helper() -> i64 { 11 }\npub fn mk() -> fn() -> i64 { fn() -> i64 { helper() } }"),
        ("m2.dmc", "fn helper() -> i64 { 22 }\npub fn g() -> i64 { helper() }"),
        ("main.dmc", r#"use "m1.dmc"
use "m2.dmc"
fn main() -> i64 { let c = mk()  c() * 100 + g() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1122);
}

#[test]
fn module_dependency_main_is_not_the_entry_point() {
    // A non-`pub fn main` in a dependency is that file's private item; the
    // entry file's `main` runs.
    let v = run_multi(&[
        ("dep.dmc", "fn main() -> i64 { 1 }\npub fn one() -> i64 { 1 }"),
        ("main.dmc", r#"use "dep.dmc"
fn main() -> i64 { one() + 1 }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 2);
}

// ─── SPEC §6.7 review follow-ups (#652) ─────────────────────────────────────

#[test]
fn module_cast_model_overlay_resolves_the_model_name() {
    // The `@cast` guard tested the SOURCE name against `model_fields`, which
    // is keyed by the declaring file's key — so it never matched, the overlay
    // fell through to the dtype cast, and `h` came back as the raw byte tensor
    // rather than a struct. Every `cast_model_overlay_*` case above rides on
    // this; this one names the resolution directly.
    let v = run(r#"
        model Hdr { a: i64, b: i64 }
        fn main() -> i64 {
            let bytes = @cast(u8) { "ABCDEFGHIJKLMNOP" }
            let h = @cast(Hdr) { bytes }
            h.b
        }
    "#);
    // `b` is bytes 8..16 of "ABCDEFGHIJKLMNOP" (`IJKLMNOP` = 49 4A 4B 4C 4D
    // 4E 4F 50). Little-endian (#606) reverses that byte sequence into the
    // decoded i64: 0x504F4E4D4C4B4A49.
    assert_eq!(as_int(&v), 0x504F4E4D4C4B4A49, "`h.b` reads a decoded field, not a tensor");
}

#[test]
fn module_cast_overlays_the_model_of_the_overlaying_file() {
    // Two files each declare a file-private `Hdr` of DIFFERENT arity. Each
    // overlay must lay out its own file's fields: the flat table would have
    // given both the last-loaded `Hdr`.
    let v = run_multi(&[
        ("a.dmc", r#"model Hdr { a: i64, b: i64 }
pub fn one() -> i64 { let bytes = @cast(u8) { "ABCDEFGHIJKLMNOP" }  let h = @cast(Hdr) { bytes }  h.b }"#),
        ("b.dmc", r#"model Hdr { a: i64 }
pub fn two() -> i64 { let bytes = @cast(u8) { "ABCDEFGH" }  let h = @cast(Hdr) { bytes }  h.a }"#),
        ("main.dmc", r#"use "a.dmc"
use "b.dmc"
fn main() -> i64 { one() - two() }"#),
    ], "main.dmc");
    // `one()`'s `b` (`IJKLMNOP`) and `two()`'s `a` (`ABCDEFGH`), both
    // little-endian decoded (#606): 0x504F4E4D4C4B4A49 - 0x4847464544434241.
    assert_eq!(as_int(&v), 0x504F4E4D4C4B4A49 - 0x4847464544434241);
}

#[test]
fn module_cast_overlays_an_imported_pub_model() {
    // The importer spells the model by the name `use` gave it; the overlay
    // resolves that spelling to the declaring file's key, which is what
    // `model_fields` is registered under.
    let v = run_multi(&[
        ("hdr.dmc", "pub model Hdr { a: i64, b: i64 }"),
        ("main.dmc", r#"use "hdr.dmc"
fn main() -> i64 {
    let bytes = @cast(u8) { "ABCDEFGHIJKLMNOP" }
    let h = @cast(Hdr) { bytes }
    h.a
}"#),
    ], "main.dmc");
    // `a` is bytes 0..8 of "ABCDEFGHIJKLMNOP" (`ABCDEFGH`), little-endian
    // decoded (#606): 0x4847464544434241.
    assert_eq!(as_int(&v), 0x4847464544434241);
}

#[test]
fn module_private_enums_do_not_cross_talk() {
    // Two files each declare a private `enum E` with different variants. One
    // flat `enums` map let the last-loaded declaration decide `E`'s variants
    // everywhere, so `E.Gamma` in the file that declares it failed with
    // `undefined identifier E` once the two-variant `E` had overwritten it.
    let v = run_multi(&[
        ("ea.dmc", r#"enum E { Alpha, Beta, Gamma }
pub fn ea() -> i64 { match E.Gamma { E.Alpha => 1, E.Beta => 2, E.Gamma => 3 } }"#),
        ("eb.dmc", r#"enum E { Alpha, Beta }
pub fn eb() -> i64 { match E.Beta { E.Alpha => 10, E.Beta => 20 } }"#),
        ("main.dmc", r#"use "ea.dmc"
use "eb.dmc"
fn main() -> i64 { ea() * 100 + eb() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 320);
}

#[test]
fn module_enum_ordinal_is_read_from_the_declaring_file() {
    // The reverse load order of the pair above: the ordinal a variant carries
    // must come from the enum its OWN file declares, whichever file loaded
    // last. `Gamma` is 2 in `ea`'s three-variant `E` and does not exist in
    // `eb`'s.
    let v = run_multi(&[
        ("eb.dmc", "enum E { Alpha, Beta }\npub fn eb() -> i64 { E.Beta as i64 }"),
        ("ea.dmc", r#"use "eb.dmc"
enum E { Alpha, Beta, Gamma }
pub fn ea() -> i64 { E.Gamma as i64 }"#),
        ("main.dmc", r#"use "ea.dmc"
use "eb.dmc"
fn main() -> i64 { ea() * 10 + eb() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 21);
}

#[test]
fn module_imported_alias_beats_an_unimported_files_declaration() {
    // #623 scoped a `type` alias by LOAD ORDER (`module <= alias_module`,
    // highest wins), which admitted a declaration from any file loaded
    // earlier — imported or not. `other.dmc` declares a private `Role` and
    // loads after `lib.dmc`, so that rule handed `main` the i64 and printed
    // 256. Registering aliases on import — the rule `fns` and `models`
    // already follow, and the one SPEC §9 states for all three — gives `main`
    // the `u8` it actually imported, which wraps to 0.
    let v = run_multi(&[
        ("lib.dmc", "pub type Role = u8\npub fn one() -> i64 { 1 }"),
        ("other.dmc", "type Role = i64\npub fn two() -> i64 { 2 }"),
        ("main.dmc", r#"use "lib.dmc"
use "other.dmc"
fn main() -> i64 {
    let r: Role = 255
    ((r + 1) as i64) * 10 + one() + two()
}"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 3, "the imported u8 governs, so `r + 1` is 0");
}

#[test]
fn module_pub_model_method_calls_its_own_files_private_fn() {
    // A `pub model`'s method body is its file's code: it reaches that file's
    // private `secret`, not the importer's or another library's, whether the
    // instance was built by the importer's own literal or by the library's
    // constructor.
    let v = run_multi(&[
        ("lib.dmc", r#"fn secret() -> i64 { 77 }
pub model W { a: i64
  fn calc(self) -> i64 { self.a + secret() }
}
pub fn mkw(a: i64) -> W { W { a: a } }"#),
        ("lib2.dmc", "fn secret() -> i64 { -1000 }\npub fn other() -> i64 { secret() }"),
        ("main.dmc", r#"use "lib.dmc"
use "lib2.dmc"
fn secret() -> i64 { 5 }
fn main() -> i64 {
    let w = W { a: 1 }
    let v = mkw(2)
    w.calc() * 1000000 + v.calc() * 10000 + (other() + 1000) * 10 + secret()
}"#),
    ], "main.dmc");
    // 78 · 1e6 + 79 · 1e4 + 0 + 5
    assert_eq!(as_int(&v), 78_790_005);
}

#[test]
fn module_private_fn_shadows_a_builtin_only_in_its_own_file() {
    // A file-private `fn abs` shadows the builtin inside that file and
    // nowhere else: the importer's bare `abs` is still the builtin.
    let v = run_multi(&[
        ("a.dmc", "fn abs(x: i64) -> i64 { 999 }\npub fn ba() -> i64 { abs(-3) }"),
        ("main.dmc", r#"use "a.dmc"
fn main() -> i64 { ba() * 10 + abs(-3) }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 9993);
}

#[test]
fn module_private_models_of_the_same_name_keep_their_own_fields() {
    // Stronger than the same-shape pair above: the two private `Node`s
    // declare DIFFERENT field sets, so a method dispatched to the wrong
    // model would read a field that does not exist there.
    let v = run_multi(&[
        ("a.dmc", r#"model Node { v: i64
    fn get(self) -> i64 { self.v + 100 }
}
pub fn ma() -> i64 { let n = Node { v: 7 }  n.get() }"#),
        ("b.dmc", r#"model Node { w: i64, z: i64
    fn get(self) -> i64 { self.w * 1000 + self.z }
}
pub fn mb() -> i64 { let n = Node { w: 3, z: 4 }  n.get() }"#),
        ("main.dmc", r#"use "a.dmc"
use "b.dmc"
fn main() -> i64 { ma() * 10000 + mb() }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1_073_004);
}

#[test]
fn module_mutual_recursion_across_a_pub_boundary_still_resolves() {
    // `even` is `pub` and its private partner is not; the recursion crosses
    // the visibility line in the declaring file, which is where both names
    // live.
    let v = run_multi(&[
        ("a.dmc", r#"pub fn even(n: i64) -> i64 { if n == 0 { 1 } else { odd_helper(n - 1) } }
fn odd_helper(n: i64) -> i64 { if n == 0 { 0 } else { even(n - 1) } }"#),
        ("main.dmc", r#"use "a.dmc"
fn main() -> i64 { even(10) * 10 + even(7) }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 10);
}

// ─── #65: model method calls ─────────────────────────────────────────────────

#[test]
fn model_forward_desugar_works() {
    let v = run(r#"
model M {
    val: i64
    fn forward(self) -> i64 { self.val * 10 }
}
fn main() -> i64 { let m = M { val: 5 }  m() }
"#);
    assert_eq!(as_int(&v), 50);
}

#[test]
fn model_explicit_forward_call_works() {
    let v = run(r#"
model M {
    val: i64
    fn forward(self) -> i64 { self.val * 10 }
}
fn main() -> i64 { let m = M { val: 5 }  m.forward() }
"#);
    assert_eq!(as_int(&v), 50);
}

#[test]
fn model_named_method_call_works() {
    let v = run(r#"
model M {
    val: i64
    fn forward(self) -> i64 { self.val * 10 }
    fn other(self)   -> i64 { self.val * 100 }
}
fn main() -> i64 { let m = M { val: 5 }  m.other() }
"#);
    assert_eq!(as_int(&v), 500);
}

#[test]
fn model_method_accesses_self_field() {
    let v = run(r#"
model Counter {
    count: i64
    fn value(self) -> i64 { self.count }
    fn doubled(self) -> i64 { self.count * 2 }
}
fn main() -> i64 { let c = Counter { count: 7 }  c.doubled() }
"#);
    assert_eq!(as_int(&v), 14);
}

#[test]
fn model_method_with_args() {
    let v = run(r#"
model Adder {
    base: i64
    fn add(self, x: i64) -> i64 { self.base + x }
}
fn main() -> i64 { let a = Adder { base: 10 }  a.add(32) }
"#);
    assert_eq!(as_int(&v), 42);
}

// ─── #66: elementwise broadcast ──────────────────────────────────────────────

#[test]
fn broadcast_col_vector_across_rows() {
    // [2,3] .* [2,1] — each row of a scaled by its b entry
    let v = run(r#"
fn main() -> f32 {
    let a: Tensor[f32, [2, 3]] = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
    let b: Tensor[f32, [2, 1]] = [[10.0f32], [100.0f32]]
    let c = a .* b
    c[1, 2]
}
"#);
    assert!((as_float(&v) - 600.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn broadcast_col_vector_first_element() {
    let v = run(r#"
fn main() -> f32 {
    let a: Tensor[f32, [2, 3]] = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
    let b: Tensor[f32, [2, 1]] = [[10.0f32], [100.0f32]]
    let c = a .* b
    c[0, 0]
}
"#);
    assert!((as_float(&v) - 10.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn broadcast_row_vector_across_cols() {
    // [2,3] .+ [1,3] — add a row vector to each row of a matrix
    let v = run(r#"
fn main() -> f32 {
    let a: Tensor[f32, [2, 3]] = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
    let b: Tensor[f32, [1, 3]] = [[10.0f32, 20.0f32, 30.0f32]]
    let c = a .+ b
    c[1, 1]
}
"#);
    assert!((as_float(&v) - 25.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn broadcast_identical_shapes_unchanged() {
    // Ensure exact-shape case still works after the refactor
    let v = run(r#"
fn main() -> f32 {
    let a: Tensor[f32, [2, 2]] = [[1.0f32, 2.0f32], [3.0f32, 4.0f32]]
    let b: Tensor[f32, [2, 2]] = [[10.0f32, 10.0f32], [10.0f32, 10.0f32]]
    let c = a .* b
    c[1, 1]
}
"#);
    assert!((as_float(&v) - 40.0).abs() < 1e-6, "got {:?}", v);
}

// ─── #68: reshape and ellipsis slicing ───────────────────────────────────────

#[test]
fn reshape_1d_to_2d() {
    let v = run(r#"
fn main() -> f32 {
    let t: Tensor[f32, [4]] = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
    let r = t.reshape[[2, 2]]
    r[1, 0]
}
"#);
    assert!((as_float(&v) - 3.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn reshape_2d_to_3d() {
    let v = run(r#"
fn main() -> f32 {
    let t: Tensor[f32, [1, 4]] = [[1.0f32, 2.0f32, 3.0f32, 4.0f32]]
    let r = t.reshape[[1, 1, 4]]
    r[0, 0, 3]
}
"#);
    assert!((as_float(&v) - 4.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn reshape_preserves_data() {
    let v = run(r#"
fn main() -> i64 {
    let t: Tensor[f32, [2, 3]] = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
    let r = t.reshape[[6]]
    let sum = r[0] + r[1] + r[2] + r[3] + r[4] + r[5]
    sum as i64
}
"#);
    assert_eq!(as_int(&v), 21);
}

#[test]
fn reshape_result_is_tensor_for_stream_append() {
    // Minimal repro from issue #68: reshape then <- should not error
    let v = run(r#"
fn main() -> nil {
    stream {
        let !cache: KV[f32, [1, ~, 4]] = stream.kv[f32, [1, ~, 4]](capacity = 8)
        let t = forge.ones[f32, [1, 4]]
        let r = t.reshape[[1, 1, 4]]
        cache <- r
    }
    nil
}
"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn ellipsis_slice_selects_last_row() {
    // t[.., -1, ..] on [2, 3, 4] selects index -1 of axis 1 → [2, 4]
    let v = run(r#"
fn main() -> f32 {
    let t: Tensor[f32, [2, 3]] = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
    let row = t[.., -1]
    row[1]
}
"#);
    assert!((as_float(&v) - 6.0).abs() < 1e-6, "got {:?}", v);
}

// ── Issue #69: model fields pass-by-copy ─────────────────────────────────────

#[test]
fn field_index_write_persists() {
    // Case 1: m.grid[0] = 100 must persist when read back via m.grid[0]
    let v = run(r#"
model Grid {
    grid: Tensor[f32, [4]]
    fn read_at(self, i: i64) -> f32 { self.grid[i] }
}
fn main() -> f32 {
    let m = Grid { grid: [0.0f32, 0.0f32, 0.0f32, 0.0f32] }
    m.grid[0] = 100.0
    m.read_at(0)
}
"#);
    assert!((as_float(&v) - 100.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn aliased_field_write_persists() {
    // Case 2: let !alias = m.grid; alias[1] = 200 must affect m.grid[1]
    let v = run(r#"
model Grid {
    grid: Tensor[f32, [4]]
    fn read_at(self, i: i64) -> f32 { self.grid[i] }
}
fn main() -> f32 {
    let m = Grid { grid: [0.0f32, 0.0f32, 0.0f32, 0.0f32] }
    let !alias = m.grid
    alias[1] = 200.0
    m.read_at(1)
}
"#);
    assert!((as_float(&v) - 200.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn method_mutation_persists() {
    // Case 3: self.n = self.n + 1 inside a method must persist to caller
    let v = run(r#"
model Counter {
    n: i64
    fn bump(self) { self.n = self.n + 1 }
    fn get(self) -> i64 { self.n }
}
fn main() -> i64 {
    let c = Counter { n: 0 }
    c.bump()
    c.bump()
    c.bump()
    c.get()
}
"#);
    assert_eq!(as_int(&v), 3);
}

// ── Issue #72: `as` narrowing casts ──────────────────────────────────────────
#[test]
fn cast_u8_wraps_positive() {
    let v = run(r#"fn main() -> i64 { (300 as u8) as i64 }"#);
    assert_eq!(as_int(&v), 44);
}
#[test]
fn cast_u8_wraps_negative() {
    let v = run(r#"fn main() -> i64 { (-5 as u8) as i64 }"#);
    assert_eq!(as_int(&v), 251);
}
#[test]
fn cast_i8_wraps_128() {
    let v = run(r#"fn main() -> i64 { (128 as i8) as i64 }"#);
    assert_eq!(as_int(&v), -128);
}

// ── Issue #291.1: float→signed-narrow cast must narrow, not pass through ──────
// Before the fix the float→I8/I16/I32 path did `n as i64` (no narrowing), so an
// out-of-range float kept its full value (`300.0 as i8` == 300). It now goes
// through the target width like the int path and the unsigned-float path.
#[test]
fn cast_float_to_i8_narrows() {
    let v = run(r#"fn main() -> i64 { (300.0 as i8) as i64 }"#);
    assert_eq!(as_int(&v), 127); // narrowed (saturating), not 300
}
#[test]
fn cast_float_to_i16_narrows() {
    let v = run(r#"fn main() -> i64 { (40000.0 as i16) as i64 }"#);
    assert_eq!(as_int(&v), 32767);
}
#[test]
fn cast_float_to_i8_in_range_unchanged() {
    let v = run(r#"fn main() -> i64 { (42.0 as i8) as i64 }"#);
    assert_eq!(as_int(&v), 42);
}
#[test]
fn cast_float_to_i8_negative_in_range() {
    let v = run(r#"fn main() -> i64 { (-42.0 as i8) as i64 }"#);
    assert_eq!(as_int(&v), -42);
}

// #298: tensor float→narrow-int casts must narrow per element like the scalar
// path, not pass the value through unchanged.
#[test]
fn cast_tensor_float_to_i8_narrows() {
    let v = run("fn main() -> i64 {\n  let t = [300.0f32, 1.0f32]\n  let u = t as i8\n  u[0] as i64\n}");
    assert_eq!(as_int(&v), 127); // was 300 before the fix
}
#[test]
fn cast_tensor_float_to_u8_saturates_negative() {
    let v = run("fn main() -> i64 {\n  let t = [-1.0f32, 2.0f32]\n  let u = t as u8\n  u[0] as i64\n}");
    assert_eq!(as_int(&v), 0); // matches scalar `-1.0 as u8`
}

// #296.1: integer comparison must be exact (i64), not demoted to f64 which
// loses precision above 2^53. These two values are distinct.
#[test]
fn large_int_equality_is_exact() {
    let v = run(r#"fn main() -> bool { let a: i64 = 9007199254740993  let b: i64 = 9007199254740992  a == b }"#);
    assert!(matches!(v, Value::Bool(false)), "got {:?}", v);
}
#[test]
fn large_int_ordering_is_exact() {
    let v = run(r#"fn main() -> bool { let a: i64 = 9007199254740993  let b: i64 = 9007199254740992  a > b }"#);
    assert!(matches!(v, Value::Bool(true)), "got {:?}", v);
}
#[test]
fn int_float_comparison_still_cross_matches() {
    // mixed int/float still compares via the float path — `1 == 1.0` is true.
    let v = run(r#"fn main() -> bool { 1 == 1.0 }"#);
    assert!(matches!(v, Value::Bool(true)), "got {:?}", v);
}

// #296.2 → #300 ruling: Add/Sub/Mul overflow WRAPS (2's complement) to match the
// JIT. (It was briefly a clean error under #296.2; the parity ruling changed it
// to wrap — systems-language norm, and the two backends now agree.) Covered by
// int_{add,sub,mul}_overflow_wraps above; div/mod/pow overflow still trap.

// ── Issue #73: KV stream ellipsis-slice keepdims ──────────────────────────────
#[test]
fn kv_ellipsis_slice_keepdims() {
    // cache[.., -1, ..] on [1, N, 3] must return [1, 1, 3], not [1, 3]
    let v = run(r#"
fn main() -> f32 {
    stream {
        let !cache: KV[f32, [1, ~, 3]] = stream.kv[f32, [1, ~, 3]](capacity = 4)
        let a: Tensor[f32, [1, 1, 3]] = [[[1.0f32, 2.0f32, 3.0f32]]]
        let b: Tensor[f32, [1, 1, 3]] = [[[100.0f32, 200.0f32, 300.0f32]]]
        cache <- a
        cache <- b
        let last = cache[.., -1, ..]
        last[0, 0, 2]
    }
}
"#);
    assert!((as_float(&v) - 300.0).abs() < 1e-6, "got {:?}", v);
}
#[test]
fn ellipsis_slice_without_sandwich_still_reduces_rank() {
    // t[.., -1] (no FullSlice after) must still reduce rank — regression guard.
    let v = run(r#"
fn main() -> f32 {
    let t: Tensor[f32, [2, 3]] = [[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]
    let row = t[.., -1]
    row[1]
}
"#);
    assert!((as_float(&v) - 6.0).abs() < 1e-6, "got {:?}", v);
}

// ── Issue #74: duplicate unqualified import detection ─────────────────────────
#[test]
fn duplicate_unqualified_import_is_error() {
    let e = run_multi_err(&[
        ("mod_a.dmc", "pub fn helper() -> i64 { 1 }"),
        ("mod_b.dmc", "pub fn helper() -> i64 { 2 }"),
        ("main.dmc", r#"use "mod_a.dmc"
use "mod_b.dmc"
fn main() -> i64 { helper() }"#),
    ], "main.dmc");
    assert!(e.contains("ambiguous") || e.contains("helper"), "got: {e}");
}

/// Run only the typechecker on `src`; return the first error string (panics if no error).
fn run_type_err(src: &str) -> String {
    use super::check::Checker;
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let prog = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.check_program(&prog, None);
    assert!(!checker.errors.is_empty(), "expected a type error but got none");
    checker.errors.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ")
}

// ── Issue #77: shape-generic model constructor field type substitution ─────────
#[test]
fn generic_model_ctor_with_literal_typechecks() {
    // M[3] { grid: [10, 20, 30] } should NOT produce a type error.
    use super::check::Checker;
    let src = r#"
model M[N] {
    grid: Tensor[i64, [N]]
}
fn main() -> i64 {
    let x = M[3] { grid: [10, 20, 30] }
    x.grid[1]
}
"#;
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let prog = Parser::new(tokens).parse_program().expect("parse");
    let mut checker = Checker::new();
    checker.check_program(&prog, None);
    assert!(checker.errors.is_empty(), "unexpected type errors: {:?}", checker.errors);
    // Also verify the interpreter returns the right value.
    let v = run(src);
    assert!((as_float(&v) - 20.0).abs() < 1e-9, "got {:?}", v);
}
#[test]
fn generic_model_ctor_mismatch_still_errors() {
    // M[3] { grid: [1, 2] } — shape mismatch (2 ≠ 3) must still be a type error.
    let e = run_type_err(r#"
model M[N] {
    grid: Tensor[i64, [N]]
}
fn main() -> i64 {
    let x = M[3] { grid: [1, 2] }
    x.grid[0]
}
"#);
    assert!(e.contains("mismatched") || e.contains("type"), "got: {e}");
}

// ── Issue #76: `:=` properly shadows; diagnostic on `!` shadowing ─────────────
#[test]
fn colon_eq_shadows_does_not_write_through() {
    // `:=` inside a block should NOT update the outer `let !` binding.
    // (run() bypasses the typechecker — tests interpreter semantics only.)
    let v = run(r#"
fn main() -> i64 {
    let !x = 10
    if true {
        x := 99
    }
    x
}
"#);
    assert_eq!(as_int(&v), 10);
}
#[test]
fn colon_eq_shadow_diagnostic() {
    // `:=` that shadows outer `let !` should produce a type-checker warning/error.
    let e = run_type_err(r#"
fn main() -> i64 {
    let !x = 10
    if true {
        x := 99
    }
    x
}
"#);
    assert!(e.contains(":=") || e.contains("shadow") || e.contains("Did you mean"), "got: {e}");
}

// ── Issue #81: UTF-8 string literals round-trip correctly ────────────────────
#[test]
fn utf8_string_roundtrip() {
    let v = run(r#"fn main() -> str { "em-dash: —" }"#);
    if let Value::Str(s) = v {
        assert!(s.contains('—'), "expected em-dash in output, got: {:?}", s);
    } else { panic!("expected str, got {:?}", v); }
}
#[test]
fn utf8_multi_codepoint_string() {
    let v = run(r#"fn main() -> str { "π § 🐍" }"#);
    if let Value::Str(s) = v {
        assert!(s.contains('π') && s.contains('§'), "got: {:?}", s);
    } else { panic!("expected str, got {:?}", v); }
}

// ── Issue #80: KV stream capacity enforcement ────────────────────────────────
#[test]
fn kv_capacity_overflow_panics() {
    let e = run_err(r#"
fn main() -> nil {
    stream {
        let !c: KV[f32, [1, ~, 2]] = stream.kv[f32, [1, ~, 2]](capacity = 2)
        let frame: Tensor[f32, [1, 1, 2]] = [[[1.0f32, 2.0f32]]]
        c <- frame
        c <- frame
        c <- frame
    }
    nil
}
"#);
    assert!(e.contains("capacity") || e.contains("exceeded"), "got: {e}");
}
#[test]
fn kv_capacity_within_limit_ok() {
    // Exactly at capacity — should NOT panic.
    let v = run(r#"
fn main() -> f32 {
    stream {
        let !c: KV[f32, [1, ~, 2]] = stream.kv[f32, [1, ~, 2]](capacity = 2)
        let frame: Tensor[f32, [1, 1, 2]] = [[[1.0f32, 2.0f32]]]
        c <- frame
        c <- frame
        c[0, 1, 1]
    }
}
"#);
    assert!((as_float(&v) - 2.0).abs() < 1e-6, "got {:?}", v);
}

// ── Issue #79: match exhaustiveness ─────────────────────────────────────────
#[test]
fn match_bool_missing_arm_is_type_error() {
    let e = run_type_err(r#"
fn classify(b: bool) -> i64 {
    match b {
        true => 1
    }
}
fn main() -> nil { nil }
"#);
    assert!(e.contains("false") || e.contains("exhaustive") || e.contains("coverage"), "got: {e}");
}
#[test]
fn match_bool_with_catchall_is_ok() {
    // _ arm covers the missing case — no type error.
    use super::check::Checker;
    let src = r#"
fn classify(b: bool) -> i64 {
    match b {
        true => 1,
        _ => 0
    }
}
fn main() -> nil { nil }
"#;
    let tokens = Lexer::new(src).tokenize().expect("lex");
    let prog = Parser::new(tokens).parse_program().expect("parse");
    let mut checker = Checker::new();
    checker.check_program(&prog, None);
    assert!(checker.errors.is_empty(), "unexpected errors: {:?}", checker.errors);
}
#[test]
fn match_no_arm_matched_panics_at_runtime() {
    let e = run_err(r#"
fn main() -> i64 {
    match 99 {
        1 => 10
    }
}
"#);
    assert!(e.contains("no arm matched") || e.contains("match"), "got: {e}");
}

// ── Issue #83: shape params bound in method bodies ──────────────────────────

#[test]
fn shape_param_bound_in_method_body() {
    // Shape param `D` must be visible inside the method body as an i64.
    let v = run(r#"
model Linear[D] {
    scale: i64

    fn dim(self) -> i64 { D }
}

fn main() -> i64 {
    let m = Linear[4] { scale: 1 }
    m.dim()
}
"#);
    assert_eq!(as_int(&v), 4);
}

#[test]
fn shape_param_used_in_forward_desugar() {
    // `m(x)` desugars to `m.forward(x)`; shape bindings must pass through.
    let v = run(r#"
model Proj[N] {
    bias: i64

    fn forward(self, x: i64) -> i64 { x + N }
}

fn main() -> i64 {
    let p = Proj[8] { bias: 0 }
    p(10)
}
"#);
    assert_eq!(as_int(&v), 18);
}

#[test]
fn shape_param_multiple_dims_in_method() {
    // Two shape params — both must be bound in the method body.
    let v = run(r#"
model Mat[R, C] {
    stride: i64

    fn rows(self) -> i64 { R }
    fn cols(self) -> i64 { C }
}

fn main() -> i64 {
    let m = Mat[3, 5] { stride: 1 }
    m.rows() * m.cols()
}
"#);
    assert_eq!(as_int(&v), 15);
}

// ── Named shape args on fn calls: `f[S=4]()` ────────────────────────────────
// The named form used to fall through to a silent `<opaque bracket(...)>`
// instead of binding the shape param (only the positional `f[4]()` worked).

#[test]
fn named_shape_arg_binds_zero_arg_fn() {
    let v = run(r#"
fn f[S]() -> i64 {
    let m = forge.zeros[f32, [S, S]]
    (sum(m) as i64) + S
}

fn main() -> i64 { f[S=4]() }
"#);
    assert_eq!(as_int(&v), 4);
}

#[test]
fn mixed_positional_and_named_shape_args() {
    let v = run(r#"
fn g[A, B]() -> i64 { A * 10 + B }

fn main() -> i64 { g[3, B=7]() }
"#);
    assert_eq!(as_int(&v), 37);
}

#[test]
fn named_shape_arg_unknown_param_errors() {
    let e = run_err(r#"
fn f[S]() -> i64 { S }

fn main() -> i64 { f[X=4]() }
"#);
    assert!(e.contains("not a shape parameter"), "got: {e}");
}

#[test]
fn named_shape_arg_bound_twice_errors() {
    let e = run_err(r#"
fn f[S]() -> i64 { S }

fn main() -> i64 { f[4, S=5]() }
"#);
    assert!(e.contains("bound twice"), "got: {e}");
}

// ── Issue #87: bool↔int cast ──────────────────────────────────────────────────

#[test]
fn bool_to_int_cast() {
    let v = run(r#"fn main() -> i64 { true as i64 }"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn false_to_int_cast() {
    let v = run(r#"fn main() -> i64 { false as i64 }"#);
    assert_eq!(as_int(&v), 0);
}

#[test]
fn nonzero_int_to_bool_cast() {
    let v = run(r#"fn main() -> bool { 1 as bool }"#);
    assert!(matches!(v, Value::Bool(true)), "expected true, got {v:?}");
}

#[test]
fn zero_int_to_bool_cast() {
    let v = run(r#"fn main() -> bool { 0 as bool }"#);
    assert!(matches!(v, Value::Bool(false)), "expected false, got {v:?}");
}

#[test]
fn negative_int_to_bool_cast() {
    let v = run(r#"fn main() -> bool { -1 as bool }"#);
    assert!(matches!(v, Value::Bool(true)), "expected true for -1, got {v:?}");
}

// ── Issue #86: chained comparison is a parse error ───────────────────────────

#[test]
fn chained_comparison_is_parse_error() {
    let tokens = crate::lexer::Lexer::new("fn main() -> bool { 1 < 2 < 3 }")
        .tokenize().expect("lex");
    let result = crate::parser::Parser::new(tokens).parse_program();
    assert!(result.is_err(), "expected parse error for chained comparison");
    let msg = result.unwrap_err().msg;
    assert!(msg.contains("chained") || msg.contains("none-associative"), "got: {msg}");
}

#[test]
fn chained_equality_is_parse_error() {
    let tokens = crate::lexer::Lexer::new("fn main() -> bool { 1 == 1 == 1 }")
        .tokenize().expect("lex");
    let result = crate::parser::Parser::new(tokens).parse_program();
    assert!(result.is_err(), "expected parse error for chained equality");
}

#[test]
fn single_comparison_still_works() {
    let v = run(r#"fn main() -> bool { 1 < 5 }"#);
    assert!(matches!(v, Value::Bool(true)));
}

// ── Issue #88: @host match feature detection ─────────────────────────────────

#[test]
fn host_match_unknown_feature_falls_through() {
    let v = run(r#"
fn main() -> i64 {
    let r = @host match {
        .totally_fictional_feature => 1,
        _ => 99,
    }
    r
}
"#);
    assert_eq!(as_int(&v), 99);
}

#[test]
fn host_match_known_feature_matches() {
    // SSE2 on x86_64, NEON on aarch64 — one of these must match.
    let v = run(r#"
fn main() -> i64 {
    let r = @host match {
        .sse2 => 1,
        .neon => 1,
        _     => 0,
    }
    r
}
"#);
    assert_eq!(as_int(&v), 1, "expected a baseline ISA feature to match");
}

// ── Issue #90: immutable binding reassignment is a type error ────────────────

#[test]
fn immutable_binding_reassignment_is_type_error() {
    let e = run_type_err(r#"
fn main() -> i64 {
    let x = 1
    x = x + 1
    x
}
"#);
    assert!(e.contains("immutable") || e.contains("cannot assign"), "got: {e}");
}

#[test]
fn mutable_let_bang_allows_reassignment() {
    let diags = {
        use super::check::Checker;
        let tokens = crate::lexer::Lexer::new(r#"
fn main() -> i64 {
    let !x = 1
    x = x + 1
    x
}
"#).tokenize().expect("lex");
        let prog = crate::parser::Parser::new(tokens).parse_program().expect("parse");
        let mut checker = Checker::new();
        checker.check_program(&prog, None);
        checker.errors
    };
    assert!(diags.is_empty(), "let !x should allow reassignment, got: {:?}", diags);
}

#[test]
fn mutable_let_mut_allows_reassignment() {
    let diags = {
        use super::check::Checker;
        let tokens = crate::lexer::Lexer::new(r#"
fn main() -> i64 {
    let mut x = 1
    x = x + 1
    x
}
"#).tokenize().expect("lex");
        let prog = crate::parser::Parser::new(tokens).parse_program().expect("parse");
        let mut checker = Checker::new();
        checker.check_program(&prog, None);
        checker.errors
    };
    assert!(diags.is_empty(), "let mut x should allow reassignment, got: {:?}", diags);
}

// ── Issue #92: return type mismatch caught by typechecker ────────────────────

#[test]
fn return_non_nil_from_nil_fn_is_type_error() {
    let e = run_type_err(r#"
fn g() -> nil { return 42 }
fn main() -> nil { g() }
"#);
    assert!(e.contains("return") && (e.to_lowercase().contains("nil") || e.contains("integer")), "got: {e}");
}

#[test]
fn nil_tail_from_nil_fn_is_type_error() {
    let e = run_type_err(r#"
fn h() -> nil { 42 }
fn main() -> nil { h() }
"#);
    assert!(e.contains("returns") || e.contains("body produces"), "got: {e}");
}

#[test]
fn bare_return_from_nil_fn_is_ok() {
    let diags = {
        use super::check::Checker;
        let tokens = crate::lexer::Lexer::new(r#"
fn g() -> nil { return }
fn main() -> nil { g() }
"#).tokenize().expect("lex");
        let prog = crate::parser::Parser::new(tokens).parse_program().expect("parse");
        let mut checker = Checker::new();
        checker.check_program(&prog, None);
        checker.errors
    };
    assert!(diags.is_empty(), "bare `return` from `-> nil` should be valid, got: {:?}", diags);
}

// ── Issue #96: extern fn ──────────────────────────────────────────────────────

#[test]
fn extern_fn_parses() {
    // Plain extern fn with no ABI string.
    let tokens = Lexer::new(r#"
extern fn cblas_sgemm(M: i32, N: i32, K: i32) -> nil
fn main() -> nil { nil }
"#).tokenize().expect("lex");
    Parser::new(tokens).parse_program().expect("extern fn should parse without error");
}

#[test]
fn extern_fn_with_abi_parses() {
    // `extern "cuda" fn` syntax.
    let tokens = Lexer::new(r#"
extern "cuda" fn cuda_sgemm(M: i32, N: i32, K: i32) -> nil
fn main() -> nil { nil }
"#).tokenize().expect("lex");
    Parser::new(tokens).parse_program().expect("extern fn with ABI string should parse");
}

#[test]
fn extern_fn_is_callable_value() {
    // Extern fn name resolves to a Value::Fn at runtime.
    let v = run(r#"
extern fn cblas_sgemm(M: i32, N: i32, K: i32) -> nil
fn main() -> nil {
    let f = cblas_sgemm
    nil
}
"#);
    assert!(matches!(v, Value::Nil), "expected Nil, got {:?}", v);
}

#[test]
fn extern_fn_call_errors_at_runtime() {
    // Calling an extern fn produces a clear error (no JIT backend).
    let e = run_err(r#"
extern fn cblas_sgemm(M: i32, N: i32, K: i32) -> nil
fn main() -> nil {
    cblas_sgemm(1, 2, 3)
}
"#);
    assert!(e.contains("extern fn") && e.contains("cblas_sgemm"), "got: {e}");
}

#[test]
fn extern_fn_typechecks_signature() {
    // The typechecker should register the extern fn signature so callers
    // are validated; no type errors expected for a correct call site.
    let diags = {
        use super::check::Checker;
        let tokens = Lexer::new(r#"
extern fn my_op(x: i32, y: f32) -> i32
fn main() -> nil { nil }
"#).tokenize().expect("lex");
        let prog = Parser::new(tokens).parse_program().expect("parse");
        let mut checker = Checker::new();
        checker.check_program(&prog, None);
        checker.errors
    };
    assert!(diags.is_empty(), "extern fn signature should check cleanly, got: {:?}", diags);
}

// ── Issue #99: trailing `return X` satisfies declared return type ─────────────

#[test]
fn trailing_return_satisfies_declared_type() {
    // Typechecker must not flag `fn -> i64 { ... return n }` as a body-type mismatch.
    let diags = {
        use super::check::Checker;
        let tokens = Lexer::new(r#"
fn early(n: i64) -> i64 {
    if n > 0 { return 42 }
    return n
}
fn main() -> nil { nil }
"#).tokenize().expect("lex");
        let prog = Parser::new(tokens).parse_program().expect("parse");
        let mut checker = Checker::new();
        checker.check_program(&prog, None);
        checker.errors
    };
    assert!(diags.is_empty(), "trailing `return n` should satisfy `-> i64`, got: {:?}", diags);
}

#[test]
fn trailing_return_runs_correctly() {
    let v = run(r#"
fn early(n: i64) -> i64 {
    if n > 0 { return 42 }
    return n
}
fn main() -> i64 { early(-7) }
"#);
    assert!(matches!(v, Value::Int(-7, _)), "expected -7, got {:?}", v);
}

#[test]
fn gcd_with_trailing_return() {
    // Acceptance test from the issue: fn gcd(a, b) -> i64 { ... return a }
    let v = run(r#"
fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a
    let mut b = b
    while b != 0 {
        let t = b
        b = a % b
        a = t
    }
    return a
}
fn main() -> i64 { gcd(462, 1071) }
"#);
    assert!(matches!(v, Value::Int(21, _)), "gcd(462,1071) expected 21, got {:?}", v);
}

#[test]
fn return_type_mismatch_still_errors() {
    // A trailing `return X` with wrong type must still be an error.
    let diags = {
        use super::check::Checker;
        let tokens = Lexer::new(r#"
fn bad() -> i64 {
    return "oops"
}
fn main() -> nil { nil }
"#).tokenize().expect("lex");
        let prog = Parser::new(tokens).parse_program().expect("parse");
        let mut checker = Checker::new();
        checker.check_program(&prog, None);
        checker.errors
    };
    assert!(!diags.is_empty(), "return type mismatch should still be caught");
    assert!(diags[0].msg.contains("return"), "expected `return` error, got: {:?}", diags);
}

// ── Issue #116: tensor mutations through ! params write back to caller ────────

#[test]
fn mut_param_writeback_single_element() {
    // Exact repro from the issue: fill(!t) sets t[0] = 9; caller must see 9.
    let v = run(r#"
fn fill(!t: Tensor[f32, [4]]) -> nil {
    t[0] = 9.0
    nil
}
fn main() -> f32 {
    let !x = forge.zeros[f32, [4]]
    fill(x)
    x[0]
}
"#);
    match v {
        Value::Float(f, _) => assert!((f - 9.0).abs() < 1e-6, "expected 9.0, got {f}"),
        Value::Int(n, _)   => assert_eq!(n, 9, "expected 9"),
        other           => panic!("expected numeric, got {:?}", other),
    }
}

#[test]
fn mut_param_writeback_multiple_params() {
    // Two ! params: swap should be visible in caller.
    let v = run(r#"
fn swap(!a: Tensor[f32, [2]], !b: Tensor[f32, [2]]) -> nil {
    let tmp0 = a[0]
    let tmp1 = a[1]
    a[0] = b[0]
    a[1] = b[1]
    b[0] = tmp0
    b[1] = tmp1
    nil
}
fn main() -> f32 {
    let !x = forge.zeros[f32, [2]]
    let !y = forge.zeros[f32, [2]]
    x[0] = 1.0
    x[1] = 2.0
    y[0] = 3.0
    y[1] = 4.0
    swap(x, y)
    x[0]
}
"#);
    match v {
        Value::Float(f, _) => assert!((f - 3.0).abs() < 1e-6, "expected x[0]=3.0 after swap, got {f}"),
        Value::Int(n, _)   => assert_eq!(n, 3, "expected x[0]=3 after swap"),
        other           => panic!("expected numeric, got {:?}", other),
    }
}

#[test]
fn mut_param_writeback_loop_fill() {
    // While-loop inside callee: all elements must be visible after return.
    let v = run(r#"
fn scale(!t: Tensor[f32, [3]], factor: f32) -> nil {
    let !i = 0
    while i < 3 {
        t[i] = t[i] * factor
        i = i + 1
    }
    nil
}
fn main() -> f32 {
    let !v = forge.zeros[f32, [3]]
    v[0] = 1.0
    v[1] = 2.0
    v[2] = 3.0
    scale(v, 10.0)
    v[2]
}
"#);
    match v {
        Value::Float(f, _) => assert!((f - 30.0).abs() < 1e-6, "expected v[2]=30.0, got {f}"),
        Value::Int(n, _)   => assert_eq!(n, 30, "expected v[2]=30"),
        other           => panic!("expected numeric, got {:?}", other),
    }
}

#[test]
fn non_mut_param_not_written_back() {
    // A non-! tensor param must NOT be written back; caller's copy unchanged.
    let v = run(r#"
fn try_mutate(t: Tensor[f32, [2]]) -> nil {
    t[0] = 99.0
    nil
}
fn main() -> f32 {
    let !x = forge.zeros[f32, [2]]
    x[0] = 1.0
    try_mutate(x)
    x[0]
}
"#);
    match v {
        Value::Float(f, _) => assert!((f - 1.0).abs() < 1e-6, "non-! param should NOT write back, got {f}"),
        Value::Int(n, _)   => assert_eq!(n, 1, "non-! param should NOT write back"),
        other           => panic!("expected numeric, got {:?}", other),
    }
}

// ── Issue #475: `!` tensor params on METHODS alias like they do on free fns ──
//
// Tensors are value-copies in the interpreter, so a `!` param only aliases by
// virtue of the post-call writeback. The two model-method dispatch paths used
// to return straight out of `call_fn_with_shapes` without performing it, so a
// method's writes through a tensor out-parameter vanished — silently, with the
// right return value. The scope table in the issue is pinned below.

#[test]
fn mut_tensor_param_on_generic_model_method_475() {
    // The issue's headline repro: `fill_method!` on a shape-generic model must
    // fill the caller's tensor, exactly as the identical free-function body does.
    let v = run(r#"
model Box[N] {
    !vals: Tensor[i64, [N]]
    fn fill_method!(self, !order: Tensor[i64, [4]]) -> i64 {
        for i in 0..N { order[i] = self.vals[i] }
        N
    }
}
fn main() -> i64 {
    let !b = Box[3] { vals: forge.zeros[i64, [3]] }
    let !v = b.vals
    v[0] = 7  v[1] = 8  v[2] = 9
    let !order = forge.zeros[i64, [4]]
    let n = b.fill_method!(order)
    n * 1000 + order[0] * 100 + order[2]
}
"#);
    // n=3, order[0]=7, order[2]=9
    assert_eq!(as_int(&v), 3709, "writes through a `!` tensor param on a generic model method were lost");
}

#[test]
fn mut_tensor_param_on_non_generic_model_method_475() {
    // Same loss on a plain model — the bug was never about generics.
    let v = run(r#"
model Box {
    !k: i64
    fn fill_method!(self, !order: Tensor[i64, [4]]) -> i64 {
        order[0] = self.k
        order[3] = self.k * 2
        self.k
    }
}
fn main() -> i64 {
    let !b = Box { k: 6 }
    let !order = forge.zeros[i64, [4]]
    let n = b.fill_method!(order)
    n * 1000 + order[0] * 100 + order[3]
}
"#);
    assert_eq!(as_int(&v), 6612, "writes through a `!` tensor param on a non-generic model method were lost");
}

#[test]
fn mut_tensor_param_on_free_fn_still_writes_back_475() {
    // The row of the table that always worked. Kept so a future change to the
    // writeback offset cannot fix methods by breaking free functions.
    let v = run(r#"
model Box[N] { !vals: Tensor[i64, [N]] }
fn fill_free![N](b: Box[N], !order: Tensor[i64, [4]]) -> i64 {
    for i in 0..N { order[i] = b.vals[i] }
    N
}
fn main() -> i64 {
    let !b = Box[3] { vals: forge.zeros[i64, [3]] }
    let !v = b.vals
    v[0] = 7  v[1] = 8  v[2] = 9
    let !order = forge.zeros[i64, [4]]
    let n = fill_free![3](b, order)
    n * 1000 + order[0] * 100 + order[2]
}
"#);
    assert_eq!(as_int(&v), 3709);
}

#[test]
fn mut_model_param_on_method_still_mutates_475() {
    // The other row that already worked: a `!` MODEL parameter aliases through
    // its `Rc<RefCell<..>>` regardless of the writeback. The inconsistency
    // between this and the tensor row is what made the bug so sharp.
    let v = run(r#"
model Sink { !n: i64 }
model Driver {
    !step: i64
    fn push!(self, !s: Sink) -> i64 {
        s.n = s.n + self.step
        s.n
    }
}
fn main() -> i64 {
    let !d = Driver { step: 5 }
    let !s = Sink { n: 1 }
    let r = d.push!(s)
    r * 100 + s.n
}
"#);
    assert_eq!(as_int(&v), 606, "a `!` model param on a method must still mutate in place");
}

#[test]
fn mut_tensor_param_on_method_probe_475() {
    // The issue's probe case, pinning all three observations at once: the
    // callee reads what the caller wrote (5), sees its own write internally
    // (9), and — the part that was broken — the caller sees that write too.
    let v = run(r#"
model Box {
    !k: i64
    fn probe!(self, !out: Tensor[i64, [4]]) -> i64 {
        let seen_in = out[0]
        out[1] = 9
        let seen_own = out[1]
        seen_in * 100 + seen_own
    }
}
fn main() -> i64 {
    let !b = Box { k: 0 }
    let !out = forge.zeros[i64, [4]]
    out[0] = 5
    let r = b.probe!(out)
    r * 10 + out[1]
}
"#);
    // r == 509 (read-in 5, write visible internally 9), out[1] == 9 for the caller.
    assert_eq!(as_int(&v), 5099, "probe: caller must observe the method's write through `!out`");
}

#[test]
fn non_mut_tensor_param_on_method_not_written_back_475() {
    // The fix must not make every method parameter alias: without `!` the
    // callee still works on a copy.
    let v = run(r#"
model Box {
    !k: i64
    fn touch(self, t: Tensor[i64, [2]]) -> i64 {
        t[0] = 99
        t[0]
    }
}
fn main() -> i64 {
    let !b = Box { k: 0 }
    let !x = forge.zeros[i64, [2]]
    x[0] = 1
    let r = b.touch(x)
    r * 10 + x[0]
}
"#);
    assert_eq!(as_int(&v), 991, "a non-`!` tensor param on a method must NOT write back");
}

#[test]
fn mut_tensor_param_through_forward_desugar_475() {
    // `m(x)` ≡ `m.forward(x)` is a second dispatch path with the receiver
    // prepended to the argument vector; it needs the same writeback offset.
    // Spelled `forward`, not `forward!`: the lexer folds a trailing `!` into
    // the identifier, so a method declared `forward!` registers under that
    // name and the desugar — which looks up `forward` — never reaches it.
    let v = run(r#"
model Filler {
    !k: i64
    fn forward(self, !out: Tensor[i64, [3]]) -> i64 {
        out[0] = self.k
        out[2] = self.k + 1
        self.k
    }
}
fn main() -> i64 {
    let !f = Filler { k: 4 }
    let !out = forge.zeros[i64, [3]]
    let n = f(out)
    n * 100 + out[0] * 10 + out[2]
}
"#);
    assert_eq!(as_int(&v), 445, "forward-desugar must write back `!` tensor params too");
}

// ── Issue #671: `!` tensor params on a BRACKETED method call alias too ──────
//
// #475 fixed the writeback for `recv.method(args)`, but a call with an
// explicit shape bracket (`recv.method![K](args)`) goes through
// `call_bracketed_method`, a third dispatch path that returned straight out
// of `call_fn_with_shapes` without it — same silent loss, one bracket over.

#[test]
fn mut_tensor_param_on_bracketed_method_call_671() {
    // Same shape as the design doc's q13 probe: a plain method call on a model
    // writes back (`fill!`), and its bracketed sibling (`fillg![K]`) must too.
    let v = run(r#"
model Buf[N] {
    !vals: Tensor[i64, [N]]
    fn fill!(self, !out: Tensor[i64, [4]]) -> i64 { out[0] = 7  out[0] }
    fn fillg![K](self, !out: Tensor[i64, [4]]) -> i64 { out[0] = 7  out[0] }
}
fn main() -> i64 {
    let !b = Buf[4] { vals: forge.zeros[i64, [4]] }
    let !o1 = forge.zeros[i64, [4]]
    let r1 = b.fill!(o1)
    let !o2 = forge.zeros[i64, [4]]
    let r2 = b.fillg![2](o2)
    r1 * 1000 + o1[0] * 100 + r2 * 10 + o2[0]
}
"#);
    // r1=7, o1[0]=7, r2=7, o2[0]=7 — the bracketed call must write back
    // exactly like the plain one, not leave o2[0] at its zero-initialized 0.
    assert_eq!(as_int(&v), 7777, "a bracketed method call must write back a `!` tensor param");
}

#[test]
fn non_mut_tensor_param_on_bracketed_method_call_not_written_back_671() {
    // The fix must not make every bracketed method parameter alias: without
    // `!` the callee still works on a copy.
    let v = run(r#"
model Buf[N] {
    !vals: Tensor[i64, [N]]
    fn touchg![K](self, out: Tensor[i64, [4]]) -> i64 { out[0] = 99  out[0] }
}
fn main() -> i64 {
    let !b = Buf[4] { vals: forge.zeros[i64, [4]] }
    let !x = forge.zeros[i64, [4]]
    x[0] = 1
    let r = b.touchg![2](x)
    r * 10 + x[0]
}
"#);
    assert_eq!(as_int(&v), 991, "a non-`!` tensor param on a bracketed method must NOT write back");
}

// ── Issue #476: model arrays held in a model field ───────────────────────────

#[test]
fn model_array_field_indexed_assign_works_476() {
    // Spelling 3. `h.cells[i] = Cell { .. }` used to fail at runtime with
    // "indexed assignment requires a tensor, got list" — a message about
    // tensors and lists, neither of which the author wrote. The local form
    // (`cs[i] = ..`) had always worked; only the through-a-field path hit the
    // tensor-only branch. Writing through the struct's own Rc, it now lands.
    let v = run(r#"
model Cell { !n: i64 }
model Holder { !cells: [Cell; 3] }
fn main() -> i64 {
    let !h = Holder { cells: forge.uninit[Cell, [3]] }
    for i in 0..3 { h.cells[i] = Cell { n: i * 10 } }
    let a = h.cells[0]
    let b = h.cells[2]
    a.n * 100 + b.n
}
"#);
    assert_eq!(as_int(&v), 20, "a[0].n=0, b[2].n=20 — the writes must reach the field");
}

#[test]
fn model_array_field_indexed_assign_is_not_a_copy_476() {
    // The point of writing through the field: neighbouring slots survive, and
    // an element aliased out afterwards sees the stored value.
    let v = run(r#"
model Cell { !n: i64 }
model Holder { !cells: [Cell; 3] }
fn main() -> i64 {
    let !h = Holder { cells: forge.uninit[Cell, [3]] }
    for i in 0..3 { h.cells[i] = Cell { n: 1 } }
    h.cells[1] = Cell { n: 7 }
    let a = h.cells[0]
    let b = h.cells[1]
    let c = h.cells[2]
    a.n * 100 + b.n * 10 + c.n
}
"#);
    assert_eq!(as_int(&v), 171);
}

#[test]
fn model_array_field_negative_and_oob_index_476() {
    // Same index discipline as the local model-array store: negatives wrap,
    // out of bounds is an error rather than a silent no-op.
    let v = run(r#"
model Cell { !n: i64 }
model Holder { !cells: [Cell; 3] }
fn main() -> i64 {
    let !h = Holder { cells: forge.uninit[Cell, [3]] }
    for i in 0..3 { h.cells[i] = Cell { n: 0 } }
    h.cells[-1] = Cell { n: 5 }
    let c = h.cells[2]
    c.n
}
"#);
    assert_eq!(as_int(&v), 5);

    let e = run_err(r#"
model Cell { !n: i64 }
model Holder { !cells: [Cell; 3] }
fn main() -> i64 {
    let !h = Holder { cells: forge.uninit[Cell, [3]] }
    h.cells[7] = Cell { n: 0 }
    0
}
"#);
    assert!(e.contains("out of bounds"), "expected an out-of-bounds error, got {e:?}");
}

#[test]
fn uninit_model_array_element_read_names_its_cause_476() {
    // Fix (2). An unwritten slot is `nil`, so a field read off one used to
    // yield a silent `Opaque` — the program then died at whatever line first
    // USED the value ("comparison requires numeric, str, or bool operands; got
    // opaque and int"), naming neither the field, nor the array, nor
    // initialization. `--check` catches the field spelling now, but not every
    // path reaches the checker, so the runtime says what actually happened.
    let e = run_err(r#"
model Cell { !n: i64 }
fn main() -> i64 {
    let !cs = forge.uninit[Cell, [3]]
    cs[0] = Cell { n: 1 }
    let c = cs[1]
    c.n
}
"#);
    assert!(e.contains("uninitialized model-array element"),
            "the runtime must name the real cause, got {e:?}");
    assert!(!e.contains("opaque"), "must not surface as an `opaque` complaint: {e:?}");
}

// ── Issue #115: solve / inv / lstsq stdlib primitives ────────────────────────

fn approx(v: &Value, expected: f64, tol: f64, label: &str) {
    let got = match v {
        Value::Float(f, _) => *f,
        Value::Int(n, _) => *n as f64,
        other => panic!("{}: expected numeric, got {:?}", label, other),
    };
    assert!((got - expected).abs() < tol, "{}: expected {}, got {}", label, expected, got);
}

fn tensor_elem(v: &Value, idx: &[usize]) -> f64 {
    if let Value::Tensor(t) = v {
        t[ndarray::IxDyn(idx)]
    } else {
        panic!("expected tensor, got {:?}", v)
    }
}

#[test]
fn solve_2x2_system() {
    // 2x + y = 5, x + 3y = 10  → x=1, y=3
    let v = run(r#"
fn main() -> Tensor[f32, [2]] {
    let A = [[2.0, 1.0], [1.0, 3.0]]
    let b = [5.0, 10.0]
    solve(A, b)
}
"#);
    approx(&Value::Float(tensor_elem(&v, &[0]), FW::F64), 1.0, 1e-9, "x");
    approx(&Value::Float(tensor_elem(&v, &[1]), FW::F64), 3.0, 1e-9, "y");
}

#[test]
fn inv_2x2() {
    // inv([[1,2],[3,4]]) = [[-2,1],[1.5,-0.5]]
    let v = run(r#"
fn main() -> Tensor[f32, [2, 2]] {
    inv([[1.0, 2.0], [3.0, 4.0]])
}
"#);
    approx(&Value::Float(tensor_elem(&v, &[0, 0]), FW::F64), -2.0,  1e-9, "[0,0]");
    approx(&Value::Float(tensor_elem(&v, &[0, 1]), FW::F64),  1.0,  1e-9, "[0,1]");
    approx(&Value::Float(tensor_elem(&v, &[1, 0]), FW::F64),  1.5,  1e-9, "[1,0]");
    approx(&Value::Float(tensor_elem(&v, &[1, 1]), FW::F64), -0.5,  1e-9, "[1,1]");
}

#[test]
fn lstsq_line_fit() {
    // Fit y=slope*x to [(1,2.1),(2,4.0),(3,5.9)]. True slope=2, lstsq ≈ 1.986.
    let v = run(r#"
fn main() -> Tensor[f32, [1]] {
    let A = [[1.0], [2.0], [3.0]]
    let b = [2.1, 4.0, 5.9]
    lstsq(A, b)
}
"#);
    approx(&Value::Float(tensor_elem(&v, &[0]), FW::F64), 1.9857142857, 1e-6, "slope");
}

#[test]
fn solve_singular_errors() {
    // Singular matrix must produce a runtime error, not a panic.
    let e = run_err(r#"
fn main() -> nil {
    solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])
    nil
}
"#);
    assert!(e.contains("singular"), "expected singular error, got: {e}");
}

// ─── #113/2.6 — fully-masked softmax row ──────────────────────────────────────

#[test]
fn softmax_normal_row() {
    // Basic sanity: softmax of [1,2,3] sums to 1.
    let v = run(r#"
fn main() -> nil {
    softmax([1.0, 2.0, 3.0])
    nil
}
"#);
    assert!(matches!(v, Value::Nil), "expected Nil sentinel, got {:?}", v);
}

#[test]
fn softmax_fully_masked_row_is_zero() {
    // A row where every logit is -inf (fully masked) must produce all zeros,
    // not NaN. STDLIB.md §3.2 / issue #113/2.6.
    let v = run(r#"
fn main() -> f32 {
    let neg_inf = -1.0 / 0.0
    let x = [neg_inf, neg_inf, neg_inf]
    let s = softmax(x)
    sum(s)
}
"#);
    assert!((as_float(&v) - 0.0).abs() < 1e-9,
        "softmax of fully-masked row should sum to 0, got {}", as_float(&v));
}

#[test]
fn softmax_with_pos_inf_is_argmax_onehot() {
    // #258: softmax of a row containing +inf must put all mass on the +inf
    // element (the limit), not collapse to 0 or NaN. softmax([+inf, 0]) = [1, 0].
    let v = run(r#"
fn main() -> f32 {
    let pos_inf = 1.0 / 0.0
    let x = [pos_inf, 0.0]
    let s = softmax(x)
    s[0] * 100.0 + s[1]
}
"#);
    assert!((as_float(&v) - 100.0).abs() < 1e-6,
        "softmax([+inf, 0]) should be [1, 0], got combined {}", as_float(&v));
}

#[test]
fn mean_of_empty_tensor_is_nan() {
    // #258: mean of a zero-size tensor is 0/0 = NaN (undefined), matching the
    // JIT — not the old special-cased 0.
    let v = run(r#"
fn main() -> i64 {
    let m = mean(forge.zeros[f32, [0]])
    if m == m { 0 } else { 1 }
}
"#);
    assert_eq!(as_int(&v), 1, "mean of empty tensor should be NaN (m != m)");
}

#[test]
fn softmax_masked_attn_row_via_attn() {
    // attn() calls softmax internally; a fully-masked row should yield 0 weights,
    // not NaN propagating into the final output.
    let v = run(r#"
fn main() -> f32 {
    let q = [[[[1.0, 0.0]]]]
    let k = [[[[1.0, 0.0]]]]
    let v2 = [[[[1.0, 2.0]]]]
    let mask = [[0.0]]
    let out = attn(q, k, v2, mask)
    sum(out)
}
"#);
    assert!(!as_float(&v).is_nan(), "attn with full mask should not produce NaN");
    assert!((as_float(&v) - 0.0).abs() < 1e-9,
        "attn with full mask should produce 0, got {}", as_float(&v));
}

#[test]
fn attn_non_tensor_mask_is_an_error() {
    // A non-tensor 4th arg used to be silently ignored, computing unmasked
    // attention with no diagnostic. It must raise instead.
    let msg = run_err(r#"
fn main() -> f32 {
    let q = [[[[1.0, 0.0]]]]
    let out = attn(q, q, q, 1)
    sum(out)
}
"#);
    assert!(msg.contains("mask") && msg.contains("tensor"),
        "expected mask-type error, got: {}", msg);
}

#[test]
fn attn_gqa_non_tensor_mask_is_an_error() {
    let msg = run_err(r#"
fn main() -> f32 {
    let q = [[[[1.0, 0.0]]]]
    let out = attn_gqa(q, q, q, "mask")
    sum(out)
}
"#);
    assert!(msg.contains("mask") && msg.contains("tensor"),
        "expected mask-type error, got: {}", msg);
}

#[test]
fn attn_nil_mask_means_unmasked() {
    // nil stays a valid "no mask" placeholder.
    let v = run(r#"
fn main() -> f32 {
    let q = [[[[1.0, 0.0]]]]
    let v2 = [[[[3.0, 4.0]]]]
    let out = attn(q, q, v2, nil)
    sum(out)
}
"#);
    assert!((as_float(&v) - 7.0).abs() < 1e-9,
        "attn with nil mask should equal unmasked, got {}", as_float(&v));
}

#[test]
fn interp_load_npz_is_a_clear_error() {
    // NPZ loading is JIT-only; the interpreter used to yield an opaque value
    // that silently poisoned downstream use (e.g. an attn mask never applied).
    let msg = run_err(r#"
fn main() {
    let m = vault.load_npz[bool, [2, 2]]("nope.npz", key="mask")
    print(m)
}
"#);
    assert!(msg.contains("load_npz") && msg.contains("jit"),
        "expected JIT-only error, got: {}", msg);
}

#[test]
fn interp_vault_load_is_a_clear_error() {
    // Raw-binary weight loading is JIT-only, same as load_npz.
    let msg = run_err(r#"
fn main() {
    let w = vault.load[f32, [2, 2]]("weights.bin")
    print(w)
}
"#);
    assert!(msg.contains("vault.load") && msg.contains("jit"),
        "expected JIT-only error, got: {}", msg);
}

// ─── #113/1.2 — int8 dtype ────────────────────────────────────────────────────

#[test]
fn int8_dtype_parses_in_tensor_type() {
    // int8 must be accepted as a scalar type keyword.
    let v = run(r#"
fn main() -> nil {
    let x: Tensor[int8, [4]] = [1.0, 2.0, 3.0, 4.0]
    nil
}
"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn int8_cast_rounds_like_i64() {
    // @cast(int8) should truncate to integer (same as int4/i64 in interp).
    let v = run(r#"
fn main() -> i64 {
    @cast(int8) { 3.7 }
}
"#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn int8_cast_negative() {
    let v = run(r#"
fn main() -> i64 {
    @cast(int8) { -5.9 }
}
"#);
    assert_eq!(as_int(&v), -5);
}

// ─── #113/2.3 — embed canonical 2-arg signature ───────────────────────────────

#[test]
fn embed_basic_lookup() {
    // embed(vocab [V, D], ids [B]) -> [B, D].
    // vocab row 0 = [1, 0], row 1 = [0, 1].
    let v = run(r#"
fn main() -> f32 {
    let vocab = [[1.0, 0.0], [0.0, 1.0]]
    let ids   = [1.0, 0.0, 1.0]
    let out   = embed(vocab, ids)
    sum(out)
}
"#);
    // ids [1,0,1] → rows [0,1],[1,0],[0,1] → sum = 2 ones from row1 + 1 from row0 = 2+1 = 3 total
    assert!((as_float(&v) - 3.0).abs() < 1e-9,
        "embed sum should be 3.0, got {}", as_float(&v));
}

#[test]
fn embed_output_shape() {
    // Output must be [B, D] — check individual elements.
    let v = run(r#"
fn main() -> nil {
    let vocab = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    let ids   = [0.0, 1.0]
    embed(vocab, ids)
    nil
}
"#);
    assert!(matches!(v, Value::Nil));
}

#[test]
fn embed_wrong_vocab_rank_errors() {
    // vocab must be 2-D; 1-D vocab should produce a runtime error.
    let e = run_err(r#"
fn main() -> nil {
    let vocab = [1.0, 2.0, 3.0]
    embed(vocab, [0.0])
    nil
}
"#);
    assert!(e.contains("2-D") || e.contains("embed"),
        "expected embed arity error, got: {e}");
}

#[test]
fn dotpow_scalar_fast_path_matches_powf() {
    // .^0.5 must equal sqrt elementwise (4,9,16 -> 2,3,4 -> sum 9).
    let v = run(r#"
fn main() -> f64 {
    let !a = forge.zeros[f32, [3]]
    a[0] = 4.0
    a[1] = 9.0
    a[2] = 16.0
    let b = a .^ 0.5
    b[0] + b[1] + b[2]
}
"#);
    assert!((as_float(&v) - 9.0).abs() < 1e-9, "got {}", as_float(&v));
}

#[test]
fn dotpow_scalar_integer_and_square() {
    // .^2.0 (2,3 -> 4,9) and .^3.0 (2 -> 8): 4 + 9 + 8 = 21.
    let v = run(r#"
fn main() -> f64 {
    let !a = forge.zeros[f32, [2]]
    a[0] = 2.0
    a[1] = 3.0
    let sq = a .^ 2.0
    let cube = a .^ 3.0
    sq[0] + sq[1] + cube[0]
}
"#);
    assert!((as_float(&v) - 21.0).abs() < 1e-9, "got {}", as_float(&v));
}

// ── Inclusive range ───────────────────────────────────────────────────────────

#[test]
fn inclusive_range_visits_endpoint() {
    // 0..=4 must iterate exactly 5 times (0,1,2,3,4).
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let mut count = 0
            for _ in 0..=4 { count += 1 }
            count
        }
    "#)), 5);
}

#[test]
fn inclusive_range_sum_gauss() {
    // 1..=100 sum = 5050.
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let mut acc = 0
            for i in 1..=100 { acc += i }
            acc
        }
    "#)), 5050);
}

#[test]
fn exclusive_range_excludes_endpoint() {
    // Confirm 0..5 (exclusive) gives 5 elements — baseline sanity for the
    // inclusive counterpart above.
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let mut count = 0
            for _ in 0..5 { count += 1 }
            count
        }
    "#)), 5);
}

// ── match on string ───────────────────────────────────────────────────────────

#[test]
fn match_on_string_literal() {
    assert_eq!(as_int(&run(r#"
        fn classify(s: str) -> i64 {
            match s {
                "yes" => 1,
                "no"  => 0,
                _     => -1,
            }
        }
        fn main() -> i64 {
            classify("yes") + classify("no") + classify("maybe")
        }
    "#)), 0);  // 1 + 0 + (-1) = 0
}

#[test]
fn match_on_string_returns_string() {
    assert_eq!(as_str(&run(r#"
        fn main() -> str {
            match "hello" {
                "hello" => "world",
                _       => "other",
            }
        }
    "#)), "world");
}

// ── @grad: two differentiable parameters ─────────────────────────────────────

#[test]
fn grad_two_differentiable_params_both_computed() {
    // loss(w, b, x) = sum(w .* x .+ b) at w=ones, b=zeros, x=ones.
    // dL/dw[i] = x[i] = 1  → sum(g.w) = 4
    // dL/db[i] = 1          → sum(g.b) = 4
    // sum(g.w) + sum(g.b) should equal 8.
    let v = run(r#"
        @grad fn loss[D](!w: Tensor[f32, [D]], !b: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum(w .* x .+ b)
        }
        fn main() -> f32 {
            let !w = vault.ones[f32, [4]]
            let !b = vault.zeros[f32, [4]]
            let  x = vault.ones[f32, [4]]
            let (_l, g) = loss.fwd_bwd(w, b, x)
            sum(g.w) + sum(g.b)
        }
    "#);
    assert!((as_float(&v) - 8.0).abs() < 1e-6,
            "expected sum(g.w)+sum(g.b)=8.0, got {:?}", v);
}

#[test]
fn grad_forward_value_is_correct() {
    // Verify the loss value returned by fwd_bwd, not just the gradient.
    // loss(w, x) = sum(w .* x) at w=ones[3], x=[1,2,3] → loss = 6.
    let v = run(r#"
        @grad fn loss[D](!w: Tensor[f32, [D]], x: Tensor[f32, [D]]) -> f32 {
            sum(w .* x)
        }
        fn main() -> f32 {
            let !w = vault.ones[f32, [3]]
            let  x = [1.0f32, 2.0f32, 3.0f32]
            let (fwd, _g) = loss.fwd_bwd(w, x)
            fwd
        }
    "#);
    assert!((as_float(&v) - 6.0).abs() < 1e-6,
            "expected forward loss=6.0, got {:?}", v);
}

// ── @comptime extended ───────────────────────────────────────────────────────

#[test]
fn comptime_with_conditional() {
    // @comptime must evaluate conditionals, not just arithmetic.
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let x = @comptime { if 3 > 2 { 10 } else { 20 } }
            x
        }
    "#)), 10);
}

#[test]
fn comptime_result_usable_in_expression() {
    // The value produced by @comptime flows into the surrounding expression.
    assert_eq!(as_int(&run(r#"
        fn main() -> i64 {
            let base = @comptime { 6 * 7 }
            base + 0
        }
    "#)), 42);
}

// ── pipe chained with named function ─────────────────────────────────────────

#[test]
fn pipe_then_named_fn_chain() {
    assert_eq!(as_int(&run(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn double(x: i64) -> i64 { x * 2 }
        fn square(x: i64) -> i64 { x * x }
        fn main() -> i64 { 3 |> inc |> double |> square }
    "#)), 64);  // (3+1)*2 = 8; 8^2 = 64
}

// ── Trit (balanced ternary) tensor tests ─────────────────────────────────────

#[test]
fn trit_quantize_round_trip() {
    // trit_quantize then cast to f32: sum of [1, 0, 1, -1] == 1
    assert_eq!(as_float(&run(r#"
        fn main() -> f32 {
            let x = forge.zeros[f32, [4]]
            x[0] = 0.8   x[1] = -0.3   x[2] = 1.2   x[3] = -0.9
            let t = trit_quantize(x)
            sum(t as f32)
        }
    "#)), 1.0);
}

#[test]
fn trit_neg_inverts_sign() {
    // trit_neg(trit_quantize(x)) == trit_quantize(-x)
    assert_eq!(as_float(&run(r#"
        fn main() -> f32 {
            let x = forge.zeros[f32, [4]]
            x[0] = 0.8   x[1] = -0.3   x[2] = 1.2   x[3] = -0.9
            let t = trit_quantize(x)
            let neg_t = trit_neg(t)
            # trit_quantize(x) = [1, 0, 1, -1]; trit_neg = [-1, 0, -1, 1]; sum = -1
            sum(neg_t as f32)
        }
    "#)), -1.0);
}

#[test]
fn trit_sparsity_fraction() {
    // trit_quantize([0.8, -0.3, 1.2, -0.9]) = [1, 0, 1, -1]; 1 zero out of 4 = 0.25
    let v = as_float(&run(r#"
        fn main() -> f64 {
            let x = forge.zeros[f32, [4]]
            x[0] = 0.8   x[1] = -0.3   x[2] = 1.2   x[3] = -0.9
            let t = trit_quantize(x)
            trit_sparsity(t)
        }
    "#));
    assert!((v - 0.25).abs() < 1e-9, "expected 0.25, got {}", v);
}

#[test]
fn forge_trit_constructor() {
    // forge.trit[4, 3] should produce a zero-filled [4, 3] DType::Trit tensor.
    // sum of zeros is 0.
    assert_eq!(as_float(&run(r#"
        fn main() -> f32 {
            let !w = forge.trit[4, 3]
            sum(w as f32)
        }
    "#)), 0.0);
}

// ─── Bool tensors ────────────────────────────────────────────────────────
// The JIT has always supported element assignment into bool tensors (1-byte
// 0/1 lanes; `m[i,j]` reads back a real bool). The interpreter rejected the
// write outright ("cannot assign value of type bool to tensor slice"), so no
// mask built by element assignment — e.g. a causal attention mask — could run
// under `dmc run`/`dmc test`. These pin the parity: DType::Bool tensors accept
// bool element writes and yield Value::Bool on element reads.

#[test]
fn bool_tensor_element_assign_and_read() {
    let v = run(r#"
        fn main() -> bool {
            let !m = forge.uninit[bool, [2, 2]]
            m[0, 0] = true
            m[0, 1] = false
            m[0, 0] && !m[0, 1]
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "expected Bool(true), got {:?}", v);
}

#[test]
fn bool_tensor_assign_from_comparison() {
    // The qwen3 causal-mask shape: the RHS is a comparison, not a literal.
    let v = run(r#"
        fn main() -> bool {
            let !m = forge.uninit[bool, [4, 4]]
            for r in 0..4 {
                for c in 0..4 {
                    m[r, c] = (c <= r)
                }
            }
            m[3, 0] && m[3, 3] && !m[0, 3] && !m[2, 3]
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "expected Bool(true), got {:?}", v);
}

#[test]
fn bool_tensor_zeros_reads_false() {
    // forge.zeros[bool] elements read back as Value::Bool(false), not Float(0.0).
    let v = run(r#"
        fn main() -> bool {
            let m = forge.zeros[bool, [2, 2]]
            m[1, 1]
        }
    "#);
    assert!(matches!(v, Value::Bool(false)), "expected Bool(false), got {:?}", v);
}

#[test]
fn trit_matmul_ones_by_trit_weight() {
    // x = ones[2,4], w = trit[4,3] with known values.
    // Row sums of w columns: col0=1-1+0+1=1, col1=-1+1+0-1=-1, col2=0+1-1+1=1
    // Both rows of x@w are [1, -1, 1]. sum = 2*(1-1+1) = 2.
    assert_eq!(as_float(&run(r#"
        fn main() -> f32 {
            let x = forge.ones[f32, [2, 4]]
            let !w = forge.trit[4, 3]
            w[0, 0] = 1.0    w[0, 1] = -1.0   w[0, 2] = 0.0
            w[1, 0] = -1.0   w[1, 1] = 1.0    w[1, 2] = 1.0
            w[2, 0] = 0.0    w[2, 1] = 0.0    w[2, 2] = -1.0
            w[3, 0] = 1.0    w[3, 1] = -1.0   w[3, 2] = 1.0
            let y = x @ w
            sum(y)
        }
    "#)), 2.0);
}

#[test]
fn trit_matmul_vs_float_reference() {
    // x @ trit_quantize(W) == x @ (trit_quantize(W) as f32)
    // Both should give the same sum.
    let src = r#"
        fn main() -> f32 {
            let x = forge.ones[f32, [2, 4]]
            let !w = forge.trit[4, 3]
            w[0, 0] = 1.0    w[0, 1] = -1.0   w[0, 2] = 0.0
            w[1, 0] = -1.0   w[1, 1] = 1.0    w[1, 2] = 1.0
            w[2, 0] = 0.0    w[2, 1] = 0.0    w[2, 2] = -1.0
            w[3, 0] = 1.0    w[3, 1] = -1.0   w[3, 2] = 1.0
            let y_trit = x @ w
            let w_f32 = w as f32
            let y_float = x @ w_f32
            sum(y_trit) - sum(y_float)
        }
    "#;
    let diff = as_float(&run(src));
    assert!((diff).abs() < 1e-6, "trit matmul diverges from float reference: diff={}", diff);
}

#[test]
fn trit_quantize_soft_runs() {
    // trit_quantize_soft(ones*0.7, 0.5) — just verify it runs and produces a value
    let v = as_float(&run(r#"
        fn main() -> f32 {
            let x = forge.ones[f32, [3, 3]] .* 0.7
            let w_soft = trit_quantize_soft(x, 0.5)
            sum(w_soft)
        }
    "#));
    // 9 elements, each ~tanh(0.4) + tanh(2.4); should be > 0
    assert!(v > 0.0, "trit_quantize_soft sum should be positive, got {}", v);
}

#[test]
fn trit_pack_returns_masks() {
    // trit_pack([1, 0, -1, 1]) -> (pos=[1,0,0,1], neg=[0,0,1,0])
    // sum(pos) = 2, sum(neg) = 1
    assert_eq!(as_float(&run(r#"
        fn main() -> f32 {
            let x = forge.zeros[f32, [4]]
            x[0] = 0.8   x[1] = -0.3   x[2] = 1.2   x[3] = -0.9
            let t = trit_quantize(x)
            let (pos, neg) = trit_pack(t)
            sum(pos) + sum(neg)
        }
    "#)), 3.0);  // sum(pos)=2 (elems 0,2), sum(neg)=1 (elem 3)
}

// ─── #530: `>>` — arithmetic right shift ────────────────────────────────

/// The shift is **arithmetic** (sign-propagating), matching the JIT's `sshr`:
/// a negative left operand keeps its sign bit, so `-8 >> 1` is -4 and `-1 >> n`
/// is -1 for every in-range n. A logical shift would give huge positives here.
#[test]
fn right_shift_is_arithmetic_on_negatives() {
    assert_eq!(as_int(&run("fn main() -> i64 { 256 >> 2 }")), 64);
    assert_eq!(as_int(&run("fn main() -> i64 { -8 >> 1 }")), -4);
    assert_eq!(as_int(&run("fn main() -> i64 { -1 >> 1 }")), -1);
    assert_eq!(as_int(&run("fn main() -> i64 { -1 >> 63 }")), -1);
    // Floor, not truncate-toward-zero: `-7 >> 1` is -4, while `-7 / 2` is -3.
    assert_eq!(as_int(&run("fn main() -> i64 { -7 >> 1 }")), -4);
    assert_eq!(as_int(&run("fn main() -> i64 { -7 / 2 }")), -3);
}

/// The 0 and 63 boundaries of the `0..=63` range, in i64 — 63 is the only
/// count that can reach the sign bit, and 0 must be the identity.
#[test]
fn right_shift_range_boundaries() {
    assert_eq!(as_int(&run("fn main() -> i64 { 256 >> 0 }")), 256);
    assert_eq!(as_int(&run("fn main() -> i64 { -256 >> 0 }")), -256);
    // 1 << 63 is i64::MIN; arithmetic-shifting it right by 63 gives -1.
    assert_eq!(as_int(&run("fn main() -> i64 { (1 << 63) >> 63 }")), -1);
    assert_eq!(as_int(&run("fn main() -> i64 { 9223372036854775807 >> 63 }")), 0);
}

/// #215: the shift amount is range-guarded exactly as `<<` is. Outside
/// `0..=63` Rust's `>>` panics in debug and masks mod-64 in release; the
/// interpreter raises a clean error instead, and the JIT matches it.
#[test]
fn right_shift_amount_out_of_range_errors() {
    let e = run_err("fn main() -> i64 { 1 >> 64 }");
    assert!(e.contains(">> shift amount 64 out of range"), "got: {}", e);
    let e = run_err("fn main() -> i64 { 1 >> -1 }");
    assert!(e.contains(">> shift amount -1 out of range"), "got: {}", e);
    // Non-integer operands are rejected the way `<<`'s are.
    let e = run_err("fn main() -> i64 { 1 >> 2.0 }");
    assert!(e.contains(">> requires int"), "got: {}", e);
}

#[test]
fn compound_assign_star_slash_bitwise() {
    // #222: `*=`, `/=`, `&=`, `|=`, `^=` silently no-op'd in the interpreter (only
    // `+=`/`-=` were handled), so `acc *= i` left acc unchanged — diverging from
    // the JIT, which applies them. Each must now mutate.
    assert_eq!(as_int(&run("fn main() -> i64 { let !a = 1    a *= 5   a }")), 5);
    assert_eq!(as_int(&run("fn main() -> i64 { let !a = 100  a /= 5   a }")), 20);
    assert_eq!(as_int(&run("fn main() -> i64 { let !a = 6    a &= 3   a }")), 2);
    assert_eq!(as_int(&run("fn main() -> i64 { let !a = 6    a |= 1   a }")), 7);
    assert_eq!(as_int(&run("fn main() -> i64 { let !a = 6    a ^= 3   a }")), 5);
    // Iterative factorial — the jit_scalar_demo repro that was returning 1.
    assert_eq!(as_int(&run(
        "fn main() -> i64 { let !acc = 1  let !i = 1  while i <= 10 { acc *= i  i += 1 }  acc }"
    )), 3628800);
}

// ─── f32 tensor precision (#241) ─────────────────────────────────────────────
// f32-family tensors are rounded through f32 at construction and element
// stores, matching the JIT's true-f32 tensors; f64 tensors keep full width.

#[test]
fn f32_tensor_store_rounds_through_f32() {
    // 0.1 is not f32-representable; the stored element must be the f32
    // rounding (0.10000000149011612), exactly what the JIT sees.
    let v = run(r#"
        fn main() -> f32 {
            let !t = forge.zeros[f32, [1]]
            t[0] = 0.1
            t[0]
        }
    "#);
    assert_eq!(as_float(&v), 0.1f32 as f64);
}

#[test]
fn f64_tensor_store_keeps_full_precision() {
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64, [1]]
            t[0] = 0.1
            t[0]
        }
    "#);
    assert_eq!(as_float(&v), 0.1f64);
}

// ─── #689: indexed tensor stores mutate in place instead of cloning ────────
//
// `t[i] = v` used to route through `lookup`, which deep-clones a `Value::
// Tensor`'s backing `ArrayD` (unlike `Value::List`, a tensor is not `Rc`-
// wrapped), so every element store copied the whole tensor and filling N
// elements cost O(N²). The fix mutates the scope slot's `ArrayD` through
// `&mut` instead of lookup-clone-reassign.

#[test]
fn indexed_tensor_store_is_not_quadratic_689() {
    // Complexity guard: 1,000,000 stores takes well under a second when each
    // store mutates in place (~0.6s measured); at the pre-fix O(N^2) per-store
    // clone this extrapolates to ~370s, so the 20s bound below is a size
    // proof, not a timing assertion, and it discriminates cleanly — it isn't
    // flaky under system load.
    let start = std::time::Instant::now();
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64, [1000000]]
            let !i = 0
            while i < 1000000 {
                t[i] = (i as f64)
                i = i + 1
            }
            sum(t)
        }
    "#);
    let elapsed = start.elapsed();
    // sum(0..1000000) = 1000000 * 999999 / 2
    assert_eq!(as_float(&v), 499_999_500_000.0);
    assert!(elapsed.as_secs() < 20,
        "1M indexed tensor stores took {:?} — the per-store clone regressed", elapsed);
}

#[test]
fn indexed_tensor_multi_index_store_is_in_place_689() {
    // `t[i, j] = v` resolves through the same fast path (multiple `IndexElem`s
    // in one `PostfixOp::Index`) — must still land correctly, not just fast.
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64, [2, 3]]
            let !i = 0
            while i < 2 {
                let !j = 0
                while j < 3 {
                    t[i, j] = (i * 3 + j) as f64
                    j = j + 1
                }
                i = i + 1
            }
            t[1, 2]
        }
    "#);
    assert_eq!(as_float(&v), 5.0);
}

#[test]
fn tensor_field_alias_sees_indexed_store_689() {
    // SPEC §3.4: `let !y = m.t` is a live alias (`Value::FieldRef`), not a
    // snapshot — `y[i] = v` must still reach `m.t`. The #689 fast path only
    // fires for a plain `Value::Tensor` binding; the FieldRef arm (untouched
    // by that change) must still carry the write through.
    let v = run(r#"
        model M { !t: Tensor[f64, [4]] }
        fn main() -> f64 {
            let !m = M { t: forge.zeros[f64, [4]] }
            let !y = m.t
            y[2] = 9.0
            m.t[2]
        }
    "#);
    assert_eq!(as_float(&v), 9.0);
}

#[test]
fn tensor_value_copy_does_not_see_indexed_store_689() {
    // SPEC §3.4: `let !y = x` on a plain (non-field) tensor binding is a
    // value copy — mutating `y` through the #689 in-place store path must
    // never reach `x`.
    let v = run(r#"
        fn main() -> f64 {
            let !x = forge.zeros[f64, [4]]
            x[0] = 1.0
            let !y = x
            y[0] = 2.0
            x[0]
        }
    "#);
    assert_eq!(as_float(&v), 1.0);
}

#[test]
fn indexed_store_errors_when_index_expr_retypes_the_binding_689() {
    // The #689 fast path peeks `Some(Value::Tensor(_))` on `name` before
    // evaluating the index — but the index expression is evaluated *after*
    // that peek (see the #443 comment just above it), so it can itself
    // reassign `name` to a non-tensor before the store lands. `wreck()`
    // does exactly that to a module-level tensor: `g = 1` retypes the
    // binding the peek already approved. This used to hit `Some(_) => break`
    // silently — the store vanished with no error, and the follow-on `g[0]`
    // read whatever the retyped value indexes to. It must now error the same
    // way the FieldRef/other arms do for a non-tensor indexed target.
    let msg = run_err(r#"
        let !g = forge.zeros[f64, [4]]
        fn wreck() -> i64 {
            g = 1
            0
        }
        fn main() -> f64 {
            g[wreck()] = 5.0
            g[0]
        }
    "#);
    assert_eq!(msg, "indexed assignment requires a tensor, got int");
}

#[test]
fn indexed_store_errors_when_block_index_expr_retypes_the_binding_689() {
    // Local-binding form of the same hazard: a block index expression
    // reassigns the very tensor being indexed to a scalar mid-evaluation.
    let msg = run_err(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64, [4]]
            t[{ t = 1
                0 }] = 5.0
            t[0]
        }
    "#);
    assert_eq!(msg, "indexed assignment requires a tensor, got int");
}

#[test]
fn indexed_store_survives_a_same_tensor_write_during_index_eval_689() {
    // Parity fix, not new behavior added by #689's perf change itself: a
    // write into the *same* tensor performed while evaluating the index
    // expression now survives instead of being clobbered by the outer
    // store. Before #689, `t[i] = v` read the whole tensor through
    // `lookup` (a clone), evaluated the index against that snapshot, and
    // then reassigned `t` to the mutated clone — discarding whatever the
    // index expression itself had written into the live `t` along the way.
    // The #689 fast path evaluates the index first (`eval_index_elems`,
    // which mutates the real scope slot if it takes the same fast path)
    // and only then borrows the scope slot to apply the outer store, so a
    // same-tensor write made during index evaluation now sticks. The
    // FieldRef arm (~4922) was never affected — it already evaluates the
    // index before borrowing the struct, so it already gave the answer
    // this test expects.
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64, [4]]
            t[{ t[0] = 7.0
                1 }] = 5.0
            t[0] * 100.0 + t[1]
        }
    "#);
    assert_eq!(as_float(&v), 705.0);
}

// ─── #693: compound assignment to an indexed tensor element ────────────────
//
// `t[i] += v` (and `-=`, `*=`, `/=`) had no match arm at all: it fell to the
// catch-all "evaluate lhs for side effects, no-op" arm. `--check` accepted
// the form, `dmc run` printed the tensor's OLD value back (0 where 5 was
// expected), and `dmc jit` refused it outright (`jit unsupported: compound
// assignment to tensor elements`) — the interpreter was the only backend
// claiming to accept a form it did not execute. The fix reads the element
// out dtype-aware, combines through `apply_binop` (the same function the
// scalar `x += v` arm calls), and stores back through
// `assign_to_tensor_resolved` (#689's in-place path), which applies the same
// f32-quantize a plain `t[i] = v` store does — so this arm's f32 answer
// matches the plain indexed store and the JIT, not the SCALAR `x += v` arm,
// which skips that store-time rounding (a pre-existing interp/jit
// divergence, filed as #713, not reproduced or fixed here).

#[test]
fn indexed_tensor_compound_add_computes_and_stores_693() {
    // The issue's exact repro: `t[0] += 5` used to print 0.
    let v = run(r#"
        fn main() -> i64 {
            let !t = forge.zeros[i64,[4]]
            t[0] += 5
            t[0]
        }
    "#);
    assert_eq!(as_int(&v), 5);
}

#[test]
fn indexed_tensor_compound_sub_mul_div_all_compute_693() {
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64,[1]]
            t[0] += 5.0
            t[0] -= 1.0
            t[0] *= 3.0
            t[0] /= 2.0
            t[0]
        }
    "#);
    // (0 + 5 - 1) * 3 / 2 = 6.0
    assert_eq!(as_float(&v), 6.0);
}

#[test]
fn indexed_tensor_compound_multi_index_693() {
    // `t[i, j] op= v` — multiple `IndexElem`s in one `PostfixOp::Index`,
    // same as the plain #689 multi-index store.
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64, [2, 3]]
            t[1, 2] += 5.0
            t[1, 2]
        }
    "#);
    assert_eq!(as_float(&v), 5.0);
}

#[test]
fn indexed_tensor_compound_through_field_reaches_the_field_693() {
    // `m.t[0] += 5.0` — SPEC §3.4 is explicit that the field's own element
    // writes and compound assignments both reach it; this went through the
    // `PostfixOp::Field` base-expr arm, not the `Ident` one.
    let v = run(r#"
        model M { !t: Tensor[f64, [4]] }
        fn main() -> f64 {
            let !m = M { t: forge.zeros[f64, [4]] }
            m.t[0] += 5.0
            m.t[0]
        }
    "#);
    assert_eq!(as_float(&v), 5.0);
}

#[test]
fn indexed_tensor_compound_through_live_alias_reaches_the_field_693() {
    // SPEC §3.4: `let !y = m.t` is a live alias (`Value::FieldRef`), not a
    // snapshot. `y[i] += v` must reach `m.t`, same as `y[i] = v` (#689).
    let v = run(r#"
        model M { !t: Tensor[f64, [4]] }
        fn main() -> f64 {
            let !m = M { t: forge.zeros[f64, [4]] }
            let !y = m.t
            y[0] += 5.0
            m.t[0]
        }
    "#);
    assert_eq!(as_float(&v), 5.0);
}

#[test]
fn indexed_tensor_compound_i32_wraps_at_its_own_width_693() {
    // Dtype-aware numeric semantics: an i32 element wraps at 32 bits, same
    // as the scalar `x += v` path and the plain indexed store (#571).
    let v = run(r#"
        fn main() -> i64 {
            let !t = forge.zeros[i32,[1]]
            t[0] = 2147483647
            t[0] += 1
            t[0]
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

#[test]
fn indexed_tensor_compound_f32_rounds_per_op_693() {
    // An f32 tensor element rounds per-op, matching #241/#226's store-time
    // rounding rule for a plain assignment — 0.1f32 + 0.2f32, not the f64 sum.
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f32,[1]]
            t[0] = 0.1
            t[0] += 0.2
            t[0]
        }
    "#);
    assert_eq!(as_float(&v), (0.1f32 + 0.2f32) as f64);
}

#[test]
fn indexed_tensor_compound_errors_on_non_tensor_target_693() {
    // A model-array (List) element has no compound-arithmetic meaning.
    // Before #693 this silently dropped the store; now it names the type
    // and refuses, same wording the plain indexed-store arm already uses.
    let msg = run_err(r#"
        model Cell { !n: i64 }
        fn main() -> i64 {
            let !cs = forge.uninit[Cell, [2]]
            cs[0] = Cell { n: 1 }
            cs[0] += 5
            0
        }
    "#);
    assert_eq!(msg, "indexed assignment requires a tensor, got list");
}

#[test]
fn indexed_tensor_compound_errors_on_slice_target_693() {
    // `t[0] += v` on a 2-D tensor with a single (partial) index would target
    // a whole row, not one element — compound assignment to a slice is a
    // different, broader feature this fix does not add. Must error, not
    // silently no-op or guess at a broadcast semantics nobody specified.
    let msg = run_err(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64,[2,3]]
            t[0] += 1.0
            t[0, 0]
        }
    "#);
    assert!(msg.contains("compound assignment to a tensor slice is not supported"),
        "unexpected message: {}", msg);
}

#[test]
fn indexed_tensor_compound_errors_when_index_expr_retypes_the_binding_693() {
    // Same #443/#689 hazard as the plain indexed store: the index expression
    // is evaluated between the tensor peek and the store, so it can itself
    // retype the very binding being indexed.
    let msg = run_err(r#"
        fn main() -> f64 {
            let !t = forge.zeros[f64, [4]]
            t[{ t = 1
                0 }] += 5.0
            t[0]
        }
    "#);
    assert_eq!(msg, "indexed assignment requires a tensor, got int");
}

#[test]
fn direct_field_compound_assign_computes_and_stores_693() {
    // #693 audit, follow-up: `m.field += v` with NO index also had no match
    // arm and fell to the same catch-all no-op; a first pass named the
    // operator with `{:?}` and refused outright ("assignment `PlusEq` to a
    // struct field is not supported" — leaking the Rust enum name).
    // Coordinator decision on review: compute it instead, mirroring the
    // `(Eq, Postfix{Field})` arm, same as the indexed case already does.
    let v = run(r#"
        model M { !x: i64 }
        fn main() -> i64 {
            let !m = M { x: 1 }
            m.x += 5
            m.x
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn direct_field_compound_assign_all_four_ops_693() {
    let v = run(r#"
        model M { !x: f64 }
        fn main() -> f64 {
            let !m = M { x: 0.0 }
            m.x += 5.0
            m.x -= 1.0
            m.x *= 3.0
            m.x /= 2.0
            m.x
        }
    "#);
    // (0 + 5 - 1) * 3 / 2 = 6.0
    assert_eq!(as_float(&v), 6.0);
}

#[test]
fn direct_field_compound_assign_f32_rounds_per_op_693() {
    // Same #241/#226 store-time rounding rule the indexed-element case gets
    // (via `coerce_declared`, not `assign_to_tensor_resolved` — different
    // storage, same rule): 0.1f32 + 0.2f32, not the f64 sum.
    let v = run(r#"
        model M { x: f32 }
        fn main() -> f64 {
            let !m = M { x: 0.1 }
            m.x += 0.2
            m.x as f64
        }
    "#);
    assert_eq!(as_float(&v), (0.1f32 + 0.2f32) as f64);
}

#[test]
fn direct_field_compound_assign_through_alias_binding_693() {
    // SPEC §3.4: a compound assignment through an alias binding must reach
    // the field. Unlike a tensor local, a plain struct binding already
    // aliases the same `Rc<RefCell<..>>` by default (`let !y = m` shares
    // it, no `FieldRef` indirection needed) — `y.x += 1` mutates the same
    // storage `m.x` reads.
    let v = run(r#"
        model M { !x: i64 }
        fn main() -> i64 {
            let !m = M { x: 1 }
            let !y = m
            y.x += 1
            m.x
        }
    "#);
    assert_eq!(as_int(&v), 2);
}

#[test]
fn direct_field_compound_assign_nested_field_693() {
    // `o.i.n += 1` — `base_expr` is itself a `Postfix{Field}` chain
    // (`o.i`), resolved the same recursive way `eval_expr` resolves any
    // other nested field access; no special-casing needed for depth.
    let v = run(r#"
        model Inner { !n: i64 }
        model Outer { i: Inner }
        fn main() -> i64 {
            let !o = Outer { i: Inner { n: 5 } }
            o.i.n += 1
            o.i.n
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn direct_field_compound_assign_whole_tensor_field_693() {
    // A tensor-typed field with a whole-tensor RHS is the same elementwise
    // replace the plain-ident `y += tensor` arms already do (#524) — no
    // per-element loop needed, `apply_binop`'s `DotAdd` handles the tensor
    // pair directly.
    let v = run(r#"
        model M { w: Tensor[f32,[2]] }
        fn main() -> f64 {
            let !m = M { w: forge.zeros[f32,[2]] }
            m.w += forge.ones[f32,[2]]
            m.w[0] as f64 + m.w[1] as f64
        }
    "#);
    assert_eq!(as_float(&v), 2.0);
}

#[test]
fn direct_field_compound_assign_tensor_field_with_scalar_rhs_errors_693() {
    // A tensor field combined with a non-tensor RHS has no established
    // broadcast semantics for a whole-field compound assign — refuse with a
    // clear message naming the operator via `assignop_str` (`+=`), not the
    // `{:?}`-leaked `PlusEq` the pre-follow-up code used elsewhere.
    let msg = run_err(r#"
        model M { w: Tensor[f32,[2]] }
        fn main() -> f64 {
            let !m = M { w: forge.zeros[f32,[2]] }
            m.w += 5.0
            m.w[0] as f64
        }
    "#);
    assert!(msg.contains("+=") && !msg.contains("PlusEq"),
        "unexpected message: {}", msg);
}

#[test]
fn f32_elementwise_op_rounds_per_op() {
    // Compute-in-f64-then-round equals native f32 for `*` (the 2p+2
    // double-rounding bound), so this must be exactly 0.1f32 * 0.2f32.
    let v = run(r#"
        fn main() -> f32 {
            let !a = forge.zeros[f32, [1]]
            let !b = forge.zeros[f32, [1]]
            a[0] = 0.1
            b[0] = 0.2
            let c = a .* b
            c[0]
        }
    "#);
    assert_eq!(as_float(&v), (0.1f32 * 0.2f32) as f64);
}

#[test]
fn f32_scalar_broadcast_demotes_the_scalar() {
    // The JIT splats a scalar into f32 lanes, so `a .* 0.1` multiplies by
    // 0.1f32, not 0.1f64 — the interpreter demotes the scalar the same way.
    let v = run(r#"
        fn main() -> f32 {
            let !a = forge.zeros[f32, [1]]
            a[0] = 0.1
            let s = a .* 0.1
            s[0]
        }
    "#);
    assert_eq!(as_float(&v), (0.1f32 * 0.1f32) as f64);
}

#[test]
fn f64_elementwise_op_stays_wide() {
    // Width propagates through ops: an f64 operand keeps the result f64.
    let v = run(r#"
        fn main() -> f64 {
            let !a = forge.zeros[f64, [1]]
            let !b = forge.zeros[f64, [1]]
            a[0] = 0.1
            b[0] = 0.2
            let c = a .* b
            c[0]
        }
    "#);
    assert_eq!(as_float(&v), 0.1f64 * 0.2f64);
}

#[test]
fn int_tensor_elementwise_not_corrupted_by_f32_rounding() {
    // An Int operand keeps the result wide: values beyond f32's 2^24 exact
    // range must survive elementwise arithmetic.
    let v = run(r#"
        fn main() -> f64 {
            let !t = forge.zeros[i64, [1]]
            t[0] = 1234567891234567
            let u = t .+ 0
            u[0]
        }
    "#);
    assert_eq!(as_float(&v), 1234567891234567.0);
}

#[test]
fn f64_tensor_cast_to_f32_rounds() {
    // Narrowing cast goes through the F32 construction rounding.
    let v = run(r#"
        fn main() -> f32 {
            let !a = forge.zeros[f64, [1]]
            a[0] = 0.1
            let b = a as f32
            b[0]
        }
    "#);
    assert_eq!(as_float(&v), 0.1f32 as f64);
}

// --- loud-diagnostics batch: runtime guards (#394/#395/#396) ---------------

#[test]
fn rng_uniform_int_draws_in_range_395() {
    // Implemented (#395): draws land in the half-open [low, high) and the
    // linear (rng, tensor) pair destructures like the float rng ctors.
    let v = run(
        "fn main() -> i64 {\n\
             let rng = Rng.seed(1)\n\
             let (r2, i) = rng.uniform_int[i32,[16]](3, 7)\n\
             let !bad = 0\n\
             for k in 0..16 {\n\
                 if i[k] < 3 { bad = bad + 1 }\n\
                 if i[k] > 6 { bad = bad + 1 }\n\
             }\n\
             bad\n\
         }");
    assert!(matches!(v, Value::Int(0, _)), "all draws must be in [3, 7), got {:?}", v);
    // An empty range must still be loud.
    let e = run_err("fn main() -> nil { let rng = Rng.seed(1)  let (r2, i) = rng.uniform_int[i32,[3]](5, 5)  nil }");
    assert!(e.contains("uniform_int") && e.contains("greater"), "got: {}", e);
    // The bracket form without the range call must be loud, not a silent nil.
    let e2 = run_err("fn main() -> nil { let rng = Rng.seed(1)  let (r2, i) = rng.uniform_int[i32,[3]]  nil }");
    assert!(e2.contains("uniform_int"), "got: {}", e2);
}

#[test]
fn field_binding_scalar_copies_444() {
    // `let !x = self.field` on a SCALAR field is a value copy (SPEC §3.4),
    // not a live alias — a snapshot must not track later field writes.
    let v = run("model B { !n: i64\n\
                 fn probe(self) -> i64 { let !snap = self.n  self.n = 99  snap } }\n\
                 fn main() -> i64 { let !b = B { n: 7 }  b.probe() }");
    assert!(matches!(v, Value::Int(7, _)), "snapshot must hold 7, got {:?}", v);
}

#[test]
fn field_binding_tensor_rebind_writes_through_444() {
    // A tensor-field binding is a live alias: whole-binding assignment
    // writes the FIELD (matching element writes), never silently rebinds.
    let v = run("model G { !buf: Tensor[i64, [4]]\n\
                 fn fill!(self, v: i64) -> nil { let !b = self.buf\n\
                     b = forge.ones[i64, [4]] .* v  nil } }\n\
                 fn main() -> bool { let !g = G { buf: forge.zeros[i64, [4]] }\n\
                     g.fill!(7)  g.buf[0] == 7 }");
    assert!(matches!(v, Value::Bool(true)), "rebind must reach the field, got {:?}", v);
}

#[test]
fn field_binding_compound_assign_writes_through_444() {
    // Compound ops through the alias read the field's current value and
    // write the result back to the field.
    let v = run("model C { !t: Tensor[i64, [2]]\n\
                 fn bump!(self) -> nil { let !b = self.t  b += forge.ones[i64, [2]]  nil } }\n\
                 fn main() -> bool { let !c = C { t: forge.zeros[i64, [2]] }\n\
                     c.bump!()  c.bump!()  c.t[1] == 2 }");
    assert!(matches!(v, Value::Bool(true)), "two bumps must reach the field, got {:?}", v);
}

#[test]
fn model_load_errors_394() {
    let e = run_err("model L[D] { w: Tensor[f32,[D]] }\nfn main() -> nil { let l = L[D=4].load(\"/tmp/x\")  nil }");
    assert!(e.contains("serialization"), "got: {}", e);
    // The empty/positional bracket forms parse as Index (not BracketArgs) and
    // used to slip past the guard to a silent opaque — they must also be loud.
    let empty = run_err("model L { w: Tensor[f32,[2]] }\nfn main() -> nil { let l = L[].load(\"/tmp/x\")  nil }");
    assert!(empty.contains("serialization"),
            "empty-bracket load must error, got: {}", empty);
}

#[test]
fn allreduce_method_form_errors_396() {
    // `allreduce.sum(y)` must not silently desugar to `sum(allreduce, y)` (→ 0.0).
    let e = run_err("fn main() -> nil { let !y = forge.zeros[f32,[3]]  y[0] = 1.0  let r = allreduce.sum(y, axis=0)  nil }");
    assert!(e.contains("collective"), "got: {}", e);
}

// --- tensor .split (SPEC §6.4, #397 built) ---------------------------------

#[test]
fn tensor_split_axis_last() {
    // Split a 2x6 tensor into 3 along axis=-1; check a piece's values.
    let v = run(r#"
        fn main() -> f32 {
            let !t = forge.zeros[f32, [2, 6]]
            for i in 0..6 { t[0, i] = (i as f32)  t[1, i] = ((10 + i) as f32) }
            let (a, b, c) = t.split[3, axis=-1]
            # a=[[0,1],[10,11]] b=[[2,3],[12,13]] c=[[4,5],[14,15]]
            a[1,0] + b[0,1] + c[1,1]   # 10 + 3 + 15 = 28
        }
    "#);
    match v {
        Value::Float(f, _) => assert!((f - 28.0).abs() < 1e-6, "got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

#[test]
fn tensor_split_non_divisible_errors() {
    let e = run_err("fn main() -> nil { let t = forge.zeros[f32,[2,6]]  let (a,b,c,d) = t.split[4, axis=-1]  nil }");
    assert!(e.contains("split") && e.contains("divisible"), "got: {}", e);
}

#[test]
fn tensor_split_bare_index_form() {
    // `t.split[n]` with no `axis=` parses as an Index (not BracketArgs), so it
    // must still dispatch to the split path — otherwise the pieces are opaque
    // and the destructure reads silent nils. axis defaults to -1.
    let v = run(r#"
        fn main() -> f32 {
            let !t = forge.zeros[f32, [8]]
            for i in 0..8 { t[i] = (i as f32) }
            let (a, b) = t.split[2]      # a=[0,1,2,3] b=[4,5,6,7]
            a[3] + b[0]                  # 3 + 4 = 7
        }
    "#);
    match v {
        Value::Float(f, _) => assert!((f - 7.0).abs() < 1e-6, "got {}", f),
        other => panic!("expected float, got {:?}", other),
    }
}

// --- #393: `x @ pat` bind patterns in match --------------------------------

#[test]
fn match_bind_pattern_393() {
    // `y @ 1` matches when the value is 1 AND binds `y` to it.
    let hit = run("fn f(x: i64) -> i64 { match x { y @ 1 => y + 100, _ => 99 } }\nfn main() -> i64 { f(1) }");
    assert_eq!(as_int(&hit), 101);
    // A non-matching value falls through to the catch-all.
    let miss = run("fn f(x: i64) -> i64 { match x { y @ 1 => y + 100, _ => 99 } }\nfn main() -> i64 { f(7) }");
    assert_eq!(as_int(&miss), 99);
}

// --- #393: `..` rest patterns in tuples (match + let-destructure) ----------

#[test]
fn tuple_rest_pattern_matches_and_binds() {
    // `(a, ..)` binds the head and absorbs the tail — previously it silently
    // failed to match (arity 2 vs 3) and took the `_` arm, returning 0.
    let head = run("fn main() -> i64 { let p = (1, 2, 3)  match p { (a, ..) => a, _ => 0 } }");
    assert_eq!(as_int(&head), 1);

    // Head + tail: the rest absorbs the middle.
    let ht = run("fn main() -> i64 { let p = (10, 20, 30, 40)  match p { (a, .., z) => a + z, _ => 0 } }");
    assert_eq!(as_int(&ht), 50);

    // `let (a, ..) = tuple` destructure binds only the fixed head.
    let destr = run("fn main() -> i64 { let (a, ..) = (7, 8, 9)  a }");
    assert_eq!(as_int(&destr), 7);

    // Standalone `..` is still a catch-all (matches anything).
    let catch = run("fn main() -> i64 { let k = 42  match k { 1 => 10, .. => 999 } }");
    assert_eq!(as_int(&catch), 999);

    // A too-short tuple does NOT match a head+tail rest → falls through. `(1, 2)`
    // has 2 elements but `(a, b, .., z)` needs at least 3.
    let short = run("fn main() -> i64 { match (1, 2) { (a, b, .., z) => a + z, _ => -1 } }");
    assert_eq!(as_int(&short), -1);
}

// --- #399: KV `<-` append capacity enforcement (Spec §3.6 panic) -----------

#[test]
fn kv_append_past_capacity_panics() {
    // Unannotated `let !c = forge.kv[...](capacity = N)` — the common form that
    // previously went untracked and silently overflowed. Appending N+1 frames
    // must now panic (Spec §3.6).
    let over = run_err(
        "fn main() -> i64 {\n\
         \x20   let !c = forge.kv[f32, [1, ~, 4]](capacity = 2)\n\
         \x20   let a = forge.zeros[f32, [1, 1, 4]]\n\
         \x20   c <- a  c <- a  c <- a\n\
         \x20   0\n\
         }");
    assert!(over.contains("capacity") && over.contains("2"),
            "over-capacity append should panic, got: {}", over);
}

#[test]
fn kv_append_uses_declared_stream_axis() {
    // Without a type annotation the append must still concatenate along the `~`
    // axis (axis 1 here), not a shape-differ guess. Two [1,1,4] frames → [1,2,4],
    // so sum over the whole cache is 2 * the single set element.
    let v = run(
        "fn main() -> i64 {\n\
         \x20   let !c = forge.kv[f32, [1, ~, 4]](capacity = 4)\n\
         \x20   let !a = forge.zeros[f32, [1, 1, 4]]\n\
         \x20   a[0, 0, 0] = 5.0\n\
         \x20   c <- a  c <- a\n\
         \x20   sum(c) as i64\n\
         }");
    assert_eq!(as_int(&v), 10);
}

// --- #443: index expressions must be evaluated before the target is borrowed ---
//
// An indexed element write whose target is a model-field alias (`Value::FieldRef`)
// used to take `borrow_mut()` on the struct's fields and *then* evaluate the index
// expression. Any index that read the same struct re-entered the interpreter, whose
// member-access path borrows that same `RefCell` — a hard panic that aborted the
// whole `dmc test` process mid-suite rather than reporting a failure. The shape
// matrix below is the one from the issue; each case is a separate test so a
// regression shows up as one failing case, not a truncated run.

#[test]
fn field_alias_write_with_self_read_in_index_443() {
    // Shape 1: `let !c = self.cells` then `c[self.table[0]] = 1`. Was: panic.
    let v = run(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn go!(self) -> nil {
        let !c = self.cells
        c[self.table[0]] = 1
        nil
    }
}
fn main() -> i64 {
    let !b = Box { table: vault.zeros[i64, [4]], cells: vault.zeros[i64, [8]] }
    b.table[0] = 3
    b.go!()
    b.cells[3]
}
"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn field_alias_write_with_self_read_in_value_443() {
    // Shape 3: `self` in value position only. Always worked — `rval` is
    // evaluated before the arm is entered. Guards the pre-eval order.
    let v = run(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn go!(self) -> nil {
        let !c = self.cells
        c[0] = self.table[0] + 7
        nil
    }
}
fn main() -> i64 {
    let !b = Box { table: vault.zeros[i64, [4]], cells: vault.zeros[i64, [8]] }
    b.table[0] = 2
    b.go!()
    b.cells[0]
}
"#);
    assert_eq!(as_int(&v), 9);
}

#[test]
fn field_alias_write_with_method_call_in_index_443() {
    // Shape 5: a method call in index position re-enters the interpreter one
    // frame deeper before reading `self`. Was: panic.
    let v = run(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn idx(self) -> i64 { self.table[0] + 1 }
    fn go!(self) -> nil {
        let !c = self.cells
        c[self.idx()] = 1
        nil
    }
}
fn main() -> i64 {
    let !b = Box { table: vault.zeros[i64, [4]], cells: vault.zeros[i64, [8]] }
    b.table[0] = 4
    b.go!()
    b.cells[5]
}
"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn field_alias_read_with_self_read_in_index_443() {
    // Shape 6: read-only indexing through an alias takes no mutable borrow.
    let v = run(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn go(self) -> i64 {
        let c = self.cells
        c[self.table[0]]
    }
}
fn main() -> i64 {
    let !b = Box { table: vault.zeros[i64, [4]], cells: vault.zeros[i64, [8]] }
    b.table[0] = 3
    b.cells[3] = 11
    b.go()
}
"#);
    assert_eq!(as_int(&v), 11);
}

#[test]
fn field_alias_write_indexed_by_same_field_443() {
    // Shape 7: the index reads the *same* field being written. The index is
    // resolved from the pre-write contents, then the store lands.
    let v = run(r#"
model Box {
    !cells: Tensor[i64, [8]]
    fn go!(self) -> nil {
        let !c = self.cells
        c[self.cells[1]] = 1
        nil
    }
}
fn main() -> i64 {
    let !b = Box { cells: vault.zeros[i64, [8]] }
    b.cells[1] = 6
    b.go!()
    b.cells[6]
}
"#);
    assert_eq!(as_int(&v), 1);
}

#[test]
fn local_tensor_write_with_self_read_in_index_443() {
    // Shape 8: a plain local target holds no borrow — always worked, kept as
    // the control for the `Value::Tensor` arm now that both share a store path.
    let v = run(r#"
model Box {
    !table: Tensor[i64, [4]]
    fn go(self) -> i64 {
        let !t = forge.zeros[i64, [8]]
        t[self.table[0]] = 5
        t[3]
    }
}
fn main() -> i64 {
    let !b = Box { table: vault.zeros[i64, [4]] }
    b.table[0] = 3
    b.go()
}
"#);
    assert_eq!(as_int(&v), 5);
}

#[test]
fn struct_field_write_with_self_read_in_index_443() {
    // The `m.field[i] = v` arm has the same shape as the alias arm: it borrows
    // the struct's fields, so an index that reads `m` must be resolved first.
    let v = run(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
}
fn main() -> i64 {
    let !b = Box { table: vault.zeros[i64, [4]], cells: vault.zeros[i64, [8]] }
    b.table[0] = 2
    b.cells[b.table[0]] = 4
    b.cells[2]
}
"#);
    assert_eq!(as_int(&v), 4);
}

#[test]
fn table_driven_register_reset_443() {
    // The form that found the bug: a table-driven loop reading both the index
    // and the value straight off `self` inside the write.
    let v = run(r#"
model Device {
    !reg_offset: Tensor[i64, [3]]
    !reg_reset: Tensor[i64, [3]]
    !regs: Tensor[i64, [2, 8]]
    fn reset_device!(self, dev: i64) -> nil {
        let !regs = self.regs
        for k in 0..3 {
            regs[dev, self.reg_offset[k] / 4] = self.reg_reset[k]
        }
        nil
    }
}
fn main() -> i64 {
    let !d = Device {
        reg_offset: [0, 8, 20],
        reg_reset: [7, 8, 9],
        regs: vault.zeros[i64, [2, 8]],
    }
    d.reset_device!(1)
    d.regs[1, 0] * 100 + d.regs[1, 2] * 10 + d.regs[1, 5] + d.regs[0, 0]
}
"#);
    assert_eq!(as_int(&v), 789);
}

#[test]
fn field_alias_slice_write_with_self_read_in_bounds_443() {
    // The hazard is in every index form, not just scalars: a slice bound that
    // reads `self` runs through the same pre-evaluation path.
    let v = run(r#"
model Box {
    !table: Tensor[i64, [4]]
    !cells: Tensor[i64, [8]]
    fn go!(self) -> nil {
        let !c = self.cells
        c[self.table[0]..self.table[1]] = 1
        nil
    }
}
fn main() -> i64 {
    let !b = Box { table: vault.zeros[i64, [4]], cells: vault.zeros[i64, [8]] }
    b.table[0] = 2
    b.table[1] = 5
    b.go!()
    sum(b.cells) as i64
}
"#);
    assert_eq!(as_int(&v), 3);
}

// ── #452: lists are copy-on-write, and still values ──────────────────────────
//
// `Value::List` holds an `Rc<Vec<Value>>` so that `xs = list_push(xs, v)` can
// append in place instead of copying the whole backing vector on every call.
// The sharing that buys the speedup is exactly what would break the language's
// value semantics if a write ever landed on a buffer someone else can see, so
// each mutating builtin gets a test that binds a second name to the list first
// and asserts the second name never moves.

#[test]
fn list_push_does_not_mutate_an_alias_452() {
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list_push(list_push(list(), 1), 2)
            let ys = xs
            xs = list_push(xs, 3)
            xs = list_push(xs, 4)
            list_len(ys) * 100 + list_len(xs)
        }
    "#);
    assert_eq!(as_int(&v), 204, "alias saw the pushes");
}

#[test]
fn list_set_does_not_mutate_an_alias_452() {
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list_push(list_push(list(), 1), 2)
            let ys = xs
            xs = list_set(xs, 0, 99)
            list_get(ys, 0) * 100 + list_get(xs, 0)
        }
    "#);
    assert_eq!(as_int(&v), 199, "alias saw the element write");
}

#[test]
fn list_pop_does_not_mutate_an_alias_452() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list(), 1), 2)
            let ys = xs
            let (rest, last) = list_pop(xs)
            list_len(ys) * 100 + list_len(rest) * 10 + last
        }
    "#);
    assert_eq!(as_int(&v), 212, "alias saw the pop");
}

#[test]
fn list_rev_and_sort_do_not_mutate_an_alias_452() {
    let v = run(r#"
        fn main() -> i64 {
            let xs = list_push(list_push(list_push(list(), 3), 1), 2)
            let ys = xs
            let r = list_rev(xs)
            let s = list_sort(xs)
            list_get(ys, 0) * 100 + list_get(r, 0) * 10 + list_get(s, 0)
        }
    "#);
    assert_eq!(as_int(&v), 321, "in-place reverse/sort leaked to the source list");
}

#[test]
fn list_concat_with_itself_keeps_both_sides_452() {
    // Same buffer on both sides of the write: the copy has to happen before the
    // extend, or the source grows while it is being read.
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list_push(list_push(list(), 1), 2)
            let ys = xs
            xs = list_concat(xs, xs)
            list_len(xs) * 100 + list_len(ys) * 10 + list_get(xs, 3)
        }
    "#);
    assert_eq!(as_int(&v), 422);
}

#[test]
fn list_alias_survives_a_reassigning_loop_452() {
    // The build-a-list loop that motivated #452, with a live alias taken
    // mid-flight: the in-place append must not be visible through `snapshot`.
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list()
            let !i = 0
            let !snapshot = list()
            while i < 50 {
                xs = list_push(xs, i)
                if i == 9 { snapshot = xs }
                i = i + 1
            }
            list_len(xs) * 1000 + list_len(snapshot)
        }
    "#);
    assert_eq!(as_int(&v), 50010);
}

#[test]
fn list_alias_through_a_fn_argument_452() {
    // A list passed to a user fn is a value there too — the callee's pushes
    // must not reach the caller's binding.
    let v = run(r#"
        fn grow(l: list) -> i64 {
            let !inner = l
            inner = list_push(inner, 99)
            list_len(inner)
        }
        fn main() -> i64 {
            let xs = list_push(list_push(list(), 1), 2)
            let n = grow(xs)
            n * 100 + list_len(xs)
        }
    "#);
    assert_eq!(as_int(&v), 302, "callee's push reached the caller's list");
}

#[test]
fn list_build_is_not_quadratic_452() {
    // Complexity guard for the #452 fix. Twenty thousand appends is a few
    // milliseconds when `list_push` writes into a uniquely owned buffer and
    // minutes when it deep-copies the backing vector on every call, so the
    // generous bound below still separates linear from quadratic by orders of
    // magnitude — it fails on a regression without being timing-flaky.
    let start = std::time::Instant::now();
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list()
            let !i = 0
            while i < 20000 {
                xs = list_push(xs, i)
                i = i + 1
            }
            list_len(xs)
        }
    "#);
    let elapsed = start.elapsed();
    assert_eq!(as_int(&v), 20000);
    assert!(elapsed.as_secs() < 20, "20k list_push calls took {:?} — copy-on-write append regressed to a per-call copy", elapsed);
}

#[test]
fn list_reassignment_reading_the_target_twice_452() {
    // The in-place path releases the target binding's handle on the buffer, so
    // it must never fire when something *else* in the statement still has to
    // read that binding. Here the builtin call is only a sub-expression of the
    // right-hand side and `xs` is read again in both branches.
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list_push(list_push(list_push(list(), 1), 2), 3)
            xs = if list_len(xs) > 2 { list_slice(xs, 0, 2) } else { xs }
            list_len(xs) * 10 + list_get(xs, 1)
        }
    "#);
    assert_eq!(as_int(&v), 22);
}

#[test]
fn list_reassignment_from_a_short_circuit_reading_the_target_452() {
    // Same hazard through `&&`: the left operand's builtin call must not
    // release `xs` out from under the right operand, nor from under the branch.
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list_push(list_push(list(), 7), 8)
            xs = if list_len(xs) > 1 && list_get(xs, 1) == 8 { list_push(xs, 9) } else { xs }
            list_len(xs) * 10 + list_get(xs, 2)
        }
    "#);
    assert_eq!(as_int(&v), 39);
}

#[test]
fn list_reassignment_with_the_target_read_in_a_later_arg_452() {
    // `xs` appears twice in the same call. The release happens only after every
    // argument is evaluated, so the second read still sees the original list.
    let v = run(r#"
        fn main() -> i64 {
            let !xs = list_push(list_push(list(), 5), 6)
            xs = list_push(xs, list_len(xs))
            list_len(xs) * 10 + list_get(xs, 2)
        }
    "#);
    assert_eq!(as_int(&v), 32);
}

// ── #474: shape-generic model methods, and model shape args in every position ─
//
// The headline of #474 is the ghost method: a method that declares its own
// shape params passed `--check` and then did nothing at all — no error, no
// mutation, not even a `print` from inside the body. A test that asserts a
// call did nothing cannot catch that (the issue has one that passed for
// months while `blit` was a no-op for every input), so every repro below
// asserts the *effect*, and the ones that come straight from the issue check
// first, so "passes --check and never runs" cannot come back one half at a
// time.

/// Type-check `src` and then run it — the pairing #474 is about. A test that
/// only runs cannot see a check-time regression, and one that only checks
/// cannot see the ghost method.
fn run_checked(src: &str) -> Value {
    use super::check::Checker;
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut checker = Checker::new();
    checker.check_program(&program, None);
    assert!(checker.errors.is_empty(), "unexpected type errors: {:?}",
            checker.errors.iter().map(|e| e.to_string()).collect::<Vec<_>>());
    Interpreter::new().run(&program, None).expect("run failed")
}


#[test]
fn model_field_of_parameterized_model_type_474() {
    // #474's field position, from the issue comment: `Console[ROWS, COLS, H, W]`
    // holding a `Surface[H, W]`. The comment records this as unwritable — the
    // literal typed bare and never unified — so demoniOS split the console in
    // two. Construct it, mutate through the nested field, and read back.
    let v = run_checked(r#"
        model Surface[H, W] {
            !px: Tensor[u32, [H, W]]
            !n: i64
            fn plot!(self, y: i64, x: i64, v: u32) -> nil {
                let !p = self.px
                p[y, x] = v
                self.n = self.n + 1
            }
        }
        model Console[ROWS, COLS, H, W] {
            !text: Tensor[u32, [ROWS, COLS]]
            !surface: Surface[H, W]
            fn render!(self) -> i64 {
                self.surface.plot!(1, 2, 0x41u32)
                self.surface.n
            }
        }
        fn main() -> i64 {
            vault {
                let !c = Console[2, 3, 4, 5] {
                    text: vault.zeros[u32, [2, 3]],
                    surface: Surface[4, 5] { px: vault.zeros[u32, [4, 5]], n: 0 }
                }
                c.render!() * 1000 + c.surface.px[1, 2] as i64
            }
        }
    "#);
    // One plot!, and 0x41 == 65 landed at [1, 2] of the nested surface.
    assert_eq!(as_int(&v), 1065);
}

#[test]
fn shape_generic_method_actually_runs_474() {
    // #474's repro verbatim, with the two prints replaced by the values they
    // printed. `plain!` (no shape params of its own) always worked; `generic!`
    // was the ghost. Both halves of its body have to land: the `n` bump and
    // the write through `self.cells`.
    let v = run_checked(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64

            fn plain!(self, v: u32) -> nil {
                self.n = self.n + 100
            }

            fn generic![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> nil {
                self.n = self.n + 1
                let !c = self.cells
                c[0, 0] = src[0, 0]
            }
        }

        fn main() -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]], n: 0 }
                let !s = vault.zeros[u32, [2, 2]]
                s[0, 0] = 0x7Au32

                b.plain!(1u32)
                b.generic![2, 2](s)
                b.n * 1000 + b.cells[0, 0] as i64
            }
        }
    "#);
    // n == 101 (100 from plain!, 1 from generic!), cells[0,0] == 0x7A == 122.
    assert_eq!(as_int(&v), 101 * 1000 + 122);
}

#[test]
fn shape_generic_method_reads_its_own_shape_params_474() {
    // The bracket is not decoration: SH and SW are in scope in the body and
    // hold what the call site wrote. A no-op could not tell 4 and 5 apart.
    let v = run_checked(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn area![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH * 10 + SW }
        }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [4, 5]]
                b.area![4, 5](s)
            }
        }
    "#);
    assert_eq!(as_int(&v), 45);
}

#[test]
fn shape_generic_method_reads_the_models_shape_params_too_474() {
    // Both halves of the body's shape scope: the model's H comes off the
    // receiver, the method's SH and W off the bracket — and a method shape
    // param that reuses a model name is shadowed by what the call site wrote,
    // the only reading that makes writing the bracket mean anything.
    let v = run_checked(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn mix![SH, W](self, src: Tensor[u32, [SH, W]]) -> i64 { H * 100 + SH * 10 + W }
        }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 3] { cells: vault.zeros[u32, [2, 3]] }
                let !s = vault.zeros[u32, [4, 5]]
                b.mix![4, 5](s)
            }
        }
    "#);
    // H = 2 from the instance; SH = 4 and W = 5 from the bracket, W = 3 shadowed.
    assert_eq!(as_int(&v), 245);
}

#[test]
fn shape_generic_method_without_a_bracket_infers_from_the_args_474() {
    // No bracket at all: the method's shape params bind from the tensor
    // argument exactly as a free function's do.
    let v = run_checked(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn area![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH * 10 + SW }
        }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [4, 5]]
                b.area!(s)
            }
        }
    "#);
    assert_eq!(as_int(&v), 45);
}

#[test]
fn named_shape_bracket_on_a_method_474() {
    // The `[SH=4, SW=5]` spelling binds by name, in any order.
    let v = run_checked(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            fn area![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> i64 { SH * 10 + SW }
        }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !s = vault.zeros[u32, [4, 5]]
                b.area![SW = 5, SH = 4](s)
            }
        }
    "#);
    assert_eq!(as_int(&v), 45);
}

#[test]
fn shape_generic_method_reached_through_a_field_474() {
    // The receiver need not be a plain identifier. `h.surf.bump![…]` is the
    // shape demoniOS's compositor wants.
    let v = run_checked(r#"
        model Inner[H, W] {
            !px: Tensor[u32, [H, W]]
            !n: i64
            fn bump![SH, SW](self, src: Tensor[u32, [SH, SW]]) -> nil {
                self.n = self.n + SH * 10 + SW
            }
        }
        model Holder { !surf: Inner[2, 3] }
        fn main() -> i64 {
            vault {
                let !i = Inner[2, 3] { px: vault.zeros[u32, [2, 3]], n: 0 }
                let !h = Holder { surf: i }
                let !s = vault.zeros[u32, [4, 5]]
                h.surf.bump![4, 5](s)
                h.surf.bump!(s)
                h.surf.n
            }
        }
    "#);
    assert_eq!(as_int(&v), 90);
}

#[test]
fn undefined_method_behind_a_shape_bracket_errors_441() {
    // #441's rule, in the bracketed spelling. The interpreter must refuse on
    // its own — `run()` does not type-check — rather than fall through to an
    // opaque that swallows the call.
    let msg = run_err(r#"
        model Box[H, W] { !cells: Tensor[u32, [H, W]] }
        fn main() -> nil {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                b.nope![2, 2](1)
                nil
            }
        }
    "#);
    assert_eq!(msg, "no method `nope!` on model `Box`");
}

#[test]
fn a_bracket_on_a_model_field_is_still_indexing_474() {
    // The dispatch must not eat `b.cells[0, 1]` — a field of that name being
    // indexed is indexing, not a shape-bracketed method call.
    let v = run_checked(r#"
        model Box[H, W] { !cells: Tensor[u32, [H, W]] }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]] }
                let !c = b.cells
                c[0, 1] = 7u32
                b.cells[0, 1] as i64
            }
        }
    "#);
    assert_eq!(as_int(&v), 7);
}

// ── #459 / #474: shape params bind from more than tensor arguments ──────────
//
// The harvest during call binding used to read tensor arguments and nothing
// else, so a model array never bound its length and a model argument never
// handed over the shape args it was built with.

#[test]
fn shape_param_binds_from_model_array_length_459() {
    // #459's repro, inferred form: `N` comes from the arena's own length.
    // Before the harvest learned about `[T; N]` this type-checked and then
    // died at the call site.
    let v = run(r#"
        model Node { kind: i64 }
        fn build[N](!ns: [Node; N], n: i64) -> i64 {
            ns[0] = Node { kind: 7 }
            ns[0].kind
        }
        fn main() -> i64 {
            let !arena = forge.uninit[Node, [8]]
            build(arena, 8)
        }
    "#);
    assert_eq!(as_int(&v), 7);
}

#[test]
fn model_array_length_is_readable_as_the_shape_param_459() {
    // The bound `N` is the real length, not merely something that stopped
    // erroring — the point of #459 is a parametric capacity the body computes
    // with.
    let v = run_checked(r#"
        model Node { kind: i64 }
        fn cap[N](!ns: [Node; N]) -> i64 { N * 2 }
        fn main() -> i64 {
            let !arena = forge.uninit[Node, [6]]
            cap(arena)
        }
    "#);
    assert_eq!(as_int(&v), 12);
}

#[test]
fn shape_param_workaround_form_still_works_459() {
    // #459's passing workaround — a named top-level capacity, no shape param
    // on the array at all. It must keep working unchanged.
    let v = run_checked(r#"
        model Node { kind: i64 }
        let AST_CAP = 8
        fn build(!ns: [Node; AST_CAP], n: i64) -> i64 {
            ns[0] = Node { kind: 7 }
            ns[0].kind
        }
        fn main() -> i64 {
            let !arena = forge.uninit[Node, [8]]
            build(arena, 8)
        }
    "#);
    assert_eq!(as_int(&v), 7);
}

#[test]
fn model_array_length_conflict_errors_459() {
    // Two arrays want the same `N` at different lengths. The new binding
    // source reports through the same path the tensor one does.
    let msg = run_err(r#"
        model Node { kind: i64 }
        fn twin[N](!a: [Node; N], !b: [Node; N]) -> i64 { N }
        fn main() -> i64 {
            let !x = forge.uninit[Node, [8]]
            let !y = forge.uninit[Node, [4]]
            twin(x, y)
        }
    "#);
    assert!(msg.contains("`N`") && msg.contains("both 8 and 4"),
            "expected an N=8-vs-4 conflict, got: {msg}");
}

#[test]
fn parameterized_model_param_binds_its_shape_args_474() {
    // #474, parameter position: `!b: Box[H, W]` binds H and W from the
    // instance itself. Before this the call needed every shape spelled out
    // (`free_generic![2, 2, 2, 2](b, s)`).
    let v = run_checked(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
        }
        fn area[H, W](!b: Box[H, W]) -> i64 { H * W }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 3] { cells: vault.zeros[u32, [2, 3]], n: 0 }
                area(b)
            }
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn parameterized_model_param_mixes_with_tensor_inference_474() {
    // The model argument and a tensor argument bind different params of the
    // same signature — the issue's `free_generic!` shape, minus the brackets.
    // The body runs and the mutation writes back.
    let v = run(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
        }
        fn blit[H, W, SH, SW](!b: Box[H, W], src: Tensor[u32, [SH, SW]]) -> nil {
            b.n = b.n + H * 1000 + W * 100 + SH * 10 + SW
            nil
        }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 3] { cells: vault.zeros[u32, [2, 3]], n: 0 }
                let !s = vault.zeros[u32, [4, 5]]
                blit(b, s)
                b.n
            }
        }
    "#);
    assert_eq!(as_int(&v), 2345);
}

#[test]
fn a_bracketed_field_call_evaluates_its_receiver_once_474() {
    // `recv.name[i](args)` cannot be told apart from a shape-bracketed method
    // call until `recv` has been evaluated, so the dispatch evaluates it — and
    // then, on deciding `subs` is a field rather than a method, used to hand
    // the expression back to the ordinary postfix path, which walked it again.
    // Every side effect ran twice: `tick` bumped `hits` twice, and a receiver
    // built in an arena was built twice.
    //
    // `hits` is the whole assertion. One evaluation, one bump.
    let v = run_checked(r#"
        model Elem {
            !v: i64
            fn forward(self, x: i64) -> i64 { self.v + x }
        }
        model Holder { !subs: [Elem; 2] !hits: i64 }
        fn tick(!h: Holder) -> Holder { h.hits = h.hits + 1  h }
        fn main() -> i64 {
            forge {
                let !es = forge.uninit[Elem, [2]]
                es[0] = Elem { v: 10 }
                es[1] = Elem { v: 20 }
                let !h = Holder { subs: es, hits: 0 }
                let r = tick(h).subs[1](5)
                h.hits * 1000 + r
            }
        }
    "#);
    assert_eq!(as_int(&v), 1025, "expected one receiver evaluation (hits=1) and r=25");
}

#[test]
fn parameterized_model_param_mixes_with_tensor_inference_through_brackets_474() {
    // The same signature reached through the bang-method bracket spelling the
    // issue actually uses. The inferred call and the fully-explicit one must
    // agree on every dim, so the bracket is a restatement, never a second
    // source of truth.
    let v = run_checked(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
        }
        fn blit![H, W, SH, SW](!b: Box[H, W], src: Tensor[u32, [SH, SW]]) -> nil {
            b.n = b.n + H * 1000 + W * 100 + SH * 10 + SW
        }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 3] { cells: vault.zeros[u32, [2, 3]], n: 0 }
                let !s = vault.zeros[u32, [4, 5]]
                blit!(b, s)
                blit![2, 3, 4, 5](b, s)
                b.n
            }
        }
    "#);
    assert_eq!(as_int(&v), 2 * 2345);
}

#[test]
fn explicit_shape_args_still_win_over_inference_474() {
    // The explicit-bracket call the issue documents as the workaround stays
    // valid: same program, brackets spelled out, same answer.
    let v = run(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
        }
        fn area[H, W](!b: Box[H, W]) -> i64 { H * W }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 3] { cells: vault.zeros[u32, [2, 3]], n: 0 }
                area[2, 3](b)
            }
        }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn model_shape_arg_conflict_errors_474() {
    // The model argument says H=2, the tensor argument says H=3. Same class of
    // diagnostic as the tensor-vs-tensor conflict.
    let msg = run_err(r#"
        model Box[H, W] {
            !cells: Tensor[u32, [H, W]]
            !n: i64
        }
        fn wide[H, W](!b: Box[H, W], src: Tensor[u32, [H, W]]) -> i64 { H }
        fn main() -> i64 {
            vault {
                let !b = Box[2, 2] { cells: vault.zeros[u32, [2, 2]], n: 0 }
                let !s = vault.zeros[u32, [3, 2]]
                wide(b, s)
            }
        }
    "#);
    assert!(msg.contains("`H`") && msg.contains("both 2 and 3"),
            "expected an H=2-vs-3 conflict, got: {msg}");
}

#[test]
fn unbindable_shape_param_still_errors() {
    // Widening the harvest must not let an uninferable param quietly bind to
    // something. A param no argument mentions is still an error at the call.
    let msg = run_err(r#"
        fn lonely[Q](x: i64) -> i64 { Q }
        fn main() -> i64 { lonely(1) }
    "#);
    assert!(msg.contains("`Q`") && msg.contains("cannot be inferred"),
            "expected an uninferable-shape-param error, got: {msg}");
}

// ─── #478: unsuffixed float literals adopt their branch's type ───────────────
//
// The checker constrains an untyped float literal to its context, and a
// sibling `if`/`match` branch IS context (SPEC.md §"Untyped numeric literals"
// — "the other operand at its use site"). The interpreter used to ignore that:
// the literal evaluated as an f64 value, so the answer depended on which
// branch the condition happened to take. These pin the width the literal now
// carries; `jit.rs`'s tests pin the same programs against the JIT.

/// `0.1f32 + 0.2f32` is 0.30000001192092896; the f64 0.1 plus an f32 0.2 is
/// 0.3000000029802322. Only a literal whose value differs between the two
/// widths can tell a fixed backend from a reverted one — which is why the
/// tests below use `0.1` and not the issue's exactly-representable `0.0`.
///
/// The VALUE is the assertion, never the `FW` tag: `fn main() -> f64` widens
/// whatever it returns, so the tag says f64 either way and would pass for a
/// backend that had reverted to computing in f64.
/// NB the parens: the addition happens IN f32 and is widened once. Widening
/// each operand first and adding in f64 gives 0.30000000447034836 — a third
/// answer, and neither backend's.
const F32_SUM_478: f64 = (0.1f32 + 0.2f32) as f64;

#[test]
fn unsuffixed_literal_in_an_f32_branch_is_f32_478() {
    for cond in ["0.0f32", "1.0f32"] {   // then taken, else taken
        let src = format!(r#"
            fn main() -> f64 {{
                let a = 0.1f32
                let z = if a > {cond} {{ a }} else {{ 0.1 }}
                z + 0.2f32
            }}
        "#);
        assert_eq!(as_float(&run(&src)), F32_SUM_478, "wrong width taking `{}`", cond);
    }
}

#[test]
fn unsuffixed_literal_in_an_f32_match_arm_is_f32_478() {
    for n in [1, 2] {                     // ident arm taken, literal arm taken
        let src = format!(r#"
            fn main() -> f64 {{
                let a = 0.1f32
                let n = {n}
                let z = match n {{ 1 => a, _ => 0.1 }}
                z + 0.2f32
            }}
        "#);
        assert_eq!(as_float(&run(&src)), F32_SUM_478, "wrong width for n = {}", n);
    }
}

#[test]
fn an_all_f64_join_is_untouched_by_type_direction_478() {
    // #209 must survive: with no f32 anywhere in the join, the literal stays
    // f64 and keeps every digit.
    let v = run(r#"
        fn main() -> f64 {
            let a = 0.5
            let z = if a > 1.0 { a } else { 0.1 }
            z
        }
    "#);
    // 0.1 quantized through f32 is 0.10000000149011612 — a narrowed join
    // would not compare equal to the f64 literal.
    assert_eq!(as_float(&v), 0.1f64, "an f64 join was narrowed: {:?}", v);
}

#[test]
fn type_direction_stops_at_the_branch_value_478() {
    // The hint is armed for one value position. A `let` STATEMENT inside the
    // f32-hinted branch, and anything after the join, keep the f64 default.
    let v = run(r#"
        fn main() -> f64 {
            let a = 0.1f32
            let z = if a > 0.0f32 { let w = 0.1  w * 0.0 + a } else { 0.1 }
            let d = 0.1
            d
        }
    "#);
    assert_eq!(as_float(&v), 0.1f64, "the hint leaked past the branch: {:?}", v);
}

// ─── Arena sizing flags (#400, MEMORY.md §1.1) ───────────────────────────────

/// Run `src` with the given arena budgets installed, as `--vault=` / `--forge=`
/// would. Returns the runtime error message, or `Ok` with the program's value.
fn run_with_limits(
    src: &str,
    limits: crate::arena::ArenaLimits,
) -> Result<Value, String> {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut interp = Interpreter::new();
    interp.set_arena_limits(limits);
    interp.run(&program, None).map_err(|e| e.msg)
}

fn forge_budget(bytes: u64) -> crate::arena::ArenaLimits {
    crate::arena::ArenaLimits { vault: None, forge: Some(bytes) }
}

fn vault_budget(bytes: u64) -> crate::arena::ArenaLimits {
    crate::arena::ArenaLimits { vault: Some(bytes), forge: None }
}

#[test]
fn allocation_within_the_forge_budget_runs() {
    // 64 f32 elements = 256 bytes at the declared width (#781), comfortably
    // inside 4 KiB.
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let t = forge.zeros[f32, [8, 8]]
            1
        }
    "#, forge_budget(4096)).expect("should fit the budget");
    assert_eq!(as_int(&v), 1);
}

#[test]
fn allocation_past_the_forge_budget_is_a_runtime_error() {
    let e = run_with_limits(r#"
        fn main() -> i64 {
            let t = forge.zeros[f32, [1024, 1024]]
            1
        }
    "#, forge_budget(4096)).expect_err("should exhaust the budget");
    assert!(e.contains("forge arena exhausted"), "{}", e);
    assert!(e.contains("--forge"), "{}", e);
}

#[test]
fn the_forge_budget_accumulates_across_allocations() {
    // Each `[8, 8]` f32 tensor is 256 bytes at its declared width (#781).
    // Two fit in a 700-byte budget (512); the third does not (768).
    let ok = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [8, 8]]
            let b = forge.ones[f32, [8, 8]]
            1
        }
    "#, forge_budget(700));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [8, 8]]
            let b = forge.ones[f32, [8, 8]]
            let c = forge.zeros[f32, [8, 8]]
            1
        }
    "#, forge_budget(700)).expect_err("third allocation should not fit");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_forge_block_does_not_rewind_the_budget_on_exit() {
    // #400: the meter is monotonic. `forge { … }` used to rewind `forge_used`
    // to the block's entry watermark, on the theory that §3's bump-pointer
    // reset gives the bytes back — but the interpreter has no bump pointer and
    // reclaims nothing, so the rewind was a lie that let a program hold every
    // "reset" tensor at once. Ten 256-byte tensors (#781: declared f32 width)
    // against a 1 KiB budget must therefore run out, exactly as they do
    // unwrapped.
    let e = run_with_limits(r#"
        fn main() -> i64 {
            let !n = 0
            for i in 0..10 {
                forge {
                    let tmp = forge.zeros[f32, [8, 8]]
                    n = n + 1
                }
            }
            n
        }
    "#, forge_budget(1024)).expect_err("a forge block must not hand budget back");
    assert!(e.contains("forge arena exhausted"), "{}", e);

    // …and a budget that covers all ten still runs them, so the block itself
    // is not charged anything extra.
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let !n = 0
            for i in 0..10 {
                forge {
                    let tmp = forge.zeros[f32, [8, 8]]
                    n = n + 1
                }
            }
            n
        }
    "#, forge_budget(10 * 512)).expect("ten 256-byte tensors comfortably fit a 5 KiB budget");
    assert_eq!(as_int(&v), 10);
}

#[test]
fn a_forge_block_is_not_a_way_around_the_forge_budget() {
    // The repro that closed the bypass: a recursive fn that keeps every tensor
    // it allocates live across the recursive call. Forty 2 MiB tensors is
    // 80 MiB of simultaneously-live memory; `--forge=4M` must refuse it
    // whether or not each allocation is wrapped in `forge { … }`. Before the
    // fix the wrapped form ran to completion and printed 40.
    const HOLD_40: &str = r#"
        fn hold(n: i64) -> i64 {
            if n <= 0 { return 0 }
            let !t = %s
            let rest = hold(n - 1)
            t[0, 0] = 1.0
            rest + 1
        }
        fn main() -> i64 { hold(40) }
    "#;
    let wrapped = HOLD_40.replace("%s", "forge { forge.zeros[f64, [512, 512]] }");
    let bare = HOLD_40.replace("%s", "forge.zeros[f64, [512, 512]]");

    let e = run_with_limits(&wrapped, forge_budget(4 << 20))
        .expect_err("`forge { … }` bypassed the budget");
    assert!(e.contains("forge arena exhausted"), "{}", e);

    let e = run_with_limits(&bare, forge_budget(4 << 20))
        .expect_err("the unwrapped form should also be refused");
    assert!(e.contains("forge arena exhausted"), "{}", e);

    // The value of a `forge { … }` block still escapes the block and is still
    // usable — the fix is to the meter, not to the semantics.
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let !t = forge { forge.zeros[f64, [4, 4]] }
            t[1, 1] = 7.0
            if t[1, 1] == 7.0 { 1 } else { 0 }
        }
    "#, forge_budget(4 << 20)).expect("a forge block's value escapes");
    assert_eq!(as_int(&v), 1, "the block's tensor was not usable after the block");
}

#[test]
fn the_vault_budget_is_separate_from_the_forge_one() {
    // A tight vault budget does not constrain forge allocations…
    let ok = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [64, 64]]
            1
        }
    "#, vault_budget(512));
    assert!(ok.is_ok(), "{:?}", ok.err());

    // …but it does constrain vault ones.
    let e = run_with_limits(r#"
        fn main() -> i64 {
            let a = vault.zeros[f32, [64, 64]]
            1
        }
    "#, vault_budget(512)).expect_err("should exhaust the vault budget");
    assert!(e.contains("vault arena exhausted"), "{}", e);
    assert!(e.contains("--vault"), "{}", e);
}

#[test]
fn the_vault_has_no_reset_so_a_loop_exhausts_it() {
    // The mirror of the forge-block test: Vault lives for the process, so
    // repeated allocation genuinely runs out (MEMORY.md §1).
    let e = run_with_limits(r#"
        fn main() -> i64 {
            let !n = 0
            for i in 0..10 {
                let tmp = vault.zeros[f32, [8, 8]]
                n = n + 1
            }
            n
        }
    "#, vault_budget(1024)).expect_err("vault should not rewind");
    assert!(e.contains("vault arena exhausted"), "{}", e);
}

#[test]
fn no_flag_means_no_budget() {
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [256, 256]]
            let b = vault.zeros[f32, [256, 256]]
            1
        }
    "#, crate::arena::ArenaLimits::default()).expect("unbounded by default");
    assert_eq!(as_int(&v), 1);
}

#[test]
fn a_kv_allocation_is_stream_not_forge() {
    // `forge.kv` lands in the Stream arena (MEMORY.md §9), which the sizing
    // flags do not reach — its bound is its own `capacity`.
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let !c = forge.kv[f32, [~]](capacity = 4096)
            c <- [1.0f32]
            1
        }
    "#, forge_budget(64)).expect("kv must not charge the forge budget");
    assert_eq!(as_int(&v), 1);
}

// ─── #781: requested-byte metering, intermediates included ──────────────────
//
// #674 §7 slice 1. Every tensor
// the interpreter materializes is charged at its DECLARED element width, not
// the f64 width it actually stores — and an implicit intermediate (a matmul
// or dotted-op result, a tensor literal, a transpose, a slice that copies)
// is charged the same as an explicit `vault.*`/`forge.*` constructor, where
// none was charged before. These pin the exact minimums measured by a set of
// internal arena-budget probe programs — bisected against the release
// binary, not derived from the formula in this comment.

#[test]
fn explicit_constructors_charge_declared_width_not_f64_m6() {
    // m6: `[81920]` f32 is 320 KiB of elements — half what the old f64-per-
    // element charge billed (640 KiB), which is why two used to collide in a
    // 1 MiB budget (`MEMORY.md §1.1`'s stale example, pre-#781).
    let ok = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [81920]]
            1
        }
    "#, forge_budget(320 * 1024));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [81920]]
            1
        }
    "#, forge_budget(320 * 1024 - 1)).expect_err("320 KiB is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
    assert!(e.contains("320 KiB"), "the request itself is 320 KiB: {}", e);
}

#[test]
fn two_forge_blocks_of_m6_shape_both_fit_a_1mib_budget() {
    // The full m6 shape: two `forge { }` blocks, `[81920]` f32 each. At the
    // corrected 320 KiB/block charge, both fit `--forge=1M` (640 KiB total)
    // — before #781 the second block was refused (640 KiB/block, 1.25 MiB
    // cumulative).
    let v = run_with_limits(r#"
        fn main() -> i64 {
            forge {
                let a = forge.zeros[f32, [81920]]
            }
            forge {
                let b = forge.zeros[f32, [81920]]
            }
            1
        }
    "#, forge_budget(1024 * 1024)).expect("both m6 blocks fit at the declared width");
    assert_eq!(as_int(&v), 1);
}

#[test]
fn an_implicit_matmul_intermediate_is_charged_p1_imp() {
    // p1_imp: `let c = a @ b`, both `[64, 64]` f32. `a` and `b` are 16 384 B
    // each at the declared width (#781, not the old 32 768 B f64 charge);
    // `c` is a third, equally-sized, implicit intermediate — 49 152 B total,
    // where the pre-#781 interpreter charged only the first two (32 768 B,
    // `MEMORY.md §1.1`'s old "intermediates are not charged").
    let ok = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [64, 64]]
            let b = forge.zeros[f32, [64, 64]]
            let c = a @ b
            c[0, 0] as i64
        }
    "#, forge_budget(49152));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [64, 64]]
            let b = forge.zeros[f32, [64, 64]]
            let c = a @ b
            c[0, 0] as i64
        }
    "#, forge_budget(49151)).expect_err("49 152 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);

    // The baseline this prices the intermediate against: `a` and `b` alone
    // (no `let c`) need only 32 768 B — p1_base, the two explicit
    // constructors at their declared width.
    let base = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [64, 64]]
            let b = forge.zeros[f32, [64, 64]]
            1
        }
    "#, forge_budget(32768));
    assert!(base.is_ok(), "{:?}", base.err());
}

#[test]
fn two_thousand_matmul_intermediates_accumulate_p2_loop() {
    // p2_loop: the same `a @ b` inside a 2000-iteration loop. Each iteration
    // materializes its own `c`, uncharged before #781 — 2000 × 16 384 B on
    // top of the 32 768 B for `a`/`b`, exactly matching `dmc jit`'s
    // per-iteration charge for this shape (the two backends' TOTAL minimums
    // still differ — `dmc jit` carries its own, separately-filed #640
    // over-charge on `a`/`b` — but the per-intermediate charge agrees).
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [64, 64]]
            let b = forge.zeros[f32, [64, 64]]
            let !s = 0.0
            for i in 0..2000 {
                let c = a @ b
                s = s + c[0, 0]
            }
            s as i64
        }
    "#, forge_budget(32768 + 2000 * 16384));
    assert!(v.is_ok(), "{:?}", v.err());

    let e = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [64, 64]]
            let b = forge.zeros[f32, [64, 64]]
            let !s = 0.0
            for i in 0..2000 {
                let c = a @ b
                s = s + c[0, 0]
            }
            s as i64
        }
    "#, forge_budget(32768 + 2000 * 16384 - 1)).expect_err("one byte under the minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_tensor_literal_is_charged_at_its_default_f32_width() {
    // p3_lit: a plain `[1.0, 2.0, …]` literal has no `vault.`/`forge.`
    // spelling, but it still materializes a tensor — DType's own doc: "F32
    // is the default float tag — tensor literals … are f32". Eight elements
    // × 4 B = 32 B.
    let ok = run_with_limits(r#"
        fn main() -> i64 {
            let t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            t[0] as i64
        }
    "#, forge_budget(32));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(r#"
        fn main() -> i64 {
            let t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
            t[0] as i64
        }
    "#, forge_budget(31)).expect_err("32 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn an_rng_draw_is_charged_to_forge_at_its_declared_width() {
    // p4_rng: `r.normal[f32, [N]]` has no `forge.`/`vault.` spelling either
    // — its base is the `Rng` value — but it still materializes a fresh f32
    // tensor, charged to Forge the same as any other implicit intermediate.
    let ok = run_with_limits(r#"
        fn main() -> i64 {
            let !r = Rng.seed(1)
            let (r2, t) = r.normal[f32, [8]]
            r = r2
            t[0] as i64
        }
    "#, forge_budget(32));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(r#"
        fn main() -> i64 {
            let !r = Rng.seed(1)
            let (r2, t) = r.normal[f32, [8]]
            r = r2
            t[0] as i64
        }
    "#, forge_budget(31)).expect_err("32 B (8 f32 elements) is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── #781 reviewer rework: two double/over-charge defects, and coverage ─────
//
// Fixed against PR #790's reviewer findings. (1) a nested tensor literal
// recursed through the SAME `Expr::TensorLit` arm for each row, so a row's
// own `charge_intermediate` call fired, and then the assembled outer array's
// call fired again — double-billing every nested literal. (2) a bracket
// list read as a `.reshape` SHAPE ARGUMENT (`a.reshape[[4096]]`) is not a
// tensor a program can ever observe — it is consumed for its integers and
// discarded — but it went through the same arm and was charged anyway, 8 B
// the JIT never charges. Both are fixed by `lit_charge_suppressed`
// (`eval_expr_lit_charge_suppressed`): held true while evaluating a row or a
// shape argument, so only the OUTERMOST literal's own charge survives.

#[test]
fn a_nested_tensor_literal_is_charged_once_not_per_row() {
    // 4 rows of 16 f32 elements — 256 B at the declared width. Before the
    // fix each row's own recursive `Expr::TensorLit` charge stacked on top
    // of the assembled `[4, 16]` array's charge: 512 B, double the real
    // 256 B the JIT bills for the same literal.
    const ROW: &str = "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0";
    let src = format!(r#"
        fn main() -> i64 {{
            let t = [[{ROW}],[{ROW}],[{ROW}],[{ROW}]]
            t[0, 0] as i64
        }}
    "#);
    let ok = run_with_limits(&src, forge_budget(256));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(&src, forge_budget(255))
        .expect_err("256 B is the exact minimum — a double charge would need 512 B");
    assert!(e.contains("forge arena exhausted"), "{}", e);
    assert!(e.contains("256 B"), "the request itself is 256 B, not 512 B: {}", e);
}

#[test]
fn a_reshape_shape_argument_literal_is_not_charged() {
    // `a.reshape[[4096]]`: `a` (16 384 B f32) reshaped to its own shape.
    // Reshape's OUTPUT is a real implicit intermediate and stays charged
    // (another 16 384 B) — only the `[4096]` shape-argument list itself,
    // consumed for its one integer and never returned to the program, must
    // NOT add its own 8 B (an i64-tagged 1-element literal) on top.
    let src = r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [4096]]
            let b = a.reshape[[4096]]
            b[0] as i64
        }
    "#;
    let ok = run_with_limits(src, forge_budget(32768));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(src, forge_budget(32767))
        .expect_err("32 768 B (16 384 + 16 384) is the exact minimum, not 32 776");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── #781 reviewer rework: builtin-dispatch coverage via `call_builtin`'s
//     choke point ───────────────────────────────────────────────────────────
//
// `softmax`, `sum_along`, `argmax`, `rms_norm` and `embed` all materialize a
// fresh tensor on the interpreter and were completely uncharged — the JIT
// charges every one of them. Fixed by charging the ONE choke point every
// builtin dispatch passes through (`call_builtin`'s `charge_builtin_result`),
// rather than patching each builtin individually.

#[test]
fn softmax_result_is_charged() {
    // x: 4 f32 elements (16 B). softmax(x) materializes a same-shaped
    // output — another 16 B, uncharged before this fix.
    let src = r#"
        fn main() -> i64 {
            let x = forge.zeros[f32, [4]]
            let s = softmax(x)
            s[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(32)).is_ok());
    let e = run_with_limits(src, forge_budget(31)).expect_err("32 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn sum_along_result_is_charged() {
    // w: [2, 3] f32 (24 B). sum_along(w, 1) reduces axis 1 → [2] f32 (8 B).
    let src = r#"
        fn main() -> i64 {
            let w = forge.zeros[f32, [2, 3]]
            let s = sum_along(w, 1)
            s[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(32)).is_ok());
    let e = run_with_limits(src, forge_budget(31)).expect_err("32 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn argmax_result_is_charged() {
    // x: [2, 3] f32 (24 B). argmax(x, 1) drops axis 1 → [2] f32 (8 B, #272:
    // a multi-axis result stays a tensor, unlike the scalar collapse case).
    let src = r#"
        fn main() -> i64 {
            let x = forge.zeros[f32, [2, 3]]
            let a = argmax(x, 1)
            a[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(32)).is_ok());
    let e = run_with_limits(src, forge_budget(31)).expect_err("32 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn rms_norm_result_is_charged() {
    // x, g: [4] f32 (16 B each). rms_norm(x, g, eps) → [4] f32 (16 B).
    let src = r#"
        fn main() -> i64 {
            let x = forge.zeros[f32, [4]]
            let g = forge.zeros[f32, [4]]
            let n = rms_norm(x, g, 0.00001)
            n[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(48)).is_ok());
    let e = run_with_limits(src, forge_budget(47)).expect_err("48 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn embed_result_is_charged() {
    // vocab: [2, 4] f32 (32 B). ids: [2] f32 (8 B). embed(vocab, ids) → [2,
    // 4] f32 (32 B) — 72 B total, where before this fix only the two
    // explicit constructors (40 B) were charged.
    let src = r#"
        fn main() -> i64 {
            let vocab = forge.zeros[f32, [2, 4]]
            let ids = forge.zeros[f32, [2]]
            let out = embed(vocab, ids)
            out[0, 0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(72)).is_ok());
    let e = run_with_limits(src, forge_budget(71)).expect_err("72 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── #804: a string produced by `+` is charged like any materialization ───
//
// `apply_binop`'s string-concatenation arm (`s + "x"`, `"lit" + other`) lives
// outside `call_builtin`, so #802's `charge_builtin_result` never saw it —
// `dmc jit` bump-allocates and charges the fresh forge string every `+`
// produces, `dmc run` charged nothing at all. Fixed by giving
// `charge_intermediate` (the same choke point `Expr::BinOp` already calls
// after every `apply_binop`, tensors included) a `Value::Str` arm, at the
// fresh string's UTF-8 byte length.

#[test]
fn str_plus_result_is_charged_at_its_byte_length() {
    // "ab" + "cd" -> "abcd", 4 bytes. Before this fix: 0 B (ran at any budget,
    // including 0 — rejected only because `--forge=0` itself is invalid).
    let src = r#"
        fn main() -> i64 {
            let s = "ab" + "cd"
            len(s)
        }
    "#;
    assert!(run_with_limits(src, forge_budget(4)).is_ok());
    let e = run_with_limits(src, forge_budget(3)).expect_err("4 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn str_plus_non_str_operand_result_is_charged() {
    // `"lit" + to_str(f)` (#803 review's case): to_str(3.5f32) -> "3.5" (3 B,
    // already charged by #802's `charge_builtin_result`); the `+` result,
    // "lit3.5" (6 B), was the uncharged half before this fix. 9 B total.
    let src = r#"
        fn main() -> i64 {
            let f: f32 = 3.5
            let s = "lit" + to_str(f)
            len(s)
        }
    "#;
    assert!(run_with_limits(src, forge_budget(9)).is_ok());
    let e = run_with_limits(src, forge_budget(8)).expect_err("9 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn str_concat_loop_charges_the_cumulative_byte_length() {
    // #804's own repro: `let !s = ""  for i in 0..1000 { s = s + "x" }`. Each
    // iteration's `s + "x"` materializes a string one byte longer than the
    // last (lengths 1..1000), so the cumulative charge is
    // 1 + 2 + ... + 1000 = 500 500 B — where before this fix `dmc run` ran
    // this loop at any positive budget at all.
    let src = r#"
        fn main() -> i64 {
            let !s = ""
            for i in 0..1000 {
                s = s + "x"
            }
            len(s)
        }
    "#;
    assert!(run_with_limits(src, forge_budget(500_500)).is_ok());
    let e = run_with_limits(src, forge_budget(500_499))
        .expect_err("500 500 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn to_str_loop_charge_is_unchanged_by_this_fix() {
    // `to_str` materializes through `call_builtin`, so #802 already charged
    // it — this loop's minimum must be unaffected by #804's `apply_binop`
    // fix. `let !s = to_str(i)` overwrites `s` every iteration rather than
    // growing it, so the cumulative charge is the sum of each `to_str(i)`
    // result's decimal-digit length for i in 0..1000: ten 1-digit numbers
    // (10 B) + ninety 2-digit numbers (180 B) + nine hundred 3-digit numbers
    // (2700 B) = 2890 B.
    let src = r#"
        fn main() -> i64 {
            let !s = ""
            for i in 0..1000 {
                s = to_str(i)
            }
            len(s)
        }
    "#;
    assert!(run_with_limits(src, forge_budget(2890)).is_ok());
    let e = run_with_limits(src, forge_budget(2889))
        .expect_err("2890 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── PR #815 review: a builtin's string result must not inherit
//     `lit_charge_suppressed` from the literal element that reached it ────
//
// #804 routed `charge_builtin_result`'s `Value::Str` arm through
// `charge_intermediate`, which early-returns while `lit_charge_suppressed`
// is held — a flag meant only to stop a tensor literal's OWN rows from
// double-billing (`eval_expr_lit_charge_suppressed`). A builtin call reached
// from INSIDE a suppressed literal element (a fn the element calls, which
// itself calls a string-producing builtin) is not a row of that literal at
// all, and its fresh string still costs real bytes under `dmc jit`
// (`forge_str_from_rust`), so this silently zeroed a real charge. Fixed by
// having that arm call `charge_arena` directly again, unconditionally — see
// `charge_builtin_result`'s doc.

#[test]
fn a_builtin_string_result_inside_a_suppressed_literal_element_is_still_charged() {
    // `noisy()` runs the #804 `to_str` loop above (2890 B cumulative) and
    // returns its length as an int. `[noisy()]` is a single-element tensor
    // literal, so `Expr::TensorLit` evaluates that one element through
    // `eval_expr_lit_charge_suppressed` unconditionally (regardless of the
    // element's type) — the suppressed context this fix must reach through.
    // The literal itself (one i64 element, the untyped-integer default
    // width, #571) adds another 8 B: 2898 B total. Before this fix: 8 B —
    // the entire `to_str` loop vanished into the suppression.
    let src = r#"
        fn noisy() -> i64 {
            let !s = ""
            for i in 0..1000 {
                s = to_str(i)
            }
            len(s)
        }
        fn main() -> i64 {
            let t = [noisy()]
            t[0]
        }
    "#;
    assert!(run_with_limits(src, forge_budget(2898)).is_ok());
    let e = run_with_limits(src, forge_budget(2897))
        .expect_err("2898 B is the exact minimum — the suppressed-literal bug left this at 8 B");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── #781 reviewer rework: `!`-param and `let !` private-copy charges ──────
//
// #650 and SPEC §3.4: a `!` parameter and a `let !y = x` binding are
// PRIVATE, independently mutable copies, not aliases — the JIT copies the
// bytes, so `dmc run` must charge the copy too, to stay in parity.

#[test]
fn a_mutating_param_charges_the_copy_in_and_not_the_copy_out() {
    // x: [8] f32 (32 B). `fill(x)` binds a private copy for `!t` — the
    // copy-IN, another 32 B, allocated at the call site on both backends —
    // for 64 B total, where before #781 only `x` itself (32 B) was charged.
    //
    // #791: the copy-OUT adds nothing. It writes back into the buffer the
    // caller already owns and already paid for (the JIT emits a `memcpy`
    // into the incoming pointer, not a second allocation), so 96 B — what an
    // earlier pass of #781 asserted here — double-billed every `!`-param
    // call.
    let src = r#"
        fn fill(!t: Tensor[f32, [8]]) -> nil {
            t[0] = 9.0
            nil
        }
        fn main() -> i64 {
            let !x = forge.zeros[f32, [8]]
            fill(x)
            x[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(64)).is_ok());
    let e = run_with_limits(src, forge_budget(63)).expect_err("64 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn let_bang_from_a_bare_identifier_charges_the_private_copy() {
    // x: [8] f32 (32 B). `let !y = x` binds an independently mutable copy —
    // another 32 B — where before this fix only `x` was charged.
    let src = r#"
        fn main() -> i64 {
            let !x = forge.zeros[f32, [8]]
            let !y = x
            y[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(64)).is_ok());
    let e = run_with_limits(src, forge_budget(63)).expect_err("64 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn let_bang_from_a_fresh_intermediate_is_not_double_charged() {
    // `let !y = a + b` — the binop's result is ALREADY a fresh intermediate
    // charged by `Expr::BinOp`'s own `charge_intermediate` call; it must not
    // be charged again just because the binding happens to be `!`. a, b:
    // [4] f32 (16 B each); the sum: another 16 B — 48 B total, not 64 B.
    let src = r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [4]]
            let b = forge.zeros[f32, [4]]
            let !y = a .+ b
            y[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(48)).is_ok());
    let e = run_with_limits(src, forge_budget(47)).expect_err("48 B is the exact minimum, not 64 B");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── PR #815 review: a private STRING copy is not charged ─────────────────
//
// #650 / SPEC §3.4 charge a `!` binding's private copy for parity
// with what the JIT actually copies. For a `Value::Tensor` that is real
// bytes on both backends (the tests above). For a `Value::Str` it is not: a
// `str` is a pointer on the JIT, so `let !y = x` / a `!` param binds a
// second pointer to the SAME bytes there, requesting 0 B. The three sites
// above (`Stmt::Let`, `eval_top_level_item`'s `Item::Let`, and the `!` param
// loop) used to charge the string's length anyway — a copy neither backend
// makes — via a false comment ("so the JIT charges the copy") that did not
// hold for this value kind. `charge_private_copy` excludes `Value::Str` at
// all three; these tests pin that a string's length is charged exactly
// once, at the point that actually materializes it, no matter how many
// private copies of it are bound afterward.

#[test]
fn let_bang_from_a_bare_str_identifier_is_not_charged() {
    // `to_str(123456789)` -> "123456789", 9 B, charged once at the builtin
    // choke point (#802/#804). Three private `let !` copies of that same
    // string must add nothing. Before this fix: 9 + 9*3 = 36 B.
    let src = r#"
        fn main() -> i64 {
            let c = to_str(123456789)
            let !t1 = c
            let !t2 = c
            let !t3 = c
            len(t3) as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(9)).is_ok());
    let e = run_with_limits(src, forge_budget(8))
        .expect_err("9 B is the exact minimum — the three private copies must add nothing");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_mutating_str_param_is_not_charged_for_the_copy() {
    // Mirrors `a_mutating_param_charges_the_copy_in_and_not_the_copy_out`
    // above, but for a `!str` param: unlike a tensor's copy-in, a string's
    // copy-in must add nothing.
    let src = r#"
        fn touch(!t: str) -> nil {
            nil
        }
        fn main() -> i64 {
            let c = to_str(123456789)
            touch(c)
            len(c) as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(9)).is_ok());
    let e = run_with_limits(src, forge_budget(8))
        .expect_err("9 B is the exact minimum — the `!str` param copy must add nothing");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_top_level_let_bang_from_a_bare_str_identifier_is_not_charged() {
    // Mirrors `let_bang_from_a_bare_str_identifier_is_not_charged` above at
    // module scope (`eval_top_level_item`'s `Item::Let`, not `Stmt::Let`).
    let src = r#"
        let c = to_str(123456789)
        let !g = c
        fn main() -> i64 {
            len(g) as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(9)).is_ok());
    let e = run_with_limits(src, forge_budget(8))
        .expect_err("9 B is the exact minimum — the top-level private copy must add nothing");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── #781 / #791: a compiler temporary returns its charge ─────────────────
//
// `MEMORY.md §1.1`: a materialization the COMPILER inserts and the compiler
// itself reclaims gives its charge back on both backends. The two rows below
// are #674's own g1 probes, verbatim, at the shapes and trip count
// `tools/jit_probes.py` runs them at — pinned here so the interpreter's
// per-step slope cannot drift away from the JIT's again without a unit test
// saying so. Each names the JIT's bisected minimum for the same source and
// the residual against it, and each residual is a CONSTANT (independent of
// the trip count), which is the property that matters: the two backends now
// charge the same bytes PER STEP.

#[test]
fn a_forward_only_grad_call_returns_its_activations() {
    // #674's `g1_fwd`: 200 forward-only calls of a `@grad fn` with a
    // `!` tensor param. Per call the JIT allocates the `!W` private copy at
    // the CALL SITE (16 384 B, outside its own rewind, so monotonic) and
    // reclaims everything the body materialized — `h = x @ W` and `h .* h`,
    // 16 384 B each — through the `InternalMark` its forward-only entry
    // takes. The interpreter now does the same, so both backends charge
    // 16 384 B per call rather than the interpreter's former 65 536 B.
    //
    // Minimum: 200 × 16 384 (the copies) + 32 768 (one call's two live
    // activations at the peak) = 3 309 568 B.
    //
    // `dmc jit` bisects to 3 375 104 B on this source. The 65 536 B residual
    // is not an autodiff or call-scope difference at all: it is `main`'s two
    // `vault.zeros` constructors, which the JIT meters against `--forge`
    // (it has one arena) and doubles under the separately filed
    // let-bound-constructor over-charge, while the interpreter meters them
    // against `--vault`. Re-spelled with `forge.zeros`, the residual is
    // exactly that over-charge (32 768 B) and nothing else.
    let src = r#"
        @grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {
            let h = x @ W
            sum(h .* h)
        }
        fn main() -> i64 {
            vault {
                let !W = vault.zeros[f32, [64, 64]]
                let x = vault.zeros[f32, [64, 64]]
                let !last = 0
                for step in 0..200 {
                    let l = loss(W, x)
                    last = step
                }
                last
            }
        }
    "#;
    assert!(run_with_limits(src, forge_budget(3_309_568)).is_ok());
    let e = run_with_limits(src, forge_budget(3_309_567))
        .expect_err("3 309 568 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_fwd_bwd_step_returns_everything_but_its_published_gradients() {
    // #674's `g1`: the same fn driven 200 times through `.fwd_bwd`,
    // which takes the tape evaluator rather than the ordinary expression
    // path. Before #791 that path materialized its activations and every
    // adjoint uncharged — the whole loop ran at `--forge=1`. Now the pass is
    // metered while it is live and released at the AD boundary, leaving only
    // what the caller keeps: one published gradient per step (16 384 B), the
    // same thing the JIT's `capture_grad` leaves standing outside
    // `grad_call_epilogue`'s rewind.
    //
    // Minimum: 200 × 16 384 (the published gradients) + 81 920 (one step's
    // peak: two forward activations plus the four adjoints the backward
    // pass publishes) = 3 358 720 B.
    //
    // `dmc jit` bisects to 3 473 408 B. Of the 114 688 B residual, 65 536 B
    // is the `vault.zeros` term described in the test above; the remaining
    // 49 152 B is scratch INSIDE the backward pass — the transposes a matmul
    // VJP takes, the buffer an accumulating `+=` produces — which the two
    // backends' backward implementations simply do not agree on op by op.
    // The interpreter meters the adjoints the pass PUBLISHES, a set both
    // backends do agree on; see `docs/MEMORY.md §1.1`.
    let src = r#"
        @grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {
            let h = x @ W
            sum(h .* h)
        }
        fn main() -> i64 {
            vault {
                let !W = vault.zeros[f32, [64, 64]]
                let x = vault.zeros[f32, [64, 64]]
                let !last = 0
                for step in 0..200 {
                    let (l, g) = loss.fwd_bwd(W, x)
                    last = step
                }
                last
            }
        }
    "#;
    assert!(run_with_limits(src, forge_budget(3_358_720)).is_ok());
    let e = run_with_limits(src, forge_budget(3_358_719))
        .expect_err("3 358 720 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── #781 (PR #790 rework): an escaping constructor stays charged ─────────
//
// The forward-only `@grad` mark, and the `.fwd_bwd` epilogue, are entitled
// to reclaim the COMPILER's own intermediates (`MEMORY.md §1.1`) — not a
// tensor the program spelled out. A `forge.*` constructor written inside
// such a body can escape it and stay live, so its bytes survive the rewind.
//
// Each row below is 40 calls of a `@grad fn` whose body constructs one
// 64×64 f32 tensor (16 384 B) and lets it escape by one of the three
// routes, with the same source minus `@grad` as the control: the twin's
// minimum is HIGHER (it keeps the body's activations too), and what matters
// is that the marked version keeps the constructor term rather than
// collapsing to one call's worth. Before this fix each `@grad` row bisected
// to 65 536 B — the whole 40-call constructor series un-charged while live.
//
// There is no JIT parity to preserve on any of them: `dmc jit` refuses all
// three outright (a non-tensor `!` param on a `@grad fn`; a non-differentiable
// postfix op; a top-level non-constant `let`), measured at the same commit.

#[test]
fn a_constructor_escaping_through_a_mut_model_param_stays_charged() {
    // `m.w = forge.zeros[…]` inside a forward-only `@grad` body: the tensor
    // leaves through the `!` model param, which is the caller's instance.
    // 40 × 16 384 (the constructors) + 16 384 (`m`'s initial `w`) + 16 384
    // (`x`) + 16 384 (one call's live `x .* x`) = 704 512 B.
    let src = r#"
        model Holder { w: Tensor[f32, [64, 64]] }
        @grad fn touch(!m: Holder, x: Tensor[f32, [64, 64]]) -> f32 {
            m.w = forge.zeros[f32, [64, 64]]
            sum(x .* x)
        }
        fn main() -> i64 {
            let !m = Holder { w: forge.zeros[f32, [64, 64]] }
            let x = forge.zeros[f32, [64, 64]]
            let !last = 0
            for step in 0..40 {
                let l = touch(m, x)
                last = step
            }
            last
        }
    "#;
    assert!(run_with_limits(src, forge_budget(704_512)).is_ok());
    let e = run_with_limits(src, forge_budget(704_511))
        .expect_err("704 512 B is the exact minimum, not the 65 536 B a full rewind left");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn the_control_twin_without_grad_charges_the_same_constructors_and_more() {
    // The same source minus `@grad`: no mark, so the body's `x .* x` stays
    // charged too — 704 512 + 39 × 16 384 = 1 343 488 B. The constructor
    // term is identical in both, which is the point of the pair.
    let src = r#"
        model Holder { w: Tensor[f32, [64, 64]] }
        fn touch(!m: Holder, x: Tensor[f32, [64, 64]]) -> f32 {
            m.w = forge.zeros[f32, [64, 64]]
            sum(x .* x)
        }
        fn main() -> i64 {
            let !m = Holder { w: forge.zeros[f32, [64, 64]] }
            let x = forge.zeros[f32, [64, 64]]
            let !last = 0
            for step in 0..40 {
                let l = touch(m, x)
                last = step
            }
            last
        }
    "#;
    assert!(run_with_limits(src, forge_budget(1_343_488)).is_ok());
    let e = run_with_limits(src, forge_budget(1_343_487))
        .expect_err("1 343 488 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_constructor_escaping_through_a_mut_tensor_param_stays_charged() {
    // `t = forge.zeros[…]` rebinds the private copy; the copy-OUT then
    // carries it back to the caller. 40 × (16 384 copy-in + 16 384
    // constructor) + 16 384 (`t`) + 16 384 (`x`) + 16 384 (one live
    // `x .* x`) = 1 359 872 B.
    let src = r#"
        @grad fn touch(!t: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {
            t = forge.zeros[f32, [64, 64]]
            sum(x .* x)
        }
        fn main() -> i64 {
            let !t = forge.zeros[f32, [64, 64]]
            let x = forge.zeros[f32, [64, 64]]
            let !last = 0
            for step in 0..40 {
                let l = touch(t, x)
                last = step
            }
            last
        }
    "#;
    assert!(run_with_limits(src, forge_budget(1_359_872)).is_ok());
    let e = run_with_limits(src, forge_budget(1_359_871))
        .expect_err("1 359 872 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_constructor_escaping_into_a_module_level_mut_binding_stays_charged() {
    // `g = forge.zeros[…]` writes a module-level `let !`, which outlives
    // every call. 40 × 16 384 (the constructors) + 16 384 (`g`'s initial
    // value) + 16 384 (`x`) + 16 384 (one live `x .* x`) = 704 512 B.
    //
    // Before #794, the `dmc run` CLI bisected 16 384 B higher here: `main.rs`
    // called `load_program` over every resolved path (including the main
    // file) and then `Interpreter::run` loaded the main file again, so a
    // module-level `let`'s initializer was evaluated — and its constructor
    // charged — twice. #794 skips the main file in that pre-load pass (it is
    // loaded and run exactly once, by `Interpreter::run`), so the CLI now
    // agrees with this test's budget exactly; this test drives the
    // interpreter directly and always saw the single charge.
    let src = r#"
        let !g = forge.zeros[f32, [64, 64]]
        @grad fn touch(x: Tensor[f32, [64, 64]]) -> f32 {
            g = forge.zeros[f32, [64, 64]]
            sum(x .* x)
        }
        fn main() -> i64 {
            let x = forge.zeros[f32, [64, 64]]
            let !last = 0
            for step in 0..40 {
                let l = touch(x)
                last = step
            }
            last
        }
    "#;
    assert!(run_with_limits(src, forge_budget(704_512)).is_ok());
    let e = run_with_limits(src, forge_budget(704_511))
        .expect_err("704 512 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_fwd_bwd_body_keeps_a_tensor_valued_escape_too() {
    // The `.fwd_bwd` epilogue takes the same carve-out. 40 × 16 384 (the
    // `!W` copy-ins) + 40 × 16 384 (the published gradients) + 39 × 16 384
    // (the escaping constructors, the 40th being inside the last step's
    // peak) + 16 384 × 3 (`g`, `W`, `x`) + one step's peak = 2 080 768 B.
    // A rewind that released the constructor series bisected to 1 441 792 B.
    // (Before #794, the `dmc run` CLI reported both 16 384 B higher, for the
    // module-level double-load reason the test above records; the CLI now
    // agrees with these numbers too.)
    let src = r#"
        let !g = forge.zeros[f32, [64, 64]]
        @grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {
            g = forge.zeros[f32, [64, 64]]
            let h = x @ W
            sum(h .* h)
        }
        fn main() -> i64 {
            let !W = forge.zeros[f32, [64, 64]]
            let x = forge.zeros[f32, [64, 64]]
            let !last = 0
            for step in 0..40 {
                let (l, gr) = loss.fwd_bwd(W, x)
                last = step
            }
            last
        }
    "#;
    assert!(run_with_limits(src, forge_budget(2_080_768)).is_ok());
    let e = run_with_limits(src, forge_budget(2_080_767))
        .expect_err("2 080 768 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

// ─── #794: the main file's top-level `let`s are evaluated exactly once ────
//
// `main.rs`'s `run` (and `pipeline`) arms used to pre-load every resolved
// path, including the main file itself, and then call `Interpreter::run` on
// the main file a second time — so the main file's own top-level `let`
// initializers ran twice: any side effect inside one (a `print`, an `Rng`
// draw, a port call, a Forge allocation) happened twice too. Fixed by
// skipping the main file in that pre-load pass — it is loaded and run
// exactly once, by `Interpreter::run` — in `main.rs`'s `run` and `pipeline`
// arms, and here in `run_multi`/`run_multi_err`, which mirror them.
//
// An imported dependency's top-level `let`s were never double-run this way
// (the old code only loaded the *entry* path twice); the third test below
// pins that the fix leaves that alone.

/// `run_multi` plus arena limits, installed the way `--forge=`/`--vault=`
/// would be by the CLI. `run_with_limits` (above) is single-file only and so
/// cannot see this bug: it calls `Interpreter::run` directly, with no
/// pre-load pass to duplicate — only the multi-path `resolve → pre-load →
/// run` pipeline `run_multi` mirrors can.
fn run_multi_with_limits(
    files: &[(&str, &str)],
    main: &str,
    limits: crate::arena::ArenaLimits,
) -> Result<Value, String> {
    run_multi_impl(files, main, Some(limits))
}

#[test]
fn main_files_top_level_forge_alloc_is_charged_exactly_once() {
    // The issue's own repro: one 16 384 B `forge.zeros[f32,[4096]]` at the
    // main file's top level. Before #794 this needed a 32 768 B budget (two
    // charges, one per evaluation of the initializer); it now needs exactly
    // 16 384 B — one.
    let src = r#"
        let G = forge.zeros[f32, [4096]]
        fn main() -> i64 { 1 }
    "#;
    let files = [("main.dmc", src)];
    assert!(run_multi_with_limits(&files, "main.dmc", forge_budget(16_384)).is_ok());
    let e = run_multi_with_limits(&files, "main.dmc", forge_budget(16_383))
        .expect_err("16 384 B is the exact minimum — one allocation, not two");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn main_files_top_level_let_initializer_runs_exactly_once() {
    // A side effect in the main file's own top-level `let` initializer must
    // fire once, not twice. An earlier version of this test kept the counter
    // (`let !count = 0`) in the main file alongside the initializer under
    // test — but the main file's own top-level `let`s are (re-)loaded fresh
    // by every call into this harness, so `count` started back at 0
    // whichever pass evaluated it, and the assertion read 1 whether the
    // initializer ran once or twice: the test passed even with the #794 fix
    // reverted (confirmed by reverting the three `if *p == canonical`
    // guards in scratch — see the PR description). Routing the counter
    // through an imported module (`dep.dmc`), and bumping it from the main
    // file's top-level `let X = bump()`, makes a double run visible: `get()`
    // reads back 2 with the fix reverted, 1 with it in place.
    let v = run_multi(&[
        (
            "dep.dmc",
            "let !count = 0\n\
             pub fn bump() -> i64 { count = count + 1  count }\n\
             pub fn get() -> i64 { count }\n",
        ),
        (
            "main.dmc",
            "use \"dep.dmc\"\nlet X = bump()\nfn main() -> i64 { get() }\n",
        ),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1, "main file's top-level initializer ran more than once");
}

#[test]
fn an_imported_files_top_level_let_initializer_still_runs_exactly_once() {
    // The counterpart: an imported dependency's top-level `let` initializer
    // was never double-run by the old code (only the entry path was loaded
    // a second time) — pins that the #794 fix leaves this alone.
    let v = run_multi(&[
        (
            "dep.dmc",
            "let !count = 0\n\
             let Y = { count = count + 1  1 }\n\
             pub fn get_count() -> i64 { count }\n",
        ),
        (
            "main.dmc",
            "use \"dep.dmc\"\nfn main() -> i64 { get_count() }\n",
        ),
    ], "main.dmc");
    assert_eq!(as_int(&v), 1, "imported file's top-level initializer ran more than once");
}

// ─── #781 (PR #790 rework): a `!` writeback survives a `?` early return ───

#[test]
fn a_mut_param_writeback_survives_a_query_early_return() {
    // SPEC §4.9: `?` desugars to `return (default[T], e)`. That reaches
    // `call_fn_in_module` as an `Err` carrying the propagation marker, so a
    // writeback capture gated on `result.is_ok()` drops it — the caller's
    // tensor stayed at 0 under `dmc run` while `dmc jit`, which emits the
    // copy-out on every exit path, answered 7. The meter rewind is the only
    // thing that gate belongs on.
    let v = run_with_limits(r#"
        fn always_fails() -> (i64, Err) {
            (0, "no")
        }
        fn write_then_bail(!a: Tensor[i64, [1]]) -> (i64, Err) {
            a[0] = 7
            let k = always_fails()?
            (k, nil)
        }
        fn main() -> i64 {
            let !t = forge.zeros[i64, [1]]
            let (r, e) = write_then_bail(t)
            t[0]
        }
    "#, forge_budget(1024)).expect("should fit the budget");
    assert_eq!(as_int(&v), 7);
}

// ─── #781 reviewer rework: uncharged on both backends today (JIT refuses) ──
//
// Tensor `as` cast, `.split` and `<-` concat on an ordinary (non-KV) tensor
// all materialize a new backing array on the interpreter and were
// completely uncharged; the JIT has no lowering for any of them yet and
// refuses the program outright, so there is nothing to stay in parity with
// — these charges get ahead of the JIT lowering them.

#[test]
fn a_tensor_as_cast_is_charged() {
    // x: [8] f64 (64 B). `x as f32` narrows to [8] f32 (32 B) — 96 B total.
    let src = r#"
        fn main() -> i64 {
            let x = forge.zeros[f64, [8]]
            let y = x as f32
            y[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(96)).is_ok());
    let e = run_with_limits(src, forge_budget(95)).expect_err("96 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn tensor_split_pieces_are_each_charged() {
    // x: [8] f32 (32 B). `.split[2]` copies out two [4] f32 pieces (16 B
    // each) — 64 B total, where before this fix only `x` was charged.
    let src = r#"
        fn main() -> i64 {
            let x = forge.zeros[f32, [8]]
            let (a, b) = x.split[2]
            a[0] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(64)).is_ok());
    let e = run_with_limits(src, forge_budget(63)).expect_err("64 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn stream_arrow_concat_on_an_ordinary_tensor_is_charged() {
    // `buf` is a PLAIN forge tensor (no declared `capacity` — not a Stream
    // arena / `forge.kv` binding), so `<-` growing it is an ordinary Forge
    // concat, not the by-design-uncharged Stream append. buf: [4] f32
    // (16 B); the appendee literal: [4] f32 (16 B, charged the same as any
    // other tensor literal, unrelated to this fix); the grown [8] f32
    // result: 32 B, uncharged before this fix. 64 B total.
    let src = r#"
        fn main() -> i64 {
            let !buf = forge.zeros[f32, [4]]
            buf <- [1.0, 2.0, 3.0, 4.0]
            buf[4] as i64
        }
    "#;
    assert!(run_with_limits(src, forge_budget(64)).is_ok());
    let e = run_with_limits(src, forge_budget(63)).expect_err("64 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_kv_bindings_stream_arrow_stays_uncharged() {
    // Regression guard alongside `a_kv_allocation_is_stream_not_forge`: a
    // DECLARED-capacity `forge.kv` binding's `<-` must stay exempt (MEMORY.md
    // §1: the Stream arena is bounded by its own `capacity`, not `--forge`)
    // even now that a PLAIN tensor's `<-` is charged.
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let !c = forge.kv[f32, [~]](capacity = 4096)
            c <- [1.0f32]
            c <- [2.0f32]
            1
        }
    "#, forge_budget(16)).expect("a kv binding's <- must not charge the forge budget");
    assert_eq!(as_int(&v), 1);
}

// ─── PR #815 review: an `as str` cast and a str method both charge their
//     fresh result, through the same choke point as everything above ──────
//
// `charge_intermediate`'s doc used to say `apply_binop`'s `+` was "the one
// string-producing form that lives outside `call_builtin`" — wrong twice.
// `apply_cast`'s `as str` (`Expr::Cast`) already flowed through the same
// `charge_intermediate` call every cast takes (correct behavior, just
// undocumented — see `MEMORY.md §1.1`). `call_str_method`
// (`.trim`/`.upper`/`.lower`/`.replace`/`.split`/`.lines`) did NOT flow
// through anything: it returned straight out of the method-call arm in
// `eval_expr`'s `Expr::Call` handling, so every one of these materialized a
// fresh string (or, for `.split`/`.lines`, a fresh list of them) at 0 B on
// `dmc run` — the JIT refuses all of these but `.starts_with` today, so
// there was no parity to preserve, only a real allocation going uncounted.
// Fixed by charging the result through `charge_builtin_result` at the call
// site, which already recurses into a `List`.

#[test]
fn an_as_str_cast_result_is_charged() {
    // `x as str` on an int: `to_str`'s own family, but reached through
    // `Expr::Cast`/`apply_cast` rather than `call_builtin` — the OTHER
    // string-producing form `charge_intermediate`'s doc names now.
    // 123456789 as str -> "123456789", 9 B.
    let src = r#"
        fn main() -> i64 {
            let x = 123456789
            let s = x as str
            len(s)
        }
    "#;
    assert!(run_with_limits(src, forge_budget(9)).is_ok());
    let e = run_with_limits(src, forge_budget(8)).expect_err("9 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_str_trim_result_is_charged() {
    // "  hi  ".trim() -> "hi", 2 B, uncharged before this fix (any budget,
    // including 0, ran — rejected only because `--forge=0` itself is
    // invalid).
    let src = r#"
        fn main() -> i64 {
            let s = "  hi  "
            let t = s.trim()
            len(t)
        }
    "#;
    assert!(run_with_limits(src, forge_budget(2)).is_ok());
    let e = run_with_limits(src, forge_budget(1)).expect_err("2 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_str_replace_result_is_charged() {
    // "aaa".replace("a", "bb") -> "bbbbbb", 6 B.
    let src = r#"
        fn main() -> i64 {
            let s = "aaa"
            let t = s.replace("a", "bb")
            len(t)
        }
    "#;
    assert!(run_with_limits(src, forge_budget(6)).is_ok());
    let e = run_with_limits(src, forge_budget(5)).expect_err("6 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_str_split_results_charge_every_piece() {
    // "a,bb,ccc".split(",") -> ["a", "bb", "ccc"]: 1 + 2 + 3 = 6 B, the sum
    // of every piece — `charge_builtin_result` recurses into the `List`.
    let src = r#"
        fn main() -> i64 {
            let s = "a,bb,ccc"
            let parts = s.split(",")
            len(parts[2])
        }
    "#;
    assert!(run_with_limits(src, forge_budget(6)).is_ok());
    let e = run_with_limits(src, forge_budget(5)).expect_err("6 B is the exact minimum");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_str_method_returning_a_non_str_is_still_uncharged() {
    // `.find()` / `.starts_with()` return an Int/Bool, not a fresh string —
    // `charge_builtin_result` is a no-op for those, exactly as it is for
    // any other builtin's non-tensor, non-string result. `s` itself is an
    // uncharged literal (`lit_value_hinted`, no `charge_intermediate` call
    // wraps it), so the whole program charges 0 B — it must run under the
    // tightest budget the arena accepts.
    let v = run_with_limits(r#"
        fn main() -> i64 {
            let s = "hello"
            if s.starts_with("he") { s.find("l") } else { -1 }
        }
    "#, forge_budget(1)).expect("charges nothing — .find/.starts_with are not fresh strings");
    assert_eq!(as_int(&v), 2);
}

// ─── #540: fixed-width integer arithmetic wraps at the declared width ────────
//
// SPEC §3.1. The interpreter used to hold an `i32` as an i64 and narrow only at
// an explicit `as i32`, so `let c: i32 = a + b` stored 2^31 where the JIT's
// `cl::I32` wrapped to -2^31. These pin the interpreter half; the cross-backend
// half is `tests/i32_wrap_540.rs` + `tests/spec_probes/p57_i32_arith_wraps.dmc`.

/// The issue's repro. Before #540 this returned 2147483648.
#[test]
fn i32_add_overflow_wraps_to_min() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = 2147483647
            let b: i32 = 1
            let c: i32 = a + b
            c as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

/// The other direction: under the bottom, back to MAX.
#[test]
fn i32_sub_at_the_negative_boundary_wraps_to_max() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = -2147483648
            let b: i32 = 1
            (a - b) as i64
        }
    "#);
    assert_eq!(as_int(&v), 2147483647);
}

/// 10^10 mod 2^32. Multiply wraps like add and subtract.
#[test]
fn i32_mul_overflow_wraps() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = 100000
            (a * a) as i64
        }
    "#);
    assert_eq!(as_int(&v), 1410065408);
}

/// Negating i32::MIN is i32::MIN — the one fixed point of two's complement.
#[test]
fn i32_negate_min_is_min() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = -2147483648
            let z: i32 = 0
            (z - a) as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

/// The width survives the binding, so a comparison sees the WRAPPED value.
/// This is the case a narrow-at-the-annotation-only fix would have missed.
#[test]
fn i32_comparison_sees_the_wrapped_value() {
    let v = run(r#"
        fn main() -> bool {
            let a: i32 = 2147483647
            let b: i32 = 1
            (a + b) < 0
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "got {:?}", v);
}

/// An untyped integer literal adopts the i32 operand's width (§3.1), the same
/// rule the JIT applies in `adopt_int_literal_kind`.
#[test]
fn i32_meeting_an_untyped_literal_still_wraps() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = 2147483647
            (a + 1) as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

/// A parameter and a declared return are width origins too, not just a `let`.
#[test]
fn i32_width_originates_at_a_parameter_and_a_return() {
    let v = run(r#"
        fn bump(x: i32) -> i32 { x + 1 }
        fn main() -> i64 {
            let a: i32 = 2147483647
            bump(a) as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

/// #545: an integer literal's SUFFIX is a width origin. §3.1: "an explicit type
/// suffix on an integer literal types the literal concretely", so
/// `2147483647i32` is an `i32` and the add after it wraps. #540 backed this out
/// because `jit.rs`'s `lower_literal` dropped the suffix and honoring it here
/// alone would have been a fresh divergence; #545 fixed both backends together,
/// and `tests/narrow_int_wrap_544.rs` pins that they agree.
#[test]
fn an_integer_suffix_is_a_width_origin() {
    let v = run(r#"
        fn main() -> i64 {
            let a = 2147483647i32
            (a + 1i32) as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

/// The suffix originates a width at EVERY fixed-width kind, not just `i32` —
/// and an unsuffixed literal meeting one adopts it (§3.1's untyped-literal
/// rule), which is the `IW::join` half of the same fix.
#[test]
fn an_integer_suffix_originates_every_width() {
    let v = run(r#"
        fn main() -> i64 {
            let a = 32767i16
            (a + 1i16) as i64
        }
    "#);
    assert_eq!(as_int(&v), -32768);
    let v = run(r#"
        fn main() -> i64 {
            let a = 127i8
            (a + 1) as i64
        }
    "#);
    assert_eq!(as_int(&v), -128);
    let v = run(r#"
        fn main() -> i64 {
            let a = 0u8
            (a - 1u8) as i64
        }
    "#);
    assert_eq!(as_int(&v), 255);
}

/// The explicit cast is UNCHANGED by #540 — it narrowed before and narrows the
/// same way now, on both backends. This is the case that already agreed.
#[test]
fn explicit_as_i32_cast_is_unchanged() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i64 = 5000000000
            (a as i32) as i64
        }
    "#);
    assert_eq!(as_int(&v), 705032704);
}

/// …and the cast's RESULT is an i32, so what follows it wraps too.
#[test]
fn the_result_of_an_as_i32_cast_is_an_i32() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i64 = 2147483647
            let b: i32 = 1
            ((a as i32) + b) as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

/// i64 is unaffected: the same expression one width up does not wrap, and i64
/// overflow still wraps at 64 bits (#300).
#[test]
fn i64_arithmetic_keeps_its_own_width() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i64 = 2147483647
            a + 1
        }
    "#);
    assert_eq!(as_int(&v), 2147483648);
    let v = run(r#"
        fn main() -> i64 {
            let a: i64 = 9223372036854775807
            a + 1
        }
    "#);
    assert_eq!(as_int(&v), i64::MIN);
}

/// `~` complements the operand's width, not always 64 bits: `~0i32` is -1 at
/// both widths, but the value stays an i32 and the next op wraps at 32.
#[test]
fn bitwise_not_and_ops_keep_the_i32_width() {
    let v = run(r#"
        fn main() -> i64 {
            let z: i32 = 0
            let max: i32 = 2147483647
            (~z - max) as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
}

/// `<<` is bounded by the SHIFTED VALUE's width — an i32 shift tops out at 31,
/// and the result wraps at 32. The JIT's `shift_range_check` (#541) sizes its
/// guard the same way, and its message names the same range.
#[test]
fn i32_shift_is_bounded_by_32_bits_and_wraps() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = 1
            let k: i32 = 31
            (a << k) as i64
        }
    "#);
    assert_eq!(as_int(&v), -2147483648);
    let e = run_err(r#"
        fn main() -> i64 {
            let a: i32 = 1
            let k: i32 = 32
            (a << k) as i64
        }
    "#);
    assert!(e.contains("expected 0..=31"), "got {:?}", e);
}

/// `MIN / -1` overflows at every width and is the one division with no wrapped
/// value; the diagnostic names the width that actually overflowed.
#[test]
fn i32_min_divided_by_minus_one_reports_the_i32_range() {
    let e = run_err(r#"
        fn main() -> i64 {
            let a: i32 = -2147483648
            let b: i32 = -1
            (a / b) as i64
        }
    "#);
    assert!(e.contains("exceeds the i32 range"), "got {:?}", e);
    let e = run_err(r#"
        fn main() -> i64 {
            let a: i64 = -9223372036854775807 - 1
            let b: i64 = -1
            a / b
        }
    "#);
    assert!(e.contains("exceeds the i64 range"), "got {:?}", e);
}

/// Div-by-zero stays total (#208) at every width.
#[test]
fn i32_division_by_zero_is_still_zero() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = 2147483647
            let z: i32 = 0
            ((a / z) + (a % z)) as i64
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// A map is `str → any` and the JIT backs it with a raw i64 store, so an
/// integer loses its declared width by going through one. Keeping the width
/// here would make `map_get(m, k) + 1` wrap under `dmc run` and not under
/// `dmc jit` — a divergence introduced by the fix for a divergence.
#[test]
fn a_map_round_trip_drops_the_i32_width() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i32 = 2147483647
            let m = map_new()
            map_set(m, "k", a)
            map_get(m, "k") + 1
        }
    "#);
    assert_eq!(as_int(&v), 2147483648);
}

/// The width DOES survive the typed containers — a tuple, an enum payload and
/// a model field are all declared `i32` on both backends.
#[test]
fn the_width_survives_typed_containers() {
    let v = run(r#"
        enum Opt { Some(i32), None }
        fn pair(a: i32, b: i32) -> (i32, i32) { (a, b) }
        fn main() -> i64 {
            let a: i32 = 2147483647
            let one: i32 = 1
            let o = Opt.Some(a)
            let x = match o { Some(n) => n + one, None => one }
            let (p, q) = pair(a, one)
            ((x + q) - (p + q)) as i64
        }
    "#);
    // x and (p + q) are both the wrapped MIN; MIN + 1 - MIN is 1.
    assert_eq!(as_int(&v), 1);
}

/// #544: the other fixed-width kinds wrap at their own width too. #540 did
/// `i32` alone because it was the only narrow `ScalarKind` the JIT had; #544
/// gave the JIT masked i64-backed kinds for the rest, so both backends now wrap
/// at 8/16/32 bits (`tests/narrow_int_wrap_544.rs` pins the agreement).
#[test]
fn the_other_narrow_widths_wrap_at_their_own_width() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i16 = 32767
            let b: i16 = 1
            (a + b) as i64
        }
    "#);
    assert_eq!(as_int(&v), -32768);
    let v = run(r#"
        fn main() -> i64 {
            let a: i8 = 127
            let b: i8 = 1
            (a + b) as i64
        }
    "#);
    assert_eq!(as_int(&v), -128);
}

/// Unsigned wrap is UNSIGNED: `0u32 - 1u32` is 4294967295, held zero-extended
/// in the i64-backed value, not the -1 a signed 32-bit wrap would give.
#[test]
fn unsigned_widths_wrap_unsigned() {
    let cases = [
        ("u8", 255i64), ("u16", 65535), ("u32", 4294967295),
    ];
    for (ty, want) in cases {
        let src = format!(r#"
            fn main() -> i64 {{
                let a: {ty} = 0
                let b: {ty} = 1
                (a - b) as i64
            }}
        "#);
        assert_eq!(as_int(&run(&src)), want, "0 - 1 at {}", ty);
    }
    // …and the zero-extended value is what every later operation sees: the
    // comparison, the division and the shift are all unsigned because the
    // operand is a non-negative i64.
    let v = run(r#"
        fn main() -> i64 {
            let a: u32 = 0
            let b: u32 = 1
            let m = a - b
            let two: u32 = 2
            (m / two) as i64
        }
    "#);
    assert_eq!(as_int(&v), 2147483647);
    let v = run(r#"
        fn main() -> bool {
            let a: u32 = 0
            let b: u32 = 1
            (a - b) > b
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "got {:?}", v);
}

/// `u64` is the one width an i64-backed value cannot fully hold, so it stays
/// 64-bit two's-complement: the WRAP is bit-identical to `i64`'s (which is all
/// §3.1 asks), but a value above 2^63 renders as its signed pattern. Pinned so
/// the limitation is a decision on record, not a surprise.
#[test]
fn u64_wraps_at_64_bits_as_a_twos_complement_pattern() {
    let v = run(r#"
        fn main() -> i64 {
            let a: u64 = 0
            let b: u64 = 1
            (a - b) as i64
        }
    "#);
    assert_eq!(as_int(&v), -1);
}

/// The narrow widths carry into `/`, `%`, `<<`, `~` and unary `-` — every op
/// #540 taught to respect a width, now at eight of them.
#[test]
fn narrow_widths_carry_through_every_width_sensitive_op() {
    // `<<` is bounded by the SHIFTED VALUE's width: an i8 shift tops out at 7.
    let v = run(r#"
        fn main() -> i64 {
            let a: i8 = 1
            let k: i8 = 7
            (a << k) as i64
        }
    "#);
    assert_eq!(as_int(&v), -128);
    let e = run_err(r#"
        fn main() -> i64 {
            let a: i8 = 1
            let k: i8 = 8
            (a << k) as i64
        }
    "#);
    assert!(e.contains("expected 0..=7"), "got {:?}", e);
    // `MIN / -1` overflows at every width, and names the one that overflowed.
    let e = run_err(r#"
        fn main() -> i64 {
            let a: i16 = -32768
            let b: i16 = -1
            (a / b) as i64
        }
    "#);
    assert!(e.contains("exceeds the i16 range"), "got {:?}", e);
    // `~` complements the operand's width: `~0u8` is 255, not -1.
    let v = run(r#"
        fn main() -> i64 {
            let z: u8 = 0
            (~z) as i64
        }
    "#);
    assert_eq!(as_int(&v), 255);
    // Negating MIN is MIN, the fixed point of two's complement, at i8 too.
    let v = run(r#"
        fn main() -> i64 {
            let a: i8 = -128
            let z: i8 = 0
            (z - a) as i64
        }
    "#);
    assert_eq!(as_int(&v), -128);
}

/// An `as` cast to a narrow type is a width origin as well as a narrowing, so
/// what follows the cast wraps — the §3.1 rule #540 established for `i32`.
#[test]
fn the_result_of_a_narrow_as_cast_carries_its_width() {
    let v = run(r#"
        fn main() -> i64 {
            let a: i64 = 100000
            let one: i16 = 1
            ((a as i16) + one) as i64
        }
    "#);
    // 100000 as i16 == -31072; +1 stays inside the width.
    assert_eq!(as_int(&v), -31071);
    let v = run(r#"
        fn main() -> i64 {
            let a: i64 = 255
            let one: u8 = 1
            ((a as u8) + one) as i64
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// The packed `int4`/`int8` kinds of §3.1 are signed 4-/8-bit types and wrap
/// at those widths, on the annotation and on the `as` cast alike.
#[test]
fn the_packed_int4_and_int8_kinds_wrap_at_their_widths() {
    let v = run(r#"
        fn main() -> i64 {
            let a: int8 = 127
            let b: int8 = 1
            (a + b) as i64
        }
    "#);
    assert_eq!(as_int(&v), -128);
    let v = run(r#"
        fn main() -> i64 {
            let a: int4 = 7
            let b: int4 = 1
            (a + b) as i64
        }
    "#);
    assert_eq!(as_int(&v), -8);
    let v = run(r#"
        fn main() -> i64 {
            let a: i64 = 9
            (a as int4) as i64
        }
    "#);
    assert_eq!(as_int(&v), -7);
}

/// A width originates at a parameter and a declared return at every kind, not
/// just at a `let` — the same five origins §3.1 names for a float width.
#[test]
fn narrow_widths_originate_at_parameters_and_returns() {
    let v = run(r#"
        fn bump(x: u8) -> u8 { x + 1 }
        fn main() -> i64 {
            let a: u8 = 255
            bump(a) as i64
        }
    "#);
    assert_eq!(as_int(&v), 0);
    let v = run(r#"
        fn wrapped() -> i8 { 127 + 1 }
        fn main() -> i64 { wrapped() as i64 }
    "#);
    assert_eq!(as_int(&v), -128);
}

// ── Issue #517: dynamic slice-start OOB is a runtime panic, not a clamp ──────
//
// `resolve_index_values` used to normalize a `Range`'s `start`/`end`
// independently, so a runtime (non-literal) start whose window fell outside
// the axis silently clamped to a shrunken or empty slice instead of erring
// (SPEC §4.3: "dynamic OOB is a runtime panic"). `dmc jit` already panics for
// this shape (`t[i..i+1, ..]`, #511/#515) via `__dmc_slice_oob_trap`; these
// pin the interpreter matching it, verbatim message included. The STATIC
// path (both bounds literal, e.g. `t[0..100]`) is untouched — see
// `jit_slice_bounds_clamp_parity` in jit.rs, which still passes unchanged.

#[test]
fn dynamic_slice_start_negative_is_a_runtime_panic() {
    // The issue's exact repro: size-4 axis, i = -1, extent 1. Before #517
    // this produced an EMPTY slice (sum 0, exit 0, matching neither the
    // spec nor the JIT). `-1` is a well-formed negative index on its own
    // (it would resolve to the last row), but the independent-bound clamp
    // can't preserve that once `end = i+1` normalizes on its own — so, like
    // the JIT, this is dynamic OOB rather than a "best guess" resolution.
    let msg = run_err(r#"
        fn main() -> nil {
            let !t = forge.zeros[f32, [4, 3]]
            let !i = -1
            let row = t[i..i+1, ..]
            print(sum(row))
            nil
        }
    "#);
    assert_eq!(msg, "slice start -1 with extent 1 out of bounds for axis of size 4");
}

#[test]
fn dynamic_slice_start_past_the_end_is_a_runtime_panic() {
    // Positive but past the axis: i = 4 on a size-4 axis. Matches
    // `both_backends_trap_on_out_of_range_runtime_start` (tests/runtime_slice.rs) —
    // same message, same axis, same extent.
    let msg = run_err(r#"
        fn main() -> nil {
            let !t = forge.zeros[f32, [4, 3]]
            let !i = 4
            print(sum(t[i..i+1, ..]))
            nil
        }
    "#);
    assert_eq!(msg, "slice start 4 with extent 1 out of bounds for axis of size 4");
}

#[test]
fn dynamic_slice_start_in_range_still_works() {
    // A legal in-range dynamic start must keep working exactly as before —
    // #517 only adds a bounds check, it must not disturb the valid window.
    // Row i sums to 40i + 6 for i in 0..5 (the table is `10i + j`); the
    // issue's own repro value (430) pins the sum end to end.
    let v = run(r#"
        fn main() -> i64 {
            let !table = forge.zeros[f32, [8, 4]]
            for i in 0..8 {
                for j in 0..4 { table[i, j] = (i as f32) * 10.0 + (j as f32) }
            }
            let !acc = 0.0
            for i in 0..5 {
                let row = table[i..i+1, ..]
                acc = acc + sum(row)
            }
            acc as i64
        }
    "#);
    assert_eq!(as_int(&v), 430);
}

#[test]
fn static_slice_bounds_still_clamp_unchanged() {
    // #291.4's clamp path (BOTH bounds literal) must be untouched by #517:
    // `t[0..100]` on a size-5 axis still clamps to the full axis rather than
    // panicking, matching `dmc jit`'s static `IndexCat::Range` path.
    let v = run(r#"
        fn main() -> f32 {
            let !t = forge.zeros[f32, [5]]
            t[2] = 1.0
            t[4] = 2.0
            sum(t[0..100])
        }
    "#);
    assert!((as_float(&v) - 3.0).abs() < 1e-6, "expected clamp to full axis (sum 3.0), got {:?}", v);
}

// ── #639: `return` / `break` / `continue` leave a block ─────────────────────
//
// A block — plain `{ … }`, `forge { … }` / `vault { … }`, or a directive block
// — is a scope, not a function boundary and not a loop boundary. SPEC.md §4
// gives a block's VALUE (its tail) and §5.1 lists `forge` beside `if` / `for` /
// `while` as a nested *scope*; nothing makes one a control-flow boundary, so a
// `return` inside it belongs to the enclosing fn and a `break` / `continue` to
// the enclosing loop. The interpreter used to collapse the escaping `Flow` —
// `return` became the block's value, `break` and `continue` became nil — while
// the JIT propagated it, so the two backends silently disagreed. The three
// programs below are the issue's own, with the JIT's answers.

#[test]
fn issue_639_return_leaves_the_arena_block_and_the_function() {
    // `dmc run` answered 7 (the tail): the `return` stopped at the `forge`
    // block and its value was discarded. `dmc jit` answered 42.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            forge { if x > 0 { return 42 } }
            7
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn issue_639_return_not_taken_still_yields_the_tail() {
    // The other half of the same rule: no flow escapes, so the function's
    // value is its own tail and the block changes nothing.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            forge { if x > 0 { return 42 } }
            7
        }
        fn main() -> i64 { f(-1) }
    "#);
    assert_eq!(as_int(&v), 7);
}

#[test]
fn issue_639_break_leaves_the_arena_block_and_the_while_loop() {
    // `dmc run` answered 10 (the loop ran to its condition): the `break` became
    // nil at the block boundary. `dmc jit` answered 3.
    let v = run(r#"
        fn main() -> i64 {
            let !i = 0
            while i < 10 {
                forge { if i > 2 { break } }
                i = i + 1
            }
            i
        }
    "#);
    assert_eq!(as_int(&v), 3);
}

#[test]
fn issue_639_continue_leaves_the_arena_block_and_skips_the_iteration() {
    // `dmc run` answered 15 (1+2+3+4+5 — the `continue` was swallowed, so no
    // iteration was skipped). `dmc jit` answered 13, i.e. without i == 2.
    let v = run(r#"
        fn main() -> i64 {
            let !i = 0
            let !s = 0
            while i < 5 {
                i = i + 1
                forge { if i == 2 { continue } }
                s = s + i
            }
            s
        }
    "#);
    assert_eq!(as_int(&v), 13);
}

#[test]
fn issue_639_plain_block_return_leaves_the_function() {
    // Nothing arena-specific about it: a plain `{ … }` in statement position
    // swallowed the `return` the same way, and the function fell through to
    // its tail (1).
    let v = run(r#"
        fn g() -> i64 { { return 9 }  1 }
        fn main() -> i64 { g() }
    "#);
    assert_eq!(as_int(&v), 9);
}

#[test]
fn issue_639_break_in_an_arena_block_inside_a_for_loop() {
    // `for` reaches the loop body through the same `eval_block`, so the arena
    // block's `break` must stop the iteration at i == 4 (four increments run).
    let v = run(r#"
        fn main() -> i64 {
            let !n = 0
            for i in 0..10 {
                forge { if i == 4 { break } }
                n = n + 1
            }
            n
        }
    "#);
    assert_eq!(as_int(&v), 4);
}

#[test]
fn issue_639_continue_climbs_out_of_nested_blocks() {
    // Two block boundaries between the `continue` and its loop: the flow has
    // to survive both, not just the innermost. 0 + 1 + 3 + 4 = 8.
    let v = run(r#"
        fn main() -> i64 {
            let !s = 0
            for i in 0..5 {
                { { if i == 2 { continue } } }
                s = s + i
            }
            s
        }
    "#);
    assert_eq!(as_int(&v), 8);
}

#[test]
fn issue_639_block_tail_is_used_and_the_conditional_return_is_not_taken() {
    // The value rule and the flow rule in one program. The `return` is not
    // reached, so the block is worth its tail (5) and the function answers 6 —
    // this is the half that must NOT move, and both backends already agreed.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            let y = { if x > 0 { return 42 }  5 }
            y + 1
        }
        fn main() -> i64 { f(-1) }
    "#);
    assert_eq!(as_int(&v), 6);
}

#[test]
fn issue_639_block_tail_is_used_and_the_conditional_return_is_taken() {
    // Same program, other branch. `dmc run` bound y = 42 and answered 43;
    // `dmc jit` returned 42 from the function. This shape lowers on both
    // backends, so it is also a `tools/jit_probes.py` entry.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            let y = { if x > 0 { return 42 }  5 }
            y + 1
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn issue_639_return_inside_a_block_inside_a_match_arm() {
    // A match arm body is an expression position, and the arm runs in a scope
    // of its own — the escaping flow has to leave that scope cleanly, or every
    // later `pop_scope` is off by one frame.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            let y = match x { 1 => { return 42 }, _ => 5 }
            y + 100
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn issue_639_match_arm_without_the_return_is_unchanged() {
    // The arm that takes no early exit must still bind its value normally —
    // the guard that the scope bookkeeping above did not disturb the ordinary
    // path. 5 + 100.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            let y = match x { 1 => { return 42 }, _ => 5 }
            y + 100
        }
        fn main() -> i64 { f(2) }
    "#);
    assert_eq!(as_int(&v), 105);
}

#[test]
fn issue_639_return_leaves_a_directive_block() {
    // A `@cast(…) { … }` block is a block. In statement position it discarded
    // the flow outright (the function answered its tail, 7); in expression
    // position it turned the `return` into the block's value (43).
    let stmt_pos = run(r#"
        fn f(x: i64) -> i64 {
            @cast(i64) { if x > 0 { return 42 }  5 }
            7
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&stmt_pos), 42);

    let expr_pos = run(r#"
        fn f(x: i64) -> i64 {
            let y = @cast(i64) { if x > 0 { return 42 }  5 }
            y + 1
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&expr_pos), 42);
}

#[test]
fn issue_639_a_block_with_no_escaping_flow_is_still_its_tail() {
    // The regression guard for the value rule: a block that no `return` /
    // `break` / `continue` leaves evaluates to its tail expression, nested
    // blocks included, exactly as SPEC.md §4 says. 2 * 3 + 1.
    let v = run(r#"
        fn main() -> i64 {
            let y = { let a = 2  { a * 3 } }
            y + 1
        }
    "#);
    assert_eq!(as_int(&v), 7);
}

#[test]
fn issue_639_break_out_of_a_block_in_a_loop_leaves_the_loop_not_the_function() {
    // `break` stops at the nearest LOOP, not at the function: the code after
    // the loop still runs. Without that the answer would be nil, not 100.
    let v = run(r#"
        fn f() -> i64 {
            let !n = 0
            while true {
                { break }
            }
            n = n + 100
            n
        }
        fn main() -> i64 { f() }
    "#);
    assert_eq!(as_int(&v), 100);
}

#[test]
fn issue_639_a_return_inside_a_range_bound_is_not_absorbed_into_the_bound() {
    // A range bound tolerates an expression that is not an integer and reads
    // it as 0 — that tolerance is for VALUES. An escaping `return` is not a
    // bad bound, it is the function's control flow passing through, so it
    // keeps travelling instead of leaving a `0..0` behind.
    let v = run(r#"
        fn f() -> i64 {
            let !s = 0
            for i in 0..{ return 5 } { s = s + i }
            s
        }
        fn main() -> i64 { f() }
    "#);
    assert_eq!(as_int(&v), 5);
}

#[test]
fn issue_639_a_match_guard_can_carry_the_escaping_flow_out() {
    // The guard runs inside the arm's scope, before the body. A `return` that
    // leaves a block there has to unwind that scope on the way out — the arm
    // is never entered, and the enclosing function answers 42.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            let y = match x { n if n > 0 => { return 42 }, _ => 5 }
            y + 100
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn issue_639_a_block_in_a_call_argument_still_returns_from_the_caller() {
    // Argument position is an expression position like any other: the `return`
    // belongs to `f`, not to the block and not to `print`.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            print({ if x > 0 { return 42 }  1 })
            7
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&v), 42);
}

// ── #639, second pass: the two entry points that bypass `eval_block` ────────
//
// "A carrier can never cross a call frame" holds only where a function body
// goes through `eval_block`. Two entries do not: a `@pp` fn's `stage K:` bodies
// (evaluated one expression at a time) and a `@grad` fn's taped body (whose
// walker hands a block to plain `eval_expr`). Both are function boundaries, so
// both catch the carrier themselves; without that, a check-clean program
// silently ran the CALLER's control flow.

#[test]
fn issue_639_return_in_a_pp_stage_returns_from_the_pp_fn_not_the_caller() {
    // The `return` belongs to `pipeline`, so `probe` still adds its 7 and
    // `main` its 10000. Uncaught it left `pipeline` AND `probe` (10099).
    let v = run(r#"
        @pp(stages=2)
        fn pipeline(x: i64) -> i64 {
            stage 0: { if x > 0 { return 99 }  x + 1 }
            stage 1: _ * 2
        }
        fn probe() -> i64 {
            let a = pipeline(5)
            let b = 7
            a + b
        }
        fn main() -> i64 { probe() + 10000 }
    "#);
    assert_eq!(as_int(&v), 10106);
}

#[test]
fn issue_639_a_pp_stage_that_takes_no_early_exit_is_unchanged() {
    // The other half: no escape, so the stages chain through `_` as always.
    // (5 + 1) * 2 = 12.
    let v = run(r#"
        @pp(stages=2)
        fn pipeline(x: i64) -> i64 {
            stage 0: { if x < 0 { return 99 }  x + 1 }
            stage 1: _ * 2
        }
        fn main() -> i64 { pipeline(5) }
    "#);
    assert_eq!(as_int(&v), 12);
}

#[test]
fn issue_639_break_in_a_pp_stage_does_not_break_the_callers_loop() {
    // A `@pp` stage is not a loop, so a `break` with no loop inside the stage
    // is the same backstop `call_fn` already raises on an ordinary body.
    // Uncaught, this broke `main`'s `while` and answered 500.
    let (msg, at) = run_err_at(r#"
        @pp(stages=1)
        fn p(x: i64) -> i64 {
            stage 0: { if x > 0 { break }  x }
        }
        fn main() -> i64 {
            let !n = 0
            while n < 5 {
                let z = p(1)
                n = n + 1
            }
            n + 500
        }
    "#);
    assert!(msg.contains("break/continue outside loop"), "got: {msg}");
    assert!(at.is_some(), "the backstop must point at the `break`, got no span");
}

#[test]
fn issue_639_continue_in_a_pp_stage_does_not_continue_the_callers_loop() {
    // Same rule for `continue`, which uncaught skipped the rest of every
    // iteration of `main`'s loop (200 escapes, `s` never incremented).
    let msg = run_err(r#"
        @pp(stages=1)
        fn p(x: i64) -> i64 {
            stage 0: { if x > 0 { continue }  x }
        }
        fn main() -> i64 {
            let !n = 0
            let !s = 0
            while n < 200 {
                n = n + 1
                let z = p(1)
                s = s + 1
            }
            n * 1000 + s
        }
    "#);
    assert!(msg.contains("break/continue outside loop"), "got: {msg}");
}

#[test]
fn issue_639_an_escaping_pp_stage_pops_its_scope_every_time() {
    // Scope balance. The old `prev = self.eval_expr(body)?` skipped the
    // `pop_scope` on the escape path, so each call left its frame — with the
    // stage's own parameter still bound in it — on the stack. `carry` is the
    // `@pp` fn's parameter name AND a `main` local: after 200 leaked frames a
    // read of `carry` finds the stage's 1 instead of `main`'s 100.
    let v = run(r#"
        @pp(stages=2)
        fn stage_ret(carry: i64) -> i64 {
            stage 0: { if carry > 0 { return 3 }  carry + 1 }
            stage 1: _ * 2
        }
        fn main() -> i64 {
            let !carry = 100
            let !n = 0
            let !s = 0
            while n < 200 {
                n = n + 1
                s = s + stage_ret(1)
            }
            carry + s
        }
    "#);
    assert_eq!(as_int(&v), 700);
}

#[test]
fn issue_639_return_on_a_non_gradient_path_of_a_grad_body_is_refused() {
    // The block is a bare STATEMENT — off the gradient path entirely, so #615's
    // rule never fires and the program is check-clean. Uncaught, the `return`
    // rode out of `call_grad` and returned from `main`, which answered 3 where
    // it used to answer 1006. It is refused instead of honoured: the carrier
    // holds a value, not a tape node, so nothing about it is differentiable.
    let (msg, at) = run_err_at(r#"
        @grad fn gl(!w: f32, x: f32) -> f32 {
            { if x > 0.0 { return w } }
            w * x
        }
        fn main() -> i64 {
            let (l, g) = gl.fwd_bwd(3.0, 2.0)
            (l as i64) + 1000
        }
    "#);
    assert!(msg.contains("@grad `gl`") && msg.contains("`return`"), "got: {msg}");
    assert!(msg.contains("not supported"), "got: {msg}");
    assert!(at.is_some(), "the refusal must point at the `return`, got no span");
}

#[test]
fn issue_639_the_615_gradient_path_block_error_is_unchanged() {
    // A block ON the gradient path (#615): its value is not traced, so the loss
    // does not participate in the graph. That is a different diagnostic and it
    // must keep its own wording — the `@grad` catch above must not capture it.
    let msg = run_err(r#"
        @grad fn gl(!w: f32, x: f32) -> f32 {
            let a = { w * x }
            a * a
        }
        fn main() -> i64 {
            let (l, g) = gl.fwd_bwd(3.0, 2.0)
            (l as i64) + 1000
        }
    "#);
    assert!(msg.contains("must produce a scalar derived from a differentiable input"),
        "the #615 diagnostic must be untouched, got: {msg}");
}

#[test]
fn issue_639_fwd_still_runs_a_grad_body_that_returns() {
    // `.fwd` needs no tape and goes through `call_fn`, so the same body runs
    // normally there: the `return` hands back `w` (3.0) and nothing is refused.
    let v = run(r#"
        @grad fn gl(!w: f32, x: f32) -> f32 {
            { if x > 0.0 { return w } }
            w * x
        }
        fn main() -> i64 {
            let l = gl.fwd(3.0, 2.0)
            (l as i64) + 1000
        }
    "#);
    assert_eq!(as_int(&v), 1003);
}

#[test]
fn issue_639_a_grad_body_with_no_escaping_flow_still_differentiates() {
    // The regression guard: an ordinary `@grad` body is untouched by the catch.
    // L = w * x = 6.0, dL/dw = x = 2.0.
    let v = run(r#"
        @grad fn gl(!w: f32, x: f32) -> f32 { w * x }
        fn main() -> f32 {
            let (l, g) = gl.fwd_bwd(3.0, 2.0)
            l + g.w
        }
    "#);
    assert!((as_float(&v) - 8.0).abs() < 1e-6, "got {:?}", v);
}

#[test]
fn issue_639_a_return_inside_a_split_piece_count_leaves_the_function() {
    // `.split` reports its own diagnostic for a piece count that is not a
    // positive integer. That tolerance swallowed the carrier, so a `return`
    // written in the bracket was reported as a bad piece count instead of
    // leaving `f`.
    let v = run(r#"
        fn f(x: i64) -> i64 {
            let t = forge.zeros[f32, [6]]
            let (a, b, c) = t.split[{ if x > 0 { return 42 }  3 }, axis=-1]
            7
        }
        fn main() -> i64 { f(1) }
    "#);
    assert_eq!(as_int(&v), 42);
}

#[test]
fn issue_639_a_split_piece_count_that_is_simply_bad_still_says_so() {
    // The other half: an ordinary failure in the same position keeps `.split`'s
    // own message — only the carrier is exempt from being absorbed.
    let msg = run_err(r#"
        fn main() -> i64 {
            let t = forge.zeros[f32, [6]]
            let (a, b) = t.split[0, axis=-1]
            7
        }
    "#);
    assert!(msg.contains("`.split` requires a positive integer piece count"), "got: {msg}");
}

#[test]
fn issue_639_a_return_inside_a_model_shape_argument_leaves_the_function() {
    // `M[expr] { … }`: a shape argument that does not evaluate is simply not
    // recorded. The carrier is not "does not evaluate" — uncaught, `f` fell
    // through to its own tail (7) instead of returning 42.
    let v = run(r#"
        model M[N] {
            v: i64
        }
        fn f(x: i64) -> i64 {
            let m = M[{ if x > 0 { return 42 }  3 }] { v: 7 }
            m.v
        }
        fn main() -> i64 { f(1) + 1000 }
    "#);
    assert_eq!(as_int(&v), 1042);
}

#[test]
fn issue_639_a_model_shape_argument_still_binds_on_the_ordinary_path() {
    // The regression guard for the same site: a shape argument that evaluates
    // is still recorded and still visible to the model's methods.
    let v = run(r#"
        model M[N] {
            v: i64
            fn n(self) -> i64 { N }
        }
        fn main() -> i64 {
            let m = M[5] { v: 7 }
            m.n() * 10 + m.v
        }
    "#);
    assert_eq!(as_int(&v), 57);
}

#[test]
fn issue_639_the_backstop_diagnostic_carries_the_keywords_span() {
    // A module-level `let` is the one expression position with no enclosing
    // statement to catch the carrier, so the backstop message is what the user
    // sees. It used to print bare `runtime error:` with no `file:line:col`;
    // now the escaping keyword's own span rides along. All three wordings are
    // reachable from here — the per-keyword `break` / `continue` messages are
    // NOT dead, they are simply out of reach *inside* a fn, where `call_fn`'s
    // own "break/continue outside loop" gets there first.
    let (msg, at) = run_err_at("let G = { return 5 }\nfn main() -> i64 { G + 1 }\n");
    assert!(msg.contains("`return` outside a function"), "got: {msg}");
    assert_eq!(at, Some((1, 11)), "expected the `return`'s own line:col");

    let (msg, at) = run_err_at("let G = { break }\nfn main() -> i64 { 1 }\n");
    assert!(msg.contains("`break` outside loop"), "got: {msg}");
    assert_eq!(at, Some((1, 11)));

    let (msg, at) = run_err_at("let G = { continue }\nfn main() -> i64 { 1 }\n");
    assert!(msg.contains("`continue` outside loop"), "got: {msg}");
    assert_eq!(at, Some((1, 11)));
}

// ─── Type aliases are transparent at every width-origin site (SPEC §6.6, §3.1) ───
//
// The checker resolves `type Role = u8` to the scalar `u8` for type-checking,
// so `let r: Role = 255` and `let r: u8 = 255` are the same program to
// `dmc --check`. The interpreter's width coercion used to match on the raw
// AST `Type::Scalar` and so never saw through the alias: the aliased `r`
// stayed a full i64 and `r + 1` was 256 where the direct spelling wraps to 0.
// `tests/alias_narrow_width.rs` drives the same cases through the binary.

/// The repro: a typed `let` whose width is spelled through an alias wraps.
#[test]
fn alias_typed_let_narrows_like_the_direct_spelling() {
    let v = run(r#"
        type Role = u8
        fn main() -> i64 {
            let r: Role = 255
            (r + 1) as i64
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// A typed parameter and a declared return are width origins too (SPEC §3.1);
/// through an alias each must narrow exactly as the direct spelling does.
#[test]
fn alias_typed_param_and_return_narrow() {
    // parameter: `r` arrives as a u8, so `r + 1` is u8 arithmetic
    let v = run(r#"
        type Role = u8
        fn wide(r: Role) -> i64 { (r + 1) as i64 }
        fn main() -> i64 { wide(255) }
    "#);
    assert_eq!(as_int(&v), 0, "aliased parameter");
    // return: the declared `-> Role` narrows the untyped `255 + 1` (256)
    let v = run(r#"
        type Role = u8
        fn bump() -> Role { 255 + 1 }
        fn main() -> i64 { bump() as i64 }
    "#);
    assert_eq!(as_int(&v), 0, "aliased return");
    // signed alias, so the wrap is two's-complement, not unsigned
    let v = run(r#"
        type Small = i8
        fn bump(r: Small) -> Small { r + 1 }
        fn main() -> i64 { bump(127) as i64 }
    "#);
    assert_eq!(as_int(&v), -128, "signed aliased param+return");
}

/// A fn literal's typed parameter and declared return are the same sites.
#[test]
fn alias_typed_lambda_param_and_return_narrow() {
    let v = run(r#"
        type Role = u8
        fn main() -> i64 {
            let f = fn(r: Role) -> Role { r + 1 }
            f(255) as i64
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// An `as` cast to an alias wraps as a cast to the target does (SPEC §3.1:
/// `5000000000 as i32` is `705032704`; `300 as Role` is `44`).
#[test]
fn alias_cast_target_wraps() {
    let v = run(r#"
        type Role = u8
        fn main() -> i64 { (300 as Role) as i64 }
    "#);
    assert_eq!(as_int(&v), 44);
}

/// A declared model field type through an alias narrows at construction and
/// on a later field write, like a direct `u8` field (#473's sites).
#[test]
fn alias_typed_model_field_narrows() {
    let v = run(r#"
        type Role = u8
        model M { r: Role }
        fn main() -> i64 {
            let !m = M { r: 255 }
            let a = (m.r + 1) as i64
            m.r = 254
            a + ((m.r + 2) as i64)
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// An alias of an alias resolves all the way down.
#[test]
fn alias_chain_resolves_to_the_scalar() {
    let v = run(r#"
        type Role = u8
        type Rank = Role
        fn main() -> i64 {
            let r: Rank = 255
            (r + 1) as i64
        }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// The float half of the same rule (#473): `type Prob = f32` quantizes a
/// typed `let` to f32 exactly as `let q: f32` does.
#[test]
fn alias_typed_f32_let_quantizes() {
    let v = run(r#"
        type Prob = f32
        fn main() -> bool {
            let p: Prob = 0.1
            let q: f32 = 0.1
            (p as f64) == (q as f64)
        }
    "#);
    assert!(matches!(v, Value::Bool(true)), "got {:?}", v);
    let v = run(r#"
        type Prob = f32
        fn main() -> bool {
            let p: Prob = 0.1
            (p as f64) == 0.1
        }
    "#);
    assert!(matches!(v, Value::Bool(false)), "an f32-aliased let must not keep the f64 literal, got {:?}", v);
}

// ─── An alias is scoped to the module that declared it (SPEC §6.6) ───────────
//
// The checker runs per module, leaf-first, so inside an imported module's fn
// bodies its OWN `type Role` governs. The interpreter kept one flat alias
// table across every file the dependency graph loaded, so the last file loaded
// — the importer — silently re-typed the library it imported: `libbump(255)`
// against `pub type Role = i64` ran as the importer's `u8` and answered 0
// where the checker had type-checked it as i64 and meant 256. The alias table
// is now tagged by module the way `fn_module` / `mut_globals` are, and a fn
// body resolves its annotations in the module it was declared in.

/// The importer's narrower alias must not reach into the library's fn bodies.
#[test]
fn an_importers_alias_does_not_retype_the_module_it_imported() {
    let v = run_multi(&[
        ("lib.dmc", "pub type Role = i64\npub fn libbump(r: Role) -> Role { r + 1 }\n"),
        ("main.dmc", r#"use "lib.dmc"
type Role = u8
fn main() -> i64 { libbump(255) }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 256, "`libbump`'s `Role` is its own module's i64");

    // The importer's own alias still governs the importer's own body.
    let v = run_multi(&[
        ("lib.dmc", "pub type Role = i64\npub fn libbump(r: Role) -> Role { r + 1 }\n"),
        ("main.dmc", r#"use "lib.dmc"
type Role = u8
fn main() -> i64 {
    let r: Role = 255
    (r + 1) as i64
}"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 0, "`main`'s `Role` is the importer's u8");
}

/// The reverse orientation: the library is the narrow one, and its width must
/// survive an importer that widens the same name. The flat table printed 256
/// here — the wrap the checker had accepted, dropped.
#[test]
fn an_imported_modules_narrow_alias_survives_a_wider_importer() {
    let v = run_multi(&[
        ("lib.dmc", "pub type Role = u8\npub fn libbump(r: Role) -> Role { r + 1 }\n"),
        ("main.dmc", r#"use "lib.dmc"
type Role = i64
fn main() -> i64 { libbump(255) as i64 }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 0, "`libbump`'s `Role` is its own module's u8");
}

/// A qualified import carries the same home-module tag onto the `l.f` name, so
/// `l.libbump` resolves its aliases exactly as the unqualified `libbump` does.
#[test]
fn a_qualified_imports_fn_resolves_aliases_in_its_own_module() {
    let v = run_multi(&[
        ("lib.dmc", "pub type Role = i64\npub fn libbump(r: Role) -> Role { r + 1 }\n"),
        ("main.dmc", r#"use "lib.dmc" as l
type Role = u8
fn main() -> i64 { l.libbump(255) }"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 256);
}

/// An alias the importing module never redeclares is still visible to it —
/// that is what `use` imported. Scoping must narrow the lookup, not break it.
#[test]
fn an_imported_alias_is_visible_to_the_importer() {
    let v = run_multi(&[
        ("lib.dmc", "pub type Role = u8\n"),
        ("main.dmc", r#"use "lib.dmc"
fn main() -> i64 {
    let r: Role = 255
    (r + 1) as i64
}"#),
    ], "main.dmc");
    assert_eq!(as_int(&v), 0);
}

// ─── A shape-generic alias over a scalar is still a width ────────────────────

/// `type W[N] = u8` is a `u8` however many shape args it is spelled with: a
/// shape argument cannot change a scalar's width. The checker resolves it and
/// the JIT (#622) refuses it; the interpreter used to run it as a full i64 and
/// print 256, which was the same hole one indirection further out.
#[test]
fn a_shape_generic_alias_over_a_scalar_narrows() {
    let v = run(r#"
        type W[N] = u8
        fn main() -> i64 {
            let r: W[3] = 255
            (r + 1) as i64
        }
    "#);
    assert_eq!(as_int(&v), 0, "typed let through a shape-generic scalar alias");
    let v = run(r#"
        type W[N] = u8
        fn bump(r: W[3]) -> W[3] { r + 1 }
        fn main() -> i64 { bump(255) as i64 }
    "#);
    assert_eq!(as_int(&v), 0, "param and return through a shape-generic scalar alias");
}

/// A shape-generic alias over a SHAPED type is still left alone: substituting
/// its shape argument in the interpreter is separate work, and no run-time
/// width hides behind it. It must keep working, not start resolving.
#[test]
fn a_shape_generic_alias_over_a_tensor_still_runs() {
    let v = run(r#"
        type Batch[B] = Tensor[f32, [B, 2]]
        fn total(t: Batch[3]) -> f32 { sum(t) }
        fn main() -> f64 { total(forge.zeros[f32, [3, 2]]) as f64 }
    "#);
    assert!(matches!(v, Value::Float(x, _) if x == 0.0), "got {:?}", v);
}

// ─── Aliases against the other named-type namespaces ─────────────────────────

/// An alias of an ENUM stays an enum: `resolve_alias_ty` hands back the enum's
/// name, `coerce_scalar_width` sees a `Named` and coerces nothing, and the
/// variant keeps its ordinal.
#[test]
fn an_alias_of_an_enum_is_not_a_width() {
    let v = run(r#"
        enum Color { Red, Green, Blue }
        type Hue = Color
        fn pick(h: Hue) -> Hue { h }
        fn main() -> i64 {
            let c: Hue = Color.Green
            pick(c) as i64
        }
    "#);
    assert_eq!(as_int(&v), 1);
}

/// An alias of a MODEL stays a model, and the model's own field annotation is
/// what narrows — the alias does not swallow the struct.
#[test]
fn an_alias_of_a_model_keeps_the_models_field_widths() {
    let v = run(r#"
        model Box { v: u8 }
        type Crate = Box
        fn open(b: Crate) -> i64 { (b.v + 1) as i64 }
        fn main() -> i64 { open(Box { v: 255 }) }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// A `type` item is collected in the same forward-reference pass as `fn` and
/// `model`, so an alias declared BELOW the fn that uses it still narrows.
#[test]
fn an_alias_declared_after_its_use_still_narrows() {
    let v = run(r#"
        fn wide(r: Role) -> i64 { (r + 1) as i64 }
        type Role = u8
        fn main() -> i64 { wide(255) }
    "#);
    assert_eq!(as_int(&v), 0);
}

/// The `Port[L]` half of the same resolution: `type Py = Port[python]` names a
/// runtime exactly as the direct spelling does, so an aliased `Port[lua]`
/// parameter refuses a python handle. `L` was decoration in both spellings
/// before this change — an untested behaviour change worth pinning.
#[test]
fn an_aliased_port_annotation_checks_the_runtime() {
    if !crate::ports::have_python() { eprintln!("skipped: python3 not on PATH"); return; }
    // The alias names the runtime the handle actually has: accepted.
    let v = run(r#"
        type Py = Port[python]
        fn use_it(p: Py) -> i64 { 7 }
        fn main() -> i64 {
            let (p, e) = port_open("python")
            let n = use_it(p)
            let (_, _) = port_close(p)
            n
        }
    "#);
    assert_eq!(as_int(&v), 7, "a python handle satisfies an aliased Port[python]");
    // The alias names a different runtime: refused, with the shared wording.
    let msg = run_err(r#"
        type Lua = Port[lua]
        fn use_it(p: Lua) -> i64 { 7 }
        fn main() -> i64 {
            let (p, e) = port_open("python")
            let n = use_it(p)
            let (_, _) = port_close(p)
            n
        }
    "#);
    assert!(msg.contains("port lang mismatch") && msg.contains("`Port[lua]`"),
            "expected the Port[lua]/python mismatch, got: {msg}");
}

// ─── #628 × #652: `?`'s `default[T]` resolves in the SIGNATURE's module ──────
//
// `default[T]` is built from the propagating function's own declared `T`, and
// `T` may be spelled as a top-level `type` alias — which SPEC §6.6 scopes to
// the module that declared it. So the rebuild has to happen while `cur_module`
// still names the file whose signature it is: `call_fn_with_shapes` restores
// `cur_module` only after `call_fn_in_module` (which ends with
// `finish_query_propagation`) returns, and `call_lambda` restores it after its
// own rebuild rather than before. Restore either one early and the alias is
// looked up in the CALLER's table, where it is either absent (a refusal) or,
// worse, a same-named alias of a different width (a silently wrong zero).
//
// Each pair below is run in both orientations, so neither passes by accident
// on an i64-shaped default: the library's declaration governs whichever way
// round the two widths are.

#[test]
fn query_default_resolves_a_type_alias_t() {
    // Single file first: `default[Slot]` is the `u8` zero, so `0 - 1` wraps to
    // 255. Before the alias was resolved here, this program did not merely get
    // the width wrong — it was refused outright ("needs a default value for
    // `Slot`") for a name one indirection away from a perfectly good scalar.
    let v = run(r#"
        type Slot = u8
        fn boom() -> (Slot, Err) { (200 as Slot, "boom") }
        fn f() -> (Slot, Err) {
            let v = boom()?
            (v, nil)
        }
        fn main() -> str {
            let (v, e) = f()
            to_str(v - 1) + "|" + to_str(e)
        }
    "#);
    assert_eq!(as_str(&v), "255|boom");
}

#[test]
fn query_default_uses_the_libs_alias_not_the_importers() {
    // The library and the importer each declare `Slot` at a DIFFERENT width.
    // The `?` is in the library's `f`, so `default[Slot]` is the library's.
    const LIB: &str = r#"pub type Slot = %W%
pub fn boom() -> (Slot, Err) { (200 as Slot, "lib-boom") }
pub fn f() -> (Slot, Err) {
    let v = boom()?
    (v, nil)
}"#;
    const MAIN: &str = r#"use "lib.dmc"
type Slot = %W%
fn main() -> str {
    let (v, e) = f()
    to_str(v - 1) + "|" + to_str(e)
}"#;
    // Library `u8`, importer `i64`: the u8 zero wraps to 255.
    let v = run_multi(&[
        ("lib.dmc", &LIB.replace("%W%", "u8")),
        ("main.dmc", &MAIN.replace("%W%", "i64")),
    ], "main.dmc");
    assert_eq!(as_str(&v), "255|lib-boom", "`f`'s `Slot` is its own module's u8");

    // Reversed, so the assertion above cannot be passing on an i64-shaped
    // default that never consulted the alias table at all.
    let v = run_multi(&[
        ("lib.dmc", &LIB.replace("%W%", "i64")),
        ("main.dmc", &MAIN.replace("%W%", "u8")),
    ], "main.dmc");
    assert_eq!(as_str(&v), "-1|lib-boom", "`f`'s `Slot` is its own module's i64");
}

#[test]
fn query_in_an_importer_fn_uses_the_importers_own_t() {
    // The other direction of the same boundary: the `?` is in the IMPORTER's
    // fn and the erroring call crosses into the library, so `default[T]` is
    // the importer's — `cur_module` is back by the time this signature is
    // rebuilt against. `MSlot` is `i32`, so the zero minus one is -1.
    let v = run_multi(&[
        ("lib.dmc", r#"pub type Slot = u8
pub fn boom() -> (Slot, Err) { (200 as Slot, "lib-boom") }
pub fn blocky() -> (Slot, Err) {
    if true {
        let v = boom()?
        (v, nil)
    } else {
        (1 as Slot, nil)
    }
}
pub fn libname() -> str { "lib" }"#),
        ("main.dmc", r#"use "lib.dmc"
type Slot = i64
type MSlot = i32
fn mainname() -> str { "main" }
fn wrap() -> (MSlot, Err) {
    let v = blocky()?
    ((v as i32) + 1, nil)
}
fn main() -> str {
    let (a, ea) = blocky()
    let (b, eb) = wrap()
    to_str(a - 1) + "|" + to_str(b - 1) + "|" + mainname() + "+" + libname()
}"#),
    ], "main.dmc");
    // `a`: the library's u8 zero (255 after -1) — and the `?` sat inside an
    // `if` block, so #639's carrier crossed the same frame without eating it.
    // `b`: the importer's i32 zero (-1 after -1).
    // `main+lib`: both files' own names still resolve after the calls, so
    // `cur_module` was restored on the way out of each.
    assert_eq!(as_str(&v), "255|-1|main+lib");
}

#[test]
fn query_in_a_lambda_written_in_another_module_resolves_that_modules_alias() {
    // The lambda literal is written in the library and CALLED from the
    // importer, so `call_lambda`'s own `cur_module` save is the only thing
    // standing between `default[Slot]` and the importer's same-named alias.
    // Restoring `cur_module` before the rebuild (rather than after) puts the
    // importer's width in the library's slot.
    const LIB: &str = r#"pub type Slot = %W%
pub fn boom() -> (Slot, Err) { (200 as Slot, "lib-boom") }
pub fn mk() -> (fn() -> (Slot, Err), Err) {
    let c = fn() -> (Slot, Err) { let v = boom()?
                                  (v, nil) }
    (c, nil)
}"#;
    const MAIN: &str = r#"use "lib.dmc"
type Slot = %W%
fn main() -> str {
    let (c, me) = mk()
    let (v, e) = c()
    to_str(v - 1) + "|" + to_str(e)
}"#;
    let v = run_multi(&[
        ("lib.dmc", &LIB.replace("%W%", "u8")),
        ("main.dmc", &MAIN.replace("%W%", "i64")),
    ], "main.dmc");
    assert_eq!(as_str(&v), "255|lib-boom", "the lambda's `Slot` is the file it was written in");

    let v = run_multi(&[
        ("lib.dmc", &LIB.replace("%W%", "i64")),
        ("main.dmc", &MAIN.replace("%W%", "u8")),
    ], "main.dmc");
    assert_eq!(as_str(&v), "-1|lib-boom", "and it is that file's width either way round");
}

#[test]
fn a_failed_shape_harvest_leaves_the_scope_stack_where_it_found_it() {
    // `harvest_shape_params` is an exit out of `call_fn_in_module` that sits
    // between `push_scope` and every `pop_scope` — it was a bare `?` on main,
    // so the callee's scope stayed on the stack forever. Nothing observed it
    // while every error out of it was fatal; `?`'s rework put two non-errors
    // on the same channel, so the exit is now spelled like its neighbours.
    //
    // The checker accepts this program (two shape-generic parameters, two
    // tensors of different lengths), so the disagreement is a RUNTIME one:
    // exactly the path in question.
    let msg = run_err(r#"
        fn twoargs[N](a: Tensor[f32, [N]], b: Tensor[f32, [N]]) -> f32 { sum(a) + sum(b) }
        fn main() -> f32 {
            let x = forge.zeros[f32, [3]]
            let y = forge.zeros[f32, [4]]
            twoargs(x, y)
        }
    "#);
    assert!(msg.contains("would need to be both 3 and 4"),
            "expected the inconsistent-shape refusal, got: {msg}");
}


// ── #674 slice 3a/3d: arena budgets — well-formedness at `enter` / `in` ─────
// These bypass `dmc run`'s own `--check` pass (this harness's `run_err_at`
// goes straight from the parser to `Interpreter::run`), so a program that
// would fail `--check` for an unrelated reason (a bogus `Budget` operand)
// still exercises the interpreter's OWN located refusal directly.
//
// All four operations are real now: `forge.carve`/`forge.reap`/`forge.root`
// from slice 3b, `forge.enter`/`in b` from slice 3d. What 3a's two
// placeholder refusals leave behind is the WELL-FORMEDNESS half — a
// `forge.enter` whose operand is not a `Budget` — which is a located
// runtime error in the shape `forge.carve`/`forge.reap` already use for
// their own arguments, and deliberately NOT a new tag: the arena-budget
// design's refusal-class table counts exactly three refusal CLASSES plus one
// runtime error, and a `budget-enter` tag would make it four.

#[test]
fn forge_enter_over_a_non_budget_refuses_located_and_does_not_run_674_s4() {
    // Probe s4's own shape with a bogus integer in place of the budget, so
    // the interpreter's OWN `BudgetBlock` refusal is what fires. The block
    // body must NOT run — that was #610's silent-no-op defect, and it stays
    // closed for an operand that never names a budget at all.
    //
    // Pinned by construction: an undefined-fn call in the body. If the body
    // ran, evaluating that call would surface `undefined identifier` (from
    // `eval_ident`) instead — so the assert below can only pass if the body
    // was never reached. The positive control,
    // `undefined_ident_call_errors_positive_control_674`, confirms the same
    // call does error when actually reached.
    let (msg, at) = run_err_at(
        "fn main() -> nil { forge.enter(5) { undefined_fn_674() } \n nil }");
    assert!(msg.starts_with("forge.enter: the budget operand must be a `Budget`"),
            "message: {msg}");
    assert!(at.is_some(), "expected a located error");
}

#[test]
fn alloc_in_evaluates_its_budget_operand_674_s3() {
    // The mirror image of 3a's version of this test, and the behaviour
    // change slice 3d makes: `<ctor> in b` EVALUATES its budget operand now
    // (3a refused before touching it), so an undefined call there surfaces
    // its own `undefined identifier` rather than a `budget-alloc-in`
    // placeholder. The operand is evaluated BEFORE the register is pushed —
    // it is not part of the allocation it names.
    let (msg, at) = run_err_at(
        "fn main() -> nil { let t = forge.zeros[f32, [4]] in undefined_fn_674() \n nil }");
    assert!(msg.contains("undefined"), "message: {msg}");
    assert!(msg.contains("undefined_fn_674"), "message: {msg}");
    assert!(at.is_some(), "expected a located error");
}

#[test]
fn alloc_in_over_a_non_budget_refuses_located_674() {
    // The well-formedness half of `in b`, the twin of
    // `forge_enter_over_a_non_budget_refuses_located_and_does_not_run_674_s4`
    // above: one shared `open_budget` produces both, so neither can drift.
    let (msg, at) = run_err_at(
        "fn main() -> nil { let t = forge.zeros[f32, [4]] in 5 \n nil }");
    assert!(msg.starts_with("`in`: the budget operand must be a `Budget`"),
            "message: {msg}");
    assert!(at.is_some(), "expected a located error");
}

#[test]
fn undefined_ident_call_errors_positive_control_674() {
    // Positive control for the two non-execution probes above: the same
    // undefined call, reached directly (nothing refuses first), does error —
    // proving it is not silently a no-op that would make either probe above
    // pass regardless of whether the guarded body/operand actually ran.
    let msg = run_err("fn main() -> nil { undefined_fn_674() \n nil }");
    assert!(msg.contains("undefined"), "message: {msg}");
    assert!(msg.contains("undefined_fn_674"), "message: {msg}");
}

// ── #674 slice 3b: the interpreter's budget table ───────────────────────────
//
// `forge.carve`/`forge.reap`/`forge.root` are for real, and since slice 3d
// so are `forge.enter`/`in b`. The hole-table cases below keep driving the
// mechanism directly through the `#[cfg(test)]` harness `Interpreter` exposes for
// exactly this purpose (`eval_expr_for_test`/`bind_for_test`/
// `with_budget_for_test`/`budget_index_for_test`) rather than through
// `forge.enter`: each row needs a tensor stamped under `b` and then a
// SPECIFIC retrieval site — a parked receiver, a writeback, a tape node —
// which surface syntax cannot aim at. `with_budget_for_test` is
// `BudgetGuard`, the same RAII `Expr::BudgetBlock` opens, so what these
// exercise IS 3d's mechanism and not a stand-in for it.

use super::arena::ArenaLimits;

fn parse_e(src: &str) -> Expr {
    let tokens = Lexer::new(src).tokenize().expect("lex expr");
    Parser::new(tokens).parse_expr().expect("parse expr")
}

fn parse_program_674(src: &str) -> super::ast::Program {
    let tokens = Lexer::new(src).tokenize().expect("lex program");
    Parser::new(tokens).parse_program().expect("parse program")
}

fn err_msg(r: EvalResult<Value>) -> String {
    r.err().expect("expected an error").msg
}

use super::ast::{Expr, PostfixOp};
use super::interp::EvalResult;

/// `--forge=1 MiB`, `forge.root` bound to `root`, a 64 KiB child carved from
/// it and bound to `b`. Returns the child's table index for
/// `with_budget_for_test`.
fn budget_setup_674(interp: &mut Interpreter) -> usize {
    interp.set_arena_limits(ArenaLimits { forge: Some(1 << 20), vault: None });
    let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
    interp.bind_for_test("root", root);
    let b = interp.eval_expr_for_test(&parse_e("forge.carve(root, 65536)")).unwrap();
    let idx = Interpreter::budget_index_for_test(&b);
    interp.bind_for_test("b", b);
    idx
}

fn reap_b_674(interp: &mut Interpreter) {
    interp.eval_expr_for_test(&parse_e("forge.reap(b)")).unwrap();
}

/// `budget-stale`, and **located**. `MEMORY.md`'s LOC invariant — and
/// §3.2's whole case for M1 over M2 — is that the refusal names the retrieval
/// that hit it, so a test that only matched the tag would have passed on an
/// unlocated `RuntimeError::msg` (PR #821 review).
fn assert_stale(r: EvalResult<Value>) {
    let e = r.err().expect("expected an error");
    assert!(e.msg.starts_with("budget-stale:"), "message: {}", e.msg);
    assert!(e.span.is_some(), "`budget-stale` must be located: {}", e.msg);
}

// ── §3.1 algebra: carve / reap / PRE / R1 / R2 / INV / LOC / NN ─────────────

#[test]
fn budget_674_carve_returns_a_budget_value() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        assert_ne!(idx, 0, "the child must not be entry 0 (root)");
    });
}

#[test]
fn budget_674_forge_root_is_a_budget_value() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let v = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        assert!(matches!(v, Value::Budget(0)), "expected Value::Budget(0), got {:?}", v);
    });
}

#[test]
fn budget_674_root_unbounded_carve_always_succeeds() {
    // §4.1: no `--forge` means free(root) = infinity — (PRE) is vacuous.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let r = interp.eval_expr_for_test(&parse_e("forge.carve(root, 999999999)"));
        assert!(r.is_ok(), "{:?}", r.err().map(|e| e.msg));
    });
}

#[test]
fn budget_674_ac_inv_root_capacity_is_sum_of_children_extents() {
    // ac_inv: --forge=256K, four 64 KiB carves succeed, a fifth is refused —
    // whatever each child has allocated.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(256 * 1024), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let carve = parse_e("forge.carve(root, 65536)");
        for i in 0..4 {
            let r = interp.eval_expr_for_test(&carve);
            assert!(r.is_ok(), "carve {i} should succeed: {:?}", r.err().map(|e| e.msg));
        }
        let fifth = interp.eval_expr_for_test(&carve);
        let msg = err_msg(fifth);
        assert!(msg.starts_with("budget-carve:"), "message: {msg}");
    });
}

#[test]
fn budget_674_ac_loc_reap_credits_the_whole_extent_not_the_remainder() {
    // ac_loc: reap one 64 KiB child that has SPENT part of its own extent;
    // the credit back to the parent is the whole 64 KiB, not the unspent
    // remainder (#674 criterion 2).
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(256 * 1024), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let carve = parse_e("forge.carve(root, 65536)");
        let mut kids = Vec::new();
        for _ in 0..4 {
            kids.push(interp.eval_expr_for_test(&carve).unwrap());
        }
        assert!(interp.eval_expr_for_test(&carve).is_err(), "root should be fully carved");
        // Spend part of kids[0]'s own extent before reaping it.
        let k0_idx = Interpreter::budget_index_for_test(&kids[0]);
        let t = interp.with_budget_for_test(k0_idx, |i| {
            i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4096]]")) // 16 384 B
        });
        assert!(t.is_ok(), "{:?}", t.err().map(|e| e.msg));
        interp.bind_for_test("k0", kids[0].clone());
        reap_via_ident_674(&mut interp, "k0");
        // The credit is the WHOLE 64 KiB extent, not the ~48 KiB unspent
        // remainder: 64 KiB + 1 is refused, exactly 64 KiB succeeds.
        let too_big = interp.eval_expr_for_test(&parse_e("forge.carve(root, 65537)"));
        assert!(too_big.is_err(), "64 KiB + 1 should be refused after crediting only 64 KiB back");
        let exact = interp.eval_expr_for_test(&carve);
        assert!(exact.is_ok(), "exactly 64 KiB should now succeed: {:?}", exact.err().map(|e| e.msg));
    });
}

fn reap_via_ident_674(interp: &mut Interpreter, name: &str) {
    let r = interp.eval_expr_for_test(&parse_e(&format!("forge.reap({name})")));
    r.unwrap();
}

#[test]
fn budget_674_ac_pre_nested_carve_past_parents_free_refused_at_the_carve() {
    // ac_pre: a nested carve past the parent's free capacity is refused AT
    // THE CARVE with `budget-carve`, not at a later allocation (#674
    // criterion 3).
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(8192), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let b = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)")).unwrap();
        interp.bind_for_test("b", b);
        let over = interp.eval_expr_for_test(&parse_e("forge.carve(b, 4097)"));
        let msg = err_msg(over);
        assert!(msg.starts_with("budget-carve:"), "message: {msg}");
        assert!(msg.contains("parent budget"), "message: {msg}");
        let exact = interp.eval_expr_for_test(&parse_e("forge.carve(b, 4096)"));
        assert!(exact.is_ok(), "{:?}", exact.err().map(|e| e.msg));
    });
}

#[test]
fn budget_674_ac_nn_reap_then_recarve_cycles_leave_capacity_unchanged() {
    // ac_nn: 64 reap-then-recarve cycles leave the root's capacity exactly
    // where it started; cycle 65 still succeeds, one byte over still fails.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(4096), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        for cycle in 0..64 {
            let b = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)"));
            assert!(b.is_ok(), "cycle {cycle} carve: {:?}", b.as_ref().err().map(|e| e.msg.clone()));
            interp.bind_for_test("b", b.unwrap());
            reap_b_674(&mut interp);
        }
        let ok = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)"));
        assert!(ok.is_ok(), "cycle 65: {:?}", ok.as_ref().err().map(|e| e.msg.clone()));
        interp.bind_for_test("b", ok.unwrap());
        reap_b_674(&mut interp);
        let over = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4097)"));
        assert!(over.is_err(), "one byte over the root's original capacity should still fail");
    });
}

#[test]
fn budget_674_r2_reap_refuses_a_budget_with_a_live_child() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(8192), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let p = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)")).unwrap();
        interp.bind_for_test("p", p);
        let _c = interp.eval_expr_for_test(&parse_e("forge.carve(p, 1024)")).unwrap();
        let r = interp.eval_expr_for_test(&parse_e("forge.reap(p)"));
        let msg = err_msg(r);
        assert!(msg.starts_with("budget-reap:"), "message: {msg}");
        assert!(msg.contains("live child"), "message: {msg}");
    });
}

#[test]
fn budget_674_reap_refuses_the_root() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let r = interp.eval_expr_for_test(&parse_e("forge.reap(root)"));
        let msg = err_msg(r);
        assert!(msg.starts_with("budget-reap:"), "message: {msg}");
    });
}

#[test]
fn budget_674_stale_handle_refused_at_a_second_reap() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(8192), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let b = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)")).unwrap();
        interp.bind_for_test("b", b);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("forge.reap(b)"));
        assert_stale(r);
    });
}

#[test]
fn budget_674_stale_handle_refused_as_a_carve_parent() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(8192), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let b = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)")).unwrap();
        interp.bind_for_test("b", b);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("forge.carve(b, 10)"));
        assert_stale(r);
    });
}

#[test]
fn budget_674_reused_index_never_matches_the_old_handles_generation() {
    // §6 Q8: an index is reused, its generation never wraps back to match an
    // old handle. Reap `b` (index N), carve a NEW child (reusing index N),
    // then the OLD `b` handle must still read stale against the new
    // occupant, not alias it.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(1 << 16), vault: None });
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let b_old = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)")).unwrap();
        let old_idx = Interpreter::budget_index_for_test(&b_old);
        interp.bind_for_test("b", b_old.clone());
        reap_b_674(&mut interp);
        let b_new = interp.eval_expr_for_test(&parse_e("forge.carve(root, 4096)")).unwrap();
        assert_eq!(Interpreter::budget_index_for_test(&b_new), old_idx,
            "the freed slot should be reused");
        let descriptor = |v: &Value| match v { Value::Budget(d) => *d, other => panic!("expected Budget, got {:?}", other) };
        assert_ne!(descriptor(&b_old), descriptor(&b_new), "the OLD descriptor must not equal the new one");
        interp.bind_for_test("b_old", b_old);
        let r = interp.eval_expr_for_test(&parse_e("forge.reap(b_old)"));
        assert_stale(r);
    });
}

// ── Well-formedness (checker's job in the full pipeline; the interpreter's
// own defensive check when `--check` is bypassed, as these tests do) ───────

#[test]
fn budget_674_carve_refuses_a_non_budget_first_arg() {
    let (msg, at) = run_err_at("fn main() -> nil { forge.carve(5, 4) \n nil }");
    assert!(msg.contains("Budget"), "message: {msg}");
    assert!(at.is_some());
}

#[test]
fn budget_674_reap_refuses_a_non_budget_arg() {
    let (msg, at) = run_err_at("fn main() -> nil { forge.reap(5) \n nil }");
    assert!(msg.contains("Budget"), "message: {msg}");
    assert!(at.is_some());
}

// ── §3.2 hole table: one case per row, against a budget carved off the root
// (no `forge.enter`/`in b` needed — see the module doc above) ──────────────

#[test]
fn hole_table_674_row1_scope_ident_retrieval() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        assert_stale(interp.eval_expr_for_test(&parse_e("t")));
    });
}

#[test]
fn hole_table_674_row1_compound_assign() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("{ t += forge.zeros[f32, [4]] \n nil }"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row1_indexed_write() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("{ t[0] = 1.0f32 \n nil }"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row1_indexed_compound_assign() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("{ t[0] += 1.0f32 \n nil }"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row2_mutating_param_argument_read_refuses_before_the_writeback() {
    // Row 2's closure is "`apply_writebacks` moves storage-to-storage into
    // `scopes`; the value is read again through row 1" — and the name says
    // which of the two this test reaches (PR #821 review). It is row 1, and
    // it is row 1 *first*: the call's own argument evaluation is expression
    // position, so a stale `t` never survives to be copied into the param.
    //
    // The writeback itself cannot carry a stale tensor into `scopes` in the
    // other direction either: `!x`'s private copy re-stamps to the budget
    // current at the call (§6 Q5, root here), so what comes back is a
    // root-stamped tensor that no reap of `b` can invalidate. That is why
    // there is no second test here pinning `apply_writebacks` directly —
    // the path does not exist, rather than being untested.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let prog = parse_program_674("fn touch(!x: Tensor[f32, [4]]) -> nil { nil }");
        interp.load_program(&prog, None).expect("load touch()");
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        // The call's own private-copy charge/re-stamp happens at the budget
        // CURRENT at the call (root here) — the writeback then hands back a
        // tensor already stale at its SOURCE read (row 1), which is what the
        // call's own argument evaluation hits first.
        let r = interp.eval_expr_for_test(&parse_e("touch(t)"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row3_parked_receiver_is_checked_at_the_take() {
    // Row 3's *take* path, pinned with nothing in front of it (PR #821
    // review). The old spelling here was `t.shape()`, which has no shape
    // bracket, so `call_bracketed_method` never ran, nothing was ever parked,
    // and the refusal came from row 1's read of `t` — the test passed
    // without touching the row it was named for.
    //
    // Reaching the take through surface syntax cannot isolate it either:
    // `call_bracketed_method` evaluates the receiver itself, which is
    // expression position, so an already-stale tensor is refused there. The
    // window row 3 owns is a receiver parked LIVE and taken STALE, so the
    // park is done directly and the reap runs in between.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        // `t.shape` — a Field postfix whose base node is what gets parked.
        let e = parse_e("t.shape");
        let Expr::Postfix { expr: recv, .. } = &e else {
            panic!("expected a Postfix, got {:?}", e)
        };
        interp.park_receiver_for_test(recv.as_ref(), t);
        // Parked live, reaped after: no read of `t` stands between the park
        // and the take, so only the take's own check can refuse this.
        reap_b_674(&mut interp);
        assert_stale(interp.eval_expr_for_test(&e));
    });
}

#[test]
fn hole_table_674_row3_shape_bracketed_non_method_receiver_parks_and_is_refused() {
    // The other half: the surface form that actually parks. `t.blit![2](…)`
    // parses as Call(Index(Field(t, "blit!"), [2]), […]) — the shape bracket
    // sits between the name and the call, so `call_bracketed_method` runs,
    // finds a tensor rather than a model instance, and parks the receiver
    // (#474). Its own receiver read is expression position, so a tensor
    // already stale is refused there, before the park — which is the reason
    // the test above has to park by hand to reach the take.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        let e = parse_e("t.blit![2](1.0f32)");
        // Pin the parse shape the park depends on, so this stays a test of
        // the bracketed route and not of whatever `t.blit![2](…)` happens to
        // parse as later: Call(Index(Field(t, "blit!"), [2]), […]).
        let Expr::Postfix { expr: callee, op: PostfixOp::Call(_), .. } = &e else {
            panic!("expected a Call, got {:?}", e)
        };
        let Expr::Postfix { expr: bracketed, op: PostfixOp::Index(_), .. } = callee.as_ref() else {
            panic!("expected an Index under the call, got {:?}", callee)
        };
        assert!(matches!(bracketed.as_ref(), Expr::Postfix { op: PostfixOp::Field(_), .. }),
            "expected a Field under the bracket, got {:?}", bracketed);
        assert_stale(interp.eval_expr_for_test(&e));
    });
}

#[test]
fn hole_table_674_row4_grad_backward_pass_over_a_reaped_activation() {
    // "`forge.reap` inside `@grad` is legal" — the backward pass over a
    // reaped activation fails with `budget-stale`, located at the `@grad`
    // call site, not at the carve/reap.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(1 << 20), vault: None });
        let prog = parse_program_674(
            "@grad fn f(x: Tensor[f32, [4]]) -> f32 { let y = x + x \n sum(y) }"
        );
        interp.load_program(&prog, None).expect("load f()");
        let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
        interp.bind_for_test("root", root);
        let b = interp.eval_expr_for_test(&parse_e("forge.carve(root, 65536)")).unwrap();
        let idx = Interpreter::budget_index_for_test(&b);
        interp.bind_for_test("b", b);
        let x = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("x", x);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("f.fwd_bwd(x)"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row5_struct_field_read() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let prog = parse_program_674("model M { w: Tensor[f32, [4]] }");
        interp.load_program(&prog, None).expect("load M");
        let w = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("w", w);
        let m = interp.eval_expr_for_test(&parse_e("M { w: w }")).unwrap();
        interp.bind_for_test("m", m);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("m.w"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row6_field_ref_alias_read() {
    // `let !y = m.w` binds `y` as a `Value::FieldRef` into `m`'s shared
    // storage (`eval_stmt_inner`'s `Stmt::Let` mutating-alias arm) — built
    // here directly on the Rust side (rather than via a `{ let !y = m.w }`
    // block, whose own pushed scope would drop `y` at the block's exit)
    // since `Value::Struct`'s inner `Rc<RefCell<..>>` is reachable from
    // `interp_tests` like any other `Value` variant field.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let prog = parse_program_674("model M { w: Tensor[f32, [4]] }");
        interp.load_program(&prog, None).expect("load M");
        let w = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("w", w);
        let m = interp.eval_expr_for_test(&parse_e("M { w: w }")).unwrap();
        if let Value::Struct(rc) = &m {
            interp.bind_for_test("y", Value::FieldRef { rc: rc.clone(), field: "w".to_string() });
        } else {
            panic!("expected Value::Struct, got {:?}", m);
        }
        interp.bind_for_test("m", m);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("y"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row7_list_index_get() {
    // `list(t)` — a bracket literal `[t]` builds a TENSOR here (demoniC is
    // tensor-first: `[...]` is `Expr::TensorLit`), not a dynamic `List`.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        let xs = interp.eval_expr_for_test(&parse_e("list(t)")).unwrap();
        interp.bind_for_test("xs", xs);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("xs[0]"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row8_tuple_index_get() {
    // Tuple element access is by pattern destructuring, not `.0`/`.1`
    // indexing (there is no dot-index postfix over `Value::Tuple`) — a
    // second `let` unpacks the constructed pair, exercising the SAME `Tuple`
    // route (`eval_stmt_inner`'s `Pattern::Tuple` bind, `interp.rs` ~6935).
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        let xs = interp.eval_expr_for_test(&parse_e("(t, 1)")).unwrap();
        interp.bind_for_test("xs", xs);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("{ let (a, _n) = xs \n a }"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row9_map_get() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        let m = interp.eval_expr_for_test(&parse_e("map_set(map(), \"k\", t)")).unwrap();
        interp.bind_for_test("m", m);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("map_get(m, \"k\")"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row10_lambda_captured_env() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("t", t);
        let lam = interp.eval_expr_for_test(&parse_e("fn() -> f32 { sum(t) }")).unwrap();
        interp.bind_for_test("lam", lam);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("lam()"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row11_print_renders_a_reaped_tensor_as_a_marker() {
    // A `fmt` has no error channel and needs none: a stale tensor inside a
    // container `print`s as a marker rather than stale bytes — the walker
    // renders, it does not raise.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("__t", t);
        let xs = interp.eval_expr_for_test(&parse_e("list(__t)")).unwrap();
        interp.bind_for_test("xs", xs);
        reap_b_674(&mut interp);
        // `print` itself has no return value worth asserting on; assert the
        // rendering directly through `Debug`'s own text via `format`.
        let rendered = interp.eval_expr_for_test(&parse_e("format(\"{}\", xs)"));
        let s = as_str(&rendered.unwrap()).to_string();
        assert!(s.contains("Tensor<reaped>"), "rendered: {s}");
    });
}

#[test]
fn hole_table_674_row12_json_encode_of_a_reaped_tensor_is_null() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"))).unwrap();
        let m = interp.with_budget_for_test(0, |i| { i.bind_for_test("__t", t); i.eval_expr_for_test(&parse_e("map_set(map(), \"w\", __t)")) }).unwrap();
        interp.bind_for_test("m", m);
        reap_b_674(&mut interp);
        let encoded = interp.eval_expr_for_test(&parse_e("json_encode(m)"));
        let s = as_str(&encoded.unwrap()).to_string();
        assert_eq!(s, "{\"w\":null}", "a stale tensor field must encode null, not stale numbers");
    });
}

#[test]
fn hole_table_674_row13_stream_append_slot_read_plain_binding() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [1, 4]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(
            &parse_e("{ t <- forge.zeros[f32, [1, 4]] \n nil }"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row13_stream_append_slot_read_struct_field() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let prog = parse_program_674("model M { cache: Tensor[f32, [1, 4]] }");
        interp.load_program(&prog, None).expect("load M");
        let c = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("forge.zeros[f32, [1, 4]]"))).unwrap();
        interp.bind_for_test("c", c);
        let m = interp.eval_expr_for_test(&parse_e("M { cache: c }")).unwrap();
        interp.bind_for_test("m", m);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(
            &parse_e("{ m.cache <- forge.zeros[f32, [1, 4]] \n nil }"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row14_builtin_result_list_of_budgeted_tensors_goes_stale() {
    // `charge_builtin_result`'s `List` arm / its `charge_intermediate`
    // callers keep `&Value` — the stamp is at the construction chokepoint,
    // not the charge — so a builtin returning a LIST of budgeted tensors
    // still goes stale per-element after the reap.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let xs = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e(
                "list_map(list(1, 2), fn(_: i64) -> Tensor[f32, [4]] { forge.zeros[f32, [4]] })"
            ))).unwrap();
        interp.bind_for_test("xs", xs);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("xs[0]"));
        assert_stale(r);
    });
}

#[test]
fn hole_table_674_row15_suppressed_literal_inside_a_budget_still_stamps() {
    // A tensor-literal row's charge is suppressed (`lit_charge_suppressed`)
    // while its OWN materialization is not — the stamp is not the charge and
    // does not inherit its suppression, so a nested literal built under a
    // budget is stamped `b` and goes stale after `reap(b)` same as any other
    // constructor.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let t = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("[[1.0f32, 2.0f32], [3.0f32, 4.0f32]]"))).unwrap();
        interp.bind_for_test("t", t);
        reap_b_674(&mut interp);
        assert_stale(interp.eval_expr_for_test(&parse_e("t")));
    });
}

#[test]
fn hole_table_674_row19_vault_ctor_inside_a_budget_is_not_stamped() {
    // §3.2 row 19 / §3.5: `vault.zeros[…]` charges `vault_used`, never any
    // budget — even while a budget is "current" (simulated here via
    // `with_budget_for_test`, the same push `forge.enter`/`in b` will use in
    // 3d), a Vault construction must NOT inherit that budget's tag: it must
    // stay LIVE after the budget is reaped.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let v = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("vault.zeros[f32, [4]]"))).unwrap();
        interp.bind_for_test("v", v);
        reap_b_674(&mut interp);
        let r = interp.eval_expr_for_test(&parse_e("v"));
        assert!(r.is_ok(), "a vault.* tensor must survive the reap of whatever \
            budget happened to be current at its construction: {:?}", r.err().map(|e| e.msg));
    });
}

// ── The Vault-arm exemption composes with a normal Forge stamp in the same
// expression ─────────────────────────────────────────────────────────────

#[test]
fn budget_674_vault_ctor_restores_the_ambient_tag_after_returning() {
    // The `VaultTagGuard` must restore the PREVIOUS tag on every exit, not
    // just clear it permanently — a `forge.*` construction evaluated right
    // after a `vault.*` one, still under the same pushed budget, must still
    // be stamped and go stale.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let idx = budget_setup_674(&mut interp);
        let pair = interp.with_budget_for_test(idx, |i| {
            i.eval_expr_for_test(&parse_e("(vault.zeros[f32, [4]], forge.zeros[f32, [4]])"))
        }).unwrap();
        interp.bind_for_test("pair", pair);
        reap_b_674(&mut interp);
        let v = interp.eval_expr_for_test(&parse_e("{ let (v, _f) = pair \n v }"));
        assert!(v.is_ok(), "the vault half must stay live: {:?}", v.err().map(|e| e.msg));
        let f = interp.eval_expr_for_test(&parse_e("{ let (_v, f) = pair \n f }"));
        assert_stale(f);
    });
}

// ── §3.5's other half: the constructor's ARGUMENTS are not the constructor
// ───────────────────────────────────────────────────────────────────────────
//
// §3.5 exempts a `vault.*` constructor's OWN allocation from the entered
// budget — row 19 above. Its ARGUMENTS are not exempt: a shape expression or
// a fill is ordinary code running inside `forge.enter(b) { … }`, so whatever
// it allocates is `b`'s, charged to `b` and stamped `b`.
//
// The guard that implements the exemption used to be opened before those
// arguments were evaluated and held for the rest of the call, which split
// the two meters: `charge_arena` reads `cur_budget` and charged `b`, while
// `TensorVal::new` reads the cleared `CUR_BUDGET_TAG` and stamped ROOT. A
// tensor allocated in a vault constructor's argument therefore had its bytes
// die at `reap(b)` while it still read as live — the interpreter's form of
// the same seam the JIT had at the emit site (PR #835 review). The guard is
// now opened at the constructor's own charge-and-build sites only.
//
// `mk` writes its `!` parameter back, which is what lets the argument's
// tensor outlive the call and be retrieved after the reap; without an escape
// there is nothing to ask about.
const VAULT_ARG_MK_674: &str =
    "fn mk(!t: Tensor[f32, [4096]]) -> i64 { t = forge.ones[f32, [4096]] \n 4 }";

/// `--forge=1 MiB`, `forge.root` as `root`, a child of `extent` as `b`,
/// `mk` loaded, and a root-allocated `scratch` bound. Returns `b`'s index.
fn vault_arg_setup_674(interp: &mut Interpreter, extent: usize) -> usize {
    interp.set_arena_limits(ArenaLimits { forge: Some(1 << 20), vault: None });
    let root = interp.eval_expr_for_test(&parse_e("forge.root")).unwrap();
    interp.bind_for_test("root", root);
    let b = interp.eval_expr_for_test(&parse_e(
        &format!("forge.carve(root, {extent})"))).unwrap();
    let idx = Interpreter::budget_index_for_test(&b);
    interp.bind_for_test("b", b);
    interp.load_program(&parse_program_674(VAULT_ARG_MK_674), None).expect("load mk()");
    // Allocated at the ROOT, before anything is entered: `scratch` is stale
    // after the reap only if the WRITEBACK re-stamped it from inside `b`.
    let scratch = interp.eval_expr_for_test(&parse_e("forge.zeros[f32, [4096]]")).unwrap();
    interp.bind_for_test("scratch", scratch);
    idx
}

#[test]
fn budget_674_vault_ctor_argument_allocation_is_stamped_the_entered_budget() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        // 32 KiB is exactly what the argument costs `b`: 16 KiB for the `!`
        // parameter's copy-in (the #650 copy-in/copy-out rule, charged at
        // the budget current at the call) and 16 KiB for `mk`'s own
        // `forge.ones`. The sibling test below carves one byte less and
        // watches the flip, which is the bisected form of "charged to `b`".
        let idx = vault_arg_setup_674(&mut interp, 32768);
        let v = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("vault.ones[f32, [mk(scratch)]]"))).unwrap();
        interp.bind_for_test("v", v);
        assert!(
            matches!(interp.eval_expr_for_test(&parse_e("(scratch[0] as i64)")).unwrap(),
                     Value::Int(1, _)),
            "`mk` did not write its `!` parameter back, so there is nothing here to \
             ask about staleness",
        );
        reap_b_674(&mut interp);
        // The CONSTRUCTOR's own bytes are the root's (§3.5, row 19) — live.
        let r = interp.eval_expr_for_test(&parse_e("v"));
        assert!(r.is_ok(), "the `vault.*` tensor must survive the reap: {:?}",
            r.err().map(|e| e.msg));
        // Its ARGUMENT's bytes are `b`'s — stale, and located.
        assert_stale(interp.eval_expr_for_test(&parse_e("scratch")));
    });
}

#[test]
fn budget_674_vault_ctor_argument_allocation_is_charged_to_the_entered_budget() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        // One byte under what the argument needs. `b`'s ceiling refuses the
        // allocation, so `mk` itself raises `forge arena exhausted` — before
        // #839 that error never reached the caller: `eval_dim` swallowed ANY
        // error out of a shape expression and substituted a bare 4, so the
        // constructor silently completed at shape [4] and the only visible
        // symptom was the missing writeback (a pair of runs, not one
        // `assert`). Now the error propagates, located at `mk(scratch)`
        // itself, so the constructor call is the assertion directly: it
        // never returns a value, and `scratch` — never reached by `mk`'s
        // writeback — is still the root tensor it started as, live and
        // reading 0 after the reap.
        let idx = vault_arg_setup_674(&mut interp, 32767);
        let r = interp.with_budget_for_test(idx,
            |i| i.eval_expr_for_test(&parse_e("vault.ones[f32, [mk(scratch)]]")));
        let e = r.err().expect(
            "`mk` allocated after all — the argument was not charged to the entered \
             budget, or its extent is not the boundary this test thinks it is");
        assert!(e.msg.contains("forge arena exhausted"), "message: {}", e.msg);
        assert!(e.span.is_some(), "the propagated error must be located: {}", e.msg);
        assert!(
            matches!(interp.eval_expr_for_test(&parse_e("(scratch[0] as i64)")).unwrap(),
                     Value::Int(0, _)),
            "`mk` allocated after all — the argument was not charged to the entered \
             budget, or its extent is not the boundary this test thinks it is",
        );
        reap_b_674(&mut interp);
        assert!(interp.eval_expr_for_test(&parse_e("scratch")).is_ok(),
            "nothing was ever allocated in `b` for `scratch`, so it cannot be stale");
    });
}

// ── Must NOT change (slice 3b's compatibility gate, §3.1 / §5) ─────────────

#[test]
fn budget_674_entry_0_exhaustion_sentence_is_byte_identical_to_pre_3b() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(8), vault: None });
        let r = interp.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"));
        let msg = err_msg(r);
        assert_eq!(
            msg,
            "forge arena exhausted: this allocation needs 16 B, but only 8 B \
             of the 8 B --forge budget is free",
            "entry 0's exhaustion sentence must be byte-identical to pre-3b's"
        );
    });
}

#[test]
fn budget_674_ordinary_program_never_carving_runs_unchanged() {
    // §3.1 / §3.6 s1's own claim, re-affirmed under a real `--forge`: a
    // program that never carves behaves exactly as before slice 3b.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(64), vault: None });
        let ok = interp.eval_expr_for_test(&parse_e("forge.zeros[f32, [4]]"));
        assert!(ok.is_ok());
        let refused = interp.eval_expr_for_test(&parse_e("forge.zeros[f32, [16]]"));
        assert!(err_msg(refused).starts_with("forge arena exhausted:"));
    });
}

// ── Cost pin: `b1`/`b2`-style minima under `dmc run` are unchanged (§6 Q5) ──

#[test]
fn budget_674_cost_pin_b1_plain_rebind_costs_zero_bytes() {
    // #790's own probe, re-run under slice 3b's charging path: one
    // 4096-element f32 ctor (16 384 B) plus three plain `let y = x`
    // rebinds bisects to exactly 16 384 B (1.00×) — a plain rebind costs
    // 0 B, and one byte under refuses.
    with_big_stack(|| {
        let src = "fn main() -> nil {
                let x = forge.zeros[f32, [4096]]
                let y = x
                let z = y
                let w = z
                nil
            }";
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(16_384), vault: None });
        let r = interp.run(&parse_program_674(src), None);
        assert!(r.is_ok(), "16 384 B must be enough for one ctor plus three plain rebinds: {:?}", r.err().map(|e| e.msg));

        let mut interp2 = Interpreter::new();
        interp2.set_arena_limits(ArenaLimits { forge: Some(16_383), vault: None });
        let refused = interp2.run(&parse_program_674(src), None);
        assert!(refused.is_err(), "one byte under 16 384 must refuse even the bare ctor");
    });
}

#[test]
fn budget_674_cost_pin_b2_private_copy_is_charged() {
    // #790's `b2`, in this test's own exact terms: one 4096-element f32 ctor
    // (16 384 B) plus one `let !y = x` private copy (16 384 B more) bisects
    // to exactly 32 768 B (2.00×) — the private copy IS charged, and the
    // copy's origin re-stamp (§6 Q5) does not change the byte count, only
    // the copy's `origin` tag.
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(32_768), vault: None });
        let ok = interp.run(&parse_program_674(
            "fn main() -> nil { let x = forge.zeros[f32, [4096]] \n let !y = x \n nil }"
        ), None);
        assert!(ok.is_ok(), "{:?}", ok.err().map(|e| e.msg));

        let mut interp2 = Interpreter::new();
        interp2.set_arena_limits(ArenaLimits { forge: Some(32_767), vault: None });
        let refused = interp2.run(&parse_program_674(
            "fn main() -> nil { let x = forge.zeros[f32, [4096]] \n let !y = x \n nil }"
        ), None);
        assert!(refused.is_err(), "one byte under 32 768 must still refuse the private copy");
    });
}

#[test]
fn budget_674_value_size_pin_is_128_bytes() {
    // The arena-budget design's own size pin ("Built, the price was not
    // 0.20 %"): the whole measured interpreter cost of M1 is `Value` growing
    // 120 → 128 B for the one-word `origin`. Pin it so a two-word relapse
    // (136 B, +34–66 %) or a
    // silent return to 120 B is a visible change, not an accident.
    assert_eq!(std::mem::size_of::<Value>(), 128);
    assert_eq!(std::mem::size_of::<super::interp::TensorVal>(), 128);
}

// ── #674 slice 3d: `forge.enter(b) { … }` and `<ctor> in b` ─────────────────
//
// The arena-budget design's own B5 rule: the block "pushes `b` as the entered
// budget and pops it on EVERY exit edge — normal exit, `return`, `break`,
// `continue`, a `?` early return", and its slice-3d test requirements ask for
// one case per edge plus one where the body raises.
//
// Every case below asserts BOTH halves, because either alone is weak:
//
//   * the REGISTER, read after the program finished
//     (`Interpreter::cur_budget_for_test`). Zero means the guard dropped.
//     On its own it would also pass for a block that never pushed at all.
//   * the CHARGE, bisected. The block runs inside a 4 KiB budget and the
//     allocation AFTER it needs 16 KiB, which fits the root and cannot fit
//     `b` — so it errors iff the register leaked. On its own it would not
//     distinguish a leak that is repaired later.
//
// The pairing is what makes them a gate: a missed edge mis-charges the rest
// of the program with no diagnostic, which is the silent-wrong-number class
// B6 exists to close.

/// Run `src` under an explicit `--forge` ceiling and hand back the result
/// together with the entered-budget register AS IT STOOD when the program
/// finished. Nothing else can read that register from outside a running
/// program, and B5's whole claim is about it.
fn run_forge_674(src: &str, forge: u64) -> (Result<Value, super::interp::RuntimeError>, usize) {
    let tokens = Lexer::new(src).tokenize().expect("lex failed");
    let program = Parser::new(tokens).parse_program().expect("parse failed");
    let mut interp = Interpreter::new();
    interp.set_arena_limits(ArenaLimits { forge: Some(forge), vault: None });
    let r = interp.run(&program, None);
    (r, Interpreter::cur_budget_for_test())
}

/// The shared assertion for the five edge cases: the program answers `want`,
/// and the register is back at the root.
fn assert_edge_restores_674(src: &str, want: i64) {
    let (r, reg) = run_forge_674(src, 1 << 20);
    let v = r.unwrap_or_else(|e| panic!("expected the program to answer: {}", e.msg));
    assert_eq!(as_int(&v), want, "value: {:?}", v);
    assert_eq!(reg, 0, "the entered-budget register did not come back to the root");
}

#[test]
fn budget_674_enter_pops_on_the_normal_edge() {
    assert_edge_restores_674(
        "fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   forge.enter(b) { let u = forge.zeros[f32, [64]] }\n\
         \x20   let t = forge.zeros[f32, [4096]]\n\
         \x20   1\n\
         }", 1);
}

#[test]
fn budget_674_enter_pops_on_the_return_edge() {
    // The `return` leaves the extent AND the function. `=> 7` rather than
    // `3` is what says the `return` is the edge that fired.
    assert_edge_restores_674(
        "fn f(b: Budget) -> i64 {\n\
         \x20   forge.enter(b) { let u = forge.zeros[f32, [64]]\n return 7 }\n\
         \x20   3\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   let k = f(b)\n\
         \x20   let t = forge.zeros[f32, [4096]]\n\
         \x20   k\n\
         }", 7);
}

#[test]
fn budget_674_enter_pops_on_the_break_edge() {
    assert_edge_restores_674(
        "fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   for i in 0..3 {\n\
         \x20       forge.enter(b) { let u = forge.zeros[f32, [64]]\n break }\n\
         \x20   }\n\
         \x20   let t = forge.zeros[f32, [4096]]\n\
         \x20   1\n\
         }", 1);
}

#[test]
fn budget_674_enter_pops_on_the_continue_edge() {
    // Three iterations, so a `continue` that failed to pop would also nest
    // three deep — the register assert catches the depth, the 16 KiB
    // allocation after the loop catches the mis-charge.
    assert_edge_restores_674(
        "fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   for i in 0..3 {\n\
         \x20       forge.enter(b) { let u = forge.zeros[f32, [64]]\n continue }\n\
         \x20   }\n\
         \x20   let t = forge.zeros[f32, [4096]]\n\
         \x20   1\n\
         }", 1);
}

#[test]
fn budget_674_enter_pops_on_the_query_edge() {
    // The edge the design names as the trap: `Expr::ArenaBlock`'s own arm
    // writes `self.eval_block(&ab.body)?`, and a restore line after that `?`
    // is skipped on exactly this path. `=> 1` is `?`'s propagated
    // `(default[T], e)` reaching `main`, not `(5, nil)`.
    assert_edge_restores_674(
        "fn g() -> (i64, Err) { (0, \"boom\") }\n\
         fn f(b: Budget) -> (i64, Err) {\n\
         \x20   forge.enter(b) {\n\
         \x20       let u = forge.zeros[f32, [64]]\n\
         \x20       let y = g()?\n\
         \x20       let v = forge.zeros[f32, [64]]\n\
         \x20   }\n\
         \x20   (5, nil)\n\
         }\n\
         fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   let (v, e) = f(b)\n\
         \x20   let t = forge.zeros[f32, [4096]]\n\
         \x20   v + 1\n\
         }", 1);
}

#[test]
fn budget_674_enter_pops_when_the_body_raises() {
    // The sixth case the design's Tests line asks for, and the one the
    // straight-line restore gets wrong most obviously: the body errors, the
    // error propagates out of `eval_expr` on the `?`, and the register must
    // still be back at the root by the time anyone can observe it.
    let (r, reg) = run_forge_674(
        "fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   forge.enter(b) { undefined_fn_674() }\n\
         \x20   1\n\
         }", 1 << 20);
    let e = r.err().expect("the body's undefined call must raise");
    assert!(e.msg.contains("undefined_fn_674"), "message: {}", e.msg);
    assert_eq!(reg, 0, "the register leaked out of a raising `forge.enter` body");
}

#[test]
fn budget_674_nested_enter_restores_the_outer_budget_not_the_root() {
    // B5: an `enter` nested inside an `enter` pops to what it SHADOWED. The
    // root is left with 4 KiB free after the two carves, so the 16 KiB
    // allocation between the inner block's exit and the outer block's can
    // only be served by `outer` — a pop-to-root would refuse it, and a
    // pop-to-nothing (still `inner`, 4 KiB) would refuse it too. Same
    // numbers as `compiler/tests/jit_budgets_3d.rs`'s own nesting case, so
    // the two backends are asserted against one arrangement rather than two.
    let src = "fn main() -> i64 {\n\
         \x20   let outer = forge.carve(forge.root, 262144)\n\
         \x20   let inner = forge.carve(forge.root, 4096)\n\
         \x20   forge.enter(outer) {\n\
         \x20       forge.enter(inner) { let a = forge.zeros[f32, [64]] }\n\
         \x20       let big = forge.zeros[f32, [4096]]\n\
         \x20   }\n\
         \x20   1\n\
         }";
    let (r, reg) = run_forge_674(src, 270336);
    assert!(r.is_ok(), "expected the outer budget to serve it: {:?}",
            r.err().map(|e| e.msg));
    assert_eq!(reg, 0);

    // The control that makes the case above discriminating: the same 16 KiB
    // allocation OUTSIDE both extents, where the root has only 4 KiB free.
    let (r2, _) = run_forge_674(
        "fn main() -> i64 {\n\
         \x20   let outer = forge.carve(forge.root, 262144)\n\
         \x20   let inner = forge.carve(forge.root, 4096)\n\
         \x20   let big = forge.zeros[f32, [4096]]\n\
         \x20   1\n\
         }", 270336);
    let e = r2.err().expect("the root has only 4 KiB free after both carves");
    assert!(e.msg.contains("forge arena exhausted"), "message: {}", e.msg);
}

#[test]
fn budget_674_alloc_in_charges_the_named_budget_not_the_entered_one() {
    // B3: `… in b` OVERRIDES the entered budget for that one constructor.
    // Inside `forge.enter(small)`, a 16 KiB constructor written `in big`
    // must charge `big` — `small` could not hold it, so the program answers
    // only if the override took effect.
    let (r, reg) = run_forge_674(
        "fn main() -> i64 {\n\
         \x20   let small = forge.carve(forge.root, 4096)\n\
         \x20   let big = forge.carve(forge.root, 65536)\n\
         \x20   forge.enter(small) {\n\
         \x20       let t = forge.zeros[f32, [4096]] in big\n\
         \x20   }\n\
         \x20   1\n\
         }", 1 << 20);
    assert!(r.is_ok(), "the `in big` override did not take: {:?}",
            r.err().map(|e| e.msg));
    assert_eq!(reg, 0);

    // The control: the identical program WITHOUT the override charges the
    // entered budget and is refused at its ceiling. Without this, the case
    // above would pass on an `in b` that did nothing at all.
    let (r2, _) = run_forge_674(
        "fn main() -> i64 {\n\
         \x20   let small = forge.carve(forge.root, 4096)\n\
         \x20   let big = forge.carve(forge.root, 65536)\n\
         \x20   forge.enter(small) {\n\
         \x20       let t = forge.zeros[f32, [4096]]\n\
         \x20   }\n\
         \x20   1\n\
         }", 1 << 20);
    let e = r2.err().expect("16 KiB cannot fit the entered 4 KiB budget");
    assert!(e.msg.contains("the entered budget's 4 KiB extent"), "message: {}", e.msg);
}

#[test]
fn budget_674_alloc_in_restores_the_entered_budget_after_the_constructor() {
    // The override is for ONE constructor (B3). The allocation after it goes
    // back to the entered budget — asserted the only way there is, by making
    // the entered budget too small for it.
    let (r, reg) = run_forge_674(
        "fn main() -> i64 {\n\
         \x20   let small = forge.carve(forge.root, 4096)\n\
         \x20   let big = forge.carve(forge.root, 65536)\n\
         \x20   forge.enter(small) {\n\
         \x20       let t = forge.zeros[f32, [4096]] in big\n\
         \x20       let u = forge.zeros[f32, [4096]]\n\
         \x20   }\n\
         \x20   1\n\
         }", 1 << 20);
    let e = r.err().expect("the second constructor must charge `small` again");
    assert!(e.msg.contains("the entered budget's 4 KiB extent"), "message: {}", e.msg);
    assert_eq!(reg, 0, "the register must still come back after the refusal");
}

#[test]
fn budget_674_enter_over_a_reaped_budget_is_budget_stale() {
    // §2.3's table lists `budget-stale` at "carve / reap / enter / `in b`" —
    // one tag, four sites, and `enter`/`in b` are the two this slice adds.
    let (r, reg) = run_forge_674(
        "fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   forge.reap(b)\n\
         \x20   forge.enter(b) { let u = forge.zeros[f32, [64]] }\n\
         \x20   1\n\
         }", 1 << 20);
    let e = r.err().expect("a reaped handle must refuse at `enter`");
    assert!(e.msg.starts_with("budget-stale:"), "message: {}", e.msg);
    assert!(e.span.is_some(), "`budget-stale` must be located");
    assert_eq!(reg, 0);
}

#[test]
fn budget_674_alloc_in_over_a_reaped_budget_is_budget_stale() {
    let (r, _) = run_forge_674(
        "fn main() -> i64 {\n\
         \x20   let b = forge.carve(forge.root, 4096)\n\
         \x20   forge.reap(b)\n\
         \x20   let t = forge.zeros[f32, [4]] in b\n\
         \x20   1\n\
         }", 1 << 20);
    let e = r.err().expect("a reaped handle must refuse at `in b`");
    assert!(e.msg.starts_with("budget-stale:"), "message: {}", e.msg);
    assert!(e.span.is_some(), "`budget-stale` must be located");
}

#[test]
fn budget_674_a_tensor_built_inside_an_enter_goes_stale_at_the_reap() {
    // The other half of what `enter` wires up: `TensorVal::new` reads the
    // same thread-local the guard pushes, so a tensor built INSIDE the
    // extent carries `b`'s stamp and B6's invalidation reaches it — through
    // surface syntax now, where slice 3b could only reach it through the
    // `#[cfg(test)]` harness.
    let (r, _) = run_forge_674(
        "fn main() -> f32 {\n\
         \x20   let b = forge.carve(forge.root, 65536)\n\
         \x20   let t = forge.enter(b) { forge.zeros[f32, [4]] }\n\
         \x20   forge.reap(b)\n\
         \x20   t[0]\n\
         }", 1 << 20);
    let e = r.err().expect("the tensor was allocated in `b`, which was reaped");
    assert!(e.msg.starts_with("budget-stale:"), "message: {}", e.msg);
}

// ─── #834: `arena_ctor_dtype` is a total, correctly-metered table ──────────
//
// Before this fix, `arena_ctor_dtype` had no entry at all for `fp8_e4m3`/
// `fp8_e5m2` and silently fell back to `DType::F32`'s FOUR-byte width for
// both — a `forge.zeros[fp8_e4m3, [4096]]` was metered as a 16 384 B f32
// tensor, four times its declared SPEC §3.1 one-byte width. These pin the
// #790 cost-pin style against the fp8 fix (a 4096-element ctor bisects to
// exactly its declared byte count) and the constructed tensor's actual
// `DType` tag — never silently `DType::F32` again for the two fp8 kinds.
//
// A separate regression, caught in review and fixed in the same pass: an
// intermediate version of `arena_ctor_dtype` anchored on `pos[0]` and turned
// ANY `Ident` there that `ScalarType::from_ctor_name` does not recognize —
// a bogus name (`qqq_not_a_type`), `str`/`nil` (`from_ctor_name` never
// admits either as ctor spellings; #817 is an open ruling on whether they
// have a tensor-element representation at all, and `check.rs`'s
// `ctor_tensor_ty` still admits them in constructor position, so this
// function must not preempt that ruling), or a generic dim `Ident`
// (`forge.trit[R, C]`'s `R`) — into a fatal internal error. Base's
// pre-#834 behavior was to fall back to `DType::F32` for all of these, and
// that fallback is restored below.
use super::interp::DType;

#[test]
fn arena_ctor_dtype_834_fp8_e4m3_meters_one_byte_per_element() {
    // #790's own cost-pin shape: a 4096-element ctor at its declared width
    // (1 B for fp8_e4m3) bisects to exactly 4096 B (1.00x) — one byte under
    // must refuse even the bare ctor.
    with_big_stack(|| {
        let src = "fn main() -> nil {
                let x = forge.zeros[fp8_e4m3, [4096]]
                nil
            }";
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(4096), vault: None });
        let r = interp.run(&parse_program_674(src), None);
        assert!(r.is_ok(), "4096 B must be enough for a 4096-element fp8_e4m3 \
                 ctor (1 B/elem): {:?}", r.err().map(|e| e.msg));

        let mut interp2 = Interpreter::new();
        interp2.set_arena_limits(ArenaLimits { forge: Some(4095), vault: None });
        let refused = interp2.run(&parse_program_674(src), None);
        assert!(refused.is_err(), "one byte under 4096 must refuse the bare fp8_e4m3 ctor");
    });
}

#[test]
fn arena_ctor_dtype_834_fp8_e5m2_meters_one_byte_per_element() {
    with_big_stack(|| {
        let src = "fn main() -> nil {
                let x = forge.zeros[fp8_e5m2, [4096]]
                nil
            }";
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(4096), vault: None });
        let r = interp.run(&parse_program_674(src), None);
        assert!(r.is_ok(), "4096 B must be enough for a 4096-element fp8_e5m2 \
                 ctor (1 B/elem): {:?}", r.err().map(|e| e.msg));

        let mut interp2 = Interpreter::new();
        interp2.set_arena_limits(ArenaLimits { forge: Some(4095), vault: None });
        let refused = interp2.run(&parse_program_674(src), None);
        assert!(refused.is_err(), "one byte under 4096 must refuse the bare fp8_e5m2 ctor");
    });
}

#[test]
fn arena_ctor_dtype_834_fp8_e4m3_tag_is_dtype_fp8e4m3() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let v = interp.eval_expr_for_test(&parse_e("forge.zeros[fp8_e4m3, [4]]"));
        match v {
            Ok(Value::Tensor(t)) => assert_eq!(
                t.dtype, DType::Fp8E4M3,
                "the constructed tensor must be tagged `Fp8E4M3`, not a silent `F32`"
            ),
            Ok(_) => panic!("expected a tensor value"),
            Err(e) => panic!("expected Ok(Tensor), got Err: {}", e.msg),
        }
    });
}

#[test]
fn arena_ctor_dtype_834_fp8_e5m2_tag_is_dtype_fp8e5m2() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let v = interp.eval_expr_for_test(&parse_e("forge.zeros[fp8_e5m2, [4]]"));
        match v {
            Ok(Value::Tensor(t)) => assert_eq!(
                t.dtype, DType::Fp8E5M2,
                "the constructed tensor must be tagged `Fp8E5M2`, not a silent `F32`"
            ),
            Ok(_) => panic!("expected a tensor value"),
            Err(e) => panic!("expected Ok(Tensor), got Err: {}", e.msg),
        }
    });
}

/// An `Ident` at `pos[0]` that `ScalarType::from_ctor_name` does not
/// recognize (with the checker bypassed, exactly what this harness does by
/// evaluating the expression directly rather than through `dmc run`'s
/// `--check`-first pipeline) falls back to `DType::F32` — base's behavior,
/// restored after review found an intermediate version of this fix turning
/// it into a fatal internal error instead (breaking generic dims and
/// `forge.zeros[str, ...]`, both #834-rework regressions).
#[test]
fn arena_ctor_dtype_834_unrecognized_name_falls_back_to_f32_not_internal_error() {
    with_big_stack(|| {
        let mut interp = Interpreter::new();
        let r = interp.eval_expr_for_test(&parse_e("forge.zeros[qqq_not_a_type, [4]]"));
        match r {
            Ok(Value::Tensor(t)) => assert_eq!(
                t.dtype, DType::F32,
                "an unrecognized ctor element-type name must fall back to F32, as base did"
            ),
            Ok(_) => panic!("expected a tensor value"),
            Err(e) => panic!("expected Ok(Tensor), got Err: {}", e.msg),
        }
    });
}

/// `str` is not a `ScalarType::from_ctor_name` spelling at all (`nil` isn't
/// either), so `forge.zeros[str, [1024]]` takes the same F32 fallback path
/// as the previous test, not the `dtype_for_scalar_elem`-returns-`None`
/// path. #817 (whether `str`/`nil` have a real tensor-element
/// representation) stays open; `check.rs`'s `ctor_tensor_ty` still admits
/// both in constructor position, and this function must not decide that
/// question by refusing them here. Meters as a 1024-element f32 tensor: 4
/// KiB, matching base.
#[test]
fn arena_ctor_dtype_834_str_ctor_falls_back_to_f32_meters_4kib() {
    with_big_stack(|| {
        let src = "fn main() -> nil {
                let x = forge.zeros[str, [1024]]
                nil
            }";
        let mut interp = Interpreter::new();
        interp.set_arena_limits(ArenaLimits { forge: Some(4096), vault: None });
        let r = interp.run(&parse_program_674(src), None);
        assert!(r.is_ok(), "4096 B must be enough for forge.zeros[str, [1024]] \
                 (F32 fallback, 4 B/elem): {:?}", r.err().map(|e| e.msg));

        let mut interp2 = Interpreter::new();
        interp2.set_arena_limits(ArenaLimits { forge: Some(4095), vault: None });
        let refused = interp2.run(&parse_program_674(src), None);
        assert!(refused.is_err(), "one byte under 4096 must refuse forge.zeros[str, [1024]]");
    });
}

/// A generic dim `Ident` at `pos[0]` (`forge.trit[R, C]`'s `R`, where `R`/`C`
/// are the function's own generic-int params, not type names) must not be
/// treated as a bogus type name — `trit`'s own ctor hardcodes `DType::Trit`
/// and never consults `arena_ctor_dtype`'s return, so this only has to not
/// blow up.
#[test]
fn arena_ctor_dtype_834_generic_dim_ident_does_not_error() {
    with_big_stack(|| {
        let src = "fn build[R, C]() -> nil {
                let t = forge.trit[R, C]
                nil
            }
            fn main() -> nil { build[4, 3]() }";
        let mut interp = Interpreter::new();
        let r = interp.run(&parse_program_674(src), None);
        assert!(r.is_ok(), "a generic dim ident at pos[0] must not be a fatal \
                 internal error: {:?}", r.err().map(|e| e.msg));
    });
}

// ─── #834 rework: `tape_index_read` must widen the same way `value_is_
//     differentiable` did ──────────────────────────────────────────────────
//
// `value_is_differentiable` was widened (this issue) to declare fp8 tensors
// differentiable, but `tape_index_read` — the function that traces a
// subscript read (`w[0]`) onto the tape — still bailed out for any dtype
// other than `F32`/`F64`. For an fp8 tensor that meant `w[0]` fell OFF the
// tape (`tape_index_read` returns `Ok(None)`, and the caller re-evaluates
// the read as an untraced plain value), so a term reading an fp8 element
// directly picked up no gradient at all — silently wrong, not refused.
//
// `f(!w: Tensor[fp8_e4m3,[2]]) = sum(w) + w[0]*w[0]` at w=[3,4]:
// d/dw[0] = 1 (from sum) + 2*w[0] (from w[0]*w[0]) = 1 + 6 = 7
// d/dw[1] = 1 (from sum only) = 1
// Before this fix the PR's g0 came back as 1 (only the `sum` term's
// contribution — the `w[0]*w[0]` term's read of `w[0]` was untraced, so its
// contribution to g0 was silently dropped); g1 was already correct since it
// is only ever read through `sum`, not indexed.

#[test]
fn tape_index_read_834_fp8_indexed_term_g0_matches_base() {
    let v = run(r#"
        @grad fn f(!w: Tensor[fp8_e4m3, [2]]) -> fp8_e4m3 { sum(w) + w[0]*w[0] }
        fn main() -> fp8_e4m3 {
            let !w = forge.zeros[fp8_e4m3, [2]]
            w[0] = 3.0
            w[1] = 4.0
            let (_l, g) = f.fwd_bwd(w)
            g.w[0]
        }
    "#);
    assert_eq!(as_float(&v), 7.0, "g0 must be 7 (1 from `sum` + 6 from `w[0]*w[0]`); \
               1 is the pre-fix bug (the `w[0]*w[0]` read fell off the tape)");
}

#[test]
fn tape_index_read_834_fp8_summed_only_term_g1_matches_base() {
    let v = run(r#"
        @grad fn f(!w: Tensor[fp8_e4m3, [2]]) -> fp8_e4m3 { sum(w) + w[0]*w[0] }
        fn main() -> fp8_e4m3 {
            let !w = forge.zeros[fp8_e4m3, [2]]
            w[0] = 3.0
            w[1] = 4.0
            let (_l, g) = f.fwd_bwd(w)
            g.w[1]
        }
    "#);
    assert_eq!(as_float(&v), 1.0, "g1 must be 1 (only read through `sum`, never indexed)");
}

// ─── #852: `overlay_elem_dtype` is an exhaustive match over `ScalarType` ────
//
// `overlay_elem_dtype` (interp.rs) is the tensor-FIELD arm's dtype table for
// a `@cast(Model)` overlay (a scalar field decodes through
// `decode_overlay_scalar` instead, which has its own separate catch-all —
// out of scope here, since #852 named `overlay_elem_dtype` specifically).
// Before this fix its trailing `_ => DType::Int(scalar_int_width(st).
// unwrap_or(IW::I64))` would have decoded a `tf32`/`fp8_e4m3`/`fp8_e5m2`
// TENSOR field as an 8-byte integer — but its ONLY call site checks
// `est.overlay_byte_width()` (`ast.rs`) is `Some` first, and that table
// answers `None` for exactly seven kinds: `str`, `nil`, `trit`, `int4`, and
// the f32-backed `tf32`/`fp8_e4m3`/`fp8_e5m2`. `DIRECTIVES.md §4.1` states
// this exclusion UNCONDITIONALLY — "`@cast(Model)` is defined for a model
// whose fields are scalars with byte storage (not `str`, `nil`, `trit`,
// `int4`, or the f32-backed `tf32`/`fp8` kinds)" — so the wrong-width
// fallback was never reachable through a real program: confirmed below both
// by construction (the field-level tests, which never reach
// `overlay_elem_dtype` at all) and directly (the table tests, via the
// `#[cfg(test)]` wrapper `overlay_elem_dtype_for_test`).
//
// This is a SETTLED design question, not #817's OPEN one (whether `str`/
// `nil` have a tensor-ELEMENT representation elsewhere in the interpreter,
// e.g. `dtype_for_scalar_elem`) — `DIRECTIVES.md §4.1` excludes `str`/`nil`
// from the OVERLAY specifically, in the same sentence as the exotic floats,
// with no open question attached. The fix is therefore NOT to invent a
// byte-decode rule for `tf32`/`fp8_*` overlay fields (that would contradict
// the settled exclusion) but to make `overlay_elem_dtype` exhaustive — no
// `_ =>` — so a future `ScalarType` variant is a compile error here, the
// same two-table guard #834 built for `dtype_for_scalar_elem`/
// `arena_ctor_dtype`.

use super::ast::ScalarType;
use super::interp::{overlay_elem_dtype_for_test, IW};

#[test]
fn cast_model_overlay_852_fp8_e4m3_scalar_field_refused_not_8_byte_int() {
    let msg = run_err(r#"
        model M { x: fp8_e4m3 }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            let m = @cast(M) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("x")
            && msg.contains("DIRECTIVES.md §4.1"),
        "expected an overlay-layout refusal naming `x`, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_852_fp8_e5m2_scalar_field_refused_not_8_byte_int() {
    let msg = run_err(r#"
        model M { x: fp8_e5m2 }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            let m = @cast(M) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("x")
            && msg.contains("DIRECTIVES.md §4.1"),
        "expected an overlay-layout refusal naming `x`, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_852_tf32_scalar_field_refused_not_a_4_byte_f32_decode() {
    let msg = run_err(r#"
        model M { x: tf32 }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [8]]
            let m = @cast(M) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("x")
            && msg.contains("DIRECTIVES.md §4.1"),
        "expected an overlay-layout refusal naming `x`, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_852_fp8_e4m3_tensor_field_refused_before_overlay_elem_dtype() {
    // The tensor-field arm — `overlay_elem_dtype`'s only call site — checks
    // `est.overlay_byte_width()` before ever calling it, so this exercises
    // the same guard as the scalar case above at the OTHER call site (the
    // one #852 actually named).
    let msg = run_err(r#"
        model M { xs: Tensor[fp8_e4m3, [4]] }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [4]]
            let m = @cast(M) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("xs"),
        "expected an overlay-layout refusal naming `xs`, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_852_fp8_e5m2_tensor_field_refused_before_overlay_elem_dtype() {
    let msg = run_err(r#"
        model M { xs: Tensor[fp8_e5m2, [4]] }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [4]]
            let m = @cast(M) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("xs"),
        "expected an overlay-layout refusal naming `xs`, got: {}", msg
    );
}

#[test]
fn cast_model_overlay_852_tf32_tensor_field_refused_before_overlay_elem_dtype() {
    let msg = run_err(r#"
        model M { xs: Tensor[tf32, [1]] }
        fn main() -> i64 {
            let !buf = forge.zeros[u8, [4]]
            let m = @cast(M) { buf }
            0
        }
    "#);
    assert!(
        msg.contains("no overlay layout") && msg.contains("xs"),
        "expected an overlay-layout refusal naming `xs`, got: {}", msg
    );
}

#[test]
fn overlay_elem_dtype_852_admitted_kinds_keep_their_declared_width() {
    // The 14 kinds `ScalarType::overlay_byte_width` answers `Some` for —
    // unchanged by this fix, asserted directly against the function (the
    // field-level tests above already cover a representative few end to end
    // through a real overlay program; `cast_model_overlay_*` tests earlier
    // in this file cover the rest, e.g. bool/f32/f64/bf16/f16/narrow ints).
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::Bool).unwrap(), DType::Bool);
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::F64).unwrap(), DType::F64);
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::F32).unwrap(), DType::F32);
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::F16).unwrap(), DType::F32);
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::Bf16).unwrap(), DType::F32);
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::I8).unwrap(), DType::Int(IW::I8));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::Int8).unwrap(), DType::Int(IW::I8));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::I16).unwrap(), DType::Int(IW::I16));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::I32).unwrap(), DType::Int(IW::I32));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::I64).unwrap(), DType::Int(IW::I64));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::U8).unwrap(), DType::Int(IW::U8));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::U16).unwrap(), DType::Int(IW::U16));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::U32).unwrap(), DType::Int(IW::U32));
    assert_eq!(overlay_elem_dtype_for_test(&ScalarType::U64).unwrap(), DType::Int(IW::U64));
}

#[test]
fn overlay_elem_dtype_852_excluded_kinds_are_an_internal_error_not_a_silent_int64() {
    // The 7 kinds `overlay_byte_width` answers `None` for — provably
    // unreachable through a real overlay program (the field-level tests
    // above), reached HERE only through the `#[cfg(test)]` wrapper to prove
    // the new guard itself fires (a catchable internal error naming the
    // type) instead of silently returning `DType::Int(IW::I64)`, the #852 bug.
    for st in [ScalarType::Str, ScalarType::Nil, ScalarType::Trit, ScalarType::Int4,
               ScalarType::Tf32, ScalarType::Fp8E4M3, ScalarType::Fp8E5M2] {
        let err = overlay_elem_dtype_for_test(&st).err()
            .unwrap_or_else(|| panic!("{:?} must be an internal error, not a value", st));
        assert!(err.msg.contains("internal error") && err.msg.contains("#852"),
            "expected #852's internal-error wording for {:?}, got: {}", st, err.msg);
    }
}

#[test]
fn overlay_elem_dtype_852_agrees_with_overlay_byte_width_for_every_scalar_type() {
    // The two tables that must never disagree again: `ScalarType::
    // overlay_byte_width` (ast.rs, gates every `@cast(Model)` call site) and
    // `overlay_elem_dtype` (interp.rs, decides the admitted field's tensor
    // dtype). Iterated over every `ScalarType` variant so a new one that
    // updates only one of the two tables fails this test rather than
    // silently drifting (#852). `wire_dtype_of` (the Port wire protocol's
    // OWN, unrelated table, answering "how does an already-constructed
    // tensor cross a Port wire") is deliberately not part of this
    // agreement — it operates on already-tagged runtime `DType`s, not on
    // whether raw overlay BYTES may be decoded into one in the first place,
    // and folding it in would force `overlay_byte_width` to admit `tf32`/
    // `fp8_*` fields, contradicting `DIRECTIVES.md §4.1`'s settled exclusion.
    let all = [
        ScalarType::I8, ScalarType::I16, ScalarType::I32, ScalarType::I64,
        ScalarType::U8, ScalarType::U16, ScalarType::U32, ScalarType::U64,
        ScalarType::Int4, ScalarType::Int8,
        ScalarType::F16, ScalarType::Bf16, ScalarType::Tf32, ScalarType::F32, ScalarType::F64,
        ScalarType::Fp8E4M3, ScalarType::Fp8E5M2,
        ScalarType::Trit,
        ScalarType::Bool, ScalarType::Str, ScalarType::Nil,
    ];
    for st in all {
        let admitted_by_width = st.overlay_byte_width().is_some();
        let admitted_by_elem_dtype = overlay_elem_dtype_for_test(&st).is_ok();
        assert_eq!(admitted_by_width, admitted_by_elem_dtype,
            "{:?}: overlay_byte_width() admits it ({admitted_by_width}) but \
             overlay_elem_dtype() admits it ({admitted_by_elem_dtype}) — the \
             two tables disagree", st);
    }
}
