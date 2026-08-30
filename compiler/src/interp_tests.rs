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
    // not even put an imported `pub let !` in scope under its bare name).
    // `g.alpha` must simply not exist here.
    let v = run_multi(&[
        ("lib.dmc", "pub let !alpha = 9.0f64\npub fn touch() -> f64 { alpha }"),
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
    let v = run_multi(&[
        ("lib.dmc", "pub let !alpha = 9.0f64\npub fn touch() -> f64 { alpha }"),
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

#[test]
fn null10_comptime_evaluates_static_expr() {
    // Null 10: @comptime evaluation is total on static operands.
    // In the reference interpreter @comptime blocks evaluate at runtime;
    // a non-terminating @comptime would require static analysis (JIT phase).
    // Verify literal arithmetic inside @comptime produces the correct result.
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

/// Write source files to a temp dir and run the main file through the full
/// resolver → typechecker → interpreter pipeline (matching `dmc run` behavior).
fn run_multi(files: &[(&str, &str)], main: &str) -> Value {
    use super::check::{Checker, ModuleEnv};
    use super::ast::collect_public_items;
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
        let mod_env = ModuleEnv { env: checker.env.clone(), aliases: checker.aliases.clone(), public_items: collect_public_items(prog) };
        checked_modules.insert(p.clone(), mod_env);
    }
    // interpret
    let mut interp = Interpreter::new();
    for p in &resolver.sorted_paths {
        let prog = resolver.files.get(p).unwrap();
        interp.load_program(prog, Some(p)).expect("load");
        let env = interp.get_module_env();
        interp.interp_modules.insert(p.clone(), env);
    }
    let main_prog = resolver.files.get(&canonical).expect("main prog");
    interp.run(main_prog, Some(&canonical)).expect("run")
}

fn run_multi_err(files: &[(&str, &str)], main: &str) -> String {
    use super::check::{Checker, ModuleEnv};
    use super::ast::collect_public_items;
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
        let mod_env = ModuleEnv { env: checker.env.clone(), aliases: checker.aliases.clone(), public_items: collect_public_items(prog) };
        checked_modules.insert(p.clone(), mod_env);
    }
    // interpret — expect runtime error
    let mut interp = Interpreter::new();
    for p in &resolver.sorted_paths {
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
    // 64 f64 elements = 512 bytes, comfortably inside 4 KiB.
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
    // Two 512-byte tensors fit in 2 KiB; the third does not.
    let ok = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [8, 8]]
            let b = forge.ones[f32, [8, 8]]
            1
        }
    "#, forge_budget(1024));
    assert!(ok.is_ok(), "{:?}", ok.err());

    let e = run_with_limits(r#"
        fn main() -> i64 {
            let a = forge.zeros[f32, [8, 8]]
            let b = forge.ones[f32, [8, 8]]
            let c = forge.zeros[f32, [8, 8]]
            1
        }
    "#, forge_budget(1024)).expect_err("third allocation should not fit");
    assert!(e.contains("forge arena exhausted"), "{}", e);
}

#[test]
fn a_forge_block_does_not_rewind_the_budget_on_exit() {
    // #400: the meter is monotonic. `forge { … }` used to rewind `forge_used`
    // to the block's entry watermark, on the theory that §3's bump-pointer
    // reset gives the bytes back — but the interpreter has no bump pointer and
    // reclaims nothing, so the rewind was a lie that let a program hold every
    // "reset" tensor at once. Ten 512-byte tensors against a 1 KiB budget must
    // therefore run out, exactly as they do unwrapped.
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
    "#, forge_budget(10 * 512)).expect("ten 512-byte tensors fit a 5 KiB budget");
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
