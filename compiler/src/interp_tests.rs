/// Interpreter unit tests — small programs hitting specific eval paths.
/// Integration: `dmc run examples/*.dmc` from the shell.

use super::interp::{Interpreter, Value};
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
    if let Value::Int(n) = v { *n } else { panic!("expected int, got {:?}", v) }
}
fn as_float(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        Value::Int(n)   => *n as f64,
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
    // A value that exits the gradient graph (an indexed read) still errors —
    // and the hint must name the traced reductions and flag what doesn't trace
    // (#253). max/min/variance now trace (#307 Tier C), so they're no longer
    // in the "don't trace yet" set — use an indexed read to trigger the hint.
    let e = run_err(r#"
        @grad fn f[N](!w: Tensor[f32, [N]]) -> f32 { w[0] }
        fn main() -> f32 {
            let !w = forge.zeros[f32, [2]]
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
fn grad_indexed_reduction_gives_helpful_hint() {
    // Regression for #71: using indexed reads (sq[0]+sq[1]+...) breaks the
    // gradient graph. The error must mention tensor reductions as the fix.
    let msg = run_err(r#"
        @grad fn loss(!x: Tensor[f32, [4]]) -> f32 {
            let sq = x .* x
            sq[0] + sq[1] + sq[2] + sq[3]
        }
        fn main() -> nil {
            let x = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            let (_v, _g) = loss.fwd_bwd(x)
            nil
        }
    "#);
    assert!(msg.contains("indexed") || msg.contains("sum") || msg.contains("reduction"),
        "expected hint about reductions in error, got: {}", msg);
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

#[test]
fn port_roundtrip_python() {
    // #402: process-port floor — open a python port, call through the JSON
    // ABI, close. python3 is a dev prerequisite here, same as
    // examples/port_bridge.dmc. `len` avoids float-formatting assumptions.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            if e1 != nil { return "open failed" }
            let (out, e2) = port_call(p, "len", "[1, 2, 3]")
            if e2 != nil { return "call failed" }
            let (_, e3) = port_close(p)
            if e3 != nil { return "close failed" }
            out
        }
    "#);
    assert_eq!(as_str(&v), "3");
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
    // `port-call` tag, not kill the port or the interpreter.
    let v = run(r#"
        fn main() -> str {
            let (p, e1) = port_open("python")
            let (out, e2) = port_call(p, "math.sqrt", json_encode("sixteen"))
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

#[test]
fn pipe_compose_basic() {
    let v = run(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn main() -> i64 { 5 >> inc }
    "#);
    assert_eq!(as_int(&v), 6);
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
fn pipe_and_compose_equivalent() {
    let v = run(r#"
        fn inc(x: i64) -> i64 { x + 1 }
        fn double(x: i64) -> i64 { x * 2 }
        fn main() -> i64 {
            let a = 5 |> inc |> double
            let b = 5 >> inc >> double
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
    assert!(matches!(v, Value::Float(f) if f.is_infinite() && f > 0.0));
}

#[test]
fn math_const_nan() {
    let v = run(r#"fn main() -> f64 { nan }"#);
    assert!(matches!(v, Value::Float(f) if f.is_nan()));
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
    // #335: `to_string` is the Rust/Python name models reach for.
    assert_eq!(as_str(&run(r#"fn main() -> str { to_string(42) }"#)), "42");
    assert_eq!(as_str(&run(r#"fn main() -> str { to_string(true) }"#)), "true");
}

#[test]
fn to_binary_aliases_to_bin() {
    // #335: `to_binary` is the name models reach for; aliases `to_bin`.
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
fn ufcs_unlocks_harvest_targets() {
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
    assert!(matches!(v, Value::Float(f) if f.is_infinite()));
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
        return e;
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
    assert!(matches!(v, Value::Int(-7)), "expected -7, got {:?}", v);
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
    assert!(matches!(v, Value::Int(21)), "gcd(462,1071) expected 21, got {:?}", v);
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
        Value::Float(f) => assert!((f - 9.0).abs() < 1e-6, "expected 9.0, got {f}"),
        Value::Int(n)   => assert_eq!(n, 9, "expected 9"),
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
        Value::Float(f) => assert!((f - 3.0).abs() < 1e-6, "expected x[0]=3.0 after swap, got {f}"),
        Value::Int(n)   => assert_eq!(n, 3, "expected x[0]=3 after swap"),
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
        Value::Float(f) => assert!((f - 30.0).abs() < 1e-6, "expected v[2]=30.0, got {f}"),
        Value::Int(n)   => assert_eq!(n, 30, "expected v[2]=30"),
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
        Value::Float(f) => assert!((f - 1.0).abs() < 1e-6, "non-! param should NOT write back, got {f}"),
        Value::Int(n)   => assert_eq!(n, 1, "non-! param should NOT write back"),
        other           => panic!("expected numeric, got {:?}", other),
    }
}

// ── Issue #115: solve / inv / lstsq stdlib primitives ────────────────────────

fn approx(v: &Value, expected: f64, tol: f64, label: &str) {
    let got = match v {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
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
    approx(&Value::Float(tensor_elem(&v, &[0])), 1.0, 1e-9, "x");
    approx(&Value::Float(tensor_elem(&v, &[1])), 3.0, 1e-9, "y");
}

#[test]
fn inv_2x2() {
    // inv([[1,2],[3,4]]) = [[-2,1],[1.5,-0.5]]
    let v = run(r#"
fn main() -> Tensor[f32, [2, 2]] {
    inv([[1.0, 2.0], [3.0, 4.0]])
}
"#);
    approx(&Value::Float(tensor_elem(&v, &[0, 0])), -2.0,  1e-9, "[0,0]");
    approx(&Value::Float(tensor_elem(&v, &[0, 1])),  1.0,  1e-9, "[0,1]");
    approx(&Value::Float(tensor_elem(&v, &[1, 0])),  1.5,  1e-9, "[1,0]");
    approx(&Value::Float(tensor_elem(&v, &[1, 1])), -0.5,  1e-9, "[1,1]");
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
    approx(&Value::Float(tensor_elem(&v, &[0])), 1.9857142857, 1e-6, "slope");
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
    assert!(matches!(v, Value::Int(0)), "all draws must be in [3, 7), got {:?}", v);
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
    assert!(matches!(v, Value::Int(7)), "snapshot must hold 7, got {:?}", v);
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
    assert!(e.contains("#394") && e.contains("serialization"), "got: {}", e);
    // The empty/positional bracket forms parse as Index (not BracketArgs) and
    // used to slip past the guard to a silent opaque — they must also be loud.
    let empty = run_err("model L { w: Tensor[f32,[2]] }\nfn main() -> nil { let l = L[].load(\"/tmp/x\")  nil }");
    assert!(empty.contains("#394") && empty.contains("serialization"),
            "empty-bracket load must error, got: {}", empty);
}

#[test]
fn allreduce_method_form_errors_396() {
    // `allreduce.sum(y)` must not silently desugar to `sum(allreduce, y)` (→ 0.0).
    let e = run_err("fn main() -> nil { let !y = forge.zeros[f32,[3]]  y[0] = 1.0  let r = allreduce.sum(y, axis=0)  nil }");
    assert!(e.contains("#396") && e.contains("collective"), "got: {}", e);
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
        Value::Float(f) => assert!((f - 28.0).abs() < 1e-6, "got {}", f),
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
        Value::Float(f) => assert!((f - 7.0).abs() < 1e-6, "got {}", f),
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
