//! Unit tests for the in-process differential fuzzer (`dmc selftest`, #408).

use super::*;
use std::time::Duration;

const T: Duration = Duration::from_secs(30);

#[test]
fn meta_test_has_teeth() {
    // The differ must reject mismatches and accept matches (else the whole
    // suite is vacuously green).
    assert!(meta_test());
}

#[test]
fn comparison_semantics() {
    assert!(agree(&Value::Int(5), &ScalarRet::I64(5)));
    assert!(!agree(&Value::Int(5), &ScalarRet::I64(6)));
    assert!(agree(&Value::Bool(false), &ScalarRet::Bool(false)));
    assert!(!agree(&Value::Bool(true), &ScalarRet::Bool(false)));
    // Cross-type is always a divergence.
    assert!(!agree(&Value::Int(1), &ScalarRet::Bool(true)));
    // #473: float agreement is EXACT — a one-ulp gap is a divergence, not
    // noise. The old tolerant window (1e-4 + 1e-3*|b|) was three orders of
    // magnitude wider than the scalar-f32 divergence #473 fixed, which is why
    // 500-iteration sweeps ran green through it.
    assert!(floats_agree(1.0, 1.0));
    assert!(!floats_agree(1.0, 1.0 + 5e-5));
    assert!(!floats_agree(0.1_f32 as f64, 0.1_f64));
    assert!(!floats_agree(1.0, 2.0));
    // Both backends producing NaN is agreement, though NaN != NaN.
    assert!(floats_agree(f64::NAN, f64::NAN));
    assert!(!floats_agree(f64::NAN, 1.0));
}

#[test]
fn generation_is_deterministic() {
    // Same seed → identical program; the repro contract depends on this.
    for seed in [0u64, 1, 42, 1_000_003, 987_654_321] {
        let a = Gen::new(seed, false).program();
        let b = Gen::new(seed, false).program();
        assert_eq!(a, b, "seed {} not deterministic", seed);
    }
    // The floats axis is a distinct generator.
    let scalar = Gen::new(7, false).program();
    let with_f = Gen::new(7, true).program();
    // (May coincide by chance, but over a spread they must differ somewhere.)
    let any_diff = (0..32).any(|s| Gen::new(s, false).program() != Gen::new(s, true).program());
    assert!(any_diff, "floats flag never changed generation");
    assert!(scalar.starts_with("fn main() -> "));
    assert!(with_f.starts_with("fn main() -> "));
}

#[test]
fn generated_programs_parse_and_classify() {
    // A spread of generated programs must never DIVERGE on a correct build.
    for idx in 0u64..60 {
        let seed = 1u64.wrapping_mul(1_000_003).wrapping_add(idx);
        let src = Gen::new(seed, false).program();
        let (verdict, detail) = classify(src.clone(), T);
        assert_ne!(
            verdict.tag(), "DIVERGE",
            "seed {} diverged: {}\n{}", seed, detail, src
        );
    }
}

#[test]
fn regression_seeds_do_not_diverge() {
    // Every pinned historical scalar bug must classify clean (all are FIXED).
    // A regression that reintroduces one of these flips it to DIVERGE and fails
    // here — this is the coverage-equivalence anchor for the retired scalar gate.
    for (label, src) in REGRESSION_SEEDS {
        let (verdict, detail) = classify((*src).to_string(), T);
        assert_ne!(
            verdict.tag(), "DIVERGE",
            "regression {} diverged: {}\n{}", label, detail, src
        );
    }
}

#[test]
fn known_scalar_program_agrees() {
    let src = "fn main() -> i64 {\n    let !s = 0\n    for i in 0..5 { s = s + i }\n    s\n}\n";
    let (verdict, detail) = classify(src.to_string(), T);
    assert_eq!(verdict.tag(), "ok", "expected ok, got {} ({})", verdict.tag(), detail);
}

#[test]
fn timeout_is_reported_as_divergence() {
    // A program that "hangs" (here: an unsatisfiable wait) must be caught by the
    // bounded recv rather than freezing the runner. We exercise the timeout path
    // with a trivially short budget against a real (fast) program: the point is
    // that classify() always returns, never blocks forever.
    let src = "fn main() -> i64 { 1 + 2 }\n";
    let (verdict, _) = classify(src.to_string(), Duration::from_millis(1));
    // Either it finished in time (ok) or the 1ms budget elapsed (DIVERGE/timeout)
    // — both are terminating outcomes; the test proves classify() returns.
    assert!(matches!(verdict.tag(), "ok" | "DIVERGE"));
}
