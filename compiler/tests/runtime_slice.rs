//! #511: tensor slices with a RUNTIME start and a COMPILE-TIME extent
//! (`t[i..i+1, ..]` with a loop-var `i`) must work under the JIT, not only
//! the interpreter. The static extent keeps the result shape compile-time
//! (spec invariant 5); the runtime start lowers to address arithmetic.
//!
//! These drive the actual `dmc` binary under BOTH backends because the class
//! of bug being tested is a backend divergence, and because the OOB case
//! calls `exit(1)` from a JIT runtime helper (which would take an in-process
//! test harness with it).

use std::path::PathBuf;
use std::process::{Command, Output};

/// Path to the `dmc` binary built by cargo for this crate.
fn dmc_binary() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_dmc") {
        return PathBuf::from(path);
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    #[cfg(debug_assertions)]
    let profile = "debug";
    #[cfg(not(debug_assertions))]
    let profile = "release";
    crate_dir.join("target").join(profile).join("dmc")
}

fn run_mode(name: &str, mode: &str, src: &str) -> Output {
    let bin = dmc_binary();
    let tmp = std::env::temp_dir().join(format!("dmc_rt_slice_{}.dmc", name));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(&bin)
        .args([mode, tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

/// Run `src` under `dmc run` and `dmc jit`; both must exit 0 and print
/// exactly `want` on stdout.
fn assert_both_print(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        let out = run_mode(name, mode, src);
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(out.status.success(), "[{} {}] exited {:?}: {}", mode, name, out.status.code(), se);
        assert_eq!(so, want, "[{} {}] stdout (stderr: {})", mode, name, se);
    }
}

/// #517: run `src` under `dmc run` and `dmc jit`; both must exit 1 and print
/// `want` on stderr, verbatim (not just `contains` — the whole point of the
/// fix is that the two backends' diagnostics are now identical).
fn assert_both_trap(name: &str, src: &str, want: &str) {
    for mode in ["run", "jit"] {
        let out = run_mode(name, mode, src);
        assert_eq!(out.status.code(), Some(1), "[{} {}] expected exit 1", mode, name);
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert_eq!(se.trim_end(), want, "[{} {}] stderr", mode, name);
    }
}

/// The issue's repro, verbatim. Row i sums to 40i + 6; Σ over i = 0..4 is 430.
/// Before #511 the JIT rejected line 8 with "unbound shape parameter `i`".
#[test]
fn issue_511_repro_prints_430_under_both_backends() {
    let src = r#"fn main() -> nil {
    let !table = forge.zeros[f32, [8, 4]]
    for i in 0..8 {
        for j in 0..4 { table[i, j] = (i as f32) * 10.0 + (j as f32) }
    }
    let !acc = 0.0
    for i in 0..5 {
        let row = table[i..i+1, ..]   # <- runtime-index slice
        acc = acc + sum(row)
    }
    print(acc)
    nil
}
"#;
    assert_both_print("repro", src, "430\n");
}

/// Runtime start at the boundary indices: 0 and the last valid row.
#[test]
fn runtime_start_at_zero_and_last_valid_index() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[f32, [4, 3]]
    for i in 0..4 { for j in 0..3 { t[i, j] = (i as f32) * 3.0 + (j as f32) } }
    let !k = 0
    print(sum(t[k..k+1, ..]))
    let !l = 3
    print(sum(t[l..l+1, ..]))
    nil
}
"#;
    // Row 0 = 0+1+2 = 3; row 3 = 9+10+11 = 30.
    assert_both_print("bounds", src, "3\n30\n");
}

/// Multi-dim: a runtime range on a TRAILING axis exercises the strided
/// (element-wise copy) path, an extent-2 leading slab the view path, and the
/// inclusive form the `..=` extent bump.
#[test]
fn runtime_start_multi_dim_and_inclusive() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[f32, [4, 3]]
    for i in 0..4 { for j in 0..3 { t[i, j] = (i as f32) * 3.0 + (j as f32) } }
    let !m = 1
    print(sum(t[.., m..m+2]))    # cols 1..3 of every row = 48
    print(sum(t[m..m+2, ..]))    # rows 1..3 = 33
    print(sum(t[m..=m+1, ..]))   # same rows, inclusive spelling = 33
    nil
}
"#;
    assert_both_print("multidim", src, "48\n33\n33\n");
}

/// A matmul and a reduction both consuming the runtime-start slice — the
/// qwen3 decode shape (`cos_table[pos..pos+1, ..]` feeding attention math).
#[test]
fn matmul_and_reduction_consume_the_slice() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[f32, [4, 3]]
    for i in 0..4 { for j in 0..3 { t[i, j] = (i as f32) * 3.0 + (j as f32) } }
    let !w = forge.zeros[f32, [3, 1]]
    for j in 0..3 { w[j, 0] = 1.0 }
    let !acc = 0.0
    for i in 0..4 {
        let mm = t[i..i+1, ..] @ w      # [1,3] @ [3,1] = [[row sum]]
        acc = acc + sum(mm)
    }
    print(acc)                          # Σ all elements = 66
    nil
}
"#;
    assert_both_print("matmul", src, "66\n");
}

/// A runtime start whose static-extent window does not fit the axis is
/// dynamic OOB — a runtime panic under both backends (SPEC §4.3). Closed by
/// #517: the interpreter used to clamp dynamic range bounds instead
/// (interp.rs `resolve_index_values`), diverging from the JIT (which cannot
/// clamp without changing the compile-time result shape, so it always
/// followed the spec here). Both backends now emit the identical message
/// and exit(1).
#[test]
fn both_backends_trap_on_out_of_range_runtime_start() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[f32, [4, 3]]
    let !i = 4
    print(sum(t[i..i+1, ..]))
    nil
}
"#;
    assert_both_trap("oob", src, "runtime error: slice start 4 with extent 1 out of bounds for axis of size 4");
}

/// A NEGATIVE runtime start traps under both backends — it does NOT resolve
/// from the end. The static extent is `b - a`, but the pre-#517 interpreter
/// normalized each bound independently (`resolve_index_values`): `i..i+1`
/// with `i = -1` on a size-4 axis became `3..0`, an EMPTY slice (`dmc run`
/// printed 0). End-resolution instead of a trap would have returned row 3 —
/// a SILENT wrong-answer divergence from the JIT. No resolution preserves
/// both the static extent and the naively-normalized value, so both
/// backends treat a negative start as dynamic OOB and panic loudly (SPEC
/// §4.3) rather than guessing.
#[test]
fn both_backends_trap_on_negative_runtime_start() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[f32, [4, 3]]
    for i in 0..4 { for j in 0..3 { t[i, j] = (i as f32) * 3.0 + (j as f32) } }
    let !i = 0 - 1
    print(sum(t[i..i+1, ..]))
    nil
}
"#;
    let out_run = run_mode("neg", "run", src);
    let out_jit = run_mode("neg-jit", "jit", src);
    for (mode, out) in [("run", &out_run), ("jit", &out_jit)] {
        assert_eq!(out.status.code(), Some(1), "[{}] expected exit 1", mode);
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert_eq!(
            se.trim_end(),
            "runtime error: slice start -1 with extent 1 out of bounds for axis of size 4",
            "[{}] stderr", mode
        );
        // Silent-row-3 guard: had the trap not fired, stdout would hold 30.
        let so = String::from_utf8_lossy(&out.stdout).to_string();
        assert!(!so.contains("30"), "[{}] negative start silently returned a row: {}", mode, so);
    }
}

/// A runtime slice whose EXTENT is also runtime stays rejected — at compile
/// time, with an actionable message (not "unbound shape parameter"). The
/// wording says "runtime bound" because the refusal also fires when only
/// the END is runtime (`t[0..b, ..]`).
#[test]
fn runtime_extent_still_rejected_at_compile_time() {
    let src = r#"fn main() -> nil {
    let !t = forge.zeros[f32, [4, 3]]
    let !a = 1
    let !b = 3
    print(sum(t[a..b, ..]))
    nil
}
"#;
    let out = run_mode("rt_extent", "jit", src);
    assert_ne!(out.status.code(), Some(0), "expected a jit error");
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        se.contains("slice with a runtime bound needs a compile-time extent"),
        "unexpected stderr: {}", se
    );
}

// ── #517 follow-up: a SHAPE PARAMETER in a range bound must classify as
// STATIC, not dynamic ──────────────────────────────────────────────────────
//
// `S..S+1` looks exactly like the dynamic pattern `i..i+1` from #517's own
// fix, but `S` is a shape parameter: the JIT's `shape_env` resolves it to a
// literal at monomorphization, so BOTH `S` and `S+1` fold to compile-time
// constants and the range classifies as the STATIC `IndexCat::Range`
// (#291.4 clamp), never reaching `bounds_check_slice_start`/
// `__dmc_slice_oob_trap` at all. The interpreter's `affine_form` must agree
// — treat a currently-bound shape param as a constant (`is_shape_param`),
// not as a runtime term — or the two backends pick different arms of
// `resolve_index_values` for the identical range and diverge on this OOB
// case (`dmc run` used to trap here; `dmc jit` clamps to an empty slice and
// then dies on `r[0]`'s ordinary scalar-index OOB instead).

/// `S..S+1` with `S` inferred as 4 from `x`'s shape, on a size-4 axis: the
/// window `[4, 5)` is out of range. Both backends must clamp to an EMPTY
/// slice (the STATIC #291.4 path — this is not dynamic OOB), so `r[0]`
/// fails as an ordinary scalar index into a zero-length axis on both. The
/// location prefix differs by design (`dmc run`'s `RuntimeError::at` carries
/// a span, `dmc jit`'s `dmc_oob_trap` never does — a pre-existing,
/// out-of-#517-scope difference), so this checks message content, not the
/// whole line.
#[test]
fn shape_param_range_bound_classifies_as_static_not_dynamic() {
    let src = r#"fn probe[S](x: Tensor[f32, [S]]) -> nil {
    let r = x[S..S + 1]
    print(r[0])
    nil
}

fn main() -> nil {
    let !t = forge.zeros[f32, [4]]
    for i in 0..4 { t[i] = (i * 10) as f32 }
    probe(t)
    nil
}
"#;
    for mode in ["run", "jit"] {
        let out = run_mode("shape_oob", mode, src);
        assert_eq!(out.status.code(), Some(1), "[{}] expected exit 1", mode);
        let se = String::from_utf8_lossy(&out.stderr).to_string();
        assert!(
            se.contains("index 0 out of bounds for axis 0 of size 0"),
            "[{}] unexpected stderr: {}", mode, se
        );
    }
}

/// The in-bounds control case from the same shape: `S-1..S` selects the
/// last element (index 3, value 30) and must print identically on both
/// backends — this is the legal case `dyn_range_extent`/`affine_form` must
/// NOT disturb.
#[test]
fn shape_param_range_bound_in_range_still_works() {
    let src = r#"fn probe[S](x: Tensor[f32, [S]]) -> nil {
    let r = x[S - 1..S]
    print(r[0])
    nil
}

fn main() -> nil {
    let !t = forge.zeros[f32, [4]]
    for i in 0..4 { t[i] = (i * 10) as f32 }
    probe(t)
    nil
}
"#;
    assert_both_print("shape_ok", src, "30\n");
}
