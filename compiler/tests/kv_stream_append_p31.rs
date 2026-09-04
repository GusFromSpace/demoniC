//! p31: `<-` accepts an appendee with the streaming axis dropped (SPEC §4.8),
//! and counts frames rather than feature width when enforcing capacity (§3.6).
//!
//! `SPEC.md §4.8` gives the appendee two spellings for a `KV[T, S]`: `S` with
//! the streaming axis carrying the appended extent, or `S` with that axis
//! dropped, which appends one frame. The interpreter handed both operands
//! straight to `ndarray::concatenate`, which requires equal rank, so only the
//! first ever worked — the dropped form failed as
//! `ShapeError/IncompatibleShape`.
//!
//! The capacity check was wrong in the same place. It read the appended extent
//! off `shape()[axis]`, which for a dropped-form `[4, 8]` into a `[4, ~, 8]`
//! cache is `shape()[1]` — 8, the feature width, not 1 frame. That was latent
//! rather than observable before the rank fix, because the append died at the
//! concatenate a few lines later either way; it becomes reachable the moment
//! the dropped form works, and it is wrong by a factor of the feature width.
//! Promoting the appendee before the check rather than after is what keeps the
//! two consistent.
//!
//! These live here rather than only in the spec probe because a probe has to
//! exit 0, so it cannot assert on a refusal.
//!
//! The happy paths and the cursor arithmetic are covered end-to-end by
//! `spec_probes/p31_kv_stream_append.dmc`.

use std::path::PathBuf;
use std::process::{Command, Output};

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

fn run_src(name: &str, src: &str) -> Output {
    let tmp = std::env::temp_dir().join(format!("dmc_kv_p31_{}.dmc", name));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(dmc_binary())
        .args(["run", tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

fn assert_prints(name: &str, src: &str, want: &str) {
    let out = run_src(name, src);
    let so = String::from_utf8_lossy(&out.stdout).to_string();
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "[{name}] exited {:?}: {se}", out.status.code());
    assert_eq!(so, want, "[{name}] stdout (stderr: {se})");
}

/// The issue's canonical pair: a `[4, 8]` token into a `KV[f32, [4, ~, 8]]`.
/// Two appends of ones is 2 * 4 * 8 = 64 only if each advanced the cursor by 1.
#[test]
fn the_dropped_streaming_axis_form_appends_one_frame() {
    let src = r#"fn main() -> nil {
    let !cache: KV[f32, [4, ~, 8]] = forge.kv[f32, [4, ~, 8]](capacity = 64)
    let tok = forge.ones[f32, [4, 8]]
    cache <- tok
    cache <- tok
    print(sum(cache))
    nil
}
"#;
    assert_prints("dropped_form", src, "64\n");
}

/// Capacity counts frames along the streaming axis, not the appendee's other
/// dimensions. Eight `[2, 3]` appends into a `capacity = 8` cache all fit; if
/// the check read the feature width (3) it would refuse the third.
#[test]
fn capacity_counts_frames_not_feature_width() {
    let src = r#"fn main() -> nil {
    let !c = forge.kv[f32, [2, ~, 3]](capacity = 8)
    let f = forge.ones[f32, [2, 3]]
    c <- f
    c <- f
    c <- f
    c <- f
    c <- f
    c <- f
    c <- f
    c <- f
    print(sum(c))
    nil
}
"#;
    // 8 frames x 2 x 3 = 48.
    assert_prints("capacity_frames", src, "48\n");
}

/// And the limit still bites at the right place: the ninth append is one frame
/// past a `capacity = 8` cache.
#[test]
fn an_append_past_capacity_is_still_a_runtime_error() {
    let src = r#"fn main() -> nil {
    let !c = forge.kv[f32, [2, ~, 3]](capacity = 8)
    let f = forge.ones[f32, [2, 3]]
    for _ in 0..9 { c <- f }
    print(sum(c))
    nil
}
"#;
    let out = run_src("over_capacity", src);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "expected a failure, got: {combined}");
    assert!(
        combined.contains("exceeded declared capacity"),
        "expected the §3.6 capacity error, got: {combined}"
    );
}

/// A rank the spec does not allow is refused with a message that names both
/// legal spellings, rather than surfacing ndarray's `IncompatibleShape`.
#[test]
fn an_unappendable_rank_is_refused_by_name() {
    let src = r#"fn main() -> nil {
    let !c = forge.kv[f32, [2, ~, 3]](capacity = 8)
    let bad = forge.ones[f32, [3]]
    c <- bad
    nil
}
"#;
    let out = run_src("bad_rank", src);
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "expected a failure, got: {combined}");
    assert!(
        combined.contains("streaming axis"),
        "expected a message naming the streaming axis, got: {combined}"
    );
}
