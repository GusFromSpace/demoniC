//! #879: `coerce_to`'s tuple arm used to rebuild a `(T, Err)` tuple into a
//! *second* Forge slot even when every element's conversion could not change
//! a byte — the idiomatic `return (n, nil)` from a `(i64, Err)`-typed
//! function being the reachable case (`Nil -> Str` is a constant zero, and
//! both `Str` and `Scalar(Nil)` already encode `nil` as an all-zero i64
//! slot). The fix in `src/jit.rs`'s `coerce_to` tuple arm returns the SOURCE
//! pointer instead of rebuilding whenever every element is either unchanged
//! or that zero case, so the coerced tuple aliases its source rather than
//! being a second allocation.
//!
//! This drives the real `dmc jit` binary (not a unit test against `coerce_to`
//! directly) because what's being pinned is externally observable: the
//! minimum `--forge` a program needs, and the allocation trace
//! `DMC_FORGE_TRACE=1` prints — exactly #841's own method,
//! whose probes q1 and r7 from #841's own probe set are the ones this
//! file inlines.

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

struct Probe {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Probe {
    fn new(src: &str) -> Probe {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("probe.dmc");
        std::fs::write(&path, src).expect("write probe");
        Probe { _dir: dir, path }
    }

    /// Run `dmc jit <probe> <extra args>`, without the forge trace.
    fn jit(&self, extra: &[&str]) -> Output {
        Command::new(dmc_binary())
            .arg("jit")
            .arg(&self.path)
            .args(extra)
            .output()
            .expect("spawn dmc")
    }

    /// Same, with `DMC_FORGE_TRACE=1` set so stderr carries one
    /// `[forge #N] alloc <bytes> bytes` line per `forge_alloc` call.
    fn jit_traced(&self, extra: &[&str]) -> Output {
        Command::new(dmc_binary())
            .arg("jit")
            .arg(&self.path)
            .args(extra)
            .env("DMC_FORGE_TRACE", "1")
            .output()
            .expect("spawn dmc")
    }
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// Binary search for the lowest `--forge` at which `probe` runs to
/// completion under `dmc jit` — #841's own bisection method, as its own
/// probe notes describe it.
fn bisect_jit_minimum(probe: &Probe, hi: u64) -> u64 {
    let (mut lo, mut hi) = (1u64, hi);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let out = probe.jit(&[&format!("--forge={}", mid)]);
        if out.status.success() {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Count of `[forge #N] alloc <n> bytes` trace lines whose `<n>` is exactly
/// `n`. `DMC_FORGE_TRACE`'s format is `jit.rs:forge_trace` (`:28296` at the
/// commit this fix landed against).
fn count_allocs_of(trace: &str, n: u64) -> usize {
    let needle = format!("alloc {} bytes", n);
    trace.lines().filter(|l| l.contains(&needle)).count()
}

/// Baseline shared by every probe below: one 16 KiB `forge.zeros` tensor,
/// exactly #841's own control row (`p0`), so the tuple's own cost
/// is isolated as the excess over 16 384.
const BASELINE: &str = "let !t = forge.zeros[f32, [4096]]\n    t[0] = 1.0\n    print(sum(t))\n";

/// 0841 §2.3 q1 — a `(i64, Err)` return whose error half is the literal
/// `nil`. Before #879's fix this bisects to 16 512 (two 64 B Forge slots);
/// after, to 16 448 (one), matching `r7` below exactly.
fn q1_source() -> String {
    format!(
        "fn tup(n: i64) -> (i64, Err) {{ (n, nil) }}\n\
         fn main() -> nil {{\n    {}    let (a, e) = tup(2)\n    nil\n}}\n",
        BASELINE
    )
}

/// 0841 §2.3 r7 — `q1`'s shape with an `Err`-typed error half, no `nil`
/// literal: the tuple's element types already match the return type, so
/// `coerce_to`'s tuple arm never fires and there was only ever one
/// allocation. This is the control that #879's fix must not move.
fn r7_source() -> String {
    format!(
        "fn tup(n: i64, m: Err) -> (i64, Err) {{ (n, m) }}\n\
         fn main() -> nil {{\n    {}    let (a, e) = tup(2, nil)\n    nil\n}}\n",
        BASELINE
    )
}

/// A tuple coercion that genuinely changes representation — an `i64`
/// literal defaulting against a declared `f32` element — must still rebuild
/// into a fresh slot: the short-circuit is for byte-identical conversions
/// only, never a blanket "reuse the source" rule.
fn mixed_source() -> String {
    format!(
        "fn tup() -> (i64, f32) {{ (7, 2.5) }}\n\
         fn main() -> nil {{\n    {}    let (a, b) = tup()\n    nil\n}}\n",
        BASELINE
    )
}

#[test]
fn q1_bisects_to_16448_after_the_fix() {
    let probe = Probe::new(&q1_source());
    let min = bisect_jit_minimum(&probe, 1 << 20);
    assert_eq!(
        min, 16_448,
        "`(n, nil)` returned as `(i64, Err)` should cost one 64 B Forge slot \
         (16 448), not two (16 512) — #879 regressed if this is 16 512"
    );
}

#[test]
fn q1_traces_a_single_tuple_allocation() {
    let probe = Probe::new(&q1_source());
    let out = probe.jit_traced(&["--forge=16448"]);
    assert!(out.status.success(), "stderr:\n{}", stderr(&out));
    let trace = stderr(&out);
    // One `alloc 16384` for the baseline tensor, one `alloc 16` for the
    // tuple `tup` builds — never a second `alloc 16` for a rebuild.
    assert_eq!(count_allocs_of(&trace, 16), 1, "trace:\n{}", trace);
    assert_eq!(count_allocs_of(&trace, 16_384), 1, "trace:\n{}", trace);
}

#[test]
fn r7_is_unchanged_by_the_fix() {
    let probe = Probe::new(&r7_source());
    let min = bisect_jit_minimum(&probe, 1 << 20);
    assert_eq!(min, 16_448, "r7's own minimum must not move");

    let out = probe.jit_traced(&["--forge=16448"]);
    assert!(out.status.success(), "stderr:\n{}", stderr(&out));
    let trace = stderr(&out);
    assert_eq!(count_allocs_of(&trace, 16), 1, "trace:\n{}", trace);
}

#[test]
fn a_representation_changing_tuple_still_rebuilds() {
    let probe = Probe::new(&mixed_source());
    // Plenty of headroom — this test is about the allocation count, not the
    // minimum.
    let out = probe.jit_traced(&["--forge=65536"]);
    assert!(out.status.success(), "stderr:\n{}", stderr(&out));
    let trace = stderr(&out);
    // The literal tuple `(7, 2.5)` (one alloc) and its coercion to
    // `(i64, f32)` (a second alloc, because the `f32` element's bytes
    // genuinely differ from the defaulted-wide source) — the short-circuit
    // must not fire here.
    assert_eq!(
        count_allocs_of(&trace, 16), 2,
        "a tuple whose element representation actually changes must still \
         rebuild into a second slot; trace:\n{}", trace
    );
}
