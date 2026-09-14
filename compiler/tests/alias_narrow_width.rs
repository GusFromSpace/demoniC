//! A `type` alias (SPEC §6.6) is transparent at every run-time width origin
//! (SPEC §3.1): a typed `let`, a typed parameter, a declared return, a
//! declared model field and an `as` cast all narrow through `type Role = u8`
//! exactly as they do through `u8`.
//!
//! The checker already resolved aliases, so `let r: Role = 255` and
//! `let r: u8 = 255` were the same program to `dmc --check`; but the
//! interpreter's width coercion matched the raw AST `Type::Scalar` and never
//! saw through the alias, so the aliased `r` stayed a full i64 and `r + 1`
//! printed 256 under `dmc run` where the direct spelling prints 0.
//!
//! An alias is also a per-MODULE name: the checker runs per file, leaf-first,
//! so an imported module's fn bodies mean its own `type Role`, not the
//! importer's. Both orientations of that pair are pinned below.
//!
//! These drive the real `dmc` binary under `dmc run` ONLY. The JIT refuses a
//! top-level `type` item outright (`jit unsupported … `type` aliases at top
//! level`), and its alias transparency is a separate change; when it lands,
//! widen these to both backends the way `tests/narrow_int_wrap_544.rs` runs.
//! The in-process unit tests live in `src/interp_tests.rs`.

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
    let tmp = std::env::temp_dir().join(format!("dmc_alias_narrow_{}.dmc", name));
    std::fs::write(&tmp, src).expect("write temp probe");
    let out = Command::new(dmc_binary())
        .args(["run", tmp.to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    let _ = std::fs::remove_file(&tmp);
    out
}

/// The same, for a program that spans files: every `(filename, source)` pair
/// is written into one temp directory and `main` is run from there, so the
/// real resolver loads the dependency graph leaf-first exactly as it does for
/// a user's project.
fn run_files(name: &str, files: &[(&str, &str)], main: &str) -> Output {
    let dir = std::env::temp_dir().join(format!("dmc_alias_narrow_{}", name));
    let _ = std::fs::create_dir_all(&dir);
    for (file, src) in files {
        std::fs::write(dir.join(file), src).expect("write module");
    }
    let out = Command::new(dmc_binary())
        .args(["run", dir.join(main).to_str().unwrap()])
        .output()
        .expect("invoke dmc");
    for (file, _) in files {
        let _ = std::fs::remove_file(dir.join(file));
    }
    let _ = std::fs::remove_dir(&dir);
    out
}

/// `dmc run` must exit 0 and print exactly `want`. The literal expected text
/// is asserted — not "same as the direct spelling" — so the test pins the
/// rule and not merely the agreement.
fn assert_run_prints(name: &str, src: &str, want: &str) {
    let out = run_src(name, src);
    let so = String::from_utf8_lossy(&out.stdout).to_string();
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "[run {name}] exited {:?}: {se}", out.status.code());
    assert_eq!(so, want, "[run {name}] stdout (stderr: {se})");
}

/// The repro: `let r: Role = 255` then `r + 1` wraps to 0, as `let r: u8` does.
#[test]
fn aliased_let_wraps_at_the_aliased_width() {
    let src = r#"type Role = u8
fn main() -> nil {
    let r: Role = 255
    print((r + 1) as i64)
    let d: u8 = 255
    print((d + 1) as i64)
    nil
}
"#;
    assert_run_prints("let", src, "0\n0\n");
}

/// A typed parameter spelled through the alias narrows the argument.
#[test]
fn aliased_param_wraps() {
    let src = r#"type Role = u8
fn wide(r: Role) -> i64 { (r + 1) as i64 }
fn main() -> nil {
    print(wide(255))
    nil
}
"#;
    assert_run_prints("param", src, "0\n");
}

/// A declared return type spelled through the alias narrows the result;
/// a signed alias wraps two's-complement.
#[test]
fn aliased_return_wraps() {
    let src = r#"type Role = u8
type Small = i8
fn bump() -> Role { 255 + 1 }
fn sbump(r: Small) -> Small { r + 1 }
fn main() -> nil {
    print(bump() as i64)
    print(sbump(127) as i64)
    nil
}
"#;
    assert_run_prints("return", src, "0\n-128\n");
}

/// `as Role` wraps as `as u8` does, and a chain of aliases resolves fully.
#[test]
fn aliased_cast_and_alias_chain_wrap() {
    let src = r#"type Role = u8
type Rank = Role
fn main() -> nil {
    print((300 as Role) as i64)
    let r: Rank = 255
    print((r + 1) as i64)
    nil
}
"#;
    assert_run_prints("cast_chain", src, "44\n0\n");
}

/// A model field declared through the alias narrows at construction and on
/// a later field write.
#[test]
fn aliased_model_field_wraps() {
    let src = r#"type Role = u8
model M { r: Role }
fn main() -> nil {
    let !m = M { r: 255 }
    print((m.r + 1) as i64)
    m.r = 254
    print((m.r + 2) as i64)
    nil
}
"#;
    assert_run_prints("field", src, "0\n0\n");
}

/// A `type` alias is a per-MODULE name. The importer's `type Role = u8` must
/// not reach into the fn bodies of the library it imported, whose own
/// `pub type Role = i64` is what the checker type-checked them against: this
/// printed 0 (the importer's width) where the checker meant 256.
#[test]
fn an_importers_alias_does_not_retype_the_imported_module() {
    let out = run_files("xmod_wide", &[
        ("lib.dmc", "pub type Role = i64\npub fn libbump(r: Role) -> Role { r + 1 }\n"),
        ("main.dmc", r#"use "lib.dmc"
type Role = u8
fn main() -> nil {
    print(libbump(255) as i64)
    let r: Role = 255
    print((r + 1) as i64)
    nil
}
"#),
    ], "main.dmc");
    let so = String::from_utf8_lossy(&out.stdout).to_string();
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "[xmod_wide] exited {:?}: {se}", out.status.code());
    assert_eq!(so, "256\n0\n", "[xmod_wide] stdout (stderr: {se})");
}

/// The reverse orientation: the library is the narrow one. Its `u8` must
/// survive an importer that widens the same name — this printed 256, dropping
/// the wrap the checker had accepted.
#[test]
fn an_imported_modules_narrow_alias_survives_a_wider_importer() {
    let out = run_files("xmod_narrow", &[
        ("lib.dmc", "pub type Role = u8\npub fn libbump(r: Role) -> Role { r + 1 }\n"),
        ("main.dmc", r#"use "lib.dmc"
type Role = i64
fn main() -> nil {
    print(libbump(255) as i64)
    let r: Role = 255
    print((r + 1) as i64)
    nil
}
"#),
    ], "main.dmc");
    let so = String::from_utf8_lossy(&out.stdout).to_string();
    let se = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "[xmod_narrow] exited {:?}: {se}", out.status.code());
    assert_eq!(so, "0\n256\n", "[xmod_narrow] stdout (stderr: {se})");
}

/// A shape-generic alias whose target is a SCALAR is still a width: a shape
/// argument cannot change one. `let r: W[3] = 255` printed 256 here while the
/// checker resolved it to `u8` and the JIT (#622) refuses it outright.
#[test]
fn a_shape_generic_alias_over_a_scalar_wraps() {
    let src = r#"type W[N] = u8
fn bump(r: W[3]) -> W[3] { r + 1 }
fn main() -> nil {
    let r: W[3] = 255
    print((r + 1) as i64)
    print(bump(255) as i64)
    nil
}
"#;
    assert_run_prints("shape_generic_scalar", src, "0\n0\n");
}
