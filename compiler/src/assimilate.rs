//! `dmc assimilate` — industrialize hand-written port wrappers (ASSIMILATE.md).
//!
//! Reads a JSON descriptor of a foreign runtime's callable surface and emits a
//! demoniC module of port-call wrappers — the boilerplate of
//! `examples/port_python.dmc`, generated instead of typed. The generator is a
//! boundary factory (ASSIMILATE.md §2): it emits ordinary demoniC that calls
//! across the port ABI; it never inlines foreign semantics.
//!
//! Deterministic: the same descriptor produces a byte-identical module, with
//! functions emitted in descriptor order (ASSIMILATE.md §6).
//!
//! This is the §5.1 python port-wrapper altitude. Per ASSIMILATE.md §9, typed
//! JSON decode does not exist yet, so wrappers return `(str, str)` — the
//! canonical-JSON result string and an Err tag — the same shape the
//! hand-written example works with.

use crate::interp::{json_decode_str, Value};

/// Demon-side types the JSON value boundary (PORTS.md §3) carries as a scalar
/// argument. A param of any other type is unmappable and skips its fn (§7).
const MAPPABLE: &[&str] = &["i64", "f64", "f32", "bool", "str"];

/// Introspect a live runtime and emit a draft descriptor (ASSIMILATE.md §3).
/// Only `python` is wired. Python exposes function names and arity but rarely
/// types, so params resolve from annotations, then default values, else `"?"` —
/// the reviewed last mile a dynamic language cannot fill for us.
const INTROSPECT_PY: &str = r#"
import sys, json, inspect, importlib
mod_name = sys.argv[1]
mod = importlib.import_module(mod_name)
PYMAP = {bool: "bool", int: "i64", float: "f64", str: "str"}
STRMAP = {"bool": "bool", "int": "i64", "float": "f64", "str": "str"}
def infer(p):
    a = p.annotation
    if a in PYMAP: return PYMAP[a]
    if isinstance(a, str) and a in STRMAP: return STRMAP[a]
    d = p.default
    if d is not inspect._empty and type(d) in PYMAP: return PYMAP[type(d)]
    return "?"
fns = []
for name in sorted(dir(mod)):
    if name.startswith("_"): continue
    obj = getattr(mod, name)
    if not callable(obj): continue
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        continue
    params, ok = [], True
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.KEYWORD_ONLY):
            ok = False
            break
        params.append({"name": p.name, "ty": infer(p)})
    if not ok: continue
    fns.append({"name": mod_name.replace(".", "_") + "_" + name,
                "call": mod_name + "." + name, "params": params})
print(json.dumps({"runtime": "python", "module": mod_name, "fns": fns}))
"#;

/// Introspect `runtime`'s `module` and return a draft descriptor as JSON text.
pub fn introspect(runtime: &str, module: &str) -> Result<String, String> {
    if runtime != "python" {
        return Err(format!(
            "introspection is only wired for `python` (PORTS.md §7.1); `{}` has no \
             introspector — write a descriptor by hand", runtime));
    }
    let dotted_ident = !module.is_empty() && module.split('.').all(|seg| is_ident(seg));
    if !dotted_ident {
        return Err(format!("`{}` is not a valid python module path", module));
    }
    let out = std::process::Command::new("python3")
        .args(["-c", INTROSPECT_PY, module])
        .output()
        .map_err(|e| format!("could not run python3: {}", e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let last = err.lines().last().unwrap_or("python introspection failed");
        return Err(format!("python introspection of `{}` failed: {}", module, last.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Runtimes with a port wired in the interpreter today (PORTS.md §7.1). Bindings
/// for any other runtime still generate and type-check — the boundary factory is
/// runtime-parametric — but cannot execute until that runtime's port lands.
const WIRED: &[&str] = &["python"];

/// Generate the wrapper module from a JSON descriptor. Returns the module text
/// on success; an `Err(msg)` is a lowercase, one-line diagnostic (SPEC §8 voice).
pub fn generate(descriptor_src: &str) -> Result<String, String> {
    let v = json_decode_str(descriptor_src)
        .map_err(|e| format!("descriptor is not valid JSON: {}", e))?;
    let obj = as_map(&v).ok_or("descriptor must be a JSON object")?;

    let runtime = get_str(&obj, "runtime")
        .ok_or("descriptor: `runtime` (str) is required")?;
    // The runtime names a `Port[L]` type, so it must be a comptime identifier
    // (SPEC §3.11) — otherwise the emitted `Port[L]` would not even parse.
    if !is_ident(&runtime) {
        return Err(format!(
            "runtime `{}` is not a valid identifier — it names the `Port[L]` type \
             (SPEC §3.11)", runtime));
    }
    let wired = WIRED.contains(&runtime.as_str());
    let fns = get_list(&obj, "fns")
        .ok_or("descriptor: `fns` (array) is required")?;

    let mut out = String::new();
    out.push_str(&format!(
        "# generated by `dmc assimilate` — do not edit by hand.\n\
         # runtime: {rt}. each wrapper returns (str, str): the canonical-JSON\n\
         # result and an Err tag (PORTS.md §6). fix the descriptor and regenerate.\n\
         # descriptor fnv1a: {hash}\n", rt = runtime, hash = fnv1a_hex(descriptor_src)));
    if !wired {
        // Honest provenance: the bindings are valid demoniC, but no port for this
        // runtime exists yet, so `port_open(\"<rt>\")` will report `port-open`.
        out.push_str(&format!(
            "# note: no Port[{rt}] runtime is wired yet (PORTS.md §7) — these\n\
             # bindings type-check but will not execute until one lands.\n", rt = runtime));
        eprintln!("assimilate: note: no `{}` port is wired yet — bindings generate \
                   and check but cannot execute until a Port[{}] runtime lands \
                   (PORTS.md §7)", runtime, runtime);
    }

    let mut emitted = 0usize;
    for (i, f) in fns.iter().enumerate() {
        let fo = as_map(f).ok_or_else(|| format!("fns[{}] must be an object", i))?;
        let name = get_str(&fo, "name")
            .ok_or_else(|| format!("fns[{}]: `name` (str) is required", i))?;
        let call = get_str(&fo, "call")
            .ok_or_else(|| format!("fns[{}] (`{}`): `call` (str) is required", i, name))?;
        // The name becomes `fn <name>`, so it must be an identifier — skip and
        // report rather than emit a module that will not parse (§7).
        if !is_ident(&name) {
            eprintln!("assimilate: skipped `{}`: not a valid identifier for a fn name", name);
            out.push_str(&format!("\n# skipped `{}`: not a valid demoniC identifier\n", name));
            continue;
        }
        let params = match fo.get("params") {
            Some(Value::List(xs)) => xs.as_ref().clone(),
            None => Vec::new(),
            _ => return Err(format!("fns[{}] (`{}`): `params` must be an array", i, name)),
        };

        // Resolve (name, ty) pairs; a bad name or unmappable type skips the whole
        // fn (§7) rather than emitting something that will not check.
        let mut ps: Vec<(String, String)> = Vec::new();
        let mut unmappable: Option<String> = None;
        let mut bad_param: Option<String> = None;
        for (j, p) in params.iter().enumerate() {
            let po = as_map(p)
                .ok_or_else(|| format!("fns[{}].params[{}] must be an object", i, j))?;
            let pn = get_str(&po, "name")
                .ok_or_else(|| format!("fns[{}].params[{}]: `name` is required", i, j))?;
            // A param name is emitted as an identifier and must not collide with
            // the `__`-sigil generated locals (emit_wrapper).
            if !is_ident(&pn) || pn.starts_with("__") {
                bad_param = Some(pn);
                break;
            }
            let pt = get_str(&po, "ty")
                .ok_or_else(|| format!("fns[{}].params[{}] (`{}`): `ty` is required", i, j, pn))?;
            if !MAPPABLE.contains(&pt.as_str()) { unmappable = Some(pt); break; }
            ps.push((pn, pt));
        }
        if let Some(bp) = bad_param {
            eprintln!("assimilate: skipped `{}`: param `{}` is not a usable identifier", name, bp);
            out.push_str(&format!(
                "\n# skipped `{}`: param `{}` is not a usable demoniC identifier\n", name, bp));
            continue;
        }
        if let Some(bt) = unmappable {
            // `?` is the introspector's "could not infer" marker (§3) — a request
            // to supply a type, not a genuinely unmappable one. Report each
            // distinctly so a raw draft descriptor still yields partial bindings.
            let (reason, note) = if bt == "?" {
                ("needs a type — introspection could not infer it; set `ty` in the descriptor",
                 format!("\n# needs types `{}`: set the `?` params in the descriptor and regenerate\n", name))
            } else {
                ("has no JSON value mapping (PORTS.md §3)",
                 format!("\n# skipped `{}`: param type `{}` is unmappable (PORTS.md §3)\n", name, bt))
            };
            eprintln!("assimilate: skipped `{}`: param type `{}` {}", name, bt, reason);
            out.push_str(&note);
            continue;
        }

        out.push('\n');
        out.push_str(&emit_wrapper(&runtime, &name, &call, &ps));
        emitted += 1;
    }

    if emitted == 0 {
        return Err("descriptor produced no bindings — `fns` was empty or every \
                    entry was skipped".to_string());
    }
    Ok(out)
}

/// One wrapper: open handle in, `(str, Err)` out, the argument vector built as a
/// `list` and JSON-encoded per PORTS.md §2.
fn emit_wrapper(runtime: &str, name: &str, call: &str, params: &[(String, String)]) -> String {
    // Generated locals carry a `__` sigil so they cannot collide with a
    // descriptor param name (params are validated to be plain identifiers, §4),
    // and the sigil reads as "machine-generated" — matching the header's
    // do-not-edit note.
    let mut s = String::new();
    s.push_str("fn ");
    s.push_str(name);
    s.push_str(&format!("(__port: Port[{}]", runtime));
    for (pn, pt) in params {
        s.push_str(&format!(", {}: {}", pn, pt));
    }
    s.push_str(") -> (str, str) {\n");
    s.push_str("    let __args = list()\n");
    for (pn, _) in params {
        s.push_str(&format!("    let __args = list_push(__args, {})\n", pn));
    }
    s.push_str(&format!(
        "    let (__out, __err) = port_call(__port, {}, json_encode(__args))\n", dmc_str_lit(call)));
    s.push_str("    if __err != nil { return (\"\", __err) }\n");
    s.push_str("    (__out, nil)\n}\n");
    s
}

// ── descriptor helpers ──────────────────────────────────────────────────────

fn as_map(v: &Value) -> Option<std::collections::HashMap<String, Value>> {
    match v {
        Value::Map(m) => Some(m.borrow().clone()),
        _ => None,
    }
}

fn get_str(obj: &std::collections::HashMap<String, Value>, key: &str) -> Option<String> {
    match obj.get(key) {
        Some(Value::Str(s)) => Some(s.clone()),
        _ => None,
    }
}

fn get_list(obj: &std::collections::HashMap<String, Value>, key: &str) -> Option<Vec<Value>> {
    match obj.get(key) {
        Some(Value::List(xs)) => Some(xs.as_ref().clone()),
        _ => None,
    }
}

/// A comptime identifier: `[A-Za-z_][A-Za-z0-9_]*`. The runtime name must be one
/// because it becomes the `L` in the emitted `Port[L]` type (SPEC §3.11).
fn is_ident(s: &str) -> bool {
    let mut cs = s.chars();
    match cs.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    cs.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Emit a demoniC string literal, escaping the two characters the lexer's string
/// escape set treats specially in a way that would break the literal.
fn dmc_str_lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// FNV-1a over the raw descriptor bytes — a stable, dependency-free provenance
/// stamp for the generated header (ASSIMILATE.md §6). Stability across runs is
/// what the header needs; it is not a cryptographic hash.
fn fnv1a_hex(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:016x}", h)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATH: &str = r#"{
        "runtime": "python",
        "fns": [
            {"name": "gcd", "call": "math.gcd",
             "params": [{"name":"a","ty":"i64"}, {"name":"b","ty":"i64"}]},
            {"name": "sqrt", "call": "math.sqrt",
             "params": [{"name":"x","ty":"f64"}]},
            {"name": "pytime", "call": "time.time", "params": []}
        ]
    }"#;

    /// The generated module must type-check with zero errors — the real proof
    /// that assimilate emits valid demoniC, not just plausible text.
    fn check_errs(src: &str) -> Vec<String> {
        let tokens = crate::lexer::Lexer::new(src).tokenize().expect("lex");
        let program = crate::parser::Parser::new(tokens).parse_program().expect("parse");
        let mut c = crate::check::Checker::new();
        c.check_program(&program, None);
        c.errors.iter().map(|e| e.msg.clone()).collect()
    }

    #[test]
    fn generated_module_type_checks() {
        let module = generate(MATH).expect("generate");
        // A wrapper per fn, in descriptor order, with the open-handle signature.
        assert!(module.contains("fn gcd(__port: Port[python], a: i64, b: i64) -> (str, str)"),
            "module:\n{}", module);
        assert!(module.contains("fn sqrt(__port: Port[python], x: f64) -> (str, str)"));
        assert!(module.contains("fn pytime(__port: Port[python]) -> (str, str)"));
        assert!(module.contains("port_call(__port, \"math.gcd\", json_encode(__args))"));
        // The generated demoniC must check clean (add a trivial main in case a
        // program entry point is expected; unused wrappers must not error).
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(generate(MATH).unwrap(), generate(MATH).unwrap());
    }

    #[test]
    fn unwired_runtime_generates_with_a_note() {
        // A runtime with no port wired (mojo) still generates valid, checkable
        // bindings — the boundary factory is runtime-parametric — carrying a
        // note that they cannot execute yet.
        let d = r#"{
            "runtime": "mojo",
            "fns": [
                {"name": "msqrt", "call": "math.sqrt",
                 "params": [{"name":"x","ty":"f64"}]}
            ]
        }"#;
        let module = generate(d).expect("generate");
        assert!(module.contains("fn msqrt(__port: Port[mojo], x: f64) -> (str, str)"),
            "module:\n{}", module);
        assert!(module.contains("no Port[mojo] runtime is wired yet"), "module:\n{}", module);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn invalid_runtime_identifier_is_rejected() {
        // The runtime becomes `L` in `Port[L]`, so it must be an identifier.
        let d = r#"{"runtime": "mojo-lang", "fns": []}"#;
        let e = generate(d).unwrap_err();
        assert!(e.contains("valid identifier"), "got: {}", e);
    }

    #[test]
    fn unmappable_param_type_skips_the_fn() {
        // A tensor argument has no JSON value mapping (PORTS.md §3): the fn is
        // skipped with a comment, and a mappable sibling still emits.
        let d = r#"{
            "runtime": "python",
            "fns": [
                {"name": "bad", "call": "np.sum",
                 "params": [{"name":"t","ty":"Tensor[f32, [4]]"}]},
                {"name": "good", "call": "math.floor",
                 "params": [{"name":"x","ty":"f64"}]}
            ]
        }"#;
        let module = generate(d).expect("generate");
        assert!(module.contains("# skipped `bad`"), "module:\n{}", module);
        assert!(module.contains("fn good(__port: Port[python], x: f64)"));
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn param_named_like_a_generated_local_still_checks() {
        // Regression: a param named `args`/`out`/`e` must not collide with the
        // wrapper's generated locals (now `__`-sigiled). It type-checks and the
        // param reaches the arg vector.
        let d = r#"{
            "runtime": "python",
            "fns": [
                {"name": "f", "call": "m.f",
                 "params": [{"name":"args","ty":"i64"}, {"name":"out","ty":"str"},
                            {"name":"e","ty":"bool"}]}
            ]
        }"#;
        let module = generate(d).expect("generate");
        assert!(module.contains("let __args = list_push(__args, args)"), "module:\n{}", module);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn invalid_fn_or_param_name_skips_with_report() {
        // A hand-written descriptor with a non-identifier name must not emit a
        // module that fails to parse — skip and report instead (§7).
        let d = r#"{
            "runtime": "python",
            "fns": [
                {"name": "bad-name", "call": "m.a", "params": []},
                {"name": "reserved", "call": "m.b", "params": [{"name":"__x","ty":"i64"}]},
                {"name": "ok", "call": "m.c", "params": [{"name":"x","ty":"i64"}]}
            ]
        }"#;
        let module = generate(d).expect("generate");
        assert!(module.contains("# skipped `bad-name`"), "module:\n{}", module);
        assert!(module.contains("# skipped `reserved`"), "module:\n{}", module);
        assert!(module.contains("fn ok(__port: Port[python], x: i64)"));
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn empty_binding_set_is_an_error() {
        assert!(generate(r#"{"runtime":"python","fns":[]}"#).is_err());
    }

    #[test]
    fn malformed_json_is_reported() {
        let e = generate("{not json").unwrap_err();
        assert!(e.contains("not valid JSON"), "got: {}", e);
    }

    // ── introspection (python3 is a dev prerequisite, as in the port tests) ──

    #[test]
    fn introspect_python_module_yields_a_descriptor() {
        // `fractions` is pure-python with signatures and defaults, so it
        // introspects into a well-formed descriptor across python versions.
        let desc = introspect("python", "fractions").expect("introspect");
        let v = json_decode_str(&desc).expect("valid JSON descriptor");
        let obj = as_map(&v).expect("object");
        assert_eq!(get_str(&obj, "runtime").as_deref(), Some("python"));
        assert!(get_list(&obj, "fns").map(|f| !f.is_empty()).unwrap_or(false),
            "expected some fns, got: {}", desc);
    }

    #[test]
    fn introspected_descriptor_generates_partial_bindings() {
        // End to end: introspect, then generate. `statistics` has default-typed
        // params (e.g. NormalDist(mu=0.0, sigma=1.0)) so at least one fn types
        // fully and emits; `?`-param fns are skipped with a needs-types note.
        let desc = introspect("python", "statistics").expect("introspect");
        let module = generate(&desc).expect("generate");
        assert!(module.contains("fn statistics_"), "module:\n{}", module);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn introspect_rejects_bad_module_and_runtime() {
        assert!(introspect("python", "no.such.module.xyz").is_err());
        assert!(introspect("python", "bad-name").is_err());
        assert!(introspect("lua", "math").unwrap_err().contains("only wired for `python`"));
    }
}
