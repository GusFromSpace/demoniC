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
//! This is the §5.1 python port-wrapper altitude. A wrapper returns the
//! descriptor's declared `ret` decoded through the typed JSON decode family
//! (PORTS.md §3.1) — `(i64, Err)`, `(f64, Err)`, `(str, Err)`, `(bool, Err)`,
//! `(list, Err)`. A descriptor with no `ret`, or `ret: "?"`, keeps the untyped
//! shape: `(str, Err)` carrying the raw canonical-JSON result.

use crate::interp::{json_parse, Value, DECODE_TYPES};

/// Demon-side types the JSON value boundary (PORTS.md §3) carries as a scalar
/// argument. A param of any other type is unmappable and skips its fn (§7).
const MAPPABLE: &[&str] = &["i64", "f64", "f32", "bool", "str"];

/// What a descriptor's `ret` becomes in the emitted wrapper: the declared
/// demoniC type, the decode primitive that enforces it, and T's zero for the
/// error path. `None` is the untyped return — the raw canonical-JSON `str`.
///
/// `f32` is deliberately absent. JSON numbers carry no width and the decode
/// family lands them in `f64`; narrowing is an explicit `as f32` at the call
/// site, not something the boundary does behind the caller's back. It stays a
/// mappable *param* type because encoding an f32 argument loses nothing.
struct RetDecode {
    ty: &'static str,
    decoder: &'static str,
    zero: &'static str,
}

/// Resolve a descriptor `ret` against the interpreter's decode family, so the
/// two cannot drift: a type the interpreter does not decode is not emittable.
fn ret_decode(ty: &str) -> Option<RetDecode> {
    let ty = *DECODE_TYPES.iter().find(|t| **t == ty)?;
    let (decoder, zero) = match ty {
        "i64"  => ("json_decode_i64",  "0"),
        "f64"  => ("json_decode_f64",  "0.0"),
        "bool" => ("json_decode_bool", "false"),
        "list" => ("json_decode_list", "list()"),
        "str"  => ("json_decode_str",  "\"\""),
        // DECODE_TYPES gained a member this arm does not know how to emit.
        _ => return None,
    };
    Some(RetDecode { ty, decoder, zero })
}

/// Introspect a live runtime and emit a draft descriptor (ASSIMILATE.md §3).
/// Only `python` is wired. Python exposes function names and arity but rarely
/// types, so params resolve from annotations, then default values, else `"?"` —
/// the reviewed last mile a dynamic language cannot fill for us.
///
/// The surface is filtered by the callable's *kind*, not by its result type: a
/// plain python function, a builtin, or a bound method of a plain function
/// (`random.randint`, which the port resolves by the same dotted path) binds;
/// everything else is dropped. A class is a callable whose call returns an
/// instance, which never round-trips, so it is dropped rather than bound into a
/// wrapper that can only fail at call time (#496); so is any other callable (a
/// `functools.partial`, a callable instance), whose behaviour is equally
/// unknowable from `dir()`.
///
/// What a bound callable *returns* is not filtered, and cannot be: python's
/// annotations are optional and unenforced, so `decimal.getcontext` binds and
/// its call reports `port-call` when the runtime hands back an object with no
/// JSON mapping (PORTS.md §3). `ret` records the annotation when there is one
/// and `?` when there is not — the descriptor's honest "unknown" marker.
///
/// Every drop is recorded in the descriptor's `dropped` array — the introspector
/// never omits a callable silently, matching the generator's skip reports (§7).
/// The demon side prints that array as the user-facing report (`drop_report`),
/// so there is one account of what was refused, rendered twice. Each entry is
/// one line: the callable's name and the reason are whitespace-collapsed, so a
/// newline in either cannot split a report into two.
const INTROSPECT_PY: &str = r#"
import sys, json, re, inspect, importlib
try:
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass
mod_name = sys.argv[1]
mod = importlib.import_module(mod_name)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
PYMAP = {bool: "bool", int: "i64", float: "f64", str: "str"}
STRMAP = {"bool": "bool", "int": "i64", "float": "f64", "str": "str"}
# A return type additionally admits `list` — the typed decode family carries it
# (PORTS.md §3.1) even though it is not a scalar argument.
RETMAP = {**PYMAP, list: "list"}
RETSTR = {**STRMAP, "list": "list"}
def infer(p):
    try:
        a = p.annotation
        if isinstance(a, str):
            if a in STRMAP: return STRMAP[a]
        elif a in PYMAP:
            return PYMAP[a]
        d = p.default
        if d is not inspect._empty and type(d) in PYMAP: return PYMAP[type(d)]
    except Exception:
        pass
    return "?"
def infer_ret(sig):
    # Only the return annotation. There is no default value to fall back on, so
    # an unannotated callable stays `?` — the wrapper hands back the raw
    # canonical-JSON str rather than the tool inventing a contract.
    a = sig.return_annotation
    try:
        if a in RETMAP: return RETMAP[a]
    except TypeError:
        pass
    if isinstance(a, str) and a in RETSTR: return RETSTR[a]
    return "?"
dropped = []
def one_line(s):
    return " ".join(str(s).split())
def drop(path, reason):
    # The drop goes into the descriptor only. Nothing here writes to stderr:
    # the report the user reads is rendered on the demon side from this array,
    # so the two accounts cannot disagree and no line of module output can
    # impersonate one (see `drop_report`).
    dropped.append({"call": one_line(path), "reason": one_line(reason)})
def type_name(obj):
    t = type(obj)
    m = getattr(t, "__module__", "")
    n = getattr(t, "__qualname__", None) or getattr(t, "__name__", "?")
    return n if m in ("", "builtins") else m + "." + n
def bindable(obj):
    # A plain python function or a builtin. A bound method reached through a
    # module-level instance (`random.randint`) is a plain function too:
    # `signature` already elides the receiver, and the port resolves the same
    # dotted path, so binding it is sound.
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return True
    return inspect.ismethod(obj) and inspect.isfunction(obj.__func__)
fns = []
for name in sorted(dir(mod)):
    if name.startswith("_"): continue
    path = mod_name + "." + name
    try:
        obj = getattr(mod, name)
    except Exception as e:
        drop(path, "reading the attribute raised %s (%s) — nothing to introspect" % (type(e).__name__, e))
        continue
    if not callable(obj): continue
    if not IDENT.match(name):
        drop(path, "not a valid demoniC identifier for a fn name")
        continue
    if inspect.isclass(obj):
        drop(path, "a class — its call returns an instance, which has no JSON value mapping (PORTS.md §3)")
        continue
    if not bindable(obj):
        drop(path, "a %s — only a plain function, a builtin, or a bound method is bound; another callable's result may not round-trip through JSON (PORTS.md §3)" % type_name(obj))
        continue
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        drop(path, "no introspectable signature (a C builtin) — write the descriptor entry by hand")
        continue
    params, bad = [], None
    for p in sig.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            bad = "variadic parameter `*%s` — the argument vector is fixed-arity (PORTS.md §2)" % p.name
            break
        if p.kind == p.VAR_KEYWORD:
            bad = "variadic parameter `**%s` — the argument vector is fixed-arity (PORTS.md §2)" % p.name
            break
        if p.kind == p.KEYWORD_ONLY:
            bad = "keyword-only parameter `%s` — the argument vector is positional (PORTS.md §2)" % p.name
            break
        params.append({"name": p.name, "ty": infer(p)})
    if bad is not None:
        drop(path, bad)
        continue
    fns.append({"name": mod_name.replace(".", "_") + "_" + name,
                "call": path, "params": params, "ret": infer_ret(sig)})
print(json.dumps({"schema": 1, "runtime": "python", "module": mod_name,
                  "fns": fns, "dropped": dropped}))
"#;

/// Run the python introspector over `module` and return its `(stdout, stderr)`.
///
/// `sys_path`, when set, *becomes* the child's whole `PYTHONPATH` — it replaces
/// the inherited value rather than prepending to it, so a test importing a
/// fixture module gets the same search path on every machine. Production passes
/// `None` and the child inherits the environment untouched. Splitting this out
/// keeps the child's stderr assertable without capturing the test process's own.
fn run_introspector(module: &str, sys_path: Option<&str>) -> Result<(String, String), String> {
    let mut cmd = std::process::Command::new("python3");
    cmd.args(["-c", INTROSPECT_PY, module]);
    if let Some(p) = sys_path {
        cmd.env("PYTHONPATH", p);
    }
    let out = cmd.output().map_err(|e| format!("could not run python3: {}", e))?;
    let report = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        let last = report.lines().last().unwrap_or("python introspection failed");
        return Err(format!("python introspection of `{}` failed: {}", module, last.trim()));
    }
    Ok((String::from_utf8_lossy(&out.stdout).trim().to_string(), report))
}

/// The §3 drop report as the user reads it, rendered from the descriptor's own
/// `dropped` array.
///
/// The child never writes a report line itself. Deriving the report here means
/// the two accounts of what was refused — the stderr lines and the descriptor —
/// are the same data by construction, and it settles provenance: since no
/// `assimilate:` line can originate in the child, every line the child *does*
/// put on stderr is the module talking and is labelled as such unconditionally.
/// Prefix-matching the child's output would have let a module that prints
/// `assimilate: skipped ...` at import time forge a drop that no descriptor
/// records.
fn drop_report(descriptor: &str) -> Result<Vec<String>, String> {
    let v = json_parse(descriptor)
        .map_err(|e| format!("introspection produced invalid JSON: {}", e))?;
    let obj = as_map(&v).ok_or("introspection produced a non-object descriptor")?;
    dropped_entries(&obj)?.iter().enumerate().map(|(i, d)| {
        let dobj = as_map(d).ok_or_else(|| format!("dropped[{}] must be an object", i))?;
        let call = get_str(&dobj, "call")
            .ok_or_else(|| format!("dropped[{}]: `call` (str) is required", i))?;
        let reason = get_str(&dobj, "reason")
            .ok_or_else(|| format!("dropped[{}] (`{}`): `reason` (str) is required", i, call))?;
        // One line per drop: `comment_text` turns any control character into a
        // space, so a newline in a name or a reason cannot fake a second entry.
        Ok(format!("assimilate: skipped `{}`: {}", comment_text(&call), comment_text(&reason)))
    }).collect()
}

/// Whatever the child put on stderr, labelled with its provenance.
///
/// Every line gets the label, unconditionally. The introspector writes nothing
/// to stderr, so anything on it came from the module — an import warning, a
/// deprecation notice, or a line crafted to read like one of ours. Labelling by
/// where the output came from rather than by what it looks like is what makes
/// the label mean something.
fn module_output(child_stderr: &str) -> Vec<String> {
    child_stderr.lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .map(|l| format!("assimilate: python: {}", l))
        .collect()
}

/// Introspect `runtime`'s `module` and return a draft descriptor as JSON text.
///
/// Both halves of the child's output reach the user's stderr: whatever the
/// module itself printed, labelled with its provenance, and then the drop
/// report (§3) — the only account of what the live surface refused to bind.
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
    let (descriptor, child_stderr) = run_introspector(module, None)?;
    for line in module_output(&child_stderr) {
        eprintln!("{}", line);
    }
    for line in drop_report(&descriptor)? {
        eprintln!("{}", line);
    }
    Ok(descriptor)
}

/// Runtimes with a port wired in the interpreter today (PORTS.md §7.1). Bindings
/// for any other runtime still generate and type-check — the boundary factory is
/// runtime-parametric — but cannot execute until that runtime's port lands.
const WIRED: &[&str] = &["python"];

/// The descriptor schema version this reader knows (ASSIMILATE.md §4). The
/// format is on the freeze list; the version field is what lets it evolve
/// without a flag day once frozen.
const SCHEMA: i64 = 1;

/// Resolve a descriptor's `schema` (ASSIMILATE.md §4). Absent means 1 — every
/// descriptor written before the field existed. A version this reader does not
/// know is rejected whole rather than half-read; unknown *fields* within a
/// known schema are ignored, so additive evolution never bumps the version.
fn schema_version(obj: &std::collections::HashMap<String, Value>) -> Result<i64, String> {
    match obj.get("schema") {
        None => Ok(SCHEMA),
        Some(Value::Int(n, _)) if *n == SCHEMA => Ok(*n),
        Some(Value::Int(n, _)) => Err(format!(
            "descriptor: schema {} is unknown — this dmc reads schema {}; regenerate \
             the descriptor or upgrade dmc", n, SCHEMA)),
        Some(_) => Err("descriptor: `schema` must be an integer".to_string()),
    }
}

/// Generate the wrapper module from a JSON descriptor. Returns the module text
/// on success; an `Err(msg)` is a lowercase, one-line diagnostic (SPEC §8 voice).
pub fn generate(descriptor_src: &str) -> Result<String, String> {
    let v = json_parse(descriptor_src)
        .map_err(|e| format!("descriptor is not valid JSON: {}", e))?;
    let obj = as_map(&v).ok_or("descriptor must be a JSON object")?;

    // Reject an unknown schema before reading anything else: a field this
    // version cannot see may change what the ones it can see mean.
    schema_version(&obj)?;

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
         # runtime: {rt}. each wrapper returns (T, str): the descriptor's `ret`\n\
         # decoded from the canonical-JSON result, and an Err tag (PORTS.md §6).\n\
         # an untyped `ret` returns the raw canonical-JSON str. fix the\n\
         # descriptor and regenerate.\n\
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

    // The introspector records what it refused to bind in `dropped` (§3). Carry
    // it into the module so the generated file itself accounts for the missing
    // callables — the stderr report scrolls away, the comment is committed.
    let dropped = dropped_entries(&obj)?;
    for (i, d) in dropped.iter().enumerate() {
        let dobj = as_map(d).ok_or_else(|| format!("dropped[{}] must be an object", i))?;
        let call = get_str(&dobj, "call")
            .ok_or_else(|| format!("dropped[{}]: `call` (str) is required", i))?;
        let reason = get_str(&dobj, "reason")
            .ok_or_else(|| format!("dropped[{}] (`{}`): `reason` (str) is required", i, call))?;
        out.push_str(&format!("# dropped `{}`: {}\n",
            comment_text(&call), comment_text(&reason)));
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
            let name = comment_text(&name);
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
            let bp = comment_text(&bp);
            eprintln!("assimilate: skipped `{}`: param `{}` is not a usable identifier", name, bp);
            out.push_str(&format!(
                "\n# skipped `{}`: param `{}` is not a usable demoniC identifier\n", name, bp));
            continue;
        }
        if let Some(bt) = unmappable {
            let bt = comment_text(&bt);
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

        // `ret` is optional. Absent or `?` keeps the untyped shape — the raw
        // canonical-JSON str — so an introspected descriptor that could not
        // recover a return type still yields a working wrapper. A `ret` naming
        // a type the decode family does not carry skips the fn: emitting a
        // wrapper whose declared type nothing enforces is the documentation-
        // not-a-contract failure typed decode exists to end.
        let ret = match fo.get("ret") {
            None | Some(Value::Nil) => None,
            Some(Value::Str(t)) if t == "?" => None,
            Some(Value::Str(t)) => match ret_decode(t) {
                Some(rd) => Some(rd),
                None => {
                    eprintln!("assimilate: skipped `{}`: return type `{}` has no typed \
                               decode (PORTS.md §3.1); use one of {}, or `?` for the raw \
                               canonical-JSON str", name, t, DECODE_TYPES.join("/"));
                    out.push_str(&format!(
                        "\n# skipped `{}`: return type `{}` has no typed decode (PORTS.md §3.1)\n",
                        name, t));
                    continue;
                }
            },
            Some(_) => return Err(format!("fns[{}] (`{}`): `ret` must be a str", i, name)),
        };

        out.push('\n');
        out.push_str(&emit_wrapper(&runtime, &name, &call, &ps, ret.as_ref()));
        emitted += 1;
    }

    if emitted == 0 {
        // A filtered-to-nothing module is the common way here (§3): say how many
        // callables the introspector already refused, so the empty result reads
        // as a filtered surface and not as a module with nothing in it.
        let tail = match dropped.len() {
            0 => String::new(),
            1 => " (1 more callable was dropped by the introspector)".to_string(),
            n => format!(" ({} more callables were dropped by the introspector)", n),
        };
        return Err(format!("descriptor produced no bindings — `fns` was empty or every \
                            entry was skipped{}", tail));
    }
    Ok(out)
}

/// One wrapper: open handle in, `(T, Err)` out, the argument vector built as a
/// `list` and JSON-encoded per PORTS.md §2. With a `ret` the canonical-JSON
/// result runs through the matching typed decode primitive; without one the
/// wrapper hands the raw result str back, as it did before typed decode.
fn emit_wrapper(runtime: &str, name: &str, call: &str, params: &[(String, String)],
                ret: Option<&RetDecode>) -> String {
    // Generated locals carry a `__` sigil so they cannot collide with a
    // descriptor param name (params are validated to be plain identifiers, §4),
    // and the sigil reads as "machine-generated" — matching the header's
    // do-not-edit note.
    let (ty, zero) = match ret {
        Some(rd) => (rd.ty, rd.zero),
        None => ("str", "\"\""),
    };
    let mut s = String::new();
    s.push_str("fn ");
    s.push_str(name);
    s.push_str(&format!("(__port: Port[{}]", runtime));
    for (pn, pt) in params {
        s.push_str(&format!(", {}: {}", pn, pt));
    }
    s.push_str(&format!(") -> ({}, str) {{\n", ty));
    s.push_str("    let __args = list()\n");
    for (pn, _) in params {
        s.push_str(&format!("    let __args = list_push(__args, {})\n", pn));
    }
    s.push_str(&format!(
        "    let (__out, __err) = port_call(__port, {}, json_encode(__args))\n", dmc_str_lit(call)));
    s.push_str(&format!("    if __err != nil {{ return ({}, __err) }}\n", zero));
    match ret {
        // A `decode-type` here means the descriptor promised a type the runtime
        // did not return; it surfaces as an Err tag, never a coerced value.
        Some(rd) => {
            s.push_str(&format!("    let (__val, __derr) = {}(__out)\n", rd.decoder));
            s.push_str(&format!("    if __derr != nil {{ return ({}, __derr) }}\n", zero));
            s.push_str("    (__val, nil)\n}\n");
        }
        None => s.push_str("    (__out, nil)\n}\n"),
    }
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

/// The descriptor's optional `dropped` array (§4) — the introspector's account
/// of the callables it refused to bind. Absent is fine (a hand-written
/// descriptor drops nothing); present-but-not-an-array is a descriptor error,
/// not something to ignore.
fn dropped_entries(obj: &std::collections::HashMap<String, Value>) -> Result<Vec<Value>, String> {
    match obj.get("dropped") {
        None => Ok(Vec::new()),
        Some(Value::List(xs)) => Ok(xs.as_ref().clone()),
        Some(_) => Err("descriptor: `dropped` must be an array".to_string()),
    }
}

/// One line of `#` comment text from untrusted descriptor input. A newline in a
/// name or a drop reason would end the comment and let the rest of the string
/// land in the module as code, so every control character collapses to a space.
fn comment_text(s: &str) -> String {
    let cleaned: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    cleaned.trim_end().to_string()
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
        "schema": 1,
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

    /// Every `fn` the generated module actually declares. Text assertions cannot
    /// tell a wrapper from the same words inside a `#` comment; the parsed item
    /// list can.
    fn declared_fns(src: &str) -> Vec<String> {
        let tokens = crate::lexer::Lexer::new(src).tokenize().expect("lex");
        let program = crate::parser::Parser::new(tokens).parse_program().expect("parse");
        program.items.iter().filter_map(|it| match it {
            crate::ast::Item::Fn(f) => Some(f.name.clone()),
            _ => None,
        }).collect()
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

    /// Every descriptor `ret` the typed decode family carries, plus the two
    /// untyped spellings (absent, and the introspector's `?`).
    const TYPED: &str = r#"{
        "schema": 1,
        "runtime": "python",
        "fns": [
            {"name": "add",   "call": "m.add",   "ret": "i64",
             "params": [{"name":"a","ty":"i64"}, {"name":"b","ty":"i64"}]},
            {"name": "scale", "call": "m.scale", "ret": "f64",
             "params": [{"name":"x","ty":"f64"}]},
            {"name": "label", "call": "m.label", "ret": "str",
             "params": [{"name":"n","ty":"i64"}]},
            {"name": "ispos", "call": "m.ispos", "ret": "bool",
             "params": [{"name":"n","ty":"i64"}]},
            {"name": "spread","call": "m.spread","ret": "list",
             "params": [{"name":"n","ty":"i64"}]},
            {"name": "unsure","call": "m.unsure","ret": "?",
             "params": [{"name":"n","ty":"i64"}]},
            {"name": "absent","call": "m.absent",
             "params": [{"name":"n","ty":"i64"}]}
        ]
    }"#;

    #[test]
    fn generation_is_deterministic() {
        // ASSIMILATE.md §6: one descriptor maps to one byte-exact module. The
        // descriptor's objects land in `HashMap`s, whose iteration order differs
        // between instances — a generator that ever walked one instead of
        // reading by key would drift across these rounds.
        for d in [MATH, TYPED] {
            let first = generate(d).expect("generate");
            for _ in 0..16 {
                assert_eq!(generate(d).expect("generate").as_bytes(), first.as_bytes(),
                    "generation is not byte-stable for:\n{}", d);
            }
        }
        // Symbols emit in descriptor order, not sorted order (§6).
        let module = generate(MATH).unwrap();
        let at = |n: &str| module.find(n).unwrap_or_else(|| panic!("missing `{}`:\n{}", n, module));
        assert!(at("fn gcd(") < at("fn sqrt(") && at("fn sqrt(") < at("fn pytime("),
            "module:\n{}", module);
        // The provenance stamp is a pure function of the descriptor bytes.
        assert!(module.contains(&format!("# descriptor fnv1a: {}", fnv1a_hex(MATH))),
            "module:\n{}", module);
    }

    #[test]
    fn typed_ret_emits_a_typed_wrapper() {
        // ASSIMILATE.md §4/§5.1: the descriptor's `ret` drives the wrapper's
        // return type and the decode primitive that enforces it. Without this
        // the declared types are documentation, not a contract.
        let module = generate(TYPED).expect("generate");
        for (sig, decoder) in [
            ("fn add(__port: Port[python], a: i64, b: i64) -> (i64, str)", "json_decode_i64"),
            ("fn scale(__port: Port[python], x: f64) -> (f64, str)",       "json_decode_f64"),
            ("fn label(__port: Port[python], n: i64) -> (str, str)",       "json_decode_str"),
            ("fn ispos(__port: Port[python], n: i64) -> (bool, str)",      "json_decode_bool"),
            ("fn spread(__port: Port[python], n: i64) -> (list, str)",     "json_decode_list"),
        ] {
            assert!(module.contains(sig), "missing `{}`:\n{}", sig, module);
            assert!(module.contains(&format!("let (__val, __derr) = {}(__out)", decoder)),
                "missing `{}`:\n{}", decoder, module);
        }
        // T's zero rides the error path so `(T, Err)` stays well-typed.
        for zero in ["return (0, __err)", "return (0.0, __err)", "return (false, __err)",
                     "return (list(), __err)", "return (\"\", __err)"] {
            assert!(module.contains(zero), "missing `{}`:\n{}", zero, module);
        }
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn untyped_ret_stays_the_raw_json_str() {
        // `?` and an absent `ret` are the same thing: no typed decode, the raw
        // canonical-JSON result str. Unlike a `?` *param* this is not a skip —
        // the wrapper still works, it just hands back text.
        let module = generate(TYPED).expect("generate");
        for name in ["unsure", "absent"] {
            let sig = format!("fn {}(__port: Port[python], n: i64) -> (str, str)", name);
            assert!(module.contains(&sig), "missing `{}`:\n{}", sig, module);
        }
        // Exactly the five typed fns decode; the two untyped ones hand `__out`
        // straight back. Each typed wrapper mentions `__derr` three times
        // (bind, test, return), so drift in either direction trips this.
        assert_eq!(module.matches("__derr").count(), 5 * 3, "module:\n{}", module);
        assert_eq!(module.matches("    (__out, nil)\n").count(), 2, "module:\n{}", module);
    }

    #[test]
    fn undecodable_ret_skips_the_fn() {
        // A `ret` naming a type with no typed decode (f32 — JSON numbers carry
        // no width) is skipped and reported rather than emitted with a return
        // type nothing enforces. A decodable sibling still emits.
        let d = r#"{
            "runtime": "python",
            "fns": [
                {"name": "narrow", "call": "m.fabs", "ret": "f32",
                 "params": [{"name":"x","ty":"f64"}]},
                {"name": "tensor", "call": "m.zeros", "ret": "Tensor[f32, [4]]",
                 "params": [{"name":"n","ty":"i64"}]},
                {"name": "wide",   "call": "m.fabs", "ret": "f64",
                 "params": [{"name":"x","ty":"f64"}]}
            ]
        }"#;
        let module = generate(d).expect("generate");
        assert!(module.contains("# skipped `narrow`: return type `f32`"), "module:\n{}", module);
        assert!(module.contains("# skipped `tensor`: return type `Tensor[f32, [4]]`"),
            "module:\n{}", module);
        assert!(module.contains("fn wide(__port: Port[python], x: f64) -> (f64, str)"),
            "module:\n{}", module);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn generated_wrappers_propagate_with_question_mark() {
        // SPEC §4.9: `?` is legal on a `(T, Err)` call inside a fn that also
        // returns `(_, Err)`. A generated wrapper must be usable that way or
        // the fallible convention it emits is only nominal. The typed form
        // additionally unwraps to a real T — `x + y` below is integer
        // arithmetic on a port result, not string handling.
        let module = generate(TYPED).expect("generate");
        let caller = format!(r#"{}
fn typed(p: Port[python], a: i64, b: i64) -> (i64, str) {{
    let x = add(p, a, b)?
    let y = add(p, x, 100)?
    (x + y, nil)
}}

fn untyped(p: Port[python], n: i64) -> (str, str) {{
    let s = absent(p, n)?
    (s, nil)
}}

fn main() -> nil {{ nil }}
"#, module);
        assert!(check_errs(&caller).is_empty(), "errors: {:?}", check_errs(&caller));

        // And `?` is refused where the convention does not hold: a caller whose
        // return type is a bare scalar cannot swallow the wrapper's Err.
        let bad = format!("{}\nfn nope(p: Port[python], n: i64) -> i64 {{ add(p, n, n)? }}\n", module);
        let errs = check_errs(&bad);
        assert!(errs.iter().any(|e| e.contains("`?` is only legal")), "errors: {:?}", errs);
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
    fn dropped_callables_are_recorded_in_the_module() {
        // The introspector's `dropped` report (§3) rides in the descriptor and
        // must reach the generated file: a reader of the module sees why a
        // callable is missing without re-running introspection.
        // `—` (an em dash) is written escaped, as the introspector writes it
        // (`json.dumps` is ASCII-only): the generated comment carries the real
        // character, so a descriptor's prose survives into the module.
        let d = "{\"runtime\": \"python\",\n\
            \"dropped\": [\n\
              {\"call\": \"m.Klass\", \"reason\": \"a class \\u2014 its call returns an \
               instance\"},\n\
              {\"call\": \"m.varargs\", \"reason\": \"variadic parameter `*xs`\"}],\n\
            \"fns\": [{\"name\": \"ok\", \"call\": \"m.ok\",\
                      \"params\": [{\"name\":\"x\",\"ty\":\"i64\"}]}]}";
        let module = generate(d).expect("generate");
        assert!(module.contains("# dropped `m.Klass`: a class — its call returns an instance\n"),
            "module:\n{}", module);
        assert!(module.contains("# dropped `m.varargs`: variadic parameter `*xs`\n"),
            "module:\n{}", module);
        // A drop is a report, not a binding: only the mapped fn is declared.
        assert_eq!(declared_fns(&module), vec!["ok".to_string()]);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn malformed_dropped_report_is_rejected() {
        let e = generate(r#"{"runtime":"python","dropped":"nope","fns":[]}"#).unwrap_err();
        assert_eq!(e, "descriptor: `dropped` must be an array");
        let e = generate(r#"{"runtime":"python","dropped":[{"call":"m.k"}],"fns":[]}"#).unwrap_err();
        assert_eq!(e, "dropped[0] (`m.k`): `reason` (str) is required");
    }

    #[test]
    fn reported_text_cannot_inject_code_into_the_module() {
        // A descriptor is untrusted input. Every reported string — a bad fn
        // name, a bad param name, an unmappable type, a drop reason — is echoed
        // into a `#` comment, so a newline in one would close the comment and
        // land the rest in the module as code.
        let d = r#"{"runtime": "python",
            "dropped": [{"call": "m.k", "reason": "a class\nfn inject_reason() -> i64 { 1 }"}],
            "fns": [
                {"name": "bad name\nfn inject_name() -> i64 { 2 }", "call": "m.a", "params": []},
                {"name": "badparam", "call": "m.b",
                 "params": [{"name": "p q\nfn inject_param() -> i64 { 3 }", "ty": "i64"}]},
                {"name": "badty", "call": "m.c",
                 "params": [{"name": "x", "ty": "Weird\nfn inject_ty() -> i64 { 4 }"}]},
                {"name": "ok", "call": "m.ok", "params": [{"name": "x", "ty": "i64"}]}
            ]}"#;
        let module = generate(d).expect("generate");
        // The injected text survives as comment prose, never as a declaration.
        assert_eq!(declared_fns(&module), vec!["ok".to_string()], "module:\n{}", module);
        assert!(module.contains("# dropped `m.k`: a class fn inject_reason() -> i64 { 1 }\n"),
            "module:\n{}", module);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    #[test]
    fn empty_binding_set_is_an_error() {
        assert_eq!(generate(r#"{"runtime":"python","fns":[]}"#).unwrap_err(),
            "descriptor produced no bindings — `fns` was empty or every entry was skipped");
    }

    #[test]
    fn a_fully_filtered_surface_says_how_much_was_dropped() {
        // A module that filters down to nothing (§3) must not read as a module
        // with nothing in it: the error counts what the introspector refused.
        let one = r#"{"runtime":"python","fns":[],
            "dropped":[{"call":"m.K","reason":"a class"}]}"#;
        assert_eq!(generate(one).unwrap_err(),
            "descriptor produced no bindings — `fns` was empty or every entry was \
             skipped (1 more callable was dropped by the introspector)");
        let two = r#"{"runtime":"python","fns":[],
            "dropped":[{"call":"m.K","reason":"a class"},
                       {"call":"m.v","reason":"variadic parameter `*xs`"}]}"#;
        assert_eq!(generate(two).unwrap_err(),
            "descriptor produced no bindings — `fns` was empty or every entry was \
             skipped (2 more callables were dropped by the introspector)");
    }

    #[test]
    fn malformed_json_is_reported() {
        let e = generate("{not json").unwrap_err();
        assert!(e.contains("not valid JSON"), "got: {}", e);
    }

    #[test]
    fn missing_schema_reads_as_one() {
        // ASSIMILATE.md §4: every descriptor written before the field existed
        // is schema 1. Stripping `"schema": 1` must change nothing but the
        // provenance hash — the two modules' bodies are identical.
        let stripped = MATH.replacen("\"schema\": 1,\n        ", "", 1);
        assert!(!stripped.contains("schema"), "strip failed:\n{}", stripped);
        let with = generate(MATH).expect("generate");
        let without = generate(&stripped).expect("generate");
        let body = |m: &str| m.lines()
            .filter(|l| !l.starts_with("# descriptor fnv1a:"))
            .collect::<Vec<_>>().join("\n");
        assert_eq!(body(&with), body(&without));
    }

    #[test]
    fn unknown_schema_is_rejected_naming_both_versions() {
        // A version this reader does not know is refused whole, not half-read:
        // the diagnostic names the descriptor's version and the reader's.
        let d = MATH.replacen("\"schema\": 1", "\"schema\": 2", 1);
        assert_eq!(generate(&d).unwrap_err(),
            "descriptor: schema 2 is unknown — this dmc reads schema 1; regenerate \
             the descriptor or upgrade dmc");
        // And a `schema` that is not an integer is a descriptor error.
        let d = MATH.replacen("\"schema\": 1", "\"schema\": \"1\"", 1);
        assert_eq!(generate(&d).unwrap_err(), "descriptor: `schema` must be an integer");
        let d = MATH.replacen("\"schema\": 1", "\"schema\": 1.5", 1);
        assert_eq!(generate(&d).unwrap_err(), "descriptor: `schema` must be an integer");
    }

    #[test]
    fn unknown_fields_within_a_known_schema_are_ignored() {
        // Additive evolution stays cheap (§4): a schema-1 reader ignores fields
        // it does not know, at the top level and inside a fn entry alike.
        let d = r#"{
            "schema": 1,
            "runtime": "python",
            "generator": "some-future-tool 0.2",
            "fns": [
                {"name": "gcd", "call": "math.gcd", "deprecated": false,
                 "params": [{"name":"a","ty":"i64","doc":"left operand"},
                            {"name":"b","ty":"i64"}]}
            ]
        }"#;
        let module = generate(d).expect("generate");
        assert!(module.contains("fn gcd(__port: Port[python], a: i64, b: i64) -> (str, str)"),
            "module:\n{}", module);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
    }

    // ── introspection (python3 is a dev prerequisite, as in the port tests) ──

    /// A module carrying one callable per drop reason (§3) plus one that binds
    /// cleanly. Written to a temp dir and imported over `PYTHONPATH`, so the
    /// assertions below do not drift with the standard library.
    const FIXTURE_PY: &str = r##""""Fixture for the assimilate introspector tests."""
import functools
from math import log as no_signature_builtin
from math import sqrt as plain_builtin


class Widget:
    def __init__(self, mu: float = 0.0):
        self.mu = mu


class _Engine:
    def scaled(self, a: int, k: float = 2.0) -> float:
        return a * k

    def __call__(self, a: int) -> int:
        return a


def plain(a: int, b: float = 1.0) -> float:
    return a + b


def variadic(*xs: int) -> int:
    return sum(xs)


def kw_only(a: int, *, scale: float = 1.0) -> float:
    return a * scale


engine = _Engine()
bound_method = engine.scaled
partial_call = functools.partial(plain, 1)
not_callable = 42
globals()["bad-name"] = plain
globals()["multi\nline"] = plain


def __dir__():
    return sorted(list(globals()) + ["exploding"])


def __getattr__(name):
    raise RuntimeError("boom\nsecond line")
"##;

    /// `U+1F600` and `U+1F4A5` — non-BMP, so each is one UTF-16 surrogate pair
    /// in the `\uXXXX`-escaped output of `json.dumps` (#509).
    const NON_BMP_NAME: char = '\u{1F600}';
    const NON_BMP_MSG: char = '\u{1F4A5}';

    /// A module whose callable *name* and whose exception *message* both carry
    /// a non-BMP character. Both must survive introspection: reported, not a
    /// crash. Since no report crosses stderr any more, the text travels in the
    /// descriptor, where it is the JSON parser's job to reassemble the pair.
    fn emoji_py() -> String {
        format!(r##""""Fixture: non-BMP text in a name and in an exception."""


def plain(a: int) -> int:
    return a


globals()["gr{name}w"] = plain


def __dir__():
    return sorted(list(globals()) + ["boom"])


def __getattr__(attr):
    raise RuntimeError("exploded {msg} hard")
"##, name = NON_BMP_NAME, msg = NON_BMP_MSG)
    }

    /// A module that prints a line shaped exactly like a drop report at import
    /// time. It must not be able to pass that off as the introspector's.
    const SPOOF_PY: &str = r##""""Fixture: a module that forges a drop report."""
import sys

sys.stderr.write("assimilate: skipped `forged.callable`: a class\n")


def plain(a: int) -> int:
    return a
"##;

    /// Write `src` as the fixture module into a fresh temp dir and return the
    /// dir. The tag keeps parallel tests in one process from sharing one.
    fn fixture_dir_of(tag: &str, src: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("dmc_assimilate_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create fixture dir");
        std::fs::write(dir.join("dmc_assim_fixture.py"), src).expect("write fixture");
        dir
    }

    fn fixture_dir(tag: &str) -> std::path::PathBuf {
        fixture_dir_of(tag, FIXTURE_PY)
    }

    /// `(call, reason)` for every entry of a descriptor's `dropped` array.
    fn dropped_pairs(desc: &str) -> Vec<(String, String)> {
        let v = json_parse(desc).expect("valid JSON descriptor");
        let obj = as_map(&v).expect("object");
        dropped_entries(&obj).expect("dropped array").iter().map(|d| {
            let o = as_map(d).expect("dropped entry object");
            (get_str(&o, "call").expect("call"), get_str(&o, "reason").expect("reason"))
        }).collect()
    }

    /// One descriptor `fns` entry: `(name, call, ret, [(param, ty)])`.
    type FnEntry = (String, String, String, Vec<(String, String)>);

    /// `(name, call, ret, [(param, ty)])` for every entry of a descriptor's `fns`.
    fn fn_entries(desc: &str) -> Vec<FnEntry> {
        let v = json_parse(desc).expect("valid JSON descriptor");
        let obj = as_map(&v).expect("object");
        get_list(&obj, "fns").expect("fns").iter().map(|f| {
            let o = as_map(f).expect("fn object");
            let ps = match o.get("params") {
                Some(Value::List(xs)) => xs.iter().map(|p| {
                    let po = as_map(p).expect("param object");
                    (get_str(&po, "name").expect("name"), get_str(&po, "ty").expect("ty"))
                }).collect(),
                _ => Vec::new(),
            };
            (get_str(&o, "name").expect("name"), get_str(&o, "call").expect("call"),
             get_str(&o, "ret").expect("ret"), ps)
        }).collect()
    }

    #[test]
    fn introspect_python_module_yields_a_descriptor() {
        // `math` is all builtins with `__text_signature__`s, so it introspects
        // into a well-formed descriptor across python versions.
        let desc = introspect("python", "math").expect("introspect");
        let v = json_parse(&desc).expect("valid JSON descriptor");
        let obj = as_map(&v).expect("object");
        // The introspector stamps the schema version it writes (§4).
        assert!(matches!(obj.get("schema"), Some(Value::Int(1, _))), "descriptor:\n{}", desc);
        assert_eq!(get_str(&obj, "runtime").as_deref(), Some("python"));
        assert_eq!(get_str(&obj, "module").as_deref(), Some("math"));
        let fns = fn_entries(&desc);
        assert!(fns.iter().any(|(n, c, _, _)| n == "math_sqrt" && c == "math.sqrt"),
            "expected math_sqrt, got: {}", desc);
    }

    #[test]
    fn introspect_drops_classes_from_a_real_module() {
        // #496: `statistics.NormalDist` is a callable whose call returns an
        // instance — never JSON — so it must not become a binding, and the
        // descriptor must say why it is missing.
        let desc = introspect("python", "statistics").expect("introspect");
        assert!(!fn_entries(&desc).iter().any(|(_, c, _, _)| c == "statistics.NormalDist"),
            "NormalDist must not be bound; descriptor:\n{}", desc);
        let dropped = dropped_pairs(&desc);
        let reason = dropped.iter()
            .find(|(c, _)| c == "statistics.NormalDist")
            .map(|(_, r)| r.clone())
            .unwrap_or_else(|| panic!("NormalDist not reported; dropped: {:?}", dropped));
        assert_eq!(reason,
            "a class — its call returns an instance, which has no JSON value mapping \
             (PORTS.md §3)");
        // Plain module-level functions still bind.
        assert!(fn_entries(&desc).iter().any(|(_, c, _, _)| c == "statistics.mean"),
            "descriptor:\n{}", desc);
    }

    #[test]
    fn introspect_reports_every_drop_reason() {
        // Roadmap §1: nothing is silently omitted. One callable per reason,
        // recorded in the descriptor and rendered from it for the user.
        let dir = fixture_dir("reasons");
        let (desc, child_stderr) = run_introspector("dmc_assim_fixture",
            Some(&dir.to_string_lossy())).expect("introspect fixture");

        // In `sorted(dir(mod))` order, so the report is deterministic.
        let expected: Vec<(String, String)> = vec![
            ("dmc_assim_fixture.Widget",
             "a class — its call returns an instance, which has no JSON value mapping \
              (PORTS.md §3)"),
            ("dmc_assim_fixture.bad-name",
             "not a valid demoniC identifier for a fn name"),
            ("dmc_assim_fixture.engine",
             "a dmc_assim_fixture._Engine — only a plain function, a builtin, or a bound \
              method is bound; another callable's result may not round-trip through JSON \
              (PORTS.md §3)"),
            // An attribute that cannot even be read is reported, not skipped in
            // silence — and the exception's own newline is collapsed, so the
            // report stays one line per drop.
            ("dmc_assim_fixture.exploding",
             "reading the attribute raised RuntimeError (boom second line) — nothing to \
              introspect"),
            ("dmc_assim_fixture.kw_only",
             "keyword-only parameter `scale` — the argument vector is positional \
              (PORTS.md §2)"),
            // A newline inside the *name* is collapsed for the same reason.
            ("dmc_assim_fixture.multi line",
             "not a valid demoniC identifier for a fn name"),
            ("dmc_assim_fixture.no_signature_builtin",
             "no introspectable signature (a C builtin) — write the descriptor entry \
              by hand"),
            ("dmc_assim_fixture.partial_call",
             "a functools.partial — only a plain function, a builtin, or a bound method \
              is bound; another callable's result may not round-trip through JSON \
              (PORTS.md §3)"),
            ("dmc_assim_fixture.variadic",
             "variadic parameter `*xs` — the argument vector is fixed-arity \
              (PORTS.md §2)"),
        ].into_iter().map(|(c, r)| (c.to_string(), r.to_string())).collect();

        assert_eq!(dropped_pairs(&desc), expected, "descriptor:\n{}", desc);
        // The user-facing report is that same array, rendered — one full line
        // per drop and nothing else. There is no second account to drift.
        let want: Vec<String> = expected.iter()
            .map(|(c, r)| format!("assimilate: skipped `{}`: {}", c, r))
            .collect();
        assert_eq!(drop_report(&desc).expect("drop report"), want, "descriptor:\n{}", desc);
        // And the child wrote nothing to stderr at all: a quiet module leaves it
        // empty, so every line that ever appears there belongs to the module.
        assert_eq!(child_stderr, "", "child stderr should be empty");

        // Only the bindable callables survive, with their inferred types. The
        // return annotation rides along as `ret` (#485 typed decode);
        // `plain_builtin` is `math.sqrt`, which annotates neither.
        // `bound_method` binds: a bound method of a plain function is as
        // introspectable as the function, and the port resolves the same path.
        assert_eq!(fn_entries(&desc), vec![
            ("dmc_assim_fixture_bound_method".to_string(),
             "dmc_assim_fixture.bound_method".to_string(), "f64".to_string(),
             vec![("a".to_string(), "i64".to_string()), ("k".to_string(), "f64".to_string())]),
            ("dmc_assim_fixture_plain".to_string(), "dmc_assim_fixture.plain".to_string(),
             "f64".to_string(),
             vec![("a".to_string(), "i64".to_string()), ("b".to_string(), "f64".to_string())]),
            ("dmc_assim_fixture_plain_builtin".to_string(),
             "dmc_assim_fixture.plain_builtin".to_string(), "?".to_string(),
             vec![("x".to_string(), "?".to_string())]),
        ], "descriptor:\n{}", desc);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_module_cannot_forge_a_drop_report() {
        // The label is decided by where the line came from, not by what it says.
        // A module that writes an `assimilate:`-shaped line at import time gets
        // it labelled as the module's output like any other line, and it does
        // not appear among the drops — the descriptor is the only source those
        // come from.
        let dir = fixture_dir_of("spoof", SPOOF_PY);
        let (desc, child_stderr) = run_introspector("dmc_assim_fixture",
            Some(&dir.to_string_lossy())).expect("introspect fixture");
        assert_eq!(module_output(&child_stderr),
            vec!["assimilate: python: assimilate: skipped `forged.callable`: a class"],
            "child stderr:\n{}", child_stderr);
        assert!(!dropped_pairs(&desc).iter().any(|(c, _)| c == "forged.callable"),
            "a forged drop reached the descriptor:\n{}", desc);
        assert!(drop_report(&desc).expect("drop report").is_empty(),
            "descriptor:\n{}", desc);
        // The real surface still introspects normally around the noise.
        assert!(fn_entries(&desc).iter().any(|(_, c, _, _)| c == "dmc_assim_fixture.plain"),
            "descriptor:\n{}", desc);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_bmp_text_in_a_drop_survives_introspection() {
        // #509: a name and an exception message carrying a non-BMP character
        // must be *reported*, not crash the run. `json.dumps` escapes each as a
        // UTF-16 surrogate pair, so this also pins the JSON parser reassembling
        // one on the way back in.
        let dir = fixture_dir_of("emoji", &emoji_py());
        let (desc, child_stderr) = run_introspector("dmc_assim_fixture",
            Some(&dir.to_string_lossy())).expect("introspect fixture");
        assert_eq!(child_stderr, "", "child stderr should be empty");
        let dropped = dropped_pairs(&desc);
        assert!(dropped.iter().any(|(c, r)|
            *c == format!("dmc_assim_fixture.gr{}w", NON_BMP_NAME)
                && r == "not a valid demoniC identifier for a fn name"),
            "the non-BMP name was not reported; dropped: {:?}", dropped);
        assert!(dropped.iter().any(|(c, r)| c == "dmc_assim_fixture.boom"
            && *r == format!("reading the attribute raised RuntimeError (exploded {} hard) \
                              — nothing to introspect", NON_BMP_MSG)),
            "the non-BMP message was not reported; dropped: {:?}", dropped);
        // The report renders both, still one line each.
        let report = drop_report(&desc).expect("drop report");
        assert_eq!(report.len(), 2, "report: {:?}", report);
        assert!(report.iter().any(|l| l.contains(NON_BMP_NAME)), "report: {:?}", report);
        assert!(report.iter().any(|l| l.contains(NON_BMP_MSG)), "report: {:?}", report);
        // And the sound callable still binds, so the module was not abandoned.
        assert!(fn_entries(&desc).iter().any(|(_, c, r, _)|
            c == "dmc_assim_fixture.plain" && r == "i64"), "descriptor:\n{}", desc);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn introspect_binds_a_module_level_bound_method() {
        // Regression: filtering for round-trippable results must not throw away
        // `random.randint` & co., which are bound methods of a hidden `Random`
        // instance — plain python functions reached through an instance, which
        // the port resolves by the same dotted path.
        let desc = introspect("python", "random").expect("introspect");
        let fns = fn_entries(&desc);
        let randint = fns.iter().find(|(_, c, _, _)| c == "random.randint")
            .unwrap_or_else(|| panic!("random.randint must bind; descriptor:\n{}", desc));
        assert_eq!(randint.0, "random_randint");
        assert_eq!(randint.3.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>(),
            vec!["a".to_string(), "b".to_string()]);
        // The class it hangs off is still dropped — a constructor never binds.
        assert!(dropped_pairs(&desc).iter().any(|(c, r)| c == "random.Random"
            && r.starts_with("a class —")), "descriptor:\n{}", desc);
    }

    #[test]
    fn introspected_descriptor_generates_partial_bindings() {
        // End to end: introspect, then generate. The annotated fn emits with the
        // return type its annotation declared, decoded rather than handed back
        // as text; the `?`-param builtin is skipped with a needs-types note;
        // every dropped callable is accounted for in the module it is missing
        // from.
        let dir = fixture_dir("bindings");
        let (desc, _) = run_introspector("dmc_assim_fixture", Some(&dir.to_string_lossy()))
            .expect("introspect fixture");
        let module = generate(&desc).expect("generate");
        assert_eq!(declared_fns(&module),
            vec!["dmc_assim_fixture_bound_method".to_string(),
                 "dmc_assim_fixture_plain".to_string()], "module:\n{}", module);
        assert!(module.contains(
            "fn dmc_assim_fixture_plain(__port: Port[python], a: i64, b: f64) -> (f64, str)"),
            "module:\n{}", module);
        assert!(module.contains("let (__val, __derr) = json_decode_f64(__out)"),
            "module:\n{}", module);
        assert!(module.contains(
            "port_call(__port, \"dmc_assim_fixture.bound_method\", json_encode(__args))"),
            "module:\n{}", module);
        assert!(module.contains("# needs types `dmc_assim_fixture_plain_builtin`"),
            "module:\n{}", module);
        assert!(module.contains(
            "# dropped `dmc_assim_fixture.Widget`: a class — its call returns an instance, \
             which has no JSON value mapping (PORTS.md §3)\n"), "module:\n{}", module);
        assert!(module.contains("# dropped `dmc_assim_fixture.variadic`: variadic parameter \
             `*xs` — the argument vector is fixed-arity (PORTS.md §2)\n"), "module:\n{}", module);
        let full = format!("{}\nfn main() -> nil {{ nil }}\n", module);
        assert!(check_errs(&full).is_empty(), "errors: {:?}", check_errs(&full));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn introspect_rejects_bad_module_and_runtime() {
        assert!(introspect("python", "no.such.module.xyz").is_err());
        assert!(introspect("python", "bad-name").is_err());
        assert!(introspect("lua", "math").unwrap_err().contains("only wired for `python`"));
    }
}
