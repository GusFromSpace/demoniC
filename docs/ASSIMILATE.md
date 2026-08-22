# demoniC — Assimilation

**Companion to:** `PORTS.md` (whole), `SPEC.md §3.11`, `§6.7`.

Ports are the foreign-runtime boundary. Writing a port wrapper by hand —
open the runtime, encode the argument vector, call, decode the result,
thread the `(_, Err)` convention, close — is mechanical and repetitive.
`port_python.dmc` is a page of it for three functions.

Assimilation industrializes that page. You point `dmc assimilate` at a
foreign interface and it emits the demoniC binding module you would have
typed. It is a **boundary factory**, not an ecosystem importer.

This document is normative. The §5.1 port-wrapper altitude is
implemented; the rest is design. See §9 for the status line.

---

## 1. Goal

One instruction, one generated module of ordinary demoniC source:

```
dmc assimilate python:math --bindings   > math.dmc
dmc assimilate c:cblas.h                 > blas.dmc   # future (§5.2)
```

The output is source a human could have written and can read, diff, edit,
and commit. Assimilation adds no syntax, no keyword, no directive, no
type-system entry. It writes the same `port_call` and `extern fn`
declarations the spec already defines, in bulk.

---

## 2. The invariant

Assimilation absorbs **interface surface, never semantics.**

- It emits declarations that *call* foreign code across an existing ABI.
- It never inlines, transpiles, or embeds foreign code into demoniC.
- The foreign runtime stays foreign. `PORTS.md §8` still holds: the
  boundary is the feature. Assimilation mass-produces boundary crossings;
  it does not dissolve the boundary.

Source-to-source porting — reading foreign source and emitting equivalent
demoniC — is a different discipline and is **out of scope** for this
document.

---

## 3. Two stages

Assimilation is descriptor-driven. It does not guess foreign types.

```
    (introspector)            (assimilator)
 foreign interface  ──▶  descriptor  ──▶  generated .dmc bindings
     (optional)            (§4)              (§5)
```

1. **Assimilator** (normative, deterministic). Consumes a **descriptor**
   (§4) — a typed manifest of the foreign symbols to bind — and emits the
   binding module. Pure text-in, text-out: same descriptor, byte-identical
   output.
2. **Introspector** (best-effort, optional). A front-end that produces a
   descriptor from a live interface. The python introspector is implemented:
   `dmc assimilate python:<module>` imports the module and reads it with
   `inspect`, recovering each public callable's name and arity. Types it
   recovers from parameter annotations, then from default values
   (`sigma=1.0` → `f64`), and otherwise emits `"?"` — a request to supply
   the type, not a guess. A callable is dropped whole when it has no
   signature (a C builtin) or any variadic or keyword-only parameter. It
   enumerates callables, so classes and constructors are included; one whose
   result does not round-trip through JSON generates a wrapper that fails at
   call time with a port tag (`PORTS.md §6`), not a compile error. The draft
   descriptor is reviewed and its `?`s filled before it is assimilated.

The split is deliberate. Type discovery in a dynamic runtime is a guess;
code generation from a declared surface is not. demoniC owns the second
and refuses to pretend the first is exact — Python exposes names and arity
but rarely types, so the last mile is a human decision the descriptor
records, not a fabrication the tool commits.

Where a richer descriptor source comes from is an environment concern. A
bare compiler has only live introspection. Under an operating
system that installs languages, the install leaves type stubs, headers, or
a published ABI on the system, and a registry can point the descriptor at
those instead — the same assimilator, a fuller, typed descriptor.
Installing a language is what raises the ceiling on what assimilate can
recover.

---

## 4. The descriptor

A descriptor is canonical JSON — the same portable baseline the port ABI
speaks (`PORTS.md §2`). It names a target and a list of symbols with
demoniC signatures.

```json
{
  "runtime": "python",
  "module":  "math",
  "fns": [
    { "name": "math_gcd",
      "call": "math.gcd",
      "params": [ {"name": "a", "ty": "i64"}, {"name": "b", "ty": "i64"} ] },
    { "name": "math_sqrt",
      "call": "math.sqrt",
      "params": [ {"name": "x", "ty": "f64"} ] }
  ]
}
```

- `runtime` is the `L` in the emitted `Port[L]` (§5.1). `python` is the
  wired runtime; another name generates checkable bindings that cannot
  execute yet (§9). The `c`/extern boundary (§5.2) is future.
- `module` is optional metadata (the source the introspector read); the
  generator does not consume it.
- `name` is the emitted function name, used verbatim as `fn <name>` — the
  introspector qualifies it (`math_gcd`) to avoid collisions across modules.
- `call` is the foreign dotted path passed verbatim to `port_call`
  (`PORTS.md §2`).
- `params[].ty` is a demoniC **scalar** in the JSON value boundary —
  `i64`, `f64`, `f32`, `bool`, or `str` (`PORTS.md §3`). A param of any
  other type is **skipped and reported** — the whole function is dropped
  with a comment, never emitted half-formed, never silently lost. There is
  no `ret` field today: a wrapper returns `(str, Err)` (§5.1), so the
  result type is not part of the descriptor until typed decode lands (§9).

The descriptor is the reviewable artifact. It is committed alongside the
generated module so regeneration is a diff, not a surprise.

---

## 5. Targets

### 5.1 Port wrappers (interpreter boundary)

For `runtime: python`, each descriptor `fn` becomes one demoniC function
that wraps a `port_call`, encoding the argument vector per `PORTS.md §2`
and returning the `(T, Err)` fallible convention (ports never throw —
`PORTS.md §6`).

The wrapper takes an already-open `Port[L]` as its first parameter: the
caller owns `port_open`/`port_close`, and one handle serves a whole module
of calls. The altitude is **runtime-parametric** — the descriptor's
`runtime` becomes the `L` in `Port[L]`, so the same generator targets any
runtime. `python` is the only runtime with a port wired today
(`PORTS.md §7.1`); bindings for an unwired runtime (§9) still generate and
type-check, they just cannot execute until that runtime's port lands.

The result is the canonical-JSON string plus an Err tag. The argument
vector is built as a `list` and JSON-encoded per `PORTS.md §2`:

```
# generated by `dmc assimilate` — do not edit by hand.
# runtime: python. each wrapper returns (str, str): the canonical-JSON
# result and an Err tag (PORTS.md §6). fix the descriptor and regenerate.
# descriptor fnv1a: 0b4a53db0c3e169d

fn math_gcd(__port: Port[python], a: i64, b: i64) -> (str, str) {
    let __args = list()
    let __args = list_push(__args, a)
    let __args = list_push(__args, b)
    let (__out, __err) = port_call(__port, "math.gcd", json_encode(__args))
    if __err != nil { return ("", __err) }
    (__out, nil)
}
```

Generated locals carry a `__` sigil so a descriptor param can never shadow
them (param names are validated as plain identifiers, §4) — and the sigil
reads as machine-generated, matching the do-not-edit header.

The `(str, Err)` result is the stepping stone of §9: a *typed* wrapper
returning `(i64, Err)` needs a typed JSON-decode primitive the bootstrap
compiler does not yet have (`json_decode` returns a dynamic value), so the
generated wrapper hands back the canonical-JSON string — the same shape
the hand-written `port_python.dmc` works with.

Generated wrappers are port calls, so `PORTS.md §5` governs them: a
generated wrapper is illegal inside a `@grad fn` exactly as a hand-written
`port_call` is, and the checker rejects it identically.

### 5.2 Extern bindings (compiled boundary)

For `runtime: c`, each descriptor `fn` becomes one `extern fn` declaration
(`SPEC.md §6.7`). This is the literal "point at a compiler": assimilate a
library's declared C ABI into demoniC's foreign-function surface.

```
# generated by `dmc assimilate` — do not edit by hand.
# runtime: c. descriptor fnv1a: 1b77…e0 — fix the descriptor and regenerate.

extern fn cblas_sgemm(order: i32, transA: i32, transB: i32,
    m: i32, n: i32, k: i32, alpha: f32, a: *f32, lda: i32,
    b: *f32, ldb: i32, beta: f32, c: *f32, ldc: i32) -> nil
```

The value boundary is the C-ABI scalar set and raw pointers `*T`
(`SPEC.md §3.10`, `§6.7`), not JSON. `extern fn` carries its own effect
restrictions (`SPEC.md §6.7`: no `@comptime`, no `@grad fn`); assimilation
adds none.

`extern fn` bindings hand-written today (BLAS callouts) are the standing
proof this altitude is worth automating.

---

## 6. Determinism and provenance

Generated modules are build outputs held under version control, like a
lockfile.

- **Deterministic.** A descriptor maps to one byte-exact module. No
  timestamps, no ordering nondeterminism — symbols emit in descriptor
  order.
- **Stamped.** Every generated file carries a header naming the runtime
  and the descriptor's content hash (an FNV-1a of the descriptor bytes).
  Review trusts the header; CI can re-assimilate and diff to prove the
  committed module matches its descriptor.
- **Idempotent.** Re-running assimilate over an unchanged descriptor
  produces byte-identical output.

Generated modules are never hand-edited. Fixes go to the descriptor and
regenerate — the header says so.

---

## 7. Refusals

Assimilation is a generator for boundary declarations and nothing more.
It refuses to emit:

- Any construct outside the declared boundary — no new directive,
  operator, keyword, or type. If a human could not write it by hand under
  the spec, assimilate does not produce it.
- A binding for an unmappable param type, or one whose `name` or a param
  name is not a demoniC identifier (§4). The function is skipped and
  reported — never emitted half-formed.
- Code that hides allocation or bypasses the JSON / C ABI.
- `pub extern fn` — an `extern fn` is always exported (`SPEC.md §6.7`);
  the marker is a compile-time error, so it is never generated.

---

## 8. Surface

```
dmc assimilate <target> [-o <file>]
```

`<target>` is one of:

- a **descriptor path** (`math.assimilate.json`) → the generated `.dmc`
  wrapper module.
- a **`python:<module>`** introspection target → the draft descriptor (§3)
  read from the live runtime. With `--bindings`, it runs straight through
  the generator: functions whose params all typed become wrappers, the rest
  are reported.

With no `-o`, output goes to stdout. The `c:` extern altitude (§5.2) and
introspectors for other runtimes are future targets.

Exit is fallible in the port sense: a bad descriptor, an unresolvable
target, or an unmappable-only surface reports a diagnostic and a nonzero
status; a partial surface (some symbols skipped per §4) emits what mapped
and reports the rest.

---

## 9. Status

The §5.1 port-wrapper altitude is implemented: `dmc assimilate
<descriptor>` reads a hand-written descriptor and emits `(str, Err)`
port-wrapper bindings — the direct industrialization of
`examples/port_python.dmc`. Generation is deterministic and
runtime-parametric; `python` is the only runtime with a wired port
(`PORTS.md §7.1`), so bindings for another runtime (e.g. `mojo`) generate
and type-check but carry a header note and cannot execute until that
runtime's port lands.

The python introspector (§3) is implemented: `dmc assimilate
python:<module>` reads the live module and emits a draft descriptor, and
`--bindings` carries it through to wrappers in one command. Types come from
annotations and default values; the rest stay `?` for review.

Not yet built:

- **Typed decode.** Wrappers return the canonical-JSON result string as a
  stepping stone. A *typed* wrapper returning `(i64, Err)` needs a typed
  JSON-decode primitive the bootstrap compiler lacks (`json_decode`
  returns a dynamic value).
- **Introspectors for other runtimes** — only `python` is wired. A `c`
  header parser and others follow.
- **The §5.2 extern altitude** — `runtime: c` → `extern fn` bindings.

---

## 10. Philosophy

Ports keep demoniC small by refusing to absorb other ecosystems. That
refusal has a cost: every crossing is hand-written. Assimilation pays the
cost down without paying the philosophical price — it does not import the
ecosystem, it prints the boundary. The wall stays; the door is just no
longer whittled one at a time.
