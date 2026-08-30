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
   the type, not a guess. The return type comes from the return annotation
   alone — there is no default value to fall back on — so an unannotated
   callable gets `"ret": "?"` and its wrapper hands back text (§5.1). Much
   of the python stdlib is unannotated; that is the recovery ceiling this
   section closes on, not a shortcoming of the reader. The draft descriptor
   is reviewed and its `?`s filled before it is assimilated.

The introspected surface is **filtered by the callable's kind** — a plain
python function, a builtin, or a bound method of one. `dir()` reports every
callable, classes included, but a class's call returns an instance, which
never round-trips; binding one would emit a wrapper that can only fail at
call time with a port tag (`PORTS.md §6`) rather than a compile error. Any
other callable (a `functools.partial`, a callable instance) is equally
unknowable from the outside, so it is dropped too.

A *bound* method does bind. `random.randint` is a method of a module-level
`Random` instance, not a free function, but it is a plain python function
reached through an instance: `inspect.signature` already elides the
receiver, and the port resolves `random.randint` by the same dotted-path
`getattr` walk (`PORTS.md §7.1`). Excluding classes and exotic callables
must not cost the caller half of `random`.

A callable is dropped when it

- is a class — its call returns an instance,
- is not a plain function, a builtin, or a bound method of a plain function,
- cannot be read at all (a module `__getattr__` that raises),
- has no introspectable signature (a C builtin with no `__text_signature__`),
- has a variadic (`*args`, `**kwargs`) or keyword-only parameter — the
  argument vector is positional and fixed-arity (`PORTS.md §2`), or
- has a name that is not a demoniC identifier.

**The filter is on the callable's kind, not on what it returns**, and it
cannot be otherwise. Python's annotations are optional and unenforced, so
there is no way to know from the outside whether a plain function's result
will serialize. A plain function that returns a non-JSON object still binds
and still fails at call time with a port tag, exactly as it did before the
filter existed:

```
$ dmc assimilate python:decimal --bindings
...
fn decimal_getcontext(__port: Port[python]) -> (str, str) { ... }
```

```
port-call: Object of type Context is not JSON serializable
```

`os.getcwdb` (bytes) behaves the same way. What the filter changed is which
*kinds* of callable reach the module at all: classes and exotic callables
no longer do. What a bound callable returns stays the descriptor's `?` —
the unknowable last mile the marker exists to record, not to guess.

**Nothing is dropped silently.** Each drop is recorded in the descriptor's
`dropped` array (§4), which the generator carries into the emitted module as
a `# dropped` comment, and which `dmc assimilate` renders to stderr in the
generator's voice — one line per drop:

```
assimilate: skipped `statistics.NormalDist`: a class — its call returns an instance, which has no JSON value mapping (PORTS.md §3)
```

The stderr report scrolls away; the descriptor and the module keep the
account. The report is *derived from* the `dropped` array rather than
printed by the introspector, so the two cannot disagree, and one drop is
always one line: the callable's name and the reason are whitespace-collapsed
first, so a newline in either — in an exception message, or in a name a
module put in its own `__dict__` — cannot split a report in two or run past
the `#` of the comment it lands in.

The imported module gets no say in that report. Anything the module itself
writes to stderr — an import warning, a deprecation notice, a line shaped
like a drop report — is passed through with its provenance attached:

```
assimilate: python: assimilate: skipped `forged.callable`: a class
```

The label goes on by where the line came from, not by what it looks like:
the introspector writes nothing to stderr, so every line there is the
module's. A drop the descriptor does not record is not a drop.

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
  "schema":  1,
  "runtime": "python",
  "module":  "math",
  "fns": [
    { "name": "math_gcd",
      "call": "math.gcd",
      "params": [ {"name": "a", "ty": "i64"}, {"name": "b", "ty": "i64"} ],
      "ret": "i64" },
    { "name": "math_sqrt",
      "call": "math.sqrt",
      "params": [ {"name": "x", "ty": "f64"} ],
      "ret": "f64" }
  ]
}
```

- `schema` is the descriptor format's version — an integer, currently `1`
  — and the introspector stamps it into every descriptor it writes. A
  descriptor without the field is schema 1: every descriptor written
  before the field existed. A reader rejects a schema it does not know,
  with a diagnostic naming the descriptor's version and its own; unknown
  *fields* within a known schema it ignores, so adding a field never
  bumps the version. Only a change that alters the meaning of what a
  reader already parses does.
- `runtime` is the `L` in the emitted `Port[L]` (§5.1). `python` is the
  wired runtime; another name generates checkable bindings that cannot
  execute yet (§9). The `c`/extern boundary (§5.2) is future.
- `module` is optional metadata (the source the introspector read); the
  generator does not consume it.
- `dropped` is optional: the introspector's account of the callables it
  refused to bind (§3), each `{"call": …, "reason": …}`. The generator
  copies it into the module as `# dropped` comments so the generated file
  explains its own omissions; a hand-written descriptor may leave it out.
- `name` is the emitted function name, used verbatim as `fn <name>` — the
  introspector qualifies it (`math_gcd`) to avoid collisions across modules.
- `call` is the foreign dotted path passed verbatim to `port_call`
  (`PORTS.md §2`).
- `params[].ty` is a demoniC **scalar** in the JSON value boundary —
  `i64`, `f64`, `f32`, `bool`, or `str` (`PORTS.md §3`). A param of any
  other type is **skipped and reported** — the whole function is dropped
  with a comment, never emitted half-formed, never silently lost.
- `ret` is the wrapper's result type, and it is the contract typed decode
  enforces: one of `i64`, `f64`, `str`, `bool`, `list` — the types with a
  typed decode primitive (`PORTS.md §3.1`). Omitted, or `"?"`, means
  *untyped*: the wrapper returns the raw canonical-JSON `str` (§5.1). A
  `ret` naming any other type is **skipped and reported**, `f32` included
  — JSON numbers carry no width, so narrowing is an explicit `as f32` at
  the call site, not something the boundary does behind the caller's back.

`?` is asymmetric between the two, deliberately. An unknown *param* type
drops the function: without it there is no argument vector to build. An
unknown *return* type does not: the result is still a canonical-JSON str,
which is exactly what a wrapper handed back before typed decode existed.

The descriptor is the reviewable artifact. It is committed alongside the
generated module so regeneration is a diff, not a surprise.

---

## 5. Targets

### 5.1 Port wrappers (interpreter boundary)

For `runtime: python`, each descriptor `fn` becomes one demoniC function
that wraps a `port_call`, encoding the argument vector per `PORTS.md §2`
and returning the `(T, Err)` fallible convention (ports never throw —
`PORTS.md §6`). `T` is the descriptor's `ret` (§4).

The wrapper takes an already-open `Port[L]` as its first parameter: the
caller owns `port_open`/`port_close`, and one handle serves a whole module
of calls. The altitude is **runtime-parametric** — the descriptor's
`runtime` becomes the `L` in `Port[L]`, so the same generator targets any
runtime. `python` is the only runtime with a port wired today
(`PORTS.md §7.1`); bindings for an unwired runtime (§9) still generate and
type-check, they just cannot execute until that runtime's port lands.

The result is the descriptor's `ret`, decoded from the canonical-JSON
result through the matching typed decode primitive (`PORTS.md §3.1`), plus
an Err tag. The argument vector is built as a `list` and JSON-encoded per
`PORTS.md §2`:

```
# generated by `dmc assimilate` — do not edit by hand.
# runtime: python. each wrapper returns (T, str): the descriptor's `ret`
# decoded from the canonical-JSON result, and an Err tag (PORTS.md §6).
# an untyped `ret` returns the raw canonical-JSON str. fix the
# descriptor and regenerate.
# descriptor fnv1a: 0b4a53db0c3e169d

fn math_gcd(__port: Port[python], a: i64, b: i64) -> (i64, str) {
    let __args = list()
    let __args = list_push(__args, a)
    let __args = list_push(__args, b)
    let (__out, __err) = port_call(__port, "math.gcd", json_encode(__args))
    if __err != nil { return (0, __err) }
    let (__val, __derr) = json_decode_i64(__out)
    if __derr != nil { return (0, __derr) }
    (__val, nil)
}
```

Generated locals carry a `__` sigil so a descriptor param can never shadow
them (param names are validated as plain identifiers, §4) — and the sigil
reads as machine-generated, matching the do-not-edit header.

The caller gets an `i64`, not a string it has to parse. `math_gcd(p, a, b)`
composes with arithmetic and with `?` (`SPEC.md §4.9`) like any other
fallible demoniC function. `T`'s zero rides both error paths so `(T, Err)`
stays well-typed on failure.

The two error paths stay distinguishable. A `port-` tag is the runtime
failing; a `decode-type` tag is the *descriptor* being wrong — the foreign
function returned something other than what `ret` promised. That is the
point of the typed form: a descriptor's declared types are a contract the
boundary enforces at every call, not a comment. Nothing is truncated or
reinterpreted to make a mismatch go away.

The contract is enforced as sharply as the wire allows, and on numbers the
wire is blunt (`PORTS.md §3.1`). JSON has one number type and the canonical
writer prints `2.0` as `2`, so a whole-valued float and an integer are
byte-identical by the time the wrapper reads them. A descriptor claiming
`ret: "i64"` over a python `-> float` that returns `5.0` gets `(5, nil)`:
the descriptor is wrong and no check at this boundary can say so. State it
plainly rather than let the generated header imply otherwise. What the
boundary does catch is every result that survives the writer as a distinct
token — against `ret: "i64"`, a `5.5` is a `decode-type`, and so is a str,
a bool, a null, a list, a map, or an integer past `i64`'s range. Read `ret`
as checked against the *kind* of the result, with integers and whole floats
a single kind.

Without a `ret` (or with `"?"`) the wrapper keeps the untyped shape —
`(str, Err)` carrying the raw canonical-JSON result, the same thing the
hand-written `port_python.dmc` works with. That is the honest output when
the introspector could not recover a return type: text, plus a `?` in the
descriptor asking a human to say what it is.

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
- A binding for an unmappable param type, a `ret` with no typed decode, or
  one whose `name` or a param name is not a demoniC identifier (§4). The
  function is skipped and reported — never emitted half-formed.
- A wrapper whose declared return type nothing enforces. If `ret` says
  `i64`, the emitted code decodes an `i64` or returns an Err — never the
  raw str, never a value of another JSON kind. One gap is the wire
  format's, not the generator's, and it is disclosed rather than papered
  over: a whole-valued float is written `2`, indistinguishable from the
  integer `2` (`PORTS.md §3.1`), so an `i64` `ret` over a float-returning
  function is caught only for non-whole results (§5.1).
- A binding for an introspected callable that is not a plain function, a
  builtin, or a bound method of one — a class above all (§3). A wrapper
  that can only ever fail at call time is worse than no wrapper: it is
  dropped and reported.

What it does *not* refuse is a bound callable whose result turns out not to
serialize. That is unknowable from python's surface (§3); it is left to
report `port-call` at call time, and `ret: "?"` says the boundary makes no
claim about it.

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
<descriptor>` reads a hand-written descriptor and emits port-wrapper
bindings — the direct industrialization of `examples/port_python.dmc`.
Generation is deterministic and runtime-parametric; `python` is the only
runtime with a wired port (`PORTS.md §7.1`), so bindings for another
runtime (e.g. `mojo`) generate and type-check but carry a header note and
cannot execute until that runtime's port lands.

Typed decode is implemented. `PORTS.md §3.1` adds one decode primitive per
boundary type, and a descriptor's `ret` drives which one a wrapper emits:
`(i64, Err)`, `(f64, Err)`, `(str, Err)`, `(bool, Err)`, `(list, Err)`. A
type the foreign runtime did not honor surfaces as a `decode-type` tag, so
the descriptor's types are now a contract the boundary checks rather than
documentation. `ret` is optional; without it a wrapper keeps the untyped
`(str, Err)` shape.

The python introspector (§3) is implemented: `dmc assimilate
python:<module>` reads the live module and emits a draft descriptor, and
`--bindings` carries it through to wrappers in one command. Param types
come from annotations and default values, the return type from the return
annotation; the rest stay `?` for review. The surface is filtered to plain
functions, builtins, and bound methods — classes and exotic callables are
excluded by kind, not by result type (§3) — and every dropped callable is
named with its reason on stderr and in the descriptor's `dropped` array.

A module can filter down to nothing bindable: most of the standard library
annotates neither parameters nor returns, so `--bindings` over it reports
every skip and exits nonzero. `textwrap` is the whole shape in seven lines:

```
$ dmc assimilate python:textwrap --bindings
assimilate: skipped `textwrap.TextWrapper`: a class — its call returns an instance, which has no JSON value mapping (PORTS.md §3)
assimilate: skipped `textwrap.fill`: variadic parameter `**kwargs` — the argument vector is fixed-arity (PORTS.md §2)
assimilate: skipped `textwrap.shorten`: variadic parameter `**kwargs` — the argument vector is fixed-arity (PORTS.md §2)
assimilate: skipped `textwrap.wrap`: variadic parameter `**kwargs` — the argument vector is fixed-arity (PORTS.md §2)
assimilate: skipped `textwrap_dedent`: param type `?` needs a type — introspection could not infer it; set `ty` in the descriptor
assimilate: skipped `textwrap_indent`: param type `?` needs a type — introspection could not infer it; set `ty` in the descriptor
assimilate: descriptor produced no bindings — `fns` was empty or every entry was skipped (4 more callables were dropped by the introspector)
```

The first four lines are the introspector's drops, the next two the
generator's skips, and the count on the last names the filtered surface, so
an empty result is not mistaken for an empty module. That is the honest
outcome, and the draft descriptor (without `--bindings`) is the artifact to
fill in.

Not yet built:

- **Introspectors for other runtimes** — only `python` is wired. A `c`
  header parser and others follow.
- **The §5.2 extern altitude** — `runtime: c` → `extern fn` bindings.
- **JIT lowering of port calls.** Generated wrappers run on the
  interpreter; the JIT does not lower `port_open`/`port_call`/`port_close`
  yet (`PORTS.md §2`).

---

## 10. Philosophy

Ports keep demoniC small by refusing to absorb other ecosystems. That
refusal has a cost: every crossing is hand-written. Assimilation pays the
cost down without paying the philosophical price — it does not import the
ecosystem, it prints the boundary. The wall stays; the door is just no
longer whittled one at a time.
