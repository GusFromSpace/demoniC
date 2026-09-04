# demoniC — Ports

**Companion to:** `docs/SPEC.md §3.11` (Ports).

Ports are the foreign-runtime boundary. demoniC does not need a large
native ecosystem. It needs hard, explicit exits to runtimes that
already exist.

This document is normative.

---

## 1. Goal

A port lets demoniC call an interpreter, process, or embedded runtime
without admitting that runtime into demoniC's type system.

Examples:

```
Port[python]
Port[lua]
Port[wasm]
Port[posix]
```

The foreign language stays foreign. demoniC values cross the boundary
only through the ABI below.

---

## 2. Core operations

Every implementation must provide these runtime builtins *(implemented
as the §7.1 process-port floor, `python` runtime only; both the
interpreter and the JIT lower them, sharing one registry so neither
backend owns a private copy of the protocol)*:

```
port_open(lang: str) -> (Port[lang], Err)
port_call(p: Port[L], name: str, payload: str) -> (str, Err)
port_close(p: Port[L]) -> (nil, Err)
```

For the python process port, `name` is a dotted import path
(`math.sqrt`, `json.dumps`, or a bare `builtins` name like `len`). The
`payload` str is the call's argument vector, decoded from JSON:

| `payload` decodes to | binding                                          |
| -------------------- | ------------------------------------------------ |
| `null` or empty      | called with no arguments                         |
| a JSON array         | the elements are the positional arguments        |
| a JSON object        | the `{args, kwargs}` envelope (below)            |
| any other JSON value | `port-protocol` — a bare scalar is not a vector  |

A single argument is therefore a one-element array (`[x]`), and a single
`array`/`object` argument nests one level inside it. A top-level object
is always the envelope, so an `object` argument is never mistaken for
it. The envelope carries `args` (a JSON array, spread positionally) and
`kwargs` (a JSON object, spread by name); both are optional and any
other key is a `port-protocol` error. The result is re-encoded to
canonical JSON before `port_call` returns it. A runtime other than
`python` reports `port-open`.

`payload` and the returned `str` are canonical JSON unless the port was
opened with an implementation-specific binary channel. JSON is the
portable baseline because it is inspectable, hostile to pointer leaks,
and already available in the bootstrap compiler. The wire text is
**UTF-8** (JSON's definition): a reader accepts raw multi-byte
sequences and `\uXXXX` escapes — surrogate pairs included — as the same
string, and rejects ill-formed UTF-8 as a parse failure rather than
passing it through.

The language does not define `import python`, `foreign fn`, or a
syntax-level module bridge in 0.0.4.

---

## 3. Value boundary

The portable JSON ABI supports:

| demoniC value | JSON form       |
| ------------- | --------------- |
| `nil`         | `null`          |
| `bool`        | boolean         |
| integer       | number          |
| float         | number          |
| `str`         | string          |
| tuple/list    | array           |
| map           | object          |

### 3.1 Typed decode

A result crosses back as the canonical-JSON `str` `port_call` returns.
Turning that text into a demoniC value is `json_decode` — which hands
back a dynamic value — or the typed decode family, one primitive per
type in the table above:

```
json_decode_i64(s: str)  -> (i64,  Err)
json_decode_f64(s: str)  -> (f64,  Err)
json_decode_str(s: str)  -> (str,  Err)
json_decode_bool(s: str) -> (bool, Err)
json_decode_list(s: str) -> (list, Err)
```

A typed decode **never coerces across a JSON kind**. Text that is JSON
of another kind returns `T`'s zero and a `decode-type` tag (§6): `"7"`
does not become `7`, `2.5` does not truncate, `1` is not `true`, `null`
is not a zero. Declaring the type is how you get it checked; silence
would make the declaration worthless.

Numbers are the exception, and the exception belongs to JSON, not to
demoniC. JSON has a single number type, and the canonical writer prints
a whole float without its fraction (`2.0` is written `2`, §2). **A
whole-valued float and an integer are the same bytes on the wire, and
nothing downstream can tell them apart.** The widening is therefore
effectively bidirectional *for whole values*, and only for those:

| result on the wire  | `json_decode_i64`      | `json_decode_f64` |
| ------------------- | ---------------------- | ----------------- |
| `2` (an integer)    | `2`                    | `2.0` — widened   |
| `2` (was the float `2.0`) | `2` — *not catchable* | `2.0`       |
| `2.5`               | `decode-type`          | `2.5`             |
| `100000000000000000000` | `decode-type`      | `1e20`            |
| `"2"`, `null`, `true` | `decode-type`        | `decode-type`     |

Read down the two columns. An integer always satisfies an `f64` decode,
by design: refusing it would fail on every whole-valued float that
crosses the boundary. A whole-valued float always satisfies an `i64`
decode too — not by design, but because the writer erased the
difference before the decode could look. A descriptor that promises
`i64` for a function returning `5.0` gets `5` and no error, and no
check at this layer can catch it.

What the `i64` direction *does* catch is every value that survives the
writer as a distinct token: a str, a bool, a null, a list, a map, and
every non-whole number. `2.5` as `i64` is a `decode-type`, because
truncation is a real coercion and the one this family exists to
prevent.

Magnitude follows the same rule. A fraction-less JSON number too large
for `i64` — anything past `9223372036854775807`, which is where whole
floats land once they pass `2^63` — is still a well-formed JSON number,
so it decodes as `f64`. `json_decode_f64` accepts it; `json_decode_i64`
reports `decode-type: expected i64, got f64`. It is never a
`decode-parse`: that tag means the text is not JSON (§6), and a long
integer is.

Closing the whole-value gap would mean making the canonical writer emit
`2.0`, and that format is the port ABI (§2) — every reader already
depends on it. A caller that must distinguish an integer from a whole
float carries the distinction in the payload, in an envelope field or a
tagged object, not in the shape of a bare number.

`json_decode_str` yields the *decoded* string (`"hi"` → `hi`), not the
JSON text. A caller that wants the raw canonical JSON already has it:
that is what `port_call` returned.

`f32` has no typed decode. JSON numbers carry no width, so the family
lands them in `f64`; narrow with `as f32` where you want it, explicitly.

**Ratified 2026-09-02.** The whole-value int/float wire decision above —
the canonical writer emits `2.0` as `2`, the ambiguity is permanent by
design, and a caller needing the distinction carries it in the payload —
was reviewed and confirmed as a ratification rather than an open
question. It stands unchanged. `docs/STABILITY.md §2` lists this ABI as
a stable surface and §3 of that document is the procedure that now
governs changing it.

### 3.2 Tensors

Tensor values do not implicitly become JSON arrays. That would hide
allocation and destroy shape information. Tensor exchange uses one of
two explicit modes:

1. **Copy:** demoniC writes bytes into a declared payload buffer and
   passes metadata `{dtype, shape, layout}`.
2. **Borrow:** demoniC lends a read-only `View` for the dynamic extent
   of a single `port_call`.

Borrow is optional. Copy is required.

---

## 4. Borrow rules

A borrowed view:

- Is read-only to the port.
- Cannot escape the dynamic extent of the call.
- Cannot be stored by the foreign runtime.
- Cannot be used after the call returns.
- Cannot be passed while a mutable alias exists in demoniC.

If the implementation cannot prove the foreign runtime honors these
rules, it must copy. Fast unsafe sharing is not a portability feature.

---

## 5. Effects

Port calls are effect boundaries.

- No fusion crosses a port call.
- No `@comptime` evaluation may call a port.
- No `@grad fn` may call a port.
- No `@fuse` block may call a port.
- `@deterministic` may call only deterministic ports.

A deterministic port is a port whose run inputs are pinned by a
**manifest** (§5.1) naming the runtime, its version, the environment it
sees, the files it reads, and the argument vector it launches with.
`@deterministic` refuses any other port at compile time — the refusal
rule in §5.2.

All four restrictions are enforced. The checker rejects a port call
inside a `@grad fn`, a `@fuse` block, or a `@deterministic` block —
closures they enclose included — with a `port-forbidden` diagnostic. The
scope is lexical, like every other directive scope (`docs/DIRECTIVES.md`
§4): a port call in a function that one of those constructs *calls* is
not caught.

`@comptime` is the exception to that last caveat, and it is stricter
rather than weaker. Its body admits **no call of any kind**
(`docs/SPEC.md §7.5` (Comptime evaluation (`@comptime`))), so a port
call is refused as `port-forbidden` and the transitive case cannot
arise: there is no call through which a port could be reached.
Compile-time evaluation therefore performs no effect by construction,
not by analysis.

`@deterministic` still rejects every port today: the manifest format
below is *specified, not yet implemented*, so no port can present one.
When resolution lands, the rule narrows from all ports to un-manifested
ones, at the same checker gate that rejects them now.

### 5.1 The deterministic-port manifest

*Everything in §5.1–§5.2 is specified, not yet implemented.*

A manifest is one canonical-JSON file per runtime:
`ports/<runtime>.port.json` under the package root — the directory
holding `demoni.json` — or, for a file compiled outside any package,
next to the source file. It is a **sibling** of the assimilate
descriptor (`ASSIMILATE.md §4`), not a block inside it: the descriptor
names the symbols a binding module calls; the manifest pins the runtime
instance those calls execute on. One runtime serves many descriptors
and hand-written modules alike, so the pins live once, beside the
package, not once per binding.

The manifest for the §7.1 python process port:

```json
{
  "schema": 1,
  "runtime": "python",
  "argv": ["python3", "-u", "-c", "${harness}"],
  "version": "3.12.3",
  "env": {
    "LC_ALL": "C",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0"
  },
  "files": []
}
```

And one for the planned rust cdylib port (in-process, §7.2 — the
`Port[rust]` convention on the roadmap), proving the format spans both
runtime classes:

```json
{
  "schema": 1,
  "runtime": "rust",
  "library": {
    "path": "native/libdmc_kernels.so",
    "sha256": "9c56cc51b374c3ba189210d5b6d4bf57790d351c96c47c02190ecf1e430635ab"
  },
  "env": {},
  "files": [
    { "path": "data/tuning.json",
      "sha256": "b5bb9d8014a0f9b1d61e21e796d78dccdf1352f23cd32812f4850b878ae4944c" }
  ]
}
```

| Field     | Pins                                                            |
| --------- | --------------------------------------------------------------- |
| `schema`  | the manifest format itself; this section defines `1`            |
| `runtime` | the `L` in `Port[L]` — equals the filename stem and the `port_open` literal |
| `argv`    | process ports (§7.1): the exact launch vector                   |
| `library` | in-process ports (§7.2): the loaded artifact, content-hashed    |
| `version` | the exact version string the runtime must report                |
| `env`     | the complete environment the runtime consumes                   |
| `files`   | every file the port reads that affects output, content-hashed   |

Exactly one of `argv`/`library` appears, and the choice is the §7
runtime class. `version` is required with `argv` and optional with
`library`: a content hash pins the runtime byte-exactly, stronger than
any version string, while a process port's executable lives outside the
package tree, so a probed version string is the pin it can have.

Field rules:

- **`schema`.** A manifest whose `schema` the implementation does not
  know is refused, naming both versions. So is one carrying a field
  this schema does not define. That second rule is deliberately the
  **opposite** of the descriptor's additive-evolution rule
  (`ASSIMILATE.md §4`): a descriptor's unknown field is metadata a
  generator may skip, but a manifest is a refusal contract — an
  unenforced pin silently weakens the very guarantee the file exists to
  give. New pins therefore cost a schema bump, on purpose.
- **`argv`.** Launched exactly as written. The one placeholder,
  `${harness}` as a whole element, is replaced by the implementation's
  §7.1 harness program. The harness bytes are not a pin — they are part
  of the compiler and travel with its version.
- **`env`.** For a process port the pin is **constructed, not
  inspected**: the child is launched with exactly this environment,
  plus inherited `PATH` — the one variable needed to resolve `argv[0]`
  at all — and nothing else from the ambient environment. Listing
  `PATH` in `env` pins even that. An in-process port shares the
  compiler's process and cannot be relaunched, so its `env` entries are
  *checked* against the live environment at open instead; a mismatch
  fails verification. Construction where possible, comparison where
  not.
- **`files`** (and `library`). `sha256`, lowercase hex, over the exact
  bytes; paths resolve against the package root. The descriptor stamps
  its output with FNV-1a (`ASSIMILATE.md §6`) — a change fingerprint
  for human review. A manifest hash backs a compiler-enforced contract,
  so it takes a collision-resistant hash — and one every machine can
  produce and audit without `dmc` (`sha256sum`).
- **`version`.** Exact string equality against a probe the runtime
  class defines. For the §7.1 python port, the harness reports
  `platform.python_version()` before the first request is served; how
  the implementation asks is its business, because the harness is its
  code.

**Verification at `port_open`.** A `port_open` lexically inside
`@deterministic` compiles to a *verifying open*; outside, `port_open`
never reads a manifest and costs nothing new. The verifying open
re-reads and re-validates the manifest (it may have changed since the
check ran), hashes `library` and every `files` entry, constructs (or
checks) `env`, launches `argv`, and matches `version`. Any pin that
fails yields no handle and a `port-manifest` error (§6) naming the pin
and both values. A runtime that fails to start at all is still
`port-open`, manifest or none. The cost is one SHA-256 pass over the
listed bytes plus one handshake line, once per open and never per call
— amortized over every call the handle serves.

What verification does **not** claim, disclosed like the §3.1
whole-value gap: it pins the inputs the manifest *names*, and cannot
prove the foreign code consumes nothing else. A python function that
reads the clock, spawns threads, or pulls `/dev/urandom` is
nondeterministic behind a fully verified manifest. The manifest makes
the named inputs fixed and the claim auditable; it is a pinning
discipline, not a sandbox. And the bar is the determinism contract's
(`docs/SPEC.md`, "Determinism contract (`@deterministic`)"): repetition
on the same host — a matching version string on another machine does
not promise bit-equal floats.

### 5.2 The refusal rule

*(Specified, not yet implemented.)* Inside a `@deterministic` extent,
every port builtin must name a manifested runtime, statically:

- `port_open(lang)` requires `lang` to be a `str` literal. A runtime
  the checker cannot name is refused — there is nothing to resolve a
  manifest against.
- The named runtime's manifest must resolve and validate. Missing: the
  diagnostic names the path that was searched. Malformed, unknown
  `schema`, unknown field, or a `runtime` that does not match the
  filename stem and the literal: the diagnostic names the offending
  field and value.
- `port_call`/`port_close` are checked against the manifest of the `L`
  in the handle's `Port[L]` type, under the same rules.

The refusal is the existing §5 gate with its condition narrowed, not a
second mechanism: the same checker walk, the same lexical extent, the
same `port-forbidden` tag — only the *because* changes, from "no port
carries a manifest" to naming exactly what this port is missing.

---

## 6. Errors

Ports do not throw. Every core operation returns `(_, Err)`.

Port errors are `str` tags. The minimum tags are:

| Tag              | Meaning                              |
| ---------------- | ------------------------------------ |
| `port-open`      | runtime could not be started         |
| `port-call`      | call failed inside the runtime       |
| `port-protocol`  | JSON or binary ABI was malformed     |
| `port-closed`    | handle was already closed            |
| `port-forbidden` | call violates a directive or borrow  |
| `port-manifest`  | a manifest pin failed at a verifying open (§5.1) |

Implementations may append detail after `:`. Code must match only the
tag prefix.

`port-manifest` *(specified, not yet implemented)* is reachable only
from a verifying open (§5.1). It is deliberately not `port-open`: that
tag means the runtime could not be started, and code matching it treats
the failure as availability — install the runtime, retry. A manifest
failure is the opposite case: the runtime may start fine, it just is
not the one that was pinned, and the remediation is to fix the
environment or re-pin, not to install. Folding the two together would
hide exactly the distinction `@deterministic` exists to surface.

Decoding a result (§3.1) is a step after the call, so it carries its own
tags. They are deliberately outside the `port-` family: a caller that
asked for the wrong type is not a runtime that failed.

| Tag            | Meaning                                     |
| -------------- | ------------------------------------------- |
| `decode-parse` | the result str is not JSON                  |
| `decode-type`  | the result is JSON of another kind (§3.1)   |

The split is text versus kind, and the line does not move with
magnitude. Every well-formed JSON number parses however long it is, so
an integer past `i64`'s range is a `decode-type` against `i64` and a
success against `f64` — never a `decode-parse` (§3.1).

A generated wrapper (`ASSIMILATE.md §5.1`) threads both families out
through its single `Err`, so matching `port-` finds the runtime failures
and matching `decode-` finds the contract failures.

---

## 7. Runtime classes

### 7.1 Process ports

A process port launches an external command and speaks JSON over
stdin/stdout. This is the portability floor.

Process ports are slow, explicit, and easy to inspect. They are the
right first implementation.

Both backends run the same one. The child process, the line framing,
the argument-vector envelope of §2, and the §6 tag decisions live in a
single module that the interpreter and the JIT each call into; neither
holds a copy. The `Err` half of a result is `nil` when there is no
error, which the JIT spells as a null string pointer — so a real `str`,
the empty one included, is never `nil`, exactly as in the interpreter.
Handle ids are never reused, so a stale handle reports `port-closed`
instead of reaching a port opened later.

A handle is opaque in both backends, and stays opaque under the JIT:
the interpreter carries one as an opaque value, the JIT as a distinct
`Port` type over the same pointer. It compares only against `nil`
(`p == nil` is how a failed open is detected), renders — for display
only — as `<opaque port#<id>:<lang>>`, and is otherwise inert: it has
no length, no elements, and no equality against another handle or
against a `str`. A `str` is therefore not accepted where a handle
belongs. Handle text is ordinary text, so accepting one would let a
program forge a handle it never opened; the interpreter refuses that
when the call is reached, the JIT when the program is compiled.
`Port[L]` may be written in a signature, so a handle can be passed and
returned, as the "Ports" section of the language spec requires — and `L`
is not decoration: binding a handle to a `Port[L]` parameter or return
position compares `L` against the runtime the handle was opened for. A
`Port[lua]` parameter refuses a python handle, in both backends, at the
binding. It cannot be a compile-time check in general — the handle's
runtime name is a value, chosen by whatever `str` reached `port_open` —
so it is a located runtime error, worded the same on both sides.

`nil` is not a handle either — that is what a failed `port_open` hands
back — and reaching a port operation with one is the same error in both
backends, at the same span: `port_call: first arg must be a Port handle
from port_open`. The same holds for a `str`-typed port argument that is
nil at run time: `port_open(e)` and `port_call(p, e, …)` on a nil `e`
raise `lang must be str` / `name must be str` rather than sending an
empty name across the pipe. The one exception is the payload, where
`nil` *is* meaningful — it is the written-out form of "call with no
arguments" (§2).

`nil` also has no methods, so the §6 idiom written *without* its guard,
`e.starts_with("port-open")` on an `Err` that is nil because nothing
went wrong, is a located runtime error in both, never a quiet "no". A
handle is inert in the same way: it has no methods and no elements, so
`p.starts_with(..)` and `p[0]` are errors, not a falsy answer. The
interpreter raises them when the call is reached; the JIT, which knows
the receiver is a `Port` while compiling, refuses the program.

### 7.2 Embedded interpreter ports

An embedded port hosts an interpreter in-process: Lua, Python, or a
WASM engine. It must expose the same ABI as a process port. In-process
state is foreign state, not demoniC arena state.

### 7.3 Device ports

Device ports cover audio devices, windows, sockets, and other runtime
surfaces. They are still ports. They do not earn language syntax just
because they are useful.

---

## 8. Philosophy

demoniC stays small by refusing to absorb other ecosystems.

The language owns:

- Typed numeric kernels.
- Shape checking.
- Arena discipline.
- Deterministic contracts.

Ports own:

- Plotting.
- Audio playback.
- UI toolkits.
- File format churn.
- Foreign package ecosystems.

The boundary is the feature.
