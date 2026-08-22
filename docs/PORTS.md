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
in the interpreter — the §7.1 process-port floor, `python` runtime
only; the JIT does not lower them yet)*:

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
and already available in the bootstrap compiler.

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

A deterministic port has a manifest naming the runtime, version,
environment variables, files, and command arguments that affect output.
Without that manifest, the compiler is specified to emit a directive
diagnostic when the port is called inside `@deterministic`.

The `@grad fn` restriction is enforced: the checker rejects a port call
inside a `@grad fn` — closures it encloses included — with a
`port-forbidden` diagnostic. The `@comptime`, `@fuse`, and
`@deterministic` restrictions are specified but not yet enforced: those
directives parse but do not yet gate, and no port carries a manifest.

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

Implementations may append detail after `:`. Code must match only the
tag prefix.

---

## 7. Runtime classes

### 7.1 Process ports

A process port launches an external command and speaks JSON over
stdin/stdout. This is the portability floor.

Process ports are slow, explicit, and easy to inspect. They are the
right first implementation.

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
