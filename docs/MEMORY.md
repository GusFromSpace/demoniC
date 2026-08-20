# demoniC — Memory Model

**Companion to:** `docs/SPEC.md §3` (Types), `§6.3` (Arenas as scopes),
`§7` (Execution model).

This document defines how demoniC allocates, mutates, and reclaims memory.
It is normative for memory behavior. Normative means the *observable*
semantics: aliasing, copy-on-write visibility, arena lifetimes, and the
listed diagnostics all hold today on both backends. Mechanism narrative
(the bump-pointer layout in §1.2, the sibling-count protocol in §4, the
code-arena reclamation in §6) describes the specified implementation;
the current runtime achieves the same observable behavior by simpler
means — e.g. JIT code pages currently live until process exit. Where this file and `docs/SPEC.md`
disagree, **the spec wins** (`docs/SPEC.md`, preamble); file a
`spec-conflict` issue so the loser gets patched.

---

## 1. The three arenas

demoniC has exactly three data arenas. There is no general heap. (The
JIT's code region — §6 — holds machine code, not data, and is not
user-allocatable.)

| Arena   | Lifetime          | Reclamation                  | Holds                                    |
| ------- | ----------------- | ----------------------------- | ----------------------------------------- |
| Vault   | program lifetime  | only on process exit         | weights, embeddings, vocab, config       |
| Forge   | one "step"        | bump-pointer reset to 0      | activations, gradients, scratch          |
| Stream  | until released    | per-allocation cursor + free | KV caches, growable `~` axes (§9)        |

The Stream arena was added to support `KV[T, [..., ~, ...]]` types
(`docs/SPEC.md §3.6`, Streaming types (KV caches)). It behaves like a
Forge slab whose allocations **survive** Forge resets, plus a
per-allocation cursor that supports appends via `<-`.

The Vault and Forge are **bump allocators** over a pre-reserved virtual address
range. Allocation is one atomic `fetch_add` on the bump pointer plus a
write of zeroes if the type demands them. There is no free list, no
coalescing, no fragmentation.

### 1.1 Sizes

The arenas are sized at program startup:

```
dmc run model.dmc --vault=16G --forge=2G
```

*(Specified, not yet implemented: `dmc` rejects the sizing flags with a
diagnostic saying so; arenas size dynamically today.)*

Specified defaults, once sizing lands:

- Vault: half of physical RAM, rounded down to nearest huge-page boundary.
- Forge: 2 GiB.

Both arenas reserve virtual address space up front (`mmap(MAP_NORESERVE)`
on Linux, `VirtualAlloc(MEM_RESERVE)` on Windows) and commit pages lazily
on first touch. This means a 16 GiB Vault costs nothing until you fill it.

### 1.2 Alignment

Every allocation is aligned to the maximum SIMD lane width of the host —
64 bytes on AVX-512, 32 on AVX2, 16 on NEON. The bump pointer is rounded
up before each allocation. Padding is wasted; nobody cares.

---

## 2. Allocation API surface

User code rarely allocates directly. It happens implicitly:

```
let A = forge.zeros[f32, [B, 768]]      # explicit, in Forge
let B = vault.load("weights.bin")        # explicit, in Vault
let C = A @ B                            # implicit; arena chosen by §3
```

The methods on `vault` and `forge` are:

```
.zeros[T, shape]
.ones[T, shape]
.identity[T, N]      # N×N matrix, 1 on the diagonal, 0 elsewhere
.uninit[T, shape]    # !! contents undefined, must be written before read
.load(path)          # only on `vault`; runs under `dmc jit` only
.snapshot()          # returns an opaque token, see §5
.restore(token)
```

`.identity` accepts either a scalar dimension (`vault.identity[f32, 3]` → 3×3)
or a square shape literal (`vault.identity[f32, [3, 3]]`). It completes the
zeros / ones / identity triple: the additive-identity tensor, the all-ones
tensor, and the multiplicative-identity matrix. `I @ M == M` and `M @ I == M`
hold by construction.

`forge.uninit` is the brutalist's tool — no zero-fill, no init checks at
runtime. Reading an uninit binding before any write lands on it is a
compile-time error. The analysis is binding-level and
deliberately coarse, in statement order: the first write — an element
or whole assignment, a `<-` append, or passing the binding to a `!`
parameter — initializes it; per-element coverage is not tracked.
Passing it to a plain parameter or a builtin counts as a read.

---

## 3. Arena selection rules

When an expression allocates and no arena is explicit, demoniC picks one
by this algorithm:

1. If the expression is lexically inside `vault { ... }`, use Vault.
2. If the expression is lexically inside `forge { ... }`, use Forge
   explicitly (same as the default, but makes the intent clear).
3. Otherwise, use Forge.

That's it. No reachability analysis, no escape analysis, no surprises.

### forge { ... } blocks

`forge { ... }` is the explicit Forge-arena scope block — the symmetric
counterpart to `vault { ... }`. Code inside executes with Forge as the
active arena. Forge allocations made inside the block are reset at the
end of the block (the bump pointer is restored to where it was on entry).

```
forge {
    let tmp = forge.zeros[f32, [B, 768]]   # allocated in Forge
    ...                                     # `tmp` is alive here
}                                           # Forge resets to entry state
```

Want a tensor in Vault? Put it in a `vault {}` block or call
`vault.alloc` explicitly. This is intentional — implicit promotion would
silently leak training memory into the model.

### 3.1 Crossing arenas

A `View` may point into either arena. Operations that mix Vault and Forge
inputs are legal; the **output** goes to the arena selected by the rules
above. The compiler emits no warning for cross-arena reads. Cross-arena
**writes** (mutating Vault data from a Forge context) require an explicit
`vault { ... }` block — otherwise they are a **compile-time error**
(enforced). Like the §2 uninit-read error this is a spec
violation rather than a lint, so demon mode does not suppress it. The
check is binding-level: it fires on element writes, compound assigns,
and `<-` appends through a binding whose value visibly comes from a
`vault.*` constructor or a `vault { ... }` block expression, whenever
the innermost lexical arena block is not `vault`. A plain whole-`=`
rebind is not a mutation, and re-tags the binding by its new value's
arena. Vault data reaching a write site through an alias, a `!`
parameter, or a model field is not yet tracked (those need the arena
tag to travel with the *type*, which is a spec-level change).

---

## 4. Copy-on-Write

The defining trick of the memory model. A `View` is a fat pointer; many
views can share the same underlying bytes.

When the program writes through a view, the compiler must guarantee the
write doesn't corrupt some other view that happens to alias the same
memory. demoniC handles this with CoW, not with aliasing analysis.

### 4.1 The CoW protocol

For any mutating operation on a view `v`:

1. The JIT checks the view's **sibling count** — a tiny integer kept
   next to the bump pointer of the backing arena page, not per-byte.
   (This is not a reference count; see §4.2.)
2. If the count is 1 (this view is the only one looking at these bytes),
   the write proceeds in place.
3. If the count is >1:
   - bump the arena pointer by `sizeof(elements)`,
   - emit a hardware `rep movsq` (x86) or NEON-aligned copy (aarch64)
     to clone the bytes,
   - rewrite the view to point at the new region,
   - decrement the original's sibling count,
   - then perform the write on the new region.

### 4.2 Why this is not refcounting

The sibling count is **per-page-region**, not per-object. Pages allocated
together in the same bump stride share one counter. There is no
graph-walking, no cycle collector, no deferred reclamation. When the
Forge resets, all counters reset to 0 in a single `memset`.

To be precise about what the counter values mean:

- **0** — unallocated / not yet owned. A freshly reset Forge page starts
  here. No view points at this region.
- **1** — one owner; this is the only view. Writes proceed in-place (see
  §4.1 step 2). This is the unaliased-but-owned state.
- **>1** — aliased. Writes trigger the CoW copy path (§4.1 step 3).

The distinction between 0 and 1 matters: a count of 1 is the "safe to
mutate in place" signal; a count of 0 means the region has never been
handed to user code and must not be mutated through any view pointer.

### 4.3 Opting out

`mut` (`!`) on a binding is a *permission*, not a guarantee of in-place.
To force in-place mutation and fail loudly if CoW would trigger:

```
let !x = forge.uninit[f32, [N]]
@inplace x += bias                 # compile-time error if x has aliases
```

The `@inplace` directive is *specified* to turn aliasing into a
diagnostic. Not yet implemented: today it parses and the checker warns
that it has no effect, like the other unimplemented directives in
`docs/DIRECTIVES.md §1` (Catalog).

---

## 5. Snapshots

The Forge supports **snapshots** — first-class checkpoint tokens of the
bump pointer.

```
let snap = forge.snapshot()
let intermediate = expensive_kernel(x)
... # uses `intermediate`
forge.restore(snap)                 # everything after `snap` is gone
```

This is the standard pattern for sub-step scratch. Cheaper than a full
Forge reset, structurally analogous to an arena scope in a game engine.

Snapshots are LIFO. Restoring an old snapshot while a newer one is still
live is undefined behavior today and specified to become a compile-time error.

---

## 6. Interaction with the JIT

The JIT writes machine code into a separate non-data region — the
**Code arena** — technically a Vault sibling but with `PROT_EXEC`.
Users cannot allocate into it.

When the JIT specializes a hot kernel (per `docs/SPEC.md §7`, Execution
model), it:

1. Writes the new opcodes into a fresh region of the Code arena.
2. Atomically swaps the function pointer in the dispatch table.
3. Marks the old region as reclaimable on the next Forge reset.

Code is **never** patched in place under concurrent execution. JIT
rewriting operates at function boundaries, not at instruction
boundaries inside a running kernel. This
distinction matters and will be enforced by the runtime.

---

## 7. What does not exist

To make the absence explicit, the following are **not features** of
demoniC and never will be:

- A garbage collector of any kind.
- Reference counting (`Rc`, `Arc`, `shared_ptr`).
- Weak references.
- Finalizers, destructors, `Drop`, RAII.
- A free function. `forge.reset()` is the only reclamation primitive.
- Thread-local heaps. The Vault is shared; the Forge is per-worker
  (§8).

If you find yourself wanting one of these, you are writing a different
language. Stop.

---

## 8. Multi-worker layout (sketch)

Distribution directives (`@shard`, `@tp`, `@pp`) are parsed and
type-checked today but do not alter code generation. A normative
multi-worker arena layout — how Forge, Stream, and Vault partition
across workers — will be specified in a future revision.

---

## 9. The Stream arena

The Stream arena supports values whose type contains a `~` (streaming)
axis. Its allocation API:

```
forge.kv[T, S](capacity = N)        # reserves N elements along `~`
```

A KV allocation made with `forge.kv` automatically survives
`forge.reset()` — the stream-persistence behavior is built in, not a
separate constructor. `stream.kv[T, S](capacity = N)` is accepted as an
explicit synonym of `forge.kv` and lowers identically — both the
interpreter and the JIT treat `forge` and `stream` the same for `.kv`.
Use it when the streaming intent is worth spelling out at the call site.

A streaming allocation reserves `capacity` elements along the `~` axis
up front. The cursor starts at 0. Appends via `<-` (see
`docs/SPEC.md §4.8`, Stream append: `<-`) advance the cursor by the
appended extent.

Implementation:

- Backed by the Stream slab, page-committed on first append.
- Per-allocation header: cursor, capacity, dtype, full shape.
- Appending past `capacity` is a **runtime error** — never a realloc.
  Enforced on both backends.
- `release(kv)` *(specified, not yet implemented — an undefined
  identifier today)* returns the slab pages to the Stream free list.
  The Stream arena does **not** have a global reset operator.

Why a separate arena, and not a Forge sub-region: KV caches outlive a
single training/inference step. Forge reset must not free them. Vault
would work but appends-as-bumps require a writable cursor, which the
Vault's "locked until process death" model disallows. Stream is the
minimum addition that satisfies both constraints.

### 9.1 Iteration safety

Appending to a stream via `<-` while inside a loop that reads from the
same `KV` value is a **compile-time error**
(`stream-iteration-aliasing`; enforced).
The hazard is the standard mutate-while-iterating bug class: a loop
that grows its own iterable either skips or revisits elements depending
on cursor semantics, with no good answer.

To grow a stream from inside an iteration over it, bind an explicit
snapshot first:

```
let !c = forge.kv[f32, [~]](capacity = 8)
c <- [1.0f32]
c <- [2.0f32]
let snap = c                  # captures the cursor as a View[f32, [2]]
for v in snap {
    c <- [v * 10.0f32]         # appends past `snap`'s frozen extent
}
# c is now [1, 2, 10, 20]; snap still sees [1, 2]
```

`snap` is a `View` (`docs/SPEC.md §3.6`, Streaming types (KV caches)) —
the conversion is free and the captured cursor value is part of the
view's shape. New appends grow `c` past `snap`'s extent; the next
iteration would need a fresh snapshot to see them.

The rule fires on lexical scope, not data-flow analysis: any `<-` on a
`KV` in the body of a `for` loop whose iterable is the same `KV` is
the error, regardless of branches or conditionals. This is intentional —
the brutalist choice keeps the diagnostic deterministic and the user
in control of when a snapshot is taken.

---

## 10. Reserved

- NUMA-aware arena pinning.
- GPU/device arenas (`device.forge`, `device.vault`).
- mmap-backed Vaults for checkpoint resume.
- Stream slab reuse across workers for KV-cache pooling.

These will be specified before 0.1.
