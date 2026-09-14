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

Both spellings work — `--forge=2G` and `--forge 2G`. A size is an integer
with an optional binary unit suffix (`B`, `K`, `M`, `G`, `T`, or the
spelled-out `KiB`/`MiB`/…, case-insensitive). No floats: write `1536M`,
not `1.5G`. Zero, a non-numeric value, an unknown suffix, and a count that
overflows a 64-bit byte total are all rejected on the command line, before
the program starts.

Exhausting a sized arena is a **runtime error at the allocation site** —
never an OOM-kill, never a silent grow:

```
runtime error at 12:9: forge arena exhausted: this allocation needs 2 MiB,
but only 1 KiB of the 1 KiB --forge budget is free
```

Omit a flag and that arena is unbounded. What each backend meters:

- **`dmc run`** meters both arenas. The interpreter has no bump pointer, so
  the budget is byte accounting charged wherever a tensor is materialized —
  an explicit `vault.*`/`forge.*` constructor, an implicit intermediate (a
  matmul or dotted-op result, a tensor literal, a transpose, a copying
  slice, a reshape), a builtin's fresh result (`softmax`, `sum_along`,
  `argmax`, `rms_norm`, `embed`, `diag`, an `as` cast, `.split`, a `@cast`
  overlay field, a non-KV `<-` concat), a `!` parameter's copy-in or a
  `let !y = x` bare-identifier's private copy, or an `@grad` pass's
  activations and published gradients — at that tensor's DECLARED element
  width, not the interpreter's f64 storage width. A `!` parameter's
  copy-OUT is not a materialization and is not charged: it writes back into
  the buffer the caller already owns, which both backends already charged.
  A fresh STRING is charged the same way, at its UTF-8 byte length, wherever
  one materializes — a builtin's result (`to_str`, `read_file`, …), `+`'s
  concatenation arm, an `as str` cast, and
  `.trim`/`.upper`/`.lower`/`.replace`/`.split`/`.lines` (`.split` and
  `.lines` charge every piece of the list they return, not just the list).
  A `!` parameter's or `let !y = x`'s private copy is the one place a string
  is DIFFERENT from a tensor: a `str` is a pointer on the JIT
  (`docs/SPEC.md §3.1`, Scalar types), so that binding copies a pointer
  there, not the string's bytes, and requests 0 B for it — charging the
  copy's length here would charge a copy the JIT never makes, so it is
  excluded.
  The Stream arena (`forge.kv` / `stream.kv`, bounded by its own `capacity`
  — §9) is not charged, by design.
- **`dmc jit`** honors `--forge` against the cumulative requested bytes the
  bump allocator has ever been asked for — not against the bytes
  physically committed — and an allocation that does not fit reports
  the same message with the same numbers, minus the source
  location — the JIT's allocator is a runtime callback and has no span to
  point at:

  ```
  runtime error: forge arena exhausted: this allocation needs 1 MiB, but
  only 1 KiB of the 1 KiB --forge budget is free
  ```

  `--vault` is **rejected**
  under `dmc jit` rather than ignored — the JIT lowers `vault.*` and
  `forge.*` into that one Forge arena, so there is no Vault region to size.
  Use `dmc run` when you want a Vault budget.

Both meters are **hierarchical, not monotonic**. A meter's `used` rises
when an allocation is charged to it and falls only when a budget it carved
is reaped, or when
the compiler reclaims a temporary it inserted itself: reaping a carve
credits the carve's whole extent — spent and unspent alike — to the meter
it was carved from, and **a compiler temporary released by the compiler
returns its charge on both backends** — the `@grad` epilogue's rewind after
`.fwd_bwd`, and the matching rewind at the exit of a forward-only `@grad
fn` entry, are the two such scopes today. Both are invisible to the
program: the user never asked for those bytes and cannot name them, and
neither scope rewinds an explicit constructor the program can still name
(a computed tensor that escapes the same way is still rewound today): an
explicit `vault.*`/`forge.*` constructor written inside either body keeps
its charge, because the tensor it built can escape — into a model field,
through a `!` parameter, into a module-level `let !` — and stay live after
the rewind. **Nothing else returns budget.** `forge.restore(mark)` rewinds
the bump cursor so the memory is reused and returns nothing to the meter, on
both backends. An undeclared `forge { … }` block carves nothing and returns
nothing (§3): a block is not a way to fit more into a budget than the
program actually holds live.

Both backends meter **requested** bytes; the JIT's chunk commitment is an
allocator detail that asks the meter for the allocation's own need before it
commits, and is not itself the meter. A chunk is sized from that need —
aligned, never from what is left of `--forge` — and the bytes a chunk
physically commits are compared against nothing: the refusal decision and
the "free" figure in an exhaustion message are both read off the
requested-byte meter alone.

The two backends are **not** interchangeable for a given budget. Both now
charge every materialized tensor, intermediates included, at its declared
element width, and both return a compiler temporary's charge at the same
two scopes — but the interpreter's real storage is still f64 where the
JIT's is f32, and the two carry different, independently-filed
over-charges (the JIT's chunk alignment; its per-rebind copy for `let y = x`,
which the interpreter makes on the Rust heap and does not meter; the JIT's
per-string charge, below; and the JIT's Forge
materialization of a model literal and of a `?`-propagated `(T, Err)`
tuple — 64 B per value, measured by the budget acceptance program, which
`dmc run` builds outside any arena), so the
same program can still fit under `dmc run --forge=N` and
exhaust under `dmc jit --forge=N`, or the reverse. Two systematic
differences remain, both measured rather than estimated: `--forge` covers a
`vault.*` constructor under `dmc jit` (one arena) but not under `dmc run`
(two), and the scratch a backward pass takes INSIDE one VJP rule — a
matmul's transposes, the buffer an accumulating `+=` produces — differs
between the two backward implementations, so only the adjoints a pass
*publishes* are a set they agree on. Size each backend against its own
measurements.

The two backends' string minimums agree up to a per-string residual. The
JIT's forge string layout is `[len: i64][bytes]`, so every fresh string —
the `+` result included — costs `dmc jit` an extra 8-byte header and then
rounds up to the 64-byte SIMD alignment (§1.2); the JIT also charges every
STRING LITERAL occurrence as its own materialization, which `dmc run` never
does (a bare literal is not *produced* by anything). `"lit" + to_str(f32)`
costs 9 B under `dmc run` (3 B for `to_str`'s builtin result, 6 B for the
concat) and 192 B under `dmc jit` (three forge strings — `"lit"`,
`to_str`'s result, and the concat — each header-plus-content rounded up to
64 B). `let !s = ""  for i in 0..1000 { s = s + "x" }` costs `dmc run`
500 500 B (1+2+…+1000, the exact cumulative byte length of each concat
result) and `dmc jit` 604 224 B. The 604 224 is exact, not estimated:
Σ align_up₆₄(n + 8) for n = 1..1000 — the 1000 concat results, each an
8-byte header plus its `n`-byte content — is 540 160, and the 1001 literal
allocations (the loop's 1000 `"x"`s plus the initial `""`) each round up to
the one-chunk minimum, 64 B, for another 1001 × 64 = 64 064. Neither the
header nor the literal-materialization charge is chased to zero; they sit
beside the chunk alignment and the per-rebind copy in the over-charge list
above. A `let`-bound arena constructor is no longer among them: `let t =
forge.ones[…]` was charged twice on the JIT and once on the interpreter,
and the JIT now charges it once, like the block form.

Under `dmc test --jit`, every test gets the whole budget: the interpreter
is rebuilt per test and the JIT's arena is reset alongside it. A test that
exhausts the budget is reported as that test's `FAIL` and the run
continues — the allocation is still served, so the budget acts as a
detector there rather than the hard ceiling it is under `dmc run` and
`dmc jit`.

Specified defaults, once sizing is a reservation rather than a budget:

- Vault: half of physical RAM, rounded down to nearest huge-page boundary.
- Forge: 2 GiB.

*(Not yet applied: an omitted flag means unbounded today, not the default
above.)*

Both arenas reserve virtual address space up front (`mmap(MAP_NORESERVE)`
on Linux, `VirtualAlloc(MEM_RESERVE)` on Windows) and commit pages lazily
on first touch. This means a 16 GiB Vault costs nothing until you fill it.
*(Specified reservation model. Today the interpreter allocates per tensor
and the JIT commits chunked bump slabs on demand; neither reserves the
whole range up front, so `--vault` and `--forge` are budgets rather than
reservations.)*

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

The check reaches one level into a model's fields. A **model-array**
field built straight from `uninit` is tracked under its dotted path, so
`h.cells` obeys exactly the rules above. Fill it in place, through the field:

```
model Cell   { !n: i64 }
model Holder { !cells: [Cell; 3] }

let !h = Holder { cells: vault.uninit[Cell, [3]] }
vault { for i in 0..3 { h.cells[i] = Cell { n: 0 } } }   # this is the write
```

Copying the field out first — `let !cs = h.cells` — does **not** work: a
model-array field binds by *value*, so the fill lands in the copy and the
field itself stays uninitialized. Reading the field to make that copy is the
error. Building the array as a local and storing it at construction
(`Holder { cells: cs }`) is equally correct, and never marks the field.

Two deliberate boundaries. **Tensor** fields are not tracked: they bind as a
live alias, so filling through `let !v = h.buf` really does reach the field
and flagging it would be a false report. And the reach is one level and coarse
in the same direction as the binding-level check — handing the model to
anything that could fill it (a method call on it, passing it as an argument)
drops the mark rather than risk a false error.

A bracket literal never builds a model array: `[Cell { n: 0 }, …]` is a tensor
literal, and is rejected at the field's declared type.

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

The reset is a bump-pointer rewind, not a refund: an arena that is *sized*
(§1.1) does not get those bytes back when the block ends. On `dmc run`
there is no bump pointer at all — the block is a scope, its allocations are
ordinary heap tensors, and its value escapes as the block's result — so the
`--forge` meter simply keeps climbing. `forge { … }` is therefore never a
way to fit more into a budget than the program actually holds live.

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

### 3.2 Budgets

*(Implemented on both backends; see the end of this section.)* A **budget** is a host-lifetime handle on a region of Forge, carved from a
parent budget and bounded by the parent's free capacity at the time of the
carve. `forge.root` names the process-wide root budget that `--forge`
sizes. Four operations:

```
forge.carve(parent: Budget, bytes: i64) -> Budget   # carve a child, checked against the parent's free capacity
forge.zeros[f32, [N]]                               # charged to the entered budget
forge.zeros[f32, [N]] in b                          # charged to `b` instead, an explicit override
forge.reap(b: Budget) -> nil                        # release `b`'s region; credit its whole extent to its parent
forge.enter(b) { ... }                              # push `b` as the entered budget for the block
```

`forge.enter(b) { ... }` has a lexical **attribution** extent and a
host-controlled **lifetime**: the entered-budget register it pushes is
per-invocation state, restored on every exit edge — normal exit, `return`,
`break`, `continue`, and `?` (an early return, the fifth edge) — and not
carried across a host call or a scheduler yield, both being returns out of
demoniC code. Nothing is reaped when the block exits; `b` outlives it. Only
allocation is ambient — `forge.carve` and `forge.reap` name their budget
argument directly.

A `Budget` is an opaque descriptor — one machine word, no arithmetic, no
ordering, no literal, not a tensor and not a model. Assignment copies the
descriptor, and both copies name the *same* budget, so §4's CoW does not
apply. A `Budget` may be stored in a model field, held in a `[Tenant; N]`,
and returned from a fn; no `!` marks one anywhere.

**Refusals, and the runtime errors:**

- A `forge.carve` past the parent's free capacity is a **runtime refusal**
  (`budget-carve`), at the carve. Not check-time: `--forge` is a runtime
  flag, so no static parent capacity exists for `dmc check` to read.
- A `forge.reap` with a live child is a **runtime refusal**
  (`budget-reap`), at the reap.
- **Well-formedness stays check-time**: a `Budget`-typed first argument to
  `forge.carve`, an integral second, and a non-positive `bytes` *literal*
  are ordinary type errors — the one static slice of the two rules above
  that a checker with no arena state can still decide.
- `budget-arena` — one check-time class, two instances, always "an
  allocation whose bytes no budget can own, written as though a budget
  owned them": **(i)** `forge.kv` / `stream.kv` inside a budgeted extent —
  Stream is not a region a `forge.reap` can release (§9); **(ii)**
  `vault.<ctor>[…] in b` — a budget is a region of Forge, so `b` cannot own
  vault-charged bytes. The complement is not a refusal, and it is the line
  that lets a tenant's weights work: `vault.<ctor>[…]` with **no** `in b`,
  inside `forge.enter(b) { … }`, charges the **root** on both backends and
  therefore survives `forge.reap(b)`. Its *arguments* do not — a shape
  expression or a fill is ordinary code in the entered extent, charged to
  `b` and invalid after the reap.
- A stale budget handle — naming a budget already reaped — is a
  **runtime error**, at carve, reap, enter, or `in b`. Never a check-time
  refusal: a `Budget` in a model field puts the reap and the use in
  different functions by design.

**Invalidity.** Every value allocated in `b` is **invalid after
`reap(b)`** — the same status a tensor already has after `forge.restore`
(§1.1). The language tracks no reference into a budget's bytes, so nothing
checks this: the host is expected to exit the tenant's processes before
reaping, and a value the host designates to survive a reap is *copied*
into the parent's budget and charged there, the original left
unreachable.

All four operations are built, on both backends: `Budget` is a real type,
`forge.root` names entry 0, `forge.carve` and `forge.reap` move the meter,
and `forge.enter(b) { … }` and `... in b` push the entered budget — the
first charging allocations inside a lexical extent, the second overriding
it for one constructor. A program that never carves runs on entry 0 and
behaves exactly as it did before any of this, which is the compatibility
gate each step held to. The acceptance program that asserts §1.1's
hierarchical rule end to end is `examples/tenant_budgets.dmc`: every budget
in it is carved at exactly what the code inside it needs, so each figure is
a refusal boundary bisected on both backends, and the refused half of each
section — which aborts rather than returning `false`, and so cannot live in
an example — is pinned in `tools/jit_probes.py`, with the two-sided bracket
in `compiler/tests/budgets_acceptance.rs`.

One property the acceptance program does **not** discharge, and which a host
embedding owes its own test: a budget entered either side of a HOST call is
unobservable from inside the language today — `dmc run` refuses to call an
`extern fn`, and `dmc jit`'s never re-enters demoniC — so the entered-budget
register has no host boundary here to survive.

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
`docs/DIRECTIVES.md §1` (Catalog). What *is* enforced is where it may be
written — `@inplace` on anything but an assignment statement is an
illegal stack (`docs/DIRECTIVES.md §3` (Stacking)), because nothing else
here holds a write to guard.

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
demoniC and never will be, except where a bullet below names its own
exception:

- A garbage collector of any kind.
- Reference counting (`Rc`, `Arc`, `shared_ptr`).
- Weak references.
- Finalizers, destructors, `Drop`, RAII.
- A free function in the general sense. `forge.reset()` is a reclamation
  primitive; so is `forge.reap(b)` *(§3.2)*, which releases a carved
  budget's region and credits its whole
  extent to the meter it was carved from *because* it releases it; so,
  once implemented, is `release(kv)` *(specified, not yet implemented —
  an undefined identifier today; §9)*. No other reclamation primitive
  exists.
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
