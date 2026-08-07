# Pending probes

Two probes from the spec-coverage audit that don't pass cleanly today
and therefore aren't part of the active regression suite. Both surface
real edges, not malformed tests; move them into the suite once the
underlying gap closes.

## p31 — KV stream `<-` append

```dmc
fn main() -> nil {
    stream {
        let !cache: KV[f32, [4, ~, 8]] = forge.kv[f32, [4, ~, 8]](capacity = 64)
        let new_token = forge.zeros[f32, [4, 8]]
        cache <- new_token
    }
    nil
}
```

**Current state:** parse + check pass; interpreter rejects the append
with `ShapeError/IncompatibleShape`.

**Spec reference:** §3.6 declares `KV[T, S]` types with `~` axis markers;
§4.8 specifies that `c <- v` appends a view `v: View[T, S_inner]` where
`S_inner` is `S` with the streaming axis dropped. The probe shapes
match that contract — `KV[f32, [4, ~, 8]]` should accept a `[4, 8]`
appendee.

**Activation criterion:** interpreter accepts the canonical shape pair
and the cache's streaming-axis cursor advances by `v`'s extent. Once
that lands, move this file into the active suite.

## p41 — `@shard` axis must include mesh divisor

```dmc
fn main() -> nil {
    let mesh = Mesh[dp=2, tp=2]
    @shard(axis=0, mesh=mesh.dp)
    let x: Tensor[f32, [4, 768]] = forge.zeros[f32, [4, 768]]
    print(42); print("\n")
    nil
}
```

**Current state:** typechecker correctly rejects with `@shard axis 0
shape '4' must include divisor 'dp'`.

This is actually correct compiler behavior — the probe as written is
**malformed**. Spec §3.7 requires sharded shapes to express the divisor
in the type (e.g. `Tensor[f32, [4/dp, 768]]`). The probe is preserved
here as a record of the negative case; rewriting it to the positive
form (with the divisor in the shape) and confirming it runs cleanly
would make it activation-worthy.

**Activation criterion:** rewrite to use `[4/dp, 768]` shape and confirm
the resulting program lexes, parses, checks, and runs. Then move into
the active suite as a positive `@shard` test.
