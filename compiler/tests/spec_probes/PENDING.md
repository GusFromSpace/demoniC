# Pending probes

Probes from the spec-coverage audit that don't pass cleanly today and
therefore aren't part of the active regression suite. Move one into the
suite once the underlying gap closes.

p31 (KV stream `<-` append) was activated: the interpreter concatenated
the two `<-` operands directly, which requires equal rank, so the
dropped-streaming-axis spelling §4.8 allows could never work. It lives
in the suite now as `p31_kv_stream_append.dmc`.

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
