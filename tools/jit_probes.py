#!/usr/bin/env python3
"""jit_probes.py — curated run/jit differential edge-case battery.

Complements `diff_backends.py` (which diffs whole example files): this is a hand-
written set of *tiny synthetic probes* hitting the edge cases real examples rarely
exercise — NaN/inf propagation, integer div/mod by zero, saturating casts,
negative and out-of-bounds indexing, degenerate/nonfinite reductions, matmul edge
shapes, bitwise/shift/pow corners. Each probe returns a scalar so the two
backends' outputs compare directly.

Classification per probe:
    * OK            — both backends succeed and agree
    * DIVERGE       — both succeed but DISAGREE  → a miscompile/semantics bug (FAIL)
    * jit-gap       — run ok, jit emits a clean "not lowered" error (informational;
                    allowlist the tracked ones below so only NEW gaps surface)
    * interp-refusal — the mirror image: run refuses, jit answers. A defect for
                    every construct not named in INTERP_REFUSAL — `extern fn`
                    (SPEC §9 REQUIRES it) and, separately, a small number of
                    known, FILED jit bugs tracked there until fixed
    * MUST-LOWER    — a jit-gap on a probe named in MUST_LOWER, i.e. a feature
                    the JIT is supposed to lower now (FAIL, not informational)
    * both-fail     — both reject (e.g. a deliberate trap probe); fine, and
                    counted (but see the gates below: a name in GAP_ALLOWLIST
                    that lands here is a dead entry, not an ordinary
                    both-fail, and a name in SAME_MESSAGE whose two refusals
                    no longer read alike is a WORDING-DIVERGE, not one either)
    * WORDING-DIVERGE — a both-fail probe named in SAME_MESSAGE whose two
                    backends' stderr texts do NOT match once normalised
                    (FAIL) — #867's gate: exit codes alone can't tell a
                    program `dmc run` and `dmc jit` refuse for the SAME
                    reason apart from one they refuse for two DIFFERENT
                    reasons, and #767/#865 found a real bug living in
                    exactly that blind spot

Exits non-zero iff a probe DIVERGES (a silent run/jit mismatch), a MUST_LOWER
probe is not lowered, a MUST_ANSWER probe has a backend that doesn't answer,
a SAME_MESSAGE probe's two refusals no longer read alike (WORDING-DIVERGE),
or a GAP_ALLOWLIST/INTERP_REFUSAL entry no longer matches the state its list
describes. The last is #771's gate: GAP_ALLOWLIST's every other branch is
gated on `rs == "ok"` ("run answers, jit refuses"), so a probe that regresses
to both-fail (e.g. a checker fix that makes `dmc run` itself start refusing
first) leaves a stale entry checked by nothing — reported as
`STALE-ALLOWLIST`. INTERP_REFUSAL is symmetric but was already gated: its own
arm at the top of the loop asserts "interp refuses, jit answers" against
every other outcome (including the interpreter now answering), scoring a
mismatch as an ordinary DIVERGE. This battery found #270 (negative index),
#271 (zero-return print), #272 (max/argmax) — keep it green and add a probe
whenever a new edge case is hardened.

    python3 tools/jit_probes.py [--dmc PATH] [--timeout SECS] [--verbose]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DMC = REPO / "compiler" / "target" / "release" / "dmc"

# #802: the `file_io_read_result_charged_to_forge_*` probes below exercise a
# REAL file round trip (write then read) to pin the Forge charge on a
# materialized `read_file` result — unlike the missing-path probes further
# down, which touch no file at all. The fixture lives under the OS temp dir,
# not the repo (so it can never show up as an untracked file) and not a
# probe's own `run()`-scoped `tempfile.TemporaryDirectory` (torn down before
# the fixture could be reused, and not known until after `PROBES`'s static
# source strings are already built). The name is a FIXED string, not salted
# with this process's pid: the two probes' FORGE_FLAG budgets below are
# bisected exactly against this path's own byte length (baked into the
# literal that `write_file`/`read_file` allocate for it on `dmc jit`), so a
# variable-length name would shift their minimum from run to run. A
# concurrent `jit_probes.py` run racing this one is content-compatible (both
# always write the identical 4096-byte payload before reading it back), so a
# shared fixed name costs nothing. `main()` removes it best-effort on exit.
FORGE_FILE_IO_FIXTURE = os.path.join(tempfile.gettempdir(), "dmc_jit_probes_802_file_io.txt")

# A jit "gap" is a clean REFUSAL to lower a feature (not a silent miscompile).
# #480: the JIT states which it is — `jit unsupported at L:C:` vs `jit error at
# L:C:` — so this matches one structural prefix instead of grepping the message
# for five English fragments. The old guess was wrong in both directions: a
# clean refusal worded differently ("`sum` is f32-only") was scored as a
# DIVERGENCE and could not be allowlisted, and a genuine miscompile whose text
# happened to contain "slice 1" would have been downgraded to a gap.
_UNSUPPORTED = "jit unsupported"
GAP_ALLOWLIST = {
    "pow_int": "#215 — integer `**` not lowered in the JIT",
    # #674 slice 3d: `for x in <ctor> in b { }` (design probe s5). The `in b`
    # override itself lowers — the JIT's refusal is `for-loop over
    # collections (only range iteration supported)`, which predates every
    # slice of 0674 and has nothing to do with budgets. Tracked here rather
    # than left an untracked gap so the day the JIT lowers for-over-a-tensor,
    # this line has to move.
    "design_probe_s5_for_ctor_in_budget_674":
        "#674 slice 3d — `<ctor> in b` lowers; the JIT's remaining refusal is its "
        "pre-existing for-over-a-collection gap (only range iteration is lowered)",
    "f64r_sum_keeps_width": "#481 follow-up — the JIT's `sum` is f32-only",
    # (All three `copy_*_mut_arg` probes have left this allowlist, and it took
    # both branches to empty it. #637 made `!` parameters lower outside a
    # `@grad fn`, which freed `copy_i64_mut_arg`; it noted that the other two
    # stopped one LAYER down, on their tensor element type
    # (`jit-element-type`), before mutability was ever considered, and that
    # each would go live on the day the JIT built tensors of that element type
    # — deliberately saying nothing about which types were on the roster,
    # because the roster moves. #632 is the branch that moved it: `f64` and
    # `bool` both have their own storage lane now, so both probes score OK on
    # both backends and neither is a gap to track.)
    # #650 — the two `!` forms the JIT REFUSES rather than lowers, pinned as
    # gaps so a later branch that starts lowering either has to say so here.
    # These are pins, not agreement: `dmc run`'s answer is the conformant one
    # in both, and the spec ruling in #650 decides whether the JIT implements
    # it or the language drops the form.
    "mut_param_fn_as_a_value": "#650 — a fn with a `!` param cannot be a fn-typed value: "
                               "`fn(T,T)->R` carries no `!`, so the indirect call site "
                               "has nothing to copy in and back out",
    "mut_model_param_rebind": "#650 — a `!` model/`KV` param is the caller's pointer, "
                              "passed by value: a wholesale rebind in the callee has no "
                              "slot to travel back in",
    # (#572's five narrow-tensor entries — `narrow_tensor_i8_wrap`,
    # `narrow_tensor_u8_wrap`, `narrow_tensor_i16_wrap`, `zeros_bool_tensor`
    # and `ones_i64_tensor` — left this allowlist in #632, which gave every
    # byte-aligned element type its own storage lane: they are scored OK on
    # both backends now, along with the probes added below them.)
    # #632 review: two refusals that became reachable once narrow elements
    # existed, reclassified from `err` (a divergence) to a clean `ElementType`
    # gap — the interpreter answers, the JIT declines.
    "elem_slice_assign_type_mismatch": "#632 — slice-assign refuses when src/dst element types differ",
    "port_tensor_u8_no_wire_dtype": "#632 — a `u8` tensor has no PORTS.md §3.2 copy-mode wire dtype in the JIT",
    # #818 — the underscore-dropping parser bug hid this gap behind a hard
    # parse error; now that `forge.zeros[fp8_e4m3, …]` parses, the JIT's
    # pre-existing refusal to allocate an `fp8_e4m3`/`fp8_e5m2` tensor is
    # visible and allowlisted rather than widened in this PR.
    "elem_fp8_e4m3_ctor_gap": "#818 — `fp8_e4m3` tensor element type is not lowered by the JIT",
    # #833 — `fp8_e5m2`'s sibling of the probe just above: #831 only added
    # `elem_fp8_e4m3_ctor_gap`, leaving this dtype's identical ctor gap
    # unprobed even though the parser fix (#818/#831) covers both dtypes
    # alike (they share one lexer rule and one `ScalarType` table).
    "elem_fp8_e5m2_ctor_gap": "#818 — `fp8_e5m2` tensor element type is not lowered by the JIT",
    # #632 review: integer tensor COMPARISONS were left out of the PR's
    # elementwise int lowering (only `.+ .- .* ./` were done) but were never
    # documented or tracked — untracked because nothing probed them.
    "elem_u8_dot_cmp_gap": "#632 — `.> .< .>= .<=` on integer tensor elements are still f32-only",
    # #563 — a colon slab needs literal bounds, and no stride, in the JIT.
    # Same story: `jit error` until #563, so the colon spelling was unprobeable.
    "slab_shape_param_end": "#563 — a colon slab's end must be a literal int in the JIT",
    "slab_strided": "#563 — strided slices (`a:b:c`) are not lowered in the JIT",
    "slab_open_end": "#563 — a range slice needs an explicit end in the JIT",
    # `use_import_scalar` was here, allowlisted as "temporary while the JIT is
    # single-file". The multi-file merge (slice 3) removed it: `dmc jit` now merges
    # the whole import DAG into one item list before lowering, so a `use` is not
    # a gap to track but a construct both backends answer. It is in MUST_LOWER
    # below, with the diamond and the alias probe.
    "cast_model_overlay": "#598 — `@cast(Model) { bytes }` is not lowered in the JIT: "
                          "the directive is a no-op and the field read is then refused",
    # #693 — the interpreter now computes `t[i] += v` (and `-=`/`*=`/`/=`) via
    # #689's in-place indexed-store path; the JIT is unchanged and still
    # refuses the form outright (`jit unsupported: compound assignment to
    # tensor elements`, jit.rs `Refusal::AssignForm`). Tracked gap, not a
    # divergence: the JIT's answer to this construct did not move.
    "indexed_tensor_compound_assign_693": "#693 — compound assignment to an indexed tensor "
                                           "element is computed by `dmc run`; `dmc jit` still "
                                           "refuses it (unchanged by this fix)",
    "direct_field_compound_assign_693": "#693 follow-up (#710 review) — compound assignment "
                                         "to a model field (no index) is computed by `dmc run`; "
                                         "`dmc jit` still refuses it (`compound assignment to a "
                                         "model field`, unchanged by this fix)",
    # #706 — the READ-side twin of #683's store guard. A tensor field of a
    # shape-generic model read through a type-arg-less parameter now refuses
    # (`jit-type-form`, naming the field and model) instead of hitting the
    # uncoded `jit error: indexing requires a tensor or str, got i64` (#480
    # class) it used to. Tracked here so a regression back to that jit-error
    # is a divergence, not a silent gap.
    "generic_field_read_type_arg_less_receiver": "#706 — a tensor field of a "
                          "shape-generic model, read through a type-arg-less "
                          "receiver, refuses (the receiver's shape args aren't "
                          "known). Slice 2 does NOT remove this, contrary to "
                          "what this note said when it was written: slice 2 "
                          "binds a method's shape env from the receiver's own "
                          "`type_args`, and this receiver has none. Measured "
                          "on the slice-2 branch — still refused, same class, "
                          "same message",
    # #718 rework (R5) — SPEC.md §3.5.1's JIT note: "Capturing closures are
    # supported in both `dmc run` and `dmc jit`." A #712-foldable LOCAL `let`
    # captured by a lambda and used as a shape dimension inside its body is
    # not one of those cases yet: the JIT's per-fn `local_const_dims` fold
    # table isn't visible inside a deferred lambda body, so the capture reads
    # as an ordinary (unfoldable) runtime value — `jit-dynamic-shape`, named
    # and classed. A tracked gap against the SPEC note, not a divergence
    # (`dmc run` closes over it normally); pinned here so a fix that starts
    # closing the fold table over into a lambda body removes this entry.
    "shape_local_let_capture_in_lambda_body_718": "#718 (SPEC.md §3.5.1) — a "
                          "foldable local `let` captured by a lambda is not "
                          "foldable as a shape dimension INSIDE the lambda "
                          "body; the enclosing fn's own shape parameters "
                          "already are (see `shape_param_in_lambda_body_dim`)",
    # #634, the model-methods design's slice 2 — the two method-call spellings
    # slice 2 deliberately refuses rather than guesses at. Both are `dmc run`
    # answers / `dmc jit` refuses, i.e. gaps; both are pinned with their class
    # and message in `compiler/tests/jit_model_methods.rs`, which is the only
    # place a message can be asserted.
    "method_generic_type_arg_less_receiver": "#634 slice 2 — a method called "
                          "through a type-arg-less receiver (`m: M` for `model "
                          "M[N]`) has no instantiation key; refused as "
                          "`jit-type-form` with #706/#709's wording rather "
                          "than bound to an invented shape (#474's shape)",
    "method_own_bracket_not_spelled": "#634 slice 2 — a method's OWN shape "
                          "param, inferable from an argument by `dmc run` but "
                          "not spelled at the call. The JIT does not infer a "
                          "method's bracket from arguments (they are lowered "
                          "inside `lower_call`; unifying against them first "
                          "would emit their side effects twice), so the "
                          "bracket-less spelling refuses by name",
    # #695 — a `let` binding the result of a fn with no declared return type.
    # The JIT's signature builder assumes such a fn returns nothing (right for
    # a call made only for effect, wrong the moment a caller keeps the
    # result), and used to hand the binding a bare `nil` and let a later
    # consumer (`map_get` was how #695 found it) discover the mistype as an
    # uncoded `jit error` with no code — #480's exact defect class. Now a
    # located, classified `jit-signature` refusal at the point of consumption,
    # naming the callee.
    #
    # #753 review: the three entries this comment used to carry
    # (`omitted_return_type_let_bound_call`, `omitted_return_provenance_block_shadow`,
    # `omitted_return_shape_generic_model_method`, probes still defined below)
    # are RETIRED, not just renamed. `compatible_with`'s new `(Unit, Map)`
    # carve-out (types.rs ~146) means the GENERAL call-arg mechanism now
    # refuses ANY `Unit`-typed value at `map_get`/`map_set`/`map_contains`'s
    # map-typed arg 0 on its own — and every omitted-return-type fn's call
    # site types as `Unit` (`register_fn`: `f.ret_type.unwrap_or(Unit)`), with
    # no exception. So the one shape these probes exist to reach — a
    # `--check`-clean program whose omitted-return call is consumed as that
    # arg 0 — no longer type-checks at all: `dmc run` now refuses it at the
    # SAME `--check` stage `dmc jit` does, before either backend's own
    # run/jit logic ever executes. All three probes now score `both-fail`
    # (`run=checkerr jit=checkerr`), which this script already classifies as
    # fine (`both-fail — both reject … fine`) — but their `GAP_ALLOWLIST`
    # entries were dead: every branch that consults this map is gated on
    # `rs == "ok"`, which a `checkerr` never is, so the entries were checked
    # by nothing and could go arbitrarily stale unnoticed.
    #
    # Rewriting the probes to reach the same JIT code path some OTHER
    # check-clean way was the preferred fix, but there isn't one: the JIT's
    # `omitted_ret_provenance`/`omitted_ret_arg_refusal` machinery (`jit.rs`)
    # is wired into exactly `map_get`/`map_set`/`map_contains`'s map-typed
    # arg — the position #753 just closed — and nowhere else, so no
    # alternative consumer reaches it. That machinery is consequently dead
    # code too (unreachable from any program the checker now accepts), which
    # is a JIT cleanup out of scope for this checker fix, not a probes-file
    # one — flagged separately, not silently dropped here.
    #
    # The three probes stay in `PROBES` below (retitled) as regression pins:
    # each still exercises this shape, and today scores `both-fail` because
    # it is refused before either backend runs. A regression that reopened
    # the `(Unit, Map)` wildcard would flip them back to `jit-gap` — and,
    # with no entry left here, an UNTRACKED one, printed as a warning line in
    # this script's own output (not a nonzero exit; `new_gaps` doesn't fail
    # the run) — still the visible signal a silently-restored allowlist entry
    # would have hidden.
    # #747/#755 — a non-generic model method whose OWN signature the JIT
    # cannot build (a `View` param, say) used to take EVERY sibling method
    # down with it (one method's `?` propagating out of the whole
    # `Item::Model` arm), then, once #747 isolated it, re-surfaced at a
    # bracket-less call site as a fabricated "shape bracket it does not
    # declare" — the wrong reason entirely, naming a bracket the call never
    # wrote. Now refuses as `jit-type-form`, the true `ty_from_ast` reason a
    # free fn with the identical param gets. #755 reworked the isolation
    # itself: it must not also swallow a genuine defect (two colliding
    # method declarations), and a broken method's own refusal must still
    # cost the whole PROGRAM in plain `dmc jit`, called or not — this probe
    # only exercises the called, single-method shape; the defect and
    # posture regressions have their own dedicated coverage in
    # `compiler/tests/jit_model_methods.rs`.
    "model_method_own_signature_refuses_honestly_747": "#747/#755 — a "
                          "non-generic model method whose own signature "
                          "cannot declare (a `View` param) refuses at a "
                          "bracket-less call site as `jit-type-form`, the "
                          "true reason — not a fabricated \"shape bracket it "
                          "does not declare\" and not the uncoded jit-error",
    # #721 — `map_set` used to coerce every value to i64 (silently, e.g. a bool
    # or f64 truncated, a tensor's pointer stored raw); only `str` refused. It
    # now refuses any non-i64 value with a located `jit-conversion` naming the
    # value's type, so a caller storing a float/bool/tensor gets a loud refusal
    # instead of a silently wrong number. #721 tracks the eventual typed-value
    # representation that would let this lower for real.
    "map_set_non_i64_bool": "#721 — `map_set` refuses a `bool` value as `jit-conversion` "
                          "rather than coercing it to 0/1",
    "map_set_non_i64_float": "#721 — `map_set` refuses an `f64`/`f32` value as "
                          "`jit-conversion` rather than truncating it to i64",
    "map_set_non_i64_tensor": "#721 — `map_set` refuses a tensor value as "
                          "`jit-conversion` rather than storing its raw pointer",
    # #742 — the same `map_set` value-only-integer arm above already caught a
    # `Map` value too (`as_scalar()` is `None` for `TyKind::Map`, same as
    # bool/float/tensor), but had no dedicated probe: `map_set(defs, "rec",
    # rec)` where `rec` is itself a map — one map stored as another's VALUE,
    # i.e. nested maps — found by `dmc test --jit --all` on the downstream
    # corpus right after #720 landed. `dmc run` answers fully (a map
    # holding another map is ordinary dynamically-typed behavior); `dmc jit`
    # refuses as `jit-conversion` naming `Map`, citing #721 — never the
    # uncoded `jit error`. `compiler/tests/jit_map_nested_742.rs` has the
    # dedicated coverage for this and the sibling key-side gap (a `map_get`
    # result used as a KEY), which this differential battery cannot probe on
    # its own: any real program reaching that KEY check with `dmc run`
    # succeeding would first have to store a genuine `str` value in a map,
    # which the pre-existing `str` refusal (unrelated to #742) always
    # intercepts first.
    "map_set_non_i64_map": "#721/#742 — `map_set` refuses a `Map` value "
                          "(nested maps) as `jit-conversion` rather than "
                          "storing its raw pointer",
    # #760 — the issue's own corpus shape: `let fns = map_get(d, "fns")
    # fns[i]`, a list value stored in a map and indexed after the round
    # trip. `map_get`'s result is always `TyKind::Scalar(OptI64)` in the
    # JIT (an opaque i64 payload, #721) no matter what the interpreter's
    # map actually held there, so indexing it used to fall through
    # `lower_index`'s tensor/str-only fallback to an uncoded `jit error`
    # (#480's exact defect class) — the same shape #742/#743/#754 already
    # closed for `map_get`/`map_set`/`map_contains`/`len`, just not yet for
    # indexing, slicing, element/slice assignment, arithmetic/comparison,
    # transpose, or the tensor-only builtin reductions (`sum`, `max`/`min`,
    # `argmax`/`argmin`, `variance`, `sum_along`/`mean_along`/`max_along`/
    # `min_along`, `softmax`, `print_tensor`). Every one of those now
    # special-cases `OptI64` first (`opt_i64_refusal`, `compiler/src/
    # jit.rs`) and cites #721, never the uncoded `jit error`; the located-
    # refusal coverage per site lives in `jit.rs`'s own `jit_760_*` inline
    # tests, not duplicated here — this probe only pins that `dmc run`
    # keeps succeeding while `dmc jit` keeps cleanly gapping.
    "map_get_result_indexed": "#760/#721 — indexing a `map_get` result "
                          "(`OptI64`) refuses as `jit-conversion` rather "
                          "than the uncoded `jit error` #760 reported",
    # #769 — the issue's own repro: a `map_get` result passed to a
    # `Tensor`-typed parameter. `dmc run` answers (the interpreter hands
    # the plain int straight to `sum`, which answers it back — its own
    # oddity, filed separately as #860, not this issue's fix); `dmc jit` used to
    # SEGFAULT (`coerce_to` decoded the `OptI64` payload and let it flow
    # into `(Scalar(I64), Tensor(_)) => Ok(v)`, a bitcast meant for a
    # genuine raw-pointer model-field initializer, reinterpreting the
    # map's plain int as a tensor data pointer). Now a located
    # `jit-conversion` citing #721, same shape as every other
    # `map_get`-result site #760/#764 audited.
    "map_get_result_tensor_param": "#769/#721 — a map_get result passed "
                          "to a `Tensor` parameter refuses as "
                          "`jit-conversion` instead of segfaulting",
    # PR #790 reviewer rework, defect 4: NOT a `jit unsupported` refusal but
    # a tracked FAILURE — `dmc jit` exhausts its `--forge` budget on a
    # program `dmc run` completes, at a budget deliberately chosen to sit
    # between the two backends' bisected minimums on 0674's g1-fwd source.
    # The 32 768 B between them (65 536 B until #640) is `MEMORY.md §1.1`'s
    # one-arena sentence made
    # measurable: the JIT lowers `vault.*` and `forge.*` into a single Forge
    # arena, so `main`'s two `vault.zeros` are charged against `--forge`
    # there while the interpreter charges them against `--vault`. Expected
    # and explained, not a defect of this PR; the probe exists so the day
    # either minimum moves, this row stops being a tracked gap — and goes
    # silently `ok` (an allowlisted gap that starts passing is dropped, not
    # reported), which is why the re-bisection after #640 was manual and
    # load-bearing: the flag was re-bisected to one byte under the new
    # JIT minimum.
    "meter_ad_temporaries_fwd_below_jit_min":
        "`MEMORY.md §1.1` — the JIT's one Forge arena charges `main`'s two "
        "`vault.zeros` against `--forge` where `dmc run` charges them against "
        "`--vault`: 32 768 B, the whole gap between the two backends' g1-fwd "
        "minimums (65 536 B until #640 stopped doubling each of them). "
        "`dmc run` answers at this budget; `dmc jit` exhausts",
    # #810 traded-away class ("q3", reviewer pass): `bind_or_refuse_model_shape_args`
    # refuses a bare model argument purely on its TYPE (no shape args to check
    # against the callee's own shape-parameterized annotation), with no way to
    # tell whether the callee's body ever actually reads that argument's shape.
    # This probe's callee never does — only its OTHER (tensor) parameter binds
    # `M` — so before #810 it lowered and agreed with the interpreter (`=> 1.5`).
    # Acceptable per #810's own doctrine (a model value carries no runtime shape
    # for the JIT to fall back on checking, whether or not anything would ever
    # read it), but a real loss, tracked here rather than left silent.
    "bare_model_arg_shape_unused_by_callee_body_810":
        "#810 q3 — a bare model argument the callee's body never actually reads "
        "the shape of still refuses unconditionally: the JIT has no way to tell "
        "a harmless bare argument from the real out-of-bounds case #810 fixed. "
        "`dmc run` and the pre-#810 JIT both answered 1.5",
    # #811 rework (reviewer pass): `generic_view_param_kv_arg_811` used to be
    # a `MUST_LOWER` row — see the comment above that dict's Tensor-argument
    # entry for the full story. Binding a `View` param's streaming-axis
    # dimension from a `KV` argument's compile-time `capacity` answers a
    # plausible but WRONG number the moment the stream is not filled to
    # capacity (the normal case, not an edge case): `dmc run` reads the real
    # runtime length, `cap` reads the upper bound, and the two silently
    # disagree on `lenof(k)` and on every element `dump(k)` prints past the
    # fill point. The fix refuses that binding unconditionally
    # (`Refusal::DynamicShape`), known `cap` or not, rather than lower a
    # wrong answer (#480) — so this is now a tracked `jit-gap`, not a
    # MUST_LOWER failure, and is expected to stay one: there is no
    # compile-time fix for a value the JIT cannot read until runtime.
    "generic_view_param_kv_arg_811":
        "#811 rework — a `View` param's streaming-axis dimension bound to a "
        "`KV` argument now refuses unconditionally (`jit-dynamic-shape`): the "
        "JIT cannot tell a KV filled to capacity from one only partway filled, "
        "and the constructor's `capacity` is silently the wrong answer for the "
        "latter, the normal case. `dmc run` answers (a gap, not a divergence)",
}

# #578: probes where the INTERPRETER refuses and the JIT answers — the mirror
# image of a jit-gap. For `extern fn` this is the *specified* behavior, not a
# defect: SPEC.md §9 requires an implementation without a JIT backend to reject
# `extern fn` calls at runtime, so `dmc run` refusing and `dmc jit` returning a
# value is the contract being kept. Without this class the battery scored such
# a probe as a DIVERGENCE (the catch-all `else` below), which is why the JIT's
# `extern fn` lowering had no probe coverage at all.
#
# #810 briefly extended this list for a second, different case the same
# mechanism also fit: a KNOWN, FILED bug where the interpreter's refusal was
# correct and the JIT's success was the wrong answer, tracked here rather
# than left an untracked DIVERGE until fixed. That entry (a bare model
# parameter forwarded to an explicitly bracketed call) is gone now that the
# bug is fixed — the JIT refuses too, so the probe
# (`bare_model_arg_against_bracketed_callee_810`) moved to `PROBES`'
# ordinary both-fail set and needs no allowlisting here.
#
# Still deliberately an explicit allowlist and not a rule: "run failed, jit
# succeeded" is a real defect for every construct not named here, filed or
# not, and it must stay a divergence there.
INTERP_REFUSAL = {
    "extern_libc_abs": "SPEC §9 — the interpreter must reject an `extern fn` call",
    "extern_libc_sqrtf": "SPEC §9 — the interpreter must reject an `extern fn` call",
    "extern_tensor_ptr": "SPEC §9 — the interpreter must reject an `extern fn` call",
    # #674 slice 3c: `forge.root` now lowers on the JIT (entry 0 of the
    # budget table) while the interpreter still carries 3a's placeholder
    # refusal ("not yet implemented ... slice 3b") — the interpreter's own
    # budget table is slice 3b's, not this one's, work. A TRACKED gap, not a
    # divergence: once 3b lands, `dmc run` answers too and both backends
    # refuse (or answer) at the SAME PRE boundary, and this probe moves back
    # to the ordinary both-fail/OK set the way `bare_model_arg_against_
    # bracketed_callee_810` did above when its own bug was fixed. `forge.
    # carve`/`forge.reap` are NOT here: 3c's own well-formedness check on a
    # non-`Budget` parent/`Budget` argument (the two probes just above)
    # still fails the JIT before it ever reaches the runtime call, so those
    # two stay `both-fail` unchanged.
}

# Probes the JIT must LOWER, not merely fail to miscompile. GAP_ALLOWLIST's
# default verdict for "run answers, jit refuses" is `jit-gap` — a printed line,
# exit 0 — which is right for a construct nobody has claimed yet and useless as
# a gate for one that just landed: every arena-block probe below would have
# passed this battery unchanged on the day the JIT refused all ten of them. A
# name here turns that verdict into a failure, so a refusal coming back is a red
# battery instead of a warning nobody reads.
#
# The other two harnesses cannot carry this weight either: `dmc test --jit`
# scores a file the JIT refuses as *skipped* and exits 0, and `diff_backends.py`
# compares outputs of programs that ran. This set is the only place a refusal is
# an error.
#
# A name may not appear in both this set and GAP_ALLOWLIST — they say opposite
# things about the same probe — and every name must be a real probe; `main`
# refuses to run otherwise.
MUST_LOWER = {
    "arena_vault_expr_scalar":    "#612 — arena blocks lower as lexical scopes",
    "arena_stream_expr_scalar":   "#612 — arena blocks lower as lexical scopes",
    "arena_forge_stmt":           "#612 — arena blocks lower in statement position",
    "plain_block_stmt":           "#612 — a valueless block in statement position",
    "arena_nested_forge_vault":   "#612 — nested arena blocks lower",
    "arena_vault_tensor_read":    "#612 — a tensor allocated in a block outlives it",
    "arena_forge_tensor_escapes": "#612 — a tensor allocated in a block outlives it",
    "arena_escapee_is_writable":  "#612 — and stays writable after the block",
    "arena_vault_write_in_block": "#612 — the #442 cross-arena write, legalised",
    "arena_cross_write_nested":   "#612 — innermost-wins nesting (MEMORY §3.1)",
    "arena_block_shadowing_let":  "#612 — an arena block is a scope (#630, fixed by #642)",
    # The multi-file merge (slice 3) — `dmc jit` merges the import DAG into one
    # item list and lowers it. The same reasoning as #612's ten: these three
    # were refusals until the slice landed, and a refusal coming back must be a
    # red battery rather than a `jit-gap` line nobody reads.
    #
    # What is NOT here, deliberately: a probe pinning a merge COLLISION — two
    # imported files declaring one `extern fn` with two signatures, which the
    # merge refuses as `jit-ineligible` / `jit-import`. The battery has no way
    # to say "refused with class X": `GAP_ALLOWLIST` says "a gap nobody has
    # closed yet", which a permanent, deliberate refusal is not, and
    # `MUST_LOWER` says the opposite of what is wanted. Putting it in either
    # would make the battery assert something false. That pin lives in
    # `compiler/tests/jit_multifile_merge.rs`
    # (`a_merge_collision_is_refused_as_jit_ineligible_not_as_a_jit_error`),
    # where the class code itself can be asserted.
    "use_import_scalar":          "the multi-file merge — a `use` lowers through the merge",
    "use_import_diamond":         "the multi-file merge — a shared leaf's `pub model` across two edges",
    "use_import_alias":           "the multi-file merge — `use ... as` resolves to the merged name",
    # #635, the model-methods design's slice 0 — the whole-field scalar store
    # `m.f = v`. Both were a `jit-assign-form` refusal before the slice; a
    # refusal coming back must be a red battery, not an informational gap.
    "field_store_scalar_local":          "#635 — a whole-field scalar store on a local",
    "field_store_scalar_via_mut_param":  "#635 — the same store through a `!` model parameter",
    "field_store_nested_model_local":    "#683 follow-up — a whole-field store of a non-generic model VALUE (#474's citation was stale; see #800 for the shape-generic case that stays refused)",
    # #634, the model-methods design's slice 1 — a method on a model with no
    # shape parameters, lowered as the free function `M$$method`. All ten were
    # a `jit-unknown-fn` refusal before the slice, and the refusal named the
    # str-method family, so a regression here would come back looking like a
    # string diagnostic on a model receiver. `#635`'s two above are a
    # prerequisite for six of them, not a follow-on: a mutating method's body
    # IS a whole-field store.
    "method_read_scalar_field":          "#634 — p1: a method reads a scalar field",
    "method_mutate_scalar_field":        "#634 — p2: `!self` mutates a scalar field, visible to the caller",
    "method_mutate_tensor_element":      "#634 — p3: `!self` writes a tensor-field element",
    "method_call_through_self":          "#634 — p6: a method calls another through `self`",
    "method_mut_tensor_arg_bare_ident":  "#634 — p7: a `!` tensor argument writes back (#475's shape)",
    "method_call_via_mut_param":         "#634 — p8: the corpus's shape, `fn op(!g: Graph) { g.step!() }`",
    "method_call_nested_field_receiver": "#634 — p9: the receiver is another model's field",
    "method_precedence_over_free_fn":    "#634 — p10: SPEC §4.12 — the method wins over a same-named free fn",
    "method_recursion":                  "#634 — p12: a method recurses through `self`",
    "method_aliased_mut_params":         "#634 — p17: one instance through two `!` names is one instance",
    # #634, the model-methods design's slice 2 — a method on a SHAPE-GENERIC
    # model, one declaration per distinct receiver `type_args` tuple. These
    # three are `MUST_LOWER` and not allowlist entries for the reason §5 gives
    # in as many words: #474 and #475 were both *silent wrong answers*, and
    # this battery scores a refusal as a warning and only a divergence as a
    # failure. `method_generic_two_instantiations` is the #474 regression gate
    # — a shared body across two instantiations answers 808, the right one
    # answers 816, and nothing else in the tree would notice.
    "method_generic_shape_param":         "#634 — p5: a method reads the model's shape param, bound from the instance",
    "method_generic_two_instantiations":  "#634 — p16: two instantiations, two bodies (#474's regression gate)",
    "method_generic_bracketed_call":      "#634 — q13: a method's own bracket, spelled at the call (#671/#679)",
    # #708, `SPEC.md §3.10` — `m(x)` desugars to `m.forward(x)` on a model
    # instance. Neither slice 1 nor slice 2's design text covers the
    # call-as-function spelling (the callee arrives as a bare `Ident` under
    # `Call`, never `Field`-then-`Call`), so it was reachable through
    # `lower_call`'s callee match falling through to `jit-unknown-fn: unknown
    # function \`l\`` until this fix.
    "method_forward_desugar_call": "#708 — SPEC §3.10: `l(7)` desugars to `l.forward(7)`",
    # The generic-model-params design's slice 0 — `SPEC.md §3.10`'s "an
    # argument's own shape arguments bind the parameter's shape names", which
    # `unify_param_type`'s model arm never implemented for the spelling anyone
    # writes (`Buf[N]` in a type position is a `TypeArg::Type`, and the arm
    # skipped those). Each was a `jit error` on a program `dmc run` answers.
    "generic_model_param_inferred_bracket": "slice 0 — g15: `N` inferred from a model argument, at two instantiations",
    "generic_model_param_from_literal":     "slice 0 — g23: the argument written as the literal, not a binding",
    "generic_model_param_single_call":      "slice 0 — g24: the minimal inference, a single call",
    # #695 — a lowercase `map` signature and the bare `map()` constructor
    # alias both used to refuse (or, for the former, mistype and surface as
    # an uncoded `jit error` at the first `map_get` downstream). Both now
    # lower; a refusal coming back must be a red battery, not an
    # informational gap.
    "map_lowercase_type_param": "#695 — a `map`-typed (lowercase) signature lowers",
    "map_bare_ctor_alias":      "#695 — `map()`, `map_new()`'s alias, lowers",
    # #751 / #754 review — `len(m)` (map entry count) and `len` of a tuple
    # (static arity) both lower now; a refusal coming back must be a red
    # battery, not an informational gap. (`map_len`/`map_del` as standalone
    # builtin NAMES, and a `map`-typed model FIELD, remain unknown-fn/refused
    # — that is the separate, still-open #757, not either of these two.)
    "len_of_map":               "#751 — len(m) lowers to the entry count",
    "len_of_tuple":             "#754 — len of a tuple lowers to its static arity",
    # PR #790 reviewer rework, defect 1: the JIT already emits
    # `emit_mut_param_copy_out` on every path out of a callee, `?` early
    # return included, and this probe is here to keep it that way — the
    # interpreter's regression on the same shape is what the probe was
    # written from.
    "mut_param_writeback_survives_query_early_return":
        "#781 — a `!` param's copy-out is emitted on every exit path, `?` included",
    # File I/O (STDLIB.md §3.10) now lowers as JIT host calls:
    # `read_file`/`write_file`/`append_file`/`file_exists` used to refuse
    # `jit-unknown-fn` before this change; a refusal coming back is now a red
    # battery, not an informational gap. `read_bytes` is deliberately absent
    # — its `Tensor[i64,[N]]` result would need `TyKind::DynTensor`, which
    # exists but is reachable only from KV reads and isn't plumbed for an
    # i64-element host-call result, so it stays a `jit-unknown-fn` refusal
    # and is not probed here as a MUST_LOWER name.
    "file_io_write_missing_dir":  "file builtins lowering — `write_file` reaches __dmc_write_file and returns a real (nil, err) pair",
    "file_io_append_missing_dir": "file builtins lowering — `append_file` reaches __dmc_append_file and returns a real (nil, err) pair",
    "file_io_read_missing_file":  "file builtins lowering — `read_file` reaches __dmc_read_file and returns a real (nil, err) pair",
    "file_io_file_exists_false":  "file builtins lowering — `file_exists` reaches __dmc_file_exists and answers a real bool",
    # #811 — `unify_param_type` had no `Type::View` arm at all, so a
    # shape-generic `fn total[N](v: View[T, [N]])` could never bind `N` under
    # `dmc jit` — every call refused with "a template parameter outside
    # scalar / Tensor / View / KV", bracketed or not. Fixing that arm alone
    # was moot: `declare_template_signature` converts every param type
    # through the same `ty_from_ast` a non-generic `View` param still refuses
    # at (#747/#755), so this PR also gives `declare_template_signature`'s
    # own parameter conversion a `View[T,S]` → `Tensor[T,S]` rewrite
    # (`view_param_ty_for_template` in `jit.rs`) — the JIT has no runtime
    # representation for a View distinct from a contiguous Tensor. This probe
    # was `jit unsupported` before this fix, on a program `dmc run` has
    # always answered, so it is MUST_LOWER.
    #
    # The `KV`-argument sibling (`generic_view_param_kv_arg_811`) is NOT
    # here, and deliberately so — see its `GAP_ALLOWLIST` entry below. A
    # reviewer rework of this same issue found that binding the streaming
    # axis from `KV.cap` (this arm's first cut) answers a plausible-looking
    # but WRONG number for the normal case, a KV filled to less than its
    # declared capacity: `dmc run` reads the real runtime length, the old
    # `cap`-based JIT arm read the constructor's upper bound instead, and the
    # two silently disagreed on both `lenof(k)` and every element `dump(k)`
    # prints past the fill point. The fix refuses that binding
    # unconditionally (`Refusal::DynamicShape`) rather than lower a wrong
    # answer (#480), so the KV-argument probe is a tracked `jit-gap`, not a
    # MUST_LOWER row, even though `total(t)` (the Tensor-argument row above)
    # keeps lowering.
    "generic_view_param_tensor_arg_811": "#811 — `N` inferred from a `Tensor[u8,[4]]` argument to a `View[u8,[N]]` param",
}

# Two of the design's slice-1 rows are deliberately NOT probes here, and the
# reason is the same in both cases: after slice 1 they are DIVERGENCES, and
# `main` has no allowlist on that branch at all — a name in `GAP_ALLOWLIST` or
# `MUST_LOWER` would make the battery assert something false rather than
# something red.
#
#   p11 — a method returns a field tensor; `let !v = b.peek()` writes through
#         it. `dmc run` answers `0`, `dmc jit` answers `6`. The design predicted
#         agreement, but the divergence is not the method's: the identical
#         program with no model and no method — a plain tensor handed to a free
#         function and returned — diverges the same way on a binary built before
#         slice 1. Slice 1 only makes it reachable through a method.
#
#         COORDINATOR WAIVER (PR #705): pending as #707 — a ruling question,
#         the return-side clause of #650 (does a returned tensor
#         alias its source, or copy?). Waived out of `MUST_LOWER` rather than
#         fixed here: the divergence is pre-existing on `main` through the
#         free-function spelling alone, so slice 1 did not introduce it, and
#         #650 covers passing a tensor in, not returning one out.
#         Tracked divergence, not a gap: the free-function half is the pin.
#
#         #708 review: `m(x)`'s forward-desugar reaches this same root too —
#         `let !v = b(0)` on a `forward` that returns `self.t` answers `0`
#         under `dmc run` and `6` under `dmc jit`, identically to the
#         `.forward(..)`-spelling pin above. Not a new divergence: `lower_call`
#         hands the desugared call to the same `lower_model_method_call` this
#         pin already goes through, so the call-as-function spelling inherits
#         #707 rather than needing its own waiver. Pinned as
#         `call_desugar_reaches_the_returned_tensor_divergence_p11_707`.
#   p14 — a `!` tensor argument that is NOT a bare identifier (`g.heat`).
#         `interp.rs`'s `apply_writebacks` drops it; the JIT's copy-in/copy-out
#         does not. Design §5 rules the INTERPRETER the incorrect side, per
#         `SPEC.md §3.4` and the existing pin for the free-function spelling
#         (`a_mut_tensor_arg_that_is_not_an_identifier_still_diverges`). It
#         cannot enter `MUST_LOWER` until #636 is fixed.
#
#         #708 review: `m(x)`'s forward-desugar reaches this same root as
#         well — `m(g.heat)`, a `!` tensor argument that is a field read, not
#         a bare identifier, answers `100` under `dmc run` and `109` under
#         `dmc jit`. Same #636 root, same reason it stays out of
#         `MUST_LOWER`: the call-as-function spelling desugars to the ordinary
#         call machinery this pin already covers, so it is reached, not
#         introduced. Pinned as
#         `call_desugar_reaches_the_non_ident_mut_arg_divergence_p14_636`.
#
#         A second shape of the SAME #636 root, found reviewing slice 1: a
#         method forwarding its OWN tensor field to a `!` param one call
#         deeper — `fn go!(!self) { fill(self.t) }` where `fill(!t: ..)`
#         mutates `t`. `dmc run` answers `0`, `dmc jit` answers `909`. Not
#         method-specific either: the identical shape with no model method at
#         all — a plain `fn go!(!m: M) { fill(m.t) }` called from `main` —
#         diverges identically (`0` / `909`) on this same binary, so it is
#         `self.t`'s `Postfix{Field}` shape hitting `apply_writebacks`'s
#         bare-`Expr::Ident`-only check, not anything slice 1 adds. Tracked
#         divergence citing #636, same root as p14, kept separate because the
#         call is two frames deep through a written-back `!` receiver rather
#         than passed directly from `main`.
#
# All three, and their two `m(x)`-desugar twins, are pinned in
# `compiler/tests/jit_model_methods.rs` and in `jit.rs`'s own tests
# respectively, where the answers can be written down.

# name -> (source, note on what it probes). Every program returns a scalar.
PROBES = {
    # integer edges
    "int_div_floor_neg":  ("fn main()->i64{ (0-7)/2 }", "truncated division, negative"),
    "int_mod_neg":        ("fn main()->i64{ (0-7)%3 }", "truncated modulo sign follows dividend"),
    "int_div_by_zero":    ("fn main()->i64{ let !z=0  10/z }", "div-by-zero guard → 0, not SIGFPE"),
    "int_mod_by_zero":    ("fn main()->i64{ let !z=0  10%z }", "mod-by-zero guard"),
    "int_shl_63":         ("fn main()->i64{ let !n=63  1<<n }", "max in-range shift"),
    # i32 operands: the count is checked at the operand's width, not always 64
    # (`OPERATORS.md §8a`). Newlines are load-bearing — `= 2  (a<<n)` parses as
    # a call on the literal.
    "int_shl_i32":        ("fn main()->i64{\n let a:i32=256\n let n:i32=2\n (a<<n) as i64\n}", "i32 `<<`, count checked at 32 bits"),
    "int_shr_i32":        ("fn main()->i64{\n let a:i32=-8\n let n:i32=1\n (a>>n) as i64\n}", "i32 `>>`, arithmetic and 32-bit"),
    # Narrow shifts that LOSE bits. These used to be unprobeable: the
    # interpreter carried every integer as an i64, so a shift out of a narrow
    # range had no answer the two backends shared. Both narrow at the declared
    # width now, which makes the lossy cases the interesting ones — each result
    # below is reachable only if the shift really ran at the operand's width,
    # and a backend that fell back to 64 bits answers something else entirely.
    "int_shl_i32_lossy":  ("fn main()->i64{\n let a:i32=1\n let n:i32=31\n (a<<n) as i64\n}", "i32 `<<` into the sign bit → i32 MIN, not 2^31"),
    "int_shr_i32_sign":   ("fn main()->i64{\n let a:i32=-2147483648\n let n:i32=31\n (a>>n) as i64\n}", "i32 `>>` copies the 32-bit sign bit → -1"),
    "int_shr_u32_logical":("fn main()->i64{\n let z:u32=0\n let o:u32=1\n let t:u32=2\n ((z-o)>>t) as i64\n}", "u32 `>>` is LOGICAL → 1073741823, not -1"),
    "int_shl_u16_wrap":   ("fn main()->i64{\n let a:u16=65535\n let n:u16=1\n (a<<n) as i64\n}", "u16 `<<` drops the top bit → 65534"),
    "int_shr_i8_arith":   ("fn main()->i64{\n let a:i8=-8\n let n:i8=1\n (a>>n) as i64\n}", "i8 `>>` is arithmetic at 8 bits → -4"),
    "int_shr_63":         ("fn main()->i64{\n let !n=63\n (1<<n)>>n\n}", "the sign bit shifted back is -1, not 1"),
    # #597: `u64` — a 64-bit hex mask with a `u64` suffix must lex/check clean
    # bound to a declared `u64` return, and `/ % >> < >` must be UNSIGNED even
    # though the value's own i64-backed register can be negative (the top bit
    # set). `u8/u16/u32` never exercised this because their zero-extended
    # pattern is always non-negative; `u64` is the first width with no spare
    # bits to hide the sign in.
    "int_lex_u64_mask":   ("fn main()->u64{ 0xffffffffffffffffu64 }", "64-bit hex mask w/ u64 suffix lexes/checks clean → u64::MAX"),
    "int_div_u64":        ("fn main()->u64{\n let x:u64=0xffffffffffffffffu64\n let two:u64=2\n x/two\n}", "u64 `/` is UNSIGNED → 0x7fffffffffffffff, not 0"),
    "int_mod_u64":        ("fn main()->u64{\n let x:u64=0xffffffffffffffffu64\n let two:u64=2\n x%two\n}", "u64 `%` is UNSIGNED → 1"),
    "int_cmp_u64":        ("fn main()->bool{\n let x:u64=0xffffffffffffffffu64\n let o:u64=1\n x>o\n}", "u64 `>` is UNSIGNED: u64::MAX > 1 → true, not false"),
    "int_shr_u64_logical":("fn main()->u64{\n let x:u64=0x8000000000000000u64\n let o:u64=1\n x>>o\n}", "u64 `>>` is LOGICAL → 0x4000000000000000, not 0xc000000000000000"),
    # #597 (review round 2): the DECIMAL spelling of a u64 above i64::MAX, the
    # float conversion, and `abs`. Each read the pattern SIGNED (or, for the
    # decimal literal, refused to lex at all) on BOTH backends, so the parity
    # gate stayed green while both were wrong together — which is exactly the
    # class of defect this probe table exists to stop being invisible.
    "int_lex_u64_decimal":("fn main()->u64{ 18446744073709551615u64 }", "decimal u64 literal above i64::MAX lexes → u64::MAX"),
    "int_u64_to_f64":     ("fn main()->f64{\n let x:u64=0xffffffffffffffffu64\n x as f64\n}", "u64→f64 reads the UNSIGNED magnitude → 1.8446744e19, not -1"),
    "int_abs_u64":        ("fn main()->u64{\n let x:u64=0xffffffffffffffffu64\n abs(x)\n}", "abs of a u64 is the IDENTITY → u64::MAX, not 1"),
    # #597 (review round 3): giving `u64` its own ScalarKind dropped it out of
    # the JIT's untyped-literal ADOPTION path (which gated on "I32 or a masked
    # narrow kind") and out of two `coerce_scalar` arms it used to reach as
    # `I64` — so `a + 1` at a u64, and `as i32` / `as bool`, hard-errored under
    # `dmc jit` on programs `dmc run` computes. Every existing #597 probe and
    # test typed BOTH operands, which is why the regression was invisible.
    "int_lit_adopt_u64":  ("fn main()->u64{\n let a:u64=10u64\n a+1\n}", "an untyped literal adopts u64 from the other operand → 11"),
    "int_lit_adopt_u64_l":("fn main()->u64{\n let a:u64=10u64\n 1+a\n}", "...on the LEFT too → 11"),
    "int_lit_adopt_u64_div":("fn main()->u64{\n let x:u64=0xffffffffffffffffu64\n x/2\n}", "an adopted literal keeps the UNSIGNED reading → 0x7fffffffffffffff"),
    "int_u64_to_i32":     ("fn main()->i32{\n let x:u64=0xffffffffffffffffu64\n x as i32\n}", "u64→i32 keeps the low 32 bits → -1"),
    "int_u64_to_bool":    ("fn main()->bool{\n let x:u64=0xffffffffffffffffu64\n x as bool\n}", "u64→bool tests the pattern against zero → true"),
    "int_big_mul_wrap":   ("fn main()->i64{ 1000000000 * 1000000000 }", "i64 multiply wrap"),
    "int_add_overflow":   ("fn main()->i64{ let a:i64=9223372036854775807  a+1 }", "#300 add overflow wraps (both backends)"),
    "int_sub_overflow":   ("fn main()->i64{ let a:i64=-9223372036854775807-1  a-1 }", "#300 sub overflow wraps to MAX"),
    # float -> int casts
    "cast_big_float":     ("fn main()->i64{ 1.0e18 as i64 }", "large float cast"),
    "cast_inf":           ("fn main()->i64{ (1.0/0.0) as i64 }", "inf cast saturates"),
    "cast_neg_float":     ("fn main()->i64{ (0.0-3.9) as i64 }", "negative float truncates toward zero"),
    "cast_nan":           ("fn main()->i64{ (0.0/0.0) as i64 }", "NaN cast saturates to 0"),
    "cast_scalar_f32":    ("fn main()->f64{ (0.1 as f32) as f64 }", "#300 scalar as-f32 rounds (both backends)"),
    # map_get on a missing key reads as nil (#300.4)
    "map_miss_is_nil":    ("fn main()->i64{ let !m=map_new()  if map_get(m,\"k\")==nil {1} else {0} }", "#300 map miss == nil (both backends)"),
    "map_hit_value":      ("fn main()->i64{ let !m=map_new()  map_set(m,\"k\",7)  map_get(m,\"k\") }", "#300 map hit returns the value"),
    "map_hit_arith":      ("fn main()->i64{ let !m=map_new()  map_set(m,\"k\",7)  map_get(m,\"k\") + 1 }", "#300 map hit usable in arithmetic"),
    # #767 — a missing key is `nil` (#300.4 above), and `scalar_compare`'s own
    # blanket `(Nil, _) | (_, Nil)` rule makes `==`/`!=` LEGAL comparisons
    # against a `nil` operand (`Eq` false, `NotEq` true) — no runtime error,
    # in either backend. `dmc jit` used to trap here anyway, with the same
    # unlocated message every other operator on an absent `map_get` result
    # got, a divergence `make parity` never caught (no probe exercised the
    # comparison-against-a-concrete-literal shape — only `map_miss_is_nil`
    # above, which compares against the literal `nil`, already worked before
    # this fix). Ordinary OK/OK rows now: both backends genuinely answer.
    "map_get_eq_absent_no_trap": (
        "fn main()->i64{ let !m=map_new()  map_set(m,\"other\",5)  \
if map_get(m,\"missing\")==3 {1} else {0} }",
        "#767 map miss == 3 answers false, no trap (both backends)"),
    "map_get_neq_absent_no_trap": (
        "fn main()->i64{ let !m=map_new()  map_set(m,\"other\",5)  \
if map_get(m,\"missing\")!=3 {1} else {0} }",
        "#767 map miss != 3 answers true, no trap (both backends)"),
    # `+` (any operator OTHER than `==`/`!=`) on an absent key is NOT this
    # divergence: the interpreter itself errors there (`scalar_arith`'s
    # "requires numeric operands"), so both backends correctly refuse —
    # `both-fail`, not a gap. Pinned so a future change to the located-trap
    # wording (#480) has a probe watching it, same shape as the issue's own
    # repro.
    "map_get_add_absent_both_fail": (
        "fn main()->i64{ let !m=map_new()  map_set(m,\"other\",5)  \
map_get(m,\"missing\")+1 }",
        "#767 map miss + 1 still refuses on both backends (both-fail, unchanged)"),
    # #695 — `map` (lowercase) is the type's real spelling (`types.rs` parses
    # only that; a signature written `Map` never passes `--check`) and `map()`
    # is `map_new()`'s alias (SPEC.md's Map-type paragraph). Both were missing
    # from the JIT before #695: a lowercase `map` signature read as an unknown
    # model named "map", and the bare `map()` call read as an unknown function
    # — either way a program `dmc run` accepts refused (or worse, #695's
    # `map_get` mistyping) under `dmc jit`.
    "map_lowercase_type_param": (
        "fn get_k(t: map) -> i64 { map_get(t, \"k\") }\n"
        "fn main() -> i64 { let !m = map_new()  map_set(m, \"k\", 7)  get_k(m) }",
        "#695 a `t: map` (lowercase) parameter lowers and agrees with `dmc run`"),
    "map_bare_ctor_alias": (
        "fn main() -> i64 { let !m = map()  map_set(m, \"k\", 7)  map_get(m, \"k\") }",
        "#695 the `map()` alias for `map_new()` lowers and agrees with `dmc run`"),
    # #751 — `len(m)` on a map lowers to the entry count (`__dmc_map_len`,
    # the same backing `HashMap` `dmc run`'s `Value::Map` reads), landed
    # alongside a located `jit-conversion` refusal for the non-map/tensor/str
    # cases the JIT can't verify (was an uncoded `jit error`, #480's exact
    # defect class). No probe covered this when #751 landed; MUST_LOWER below.
    "len_of_map":         ("fn main()->i64{ let !m=map_new()  map_set(m,\"k\",7)  len(m) }",
                           "#751 len(m) is the entry count (both backends)"),
    # #754 review of #751 — a tuple's arity is static (`iconst`, no runtime
    # call), matching `dmc run`'s `Value::Tuple(vs) => vs.len()`. This used
    # to fall through to the same non-map/tensor/str refusal `len(true)`
    # rightly gets, rejecting a program `dmc run` answers correctly.
    "len_of_tuple":       ("fn main()->i64{ len((1,2,3)) }",
                           "#754 len of a tuple is its arity (both backends)"),
    # #695 — `dependency` has no `-> T`; `dmc run` used to return its map fine
    # regardless (the interpreter doesn't care about declared return types),
    # while the JIT's signature builder assumed an omitted return type means
    # "returns nothing" and hand `dep` a bare `nil` — surfacing three lines
    # away as an uncoded `map_get arg 1 must be a Map, got nil`. #720 gave that
    # a located `jit-signature` refusal instead, naming `dependency`.
    #
    # #753 retired this as a `jit-gap` probe (see the `GAP_ALLOWLIST` comment
    # above `map_set_non_i64_bool`, this PR): `compatible_with`'s new `(Unit,
    # Map)` carve-out means `dep`'s `Unit`-typed call now fails the GENERAL
    # call-arg check at `map_get`'s arg 0 all on its own, so `dependency`'s
    # declaration is itself required to say `-> map` — this exact program no
    # longer `--check`s clean, and `dmc run` refuses it before ever reaching
    # the interpreter, the same as `dmc jit`. Kept as a regression pin: it
    # still asserts (via `both-fail`) that this shape is refused up front.
    "omitted_return_type_let_bound_call": (
        "fn dependency(v: i64) { let !m = map()  map_set(m, \"v\", v)  m }\n"
        "fn main() -> i64 { let dep = dependency(7)  map_get(dep, \"v\") }",
        "#695/#753 an omitted-return-type fn's map tail is now a --check "
        "error at its own declaration (must say `-> map`); both backends "
        "refuse this program before running (both-fail)"),
    # Originally: a nested block's `let` of the SAME bare name as an outer
    # omitted-return-provenance binding used to wipe the outer entry for good
    # (`omitted_ret_provenance` keyed by bare name, dropped unconditionally by
    # every `let` of that name, with no notion of a block's drop being
    # temporary), so `map_get(d, ..)` after the block fell to the uncoded
    # `jit-error` #695 reported in the first place; scoping it like
    # `locals`/`local_tys`/`field_aliases` fixed it. #753 retired this as a
    # `jit-gap` probe too — same reasoning as `omitted_return_type_let_bound_call`
    # just above (the whole shape is a `--check` error now, on both backends,
    # before the provenance-tracking code this probe targeted ever runs).
    "omitted_return_provenance_block_shadow": (
        "fn dep(v: i64) { let !m = map()  map_set(m, \"v\", v)  m }\n"
        "fn main() -> i64 {\n"
        "    let d = dep(7)\n"
        "    if true { let d = 5  print(d) }\n"
        "    map_get(d, \"v\")\n"
        "}",
        "#753 an outer omitted-return-type binding's map tail is now a "
        "--check error at its own declaration; both backends refuse this "
        "program before running (both-fail)"),
    # Originally (#717 follow-up): the same omitted-return-type shape as
    # `omitted_return_type_let_bound_call` above, but through a shape-generic
    # model's method (`model B[N] { fn build(self, k) {..map..} }`) rather
    # than a plain fn — `resolve_direct_call_name` used to answer `None` for
    # ANY shape-generic model's method, which left this exact program with no
    # provenance and the uncoded `jit-error` #695's fix was supposed to have
    # closed everywhere; closed for the narrow slice needing no
    # argument-lowering to resolve. #753 retired this as a `jit-gap` probe
    # too, same reasoning as the two above.
    "omitted_return_shape_generic_model_method": (
        "model B[N] {\n"
        "    tag: i64\n"
        "    fn build(self, k: i64) { let !m = map()  map_set(m, \"v\", k)  m }\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let s = B[9] { tag: 1 }\n"
        "    let d = s.build(9)\n"
        "    map_get(d, \"v\")\n"
        "}",
        "#717/#753 a shape-generic model method's omitted-return map tail is "
        "now a --check error at its own declaration; both backends refuse "
        "this program before running (both-fail)"),
    # #747/#755 — `handle_rx!`'s own signature (a `View` param) never
    # declares; the bracket-less call re-surfaces that TRUE stored reason
    # (`jit-type-form`) rather than the fabricated "shape bracket it does
    # not declare" #747 fixed.
    "model_method_own_signature_refuses_honestly_747": (
        "model NetworkDriver {\n"
        "    !count: i64\n"
        "    fn handle_rx!(!self, raw_packet: View[u8, [128]]) -> nil {\n"
        "        self.count = self.count + 1\n"
        "    }\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let !net = NetworkDriver { count: 0 }\n"
        "    let !buf = forge.zeros[u8, [128]]\n"
        "    net.handle_rx!(buf)\n"
        "    net.count\n"
        "}",
        "#747/#755 a bracket-less method call whose own View-param signature "
        "cannot declare refuses as jit-type-form, the true reason"),
    # #720 review (F4) — `map_set` used to coerce a `bool`/float/tensor value
    # to `i64` silently (only `str` refused); it now refuses every non-
    # integer value with a located `jit-conversion`. `dmc run` answers
    # (its map is genuinely dynamically typed), `dmc jit` cleanly refuses —
    # a tracked gap, not a divergence, until #721's typed-value
    # representation lands.
    "map_set_non_i64_bool": (
        "fn main() -> i64 { let !m = map()  map_set(m, \"k\", true)  0 }",
        "#721 map_set refuses a bool value as jit-conversion rather than coercing it"),
    "map_set_non_i64_float": (
        "fn main() -> i64 { let !m = map()  map_set(m, \"k\", 1.5)  0 }",
        "#721 map_set refuses an f64 value as jit-conversion rather than truncating it"),
    "map_set_non_i64_tensor": (
        "fn main() -> i64 { let !m = map()  let t = forge.zeros[f32, [2]]  map_set(m, \"k\", t)  0 }",
        "#721 map_set refuses a tensor value as jit-conversion rather than storing its raw pointer"),
    # #742 — the sibling of the three probes above: a MAP stored as another
    # map's value (nested maps), the exact shape `dmc test --jit --all`
    # found on the downstream corpus right after #720 landed
    # (`map_set(defs, map_get(rec, "name"), rec)`, `rec` itself a map).
    # `dmc run` answers fully; `dmc jit` refuses as `jit-conversion` naming
    # `Map`, citing #721 — never the uncoded `jit error` #742 reported.
    "map_set_non_i64_map": (
        "fn main() -> i64 { let !rec = map()  map_set(rec, \"a\", 1)  "
        "let !defs = map()  map_set(defs, \"rec\", rec)  0 }",
        "#721/#742 map_set refuses a Map value (nested maps) as jit-conversion"),
    # #760 — the issue's own repro: `let fns = map_get(d, "fns")  fns[i]`.
    # `dmc run` runs (indexing an int falls through the interpreter's
    # forward-compat `Value::Opaque("index")` catch-all, interp.rs); `dmc
    # jit` used to raise an uncoded `jit error: indexing requires a tensor
    # or str, got i64?` — now a located `jit-conversion` citing #721, same
    # as every other map-value-typed shape above.
    "map_get_result_indexed": (
        "fn main() -> i64 { let !d = map_new()  map_set(d, \"fns\", 3)  "
        "let fns = map_get(d, \"fns\")  let i = 0  let v = fns[i]  0 }",
        "#760/#721 indexing a map_get result refuses as jit-conversion"),
    # #769 — the issue's own repro, adapted to an i64-returning `main` so
    # both backends' answers compare directly: `fn f(t: Tensor[f32, [2]])
    # -> i64 { sum(t) as i64 }` called with a `map_get` result. `dmc run`
    # answers 3 (the interpreter hands the plain int to `sum`, which
    # answers it back); `dmc jit` used to segfault reinterpreting the
    # decoded `OptI64` payload as a tensor pointer in `coerce_to` — now a
    # located `jit-conversion` citing #721.
    "map_get_result_tensor_param": (
        "fn f(t: Tensor[f32, [2]]) -> i64 { sum(t) as i64 }  "
        "fn main() -> i64 { let !d = map_new()  map_set(d, \"k\", 3)  "
        "let x = map_get(d, \"k\")  f(x) }",
        "#769/#721 map_get result passed to a Tensor param refuses as "
        "jit-conversion instead of segfaulting"),
    # global max/min NaN propagation (#300)
    "max_nan_propagate":  ("fn main()->f64{ let a=[0.0f32,2.0f32]  let z=[0.0f32,1.0f32]  max(a ./ z) }", "#300 max propagates NaN"),
    "min_nan_propagate":  ("fn main()->f64{ let a=[0.0f32,2.0f32]  let z=[0.0f32,1.0f32]  min(a ./ z) }", "#300 min propagates NaN"),
    # float compares with non-finite
    "float_inf_gt":       ("fn main()->i64{ if (1.0/0.0) > 1.0e300 {1} else {0} }", "inf ordering"),
    "float_nan_eq":       ("fn main()->i64{ let n=0.0/0.0  if n==n {1} else {0} }", "NaN != NaN"),
    "float_nan_lt":       ("fn main()->i64{ let n=0.0/0.0  if n<1.0 {1} else {0} }", "NaN compares false"),
    # zero/false return printing (#271)
    "ret_zero":           ("fn main()->i64{ 0 }", "#271 zero i64 prints"),
    "ret_false":          ("fn main()->bool{ false }", "#271 false bool prints"),
    # indexing (#270 negative; OOB traps both)
    "idx_neg_last":       ("fn main()->i64{ let !t=forge.zeros[f32,[4]] t[3]=9.0 t[-1] as i64 }", "#270 negative index"),
    "idx_neg_first":      ("fn main()->i64{ let !t=forge.zeros[f32,[4]] t[0]=7.0 t[-4] as i64 }", "#270 -dim index"),
    "idx_neg_write":      ("fn main()->i64{ let !t=forge.zeros[f32,[4]] t[-1]=5.0 t[3] as i64 }", "#270 negative write"),
    # reductions (#272) + degenerate/nonfinite
    "sum_size1":          ("fn main()->i64{ let !t=forge.zeros[f32,[1]] t[0]=5.0 sum(t) as i64 }", "size-1 sum"),
    "max_reduction":      ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=2.0 t[1]=9.0 t[2]=4.0 max(t) as i64 }", "#272 max"),
    "min_reduction":      ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=2.0 t[1]=9.0 t[2]=4.0 min(t) as i64 }", "#272 min"),
    "argmax_ties":        ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=2.0 t[1]=2.0 t[2]=1.0 argmax(t,0) }", "#272 argmax first-wins"),
    "argmin":             ("fn main()->i64{ let !t=forge.zeros[f32,[3]] t[0]=5.0 t[1]=1.0 t[2]=4.0 argmin(t,0) }", "#272 argmin"),
    "max_with_inf":       ("fn main()->i64{ let !t=forge.zeros[f32,[2]] t[0]=1.0/0.0 t[1]=3.0 max(t) as i64 }", "#272 max with inf"),
    # matmul / broadcast / transpose
    "matmul_1xN_Nx1":     ("fn main()->i64{ let a=forge.ones[f32,[1,4]] let b=forge.ones[f32,[4,1]] (a@b)[0,0] as i64 }", "1xN @ Nx1"),
    "bcast_row":          ("fn main()->i64{ let !a=forge.zeros[f32,[2,3]] let !b=forge.zeros[f32,[3]] b[0]=1.0 sum(a .+ b) as i64 }", "row broadcast"),
    "transpose_sum":      ("fn main()->i64{ let !g=forge.zeros[f32,[2,2]] g[0,1]=5.0 sum(g') as i64 }", "transpose"),
    # control flow / recursion / pow / bitwise
    "recursion_fib":      ("fn fib(n:i64)->i64{ if n<2 {n} else {fib(n-1)+fib(n-2)} } fn main()->i64{ fib(15) }", "recursion"),
    "match_dispatch":     ("fn main()->i64{ let !s=0  for k in 0..4 { s=s+match k {0=>10,1=>20,2=>30,_=>40} }  s }", "match jump table"),
    "bit_ops":            ("fn main()->i64{ (12 & 10) + (12 | 10) + (12 ^ 10) }", "bitwise and/or/xor"),
    "pow_float":          ("fn main()->i64{ (2.0 ** 10.0) as i64 }", "#261 float power"),
    "pow_int":            ("fn main()->i64{ 2 ** 10 }", "integer power (jit gap, #215)"),
    "argmax_then_index":  ("fn main()->i64{\n  let !t=forge.zeros[f32,[3]]\n  t[0]=5.0\n  t[2]=9.0\n  let i = argmax(t, 0)\n  t[i] as i64\n}", "#272 argmax result usable as index"),
    # ── `return` out of a block expression (#639) ────────────────────────────
    # The interpreter used to collapse a block's escaping `Flow`: a `return`
    # became the BLOCK's value instead of the function's, so `y` bound 42 and
    # the function went on to answer 43 where the JIT — which has always
    # propagated it — answered 42. Both directions of the branch are probed:
    # the taken one is the divergence, the untaken one pins the other half of
    # the rule (no flow escapes → the block's value is still its tail, 5).
    # Only the PLAIN-block spellings live here; `forge { … }` in expression
    # position is not lowered yet (#605/#629), so an arena probe would score as
    # a gap rather than as coverage.
    "block_return_taken":   ("fn main()->i64{\n let x=1\n let y={ if x>0 { return 42 }  5 }\n y+1\n}", "#639 `return` inside a block leaves the FUNCTION, not the block"),
    "block_return_untaken": ("fn main()->i64{\n let x=0\n let y={ if x>0 { return 42 }  5 }\n y+1\n}", "#639 no flow escapes → the block's value is its tail"),
    "block_return_else":    ("fn main()->i64{\n let x=1\n let y={ if x>0 { return 42 } else { 5 } }\n y+1\n}", "#639 same through an `if`/`else` block"),
    "block_tail_value":     ("fn main()->i64{\n let y={ let a=2  { a*3 } }\n y+1\n}", "#639 a nested block still evaluates to its tail"),
    # integer-element tensors (#274) — token ids / index data
    "i64_tensor_rw":      ("fn main()->i64{ let !t=forge.zeros[i64,[3]] t[1]=42 t[1] }", "#274 i64 zeros+write+read"),
    "i32_tensor_rw":      ("fn main()->i64{ let !t=forge.uninit[i32,[2]] t[0]=7 t[0] as i64 }", "#274 i32 uninit+rw"),
    "i64_literal_index":  ("fn main()->i64{ let ids=[[785,6722,315]] ids[0,1] }", "#274 i64 literal index"),
    "i64_param_read":     ("fn first[B,S](x:Tensor[i64,[B,S]])->i64{ x[0,0] } fn main()->i64{ first([[5,6]]) }", "#274 i64 tensor param"),
    "embed_gather":       ("fn main()->i64{ let ids=[10,20,30] let !e=forge.zeros[f32,[40]] e[20]=5.0 let id=ids[1] e[id] as i64 }", "#274 id-driven gather"),
    # ── narrow-integer tensor elements (#572) ───────────────────────────────
    # These are the probes #572 says the misclassification suppressed. Each
    # wraps at its declared width, so a backend that widened to i64 answers
    # something else entirely; today the JIT declines to allocate the element
    # type at all, and — since #572 — says so as a GAP, which is what lets
    # these live here instead of reddening the gate.
    "narrow_tensor_i8_wrap":  ("fn main()->i64{\n let !t=forge.uninit[i8,[2]]\n t[0]=127\n let v:i8=t[0]\n (v+1) as i64\n}", "#572 i8 tensor element wraps at 8 bits -> -128"),
    "narrow_tensor_u8_wrap":  ("fn main()->i64{\n let !t=forge.uninit[u8,[2]]\n t[0]=255\n let v:u8=t[0]\n (v+1) as i64\n}", "#572 u8 tensor element wraps at 8 bits -> 0"),
    "narrow_tensor_i16_wrap": ("fn main()->i64{\n let !t=forge.uninit[i16,[2]]\n t[0]=32767\n let v:i16=t[0]\n (v+1) as i64\n}", "#572 i16 tensor element wraps at 16 bits -> -32768"),
    "zeros_bool_tensor":      ("fn main()->i64{\n let t=forge.zeros[bool,[4]]\n if t[0] { 1 } else { 0 }\n}", "#572 `forge.zeros[bool,..]` is false everywhere"),
    "ones_i64_tensor":        ("fn main()->i64{\n let t=forge.ones[i64,[4]]\n t[0]+t[3]\n}", "#572 `forge.ones[i64,..]` is 1 everywhere -> 2"),
    # #818: `forge.zeros[fp8_e4m3,[4]]` was unparseable before this fix (the
    # constructor's type-arg position dropped the underscore off the token,
    # so the checker saw `fp8e4m3` and refused with `undefined identifier`
    # before the JIT ever got a look). Now it parses and type-checks on both
    # backends; the JIT still declines to allocate the element type itself
    # (`Refusal::ElementType`, same shape as `narrow_tensor_*` above), which
    # is a pre-existing, deliberate gap — this PR does not widen JIT support.
    "elem_fp8_e4m3_ctor_gap": ("fn main()->i64{\n let t=forge.zeros[fp8_e4m3,[4]]\n t.shape[0]\n}", "#818 `forge.zeros[fp8_e4m3,..]` now parses; JIT still declines the element type -> 4"),
    "elem_fp8_e5m2_ctor_gap": ("fn main()->i64{\n let t=forge.zeros[fp8_e5m2,[4]]\n t.shape[0]\n}", "#833 sibling of `elem_fp8_e4m3_ctor_gap`: `forge.zeros[fp8_e5m2,..]` parses; JIT still declines the element type -> 4"),
    # #632 — the byte-aligned element widths, at their own storage lane. Each
    # of these is reachable only if the element really occupies the width its
    # type names: a load or store at the 64-bit MACHINE width would read or
    # clobber a neighbouring lane, and the answer would not be this one.
    "elem_u8_lane_isolated":  ("fn main()->i64{\n let !t=forge.zeros[u8,[4]]\n t[0]=200\n t[1]=7\n t[2]=9\n (t[1]+t[2]) as i64\n}", "#632 a u8 store touches one byte, not eight -> 16"),
    "elem_u8_ones_fill":      ("fn main()->i64{\n let t=forge.ones[u8,[5]]\n (t[0]+t[4]) as i64\n}", "#632 `ones` fills every one-byte lane -> 2"),
    "elem_u8_ragged_zeros":   ("fn main()->i64{\n let t=forge.zeros[u8,[3]]\n (t[0]+t[2]) as i64\n}", "#632 zero-fill at a byte length that is not a whole word -> 0"),
    "elem_u32_wrap_unsigned": ("fn main()->i64{\n let !t=forge.zeros[u32,[2]]\n t[0]=0\n let v:u32=t[0]\n (v-1) as i64\n}", "#632 u32 element wraps UNSIGNED -> 4294967295, not -1"),
    "elem_u32_top_bit":       ("fn main()->i64{\n let !t=forge.zeros[u32,[2]]\n t[0]=4294967295\n (t[0]/2) as i64\n}", "#632 u32 division is unsigned -> 2147483647"),
    "elem_u64_high_bit":      ("fn main()->i64{\n let !t=forge.zeros[u64,[2]]\n t[0]=0\n let v:u64=t[0]\n (v-1) as i64\n}", "#632 u64 carries the 64-bit pattern (SPEC §3.1 status) -> -1"),
    "elem_u64_exact_2p53":    ("fn main()->i64{\n let !t=forge.zeros[u64,[2]]\n t[0]=9007199254740992\n t[0] as i64\n}", "#632 u64 element at f64's exact-integer bound round-trips"),
    # #643 gave `u64` its own tag so `/ % >>` and the orderings read it
    # unsigned, but it could only reach SCALARS — the JIT refused to build a
    # `Tensor[u64, …]`, so the element path was unprobeable and free to keep
    # the signed reading. These three separate the two readings: a signed
    # element load answers 0, -1 and false instead.
    "elem_u64_div_unsigned":  ("fn main()->i64{\n let !t=forge.zeros[u64,[2]]\n t[0]=0xffffffffffffffffu64\n let v:u64=t[0]\n (v/2u64) as i64\n}", "#643 a u64 ELEMENT divides unsigned -> 9223372036854775807, not 0"),
    "elem_u64_shr_logical":   ("fn main()->i64{\n let !t=forge.zeros[u64,[2]]\n t[0]=0xffffffffffffffffu64\n let v:u64=t[0]\n (v>>4u64) as i64\n}", "#643 a u64 ELEMENT shifts logically -> 2^60-1, not -1"),
    "elem_u64_cmp_unsigned":  ("fn main()->i64{\n let !t=forge.zeros[u64,[2]]\n t[0]=0xffffffffffffffffu64\n t[1]=10u64\n let a:u64=t[0]\n let b:u64=t[1]\n if a>b { 1 } else { 0 }\n}", "#643 a u64 ELEMENT compares unsigned -> 1, not 0"),
    # #667: the three probes above cover a `Tensor[u64, …]` LOCAL. A model
    # FIELD of the same declared type went through a separate, hand-rolled
    # copy of the element-kind table in `lower_ty_for_model` that still
    # mapped `u64` to `ScalarKind::I64`, so a struct literal holding
    # `forge.zeros[u64, …]` refused to convert into its own field's type
    # ("cannot convert Tensor[u64,[2]] to Tensor[i64,[2]]") — a clean
    # `jit-conversion` refusal, not a miscompile, but a refusal the local form
    # never hit. Construct, read, element-assign, and pass the model by `!`
    # reference into a callee that divides the field UNSIGNED: the answer is
    # only right if the field's storage lane, not just the local one, reads
    # the pattern unsigned.
    "elem_u64_model_field": (
        "model Ring { !d: Tensor[u64,[2]] }\n"
        "fn bump(!r: Ring, v: u64)->u64{\n r.d[0]=v\n r.d[0]/2u64\n}\n"
        "fn main()->i64{\n let !r=Ring{ d: forge.zeros[u64,[2]] }\n"
        " let half=bump(r,18446744073709551615u64)\n"
        " (half as i64) + (r.d[0] as i64)\n}",
        "#667 a Tensor[u64,…] MODEL FIELD used to refuse jit-conversion; now "
        "shares tensor_elem_kind with the local form -> 9223372036854775806"),
    "elem_i8_sign_extends":   ("fn main()->i64{\n let !t=forge.zeros[i8,[2]]\n t[0]=0-128\n t[0] as i64\n}", "#632 an i8 lane loads SIGN-extended -> -128, not 128"),
    "elem_u8_copy_snapshot":  ("fn main()->i64{\n let !a=forge.zeros[u8,[4]]\n a[0]=200\n a[3]=55\n let b=a\n a[0]=1\n (b[0]+b[3]) as i64\n}", "#632 the copy loop walks u8 lanes at one byte -> 255"),
    "elem_u8_dot_mul_wraps":  ("fn main()->i64{\n let !a=forge.zeros[u8,[2]]\n let !b=forge.zeros[u8,[2]]\n a[0]=250\n b[0]=10\n (a .* b)[0] as i64\n}", "#571/#632 `.*` wraps at the element width -> 196"),
    "elem_u8_dot_div_zero":   ("fn main()->i64{\n let !a=forge.zeros[u8,[2]]\n let !b=forge.zeros[u8,[2]]\n a[0]=7\n (a ./ b)[0] as i64\n}", "#571/#632 elementwise `/` by zero is 0, not inf"),
    "elem_u8_dot_scalar":     ("fn main()->i64{\n let !a=forge.zeros[u8,[2]]\n a[0]=255\n (a .+ 1)[0] as i64\n}", "#571/#632 a scalar operand adopts the element width -> 0"),
    "elem_u8_sum_no_wrap":    ("fn main()->i64{\n let !a=forge.zeros[u8,[3]]\n a[0]=200\n a[1]=250\n a[2]=200\n sum(a) as i64\n}", "#632 `sum` accumulates in f64 and does NOT wrap -> 650"),
    "elem_bool_sum_counts":   ("fn main()->i64{\n let !a=forge.zeros[bool,[4]]\n a[0]=true\n a[3]=true\n sum(a) as i64\n}", "#632 `sum` over a bool tensor counts set lanes -> 2"),
    "elem_slice_assign_type_mismatch": ("fn main()->i64{\n let !a=forge.zeros[u8,[7]]\n let !b=forge.zeros[i64,[3]]\n b[0]=1\n a[4..7]=b\n a[4] as i64\n}", "#632 review: a slice-assign whose src/dst element types differ is a clean gap, not a `jit error` -> 1"),
    "elem_u8_dot_cmp_gap":    ("fn main()->i64{\n let !a=forge.zeros[u8,[3]]\n let !b=forge.zeros[u8,[3]]\n a[0]=5\n a[1]=2\n a[2]=9\n b[0]=1\n b[1]=2\n b[2]=20\n sum(a .> b) as i64\n}", "#632 review: integer tensor comparisons (`.>` etc.) are still f32-only, undocumented until now -> 1"),
    # ── colon slabs (#563) ──────────────────────────────────────────────────
    # The colon spelling of a slice. `x[0:S/2]` folds only if S is bound; the
    # JIT wants a literal, and the strided and open-ended forms are not lowered
    # at all. All three were unprobeable until #563 made them announce as gaps.
    "slab_literal_bounds":  ("fn main()->i64{\n let !x=forge.zeros[f32,[4]]\n for i in 0..4 { x[i]=1.0 }\n sum(x[0:2]) as i64\n}", "#563 the literal-bound slab that DOES lower — 2"),
    "slab_shape_param_end": ("fn half[S](x:Tensor[f32,[S]])->i64{ sum(x[0 : S / 2]) as i64 }\nfn main()->i64{\n let !x=forge.zeros[f32,[4]]\n for i in 0..4 { x[i]=1.0 }\n half(x)\n}", "#563 the issue's repro: a slab end that is a shape expression"),
    "slab_strided":         ("fn main()->i64{\n let !x=forge.zeros[f32,[4]]\n for i in 0..4 { x[i]=1.0 }\n sum(x[0::2]) as i64\n}", "#563 a strided slab"),
    "slab_open_end":        ("fn main()->i64{\n let !x=forge.zeros[f32,[4]]\n for i in 0..4 { x[i]=1.0 }\n sum(x[1..]) as i64\n}", "#563 a range slice with no end"),
    # cached GQA attention (#277) — k/v from KV caches, runtime history length
    "attn_gqa_cached_kv": ("fn main()->i64{\n"
                            "  let !kc=forge.kv[f32,[1,1,~,2]](capacity=4)\n"
                            "  let !vc=forge.kv[f32,[1,1,~,2]](capacity=4)\n"
                            "  let !k=forge.uninit[f32,[1,1,2,2]]\n"
                            "  let !v=forge.uninit[f32,[1,1,2,2]]\n"
                            "  for s in 0..2 { for d in 0..2 { k[0,0,s,d]=((s*2+d) as f32)*0.1  v[0,0,s,d]=((s*2+d) as f32)*0.2 } }\n"
                            "  kc <- k[..,..,0..2,..]\n"
                            "  vc <- v[..,..,0..2,..]\n"
                            "  let !q=forge.uninit[f32,[1,2,2,2]]\n"
                            "  for h in 0..2 { for s in 0..2 { for d in 0..2 { q[0,h,s,d]=((h*4+s*2+d) as f32)*0.1-0.2 } } }\n"
                            "  let o=attn_gqa(q,kc,vc)\n"
                            "  ((o[0,0,0,0]+o[0,1,1,1]*3.0)*1000.0) as i64 }",
                            "#277 cached GQA attn: KV cache, GQA grouping (H_q=2,H_kv=1), len==S"),
    # elementwise activations (relu/sigmoid/tanh/gelu/silu) — scalar + tensor.
    # Inputs/scales chosen mid-integer to avoid f32/f64 rounding flips. Statements
    # are newline-separated so a literal is never immediately followed by `(` (the
    # `3.0  (expr)` → `3.0(...)` indirect-call parse trap).
    "relu_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=relu(x)*1000.0\n  r as i64 }", "relu scalar -> 3000"),
    "sigmoid_scalar": ("fn main()->i64{\n  let x:f32=3.0\n  let r=sigmoid(x)*1000.0\n  r as i64 }", "sigmoid scalar ~952"),
    "tanh_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=tanh(x)*1000.0\n  r as i64 }", "tanh scalar ~995"),
    "gelu_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=gelu(x)*1000.0\n  r as i64 }", "gelu (tanh approx) scalar ~2996"),
    "silu_scalar":    ("fn main()->i64{\n  let x:f32=3.0\n  let r=silu(x)*1000.0\n  r as i64 }", "silu scalar ~2857"),
    "silu_neg_scalar":("fn main()->i64{\n  let x:f32=-2.0\n  let r=silu(x)*1000.0\n  r as i64 }", "silu negative ~-238"),
    "relu_tensor":    ("fn main()->i64{\n  let t=[3.0,1.5,2.0]\n  let r=sum(relu(t))*1000.0\n  r as i64 }", "relu tensor sum -> 6500"),
    "gelu_tensor":    ("fn main()->i64{\n  let t=[3.0,1.5,2.0]\n  let r=sum(gelu(t))*1000.0\n  r as i64 }", "gelu tensor sum ~6350"),
    "silu_tensor":    ("fn main()->i64{\n  let t=[3.0,1.5,2.0]\n  let r=sum(silu(t))*1000.0\n  r as i64 }", "silu tensor sum ~5845"),
    # ── scalar f32 (#473) ────────────────────────────────────────────────────
    # The corpus reaches no multi-op f32 scalar chain, so nothing here was
    # gated before: `diff_backends.py` was green while `dmc run` and `dmc jit`
    # disagreed on `0.1f32 + 0.2f32`. Each probe returns the f64 widening of an
    # f32 result, so the comparison is on all 17 digits — a one-ulp f32
    # disagreement (~1e-8 relative) shows, which is exactly the scale at which
    # "accumulate in f64, round once" parts company with true f32.
    # A SINGLE `+ - * /` is NOT enough: #241's double-rounding property makes
    # f64-then-round bit-exact with native f32 for one op. Chains are the test.
    "f32_add_two":        ("fn main()->f64{ (0.1f32 + 0.2f32) as f64 }", "#473 the issue's repro: one add"),
    "f32_chain_three":    ("fn main()->f64{ let a=0.1f32\n let b=0.2f32\n let c=0.3f32\n (a+b+c) as f64 }", "#473 three-term chain — diverges under f64 accumulation"),
    "f32_chain_mixed_ops":("fn main()->f64{ let a=0.1f32\n let b=0.3f32\n let c=0.7f32\n ((a+b)*c-a/b) as f64 }", "#473 + - * / in one f32 expression"),
    "f32_horner":         ("fn main()->f64{ let a=1.1f32\n let b=1.3f32\n (((a*b+a)*b+a)*b+a) as f64 }", "#473 nested multiply-add, 6 ops"),
    "f32_accum_loop":     ("fn main()->f64{ let !s=0.0f32\n for i in 0..100 { s = s + 0.01f32 }\n s as f64 }", "#473 100-step f32 accumulation (the `total()` shape)"),
    "f32_accum_mul":      ("fn main()->f64{ let !s=1.0f32\n for i in 0..20 { s = s * 1.1f32 }\n s as f64 }", "#473 multiplicative accumulation"),
    "f32_accum_elems":    ("fn main()->f64{ let !t=forge.zeros[f32,[8]]\n for i in 0..8 { t[i]=0.1f32 }\n let !s=0.0f32\n for i in 0..8 { s = s + t[i] }\n s as f64 }", "#473 summing f32 tensor elements in a scalar loop"),
    "f32_mixed_f64":      ("fn main()->f64{ let a=0.1f32\n let b=0.2\n a + b }", "#473 f32 meets f64 -> promotes, never narrows"),
    "f32_mixed_rev":      ("fn main()->f64{ let a=0.2\n let b=0.1f32\n a + b }", "#473 f64 meets f32, other order"),
    "f32_mixed_tail":     ("fn main()->f64{ let a=0.1f32\n let b=0.2f32\n let c=0.3\n a + b + c }", "#473 f32 chain then an f64 tail"),
    "f32_call_ret":       ("fn g(x:f32)->f32{ x*x+x }\nfn main()->f64{ g(0.1f32) as f64 }", "#473 f32 through a parameter and a return"),
    "f32_call_nested":    ("fn g(x:f32)->f32{ x+0.1f32 }\nfn main()->f64{ g(g(g(0.1f32))) as f64 }", "#473 f32 across three nested calls"),
    "f32_ret_unsuffixed": ("fn g()->f32{ 0.1 }\nfn main()->f64{ g() as f64 }", "#473 a declared f32 RETURN binds an unsuffixed literal's width"),
    "f32_tensor_read":    ("fn main()->f64{ let !t=forge.zeros[f32,[3]]\n t[0]=0.1f32\n t[1]=0.2f32\n t[2]=0.3f32\n (t[0]+t[1]+t[2]) as f64 }", "#473 reads out of an f32 tensor stay f32"),
    "f32_tensor_mixed":   ("fn main()->f64{ let !a=forge.zeros[f32,[1]]\n let !b=forge.uninit[f64,[1]]\n a[0]=0.1f32\n b[0]=0.2\n a[0]+b[0] }", "#473 f32 element + f64 element promotes"),
    "f32_cast_chain":     ("fn main()->f64{ let a=0.1 as f32\n let b=0.2 as f32\n (a+b) as f64 }", "#473 `as f32` produces a true f32"),
    "f32_let_annotated":  ("fn main()->f64{ let a: f32 = 0.1\n let b: f32 = 0.2\n let c: f32 = 0.3\n (a+b+c) as f64 }", "#473 a typed `let` binds width (#209: the literal itself stays f64)"),
    "f32_model_field":    ("model M { x: f32 }\nfn main()->f64{ let m = M { x: 0.1 }\n (m.x + 0.2f32) as f64 }", "#473 a declared f32 FIELD binds width"),
    "f32_unary_neg":      ("fn main()->f64{ let a=0.1f32\n let b=0.0f32-a\n (b+a+a) as f64 }", "#473 negation keeps f32"),
    # ── f32 tensor reductions (#481) ────────────────────────────────────────
    # `sum(t)` used to disagree with the loop it is documented to mean, WITHIN
    # `dmc run`. Each returns the f64 widening so all 17 digits are compared.
    "f32r_sum":           ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n sum(t) as f64 }", "#481 sum accumulates at f32 width"),
    "f32r_sum_vs_loop":   ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n let !s=0.0f32\n for i in 0..12 { s=s+t[i] }\n (sum(t)-s) as f64 }", "#481 sum(t) - hand loop == 0 on both backends"),
    "f32r_mean":          ("fn main()->f64{ let !t=forge.zeros[f32,[100]]\n for i in 0..100 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n mean(t) as f64 }", "#481 mean divides by an f32 count in f32"),
    "f32r_variance":      ("fn main()->f64{ let !t=forge.zeros[f32,[100]]\n for i in 0..100 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n variance(t) as f64 }", "#481 variance is two passes at f32 width"),
    "f32r_sum_then_add":  ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n (sum(t)+0.1f32) as f64 }", "#481 the reduction RESULT is f32-wide"),
    "f32r_max_then_add":  ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n (max(t)+0.1f32) as f64 }", "#481 max selects an f32 — `max(t)` alone agreed even before"),
    "f32r_sum_along":     ("fn main()->f64{ let !m=forge.zeros[f32,[3,4]]\n for i in 0..3 { for j in 0..4 { m[i,j]=((i*4+j) as f32)*0.3f32+0.1f32 } }\n sum(sum_along(m,1)) as f64 }", "#481 per-axis lanes reduce at f32 width"),
    "f32r_mean_along":    ("fn main()->f64{ let !m=forge.zeros[f32,[3,4]]\n for i in 0..3 { for j in 0..4 { m[i,j]=((i*4+j) as f32)*0.3f32+0.1f32 } }\n sum(mean_along(m,0)) as f64 }", "#481 mean_along keeps the source dtype"),
    "f32r_variance_along":("fn main()->f64{ let !m=forge.zeros[f32,[3,4]]\n for i in 0..3 { for j in 0..4 { m[i,j]=((i*4+j) as f32)*0.3f32+0.1f32 } }\n sum(variance_along(m,1)) as f64 }", "#481 variance_along is two-pass, matching the JIT's formula"),
    "f32r_softmax_sum":   ("fn main()->f64{ let !t=forge.zeros[f32,[12]]\n for i in 0..12 { t[i]=((i as f32)*0.7f32+0.1f32)/3.0f32 }\n sum(softmax(t)) as f64 }", "#481 reducing a derived f32 tensor"),
    # Restored by #480: the JIT's `sum` is f32-only, and that refusal is now a
    # classified `jit unsupported` rather than a message this battery had to
    # score as a DIVERGENCE. Before #480 this probe could not live here at all.
    "f64r_sum_keeps_width":("fn main()->f64{ let !t=forge.uninit[f64,[3]]\n t[0]=0.1\n t[1]=0.2\n t[2]=0.3\n sum(t) }", "#481 must NOT narrow an f64 tensor's reduction (jit gap: sum is f32-only)"),
    # ── f32 matmul contraction (#481) ───────────────────────────────────────
    # C[0,0] depends only on A's row 0 and B's column 0, which are identical in
    # both probes below. Before #481 the JIT answered differently depending on
    # B's WIDTH, because `n % 4` picked the kernel and the kernel picked the
    # rounding. These two must now agree with each other AND across backends.
    "f32mm_width3":       ("fn main()->f64{ let !a=forge.zeros[f32,[8,16]]\n let !b=forge.zeros[f32,[16,3]]\n for i in 0..8 { for j in 0..16 { a[i,j]=(((i*16+j) as f32)*0.7f32+0.1f32)/3.0f32 } }\n for i in 0..16 { for j in 0..3 { b[i,j]=(((i*97+j) as f32)*0.3f32+0.2f32)/7.0f32 } }\n (a@b)[0,0] as f64 }", "#481 matmul, B width 3 (scalar kernel)"),
    "f32mm_width4":       ("fn main()->f64{ let !a=forge.zeros[f32,[8,16]]\n let !b=forge.zeros[f32,[16,4]]\n for i in 0..8 { for j in 0..16 { a[i,j]=(((i*16+j) as f32)*0.7f32+0.1f32)/3.0f32 } }\n for i in 0..16 { for j in 0..4 { b[i,j]=(((i*97+j) as f32)*0.3f32+0.2f32)/7.0f32 } }\n (a@b)[0,0] as f64 }", "#481 matmul, B width 4 (vector kernel) — same C[0,0] as width 3"),
    "f32mm_batched":      ("fn main()->f64{ let !a=forge.zeros[f32,[2,4,8]]\n let !b=forge.zeros[f32,[2,8,4]]\n for n in 0..2 { for i in 0..4 { for j in 0..8 { a[n,i,j]=(((n*32+i*8+j) as f32)*0.7f32+0.1f32)/3.0f32 } } }\n for n in 0..2 { for i in 0..8 { for j in 0..4 { b[n,i,j]=(((n*32+i*4+j) as f32)*0.3f32+0.2f32)/7.0f32 } } }\n (a@b)[1,2,3] as f64 }", "#481 batched matmul contracts too"),
    # #478: an unsuffixed literal beside an f32 branch adopts f32 on BOTH
    # backends (it was a tracked jit gap until the literal started being
    # lowered at the type the checker inferred). The literal is `0.1`, not
    # `0.0`: only a value that differs between the two widths can catch a
    # backend that reverted to f64. Both branch-taken directions are probed —
    # the else-taken one returns the literal's own value and is where `dmc run`
    # used to answer 0.3000000029802322 instead of 0.30000001192092896.
    "f32_branch_lit_then":  ("fn main()->f64{ let a=0.1f32\n let z=if a>0.0f32 {a} else {0.1}\n (z+0.2f32) as f64 }", "#478 unsuffixed literal adopts the f32 branch (then taken)"),
    "f32_branch_lit_else":  ("fn main()->f64{ let a=0.1f32\n let z=if a>1.0f32 {a} else {0.1}\n (z+0.2f32) as f64 }", "#478 unsuffixed literal adopts the f32 branch (else taken)"),
    "f32_match_lit_arm":    ("fn main()->f64{ let a=0.1f32\n let n=2\n let z=match n { 1 => a, _ => 0.1 }\n (z+0.2f32) as f64 }", "#478 same at a match join (literal arm taken)"),
    "f32_branch_lit_f64":   ("fn main()->f64{ let a=0.5\n let z=if a>1.0 {a} else {0.1}\n z }", "#478 must NOT narrow: an all-f64 join stays f64 (#209)"),
    # #566: the same rule through an ENUM PAYLOAD binding. #478's pre-pass reads
    # a bound name's width out of the JIT's local types, so an ordinary arm
    # (`match k { 0 => x_f32, _ => 0.1 }`) already worked; a payload binding is
    # not declared until the arm block is entered, i.e. after the hint has to be
    # fixed, so the literal kept its f64 default and the arms were refused as
    # disagreeing. `dmc run` answered, `dmc jit` did not — SPEC.md §4.5's own
    # worked example. These probe both arm-taken directions plus the half that
    # must not move.
    "enum_payload_lit_arm":  ("enum S { C(f32), E }\nfn a(s:S)->f32{ match s { C(r) => r*r, E => 0.1 } }\nfn main()->f64{ (a(S.E)+0.2f32) as f64 }", "#566 untyped literal beside an f32 payload arm (literal arm taken)"),
    "enum_payload_lit_pay":  ("enum S { C(f32), E }\nfn a(s:S)->f32{ match s { C(r) => r*r, E => 0.1 } }\nfn main()->f64{ (a(S.C(0.5f32))+0.2f32) as f64 }", "#566 …and the payload arm, whose f32 the literal adopts"),
    "enum_payload_lit_f64":  ("enum S { C(f64), E }\nfn a(s:S)->f64{ match s { C(r) => r*r, E => 0.1 } }\nfn main()->f64{ a(S.E) }", "#566 must NOT narrow: an f64 payload leaves the literal f64 (#209)"),
    "f32_branch_join":    ("fn main()->f64{ let a=0.1f32\n let z=if a>0.0f32 {a+0.2f32} else {0.0f32}\n (z+0.3f32) as f64 }", "#473 both if-branches f32"),
    "f32_match_join":     ("fn main()->f64{ let n=1\n let a=0.1f32\n let z=match n { 1 => a+0.2f32, _ => 0.0f32 }\n (z+0.3f32) as f64 }", "#473 both match arms f32"),
    "f32_pow":            ("fn main()->f64{ let a=1.1f32\n let b=3.0f32\n (a**b) as f64 }", "#473 `**` widens to the f64 libm call and rounds back"),
    "f32_mod":            ("fn main()->f64{ let a=5.3f32\n let b=2.1f32\n (a%b) as f64 }", "#473 `%` widens to the f64 fmod and rounds back"),
    "f32_sqrt_chain":     ("fn main()->f64{ let a=2.0f32\n (sqrt(a)+0.1f32) as f64 }", "#473 transcendentals compute in f64 on BOTH backends (#209)"),
    "f32_from_bits":      ("fn main()->f64{ let a=f32_from_bits(1036831949)\n (a+0.2f32) as f64 }", "#473 f32_from_bits yields an f32, not an f64"),
    "f32_bf16_suffix":    ("fn main()->f64{ let a=0.1bf16\n let b=0.2bf16\n (a+b) as f64 }", "#473/#179 bf16 is f32-backed"),
    "f32_f16_suffix":     ("fn main()->f64{ let a=0.1f16\n let b=0.2f16\n (a+b) as f64 }", "#473/#179 f16 is f32-backed"),
    "f32_deep_chain":     ("fn main()->f64{ let !s=1.0f32\n for i in 0..50 { s = s * 1.01f32 + 0.001f32 }\n s as f64 }", "#473 50 fused multiply-adds — drift is unmissable here"),
    "f32_loop_predicate": ("fn main()->i64{ let !s=0.0f32\n let !n=0\n while s < 1.0f32 { s = s + 0.1f32\n n = n + 1 }\n n }", "#473 f32 rounding decides the trip count"),
    "f64_unsuffixed_keeps_width": ("fn main()->f64{ let a=0.1\n let b=0.2\n let c=0.3\n a+b+c }", "#209 must NOT regress: unsuffixed literals stay f64"),
    # ── `if` joins: which side reaches the join, and carrying what ─────────
    # The join's phi param is decided before the else side is lowered, so a
    # wrong guess is a Cranelift `mismatched argument count` — a whole-function
    # compile failure, which scores as a divergence here rather than a gap.
    # These are the shapes that used to produce one; none of them is exotic,
    # and the first is how PORTS.md §6's guarded idiom is written.
    "if_else_trailing_bare_if":  ("fn main()->i64{ let n=5\n if n>3 { 0 } else { if n != 1 { print(\"x\") } }\n n }", "an else block ending in an `if` with no `else` yields nothing"),
    "if_else_trailing_bare_if_valued": ("fn main()->i64{ let n=5\n if n>3 { 0 } else { if n != 1 { 7 } }\n n }", "…still nothing when that bare `if` has a value to give"),
    "if_then_trailing_bare_if":  ("fn main()->i64{ let n=5\n if n>3 { if n != 1 { print(\"x\") } } else { 0 }\n n }", "the mirror: the THEN side is the one arriving empty-handed"),
    "if_elseif_chain_no_final_else": ("fn main()->i64{ let n=5\n if n>3 { 0 } else if n != 1 { print(\"x\") }\n n }", "an `else if` chain with no final `else` yields nothing"),
    "if_nested_both_sides_still_value": ("fn main()->i64{ let n=5\n let x=if n>3 { 1 } else { if n != 1 { 2 } else { 3 } }\n x }", "…but a trailing `if` with both sides still carries its value"),
    "if_diverging_else_keeps_then_value": ("fn main()->i64{ let n=5\n let x=if n>3 { 5 } else { return 0 }\n x }", "`else { return }` never reaches the join, so the then value is the answer"),
    "if_diverging_else_taken": ("fn main()->i64{ let n=1\n let x=if n>3 { 5 } else { return 42 }\n x }", "…and the diverging side still decides when the guard fails"),

    # ── process ports (PORTS.md §2/§6/§7.1) ─────────────────────────────────
    # Both backends now drive one `crate::ports::PortRegistry`, so these probe
    # the marshalling on each side of it: the tag texts, the `(_, Err)` shape,
    # and the nil-vs-empty-str distinction the JIT encodes as a null pointer.
    # Every python-dependent probe is guarded by its own `port_open` check, so
    # a machine without python3 answers `false` on BOTH backends (agreeing)
    # instead of crashing one of them.
    "port_open_unsupported_tag": ('fn main()->bool{ let (p,e)=port_open("lua")\n e.starts_with("port-open") }', "PORTS §6 port-open: a runtime that is not wired up"),
    "port_open_failed_handle_nil": ('fn main()->bool{ let (p,e)=port_open("lua")\n p == nil }', "a failed open hands back a nil handle, not an empty str"),
    "port_open_failed_handle_is_not_live": ('fn main()->bool{ let (p,e)=port_open("lua")\n p != nil }', "…so the `!= nil` liveness test says false, not true"),
    "port_call_error_result_is_empty_not_nil": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"math.sqrt","[\\"x\\"]")\n let (_,_)=port_close(p)\n o == "" && o != nil } }', "PORTS §6 the result half of a failed call is the EMPTY str, not nil"),
    "port_call_positional_spread": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (g,e2)=port_call(p,"math.gcd","[462, 1071]")\n let (_,_)=port_close(p)\n g == "21" && e2 == nil } }', "PORTS §2 a JSON array spreads positionally"),
    "port_call_niladic_nil_payload": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"list",nil)\n let (_,_)=port_close(p)\n o == "[]" && e2 == nil } }', "PORTS §2 a nil payload calls with no arguments"),
    "port_call_kwargs_envelope": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"round","{\\"args\\": [3.14159], \\"kwargs\\": {\\"ndigits\\": 2}}")\n let (_,_)=port_close(p)\n o == "3.14" && e2 == nil } }', "PORTS §2 the {args, kwargs} envelope"),
    "port_call_foreign_exception_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"math.sqrt","[\\"sixteen\\"]")\n let (_,_)=port_close(p)\n e2.starts_with("port-call") && o == "" } }', "PORTS §6 port-call, and the result half is the empty str"),
    "port_call_bare_scalar_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (o,e2)=port_call(p,"math.sqrt","16")\n let (_,_)=port_close(p)\n e2.starts_with("port-protocol") } }', "PORTS §2/§6 a bare scalar is not an argument vector"),
    "port_call_after_close_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,_)=port_close(p)\n let (o,e2)=port_call(p,"len","[[1, 2, 3]]")\n e2.starts_with("port-closed") } }', "PORTS §6 port-closed: a call through a closed handle"),
    "port_double_close_tag": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,first)=port_close(p)\n let (_,second)=port_close(p)\n first == nil && second.starts_with("port-closed") } }', "PORTS §6 port-closed on the second close, clean on the first"),
    "port_ids_are_not_reused": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,_)=port_close(p)\n let (q,_)=port_open("python")\n let (o,e2)=port_call(p,"len","[[1, 2, 3]]")\n let (o2,e3)=port_call(q,"len","[[1, 2, 3]]")\n let (_,_)=port_close(q)\n e2.starts_with("port-closed") && o2 == "3" && e3 == nil } }', "a stale handle cannot reach the port opened after it"),
    "port_survives_an_error": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (_,_)=port_call(p,"math.sqrt","[\\"sixteen\\"]")\n let (g,e2)=port_call(p,"math.gcd","[462, 1071]")\n let (_,_)=port_close(p)\n g == "21" && e2 == nil } }', "a foreign exception does not desync or kill the port"),
    "port_lang_annotation_is_checked": ('fn ask(q: Port[lua]) -> str { let (o,_)=port_call(q,"list",nil)\n o }\nfn main()->bool{ let (p,e)=port_open("python")\n if e != nil { true } else { let r=ask(p)\n let (_,_)=port_close(p)\n false } }', "SPEC §3.11 `Port[L]` refuses a handle opened for another runtime (both backends die)"),
    "port_lang_annotation_accepts_its_own": ('fn ask(q: Port[python]) -> str { let (o,_)=port_call(q,"list",nil)\n o }\nfn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let r=ask(p)\n let (_,_)=port_close(p)\n r == "[]" } }', "…and still binds the matching one"),
    "port_payload_escaping": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let (n,e2)=port_call(p,"len","[\\"he said \\\\\\"hi\\\\\\"\\\\n\\"]")\n let (a,e3)=port_call(p,"len","[[1, 2, 3]]")\n let (_,_)=port_close(p)\n n == "13" && a == "3" && e2 == nil && e3 == nil } }', "a quote+newline payload crosses intact and leaves the frame in sync"),
    # ── tensor copy mode (PORTS.md §3.2) ────────────────────────────────────
    # The envelope is the ABI, so these probes compare it to its literal text:
    # both backends marshal into one `crate::ports` writer, and a probe that
    # returned `true` on one side and `false` on the other would be exactly the
    # drift the shared registry exists to prevent. Only the last one needs
    # python3 — the format itself is demoniC's, and is checked without a port.
    "port_tensor_envelope_i64": ('fn main()->bool{ let !g=forge.zeros[i64,[2,3]]\n g[0,0]=1 g[0,1]=2 g[0,2]=3 g[1,0]=-4 g[1,1]=5 g[1,2]=6\n port_tensor_encode(g) == "{\\"data\\":\\"AQAAAAAAAAACAAAAAAAAAAMAAAAAAAAA/P////////8FAAAAAAAAAAYAAAAAAAAA\\",\\"dmc_tensor\\":1,\\"dtype\\":\\"i64\\",\\"layout\\":\\"row_major\\",\\"shape\\":[2,3]}" }', "PORTS §3.2 the copy-mode envelope, byte for byte"),
    "port_tensor_envelope_bool_width": ('fn main()->bool{ let !b=forge.uninit[bool,[3]]\n b[0]=true b[1]=false b[2]=true\n port_tensor_encode(b) == "{\\"data\\":\\"AQAB\\",\\"dmc_tensor\\":1,\\"dtype\\":\\"bool\\",\\"layout\\":\\"row_major\\",\\"shape\\":[3]}" }', "PORTS §3.2 the wire dtype states the payload's real width — 1 byte per bool (#292)"),
    "port_tensor_u8_no_wire_dtype": ('fn main()->bool{ let !a=forge.zeros[u8,[2]]\n a[0]=5\n port_tensor_encode(a) == "{\\"data\\":\\"BQAAAAAAAAAAAAAAAAAAAA==\\",\\"dmc_tensor\\":1,\\"dtype\\":\\"i64\\",\\"layout\\":\\"row_major\\",\\"shape\\":[2]}" }', "#632 review: the interpreter crosses a `u8` tensor tagged `i64` (PORTS §3.2); the JIT's copy-mode helpers have no wire dtype for it and cleanly refuse (jit gap)"),
    "port_tensor_round_trip_i64": ('fn main()->bool{ let !g=forge.zeros[i64,[2,2]]\n g[0,0]=10 g[0,1]=-20 g[1,0]=30 g[1,1]=-40\n let (t,e)=port_tensor_decode(port_tensor_encode(g), forge.zeros[i64,[2,2]])\n e == nil && t[0,1] == -20 && t[1,1] == -40 }', "PORTS §3.2 an integer tensor survives the round trip exactly"),
    "port_tensor_round_trip_bool": ('fn main()->bool{ let !b=forge.uninit[bool,[3]]\n b[0]=true b[1]=false b[2]=true\n let (t,e)=port_tensor_decode(port_tensor_encode(b), forge.uninit[bool,[3]])\n e == nil && t[0] && !t[1] && t[2] }', "PORTS §3.2 a bool tensor comes back as bools, not as numbers"),
    "port_tensor_decode_shape_tag": ('fn main()->bool{ let !g=forge.zeros[i64,[4]]\n let (t,e)=port_tensor_decode(port_tensor_encode(g), forge.zeros[i64,[2]])\n e == "decode-type: expected tensor shape [2], got [4]" && t[0] == 0 }', "PORTS §3.1/§3.2 an undeclared shape is decode-type, and the zero rides the error path"),
    "port_tensor_decode_kind_tag": ('fn main()->bool{ let !f=forge.zeros[f32,[2]]\n let (t,e)=port_tensor_decode(port_tensor_encode(f), forge.zeros[i64,[2]])\n e == "decode-type: expected a `i64` tensor, got `f32`" }', "PORTS §3.1 a typed tensor decode never coerces across a kind"),
    "port_tensor_decode_parse_tag": ('fn main()->bool{ let (t,e)=port_tensor_decode("{oops", forge.zeros[i64,[2]])\n e.starts_with("decode-parse") && t[0] == 0 && t[1] == 0 }', "PORTS §6 text that is not JSON is decode-parse, not decode-type"),
    "port_tensor_copy_through_the_port": ('fn main()->bool{ let (p,e)=port_open("python")\n if e != nil { false } else { let !g=forge.zeros[i64,[2,2]]\n g[0,0]=1 g[0,1]=2 g[1,0]=3 g[1,1]=4\n let (o,e2)=port_call(p,"dmc.echo","[" + port_tensor_encode(g) + "]")\n let (s,e3)=port_call(p,"dmc.shape","[" + port_tensor_encode(g) + "]")\n let (_,_)=port_close(p)\n let (t,e4)=port_tensor_decode(o, forge.zeros[i64,[2,2]])\n e2 == nil && e4 == nil && s == "[2,2]" && t[1,1] == 4 } }', "PORTS §3.2 a tensor crosses a live port by copy and comes back"),
    # ── @grad reverse-tape widths and shapes (#422 wall 2) ──────────────────
    # `f$fwd_bwd` is a second function the JIT emits from the same body, and
    # for a long time nothing in this battery compiled one. These four cover
    # the shapes that used to make Cranelift reject it outright — a compile
    # failure, so they would have scored as a divergence, not a gap.
    "grad_unsuffixed_scalar_mul": (
        "@grad fn f(!w:Tensor[f32,[3]])->f32{ let s=sum(w)\n s*2.0 }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[3]]\n w[0]=1.0 w[1]=2.0 w[2]=3.0\n"
        " let (l,g)=f.fwd_bwd(w)\n (l + g.w[0]) as f64 }",
        "#422 wall 2: unsuffixed literal (f64) times a traced scalar — 12+2"),
    "grad_unsuffixed_scalar_div": (
        "@grad fn f(!w:Tensor[f32,[3]])->f32{ let s=sum(w)\n s/4.0 }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[3]]\n w[0]=1.0 w[1]=2.0 w[2]=3.0\n"
        " let (l,g)=f.fwd_bwd(w)\n (l + g.w[0]) as f64 }",
        "#422 wall 2: the other operand-reading rule — 1.5+0.25"),
    "grad_loss_declared_width": (
        "@grad fn f(!w:Tensor[f32,[3]])->f32{ let s=sum(w)\n s+0.1 }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[3]]\n w[0]=1.0 w[1]=2.0 w[2]=3.0\n"
        " let (l,_g)=f.fwd_bwd(w)\n l as f64 }",
        "#473: `.fwd_bwd`'s loss narrows to the declared `-> f32`, as a plain call does"),
    "grad_matmul_rank3": (
        "@grad fn f(!a:Tensor[f32,[1,1,2]], b:Tensor[f32,[1,2,2]])->f32{ let y=a@b\n sum(y .* y) }\n"
        "fn main()->f64{ let !a=forge.zeros[f32,[1,1,2]]\n a[0,0,0]=1.0 a[0,0,1]=2.0\n"
        " let !b=forge.zeros[f32,[1,2,2]]\n b[0,0,0]=3.0 b[0,0,1]=4.0 b[0,1,0]=5.0 b[0,1,1]=6.0\n"
        " let (l,g)=f.fwd_bwd(a,b)\n (l + g.a[0,0,0]) as f64 }",
        "#422: rank-3 batched matmul VJP — the backward transposes, and rank 3 was missing"),
    # ── model-field bindings (#524, SPEC §3.4) ──────────────────────────────
    # `let !y = m.w` on a TENSOR field is a LIVE ALIAS on both backends: every
    # write through the binding reaches the field. The JIT used to copy here
    # (its #249 value-copy rule swept up the field case too), so each of the
    # first three probes was a silent wrong-answer divergence: `dmc run` saw
    # the field move, `dmc jit` saw it stand still. The negatives that follow
    # pin the other half of the rule — `mut`, a plain `let`, a scalar field,
    # and a plain local all still snapshot, which is what #249 fixed.
    "field_alias_elem_write": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let !y=m.w\n y[0]=42.0\n m.w[0] as i64 }",
        "#524 element write through a tensor-field alias reaches the field"),
    "field_alias_whole_assign": (
        "model M { !w: Tensor[f32,[4]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[4]] }\n"
        " let !y=m.w\n y = forge.ones[f32,[4]] .* 7.0\n (m.w[0]+m.w[3]) as i64 }",
        "#524 whole-binding `=` through the alias writes the field, not the local"),
    "field_alias_compound_assign": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let !y=m.w\n y += forge.ones[f32,[2]]\n y += forge.ones[f32,[2]]\n m.w[1] as i64 }",
        "#524 compound `+=` reads the field and writes the result back to it"),
    # #693: `t[i] += v` had NO match arm at all — it fell to the interpreter's
    # catch-all no-op (`--check` accepted it, nothing stored). `dmc jit`
    # already refused the form outright (`jit unsupported: compound
    # assignment to tensor elements`, unchanged by this fix — see jit.rs's
    # `Refusal::AssignForm`). The interpreter now computes it via #689's
    # in-place indexed-store path, so `run` goes from silently-wrong to
    # correct while `jit` keeps refusing: a tracked jit-gap (GAP_ALLOWLIST
    # below), not a new run/jit divergence — the JIT's answer didn't change.
    # Note: this probe scores `jit-gap` on `rs == "ok" and _UNSUPPORTED in
    # jv` (see the scoring loop below) WITHOUT ever comparing `rv` — so it
    # scores identically whether the interpreter's answer is 5 (this fix) or
    # the pre-#693 buggy 0. It does not pin the fix; the regression pin
    # lives in the Rust tests (`interp_tests.rs`'s
    # `indexed_tensor_compound_*_693` family, which do assert `rv`). This
    # entry stays for corpus-battery jit-gap tracking only.
    "indexed_tensor_compound_assign_693": (
        "fn main()->i64{ let !t=forge.zeros[i64,[4]]\n t[0]+=5\n t[0] }",
        "#693 run computes 5 via the #689 in-place store path; jit still "
        "refuses (`compound assignment to tensor elements`)"),
    # #693 follow-up (#710 review): `m.field += v` with NO index used to
    # error outright on both backends (no divergence, but the wrong backend
    # was the strict one). Coordinator decision: the interpreter computes it
    # now, mirroring the plain `m.field = expr` field-write arm; `dmc jit`
    # is unchanged and still refuses (`compound assignment to a model
    # field`) — a tracked jit-gap, not a new divergence, same shape as
    # `indexed_tensor_compound_assign_693` above. Same caveat applies: this
    # probe cannot pin the VALUE `run` computes (see the note above); that
    # lives in `interp_tests.rs`'s `direct_field_compound_assign_*_693`
    # family.
    "direct_field_compound_assign_693": (
        "model M { !n: i64 }\nfn main()->i64{ let !m=M { n: 1 }\n m.n += 1\n m.n }",
        "#693 follow-up (#710 review): run computes 2; jit still refuses "
        "(`compound assignment to a model field`)"),
    "field_alias_mut_spelling_copies": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let mut y=m.w\n y[0]=42.0\n (m.w[0]+y[0]*100.0) as i64 }",
        "#524 `let mut` never aliases — this is where the two spellings differ"),
    "field_alias_immutable_bind_copies": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let y=m.w\n m.w[0]=42.0\n (y[0]+m.w[0]*100.0) as i64 }",
        "#524 a plain `let` from a field is a snapshot"),
    "field_alias_scalar_field_copies": (
        "model M { !n: i64 }\n"
        "fn main()->i64{ let !m=M { n: 7 }\n let !s=m.n\n s=99\n m.n*100+s }",
        "#524/#444 a SCALAR field binds its current value, not an alias"),
    "field_alias_plain_local_copies": (
        "fn main()->i64{ let !x=forge.zeros[f32,[3]]\n x[0]=1.0\n let !y=x\n y[0]=9.0\n"
        " (x[0]+y[0]*100.0) as i64 }",
        "#249 must NOT regress: a plain local still value-copies"),
    "field_alias_constructor_copies": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !t=forge.zeros[f32,[2]]\n let !m=M { w: t }\n t[0]=9.0\n"
        " m.w[0] as i64 }",
        "#249 must NOT regress: `M { w: t }` snapshots the constructor argument"),
    "field_alias_shadowed": (
        "model M { !w: Tensor[f32,[2]] }\n"
        "fn main()->i64{ let !m=M { w: forge.zeros[f32,[2]] }\n"
        " let !y=m.w\n y[0]=5.0\n let !y=forge.zeros[f32,[2]]\n y[0]=9.0\n"
        " (m.w[0]+y[0]*100.0) as i64 }",
        "#524 a shadowing `let` drops the alias — the later write stays local"),
    # ── non-f32 tensor value copies (#552) ──────────────────────────────────
    # #249's snapshot (`let !y = x`, `M { w: t }`, a tuple element, and the
    # `!`-argument copy/writeback) lowered to a loop that loaded EVERY element
    # as f32 while striding by the tensor's real element width. For an i64/f64
    # tensor that copied only the low 4 bytes of each element and left the high
    # 4 bytes as whatever the destination arena already held. A fresh forge
    # allocation is zeroed, so small positive values came out intact — which is
    # why the whole corpus stayed green. Every probe here therefore carries
    # bits ABOVE 2^32 (i64) or in the exponent (f64), where the two backends
    # part company: `dmc jit` answered 705032704 for 5000000000, and turned an
    # ordinary f64 into a denormal near zero.
    "copy_i64_let": (
        "fn main()->i64{ let !x=forge.uninit[i64,[2]]\n x[0]=5 x[1]=5000000000\n"
        " let !y=x\n y[0]=9\n y[1] }",
        "#552 `let !y = x` copies an i64 tensor at full width"),
    "copy_i64_ctor": (
        "model M { !w: Tensor[i64,[2]] }\n"
        "fn main()->i64{ let !t=forge.uninit[i64,[2]]\n t[0]=5000000000 t[1]=1\n"
        " let !m=M { w: t }\n t[0]=1\n m.w[0] }",
        "#552 `M { w: t }` snapshots an i64 tensor at full width"),
    "copy_i64_tuple": (
        "fn main()->i64{ let !x=forge.uninit[i64,[2]]\n x[0]=5000000000 x[1]=1\n"
        " let p=(x,7)\n x[0]=1\n let (a,b)=p\n a[0] }",
        "#552 a tuple element snapshots an i64 tensor at full width"),
    "copy_f64_let": (
        "fn main()->f64{ let !x=forge.uninit[f64,[2]]\n x[0]=0.5 x[1]=0.1\n"
        " let !y=x\n y[0]=9.0\n y[1] }",
        "#552 `let !y = x` copies an f64 tensor at full width (a 4-byte read loses the exponent)"),
    "copy_f64_ctor": (
        "model M { !w: Tensor[f64,[2]] }\n"
        "fn main()->f64{ let !t=forge.uninit[f64,[2]]\n t[0]=0.1 t[1]=0.25\n"
        " let !m=M { w: t }\n t[0]=9.0\n m.w[0]+m.w[1] }",
        "#552 `M { w: t }` snapshots an f64 tensor at full width"),
    # A bool element is 1 byte, so the old f32 read returned the RIGHT value
    # (the 4-byte store just rewrote its neighbours with their own bytes) while
    # running 3 bytes past the end of both buffers. These two agreed before and
    # must keep agreeing now that the copy is exactly `n` bytes wide.
    "copy_bool_let": (
        "fn main()->i64{ let !x=forge.uninit[bool,[2]]\n x[0]=false x[1]=true\n"
        " let !y=x\n y[0]=true\n if y[1] {1} else {0} }",
        "#552 `let !y = x` copies a bool tensor without over-reading past it"),
    "copy_bool_ctor": (
        "model M { !w: Tensor[bool,[2]] }\n"
        "fn main()->i64{ let !t=forge.uninit[bool,[2]]\n t[0]=true t[1]=false\n"
        " let !m=M { w: t }\n t[0]=false\n if m.w[0] {1} else {0} }",
        "#552 `M { w: t }` snapshots a bool tensor without over-reading past it"),
    # The `!`-ARGUMENT copy/writeback uses the same loop. It was unreachable
    # while the JIT lowered `!` parameters only inside a `@grad fn` (and
    # `@grad` is f32-only), so all three were tracked gaps rather than silent
    # agreement. The i64 one is live now; the other two stop on the element
    # type instead, one layer below mutability.
    "copy_i64_mut_arg": (
        "fn bump(!t: Tensor[i64,[2]])->nil{ t[0]=t[0]+1  nil }\n"
        "fn main()->i64{ let !x=forge.uninit[i64,[2]]\n x[0]=5000000000 x[1]=0\n"
        " bump(x)\n x[0] }",
        "`!` i64 tensor argument round-trips at full width — the copy/writeback "
        "loop is width-generic, so a value past 2^32 must survive it"),
    "copy_f64_mut_arg": (
        "fn bump(!t: Tensor[f64,[2]])->nil{ t[0]=t[0]+0.25  nil }\n"
        "fn main()->f64{ let !x=forge.uninit[f64,[2]]\n x[0]=0.5 x[1]=0.1\n"
        " bump(x)\n x[0]+x[1] }",
        "#552 `!` f64 tensor argument round-trips at full width (jit gap: no f64 tensors)"),
    "copy_bool_mut_arg": (
        "fn flip(!t: Tensor[bool,[2]])->nil{ t[0]=true  nil }\n"
        "fn main()->i64{ let !x=forge.uninit[bool,[2]]\n x[0]=false x[1]=true\n"
        " flip(x)\n if x[0] {1} else {0} }",
        "#552 `!` bool tensor argument round-trips (jit gap: no bool tensors)"),
    # ── `!` parameters outside `@grad fn` ───────────────────────────────────
    # Two representations, two conventions: a tensor is copied in and copied
    # back, a model is passed as the pointer it already is. The probes that
    # matter are the ones where those two disagree about aliasing.
    "mut_model_param_field_write": (
        "model G { !k: Tensor[i64,[4]] }\n"
        "fn alloc(!g: G, id: i64, v: i64)->i64{ g.k[id]=v  id }\n"
        "fn main()->i64{ let !g=G{ k: forge.zeros[i64,[4]] }\n"
        " let a=alloc(g,1,7)\n let b=alloc(g,2,9)\n"
        " b*1000 + g.k[1]*10 + g.k[2] + a*0 }",
        "a `!` model parameter's field writes reach the caller's instance"),
    "mut_model_param_aliases_a_by_value_copy": (
        "model C { !n: Tensor[i64,[1]] }\n"
        "fn poke(!a: C, b: C)->i64{ a.n[0]=5  b.n[0] }\n"
        "fn main()->i64{ let !c=C{ n: forge.zeros[i64,[1]] }\n"
        " let seen=poke(c,c)\n seen*10 + c.n[0] }",
        "a model is one instance under both names — where a `!` TENSOR arg is "
        "isolated from an aliased by-value one, a `!` model arg is not"),
    "mut_tensor_param_chained_two_frames": (
        "fn inner(!t: Tensor[i64,[3]], v: i64)->i64{ t[1]=v  t[1] }\n"
        "fn outer(!t: Tensor[i64,[3]], v: i64)->i64{ t[0]=v*2  inner(t,v) }\n"
        "fn main()->i64{ let !t=forge.zeros[i64,[3]]\n"
        " let r=outer(t,3)\n r*1000 + t[0]*10 + t[1] }",
        "a `!` argument forwarded to another `!` parameter: three copies have "
        "to chain, and losing any one drops the write"),
    "mut_param_let_bang_inside_stays_a_copy": (
        "fn touch(!t: Tensor[i64,[2]])->i64{ let !y=t\n y[0]=99\n t[0]=1\n y[0] }\n"
        "fn main()->i64{ let !t=forge.zeros[i64,[2]]\n"
        " let seen=touch(t)\n seen*10 + t[0] }",
        "SPEC §3.4: taking the parameter by reference must not turn a "
        "`let !y = t` copy inside the callee into an alias"),
    # ── the two forms that are REFUSED, pinned as gaps (#650) ───────────────
    # Both are allowlisted above. What each probe records is `dmc run`'s
    # answer — the conformant one — so that the day the JIT lowers either
    # form, this battery says whether it got it right instead of merely
    # noticing it stopped refusing.
    "mut_param_fn_as_a_value": (
        "fn poke(!a: Tensor[i64,[2]], b: Tensor[i64,[2]])->i64{ a[0]=5  b[0] }\n"
        "fn main()->i64{ let !t=forge.zeros[i64,[2]]\n"
        " let f: fn(Tensor[i64,[2]],Tensor[i64,[2]])->i64 = poke\n"
        " let seen=f(t,t)\n seen*10 + t[0] }",
        "a `!` param through a fn VALUE: `dmc run` isolates the aliased "
        "by-value arg (5); an indirect call with no copy-in/copy-out would "
        "alias it (55), so the JIT refuses instead"),
    "mut_model_param_rebind": (
        "model C { !n: Tensor[i64,[1]] }\n"
        "fn replace(!c: C)->i64{ c = C{ n: forge.zeros[i64,[1]] }\n"
        " c.n[0]=9\n c.n[0] }\n"
        "fn main()->i64{ let !c=C{ n: forge.zeros[i64,[1]] }\n"
        " let r=replace(c)\n r*10 + c.n[0] }",
        "a `!` model param REBOUND wholesale: `dmc run` moves the caller's "
        "binding onto the fresh instance (99); the JIT has no slot to carry "
        "the new pointer back (90), so it refuses instead"),
    "copy_f32_grad_arg": (
        "@grad fn loss(!w: Tensor[f32,[2]])->f32{ sum(w .* w) }\n"
        "fn main()->f64{ let !w=forge.zeros[f32,[2]]\n w[0]=1.5 w[1]=2.0\n"
        " let (l,g)=loss.fwd_bwd(w)\n (l+g.w[0]) as f64 }",
        "#552 the one `!` tensor-argument copy the JIT does lower stays correct"),
    # ── `!` parameter copy-in/copy-out (#249, #553) ─────────────────────────
    # A `!` tensor parameter is copy-in/copy-out: the caller hands the callee a
    # private copy and copies that buffer back on return. #553: a whole-binding
    # assignment in the callee rebinds the local to a FRESH forge buffer, so the
    # buffer the caller wrote back was the stale one it passed in — `dmc run`
    # saw the new values, `dmc jit` the old. Element writes hid it, since they
    # mutate the copy in place. Note the two shapes a `!` parameter can take
    # here: a `@grad fn` (where `declare_fn` requires the directive) and a
    # template (where `declare_template_signature` accepts one without it).
    "mut_param_reassign_writes_back": (
        "@grad fn f(!w:Tensor[f32,[2]])->f32{ w = w .* 2.0\n sum(w .* w) }\n"
        "fn main()->i64{ let !w=forge.zeros[f32,[2]]\n w[0]=1.0 w[1]=2.0\n"
        " let s=f(w)\n (s + w[0]*100.0) as i64 }",
        "#553 a reassigned `!` param writes back the buffer it now names — 220, not 120"),
    "mut_param_reassign_writes_back_template": (
        "fn scale[N](!t:Tensor[f32,[N]])->f32{ t = t .* 2.0\n t[0] }\n"
        "fn main()->i64{ let !x=forge.zeros[f32,[3]]\n x[0]=1.0\n"
        " let d=scale(x)\n (d + x[0]*100.0) as i64 }",
        "#553 the same on a template `!` param — no `@grad` needed to reach one"),
    "mut_param_reassign_leaves_others_alone": (
        "@grad fn f(!a:Tensor[f32,[2]], !b:Tensor[f32,[2]])->f32{ a = a .* 3.0\n sum(a .* b) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[2]]\n let !b=forge.zeros[f32,[2]]\n"
        " a[0]=1.0 a[1]=2.0 b[0]=5.0 b[1]=7.0\n let s=f(a,b)\n"
        " (s + a[0]*100.0 + b[0]*10000.0) as i64 }",
        "#553 only the reassigned param moves — `b` comes back as it went in"),
    "mut_param_reassign_isolated_from_aliased_arg": (
        "@grad fn f(!a:Tensor[f32,[2]], b:Tensor[f32,[2]])->f32{ a = a .* 2.0\n sum(a) + sum(b) }\n"
        "fn main()->i64{ let !t=forge.zeros[f32,[2]]\n t[0]=1.0 t[1]=2.0\n"
        " let s=f(t,t)\n (s + t[0]*100.0) as i64 }",
        "#249 must NOT regress: `f(t,t)` keeps the reassignment out of the plain `b`"),
    "mut_param_no_reassign_writes_back": (
        "fn poke[N](!t:Tensor[f32,[N]])->nil{ t[0]=77.0 nil }\n"
        "fn main()->i64{ let !x=forge.zeros[f32,[3]]\n poke(x)\n x[0] as i64 }",
        "#249 must NOT regress: an element write still reaches the caller"),
    "mut_param_fwd_bwd_does_not_write_back": (
        "@grad fn f(!w:Tensor[f32,[2]])->f32{ w = w .* 2.0\n sum(w .* w) }\n"
        "fn main()->i64{ let !w=forge.zeros[f32,[2]]\n w[0]=1.0 w[1]=2.0\n"
        " let (l,_g)=f.fwd_bwd(w)\n (l + w[0]*100.0 + w[1]*10000.0) as i64 }",
        "#553 `.fwd_bwd` passes the caller's own tensor and writes back nothing, on both backends"),
    # ── slab bounds built from a SHAPE PARAMETER ────────────────────────────
    # Every slice this battery and the corpus reached had LITERAL bounds. A
    # bound that mentions `S` cannot be resolved until the generic is
    # monomorphized, and the two backends reach it by different routes: the
    # interpreter substitutes the binding at call time, the JIT folds a
    # constant into the body it emits per instantiation. Nothing gated that.
    # Each probe therefore calls one generic at TWO shapes and packs both
    # answers into a single scalar, so a backend that folded once and reused
    # the result returns a plausible number that is arithmetically wrong.
    "slab_end_is_shape_param": (
        "fn tail[S](x:Tensor[f32,[S]])->f32{ sum(x[S-2 .. S]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[6]]\n for i in 0..6 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[3]]\n for i in 0..3 { b[i]=(i+10) as f32 }\n"
        " (tail(a)*1000.0+tail(b)) as i64 }",
        "a fixed-length window anchored at `S` — 11 on [6], 23 on [3]"),
    "slab_len_is_shape_param": (
        "fn half[S](x:Tensor[f32,[S]])->f32{ sum(x[0 .. S/2]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[5]]\n for i in 0..5 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[6]]\n for i in 0..6 { b[i]=(i+1) as f32 }\n"
        " (half(a)*1000.0+half(b)) as i64 }",
        "the slab's own LENGTH moves, and `S/2` truncates on the odd shape — 3 then 6"),
    "slab_both_ends_derived": (
        "fn mid[S](x:Tensor[f32,[S]])->f32{ sum(x[1 .. S-1]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[6]]\n for i in 0..6 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[3]]\n for i in 0..3 { b[i]=(i+10) as f32 }\n"
        " (mid(a)*1000.0+mid(b)) as i64 }",
        "start and end both derived — 14 on [6], 11 on [3]"),
    "slab_full_range_via_param": (
        "fn whole[S](x:Tensor[f32,[S]])->f32{ sum(x[0 .. S]) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[6]]\n for i in 0..6 { a[i]=(i+1) as f32 }\n"
        " whole(a) as i64 }",
        "`x[0 .. S]` is the whole tensor — an off-by-one in the derived end shows only here"),
    "slab_2d_row_from_param": (
        "fn lastrow[R,C](m:Tensor[f32,[R,C]])->f32{ sum(m[R-1 .. R, ..]) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " lastrow(m) as i64 }",
        "row slab from `R` on a grid of distinct elements — 9+10+11"),
    "slab_2d_col_from_param": (
        "fn righthalf[R,C](m:Tensor[f32,[R,C]])->f32{ sum(m[.., C/2 .. C]) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " righthalf(m) as i64 }",
        "column slab from `C`, applied down every row — 48"),
    "slab_is_a_matmul_operand": (
        "fn slabmm[R,C](m:Tensor[f32,[R,C]], w:Tensor[f32,[C,C]])->f32{ sum(m[0 .. R/2, ..] @ w) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " let !w=forge.zeros[f32,[3,3]]\n for i in 0..3 { w[i,i]=1.0 }\n"
        " slabmm(m,w) as i64 }",
        "the derived extent has to survive into `@`, not just into a reduction — 15"),
    "slab_feeds_axis_reduction": (
        "fn slaba[R,C](m:Tensor[f32,[R,C]])->f32{ sum(sum_along(m[0 .. R-1, ..], 0)) }\n"
        "fn main()->i64{ let !m=forge.zeros[f32,[4,3]]\n"
        " for i in 0..4 { for j in 0..3 { m[i,j]=(i*3+j) as f32 } }\n"
        " slaba(m) as i64 }",
        "a derived slab as the argument of a per-axis reduction — 36"),
    # ── the shape parameter as a VALUE, and tensor-valued joins ─────────────
    # `N` is also an ordinary integer in the body: a `forge` extent, a loop
    # bound, an `N as f32` divisor. And the `if_*` family above joins only
    # i64 — a join whose arms are TENSORS carries a buffer through the phi,
    # which outside `@grad` nothing reached.
    "shape_param_as_divisor": (
        "fn avg[N](x:Tensor[f32,[N]])->f32{ let !acc=forge.zeros[f32,[N]]\n"
        " for i in 0..N { acc[i]=x[i]/(N as f32) }\n sum(acc) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[8]]\n for i in 0..8 { b[i]=(i+1) as f32 }\n"
        " ((avg(a)*1000.0+avg(b))*10.0) as i64 }",
        "`N` is the local's extent, the loop bound AND the divisor — 2.5 then 4.5"),
    "shape_param_bounds_a_while": (
        "fn total[N](x:Tensor[f32,[N]])->f32{ let !i=0\n let !acc=0.0f32\n"
        " while i < N { acc = acc + x[i]\n i = i + 1 }\n acc }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " let !b=forge.zeros[f32,[8]]\n for i in 0..8 { b[i]=(i+1) as f32 }\n"
        " (total(a)*1000.0+total(b)) as i64 }",
        "the parameter as a `while` predicate — corpus `while` bounds are module constants"),
    "explicit_shape_arg_sizes_result": (
        "fn ramp[N]()->Tensor[f32,[N]]{ let !t=forge.zeros[f32,[N]]\n"
        " for i in 0..N { t[i]=(i*2) as f32 }\n t }\n"
        "fn main()->i64{ let a=ramp[5]()\n let b=ramp[3]()\n"
        " (sum(a)*1000.0+sum(b)) as i64 }",
        "no argument mentions `N`, so only the explicit shape argument can size the result"),
    "tensor_valued_if_join": (
        "fn branch[N](x:Tensor[f32,[N]], d:bool)->f32{ let y=if d { x .* 2.0 } else { x .+ 1.0 }\n sum(y) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " (branch(a,true)*1000.0+branch(a,false)) as i64 }",
        "both arms are TENSORS, so the phi carries a buffer — the `if_*` family above is all i64"),
    "tensor_valued_match_join": (
        "fn pick[N](x:Tensor[f32,[N]], k:i64)->f32{ let y=match k { 0 => x .* 2.0, 1 => x .+ 1.0, _ => x }\n sum(y) }\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n for i in 0..4 { a[i]=(i+1) as f32 }\n"
        " (pick(a,0)*10000.0+pick(a,1)*100.0+pick(a,9)) as i64 }",
        "…same at a match join, and the default arm yields a buffer the callee never allocated"),
    # #574 region scoping. The JIT kept `locals` as one flat map per function,
    # so a binding introduced by a match arm or a loop overwrote an outer local
    # of the same name for the rest of the function — and did it at lowering
    # time, which knows nothing about which branch runs, so an arm that never
    # executed still destroyed the name. Silent: both backends exited 0 and
    # printed different numbers. The corpus reaches none of these shapes,
    # because no example shadows an outer local with a region binding, which is
    # why diff_backends had nothing to fire on.
    "scope_payload_untaken_arm": (
        "enum Shape { Circle(f32), Empty }\n"
        "fn probe(s:Shape)->f32{ let r=2.0f32\n"
        " let a=match s { Circle(r) => r*r, Empty => 0.0f32, }\n r+0.5f32 }\n"
        "fn main()->i64{ (probe(Shape.Empty)*100.0) as i64 }",
        "the payload arm never runs; the outer `r` must still be 2.0 → 250"),
    "scope_payload_taken_arm": (
        "enum Shape { Circle(f32), Empty }\n"
        "fn probe(s:Shape)->f32{ let r=2.0f32\n"
        " let a=match s { Circle(r) => r*r, Empty => 0.0f32, }\n a+r }\n"
        "fn main()->i64{ (probe(Shape.Circle(3.0f32))*100.0) as i64 }",
        "the other direction: the payload binding must still work inside its own arm → 1100"),
    "scope_catch_all_arm": (
        "fn probe(s:i64)->i64{ let n=7\n"
        " let a=match s { 1 => 100, n => n*2, }\n n+a }\n"
        "fn main()->i64{ probe(1)*1000 + probe(5) }",
        "a catch-all arm binding the scrutinee is the same defect → 107017"),
    "scope_for_loop_index": (
        "fn probe()->i64{ let i=42\n for i in 0..3 { }\n i }\n"
        "fn main()->i64{ probe() }",
        "a loop index does not outlive its loop → 42, not the last value reached"),
    "scope_nested_regions": (
        "fn probe(a:i64,b:i64)->i64{ let n=1\n"
        " let outer=match a { 0 => 0, n => { let inner=match b { 0 => 0, n => n*10, }\n inner+n }, }\n"
        " outer+n }\n"
        "fn main()->i64{ probe(5,7) }",
        "the inner arm restores to the OUTER arm's binding, not the function's → 76"),
    # #630 block scoping — the same root cause at the fourth site. #574 gave
    # match arms and `for` indices their bookkeeping and stopped there, so a
    # `let` inside a BLOCK still wrote into the flat map and stayed there for
    # the rest of the function: `let a=1  let b={ let a=2  a+10 }  a+b` was 13
    # under `run` and 14 under `jit`, both exiting 0. Every construct whose
    # body is a block had it, through four different lowering paths, so each
    # path gets a probe. No corpus example shadows an outer local from inside a
    # block, which is again why diff_backends had nothing to fire on.
    "scope_block_expr": (
        "fn probe()->i64{ let a=1\n let b={ let a=2\n a+10 }\n a+b }\n"
        "fn main()->i64{ probe() }",
        "#630's repro: the inner `a` dies with the block → 13, not 14"),
    "scope_block_inner_visible": (
        "fn probe()->i64{ let a=1\n let b={ let a=2\n a*10 }\n a*1000+b }\n"
        "fn main()->i64{ probe() }",
        "the other direction: the shadow is in effect INSIDE its own block → 1020"),
    "scope_block_nested": (
        "fn probe()->i64{ let a=1\n"
        " let b={ let a=2\n let c={ let a=3\n a*100 }\n c+a }\n a+b }\n"
        "fn main()->i64{ probe() }",
        "an inner block restores to the enclosing BLOCK, not the function → 303"),
    "scope_block_stmt_position": (
        "fn probe()->i64{ let a=1\n { let a=2\n a }\n a }\n"
        "fn main()->i64{ probe() }",
        "a block in statement position reaches a different arm, same scope → 1"),
    "scope_block_if_branch": (
        "fn probe(c:bool)->i64{ let a=1\n if c { let a=2 } else { let a=3 }\n a }\n"
        "fn main()->i64{ probe(true)*10 + probe(false) }",
        "an `if` branch is a block; its `let` dies with the branch → 11"),
    "scope_block_while_body": (
        "fn probe()->i64{ let a=1\n let !i=0\n while i<3 { let a=2\n i=i+a }\n a+i }\n"
        "fn main()->i64{ probe() }",
        "a `while` body is a block — the body sees a=2, the function keeps a=1 → 5"),
    "scope_block_loop_body": (
        "fn probe()->i64{ let a=1\n let !i=0\n"
        " loop { let a=2\n i=i+a\n if i>=4 { break } }\n a+i }\n"
        "fn main()->i64{ probe() }",
        "same for `loop` → 5"),
    "scope_block_for_body": (
        "fn probe()->i64{ let a=1\n let !s=0\n for i in 0..3 { let a=2\n s=s+a }\n a+s }\n"
        "fn main()->i64{ probe() }",
        "#574 scoped the `for` INDEX; the body's own `let`s are a second scope → 7"),
    "scope_block_directive": (
        "fn probe()->i64{ let a=1\n @deterministic { let a=2 }\n a }\n"
        "fn main()->i64{ probe() }",
        "a directive block is a block → 1"),
    "scope_block_match_arm": (
        "fn probe(k:i64)->i64{ let a=1\n let b=match k { 1 => { let a=9\n a+1 }, _ => 0, }\n"
        " a*100+b }\n"
        "fn main()->i64{ probe(1) }",
        "an arm whose body is a block: #574 scoped the arm, not the block → 110"),
    "scope_block_then_outer_write": (
        "fn probe()->i64{ let !x=1\n let b={ let x=100\n x+1 }\n x=x+2\n x*1000+b }\n"
        "fn main()->i64{ probe() }",
        "the shadow is undone before the later write to the outer `let !` → 3101"),
    "scope_block_write_through": (
        "fn probe()->i64{ let !x=1\n let b={ x=x+4\n x*2 }\n x*100+b }\n"
        "fn main()->i64{ probe() }",
        "must NOT regress: a non-shadowing write from inside a block still writes through → 510"),

    # #505 @comptime v1. Before it, `dmc run` answered and `dmc jit` refused
    # every one of these — the directive was a parity hole, not just a no-op.
    "comptime_fold_int": (
        "fn main()->i64{ let x = @comptime { 3 * 7 + 1 }\n x }",
        "#505 tier-1: a closed block folds to a literal both backends see → 22"),
    "comptime_fold_bool": (
        "fn main()->i64{ let c = @comptime { 2 > 1 }\n if c { 1 } else { 0 } }",
        "#505 a folded boolean is what #384 Option A needs → 1"),
    "comptime_fold_loop": (
        "fn main()->i64{ let x = @comptime { let !a=0\n let !i=1\n"
        " while i <= 4 { a += i\n i += 1 }\n a }\n x }",
        "#505 real evaluation under the step budget, not literal folding → 10"),
    "comptime_int_semantics": (
        "fn main()->i64{ let x = @comptime { (0-7)/2 }\n x }",
        "#505 the fold uses the interpreter's truncating division → -3"),
    "comptime_shape_param": (
        "fn tile[N](x:Tensor[f32,[N]])->i64{ let k = @comptime { N * 2 }\n k }\n"
        "fn main()->i64{ tile(forge.zeros[f32,[4]]) }",
        "#505 tier-2: residual block, folded per monomorphization by each backend → 8"),

    # #578 `extern fn`. Classified by INTERP_REFUSAL above: `dmc run` refuses by
    # spec, `dmc jit` answers. The three symbols are libc/libm and are in every
    # CI host's process image, so the probes need no linkage of their own.
    "extern_libc_abs": (
        "extern fn abs(x: i32) -> i32\nfn main()->i64{ abs(-7) as i64 }",
        "#578 the C ABI scalar call path, integer in / integer out → 7"),
    "extern_libc_sqrtf": (
        "extern fn sqrtf(x: f32) -> f32\n"
        "fn main()->i64{ (sqrtf(9.0f32) * 100.0f32) as i64 }",
        "#578 …and the float registers, which use a different ABI class → 300"),
    "extern_tensor_ptr": (
        "extern fn memcpy(dst: *f32, src: *f32, n: i64) -> *nil\n"
        "fn main()->i64{ let !a=forge.zeros[f32,[4]]\n let !b=forge.zeros[f32,[4]]\n"
        " b[0]=5.0f32\n memcpy(a, b, 16)\n (a[0]*10.0f32) as i64 }",
        "#578 SPEC §3.12 tensor-to-`*T` materialization, written through → 50"),
    # ── multi-file (the whole-program merge) ────────────────────────────────
    # The only probes in the battery that are not one file. All three were
    # `jit-import` refusals until slice 3 handed `compile_program` the merged
    # program; all three are in MUST_LOWER.
    "use_import_scalar": (
        "use \"probe_mod.dmc\"\nfn main()->i64{ probe_add(20, 22) }",
        "an unqualified import of a `pub fn`: both backends answer 42",
        {"probe_mod.dmc": "pub fn probe_add(a: i64, b: i64) -> i64 { a + b }\n"}),
    # The design's §0 diamond: one leaf reached over two edges, its `pub model`
    # crossing both boundaries as a field type. The shape that would go wrong
    # if the merge emitted a shared leaf twice, or renamed a `pub model` in one
    # importer's copy and not the other's.
    "use_import_diamond": (
        "use \"probe_boxed.dmc\"\nuse \"probe_sum.dmc\"\n"
        "fn main()->i64{ let bx = probe_boxed(4, 5)\n probe_sum_pt(bx.p) + bx.tag }",
        "a shared leaf's `pub model` across two import edges → 10",
        {"probe_point.dmc":
             "pub model Point { x: i64, y: i64 }\n"
             "pub fn probe_mk(x: i64, y: i64) -> Point { Point { x: x, y: y } }\n",
         "probe_boxed.dmc":
             "use \"probe_point.dmc\"\npub model Boxed { p: Point, tag: i64 }\n"
             "pub fn probe_boxed(x: i64, y: i64) -> Boxed { Boxed { p: probe_mk(x, y), tag: 1 } }\n",
         "probe_sum.dmc":
             "use \"probe_point.dmc\"\npub fn probe_sum_pt(p: Point) -> i64 { p.x + p.y }\n"}),
    # An aliased import reaches the backend as a postfix `Field` then `Call` on
    # an identifier bound to a module — the same shape a cross-file UFCS chain
    # has (#627), which is why the merge keys the rewrite on the `use ... as`
    # binding and not on the chain.
    "use_import_alias": (
        "use \"probe_alias_mod.dmc\" as u\n"
        "fn main()->i64{ u.probe_mk2(1, 3).x + u.probe_twice(4) }",
        "`use ... as` — an aliased fn and an aliased model constructor → 12",
        {"probe_alias_mod.dmc":
             "pub model P { x: i64 }\n"
             "pub fn probe_mk2(a: i64, b: i64) -> P { P { x: a + b } }\n"
             "pub fn probe_twice(v: i64) -> i64 { v * 2 }\n"}),
    # #724: a `for`-range endpoint is the one construct measured to leak an
    # imported module's PRIVATE top-level `let` past the checker (SPEC.md
    # §6.7 says a non-`pub` item is invisible to an importer). #731 closed
    # the checker half: `check_expr`'s `Expr::Range` arm now recurses into
    # `start`/`end`, so BOTH `dmc run` and `dmc jit` refuse this at
    # `--check`, before the JIT-specific `imported_private_let_module`
    # workaround is ever reached, with the checker's own `import-let`
    # diagnostic (`compiler/tests/jit_import_const_724.rs`). A regression
    # pin, not a tracked gap — no `GAP_ALLOWLIST` entry, since neither
    # backend answers.
    "use_import_private_let_leak_724": (
        "use \"probe_private_let_724.dmc\"\n"
        "fn main()->i64{ let !s = 0\n for k in 0..(CAP + 1) { s = s + k }\n s }",
        "#724/#731 SPEC §6.7 leak via a `for`-range endpoint: both backends "
        "refuse it at `--check` (`import-let`)",
        {"probe_private_let_724.dmc":
             "let CAP: i64 = 8\npub fn probe_private_let_noop() -> i64 { 0 }\n"}),

    # Error propagation `?` (SPEC §4.9) and the `(T, Err)` idiom's `nil` half
    # (SPEC §3.9) — the roadmap's JIT parity push. `Err` used to lower
    # as an opaque, unbuildable `TyKind::Model("Err", [])`, so every
    # `(T, Err)`-returning fn refused at `jit-conversion` the moment it
    # constructed its own return value — `?` itself was a flat `jit-construct`
    # refusal on top of that. Both are one gap: `Err` is now `TyKind::Str`
    # (nil = NULL pointer), the same representation `str`/`Port` already use
    # for "no error"/"no handle".
    #
    # Every probe below takes the error path with a value half that ALREADY
    # equals `default[T]` (0 / 0.0 / false / nil / a zero tensor), so the two
    # backends agree whatever the answer to #628 is. A callee whose error-path
    # value is NON-ZERO distinguishes SPEC §4.9's `return (default[T], e)`
    # (what the JIT emits) from returning the callee's whole tuple (what
    # `interp.rs`'s `propagate` does) — and those cases cannot live here,
    # because this battery fails on any run/jit divergence and has no way to
    # spell "known interpreter defect". They are JIT-value pins in
    # `jit_query_error_path_is_default_t_not_the_callees_value` (jit.rs)
    # instead, and move here once #628 lands.
    "query_unwraps_on_success": (
        "fn half(n:i64)->(i64,Err){ if n%2!=0 {return (0,\"odd\")}\n (n/2,nil) }\n"
        "fn twice_half(n:i64)->(i64,Err){ let h=half(n)?\n let h2=half(h)?\n (h2,nil) }\n"
        "fn main()->i64{ let (v,e)=twice_half(8)\n v }",
        "`?` unwraps to T on the no-error branch → 2"),
    "query_propagates_on_error": (
        "fn half(n:i64)->(i64,Err){ if n%2!=0 {return (0,\"odd\")}\n (n/2,nil) }\n"
        "fn twice_half(n:i64)->(i64,Err){ let h=half(n)?\n let h2=half(h)?\n (h2,nil) }\n"
        "fn main()->bool{ let (v,e)=twice_half(9)\n v==0 && e==\"odd\" }",
        "`?` early-returns (default[T], e) through the caller, not `t`"),
    "query_error_path_f32": (
        "fn f(n:i64)->(f32,Err){ if n<0 {return (0.0f32,\"neg\")}\n (1.0f32,nil) }\n"
        "fn g(n:i64)->(f32,Err){ let v=f(n)?\n (v,nil) }\n"
        "fn main()->bool{ let (v,e)=g(-1)\n v==0.0f32 && e==\"neg\" }",
        "`?` through an f32 value half — default[f32] is 0.0"),
    "query_error_path_bool": (
        "fn f(n:i64)->(bool,Err){ if n<0 {return (false,\"neg\")}\n (true,nil) }\n"
        "fn g(n:i64)->(bool,Err){ let v=f(n)?\n (v,nil) }\n"
        "fn main()->bool{ let (v,e)=g(-1)\n v==false && e==\"neg\" }",
        "`?` through a bool value half — default[bool] is false"),
    "query_error_path_str": (
        "fn f(n:i64)->(str,Err){ if n<0 {return (nil,\"neg\")}\n (\"ok\",nil) }\n"
        "fn g(n:i64)->(str,Err){ let v=f(n)?\n (v,nil) }\n"
        "fn main()->bool{ let (v,e)=g(-1)\n v==nil && e==\"neg\" }",
        "`?` through a str value half — default[str] is nil (a NULL pointer)"),
    "query_error_path_tensor": (
        "fn f(n:i64)->(Tensor[f32,[2]],Err){ let !t=forge.zeros[f32,[2]]\n"
        " if n<0 {return (t,\"neg\")}\n t[0]=5.0f32\n (t,nil) }\n"
        "fn g(n:i64)->(Tensor[f32,[2]],Err){ let v=f(n)?\n (v,nil) }\n"
        "fn main()->bool{ let (v,e)=g(-1)\n v[0]==0.0f32 && e==\"neg\" }",
        "`?` through a tensor value half — default[Tensor] is a zero tensor"),
    "query_inside_a_loop_body": (
        "fn f(n:i64)->(i64,Err){ if n==3 {return (0,\"three\")}\n (n,nil) }\n"
        "fn sum_to(n:i64)->(i64,Err){ let !acc=0\n"
        " for i in 0..n { let v=f(i)?\n acc=acc+v }\n (acc,nil) }\n"
        "fn main()->bool{ let (a,e)=sum_to(5)\n let (b,e2)=sum_to(3)\n"
        " a==0 && e==\"three\" && b==3 && e2==nil }",
        "`?` in a `for` body leaves the loop AND the fn"),
    "query_inside_a_nested_block": (
        "fn f(n:i64)->(i64,Err){ if n<0 {return (0,\"neg\")}\n (n*2,nil) }\n"
        "fn g(n:i64)->(i64,Err){ if n != 0 { let v=f(n)?\n return (v+1,nil) }\n (0,nil) }\n"
        "fn main()->bool{ let (v,e)=g(-1)\n let (v2,e2)=g(3)\n"
        " v==0 && e==\"neg\" && v2==7 && e2==nil }",
        "`?` inside an `if` block unwinds out of the block"),
    "query_twice_in_one_expression": (
        "fn f(n:i64)->(i64,Err){ if n<0 {return (0,\"neg\")}\n (n*2,nil) }\n"
        "fn g(a:i64,b:i64)->(i64,Err){ let s=f(a)?+f(b)?\n (s,nil) }\n"
        "fn main()->bool{ let (v,e)=g(1,2)\n let (v2,e2)=g(1,-1)\n"
        " v==6 && e==nil && v2==0 && e2==\"neg\" }",
        "two `?`s in one expression — the second is skipped once the first propagates"),
    "mutable_err_accumulator_idiom": (
        "fn check(n:i64)->str{ let !err=nil\n if n<0 {err=\"negative\"}\n"
        " if n>100 {err=\"too big\"}\n err }\n"
        "fn main()->bool{ check(5)==nil && check(-1)==\"negative\" && check(200)==\"too big\" }",
        "`let !err=nil` later reassigned a `str` — SPEC §3.9's mutable-Err idiom"),

    # #612 — arena blocks and valueless blocks, slice 1 of
    # the arena-blocks design. All ten are in MUST_LOWER: the JIT
    # refused every one of them until that slice landed, and the point of the
    # set is that the refusal cannot come back quietly.
    #
    # What each probe is FOR is the semantics the design measured on `dmc run`
    # and the JIT must not "improve" on: the block selects an arena at CHECK
    # time only (`MEMORY.md §3.1`), it rewinds nothing at the exit (#400), and
    # the tensors it allocates escape it — readable *and* writable afterwards.
    "arena_vault_expr_scalar": (
        "fn main()->i64{ let x = vault { let a = 2 + 3\n a }\n x }",
        "p1: `vault { }` in expression position is its tail value → 5"),
    "arena_stream_expr_scalar": (
        "fn main()->i64{ let x = stream { let a = 9\n a }\n x }",
        "p8: `stream { }` is the same scope, with no Stream arena behind it → 9"),
    "arena_forge_stmt": (
        "fn main()->i64{ let !s = 0\n forge { s = 7 }\n s }",
        "p2: an arena block in STATEMENT position lowers for effect → 7"),
    "plain_block_stmt": (
        "fn main()->i64{ let !s = 0\n { s = 7 }\n s }",
        "s1: a valueless PLAIN block in statement position — the #612 defect → 7"),
    "arena_nested_forge_vault": (
        "fn main()->i64{ let x = forge { vault { 2 + 2 } }\n x }",
        "p4: nesting resolves innermost-first and lowers as one scope → 4"),
    "arena_vault_tensor_read": (
        "fn main()->i64{ let t = vault { vault.ones[f32,[4]] }\n sum(t) as i64 }",
        "p5: a tensor allocated inside the block is readable after it → 4"),
    "arena_forge_tensor_escapes": (
        "fn main()->i64{ let t = forge { forge.ones[f32,[4]] }\n sum(t) as i64 }",
        "r1: the escapee is Forge-resident — a block-exit rewind would free it → 4"),
    "arena_escapee_is_writable": (
        "fn main()->i64{ let !t = forge { forge.zeros[f32,[4]] }\n t[0]=7.0\n t[0] as i64 }",
        "e1: and it is still WRITABLE — a rewind would store into freed bytes → 7"),
    "arena_vault_write_in_block": (
        "fn main()->i64{ let !w = vault.zeros[f32,[4]]\n vault { w[0]=5.0 }\n w[0] as i64 }",
        "p6: the cross-arena write `vault { }` legalises (#442, check-time) → 5"),
    "arena_cross_write_nested": (
        "fn main()->i64{ let !w = vault.zeros[f32,[4]]\n forge { vault { w[0]=3.0 } }\n w[0] as i64 }",
        "e2: innermost-wins — the mirror `vault { forge { … } }` is a check error → 3"),

    # #614 (the arena-budget design's §2, §7 slice 0) —
    # `forge.restore(mark)` rewinds the bump cursor so the memory is reused,
    # but returns nothing to the `--forge` meter on either backend. These are
    # m3, m8, x6, x7 and x8 from that brief's own probe set,
    # inlined here as scalar-returning
    # probes so the battery can score them; `FORGE_FLAG` below carries the
    # `--forge` ceiling each row needs, mirroring the flag its design-probe
    # counterpart documents in that directory's `README.md`. Before this fix
    # `dmc jit` accepted m3, m8, x6 and x8 under a budget `dmc run` refused —
    # the JIT's own bypass of #400 (`e6e8720`) through the explicit
    # snapshot/restore primitive rather than an undeclared `forge { }` block.
    # They are ordinary probes, not MUST_LOWER or GAP_ALLOWLIST entries: the
    # expected verdict is `both-fail` for m3/m8/x6/x8 (a regression to the old
    # JIT-only-accepts behavior falls into the DIVERGE catch-all and fails the
    # battery) and plain agreement for x7, whose answer #614 does not change.
    "forge_restore_no_budget_m3": (
        "fn main() -> i64 {\n"
        " for i in 0..40 {\n"
        "  let mark = forge.snapshot()\n"
        "  let t = forge.zeros[f32, [4096]]\n"
        "  forge.restore(mark)\n"
        " }\n"
        " 1\n"
        "}",
        "0674 m3: 40 iterations of snapshot/alloc[4096]/restore must exhaust "
        "`--forge=512K` on both backends now (both-fail; #781 halves this "
        "from `1M` — declared-width f32 charging) — `dmc jit` used to run "
        "all 40"),
    "forge_restore_no_budget_m8": (
        "fn main() -> i64 {\n"
        " let !i = 0\n"
        " while i < 200000 {\n"
        "  let mark = forge.snapshot()\n"
        "  let t = forge.ones[f32, [64]]\n"
        "  forge.restore(mark)\n"
        "  i = i + 1\n"
        " }\n"
        " 1\n"
        "}",
        "0674 m8: 200 000 snapshot/alloc[64]/restore iterations must exhaust "
        "`--forge=4M` on both backends now (both-fail; #781 halves this from "
        "`8M`) — `dmc jit` used to run "
        "all 200 000"),
    "forge_restore_no_budget_x6": (
        "fn hold(n: i64) -> i64 {\n"
        " if n <= 0 { return 0 }\n"
        " let mark = forge.snapshot()\n"
        " let !t = forge.zeros[f32, [8192]]\n"
        " forge.restore(mark)\n"
        " let rest = hold(n - 1)\n"
        " t[0] = 1.0f32\n"
        " rest + 1\n"
        "}\n"
        "fn main() -> i64 { hold(40) }",
        "0674 x6: #400's own repro (e6e8720), snapshot/restore in place of "
        "`forge { }` — refused under `--forge=128K` on both backends now "
        "(both-fail; #781 halves this from `256K`); `dmc jit` used to print "
        "`held 40`, granting #400's bypass"),
    "forge_restore_alive_after_restore_x7": (
        "fn main() -> i64 {\n"
        " let mark = forge.snapshot()\n"
        " let !t = forge.zeros[f32, [4]]\n"
        " forge.restore(mark)\n"
        " t[0] = 3.0f32\n"
        " t[0] as i64\n"
        "}",
        "0674 x7: `forge.restore` is memory reuse, not a reap — the tensor "
        "allocated above the mark is still alive and writable after the "
        "restore, on both backends -> 3 (no --forge flag; unaffected by "
        "#614)"),
    "forge_restore_no_budget_x8": (
        "fn hold(n: i64) -> i64 {\n"
        " if n <= 0 { return 0 }\n"
        " let mark = forge.snapshot()\n"
        " let t = forge.ones[f32, [8192]] .* (n as f32)\n"
        " forge.restore(mark)\n"
        " let rest = hold(n - 1)\n"
        " if t[0] == (n as f32) { rest + 1 } else { rest }\n"
        "}\n"
        "fn main() -> i64 { hold(40) }",
        "0674 x8: carve/allocate/reap/hold recursion (the primitives that "
        "exist) — refused under `--forge=128K` on both backends now "
        "(both-fail; #781 halves this from `256K`); `dmc jit` used to print "
        "`intact 1 of 40` with no diagnostic"),

    # #674 slice 2 (the arena-budget design's §2.1, §7 slice 2) —
    # c1's own property from the arena-budgets probe set, minus
    # the print/`to_str` machinery c1.dmc itself exercises (a separate,
    # already-measured JIT-vs-interpreter charging gap, unrelated to chunk
    # commitment — see this PR's description). Both constructors sit in
    # EXPRESSION position (not `let`-bound), so neither was ever subject to
    # #640's 2× `let`-bound double charge (fixed since; these two numbers
    # are unmoved by it, re-bisected): the small tensor costs its
    # interpreter-declared 16 B on `dmc run` and one aligned 64 B chunk on
    # `dmc jit`, the big one 1 MiB flat on both (already a 64-byte multiple).
    # Before this slice, `chunk_size_for` capped the FIRST chunk at what was
    # left of the budget, so a tiny allocation's chunk could silently
    # consume the entire `--forge` ceiling and every later exhaustion
    # message read "0 B free" regardless of what was actually live (§2.1's
    # own c1 measurement, 2026-09-05) — that mechanism is still in `main`
    # today (#779 did not touch it), but #779's own monotonic-`used` check
    # now fires first and reports the honest figure, so the observable "0 B
    # free" symptom is already gone on `main`. Slice 2 removes the
    # mechanism itself — a chunk is now sized from the request alone, so the two
    # backends' minimums for this program now differ by exactly the 64-byte
    # alignment tail on the SMALL tensor — 48 B (64 − 16) — nothing more:
    # `dmc run`'s own minimum is 1 048 592 B (16 + 1 048 576); `dmc jit`'s is
    # 1 048 640 B (64 + 1 048 576), which is therefore the SHARED minimum —
    # the smallest budget at which both backends run the program to
    # completion and agree.
    "chunk_granularity_c1_below_shared_min": (
        "fn main() -> i64 {\n"
        " sum(forge.zeros[f32, [4]]) as i64\n"
        " sum(forge.zeros[f32, [262144]]) as i64\n"
        "}",
        "0674 slice 2, one byte under `dmc run`'s own 1 048 592 B minimum "
        "(the lower of the two): both backends refuse with `forge arena "
        "exhausted` (both-fail) — pinned to that failure mode via "
        "BUDGET_EXHAUSTION_PROBES so a probe that broke some other way "
        "would not stay green"),
    "chunk_granularity_c1_at_shared_min": (
        "fn main() -> i64 {\n"
        " sum(forge.zeros[f32, [4]]) as i64\n"
        " sum(forge.zeros[f32, [262144]]) as i64\n"
        "}",
        "0674 slice 2, at `dmc jit`'s own 1 048 640 B minimum (the higher "
        "of the two, hence shared): both backends run to completion and "
        "agree (OK, => 0) — before #779 this budget refused under "
        "`dmc jit` with the arena reporting 0 B free no matter how little "
        "was actually live; `main` already prints `=> 0` here"),

    # #674 slice 2 — the divergence neither c1 probe above exercises: a
    # SECOND chunk. Two 270 000 000 B tensors (EXPRESSION position, so
    # neither was ever subject to #640's `let`-bound double charge, fixed
    # since and this number re-bisected unmoved) sum to
    # 540 000 000 B — under the aligned `--forge=556 870 912` (531.1 MiB)
    # budget used here, comfortably. Neither tensor fits alone in the
    # remaining space of the other's chunk (270 000 000 B > half of the
    # 536 870 912 B default chunk), so the second allocation must open a
    # second chunk. Before slice 2, opening that chunk charged the new
    # chunk's own size against `committed` (`536 870 912 already committed
    # for chunk 0` + `270 000 000` for chunk 1 = `806 870 912` > the
    # `556 870 912` budget) and refused — even though the two tensors'
    # actual combined request, `540 000 000`, comfortably fits. After slice
    # 2 the only quantity compared to `--forge` is that combined request,
    # so the same program now runs to completion on `dmc jit`. `dmc run`
    # was never affected (the interpreter has no chunks to commit).
    "chunk_granularity_second_chunk_matches_used": (
        "fn main() -> i64 {\n"
        " (sum(forge.zeros[f32, [67500000]]) + sum(forge.zeros[f32, "
        "[67500000]])) as i64\n"
        "}",
        "0674 slice 2: two 270 000 000 B tensors (540 000 000 B combined) "
        "that force a second chunk under `--forge=556 870 912` — before "
        "slice 2 `dmc jit` refused (the new chunk's own size charged "
        "against `committed` overshot the budget even though the combined "
        "request did not); both backends now run to completion and agree "
        "(OK, => 0); measured peak RSS ~550 MB (re-verified at slice 3c's "
        "mmap-backed `PageChunk` rework, `/usr/bin/time -l`: 550 682 624 B "
        "— the `alloc_zeroed`-under-`Layout` version this rework replaced "
        "measured 1 084 440 576 B on the same host, an eager-zero "
        "regression this figure predates and the mmap fix removes)"),

    # #674 slice 3a (the arena-budget design's §5, slice 3a) — the four
    # placeholder runtime refusals, isolated one operation at a time so each
    # scores `both-fail` on its OWN construct rather than cascading through
    # `forge.root`'s. `dmc run` refuses located with a `budget-`tagged class-7
    # error (`interp.rs`'s new interceptors); `dmc jit` refuses `.carve()`/
    # `.reap()` exactly as before (pre-existing `jit-unknown-fn`, unaffected
    # by this slice) and `forge.root` with its RECLASSIFIED code — `jit-
    # ineligible`/`jit-unknown-fn` now, where it used to be an uncoded
    # `jit error: unknown binding \`forge\``.
    #
    # #674 slice 3b: the interpreter's budget table lands (`interp.rs`'s
    # `.carve`/`.reap`/`forge.root`). These two probes' ARGUMENTS are still
    # ill-typed (`5` is not a `Budget`) on purpose, so `dmc run` still
    # refuses them — the message changed (a well-formedness error naming the
    # bad argument type, not "not yet implemented"), the PASS/FAIL verdict
    # did not: both-fail, unchanged. `dmc jit` is untouched by 3b (its own
    # table lands in slice 3c) and still refuses both with the same
    # pre-existing `jit-unknown-fn`.
    "budget_carve_not_yet_implemented_674": (
        "fn main() -> i64 {\n"
        " let b = forge.carve(5, 4096)\n"
        " 1\n"
        "}",
        "0674 slice 3b: `forge.carve` is real now, but its first argument "
        "here is `5`, not a `Budget` — both-fail unchanged (`dmc run`: a "
        "well-formedness error naming the bad type; `dmc jit`: "
        "pre-existing `jit-unknown-fn`, slice 3c's job) — #610's silent "
        "no-op stays closed"),
    "budget_reap_not_yet_implemented_674": (
        "fn main() -> i64 {\n"
        " forge.reap(5)\n"
        " 1\n"
        "}",
        "0674 slice 3b: `forge.reap` is real now, but its argument here is "
        "`5`, not a `Budget` — both-fail unchanged (`dmc run`: a "
        "well-formedness error naming the bad type; `dmc jit`: "
        "pre-existing `jit-unknown-fn`, slice 3c's job) — #610's silent "
        "no-op stays closed"),
    # #674 slices 3b + 3c: `forge.root` is a real `Budget` value on BOTH
    # backends now — entry 0 of the interpreter's table (3b) and of the
    # JIT's (3c). 3a's located `budget-root` refusal and the JIT's
    # reclassified `jit-ineligible`/`jit-unknown-fn` placeholder are both
    # gone; this probe is an ordinary OK/OK row.
    "budget_root_is_a_budget_value_674": (
        "fn main() -> i64 {\n"
        " let b = forge.root\n"
        " 1\n"
        "}",
        "0674 slices 3b + 3c: `forge.root` evaluates to the root budget "
        "descriptor on both backends (OK, => 1) — 3a's located "
        "`budget-root` refusal and the JIT placeholder are both gone"),
    # #674 slices 3b + 3c: `forge.carve`/`forge.reap` for real, over a
    # WELL-TYPED `Budget` operand this time (unlike the two well-formedness
    # probes above) — a carve against the root and a reap of it, the budget
    # table's §3.1 transitions on BOTH backends, with no allocation into
    # the child (`in b` stays refused, `budget-alloc-in`, until slice 3d,
    # so this probe does not use it).
    "budget_carve_and_reap_succeed_674": (
        "fn main() -> i64 {\n"
        " let b = forge.carve(forge.root, 4096)\n"
        " forge.reap(b)\n"
        " 1\n"
        "}",
        "0674 slices 3b + 3c: carve then reap against the root both "
        "succeed on both backends (OK, => 1) — the budget table's §3.1 "
        "transitions, for real."),
    # #674 slices 3b + 3c: R1 (`budget-carve`) and R2 (`budget-reap`) for
    # real on BOTH backends — a carve past the root's `--forge` budget, and
    # a reap of a budget with a live child. Both-fail, for the SAME located
    # reason on each backend now.
    "budget_carve_pre_refused_674": (
        "fn main() -> i64 {\n"
        " forge.carve(forge.root, 4097)\n"
        " 1\n"
        "}",
        "0674 slice 3b: PRE — this carve exceeds free(root) under "
        "`--forge=4096` (`FORGE_FLAG`, below). `dmc run`: located "
        "`budget-carve`, for real (3a's placeholder said `not yet "
        "implemented`). `dmc jit`: the same located `budget-carve` from "
        "its own table (3c) — both-fail"),
    "budget_reap_live_child_refused_674": (
        "fn main() -> i64 {\n"
        " let p = forge.carve(forge.root, 4096)\n"
        " let c = forge.carve(p, 1024)\n"
        " forge.reap(p)\n"
        " 1\n"
        "}",
        "0674 slice 3b: R2 — `p` has a live child `c`. `dmc run`: located "
        "`budget-reap`, for real. `dmc jit`: the same located "
        "`budget-reap` (3c) — both-fail"),
    # ── #674 slice 3d: `forge.enter(b) { … }`'s five exit edges (B5) ────────
    #
    # "`forge.enter(b) { … }` pushes `b` as the entered budget and pops it on
    # EVERY exit edge — normal exit, `return`, `break`, `continue`, a `?`
    # early return." A missed edge leaves the register naming `b` for the
    # rest of the program and mis-charges every later allocation with NO
    # diagnostic, so the edge set is the gate (the arena-budget design's §5 slice
    # 3d, and §4 (a), the mechanism criterion 5 depends on).
    #
    # Each edge is one BISECTED PAIR over the same source, because there is
    # no introspection API (§4). Re-bisected after #837 (`jit: flat let of an
    # arena constructor lands the tensor once`) removed the JIT's own
    # let-bound-constructor double charge (#640) that this table used to
    # paper over with a flat `1M`: the JIT no longer needs roughly double the
    # interpreter's minimum, so the flags now sit close to what the ROOT
    # itself needs — `4 096` (`b`'s carve) `+ 16 384` (the post-block
    # allocation) `= 20 480`.
    #   * `..._outer_charged_674` — `--forge=20480` on normal/`return`/
    #     `break`/`continue` (both backends share that minimum exactly); the
    #     `?` edge alone needs `--forge=20544` — `dmc jit`'s own minimum on
    #     that source sits 64 B above `dmc run`'s unchanged 20 480, from the
    #     tuple/`Err` bookkeeping the early return's `g()?` charges inside
    #     `b` before the register pops. It answers only if the register came
    #     back. In `MUST_ANSWER`, because a leak makes it fail on BOTH
    #     backends and `both-fail` is otherwise scored green.
    #   * `..._root_below_min_674` — one byte under the ROOT's own
    #     requirement on the LOWER of the two backends' minimums, so both
    #     refuse with `forge arena exhausted` (`BUDGET_EXHAUSTION_PROBES`),
    #     which is what makes the row above a real ceiling rather than a
    #     vacuous success. `20479` on all five sources, including `?`: its
    #     own below-min half brackets `dmc run`'s 20 480 (the smaller of its
    #     two backend minimums), not `dmc jit`'s 20 544 — the two flags of
    #     that one pair are not simply adjacent, unlike the other four.
    "budget_edge_normal_outer_charged_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [64]]\n"
        "    }\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    1\n"
        "}",
        "0674 slice 3d, exit edge 1/5 (normal): the 16 KiB allocation after "
        "the block charges the OUTER budget — it cannot fit `b`'s 4 KiB "
        "extent, so it answers only if `enter` popped on the normal edge"),
    "budget_edge_normal_root_below_min_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [64]]\n"
        "    }\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    1\n"
        "}",
        "0674 slice 3d: the same source one byte under what the ROOT needs "
        "for that 16 KiB — both backends refuse, `forge arena exhausted`. "
        "The bisection that makes the row above assert a real ceiling"),
    "budget_edge_return_outer_charged_674": (
        "fn f(b: Budget) -> i64 {\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [64]]\n"
        "        return 7\n"
        "    }\n"
        "    3\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    let k = f(b)\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    k\n"
        "}",
        "0674 slice 3d, exit edge 2/5 (`return`): the `return` leaves the "
        "extent AND the function; the 16 KiB in the caller then charges the "
        "root (=> 7, so the `return` really is the one that fired)"),
    "budget_edge_return_root_below_min_674": (
        "fn f(b: Budget) -> i64 {\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [64]]\n"
        "        return 7\n"
        "    }\n"
        "    3\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    let k = f(b)\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    k\n"
        "}",
        "0674 slice 3d: the `return` edge's own below-min half — both "
        "backends refuse with `forge arena exhausted` at the caller's "
        "allocation"),
    "budget_edge_break_outer_charged_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    for i in 0..3 {\n"
        "        forge.enter(b) {\n"
        "            let u = forge.zeros[f32, [64]]\n"
        "            break\n"
        "        }\n"
        "    }\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    1\n"
        "}",
        "0674 slice 3d, exit edge 3/5 (`break`): the jump leaves the extent "
        "but not the loop's own scope — the exit is emitted down to the "
        "enclosing loop's budget depth, and the 16 KiB after the loop "
        "charges the root"),
    "budget_edge_break_root_below_min_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    for i in 0..3 {\n"
        "        forge.enter(b) {\n"
        "            let u = forge.zeros[f32, [64]]\n"
        "            break\n"
        "        }\n"
        "    }\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    1\n"
        "}",
        "0674 slice 3d: the `break` edge's own below-min half — both "
        "backends refuse with `forge arena exhausted`"),
    "budget_edge_continue_outer_charged_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    for i in 0..3 {\n"
        "        forge.enter(b) {\n"
        "            let u = forge.zeros[f32, [64]]\n"
        "            continue\n"
        "        }\n"
        "    }\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    1\n"
        "}",
        "0674 slice 3d, exit edge 4/5 (`continue`): three iterations, each "
        "entering and leaving on the back-edge — a missed exit here would "
        "also nest three pushes deep, so the 16 KiB after the loop is the "
        "assertion that the register is back at the root"),
    "budget_edge_continue_root_below_min_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    for i in 0..3 {\n"
        "        forge.enter(b) {\n"
        "            let u = forge.zeros[f32, [64]]\n"
        "            continue\n"
        "        }\n"
        "    }\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    1\n"
        "}",
        "0674 slice 3d: the `continue` edge's own below-min half — both "
        "backends refuse with `forge arena exhausted`"),
    "budget_edge_query_outer_charged_674": (
        "fn g() -> (i64, Err) {\n"
        "    (0, \"boom\")\n"
        "}\n"
        "fn f(b: Budget) -> (i64, Err) {\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [64]]\n"
        "        let y = g()?\n"
        "        let v = forge.zeros[f32, [64]]\n"
        "    }\n"
        "    (5, nil)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    let (v, e) = f(b)\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    v + 1\n"
        "}",
        "0674 slice 3d, exit edge 5/5 (`?`): the early return the design "
        "names as the trap — `Expr::ArenaBlock`'s `self.eval_block(&ab."
        "body)?` shape leaks the register on exactly this path. `=> 1` (the "
        "propagated `(default[T], e)`, not `(5, nil)`) plus a 16 KiB "
        "allocation the root has to serve"),
    "budget_edge_query_root_below_min_674": (
        "fn g() -> (i64, Err) {\n"
        "    (0, \"boom\")\n"
        "}\n"
        "fn f(b: Budget) -> (i64, Err) {\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [64]]\n"
        "        let y = g()?\n"
        "        let v = forge.zeros[f32, [64]]\n"
        "    }\n"
        "    (5, nil)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    let (v, e) = f(b)\n"
        "    let t = forge.zeros[f32, [4096]]\n"
        "    v + 1\n"
        "}",
        "0674 slice 3d: the `?` edge's own below-min half — both backends "
        "refuse with `forge arena exhausted`"),

    # ── #674 slice 3d: the arena-budget design's §4 five invariants ────────
    #
    # Gated HERE and not at 3b/3c: every one of them spells `forge.enter` or
    # `… in b`, which is the first slice at which either parses AND lowers on
    # both backends (§5 slice 3d's Gate says so in as many words; 3b and 3c
    # gate the same transitions through unit tests instead).
    #
    # Each invariant is one bisected pair, exactly as §4's table spells the
    # flip: the succeeding side is in `MUST_ANSWER`, the refused side in
    # `BUDGET_EXHAUSTION_PROBES` pinned to the tag §2.3 gives that refusal
    # (`budget-carve` for PRE, `forge arena exhausted` for PRE-A). All ten
    # run under `--forge=256K`, which §4's table names.
    "ac_inv_four_carves_674": (
        "fn main() -> i64 {\n"
        "    let b1 = forge.carve(forge.root, 64K)\n"
        "    let b2 = forge.carve(forge.root, 64K)\n"
        "    let b3 = forge.carve(forge.root, 64K)\n"
        "    let b4 = forge.carve(forge.root, 64K)\n"
        "    forge.enter(b1) {\n"
        "        let t = forge.zeros[f32, [4096]]\n"
        "    }\n"
        "    1\n"
        "}",
        "0674 §4 (INV) `B_root = sum(free + used)`: under `--forge=256K`, "
        "four 64 KiB carves fit exactly, and what `b1` then allocates inside "
        "its own extent does not come out of the root's remainder"),
    "ac_inv_fifth_carve_refused_674": (
        "fn main() -> i64 {\n"
        "    let b1 = forge.carve(forge.root, 64K)\n"
        "    let b2 = forge.carve(forge.root, 64K)\n"
        "    let b3 = forge.carve(forge.root, 64K)\n"
        "    let b4 = forge.carve(forge.root, 64K)\n"
        "    forge.enter(b1) {\n"
        "        let t = forge.zeros[f32, [4096]]\n"
        "    }\n"
        "    forge.carve(forge.root, 64K)\n"
        "    1\n"
        "}",
        "0674 §4 (INV), the refused half: a FIFTH 64 KiB carve is refused "
        "`budget-carve` on both backends — the root's capacity is the "
        "children's whole extents, spent and unspent alike"),
    "ac_loc_recarve_extent_674": (
        "fn main() -> i64 {\n"
        "    let b1 = forge.carve(forge.root, 64K)\n"
        "    let b2 = forge.carve(forge.root, 64K)\n"
        "    let b3 = forge.carve(forge.root, 64K)\n"
        "    let b4 = forge.carve(forge.root, 64K)\n"
        "    forge.enter(b4) {\n"
        "        let t = forge.zeros[f32, [4096]]\n"
        "    }\n"
        "    forge.reap(b4)\n"
        "    let b5 = forge.carve(forge.root, 64K)\n"
        "    1\n"
        "}",
        "0674 §4 (LOC) `E(r) = free + used + sum E(children)`: `b4` had "
        "spent part of its extent when it was reaped, and a carve of the "
        "WHOLE 64 KiB still succeeds — the credit is the extent, not the "
        "unspent remainder (#674 criterion 2)"),
    "ac_loc_over_extent_refused_674": (
        "fn main() -> i64 {\n"
        "    let b1 = forge.carve(forge.root, 64K)\n"
        "    let b2 = forge.carve(forge.root, 64K)\n"
        "    let b3 = forge.carve(forge.root, 64K)\n"
        "    let b4 = forge.carve(forge.root, 64K)\n"
        "    forge.enter(b4) {\n"
        "        let t = forge.zeros[f32, [4096]]\n"
        "    }\n"
        "    forge.reap(b4)\n"
        "    let b5 = forge.carve(forge.root, 65537)\n"
        "    1\n"
        "}",
        "0674 §4 (LOC), the refused half: 64 KiB + 1 after the same reap is "
        "refused `budget-carve` on both backends — the credit is EXACTLY "
        "the extent, no more"),
    "ac_pre_at_free_674": (
        "fn main() -> i64 {\n"
        "    let p = forge.carve(forge.root, 64K)\n"
        "    let c = forge.carve(p, 64K)\n"
        "    1\n"
        "}",
        "0674 §4 (PRE) `carve <= free(parent)`: a nested carve of exactly "
        "the parent's free capacity is admitted"),
    "ac_pre_refused_674": (
        "fn main() -> i64 {\n"
        "    let p = forge.carve(forge.root, 64K)\n"
        "    let c = forge.carve(p, 65537)\n"
        "    let t = forge.zeros[f32, [4]] in c\n"
        "    1\n"
        "}",
        "0674 §4 (PRE), the refused half: one byte past the parent's free "
        "capacity is refused AT THE CARVE with `budget-carve` — the `in c` "
        "allocation on the next line never runs, which is the whole point "
        "(#674 criterion 3)"),
    "ac_prea_disjoint_budgets_674": (
        "fn main() -> i64 {\n"
        "    let b1 = forge.carve(forge.root, 64K)\n"
        "    let b2 = forge.carve(forge.root, 64K)\n"
        "    forge.enter(b1) {\n"
        "        let a = forge.zeros[f32, [4096]]\n"
        "    }\n"
        "    let c = forge.zeros[f32, [4096]] in b2\n"
        "    1\n"
        "}",
        "0674 §4 (PRE-A) `allocate n <= free(r)`: two disjoint budgets each "
        "serve a 16 KiB allocation from their OWN ceiling — one through "
        "`forge.enter`, one through `… in b` (#674 criterion 1, m6's "
        "failure)"),
    "ac_prea_own_ceiling_refused_674": (
        "fn main() -> i64 {\n"
        "    let b1 = forge.carve(forge.root, 64K)\n"
        "    let b2 = forge.carve(forge.root, 64K)\n"
        "    let c = forge.zeros[f32, [4096]] in b2\n"
        "    forge.enter(b1) {\n"
        "        let a = forge.zeros[f32, [65536]]\n"
        "    }\n"
        "    1\n"
        "}",
        "0674 §4 (PRE-A), the refused half: `b2` has already served, and "
        "`b1` then refuses 256 KiB at ITS OWN 64 KiB ceiling — not at the "
        "root's 256 KiB, which still has capacity"),
    "ac_nn_recarve_cycles_674": (
        "fn main() -> i64 {\n"
        "    for i in 0..64 {\n"
        "        let c = forge.carve(forge.root, 128K)\n"
        "        forge.reap(c)\n"
        "    }\n"
        "    let d = forge.carve(forge.root, 256K)\n"
        "    1\n"
        "}",
        "0674 §4 (NN) `free, used >= 0`: 64 reap-then-recarve cycles leave "
        "the root's capacity exactly where it started — cycle 65 carves the "
        "WHOLE 256 KiB"),
    "ac_nn_over_root_refused_674": (
        "fn main() -> i64 {\n"
        "    for i in 0..64 {\n"
        "        let c = forge.carve(forge.root, 128K)\n"
        "        forge.reap(c)\n"
        "    }\n"
        "    let d = forge.carve(forge.root, 262145)\n"
        "    1\n"
        "}",
        "0674 §4 (NN), the refused half: one byte over is still refused "
        "after 64 cycles. An unsigned underflow in the reap credit would "
        "show as an enormous free capacity and this row would answer"),

    # s3/s4/s5 (from the arena-budgets probe set), re-run
    # verbatim for their POST-3d outcome. Their history in one line each:
    # before 3a, s3 errored (`undefined identifier` on the bare `in`), s4 ran
    # its block unconditionally (#610's defect, printing `s4 inner 0` then
    # `s4 ran`) and s5 was a parse error; 3a made all three refuse at
    # `forge.root`; 3b and 3c moved the refusal point later, to `in b` /
    # `forge.enter` themselves.
    #
    # SLICE 3d closes them. `forge.enter(b) { … }` and `<ctor> in b` lower on
    # BOTH backends now, so s3 and s4 leave `both-fail` for the ordinary
    # ok/ok set — and go into `MUST_ANSWER`, since a probe whose whole
    # assertion is "this construct runs on both backends" is scored green by
    # the plain `both-fail` verdict it would regress to. Neither carries a
    # `=> value` (both return `nil`), so what they pin here is that the
    # program completes on both backends; the printed lines themselves are
    # `make diff-backends`'s comparison, not this battery's.
    #
    # s5 is the exception, and NOT for a budget reason: `for x in <ctor> in b`
    # lowers its `in b` fine and then meets the JIT's pre-existing
    # for-over-a-collection refusal (only range iteration is lowered). It is
    # a tracked `jit-gap` — see its `GAP_ALLOWLIST` entry.
    "design_probe_s3_alloc_in_674": (
        "fn main() -> nil {\n"
        "    let b = forge.carve(forge.root, 4M)\n"
        "    let t = forge.zeros[f32, [4]] in b\n"
        "    print(\"s3 ran \" + to_str(t[0]))\n"
        "    nil\n"
        "}",
        "0674 slice 3d, probe s3: `<ctor> in b` LOWERS now — both backends "
        "run to completion and print `s3 ran 0`. The per-constructor budget "
        "override charges `b` on both (`Expr::AllocIn` holds the register "
        "across the constructor); 3a's `budget-alloc-in` refusal and the "
        "JIT's `jit-construct` one are both gone"),
    "design_probe_s4_budget_block_674": (
        "fn main() -> nil {\n"
        "    let b = forge.carve(forge.root, 4M)\n"
        "    forge.enter(b) {\n"
        "        let t = forge.zeros[f32, [4]]\n"
        "        print(\"s4 inner \" + to_str(t[0]))\n"
        "    }\n"
        "    print(\"s4 ran\")\n"
        "    nil\n"
        "}",
        "0674 slice 3d, probe s4: `forge.enter(b) { … }` LOWERS now — both "
        "backends print `s4 inner 0` then `s4 ran`. It prints the same two "
        "lines it printed BEFORE slice 3a, and means something different: "
        "then the block ran because the `forge.enter(b)` call was parsed "
        "away and the block was an unconditional one (#610); now it runs as "
        "an extent that charges `b`, and pops on exit (the five edge probes "
        "above are what pin the popping)"),
    # #674 slice 3d, the arena-budget design's §3.5 (and §6 Q3): a VAULT-spelled
    # constructor with no `in b`, evaluated INSIDE `forge.enter(b) { … }`,
    # charges the ROOT — the spelling says Vault and no budget was named.
    # `dmc run` has always done it (Vault is a flat `vault_used` meter a
    # budget never owns); `dmc jit` lowers both spellings into the one arena
    # (`MEMORY.md §1.1`), so the distinction has to be made at the emit site,
    # and until this slice it was not made at all: the buffer went into `b`'s
    # storage and `forge.reap(b)` freed memory the program still held.
    #
    # Three rows, because the defect had two observable faces:
    #   * `..._survives_reap_674` — the USE-AFTER-FREE. The vault tensor
    #     outlives `reap(b)`, a 64 KiB `forge.zeros` reuses the storage, and
    #     the program then reads the tensor. `=> 2` on both backends; before
    #     the fix `dmc jit` died with SIGSEGV (exit 139) and, under
    #     `DMC_BUDGET_POISON=1`, faulted at the access site. In `MUST_ANSWER`
    #     because a crash on both backends would otherwise score `both-fail`
    #     and stay green.
    #   * `..._charges_root_674` / `..._charges_b_674` — the METERING, as a
    #     minimal pair that differs in EXACTLY ONE TOKEN, `vault` vs `forge`.
    #     32 KiB of constructors inside a 4 KiB `b`: the `vault.*` spelling
    #     runs on both backends (the bytes went to the root, sized at this
    #     pair's own bisected minimum of `4 096 + 2 × 16 384 = 36 864` —
    #     comfortably more than `b`'s own 4 KiB extent), the `forge.*`
    #     spelling refuses on both with the same located `forge arena
    #     exhausted` naming `b`'s 4 KiB extent regardless of the root's own
    #     size. The refusing row is what makes the answering row a real
    #     assertion rather than a vacuous success — the usual below-min
    #     bracket cannot serve here, because on `dmc run` these bytes never
    #     touch `--forge` at all.
    "budget_vault_ctor_in_enter_survives_reap_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 65536)\n"
        "    let w = forge.enter(b) { vault.ones[f32, [4]] }\n"
        "    forge.reap(b)\n"
        "    let pad = forge.zeros[f32, [16384]]\n"
        "    ((w[0] + w[3]) as i64)\n"
        "}",
        "0674 slice 3d, the arena-budget design's §3.5: a `vault.*` constructor "
        "inside `forge.enter(b)` charges the ROOT, so it SURVIVES "
        "`forge.reap(b)` and still reads 1 after a 64 KiB forge allocation "
        "has taken the reaped storage (=> 2). `dmc jit` used to place it in "
        "`b` and segfault here"),
    "budget_vault_ctor_in_enter_charges_root_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    forge.enter(b) {\n"
        "        vault.zeros[f32, [4096]]\n"
        "        vault.zeros[f32, [4096]]\n"
        "    }\n"
        "    1\n"
        "}",
        "0674 slice 3d, §3.5: 32 KiB of `vault.*` constructors inside a "
        "4 KiB entered budget run on BOTH backends — the bytes are the "
        "root's, not `b`'s. One token from the refusing row below"),
    "budget_forge_ctor_in_enter_charges_b_674": (
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 4096)\n"
        "    forge.enter(b) {\n"
        "        forge.zeros[f32, [4096]]\n"
        "        forge.zeros[f32, [4096]]\n"
        "    }\n"
        "    1\n"
        "}",
        "0674 slice 3d, §3.5: the same program with `forge` for `vault` — "
        "refused on BOTH backends, `forge arena exhausted` naming the "
        "entered budget's 4 KiB extent. The control that makes the row "
        "above an assertion about the spelling"),
    "design_probe_s5_for_ctor_in_budget_674": (
        "fn main() -> nil {\n"
        "    let b = forge.carve(forge.root, 4M)\n"
        "    for x in forge.zeros[f32, [4]] in b {\n"
        "        print(\"s5 \" + to_str(x))\n"
        "    }\n"
        "    nil\n"
        "}",
        "0674 slice 3d, probe s5: `for x in <ctor> in b { }` parses as one "
        "reading (no flag, §6 Q2) and now RUNS on `dmc run`, printing four "
        "`s5 0` lines — the loop's iterable is a constructor charged to `b`. "
        "`dmc jit` lowers the `in b` and then meets its own pre-existing "
        "for-over-a-collection refusal, so this is a tracked jit-gap and "
        "not a budget one (GAP_ALLOWLIST)"),
    # s6, z1, z2 — unchanged/changed exactly as the Gate specifies.
    "design_probe_s6_budget_name_reserved_674": (
        "model Budget { n: i64 }\n"
        "fn main() -> nil {\n"
        "    let b = Budget { n: 3 }\n"
        "    print(\"s6 \" + to_str(b.n))\n"
        "    nil\n"
        "}",
        "0674 slice 3a, probe s6 post-3a: `model Budget { … }` is now a "
        "reserved-name type error on both backends (`Gate`: must refuse "
        "where it used to print `s6 3`) — both-fail"),
    "design_probe_z1_in_still_an_identifier_674": (
        "fn main() -> nil {\n"
        "    let in = 5\n"
        "    print(\"z1 \" + to_str(in))\n"
        "    nil\n"
        "}",
        "0674 slice 3a, probe z1: `in` stays a usable identifier (Must NOT "
        "change) — both backends print `z1 5` and agree (OK)"),
    "design_probe_z2_in_arm_no_skip_newlines_674": (
        "fn main() -> nil {\n"
        "    let in = 7\n"
        "    let t = forge.zeros[f32, [4]]\n"
        "    in\n"
        "    print(\"z2 \" + to_str(in))\n"
        "    nil\n"
        "}",
        "0674 slice 3a, probe z2: the new `in` postfix arm must not "
        "`skip_newlines()` (Must NOT change) — a bare `in` on its own "
        "line, right after a constructor statement with no `in b` on it, "
        "stays two statements, not `forge.zeros[...] in in`; both "
        "backends print `z2 7` and agree (OK)"),

    # ── #674 slice 4: `examples/tenant_budgets.dmc`, section by section ─────
    #
    # The acceptance program (the arena-budget design's §5 "Slice 4") is an EXAMPLE,
    # so its nine sections already run on both backends, unflagged, under
    # `make conformance` and `make test-dmc` — `dmc test` and `dmc test --jit`
    # over `examples/`. What an example cannot carry is the refused half of a
    # boundary: a refusal aborts the program instead of returning `false`, and
    # an example that aborts is a corpus regression, not an assertion.
    #
    # These nine are that half. Each is one section transcribed as a `fn main`
    # and run one byte under the LOWER of the two backends' bisected `--forge`
    # minimums, so both refuse, and each is pinned in
    # `BUDGET_EXHAUSTION_PROBES` to `budget-carve` — the tag §2.3 gives a
    # carve past the parent's free capacity — because `both-fail` on its own
    # is also what "the file stopped parsing" scores.
    #
    # Three sections' two backends do not agree on the minimum, and the reason
    # is the same in all three: `dmc jit` materializes a `Tenant` model and a
    # propagated `(i64, Err)` tuple in Forge where the interpreter builds them
    # outside any arena, so the JIT wants exactly one 64 B aligned slot more
    # per such value. Each is 64 or 128 B, each is named in its probe's
    # description, and none of them moves the EXAMPLE's own figures, which are
    # carve extents rather than root requirements and agree exactly. The
    # at-minimum half of every one of these nine pairs — the assertion that
    # the section still FITS in the bisected figure, on each backend — is in
    # `compiler/tests/budgets_acceptance.rs`, which brackets both sides on
    # both backends; this battery carries the differential half.
    "tenant_budgets_inv_674": (
        "fn spend_all(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [16384]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let p = forge.carve(forge.root, 262144)\n"
        "    let b1 = forge.carve(p, 65536)\n"
        "    let b2 = forge.carve(p, 65536)\n"
        "    let b3 = forge.carve(p, 65536)\n"
        "    let b4 = forge.carve(p, 65536)\n"
        "    let s = spend_all(b1) + spend_all(b2) + spend_all(b3)\n"
        "    forge.reap(b1)\n"
        "    forge.reap(b2)\n"
        "    forge.reap(b3)\n"
        "    forge.reap(b4)\n"
        "    forge.reap(p)\n"
        "    s as i64\n"
        "}",
        "0674 slice 4, (INV): the (INV) section one byte under the root's "
        "own requirement — the 262 144 B carve that holds the four tenant "
        "extents is refused `budget-carve` on both backends"),
    "tenant_budgets_loc_674": (
        "fn spend_all(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [16384]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn spend_quarter(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [4096]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let p = forge.carve(forge.root, 65536)\n"
        "    let c = forge.carve(p, 65536)\n"
        "    let s = spend_quarter(c)\n"
        "    forge.reap(c)\n"
        "    let d = forge.carve(p, 65536)\n"
        "    let u = spend_all(d)\n"
        "    forge.reap(d)\n"
        "    forge.reap(p)\n"
        "    (s + u) as i64\n"
        "}",
        "0674 slice 4, (LOC): the (LOC) section one byte under — the "
        "parent whose child is reaped and re-carved whole cannot itself "
        "be carved, `budget-carve` on both backends"),
    "tenant_budgets_pre_674": (
        "fn spend_half(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [8192]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let p = forge.carve(forge.root, 65536)\n"
        "    let c1 = forge.carve(p, 32768)\n"
        "    let c2 = forge.carve(p, 32768)\n"
        "    let s = spend_half(c1) + spend_half(c2)\n"
        "    forge.reap(c1)\n"
        "    forge.reap(c2)\n"
        "    forge.reap(p)\n"
        "    s as i64\n"
        "}",
        "0674 slice 4, (PRE): the (PRE) section one byte under — the "
        "parent the two nested carves split cannot itself be carved, "
        "`budget-carve` on both backends"),
    "tenant_budgets_prea_674": (
        "model Tenant { budget: Budget, id: i64 }\n"
        "fn carve_tenant(parent: Budget, id: i64, bytes: i64) -> Tenant {\n"
        "    Tenant { budget: forge.carve(parent, bytes), id: id }\n"
        "}\n"
        "fn drop_tenant(r: Tenant) -> nil {\n"
        "    forge.reap(r.budget)\n"
        "    nil\n"
        "}\n"
        "fn spend_all(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [16384]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn tick() -> f32 {\n"
        "    let !t = forge.zeros[f32, [4096]]\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn step_tenant(r: Tenant) -> f32 {\n"
        "    let !s: f32 = 0.0\n"
        "    forge.enter(r.budget) {\n"
        "        s = tick() + tick() + tick() + tick()\n"
        "    }\n"
        "    s\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let a = carve_tenant(forge.root, 1, 65536)\n"
        "    let b = carve_tenant(forge.root, 2, 65536)\n"
        "    let sa = step_tenant(a)\n"
        "    let sb = spend_all(b.budget)\n"
        "    drop_tenant(a)\n"
        "    drop_tenant(b)\n"
        "    (sa + sb) as i64\n"
        "}",
        "0674 slice 4, (PRE-A): the (PRE-A) section one byte under `dmc "
        "run`'s own minimum, the LOWER of the two (the JIT wants 131 200: "
        "it materializes each `Tenant` model in Forge and the interpreter "
        "does not) — the second tenant's carve is refused `budget-carve` "
        "on both backends"),
    "tenant_budgets_nn_674": (
        "fn spend_all(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [16384]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let p = forge.carve(forge.root, 131072)\n"
        "    for i in 0..64 {\n"
        "        let c = forge.carve(p, 131072)\n"
        "        forge.reap(c)\n"
        "    }\n"
        "    let d = forge.carve(p, 131072)\n"
        "    let s = spend_all(d)\n"
        "    forge.reap(d)\n"
        "    forge.reap(p)\n"
        "    s as i64\n"
        "}",
        "0674 slice 4, (NN): the (NN) section one byte under — the parent "
        "the 64 reap/re-carve cycles run inside is refused `budget-carve` "
        "on both backends"),
    "tenant_budgets_reap_order_674": (
        "fn spend_all(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [16384]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let host = forge.carve(forge.root, 131072)\n"
        "    let p = forge.carve(host, 131072)\n"
        "    let c1 = forge.carve(p, 65536)\n"
        "    let c2 = forge.carve(p, 65536)\n"
        "    let s = spend_all(c1) + spend_all(c2)\n"
        "    forge.reap(c1)\n"
        "    forge.reap(c2)\n"
        "    forge.reap(p)\n"
        "    let again = forge.carve(host, 131072)\n"
        "    forge.reap(again)\n"
        "    forge.reap(host)\n"
        "    s as i64\n"
        "}",
        "0674 slice 4, criterion 4: the reap-order section one byte under "
        "— the budget that holds the parent with two live children is "
        "refused `budget-carve` on both backends"),
    "tenant_budgets_exit_edges_674": (
        "fn tick() -> f32 {\n"
        "    let !t = forge.zeros[f32, [4096]]\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn leave_normally(b: Budget) -> nil {\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [16]]\n"
        "    }\n"
        "    nil\n"
        "}\n"
        "fn leave_by_return(b: Budget) -> i64 {\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [16]]\n"
        "        return 7\n"
        "    }\n"
        "    3\n"
        "}\n"
        "fn leave_by_break(b: Budget) -> i64 {\n"
        "    let !n = 0\n"
        "    for i in 0..3 {\n"
        "        n = n + 1\n"
        "        forge.enter(b) {\n"
        "            let u = forge.zeros[f32, [16]]\n"
        "            break\n"
        "        }\n"
        "        n = n + 10\n"
        "    }\n"
        "    n\n"
        "}\n"
        "fn leave_by_continue(b: Budget) -> i64 {\n"
        "    let !n = 0\n"
        "    for i in 0..3 {\n"
        "        n = n + 1\n"
        "        forge.enter(b) {\n"
        "            let u = forge.zeros[f32, [16]]\n"
        "            continue\n"
        "        }\n"
        "        n = n + 10\n"
        "    }\n"
        "    n\n"
        "}\n"
        "fn fail_with(msg: Err) -> (i64, Err) {\n"
        "    (0, msg)\n"
        "}\n"
        "fn leave_by_query(b: Budget, msg: Err) -> (i64, Err) {\n"
        "    forge.enter(b) {\n"
        "        let u = forge.zeros[f32, [16]]\n"
        "        let y = fail_with(msg)?\n"
        "        let v = forge.zeros[f32, [16]]\n"
        "    }\n"
        "    (5, nil)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let msg = \"the query edge\"\n"
        "    let b = forge.carve(forge.root, 512)\n"
        "    let o = forge.carve(forge.root, 65536)\n"
        "    let q = forge.carve(forge.root, 16448)\n"
        "    let !acc: f32 = 0.0\n"
        "    let !k = 0\n"
        "    let !nb = 0\n"
        "    let !nc = 0\n"
        "    let !qv = 0\n"
        "    forge.enter(o) {\n"
        "        leave_normally(b)\n"
        "        acc = acc + tick()\n"
        "        k = leave_by_return(b)\n"
        "        acc = acc + tick()\n"
        "        nb = leave_by_break(b)\n"
        "        acc = acc + tick()\n"
        "        nc = leave_by_continue(b)\n"
        "        acc = acc + tick()\n"
        "    }\n"
        "    forge.enter(q) {\n"
        "        let (v, e) = leave_by_query(b, msg)\n"
        "        qv = v\n"
        "        acc = acc + tick()\n"
        "    }\n"
        "    forge.reap(q)\n"
        "    forge.reap(o)\n"
        "    forge.reap(b)\n"
        "    (acc as i64) + k + nb + nc + qv\n"
        "}",
        "0674 slice 4, (a): the five-exit-edge section one byte under "
        "`dmc run`'s own minimum, the LOWER of the two (the JIT wants 82 "
        "560, one 64 B tuple more for the `?` edge) — the third of the "
        "three budgets the edges need is refused `budget-carve` on both "
        "backends"),
    "tenant_budgets_field_reap_674": (
        "model Tenant { budget: Budget, id: i64 }\n"
        "fn carve_tenant(parent: Budget, id: i64, bytes: i64) -> Tenant {\n"
        "    Tenant { budget: forge.carve(parent, bytes), id: id }\n"
        "}\n"
        "fn drop_tenant(r: Tenant) -> nil {\n"
        "    forge.reap(r.budget)\n"
        "    nil\n"
        "}\n"
        "fn spend_all(b: Budget) -> f32 {\n"
        "    let !t = forge.zeros[f32, [16384]] in b\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let p = forge.carve(forge.root, 65536)\n"
        "    let r = carve_tenant(p, 3, 65536)\n"
        "    let s = spend_all(r.budget)\n"
        "    drop_tenant(r)\n"
        "    let again = forge.carve(p, 65536)\n"
        "    forge.reap(again)\n"
        "    forge.reap(p)\n"
        "    (s as i64) + r.id\n"
        "}",
        "0674 slice 4, (b): the model-field section one byte under `dmc "
        "run`'s own minimum, the LOWER of the two (the JIT wants 65 600, "
        "one 64 B `Tenant` more) — the parent the tenant is carved from is "
        "refused `budget-carve` on both backends"),
    "tenant_budgets_ambient_call_674": (
        "fn hop(n: i64) -> f32 {\n"
        "    deeper(n)\n"
        "}\n"
        "fn deeper(n: i64) -> f32 {\n"
        "    let !t = forge.zeros[f32, [8192]]\n"
        "    t[0] = 1.0\n"
        "    sum(t)\n"
        "}\n"
        "fn main() -> i64 {\n"
        "    let b = forge.carve(forge.root, 65536)\n"
        "    let !s: f32 = 0.0\n"
        "    forge.enter(b) {\n"
        "        s = s + hop(1)\n"
        "        s = s + hop(2)\n"
        "    }\n"
        "    forge.reap(b)\n"
        "    s as i64\n"
        "}",
        "0674 slice 4, (c): the ambient-call section one byte under — the "
        "budget the two calls allocate into is refused `budget-carve` on "
        "both backends"),

    # g1 (from the arena-budgets probe set) — #614's coordinator
    # scope note on PR #779: the monotonic charge is for the LANGUAGE-LEVEL
    # `forge.restore(mark)` only. `grad_call_epilogue`'s rewind after every
    # `fwd_bwd` is a compiler-inserted cleanup over intermediates the user
    # never asked to keep (`MEMORY.md §1.1`: implicit intermediates are not
    # charged), so it returns its own charge and a `@grad` step loop's budget
    # is unaffected by #614. Unlike m3/m8/x6/x8 above, this is an ordinary
    # agreement probe (expected verdict `OK`, both backends answering 199 —
    # the last step served) at the same `--forge=4M` this loop ran under
    # before #614 landed; PR #779 itself regressed it to needing 32 MiB by
    # charging the epilogue's rewind, which is the bug this probe pins shut.
    "forge_restore_no_budget_g1": (
        "@grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {\n"
        " let h = x @ W\n"
        " sum(h .* h)\n"
        "}\n"
        "fn main() -> i64 {\n"
        " vault {\n"
        "  let !W = vault.zeros[f32, [64, 64]]\n"
        "  let x = vault.zeros[f32, [64, 64]]\n"
        "  let !last = 0\n"
        "  for step in 0..200 {\n"
        "   let (l, g) = loss.fwd_bwd(W, x)\n"
        "   last = step\n"
        "  }\n"
        "  last\n"
        " }\n"
        "}",
        "0674 g1: a @grad step loop, 200 iterations of `fwd_bwd` — the "
        "compiler-inserted epilogue rewind must return its own charge, so "
        "all 200 steps run under `--forge=4M` on both backends (OK, => 199) "
        "— #614 does not touch this budget (PR #779 scope note). #791: "
        "`dmc run` now charges this loop too (the tape evaluator's "
        "activations and adjoints, released at the same AD boundary), and "
        "still fits the unchanged `4M`; its own minimum is 3 358 720 B "
        "against `dmc jit`'s 3 473 408 B — bracketed by "
        "`meter_ad_temporaries_fwd_bwd_below_min` below"),

    # g1's forward-only twin — PR #779's second reviewer-found defect. A
    # plain call `loss(W, x)` (no `.fwd_bwd`) is a forward-only entry of the
    # same `@grad fn`, which gets its own compiler-inserted rewind at exit
    # (`jit.rs`'s `fwd_only_scope` block) for the same reason the AD
    # epilogue does: its activations are intermediates the user never asked
    # to keep. Before that fix this rewind used the user-facing
    # `forge.snapshot`/`forge.restore` pair instead of the `InternalMark`
    # one, so it never returned its charge and 200 calls needed 9664 KiB
    # minimum under `dmc jit` (measured) against `dmc run`'s 1 KiB — a
    # divergence this probe would have caught as `both-fail` at `--forge=4M`
    # (the JIT alone refusing) had it existed then. Fixed, both backends
    # agreed (OK, => 199) at that flag; the JIT's own fixed minimum measures
    # 3296 KiB (3 375 104 B) and is UNCHANGED by everything below — none of
    # it touches `jit.rs`.
    #
    # #781 first pass: a forward-only call runs its body through the
    # ordinary expression path (unlike `.fwd_bwd`'s separate tape
    # evaluator), so `h = x @ W` and `h .* h` are each charged as an
    # implicit intermediate (`Expr::BinOp`) — 64×64 f32, 16 384 B apiece,
    # both real per step. `dmc run`'s minimum rose from 1 KiB to 200 × (2 ×
    # 16 384) = 6 553 600 B (not the 6 400 000 B an earlier pass of this
    # comment computed — 200 × 32 KiB is 6 553 600, since 32 KiB is 32 768
    # B, not 32 000), past the old `4M` ceiling this probe shared with g1;
    # raised to `8M` there.
    #
    # PR #790 reviewer rework: `!W` is a `!` param, and #650 (SPEC
    # §3.4) makes a `!` param a PRIVATE COPY — the JIT already charged that
    # copy-in; `dmc run` did not. Now it does (`call_fn_in_module`'s
    # param-binding loop). An intermediate pass of that rework charged the
    # copy-OUT as well and released neither, which took this probe to `16M`;
    # both halves of that were wrong, and #791 fixes them:
    #
    #   - the copy-out is a memcpy into the buffer the caller already owns
    #     (`emit_mut_param_copy_out`), not a second allocation, so it is not
    #     charged at all — the copy-IN is the one that survives, and it is
    #     monotonic on BOTH backends because the JIT allocates it at the
    #     call site, outside the callee's own rewind;
    #   - the body's activations are not monotonic on either backend any
    #     more: `call_fn_in_module` now takes the same forward-only mark
    #     `fwd_only_scope` does, gated the same way (`@grad` fn, non-nil
    #     scalar return), and hands `forge_used` back at the return.
    #
    # Both backends therefore charge 16 384 B per call, and the flag comes
    # back to the `4M` this probe shared with g1 before #781 touched it.
    # `dmc run`'s minimum is 3 309 568 B (200 copies + one call's two live
    # activations) against `dmc jit`'s 3 342 336 B (#640: 3 375 104 B while
    # the flat `let` of a constructor snapshotted it); the 32 768 B residual
    # is `main`'s two `vault.zeros`, which the JIT meters against `--forge`
    # (one arena) while `dmc run` meters them against `--vault` — once each
    # now, not twice. Bracketed by `meter_ad_temporaries_fwd_below_min`.
    "forge_restore_no_budget_g1_fwd": (
        "@grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {\n"
        " let h = x @ W\n"
        " sum(h .* h)\n"
        "}\n"
        "fn main() -> i64 {\n"
        " vault {\n"
        "  let !W = vault.zeros[f32, [64, 64]]\n"
        "  let x = vault.zeros[f32, [64, 64]]\n"
        "  let !last = 0\n"
        "  for step in 0..200 {\n"
        "   let l = loss(W, x)\n"
        "   last = step\n"
        "  }\n"
        "  last\n"
        " }\n"
        "}",
        "0674 g1-fwd: 200 forward-only calls of g1's loss fn (no `.fwd_bwd`) "
        "— the forward-only entry's own compiler-inserted rewind must return "
        "its charge too, on BOTH backends (#791), so all 200 calls run under "
        "`--forge=4M` (OK, => 199); `dmc run`'s own minimum is 3 309 568 B, "
        "`dmc jit`'s unchanged 3 375 104 B — see this probe's header comment "
        "for the 65 536 B residual"),

    # #791: the two rows below bracket the interpreter's OWN minimum on the
    # g1 pair from underneath — one byte short of it, so `dmc run` must
    # refuse. Both backends refuse there: `dmc jit` needs MORE on each
    # source (3 375 104 B forward-only, 3 473 408 B through `fwd_bwd`), so a
    # budget under the interpreter's minimum is under the JIT's by more
    # still. That asymmetry is the reason these budgets are pinned to the
    # interpreter's number rather than the JIT's — one byte under the JIT's
    # would leave `dmc run` succeeding and score a divergence.
    #
    # They exist because the OK-both probes above cannot see an
    # under-charge: a backend that charged nothing at all would pass
    # `--forge=4M` just as happily. #791's whole defect was of that shape —
    # `.fwd_bwd`'s tape evaluator ran the entire 200-step loop at
    # `--forge=1` — so the ceiling probes alone did not catch it. Pinned to
    # the `forge arena exhausted` failure mode via BUDGET_EXHAUSTION_PROBES.
    "meter_ad_temporaries_fwd_below_min": (
        "@grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {\n"
        " let h = x @ W\n"
        " sum(h .* h)\n"
        "}\n"
        "fn main() -> i64 {\n"
        " vault {\n"
        "  let !W = vault.zeros[f32, [64, 64]]\n"
        "  let x = vault.zeros[f32, [64, 64]]\n"
        "  let !last = 0\n"
        "  for step in 0..200 {\n"
        "   let l = loss(W, x)\n"
        "   last = step\n"
        "  }\n"
        "  last\n"
        " }\n"
        "}",
        "#791, one byte under `dmc run`'s 3 309 568 B minimum on g1-fwd "
        "(200 `!W` copy-ins at 16 384 B each, plus one call's two live "
        "activations): both backends refuse (both-fail). Pins the "
        "forward-only entry's charge from below — before #781 this loop ran "
        "at `--forge=1`"),
    # PR #790 reviewer rework, defect 1: a `!` tensor param written before a
    # `?` early return. The `?` desugars to a `return (default[T], e)`
    # (SPEC §4.9), which reaches `call_fn_in_module` as an `Err` carrying the
    # propagation marker — so an intermediate pass of this PR, which had
    # gated the writeback capture on `result.is_ok()` to keep it out of the
    # forward-only `@grad` rewind's way, dropped the writebacks on exactly
    # that path: `dmc run` printed the caller's untouched 0 where `main` and
    # `dmc jit` both printed 7. The JIT emits `emit_mut_param_copy_out` on
    # every path out of the callee and never had the bug, so this is an
    # ordinary agreement probe (OK, => 7) — and a MUST_LOWER name, so a JIT
    # that stopped answering here is a red battery rather than a warning.
    #
    # The metering half of that same gate was the part worth keeping, and it
    # kept it: only the meter rewind is conditioned on `is_ok()` now.
    "mut_param_writeback_survives_query_early_return": (
        "fn boom() -> (i64, Err) {\n"
        " (0, \"boom\")\n"
        "}\n"
        "fn fill(!t: Tensor[f32, [4]]) -> (i64, Err) {\n"
        " t[0] = 7.0\n"
        " let k = boom()?\n"
        " (k, nil)\n"
        "}\n"
        "fn main() -> i64 {\n"
        " let !x = forge.zeros[f32, [4]]\n"
        " let (r, e) = fill(x)\n"
        " x[0] as i64\n"
        "}",
        "PR #790 defect 1: a `!` tensor param written before a `?` early "
        "return writes back on BOTH backends (OK, => 7) — the `?` arrives as "
        "an `Err`, and the writeback capture must not be gated on success"),

    "meter_ad_temporaries_fwd_bwd_below_min": (
        "@grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {\n"
        " let h = x @ W\n"
        " sum(h .* h)\n"
        "}\n"
        "fn main() -> i64 {\n"
        " vault {\n"
        "  let !W = vault.zeros[f32, [64, 64]]\n"
        "  let x = vault.zeros[f32, [64, 64]]\n"
        "  let !last = 0\n"
        "  for step in 0..200 {\n"
        "   let (l, g) = loss.fwd_bwd(W, x)\n"
        "   last = step\n"
        "  }\n"
        "  last\n"
        " }\n"
        "}",
        "#791, one byte under `dmc run`'s 3 358 720 B minimum on g1 (200 "
        "published gradients at 16 384 B each, plus one step's peak): both "
        "backends refuse (both-fail). This is the probe #791 itself was "
        "filed from — the tape evaluator used to materialize every "
        "activation and adjoint uncharged, so the loop ran at `--forge=1`"),

    # PR #790 reviewer rework, defect 4: the pair above brackets each g1
    # source from below at the INTERPRETER's minimum, which is the lower of
    # the two on both — so nothing in this battery bracketed the JIT's own
    # g1-fwd minimum from below, and a JIT that started charging LESS there
    # would have gone unnoticed. This row sits one byte under it
    # (3 375 103 B), in the window between the two backends' minimums:
    # `dmc run` answers (=> 199, its own minimum being 3 309 568 B) and
    # `dmc jit` refuses with `forge arena exhausted`.
    #
    # That is a real, EXPECTED divergence, and it is the arena asymmetry
    # `MEMORY.md §1.1` states outright — the JIT lowers `vault.*` and
    # `forge.*` into one arena, so `main`'s two `vault.zeros` are charged
    # against `--forge` there and against `--vault` here: 2 × 16 384 =
    # 32 768 B, exactly the gap between the two minimums. (It was 65 536 B
    # until #640, whose let-bound-constructor over-charge doubled each of
    # the two — the flag below was re-bisected with it.) It is tracked in
    # GAP_ALLOWLIST (a `jit-gap (tracked failure)`, not a divergence) rather
    # than silently tolerated, so the day either backend's number moves this
    # row goes red.
    "meter_ad_temporaries_fwd_below_jit_min": (
        "@grad fn loss(!W: Tensor[f32, [64, 64]], x: Tensor[f32, [64, 64]]) -> f32 {\n"
        " let h = x @ W\n"
        " sum(h .* h)\n"
        "}\n"
        "fn main() -> i64 {\n"
        " vault {\n"
        "  let !W = vault.zeros[f32, [64, 64]]\n"
        "  let x = vault.zeros[f32, [64, 64]]\n"
        "  let !last = 0\n"
        "  for step in 0..200 {\n"
        "   let l = loss(W, x)\n"
        "   last = step\n"
        "  }\n"
        "  last\n"
        " }\n"
        "}",
        "PR #790 defect 4: one byte under `dmc jit`'s 3 342 336 B minimum on "
        "g1-fwd (#640: 3 375 104 B before the fix) and above `dmc run`'s "
        "unmoved 3 309 568 B — run answers (=> 199), "
        "jit refuses. Brackets the JIT's minimum from below and records the "
        "one-arena asymmetry (#640, `MEMORY.md §1.1`) as a tracked gap"),

    # #781 (the arena-budget design's §7 slice 1): the
    # interpreter now meters an implicit intermediate (`let C = A @ B`) the
    # same as an explicit `forge.*`/`vault.*` constructor — at its declared
    # element width, charged where the tensor is materialized — closing the
    # divergence #781 filed (p1_imp/p2_loop from
    # the arena-budgets probe set's c2 pair, minimum `--forge` bisected on
    # each backend independently). The two backends' minimums do NOT
    # coincide — `dmc jit` carried #640's separate 2× `let`-bound-constructor
    # over-charge, so its minimum (81 929, measured; 81 984 as flagged, its
    # own 64-byte chunk alignment) sat a whole `a`/`b` pair above the
    # interpreter's correctly-metered one (49 152), and rather than bracket
    # one shared number each pair below bracketed its OWN backend's minimum.
    # #640 is FIXED: a flat `let` of an arena constructor no longer snapshots
    # the fresh tensor, the JIT's minimum here is now the interpreter's own
    # 49 152 B, and both pairs bracket one shared number again — `_below_min`
    # one byte under it (both backends refuse), `_at_jit_min` AT it (both
    # succeed). The keys keep their `_at_jit_min` names so the battery's name
    # set does not churn; the name is historical, the number is shared.
    "meter_intermediates_p1_imp_below_min": (
        "fn main() -> i64 {\n"
        " let a = forge.zeros[f32, [64, 64]]\n"
        " let b = forge.zeros[f32, [64, 64]]\n"
        " let c = a @ b\n"
        " c[0, 0] as i64\n"
        "}",
        "#781 p1_imp, one byte under the interpreter's 49 152 B minimum: "
        "both backends refuse (both-fail) — before #781 `dmc run` charged "
        "the intermediate `c` nothing and ran at this budget"),
    "meter_intermediates_p1_imp_at_jit_min": (
        "fn main() -> i64 {\n"
        " let a = forge.zeros[f32, [64, 64]]\n"
        " let b = forge.zeros[f32, [64, 64]]\n"
        " let c = a @ b\n"
        " c[0, 0] as i64\n"
        "}",
        "#781 p1_imp, at the 49 152 B minimum both backends now share "
        "(#640 fixed: `dmc jit` needed 81 984 B while it snapshotted each "
        "`let`-bound constructor): both backends succeed and agree (OK, => 0)"),
    "meter_intermediates_p2_loop_below_min": (
        "fn main() -> i64 {\n"
        " let a = forge.zeros[f32, [64, 64]]\n"
        " let b = forge.zeros[f32, [64, 64]]\n"
        " let !s = 0.0\n"
        " for i in 0..2000 {\n"
        "  let c = a @ b\n"
        "  s = s + c[0, 0]\n"
        " }\n"
        " s as i64\n"
        "}",
        "#781 p2_loop (2000 matmul intermediates), one byte under the "
        "interpreter's 32 800 768 B minimum: both backends refuse "
        "(both-fail)"),
    "meter_intermediates_p2_loop_at_jit_min": (
        "fn main() -> i64 {\n"
        " let a = forge.zeros[f32, [64, 64]]\n"
        " let b = forge.zeros[f32, [64, 64]]\n"
        " let !s = 0.0\n"
        " for i in 0..2000 {\n"
        "  let c = a @ b\n"
        "  s = s + c[0, 0]\n"
        " }\n"
        " s as i64\n"
        "}",
        "#781 p2_loop, at the 32 800 768 B minimum both backends now share "
        "(#640 fixed: `dmc jit` needed 32 833 600 B while it snapshotted "
        "each `let`-bound constructor): both succeed and agree (OK, => 0)"),

    # #781 reviewer rework (PR #790): two verified defects in the ORIGINAL
    # #781 charging, both in the `Expr::TensorLit` arm. (1) a nested literal
    # recursed through this same arm for each row, so a row's own charge
    # fired and the assembled outer array's charge fired again on top —
    # double-billing every nested literal. (2) a bracket list read as a
    # `.reshape` SHAPE ARGUMENT went through the same arm and was charged
    # even though it is consumed for its integers and never returned to the
    # program. Fixed by suppressing the charge while evaluating a row or a
    # shape argument, so only the outermost literal's own charge survives.
    "nested_literal_charged_once_below_min": (
        "fn main() -> i64 {\n"
        " let t = [[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"
        "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"
        "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"
        "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]\n"
        " t[0, 0] as i64\n"
        "}",
        "PR #790 defect 1, one byte under the correct 256 B minimum (4 rows "
        "of 16 f32): both backends refuse (both-fail) at the SAME message — "
        "before the fix `dmc run` charged this literal twice (512 B, one "
        "row-recursion charge stacked on the outer array's) and diverged "
        "from `dmc jit` here"),
    "nested_literal_charged_once_at_min": (
        "fn main() -> i64 {\n"
        " let t = [[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"
        "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"
        "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],"
        "[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]\n"
        " t[0, 0] as i64\n"
        "}",
        "PR #790 defect 1, at the correct 256 B minimum (not the old, "
        "over-charged 512 B): both backends succeed and agree (OK, => 1)"),
    "reshape_shape_arg_not_charged_below_min": (
        "fn main() -> i64 {\n"
        " let a = forge.zeros[f32, [4096]]\n"
        " let b = a.reshape[[4096]]\n"
        " b[0] as i64\n"
        "}",
        "PR #790 defect 2, one byte under the interpreter's correct "
        "32 768 B minimum (16 384 for `a` + 16 384 for reshape's real "
        "output `b` — NOT the shape-argument list `[4096]` itself, which "
        "is consumed for its one integer and never returned to the "
        "program): both backends refuse (both-fail) — `dmc jit` needed more "
        "here while #640's 2× let-bound-constructor over-charge inflated "
        "`a` alone past this budget, so this budget was already both-fail "
        "before the #790 fix; what #790 changed is `dmc run`'s OWN minimum, "
        "from 32 776 B (the shape argument's spurious 8 B on top) to the "
        "correct 32 768 B, which #640's fix has since made the shared one"),
    "reshape_shape_arg_not_charged_at_jit_min": (
        "fn main() -> i64 {\n"
        " let a = forge.zeros[f32, [4096]]\n"
        " let b = a.reshape[[4096]]\n"
        " b[0] as i64\n"
        "}",
        "PR #790 defect 2, at the 32 768 B minimum both backends now share "
        "(#640 fixed: `dmc jit` needed 49 152 B while it snapshotted the "
        "`let`-bound `a`): both backends succeed and agree (OK, => 0)"),

    # #630, closed by #642, in its arena spelling. Slice 1 is what makes the
    # arena block reach the scoping machinery at all; before it the JIT refused
    # here on `jit-construct`, so this is a MUST_LOWER probe like the ten above.
    "arena_block_shadowing_let": (
        "fn main()->i64{ let a = 1\n let b = forge { let a = 2\n a + 10 }\n a + b }",
        "#630: the inner `let a` dies with the block — 12 + 1, not 12 + 2 → 13"),

    # #639, closed by #648 — `return` / `break` / `continue` crossing a block
    # boundary. Ordinary agreement probes: the JIT always answered these
    # numbers and the interpreter now does too. They were deliberately kept out
    # of this battery while they diverged, because a run/jit disagreement scores
    # as a `divergence` and exits non-zero, which would have turned a tracked
    # interpreter bug into a red gate on every branch until #648 landed.
    #
    # Not in MUST_LOWER, unlike the arena probes above: MUST_LOWER is the set
    # that says "the JIT refused this construct before slice 1", and control
    # flow out of a block was never the refused half. Three of the four do
    # contain a `forge { }`, so a slice-1 regression would demote them to
    # informational `jit-gap`s rather than failures — the ten MUST_LOWER names
    # above already cover that regression, and duplicating it here would say
    # these probes are about the arena lowering when they are about #639.
    "block_return": (
        "fn f()->i64{ forge { return 42 }\n 7 }\nfn main()->i64{ f() }",
        "#639: `return` inside a block returns from the FUNCTION → 42, not 7"),
    "block_break": (
        "fn main()->i64{ let !s = 0\n for i in 0..5 { forge { if i == 3 { break } }\n s = s + i }\n s }",
        "#639: `break` inside a block reaches the enclosing loop → 3, not 10"),
    "block_continue": (
        "fn main()->i64{ let !s = 0\n for i in 0..6 { forge { if i == 2 { continue } }\n s = s + i }\n s }",
        "#639: `continue` inside a block reaches the enclosing loop → 13, not 15"),
    "plain_block_return": (
        "fn g()->i64{ { return 9 }\n 4 }\nfn main()->i64{ g() }",
        "#639 with no arena keyword — the proof it was an interpreter bug → 9, not 4"),
    # top-level module constant as a shape dimension (jit-error defect, not a
    # gap): a non-generic fn whose Tensor shape uses a `let NAME = <literal>`
    # module constant used to fail the JIT with "unbound shape parameter",
    # while `--check`/`run` accepted it (the checker's SymDim treats any bare
    # shape-position ident as an opaque symbolic var, never distinguishing a
    # module constant from a real shape parameter). `resolve_shape_expr` in
    # jit.rs fixes this: in a type-level shape position, and only for a name
    # nothing in scope binds, it falls back to the folded module-constant
    # table (`ConstDim`).
    "shape_const_dim_nongeneric": (
        "let CHANNELS: u64 = 3u64\n"
        "fn retain_default() -> Tensor[f32, [CHANNELS]] {\n"
        " let !t = forge.zeros[f32, [CHANNELS]]\n"
        " t[0] = 1.0\n t[1] = 2.0\n t[2] = 3.0\n t\n}\n"
        "fn main()->i64{ sum(retain_default()) as i64 }",
        "a module constant used as a Tensor shape dim in a non-generic fn → 6"),

    # The other half of that fix: a module constant is a dimension only in a
    # TYPE-LEVEL shape position, and only when nothing in scope binds the name.
    # A draft that seeded every fn's shape_env with the constants instead folded
    # `K` to the module constant inside a runtime slice bound and inside scalar
    # arithmetic — silently, on programs that had always worked. Both forms are
    # run-vs-jit divergences, which is exactly what this battery is for.
    "shape_const_shadowed_by_local": (
        "let K = 2\n"
        "fn main()->i64{ let K = 5\n"
        " let !t = forge.zeros[f32,[8]]\n"
        " for i in 0..8 { t[i] = (i as f32) * 10.0 }\n"
        " let s = t[K..K+1]\n"
        " sum(s) as i64 }",
        "a local shadowing a module constant wins in a slice bound → 50"),
    "shape_const_shadowed_by_loop_var": (
        "let K = 2\n"
        "fn main()->i64{ let !t = forge.zeros[f32,[8]]\n"
        " for i in 0..8 { t[i] = (i as f32) * 10.0 }\n"
        " let !acc: f32 = 0.0\n"
        " for K in 4..8 { let s = t[K..K+1]\n  acc = acc + sum(s) }\n"
        " acc as i64 }",
        "a loop var shadowing a module constant wins in a slice bound (#511) → 220"),
    "narrow_width_const_scalar": (
        "let W: i32 = 7i32\n"
        "fn main()->i64{ (W * 2i32) as i64 }",
        "a narrow-width module constant keeps its declared width as a scalar (#268) → 14"),

    # #712: a LOCAL literal `let` used as a Tensor shape dimension — the
    # fold path #620 gave a MODULE constant, extended to the current fn's own
    # `let`s. Was the bare, uncoded "unbound shape parameter" jit-error (the
    # #480 defect class) on a program `--check`/`run` both accept.
    "shape_local_let_dim_712": (
        "fn main()->i64{ let H = 32\n"
        " let !t = forge.zeros[i64,[H]]\n"
        " t[0] = 1\n"
        " sum(t) }",
        "a local literal `let` used as a Tensor shape dim (#712) → 1"),
    # The arithmetic form: `let W = H * 2`, folding one already-registered
    # local `let` through another — not only a bare literal.
    "shape_local_let_dim_arithmetic_712": (
        "fn main()->i64{ let H = 32\n let W = H * 2\n"
        " let !t = forge.zeros[i64,[W]]\n"
        " t[0] = 1\n"
        " sum(t) }",
        "a local `let` folded from arithmetic on another local `let` (#712) → 1"),
    # The shadowing counterpart: a local `let` must fold to ITS OWN value, not
    # the module constant it shadows — same shadowing discipline #620 already
    # enforces for a runtime local (`shape_const_shadowed_by_local`, above),
    # now covering the fold path too.
    # #737: folded dims share `int_arith`'s rules — `/ 0` and `% 0` are 0 on
    # both backends (SPEC.md §3.1), no per-arm guard; `%` folds like `/`.
    "shape_dim_div_by_zero_folds_to_zero_737": (
        "fn main()->i64{ let H = 4 / 0\n"
        " let !t = forge.zeros[i64,[H]]\n"
        " len(t) }",
        "a folded dim of `4 / 0` is 0 on both backends (#737) → 0 (`len`, "
        "not `sum`: the sum of an empty tensor prints `-0` on `dmc run`, #725)"),
    "shape_dim_mod_by_zero_folds_to_zero_737": (
        "fn main()->i64{ let H = 7 % 0\n"
        " let !t = forge.zeros[i64,[H]]\n"
        " len(t) }",
        "a folded dim of `7 % 0` is 0 on both backends (#737; the JIT used to "
        "refuse `%` as a dynamic shape) → 0"),
    "shape_local_let_shadows_module_const_712": (
        "let N = 10\n"
        "fn main()->i64{ let N = 5\n"
        " let !t = forge.zeros[i64,[N]]\n"
        " t[0] = 1\n"
        " sum(t) }",
        "a local `let` shadowing a module constant folds to its OWN value, 5 not 10 (#712) → 1"),

    # A lambda body is lexically inside its enclosing fn, so that fn's
    # monomorphized shape parameters are in scope there. The deferred lambda
    # carried an EMPTY shape_env, so `N` was unbound inside it and the module
    # constant answered in place of the fn's own binding — the fallback
    # standing in for a binding that exists. Both positions diverged:
    # a dimension (11 for 5) and a scalar (24 for 6).
    "shape_param_in_lambda_body_dim": (
        "let N = 8\n"
        "fn outer[N](x: Tensor[f32,[N]]) -> f32 {\n"
        " let g = fn(y: f32) -> f32 { let t = forge.ones[f32,[N]]\n  y + sum(t) }\n"
        " g(sum(x)) }\n"
        "fn main()->i64{ let !t = forge.zeros[f32,[2]]\n"
        " t[0]=1.0\n t[1]=2.0\n outer(t) as i64 }",
        "an enclosing fn's shape param is the dimension inside its lambda, not the constant → 5"),
    "shape_param_in_lambda_body_scalar": (
        "let N = 8\n"
        "fn outer[N](x: Tensor[f32,[N]]) -> f32 {\n"
        " let g = fn(y: f32) -> f32 { y * (N as f32) }\n"
        " g(sum(x)) }\n"
        "fn main()->i64{ let !t = forge.zeros[f32,[2]]\n"
        " t[0]=1.0\n t[1]=2.0\n outer(t) as i64 }",
        "the same, in a scalar position inside the lambda body → 6"),

    # #718 rework (R5): a #712-foldable LOCAL `let` (not a shape param)
    # CAPTURED by a lambda and used as a shape dimension inside its body.
    # `dmc run` closes over the binding normally (SPEC.md §3.5.1: "Capturing
    # closures are supported in both `dmc run` and `dmc jit`"), but the JIT's
    # `local_const_dims` table is per-fn and a deferred lambda body is lowered
    # with none of the enclosing fn's entries visible, so the capture reads as
    # an ordinary runtime value — `jit-dynamic-shape`, named and classed, never
    # a silent divergence. Tracked here rather than promised as SPEC-conformant
    # until the JIT closes over a foldable local's *verdict*, not only its
    # runtime value, the way it already does for an enclosing fn's shape
    # parameter (`shape_param_in_lambda_body_dim`, just above).
    "shape_local_let_capture_in_lambda_body_718": (
        "fn main()->i64{ let H = 4\n"
        " let f = fn()->i64 { let !t = forge.zeros[i64,[H]]\n"
        "  t[0] = 1\n  sum(t) }\n"
        " f() }",
        "a foldable local `let` captured by a lambda, sized as its shape dim (#718) → 1"),

    # A negative extent is a verdict on the folded DIMENSION, not on a leaf: a
    # `let Z = 0` term is fine when the dimension it sits in folds positive,
    # and a zero-length tensor is legal on both backends anyway (#258), so a
    # constant that folds to 0 is a dimension like the literal `[0]`.
    "shape_const_zero_leaf_folds_positive": (
        "let Z = 0\n"
        "let A = 5\n"
        "fn main()->i64{ let !t = forge.zeros[f32,[A + Z]]\n"
        " for i in 0..5 { t[i] = 1.0 }\n"
        " sum(t) as i64 }",
        "a zero module constant is a fine term of a dimension that folds to 5 → 5"),
    "shape_const_zero_dim_is_legal": (
        "let Z = 0\n"
        "fn main()->i64{ let m = mean(forge.zeros[f32,[Z]])\n"
        " if m == m { 0 } else { 1 } }",
        "a constant that folds to 0 is a legal zero-length extent, mean is NaN (#258) → 1"),

    # The roadmap's JIT parity push: a top-level `type` alias (SPEC.md §6.6)
    # used to refuse the whole file outright (`jit-item`, "`type` aliases at
    # top level") before a single fn body was even looked at. Now the
    # declaration is transparent and every JIT type-building site resolves
    # through it — one probe per site the fix touches.
    #
    # Deliberately in-range values (not a width-wrapping case like `u8` 255+1).
    # Issue #626 is the reason: through an alias, `dmc run` loses the target's
    # narrow width, so a wrapping case would score here as a divergence that
    # belongs to the interpreter, not to the alias resolution these probes are
    # for. In-range values isolate "does the alias resolve at all" from that.
    #
    # No CLI probe for `forge.zeros[Alias, ..]`: the checker blocks that form
    # today — `scalar_type_from_ident` in check.rs (the dtype-argument matcher
    # for `forge.zeros`/`.ones`/`.uninit`) is its own hardcoded name match with
    # no alias awareness, so `forge.zeros[Count, [3]]` is rejected as
    # "undefined identifier `Count`" before either backend's alias handling is
    # reached. Pre-existing and out of scope here; the JIT side of that
    # position (`lower_arena_alloc` in `compiler/src/jit.rs`) is covered
    # directly, bypassing the checker, by a `#[test]` in `jit.rs`
    # (`jit_type_alias_forge_zeros_dtype`).
    "type_alias_scalar_signature": (
        "type Role = u8\n"
        "fn bump(r: Role) -> Role { r + 1 }\n"
        "fn main()->i64{ bump(10) as i64 }",
        "an alias in a fn param/return resolves through instead of erroring as an unknown model → 11"),
    "type_alias_model_field": (
        "type Idx = i32\n"
        "model Acct { v: Idx }\n"
        "fn main()->i64{ let a = Acct{v: 42}\n a.v as i64 }",
        "an alias as a model field type lays out as its target, not an unknown model → 42"),
    "type_alias_cast": (
        "type Byte = u8\n"
        "fn main()->i64{ let x: i64 = 12\n (x as Byte) as i64 }",
        "an alias as a cast target resolves to its target scalar type → 12"),
    "type_alias_let_annotation": (
        "type Byte = u8\n"
        "fn main()->i64{ let b: Byte = 12\n b as i64 }",
        "an alias as a `let` annotation coerces to its target, not `jit-conversion` → 12"),
    "type_alias_nested_tuple_return": (
        "type Byte = u8\n"
        "fn pair() -> (Byte, i64) { (3, 4) }\n"
        "fn main()->i64{ let (a, b) = pair()\n (a as i64) + b }",
        "an alias nested in a tuple return resolves like `(u8, i64)` → 7"),
    "type_alias_nested_tensor_element": (
        "type E = f32\n"
        "fn head(x: Tensor[E, [3]]) -> f32 { x[0] }\n"
        "fn main()->i64{ let !t = forge.zeros[f32, [3]]\n t[0] = 7.0f32\n head(t) as i64 }",
        "an alias as a tensor element type resolves like `Tensor[f32,[3]]` → 7"),
    # N1 review finding: `unify_param_type` (template instantiation's own
    # unifier, separate from `ty_from_ast`) read the raw written `Type`, so
    # this same alias-as-tensor-element form went through `jit-element-type`
    # ("a non-scalar element type") when the param was a TEMPLATE — even
    # though the sibling non-template probe above, and the un-aliased
    # template (`Tensor[f32,[N]]`), both lower.
    "type_alias_template_tensor_element": (
        "type E = f32\n"
        "fn head[N](x: Tensor[E, [N]]) -> f32 { x[0] }\n"
        "fn main()->i64{ let !t = forge.zeros[f32, [3]]\n t[0] = 7.0f32\n head(t) as i64 }",
        "N1: an alias in a TEMPLATE tensor-param element type resolves like `Tensor[f32,[N]]` → 7"),
    # ── `@cast(Model)` overlay (#598) ───────────────────────────────────────
    # The overlay materializes a model's fields from a byte tensor under
    # `dmc run`. The JIT does not lower it at all — the directive is a no-op
    # there and the field read that follows is refused as an access on a
    # non-model type. Nothing said so until this probe: `--check` accepts the
    # program, `dmc run` answers, and `dmc jit` refused with nobody counting
    # it. Now the battery reports it as a tracked gap.
    "cast_model_overlay": (
        "model M { a: i64 }\n"
        "fn main()->i64{\n let !buf=forge.zeros[i64,[8]]\n buf[7]=9\n"
        " let m=@cast(M) { buf }\n m.a\n}",
        "#598 `@cast(Model) { bytes }` decodes under `dmc run` (jit gap)"),

    # #635, the model-methods design's slice 0 — the whole-field scalar store,
    # split out of the assignment catch-all. Both are `q2`/`q3` from the
    # design's own probe set.
    "field_store_scalar_local": (
        "model B { !n: i64 }\n"
        "fn main() -> i64 { let !b = B { n: 1 }\n b.n = 7\n b.n }",
        "q2: a whole-field store on a local model binding → 7"),
    "field_store_scalar_via_mut_param": (
        "model C { !n: i64 }\n"
        "fn bump(!c: C) -> i64 { c.n = c.n + 1  c.n }\n"
        "fn main() -> i64 { let !c = C { n: 0 }\n bump(c)\n c.n }",
        "q3: the same store through a `!` model parameter, visible to the caller → 1, 1"),
    # The #683 follow-up: a whole-field store of a NON-generic MODEL value
    # (`c.mode = m`), not a scalar. Was refused `jit-assign-form` citing #474
    # (closed, unrelated — the ghost-method bug); now lowers as a raw i64
    # pointer store, the same representation `lower_struct_lit`'s own
    # field-init and the field-read arm already use for a model-typed field.
    "field_store_nested_model_local": (
        "model D { w: i64\n h: i64 }\n"
        "model C { !mode: D }\n"
        "fn main() -> i64 { let !c = C { mode: D { w: 1, h: 1 } }\n"
        " let !m = D { w: 800, h: 600 }\n c.mode = m\n c.mode.w + c.mode.h }",
        "a whole-field store of a non-generic model VALUE on a local → 1400"),

    # #634, the model-methods design's slice 1 — model methods, lowered as
    # `M$$method` free functions. One probe per row named in the slice's
    # gate; the design's own probe set carries copies of each.
    #
    # Each row there makes TWO observations — the method's return value and the
    # caller's view of the mutation afterwards — and prints the first. The
    # battery reads only the first `=> value` a program emits, so the two are
    # folded into one scalar here (`a * 10 + m.n`): a lowering that got the
    # return right and the write-back wrong would pass on either observation
    # alone and fails on the pair. The design's files stay the two-print form.
    "method_read_scalar_field": (
        "model M { !n: i64\n fn get(self) -> i64 { self.n } }\n"
        "fn main() -> i64 { let !m = M { n: 7 }\n m.get() }",
        "p1: a method reads a scalar field off `self` → 7"),
    "method_mutate_scalar_field": (
        "model M { !n: i64\n fn bump!(!self) -> i64 { self.n = self.n + 1  self.n } }\n"
        "fn main() -> i64 { let !m = M { n: 7 }\n let a = m.bump!()\n a * 10 + m.n }",
        "p2: `!self` mutates a scalar field and the caller sees it → 88"),
    "method_mutate_tensor_element": (
        "model M { !t: Tensor[i64, [3]]\n"
        " fn poke!(!self) -> i64 { self.t[0] = 5  self.t[0] } }\n"
        "fn main() -> i64 { let !m = M { t: forge.zeros[i64, [3]] }\n"
        " let a = m.poke!()\n a * 10 + m.t[0] }",
        "p3: `!self` writes a tensor-field element, visible to the caller → 55"),
    "method_call_through_self": (
        "model M { !n: i64\n"
        " fn inner!(!self) -> i64 { self.n = self.n + 1  self.n }\n"
        " fn outer!(!self) -> i64 { self.inner!() } }\n"
        "fn main() -> i64 { let !m = M { n: 1 }\n let a = m.outer!()\n a * 10 + m.n }",
        "p6: a method calls another method through `self` → 22"),
    "method_mut_tensor_arg_bare_ident": (
        "model M { !n: i64\n"
        " fn fill!(self, !t: Tensor[i64, [3]]) -> i64 { t[0] = 3  t[0] } }\n"
        "fn main() -> i64 { let !m = M { n: 0 }\n let !out = forge.zeros[i64, [3]]\n"
        " let a = m.fill!(out)\n a * 10 + out[0] }",
        "p7: a `!` tensor argument written through a method writes back (#475) → 33"),
    "method_call_via_mut_param": (
        "model Graph { !n: i64\n fn step!(!self) -> nil { self.n = self.n + 1  nil } }\n"
        "fn op(!g: Graph) -> nil { g.step!()  nil }\n"
        "fn main() -> i64 { let !g = Graph { n: 0 }\n op(g)\n g.n }",
        "p8: the corpus's shape — a method called on a `!` model parameter → 1"),
    "method_call_nested_field_receiver": (
        "model Inner { !n: i64\n fn step!(!self) -> nil { self.n = self.n + 1  nil } }\n"
        "model Outer { !b: Inner }\n"
        "fn main() -> i64 { let !a = Outer { b: Inner { n: 0 } }\n a.b.step!()\n a.b.n }",
        "p9: the receiver is another model's field — `model_of_expr`'s nested arm → 1"),
    "method_precedence_over_free_fn": (
        "model M { !n: i64\n fn tag(self) -> i64 { 1 } }\n"
        "fn tag(m: M) -> i64 { 2 }\n"
        "fn main() -> i64 { let !m = M { n: 0 }\n m.tag() * 10 + tag(m) }",
        "p10: SPEC §4.12 — the method wins, and the free fn is still callable → 12"),
    "method_recursion": (
        "model M { !n: i64\n"
        " fn down!(!self, k: i64) -> i64 {"
        " if k == 0 { self.n } else { self.n = self.n + 1  self.down!(k - 1) } } }\n"
        "fn main() -> i64 { let !m = M { n: 0 }\n m.down!(5) }",
        "p12: a method recurses through `self` — the decl must precede its body → 5"),
    "method_aliased_mut_params": (
        "model C { !n: i64\n fn bump!(!self, k: i64) -> i64 { self.n = self.n + k  self.n } }\n"
        "fn two(!a: C, !b: C) -> i64 { a.bump!(1)  b.bump!(10) }\n"
        "fn main() -> i64 { let !c = C { n: 0 }\n two(c, c) }",
        "p17: one instance reached through two `!` names is one instance → 11"),

    # #634, the model-methods design's slice 2 — methods on SHAPE-GENERIC
    # models, one symbol per distinct receiver `type_args` tuple. The design's
    # own probe set carries copies as `p5`, `p16` and `q13`; those print two values, which
    # this battery cannot read (it takes the first `=> value` only), so each is
    # folded into one scalar here.
    "method_generic_shape_param": (
        "model Buf[N] { !t: Tensor[i64, [N]]\n"
        " fn cap(self) -> i64 { N * 2 } }\n"
        "fn main() -> i64 { let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n b.cap() }",
        "p5: a method reads the model's shape param, bound from the instance → 8"),
    "method_generic_two_instantiations": (
        "model Buf[N] { !t: Tensor[i64, [N]]\n"
        " fn cap(self) -> i64 { N * 2 } }\n"
        "fn main() -> i64 { let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n"
        " let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n"
        " a.cap() * 100 + b.cap() }",
        "p16: two instantiations of one model must answer 8 and 16, not 8 and 8 → 816"),
    "method_generic_bracketed_call": (
        "model Buf[N] { !vals: Tensor[i64, [N]]\n"
        " fn fillg![K](self, !out: Tensor[i64, [4]]) -> i64 { out[0] = 7  out[0] } }\n"
        "fn main() -> i64 { let !b = Buf[4] { vals: forge.zeros[i64, [4]] }\n"
        " let !o = forge.zeros[i64, [4]]\n"
        " let r = b.fillg![2](o)\n r * 10 + o[0] }",
        "q13: a method's own bracket at the call, with its `!` write-back "
        "(#671, landed as #679) → 77"),

    # #708, `SPEC.md §3.10` — `instance(x, ...)` desugars to
    # `instance.forward(x, ...)`. The interpreter already does this (`dmc run`
    # answers `21`); the JIT's callee match saw a bare `Ident` and fell through
    # to `jit-unknown-fn: unknown function \`l\``, never reaching
    # `lower_model_method_call` at all. `compiler/tests/jit_model_methods.rs`
    # carries the shape-generic, `!`-param and `!self` variants; this one
    # scalar is the issue's own probe.
    "method_forward_desugar_call": (
        "model Lin { fn forward(self, x: i64) -> i64 { x * 3 } }\n"
        "fn main() -> i64 { let l = Lin {}\n l(7) }",
        "#708: `l(7)` desugars to `l.forward(7)` → 21"),

    # The two spellings slice 2 refuses. Both are in GAP_ALLOWLIST above, with
    # the reason each is a deliberate line rather than an unclosed gap.
    "method_generic_type_arg_less_receiver": (
        "model Buf[N] { !t: Tensor[i64, [N]]\n"
        " fn cap(self) -> i64 { N * 2 } }\n"
        "fn f(b: Buf) -> i64 { b.cap() }\n"
        "fn main() -> i64 { let !b = Buf[4] { t: forge.zeros[i64, [4]] }\n f(b) }",
        "#634 slice 2: a method call through a type-arg-less receiver (jit gap) → 8"),
    "method_own_bracket_not_spelled": (
        "model M { !n: i64\n"
        " fn blit![SH](self, t: Tensor[i64, [SH]]) -> i64 { SH } }\n"
        "fn main() -> i64 { let !m = M { n: 1 }\n let t = forge.zeros[i64, [4]]\n"
        " m.blit!(t) }",
        "#634 slice 2: a method's own bracket, inferable by `dmc run` but not "
        "spelled at the call (jit gap) → 4"),

    # #706 — the exact program from the issue: a tensor field of a
    # shape-generic model, read through a type-arg-less parameter. `--check`
    # accepts it and `dmc run` answers 7; before this fix `dmc jit` hit an
    # uncoded `jit error: indexing requires a tensor or str, got i64` (#480
    # class) — now a located `jit-type-form` refusal, the read-side twin of
    # #683's `jit-assign-form` store guard.
    "generic_field_read_type_arg_less_receiver": (
        "model M[N] {\n"
        " t: Tensor[i64,[N]]\n"
        "}\n"
        "fn first(m: M) -> i64 {\n"
        " m.t[0]\n"
        "}\n"
        "fn main() -> i64 {\n"
        " let !m = M[4]{ t: forge.zeros[i64,[4]] }\n"
        " m.t[0] = 7\n"
        " first(m)\n"
        "}",
        "#706: a shape-generic model's tensor field read through a "
        "type-arg-less receiver (jit gap) → 7"),

    # The generic-model-params design's slice 0 — a template's shape parameter
    # bound from a MODEL argument's own shape arguments (`SPEC.md §3.10`, "an
    # argument's own shape arguments bind the parameter's shape names"). All
    # three were `jit error: could not infer shape parameter `N`` before the
    # slice, on programs `dmc run` has always answered, so all three are
    # MUST_LOWER: a refusal coming back is a red battery, not a gap line.
    "generic_model_param_inferred_bracket": (
        "model Buf[N] { !t: Tensor[i64, [N]]\n n: i64 }\n"
        "fn cap[N](b: Buf[N]) -> i64 { b.t[0] + b.n + N }\n"
        "fn main() -> i64 { let !a = Buf[4] { t: forge.zeros[i64, [4]], n: 3 }\n"
        " let !c = Buf[8] { t: forge.zeros[i64, [8]], n: 9 }\n"
        " a.t[0] = 5\n c.t[0] = 7\n"
        " cap(a) * 100 + cap(c) }",
        "g15: `N` inferred from the model argument, two instantiations — "
        "1224, never 1212 (one body would serve both)"),
    "generic_model_param_from_literal": (
        "model Buf[N] { !t: Tensor[i64, [N]] }\n"
        "fn cap[N](b: Buf[N]) -> i64 { b.t[0] + N }\n"
        "fn main() -> i64 { cap(Buf[4] { t: forge.zeros[i64, [4]] }) }",
        "g23: the argument is the model literal itself, not a binding → 4"),
    "generic_model_param_single_call": (
        "model Buf[N] { !t: Tensor[i64, [N]] }\n"
        "fn cap[N](b: Buf[N]) -> i64 { b.t[0] + N }\n"
        "fn main() -> i64 { let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n"
        " cap(a) }",
        "g24: the minimal inference — one call, one instantiation → 4"),

    # The two DIVERGENCE pins slice 0 closes. Neither is in MUST_LOWER or
    # GAP_ALLOWLIST, and that is the point: both backends REFUSE these, which
    # this battery scores `both-fail`, while a regression to either old
    # behaviour — the JIT answering where `dmc run` refuses — falls into the
    # catch-all below and is scored a DIVERGENCE, i.e. a failure. That is the
    # only gate shape that holds #732 and #733 closed.
    "generic_model_param_two_args_disagree_732": (
        "model Buf[N] { !t: Tensor[i64, [N]] }\n"
        "fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n"
        "fn main() -> i64 { let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n"
        " let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n"
        " a.t[3] = 5\n b.t[3] = 100\n"
        " f[4](a, b) }",
        "#732: `f[4]` on a `Buf[4]` and a `Buf[8]` — both backends refuse the "
        "contradiction; the JIT used to answer 105, reading `y.t[3]` at N=4"),
    "lowercase_tensor_shape_param_not_inferred_733": (
        "fn head[n](x: Tensor[i64, [n]]) -> i64 { x[0] }\n"
        "fn main() -> i64 { let !a = forge.zeros[i64, [4]]\n a[0] = 5\n head(a) }",
        "#733 (`SPEC.md §3.10`: inference binds uppercase-initial names only) — "
        "both backends refuse; the JIT used to infer `n` and answer 5"),
    "lowercase_model_shape_param_not_inferred_g30": (
        "model Buf[N] { !t: Tensor[i64, [N]] }\n"
        "fn cap[n](b: Buf[n]) -> i64 { b.t[0] }\n"
        "fn main() -> i64 { let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n"
        " a.t[0] = 5\n cap(a) }",
        "g30: #733's clause on the MODEL arm — refused on both before slice 0 "
        "only because the arm never fired, and still refused now that it does"),
    # #810 — a BARE model parameter (`fn g(a: Buf, b: Buf)`) forwarded to an
    # explicitly bracketed call (`f[8](a, b)`). The forwarding erases `a`/`b`'s
    # own shape (`TyKind::Model("Buf", [])` inside `g`), so `unify_param_type`'s
    # model arm had nothing to bind or check `N` against — but the bracket had
    # already seeded `bindings["N"] = 8` before the arm ran, so the "every
    # shape_param must end up bound" sweep never fired either. Nothing objected,
    # and the JIT read `a.t[7]` off a 4-element tensor, answering `100` where
    # `dmc run`'s runtime contradiction check ("shape param `N` would need to
    # be both 8 and 4") refuses. Both backends refuse now (both-fail): the
    # model arm refuses as `jit-type-form` the moment it finds `N` already
    # bound with no argument shape of its own to check.
    "bare_model_arg_against_bracketed_callee_810": (
        "model Buf[N] { !t: Tensor[i64, [N]] }\n"
        "fn f[N](x: Buf[N], y: Buf[N]) -> i64 { x.t[N - 1] + y.t[N - 1] }\n"
        "fn g(a: Buf, b: Buf) -> i64 { f[8](a, b) }\n"
        "fn main() -> i64 { let !a = Buf[4] { t: forge.zeros[i64, [4]] }\n"
        " let !b = Buf[8] { t: forge.zeros[i64, [8]] }\n"
        " a.t[3] = 5\n b.t[7] = 100\n"
        " g(a, b) }",
        "#810: `f[8](a, b)` with bare `a: Buf`/`b: Buf` — both backends "
        "refuse; the JIT used to answer 100, reading `a.t[7]` out of bounds "
        "on a 4-element tensor"),

    # #810 traded-away class (reviewer pass, "q3"): a bare model argument
    # forwarded to a callee whose OWN parameter for it carries a shape
    # argument (`x: Buf[M]`) — but the callee's BODY never actually reads
    # `x`'s shape at all (only `t: Tensor[f32,[M]]` does). Before #810 this
    # lowered and agreed with the interpreter (`=> 1.5`, `mean(t)` on
    # `[1.0, 2.0]`); `bind_or_refuse_model_shape_args` is a purely TYPE-level
    # check, with no notion of whether the callee's body ever dereferences
    # the argument it's refusing — a bare model value carries no runtime
    # shape for the JIT to check even when nothing would ever read it, so
    # the same unconditional refusal that catches the real #810 OOB case
    # also catches this always-sound one. Acceptable (see #810's own doc:
    # "a model value at run time is only the i64 pointer to its field arena,
    # no header, so there is no runtime shape to fall back on checking
    # instead"), but a real, tracked loss — allowlisted in GAP_ALLOWLIST
    # rather than left silent, so the gap stays visible.
    "bare_model_arg_shape_unused_by_callee_body_810": (
        "model Buf[N] { !t: Tensor[i64, [N]] }\n"
        "fn f[M](x: Buf[M], t: Tensor[f32, [M]]) -> f32 { mean(t) }\n"
        "fn g(x: Buf, t: Tensor[f32, [2]]) -> f32 { f[2](x, t) }\n"
        "fn main() -> f32 { let !x = Buf[2] { t: forge.zeros[i64, [2]] }\n"
        " let !t = forge.zeros[f32, [2]]\n"
        " t[0] = 1.0\n t[1] = 2.0\n"
        " g(x, t) }",
        "#810 q3: `x: Buf` bare, forwarded to `f[M](x: Buf[M], t: Tensor[f32,[M]])`, "
        "whose body only reads `t` — `dmc run` and the pre-#810 JIT both answered "
        "1.5; #810's unconditional refusal (a model argument carries no runtime "
        "shape to check) now catches this always-sound case too, not just the "
        "real OOB one — a known, accepted jit-gap, not a new divergence"),

    # File I/O (STDLIB.md §3.10): `read_file`/`write_file`/`append_file`/
    # `file_exists` now lower under `dmc jit` as host calls into the same
    # `std::fs` operations the interpreter uses. Every path below is under a
    # fictitious top-level directory that requires root to create and is
    # never present in a sandboxed CI checkout, so these are ALWAYS the
    # failure/absent path on both backends — no real file is ever touched,
    # so the probe needs no scratch directory and leaves nothing to clean up
    # (`compiler/tests/jit_file_io.rs` covers the successful round trip, with
    # real temp files under `std::env::temp_dir()`). Each collapses `err`/
    # presence to a plain 0/1 so the two backends' outputs compare directly.
    "file_io_write_missing_dir": (
        "fn main() -> i64 {\n"
        " let (_ok, err) = write_file(\"/nonexistent/dmc_jit_probes_never_exists/x.txt\", \"y\")\n"
        " if err == nil { 1 } else { 0 }\n}",
        "write_file to an unwritable path → 0 (a real io::Error, not nil) on both backends"),
    "file_io_append_missing_dir": (
        "fn main() -> i64 {\n"
        " let (_ok, err) = append_file(\"/nonexistent/dmc_jit_probes_never_exists/x.txt\", \"y\")\n"
        " if err == nil { 1 } else { 0 }\n}",
        "append_file to an unwritable path → 0, same shape as write_file"),
    "file_io_read_missing_file": (
        "fn main() -> i64 {\n"
        " let (content, err) = read_file(\"/nonexistent/dmc_jit_probes_never_exists/x.txt\")\n"
        " if err == nil { 1 } else { 0 }\n}",
        "read_file of a missing path → 0 (content is nil, err is a real io::Error)"),
    "file_io_file_exists_false": (
        "fn main() -> i64 { if file_exists(\"/nonexistent/dmc_jit_probes_never_exists/x.txt\") { 1 } else { 0 } }",
        "file_exists on a path that cannot exist → 0"),
    # #802 review: before this PR, `read_file`'s result string was charged to
    # `--forge` under `dmc jit` (bump-allocated by `forge_str_from_rust`) but
    # NOT under `dmc run` — the same program could need a `--forge` budget on
    # one backend it never needed on the other (MEMORY.md §1.1's "every
    # materialization is charged on both backends" rule, since #790). Fixed
    # by charging a builtin's fresh `Value::Str` result the way a `Value::Tensor`
    # result already was (`interp.rs`'s `charge_builtin_result`). Both probes
    # below write and read back the same real 4096-byte fixture
    # (`compiler/tests/jit_file_io.rs` covers the round trip's correctness;
    # these cover its Forge cost) and, like the `meter_intermediates_p1_imp_*`
    # pair above, bracket each backend's OWN bisected minimum rather than one
    # shared number: `dmc run`'s is 4096 B (the content's raw UTF-8 length,
    # nothing else); `dmc jit`'s is 8704 B (that content charged twice — once
    # for the literal, once for the forge string `read_file` hands back —
    # plus the 8-byte length-header and 64-byte alignment overhead on every
    # forge allocation, `write_file`'s (nil, err) pair, and the two
    # FORGE_FILE_IO_FIXTURE path-literal allocations, one per call site).
    # `_below_min` sits one byte under the interpreter's (the smaller): both
    # backends already agree by refusing. `_at_jit_min` sits AT the JIT's
    # (the larger): both agree by succeeding. Both numbers are specific to
    # FORGE_FILE_IO_FIXTURE's exact path length — re-bisect if that path ever
    # changes shape.
    "file_io_read_result_charged_to_forge_below_min": (
        "fn main() -> i64 {\n"
        f"    let content = \"{'x' * 4096}\"\n"
        f"    let (_wv, _werr) = write_file(\"{FORGE_FILE_IO_FIXTURE}\", content)\n"
        f"    let (s, _rerr) = read_file(\"{FORGE_FILE_IO_FIXTURE}\")\n"
        "    len(s)\n"
        "}",
        "one byte under the interpreter's 4096 B minimum: both backends "
        "refuse (both-fail) — before this fix `dmc run` charged read_file's "
        "content nothing and ran at this budget"),
    "file_io_read_result_charged_to_forge_at_jit_min": (
        "fn main() -> i64 {\n"
        f"    let content = \"{'x' * 4096}\"\n"
        f"    let (_wv, _werr) = write_file(\"{FORGE_FILE_IO_FIXTURE}\", content)\n"
        f"    let (s, _rerr) = read_file(\"{FORGE_FILE_IO_FIXTURE}\")\n"
        "    len(s)\n"
        "}",
        "at `dmc jit`'s 8704 B minimum (comfortably above the interpreter's "
        "own 4096 B): both backends succeed and agree (OK, => 4096)"),

    # #811 — see the `MUST_LOWER` entry and the `GAP_ALLOWLIST` entry above
    # for the full story. The two cases the issue itself names: `total(t)`
    # with a `Tensor` argument (lowers, MUST_LOWER), and `total(kv)` with a
    # `KV` argument (SPEC.md §3.6) — the #811 rework refuses this one
    # unconditionally instead, even though this particular KV happens to be
    # filled to exactly its declared capacity: the JIT has no way to tell
    # that from a KV only partway filled, so it cannot special-case "this one
    # happens to be full" at compile time.
    "generic_view_param_tensor_arg_811": (
        "fn total[N](v: View[u8, [N]]) -> i64 {\n"
        " let !s = 0\n let !i = 0\n"
        " while i < N { s = s + (v[i] as i64)\n i = i + 1 }\n"
        " s\n}\n"
        "fn main() -> i64 {\n"
        " let !t = forge.zeros[u8, [4]]\n"
        " t[0] = 1\n t[1] = 2\n t[2] = 3\n t[3] = 4\n"
        " total(t)\n}",
        "#811: `N` inferred from a `Tensor[u8,[4]]` argument to a "
        "`View[u8,[N]]` param, unbracketed → 10"),
    "generic_view_param_kv_arg_811": (
        "fn total[N](v: View[f32, [N]]) -> f32 {\n"
        " let !s = 0.0\n let !i = 0\n"
        " while i < N { s = s + (v[i] as f64)\n i = i + 1 }\n"
        " s as f32\n}\n"
        "fn main() -> f32 {\n"
        " let !k = forge.kv[f32, [~]](capacity = 4)\n"
        " let !one = forge.zeros[f32, [1]]\n"
        " one[0] = 1.0\n k <- one\n"
        " one[0] = 2.0\n k <- one\n"
        " one[0] = 3.0\n k <- one\n"
        " one[0] = 4.0\n k <- one\n"
        " total(k)\n}",
        "#811 rework: a `View` param's streaming-axis dimension bound to a "
        "`KV` argument now refuses (jit-gap, tracked) even though this KV is "
        "filled to exactly its declared capacity — `dmc run` still answers 10"),
    # #811 reviewer rework (final pass): a NON-generic `Tensor[f32,[4]]`
    # parameter — not a `View` — coerced from a `KV` argument via `coerce_to`'s
    # ordinary call-argument path, the case the two `generic_view_param_*_811`
    # probes above never exercise (both call a `View[T,[N]]`-typed template
    # param). `coerce_to`'s `(TyKind::KV, TyKind::Tensor)` arm was left in
    # place as "inert" on the theory that `unify_param_type`'s streaming-axis
    # refusal always intercepts first — true for the `View`-template path, but
    # this arm is also reachable from a plain, non-generic call, which never
    # goes through `unify_param_type` at all. `total(k)` here is check-clean
    # and the arm fired for real: it skipped the KV's 16-byte header and
    # handed over the data region as an ordinary 4-element `Tensor[f32,[4]]`
    # pointer, so `dmc jit` answered `7` — the one real append plus three
    # stale (never-written) slots past the fill point read as zero — while
    # `dmc run` faults `index 1 out of bounds for axis 0 of size 1` (the KV's
    # real runtime length is 1, not the capacity of 4). Fixed by deleting the
    # arm outright, restoring the pre-#811 refusal both backends now share
    # (both-fail): `dmc jit` refuses with `cannot convert `KV[f32, [~]]` to
    # `Tensor[f32, [4]]`; use `dmc run` for full semantics`.
    "non_generic_kv_arg_coerced_to_tensor_param_811": (
        "fn total(v: Tensor[f32, [4]]) -> f32 {\n"
        " let !s = 0.0\n let !i = 0\n"
        " while i < 4 { s = s + (v[i] as f64)\n i = i + 1 }\n"
        " s as f32\n}\n"
        "fn main() -> f32 {\n"
        " let !k = forge.kv[f32, [~]](capacity = 4)\n"
        " let !one = forge.zeros[f32, [1]]\n"
        " one[0] = 7.0\n k <- one\n"
        " total(k)\n}",
        "#811 reviewer rework: a non-generic `Tensor[f32,[4]]` param coerced "
        "from a capacity-4 KV holding one append — both backends now refuse "
        "(both-fail); the removed `coerce_to` arm used to answer `dmc jit`'s "
        "ordinary call-argument path off a stale zero past the real fill "
        "point (`=> 7`), where `dmc run` faults `index 1 out of bounds for "
        "axis 0 of size 1`"),
    # #804: `s + "x"` (`apply_binop`'s string-concat arm) lives outside
    # `call_builtin`, so #802's `charge_builtin_result` never saw it — `dmc
    # jit` bump-allocates and charges the fresh forge string every `+`
    # produces (header + 64 B alignment, per string), `dmc run` charged
    # nothing at all. Fixed by giving `charge_intermediate` — the same choke
    # point `Expr::BinOp` already calls after every `apply_binop` — a
    # `Value::Str` arm, charged at the fresh string's byte length. Loop
    # produces strings of length 1..1000, so `dmc run`'s own minimum is
    # 1+2+...+1000 = 500 500 B; `dmc jit`'s is 604 224 B (the same 500 500 B
    # plus the 8-byte length header and 64-byte alignment on each of the 1000
    # concat results and the loop's 1000 `"x"` literal allocations, plus one
    # more for the initial `""` literal — the JIT charges every string
    # literal occurrence as its own forge materialization, an asymmetry with
    # no interpreter analogue; see `MEMORY.md §1.1`'s over-charge list).
    # `_below_min` sits one byte under the interpreter's (the smaller): both
    # backends already agree by refusing. `_at_jit_min` sits AT the JIT's
    # (the larger): both agree by succeeding.
    "str_concat_charged_to_forge_below_min": (
        "fn main() -> i64 {\n"
        " let !s = \"\"\n"
        " for i in 0..1000 {\n"
        "  s = s + \"x\"\n"
        " }\n"
        " len(s)\n"
        "}",
        "one byte under the interpreter's 500 500 B minimum: both backends "
        "refuse (both-fail) — before this fix `dmc run` charged the concat "
        "loop nothing and ran at this budget"),
    "str_concat_charged_to_forge_at_jit_min": (
        "fn main() -> i64 {\n"
        " let !s = \"\"\n"
        " for i in 0..1000 {\n"
        "  s = s + \"x\"\n"
        " }\n"
        " len(s)\n"
        "}",
        "at `dmc jit`'s 604 224 B minimum (comfortably above the "
        "interpreter's own 500 500 B): both backends succeed and agree "
        "(OK, => 1000)"),

    # #839: `eval_dim` used to substitute a bare 4 on ANY error out of a
    # constructor's shape expression, so `forge.zeros[f32, [n()]]` with a
    # raising `n()` silently built a 4-element tensor and kept running. The
    # JIT refuses a non-literal shape element outright (its own tracked gap,
    # unrelated to what `n()` does), so all three land `both-fail`: `dmc run`
    # now refuses too, at `n()`'s own error, and `dmc jit` refuses at the
    # shape position before it would ever call `n()`. Neither failure mode
    # matches the other's message, so none of these belong in
    # `BUDGET_EXHAUSTION_PROBES` — that dict pins BOTH backends to the SAME
    # substring, which would be wrong here on purpose.
    "shape_fn_runtime_error_839": (
        "fn n(k: i64) -> i64 {\n"
        " let !t = forge.zeros[f32, [2]]\n"
        " t[k] as i64\n"
        "}\n"
        "fn main() -> f32 {\n"
        " let t = forge.zeros[f32, [n(10)]]\n"
        " sum(t)\n"
        "}",
        "a shape fn's out-of-bounds index: both backends refuse (both-fail) "
        "— before #839 `dmc run` silently built a 4-element tensor and "
        "printed `=> 0`"),
    "shape_fn_forge_exhaustion_839": (
        "fn n() -> i64 {\n"
        " let !t = forge.zeros[f32, [2000]]\n"
        " 4\n"
        "}\n"
        "fn main() -> f32 {\n"
        " let t = forge.zeros[f32, [n()]]\n"
        " sum(t)\n"
        "}",
        "a shape fn's own `forge.zeros` exhausts a deliberately tiny "
        "--forge budget: both backends refuse (both-fail) — before #839 "
        "`dmc run` silently built a 4-element tensor and printed `=> 0`"),
    "shape_fn_budget_stale_839": (
        "fn n() -> i64 {\n"
        " let b = forge.carve(forge.root, 4096)\n"
        " let !t = forge.zeros[f32, [4]] in b\n"
        " forge.reap(b)\n"
        " t[0] as i64\n"
        "}\n"
        "fn main() -> f32 {\n"
        " let t = forge.zeros[f32, [n()]]\n"
        " sum(t)\n"
        "}",
        "a shape fn reads a tensor out of its own already-reaped budget "
        "(`budget-stale`): both backends refuse (both-fail) — before #839 "
        "`dmc run` silently built a 4-element tensor and printed `=> 0`"),
}

_VAL = re.compile(r"=>\s*(-?[\d.]+(?:e[-+]?\d+)?|true|false|NaN|inf)")

# #614: a probe whose behavior only exists under an arena budget maps its name
# here to the extra CLI args its `dmc` invocation needs — mirroring the header
# comment each row in the arena-budgets design's own probe notes documents
# its own flag with. `run()` appends these after the file path; `--forge=<size>`
# parses in any position (`main.rs`'s flag-stripping loop scans the whole
# argv), so the file path coming first is fine.
FORGE_FLAG = {
    # #781: the interpreter now charges these f32 tensors at their declared
    # 4-byte element width rather than its f64 storage width, halving what
    # it charges per allocation — a property of CUMULATIVE bytes charged,
    # not of the per-element width, so in principle every ceiling tuned
    # against the old f64 charge could need halving to keep exhausting at
    # the same iteration. In practice (PR #790 reviewer rework, verified by
    # re-bisecting each probe) only TWO of the six actually needed it: m3
    # (the narrowest budget, 1M → 512K) and g1_fwd (below). m8, x6 and x8
    # already held their `both-fail` verdict at their OLD, pre-#781 flags —
    # their iteration counts are high enough that even the halved per-step
    # charge still exhausts the same unchanged budget (x8's own interpreter
    # minimum, for instance, is unchanged at 2 621 440 B either way) — so
    # halving them here is DEFENSIVE headroom, not a required fix; left
    # halved anyway so this table does not silently depend on how close
    # each budget happens to sit to its actual minimum.
    # g1 (`.fwd_bwd`) keeps its `4M` too, but no longer because it is
    # exempt: #791 instruments the tape evaluator that entry point runs
    # through, so `dmc run` now charges the loop (3 358 720 B minimum) —
    # `4M` was simply always wide enough. g1_fwd (a plain forward-only
    # call) is likewise back at `4M`; see its own comment below and in the
    # PROBES table above.
    # #674 slice 3b: brackets the carve, not an allocation — `4096` free,
    # `4097` requested, refused at the carve itself (R1/PRE), not later.
    "budget_carve_pre_refused_674": ["--forge=4096"],
    # #674 slice 3d: the five exit edges. `1M` is comfortably above what
    # either backend needs for `4096 + 16384` (the JIT charges a let-bound
    # constructor twice, #640, so its own minimum is roughly double the
    # interpreter's); `20479` is one byte under the ROOT's own requirement
    # on the LOWER of the two, so both refuse. The pair brackets the
    # post-block allocation against the root — see the probes' own comments.
    "budget_edge_normal_outer_charged_674": ["--forge=20480"],
    "budget_edge_normal_root_below_min_674": ["--forge=20479"],
    "budget_edge_return_outer_charged_674": ["--forge=20480"],
    "budget_edge_return_root_below_min_674": ["--forge=20479"],
    "budget_edge_break_outer_charged_674": ["--forge=20480"],
    "budget_edge_break_root_below_min_674": ["--forge=20479"],
    "budget_edge_continue_outer_charged_674": ["--forge=20480"],
    "budget_edge_continue_root_below_min_674": ["--forge=20479"],
    "budget_edge_query_outer_charged_674": ["--forge=20544"],
    "budget_edge_query_root_below_min_674": ["--forge=20479"],
    # #674 slice 3d, §3.5: the vault-routing trio. Re-bisected after #837
    # removed #640's let-bound double charge (neither of the first row's
    # allocations is `let`-bound, so it was never inflated by #640, but it
    # shares this table with rows that were). `65 600` is the survives-reap
    # source's own bisected minimum on either backend: `forge.carve(root,
    # 65536)` for `b` plus the 64 B the tiny `vault.ones[f32, [4]]` costs on
    # `dmc jit` (routed to the root, §3.5) before `reap(b)` credits `b`'s
    # extent back — `dmc run`'s own minimum is not larger, so one flag
    # serves both. `36864` is `4 096` (`b`'s carve) `+ 2 × 16 384` (the two
    # `vault.zeros[f32, [4096]]` statements, charged to the root on both
    # backends, §3.5) — the charges-root row's own bisected minimum, shared
    # with the charges-`b` row below: the same three lines with `forge` for
    # `vault` refuse at `b`'s own 4 KiB extent regardless of the root's
    # size, so any budget that admits the carve makes that row refuse — the
    # pair's whole content is that one spelling runs and the other refuses
    # at the SAME budget.
    "budget_vault_ctor_in_enter_survives_reap_674": ["--forge=65600"],
    "budget_vault_ctor_in_enter_charges_root_674": ["--forge=36864"],
    "budget_forge_ctor_in_enter_charges_b_674": ["--forge=36864"],
    # #674 slice 3d: the arena-budget design's §4 own flag — every row of its table
    # passes `--forge` explicitly, and §4.1 says why it must (an unbounded
    # root makes PRE vacuous, so (INV)-at-the-root is untestable there).
    "ac_inv_four_carves_674": ["--forge=256K"],
    "ac_inv_fifth_carve_refused_674": ["--forge=256K"],
    "ac_loc_recarve_extent_674": ["--forge=256K"],
    "ac_loc_over_extent_refused_674": ["--forge=256K"],
    "ac_pre_at_free_674": ["--forge=256K"],
    "ac_pre_refused_674": ["--forge=256K"],
    "ac_prea_disjoint_budgets_674": ["--forge=256K"],
    "ac_prea_own_ceiling_refused_674": ["--forge=256K"],
    "ac_nn_recarve_cycles_674": ["--forge=256K"],
    "ac_nn_over_root_refused_674": ["--forge=256K"],
    # #674 slice 4: `examples/tenant_budgets.dmc`, one row per section, each
    # one byte under the LOWER of the two backends' bisected minimums for
    # that section's `fn main` transcription. Bisected on both backends
    # against the same release binary. Where the two differ the JIT is the
    # higher, always by whole 64 B slots it spends on a `Tenant` model or a
    # propagated `(i64, Err)` tuple that the interpreter does not build in
    # Forge at all: (PRE-A) 131 072 / 131 200, (a) 82 496 / 82 560, (b)
    # 65 536 / 65 600. The other six agree exactly: (INV) 262 144, (LOC)
    # 65 536, (PRE) 65 536, (NN) 131 072, criterion 4 131 072, (c) 65 536.
    "tenant_budgets_inv_674": ["--forge=262143"],
    "tenant_budgets_loc_674": ["--forge=65535"],
    "tenant_budgets_pre_674": ["--forge=65535"],
    "tenant_budgets_prea_674": ["--forge=131071"],
    "tenant_budgets_nn_674": ["--forge=131071"],
    "tenant_budgets_reap_order_674": ["--forge=131071"],
    "tenant_budgets_exit_edges_674": ["--forge=82495"],
    "tenant_budgets_field_reap_674": ["--forge=65535"],
    "tenant_budgets_ambient_call_674": ["--forge=65535"],
    "forge_restore_no_budget_m3": ["--forge=512K"],
    "forge_restore_no_budget_m8": ["--forge=4M"],
    "forge_restore_no_budget_x6": ["--forge=128K"],
    "forge_restore_no_budget_x8": ["--forge=128K"],
    # #674 slice 2: brackets c1's own shared minimum from below (`dmc run`'s
    # 1 048 592 B, the lower of the two backends' minimums) and at it (`dmc
    # jit`'s 1 048 640 B, the higher — see the probe comments above).
    "chunk_granularity_c1_below_shared_min": ["--forge=1048591"],
    "chunk_granularity_c1_at_shared_min": ["--forge=1048640"],
    "chunk_granularity_second_chunk_matches_used": ["--forge=556870912"],
    "forge_restore_no_budget_g1": ["--forge=4M"],
    # PR #790: raised twice and now back where it started — `4M` → `8M` by
    # the first #781 pass (the forward-only call's own activations), `8M` →
    # `16M` by an intermediate pass (the `!W` copy charged twice and
    # released never), and back to `4M` once #791 gave the forward-only
    # entry its rewind and stopped charging the copy-out. See the probe's
    # own header comment in the PROBES table above for the accounting.
    # `dmc run`'s minimum is 3 309 568 B; `dmc jit`'s is unchanged at
    # 3 375 104 B.
    "forge_restore_no_budget_g1_fwd": ["--forge=4M"],
    # #791: one byte under `dmc run`'s own bisected minimum on each g1
    # source — the lower of the two backends' minimums on both, so both
    # refuse. See the probes' comments above for why these are pinned to the
    # interpreter's number and not the JIT's.
    "meter_ad_temporaries_fwd_below_min": ["--forge=3309567"],
    "meter_ad_temporaries_fwd_bwd_below_min": ["--forge=3358719"],
    # PR #790 defect 4: one byte under the JIT's own g1-fwd minimum and
    # ABOVE the interpreter's, so `dmc run` answers and `dmc jit` refuses —
    # the expected, explained divergence tracked in GAP_ALLOWLIST.
    # #640: re-bisected from `3375103`. The JIT's own g1-fwd minimum fell
    # 32 768 B (3 375 104 → 3 342 336) when the flat `let` of an arena
    # constructor stopped landing the tensor twice, so the old value now sits
    # ABOVE it and both backends succeed — the probe would keep passing
    # while probing nothing. The residual gap to `dmc run`'s unmoved
    # 3 309 568 B is now 32 768 B, `main`'s two `vault.zeros` charged once
    # each against the JIT's single arena (`MEMORY.md §1.1`), not twice.
    "meter_ad_temporaries_fwd_below_jit_min": ["--forge=3342335"],
    # #781: see the probe comments above — each pair brackets its OWN
    # backend's bisected minimum. #640 re-bisected the two `_at_jit_min`
    # values (81 984 → 49 152, 32 833 600 → 32 800 768): with the flat
    # `let`'s double charge gone the JIT's minimum now COINCIDES with the
    # interpreter's on both sources, so each pair brackets one shared
    # number and the key's `_at_jit_min` name is now historical.
    "meter_intermediates_p1_imp_below_min": ["--forge=49151"],
    "meter_intermediates_p1_imp_at_jit_min": ["--forge=49152"],
    "meter_intermediates_p2_loop_below_min": ["--forge=32800767"],
    "meter_intermediates_p2_loop_at_jit_min": ["--forge=32800768"],
    # PR #790 reviewer rework (defects 1 and 2, both in `Expr::TensorLit`) —
    # see the probe comments above for what each brackets.
    "nested_literal_charged_once_below_min": ["--forge=255"],
    "nested_literal_charged_once_at_min": ["--forge=256"],
    "reshape_shape_arg_not_charged_below_min": ["--forge=32767"],
    # #640: re-bisected from `49152` — the JIT's minimum on this source fell
    # to the interpreter's own 32 768 B once `let a = forge.zeros[…]` stopped
    # landing `a` twice, so this pair, too, now brackets one shared number.
    "reshape_shape_arg_not_charged_at_jit_min": ["--forge=32768"],
    # #802 review: see the probe comments above — brackets each backend's OWN
    # bisected minimum for a real `read_file` round trip, not a shared one.
    "file_io_read_result_charged_to_forge_below_min": ["--forge=4095"],
    "file_io_read_result_charged_to_forge_at_jit_min": ["--forge=8704"],
    # #804: see the probe comments above — brackets the interpreter's own
    # 500 500 B minimum from below (both refuse) and the JIT's own 604 224 B
    # minimum at (both succeed).
    "str_concat_charged_to_forge_below_min": ["--forge=500499"],
    "str_concat_charged_to_forge_at_jit_min": ["--forge=604224"],
    # #839: comfortably below the 8000 B `n()` itself needs (2000 f32
    # elements) — no bisection here, this probe only needs `n()`'s own
    # allocation to exhaust, not a boundary byte.
    "shape_fn_forge_exhaustion_839": ["--forge=4096"],
}

# The m3/m8/x6/x8 budget probes, and the two PR #790 defect probes below
# them, are pinned to a specific FAILURE MODE (`forge arena exhausted`), not
# merely "both backends reject" — a probe that stopped parsing, or crashed
# for an unrelated reason, would still score `both-fail` under the coarse
# check and stay green.  See the assertion beside `both-fail (run=...)`
# below.
#
# #674 slice 3d: a DICT now, name -> the substring both backends' failures
# must contain, because #674's own refusals are not spelled `forge
# arena exhausted`. The arena-budget design's §4 asks for exactly this ("pinned to
# the `forge arena exhausted` / `budget-*` failure mode through
# BUDGET_EXHAUSTION_PROBES"): PRE-A exhausts an entered budget and keeps the
# old sentence, while PRE (a carve past the parent) and R2 (a reap with a
# live child) refuse with `budget-carve` / `budget-reap` instead. Every
# pre-existing member maps to the string the tuple form hard-coded, so no
# existing verdict moves.
BUDGET_EXHAUSTION_PROBES = {
    "forge_restore_no_budget_m3": "forge arena exhausted",
    "forge_restore_no_budget_m8": "forge arena exhausted",
    "forge_restore_no_budget_x6": "forge arena exhausted",
    "forge_restore_no_budget_x8": "forge arena exhausted",
    "nested_literal_charged_once_below_min": "forge arena exhausted",
    "reshape_shape_arg_not_charged_below_min": "forge arena exhausted",
    "meter_ad_temporaries_fwd_below_min": "forge arena exhausted",
    "meter_ad_temporaries_fwd_bwd_below_min": "forge arena exhausted",
    "chunk_granularity_c1_below_shared_min": "forge arena exhausted",
    "file_io_read_result_charged_to_forge_below_min": "forge arena exhausted",
    "str_concat_charged_to_forge_below_min": "forge arena exhausted",
    # #674 slice 3d, the five exit edges (the arena-budget design's B5). Each is the
    # `*_outer_charged_674` row below re-run one byte under the ROOT's own
    # capacity for the post-block allocation, so the pair brackets that
    # allocation against the root rather than asserting a vacuous success.
    "budget_edge_normal_root_below_min_674": "forge arena exhausted",
    "budget_edge_return_root_below_min_674": "forge arena exhausted",
    "budget_edge_break_root_below_min_674": "forge arena exhausted",
    "budget_edge_continue_root_below_min_674": "forge arena exhausted",
    "budget_edge_query_root_below_min_674": "forge arena exhausted",
    # #674 slice 3d, §3.5: the `forge`-spelled half of the vault-routing
    # minimal pair — it charges `b`, so it must refuse where its one-token
    # `vault.*` twin runs.
    "budget_forge_ctor_in_enter_charges_b_674": "forge arena exhausted",
    # #674 slice 3d, the arena-budget design's §4 five invariants — the refused half
    # of each bisection, pinned to the tag §2.3's table gives that refusal.
    "ac_inv_fifth_carve_refused_674": "budget-carve",
    "ac_loc_over_extent_refused_674": "budget-carve",
    "ac_pre_refused_674": "budget-carve",
    "ac_prea_own_ceiling_refused_674": "forge arena exhausted",
    "ac_nn_over_root_refused_674": "budget-carve",
    # #674 slice 3d, §4's sixth criterion (a reap with a live child) — R2,
    # already probed by `budget_reap_live_child_refused_674` above; pinned to
    # its own tag here so it cannot stay green by failing some other way.
    "budget_reap_live_child_refused_674": "budget-reap",
    # #674 slice 4: the nine `examples/tenant_budgets.dmc` sections, each one
    # byte under its own bisected root requirement. All nine bottom out at a
    # carve — the section's outermost budget is the first thing the shrunken
    # root cannot serve — so all nine pin `budget-carve` rather than
    # `forge arena exhausted`, on both backends.
    "tenant_budgets_inv_674": "budget-carve",
    "tenant_budgets_loc_674": "budget-carve",
    "tenant_budgets_pre_674": "budget-carve",
    "tenant_budgets_prea_674": "budget-carve",
    "tenant_budgets_nn_674": "budget-carve",
    "tenant_budgets_reap_order_674": "budget-carve",
    "tenant_budgets_exit_edges_674": "budget-carve",
    "tenant_budgets_field_reap_674": "budget-carve",
    "tenant_budgets_ambient_call_674": "budget-carve",
}

# #674 slice 3d: probes whose whole assertion is that BOTH backends ANSWER.
# The battery scores an `ok`/`ok` probe that regresses to `both-fail` as
# green — that branch is one line below the `MUST_LOWER` gate and has
# exactly the hole `MUST_LOWER` closed for refusals: "the file didn't even
# parse" and "the register leaked and the next allocation was mis-charged"
# both score `both-fail (run=err jit=err)` and exit 0. For the exit-edge
# probes that hole is fatal, because a LEAKED budget register makes the
# post-block allocation fail on both backends — the exact regression the
# gate exists to catch would silence itself.
#
# A name here turns "either backend did not answer" into a failure. It says
# nothing about the VALUES, which the ordinary `rv == jv` comparison still
# judges; and a name in `BUDGET_EXHAUSTION_PROBES` (a probe that must FAIL)
# or `GAP_ALLOWLIST` (one the JIT may refuse) contradicts it, so `main`
# refuses to run on an overlap.
MUST_ANSWER = {
    "budget_edge_normal_outer_charged_674":
        "#674 slice 3d — B5 edge 1/5: normal exit restores the outer budget",
    "budget_edge_return_outer_charged_674":
        "#674 slice 3d — B5 edge 2/5: `return` restores the outer budget",
    "budget_edge_break_outer_charged_674":
        "#674 slice 3d — B5 edge 3/5: `break` restores the outer budget",
    "budget_edge_continue_outer_charged_674":
        "#674 slice 3d — B5 edge 4/5: `continue` restores the outer budget",
    "budget_edge_query_outer_charged_674":
        "#674 slice 3d — B5 edge 5/5: a `?` early return restores the outer budget",
    "ac_inv_four_carves_674":
        "#674 slice 3d — the arena-budget design's §4 (INV): the root's capacity is its children's whole extents",
    "ac_loc_recarve_extent_674":
        "#674 slice 3d — §4 (LOC): a reap credits the whole extent, spent and unspent alike",
    "ac_pre_at_free_674":
        "#674 slice 3d — §4 (PRE): a carve of exactly `free(parent)` is admitted",
    "ac_prea_disjoint_budgets_674":
        "#674 slice 3d — §4 (PRE-A): two disjoint budgets each serve from their own ceiling",
    "ac_nn_recarve_cycles_674":
        "#674 slice 3d — §4 (NN): 64 reap/re-carve cycles leave the root exactly where it started",
    "budget_vault_ctor_in_enter_survives_reap_674":
        "#674 slice 3d — the arena-budget design's §3.5: a `vault.*` constructor inside `enter(b)` survives `reap(b)`",
    "budget_vault_ctor_in_enter_charges_root_674":
        "#674 slice 3d — §3.5: a `vault.*` constructor inside `enter(b)` charges the root, not `b`",
    "design_probe_s3_alloc_in_674":
        "#674 slice 3d — probe s3: `<ctor> in b` lowers on both backends",
    "design_probe_s4_budget_block_674":
        "#674 slice 3d — probe s4: `forge.enter(b) { … }` lowers on both backends",
    "budget_root_is_a_budget_value_674":
        "#674 slices 3b + 3c — `forge.root` is a real `Budget` on both backends",
    "budget_carve_and_reap_succeed_674":
        "#674 slices 3b + 3c — carve then reap both succeed on both backends",
}

# #867: `both-fail` alone only checks exit codes — two refusals that disagree
# on WORDING (e.g. `dmc run`'s `got nil and int` vs `dmc jit`'s `got nil and
# nil` for the identical program, #865's blocking finding) score exactly as
# green as two that agree, because nothing ever compared their stderr.
# GAP_ALLOWLIST/INTERP_REFUSAL and BUDGET_EXHAUSTION_PROBES's substring pin
# close every OTHER blind spot a bare exit-code/`rs`/`js` comparison has; this
# is the one both-fail branch itself still has.
#
# A name here asserts the two backends' stderr is the SAME diagnostic, not
# merely two nonzero exits — normalised (`_normalize_stderr` below) to fold
# out the two things the backends are entitled to spell differently and
# nothing else:
#   * a leading `runtime error at L:C: ` / `type error at L:C: ` / `parse
#     error at L:C: ` collapses to its bare kind word (`runtime error: ` /
#     `type error: ` / `parse error: `) — `interp.rs`'s own `Display` prints
#     the SAME kind with no location at all when the AST node it traps on
#     kept no `Span` (`RuntimeError`'s `Display`, `interp.rs`), while the
#     JIT's own trap sites always know theirs;
#   * a `=> value` echo line is dropped — it is `run()`'s own harness marker
#     for an OK probe, never part of a diagnostic, and never fires here
#     anyway (both backends already failed) but a probe whose source string
#     happens to print one of its own before trapping should not spuriously
#     poison the comparison;
#   * surrounding whitespace is trimmed.
# The two normalised texts must then be BYTE-EQUAL. A mismatch is
# `WORDING-DIVERGE`, folded into `diverged` (a failure), not `both_fail`.
#
# Audited against every both-fail row in `PROBES` as of #867, against the
# release `dmc` binary, each row run with its own `FORGE_FLAG` (if any) and
# no other flags. The #767 rows (`map_get_add_absent_both_fail`) are here:
# #865 gave the JIT's absent-operand trap the interpreter's own located
# wording, so the two now read alike modulo location.
# `budget_reap_live_child_refused_674` joined them in #842, the one
# divergence this audit itself turned up: `interp.rs` hardcoded the plural
# pronoun for a SINGLE live child while `jit.rs` conjugated. Both backends
# now build that clause through `arena::budget_live_child_clause`, and
# `compiler/tests/budget_reap_message_842.rs` compares the two backends'
# whole stderr at counts 1 and 2 independently of this table.
#
# Left OUT, and why — every other both-fail row as of this audit:
#   * `tenant_budgets_prea_674`, `tenant_budgets_exit_edges_674` — these two
#     sections are exactly the (PRE-A)/(a) rows the header comment above
#     `tenant_budgets_inv_674` (PROBES) already documents as differing: the
#     JIT is higher by the whole-64-B slots it spends materializing a
#     `Tenant` model/propagated tuple the interpreter's own accounting never
#     builds in Forge. A documented, expected backend-accounting asymmetry,
#     not a wording bug — the "needs"/"only" byte figures themselves differ
#     (e.g. `63.9 KiB` vs `63.8 KiB` free), not just their spelling.
#   * `file_io_read_result_charged_to_forge_below_min` — same shape: the
#     probe's own `#802 review` comment above documents `dmc jit` charging
#     the round-tripped content TWICE (literal + the forge string handed
#     back) plus per-allocation header/alignment overhead the interpreter's
#     accounting has no analogue for, so the "needs N B" figure itself is
#     backend-specific by design, not merely differently worded.
#   * `str_concat_charged_to_forge_below_min` — same shape again: the
#     probe's own `#804` comment documents `dmc run`'s 500 500 B minimum
#     against `dmc jit`'s 604 224 B one (string-literal materialization has
#     no interpreter analogue); the two "needs"/"only" figures are expected
#     to disagree, not just to be worded differently.
#   * `shape_fn_runtime_error_839`, `shape_fn_forge_exhaustion_839`,
#     `shape_fn_budget_stale_839` — the probes' own shared comment above
#     `shape_fn_runtime_error_839` (PROBES) already says it: "Neither
#     failure mode matches the other's message... none of these belong in
#     `BUDGET_EXHAUSTION_PROBES`... which would be wrong here on purpose."
#     `dmc run` traps on ITS OWN runtime error inside the shape fn; `dmc
#     jit` refuses at the shape POSITION before the shape fn's body — two
#     genuinely different diagnostics for two genuinely different reasons,
#     by design.
#   * `lowercase_tensor_shape_param_not_inferred_733`,
#     `lowercase_model_shape_param_not_inferred_g30` — `dmc run` raises its
#     own "shape param `n` cannot be inferred from any argument. Either
#     pass a tensor..." (a located `runtime error`); `dmc jit` raises a
#     wholly separate, differently-worded `jit error: could not infer
#     shape parameter...`. Same underlying condition, two unrelated
#     message-building code paths that were never unified — not the
#     `runtime error`/`type error`/`parse error` family this normalisation
#     covers at all.
#   * `bare_model_arg_against_bracketed_callee_810`,
#     `non_generic_kv_arg_coerced_to_tensor_param_811` — one side is `dmc
#     run`'s own runtime contradiction/bounds check (a located `runtime
#     error`), the other is `dmc jit`'s CODED refusal for a construct it
#     declines to lower at all (`jit unsupported: ...` / `jit unsupported:
#     cannot convert ...`) — a checker-shaped refusal on one backend and a
#     coded JIT gap on the other, by design; see each probe's own comment
#     in `PROBES`.
SAME_MESSAGE = {
    "map_get_add_absent_both_fail",
    "omitted_return_type_let_bound_call",
    "omitted_return_provenance_block_shadow",
    "omitted_return_shape_generic_model_method",
    "matmul_1xN_Nx1",
    "port_lang_annotation_is_checked",
    "use_import_private_let_leak_724",
    "forge_restore_no_budget_m3",
    "forge_restore_no_budget_m8",
    "forge_restore_no_budget_x6",
    "forge_restore_no_budget_x8",
    "chunk_granularity_c1_below_shared_min",
    "budget_carve_not_yet_implemented_674",
    "budget_reap_not_yet_implemented_674",
    "budget_carve_pre_refused_674",
    "budget_reap_live_child_refused_674",
    "budget_edge_normal_root_below_min_674",
    "budget_edge_return_root_below_min_674",
    "budget_edge_break_root_below_min_674",
    "budget_edge_continue_root_below_min_674",
    "budget_edge_query_root_below_min_674",
    "ac_inv_fifth_carve_refused_674",
    "ac_loc_over_extent_refused_674",
    "ac_pre_refused_674",
    "ac_prea_own_ceiling_refused_674",
    "ac_nn_over_root_refused_674",
    "budget_forge_ctor_in_enter_charges_b_674",
    "design_probe_s6_budget_name_reserved_674",
    "tenant_budgets_inv_674",
    "tenant_budgets_loc_674",
    "tenant_budgets_pre_674",
    "tenant_budgets_nn_674",
    "tenant_budgets_reap_order_674",
    "tenant_budgets_field_reap_674",
    "tenant_budgets_ambient_call_674",
    "meter_ad_temporaries_fwd_below_min",
    "meter_ad_temporaries_fwd_bwd_below_min",
    "meter_intermediates_p1_imp_below_min",
    "meter_intermediates_p2_loop_below_min",
    "nested_literal_charged_once_below_min",
    "reshape_shape_arg_not_charged_below_min",
    "generic_model_param_two_args_disagree_732",
}

# `_normalize_stderr`'s one structural rule: a LOCATED diagnostic's `at L:C:`
# is elective (the interpreter omits it when its trap site kept no `Span`);
# the kind word before it and the message after it are not.
_LOCATED_PREFIX = re.compile(r"^(runtime error|type error|parse error) at \d+:\d+: ")


def _normalize_stderr(text):
    """#867: fold `SAME_MESSAGE`'s two elective spelling differences out of a
    both-fail probe's raw stderr — see `SAME_MESSAGE`'s own header comment
    for what this does and does not paper over. Applied per-line so a
    multi-error `type error` block normalises the same way on every line,
    not just its first."""
    out_lines = []
    for line in text.splitlines():
        if line.strip().startswith("=>"):
            continue
        m = _LOCATED_PREFIX.match(line)
        if m:
            line = f"{m.group(1)}: {line[m.end():]}"
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def run(dmc, mode, src, timeout, aux=None, extra_args=None):
    # A private directory rather than a bare temp file: `use` paths are relative
    # to the importing file (SPEC §6.7), so a multi-file probe's siblings have
    # to sit beside it — and nothing else may, or `dmc test`-shaped tooling
    # would sweep up the rest of the system temp root.
    #
    # Returns (status, short_value, full_text): `short_value` is the existing
    # truncated-for-display summary every caller before #867 used (a matched
    # `=> value`, or the first/last line of stderr clipped to 90 chars);
    # `full_text` is the untruncated, unsliced `stdout+stderr` — #867's
    # `SAME_MESSAGE` needs the WHOLE diagnostic to normalise and compare, and
    # a both-fail probe's located message routinely runs past 90 characters
    # (`budget-carve: this carve needs 64 KiB, but only 0 B of the parent
    # budget's 256 KiB extent is free` alone is over 100).
    with tempfile.TemporaryDirectory(prefix="dmc_probe_") as d:
        p = os.path.join(d, "probe.dmc")
        with open(p, "w") as f:
            f.write(src)
        for name, text in (aux or {}).items():
            with open(os.path.join(d, name), "w") as f:
                f.write(text)
        try:
            r = subprocess.run([dmc, mode, p] + (extra_args or []),
                                capture_output=True, text=True, timeout=timeout)
            out = (r.stdout + r.stderr).strip()
            m = _VAL.search(out)
            if m:
                return ("ok", m.group(1), out)
            if "type error" in out or "refusing to run" in out:
                return ("checkerr", out.splitlines()[0][:90] if out else "", out)
            if r.returncode != 0:
                return ("err", out.splitlines()[-1][:90] if out else "", out)
            # Exit 0, no `=> value`: a nil-returning probe (none here) — treat as ok/empty.
            return ("ok", "", out)
        except subprocess.TimeoutExpired:
            return ("timeout", "", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmc", default=str(DEFAULT_DMC))
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--verbose", action="store_true", help="print every probe, not just problems")
    args = ap.parse_args()

    if not os.path.exists(args.dmc):
        print(f"jit_probes: error: dmc binary not found at {args.dmc} "
                f"(build it: cd compiler && cargo build --release)", file=sys.stderr)
        return 2

    # MUST_LOWER is only a gate while it is coherent: a name in both lists is a
    # contradiction, and a name in neither PROBES nor the run is a gate that
    # silently guards nothing.
    contradictory = sorted(set(MUST_LOWER) & set(GAP_ALLOWLIST))
    if contradictory:
        print(f"jit_probes: error: in both MUST_LOWER and GAP_ALLOWLIST: "
                f"{', '.join(contradictory)}", file=sys.stderr)
        return 2
    unknown = sorted(set(MUST_LOWER) - set(PROBES))
    if unknown:
        print(f"jit_probes: error: MUST_LOWER names no such probe: "
                f"{', '.join(unknown)}", file=sys.stderr)
        return 2
    # FORGE_FLAG had no such check — a typo'd key silently guarded nothing
    # (the flag was never applied and the probe ran unbudgeted) instead of
    # failing the run the way an unknown MUST_LOWER name does above.
    unknown_forge_flag = sorted(set(FORGE_FLAG) - set(PROBES))
    if unknown_forge_flag:
        print(f"jit_probes: error: FORGE_FLAG names no such probe: "
                f"{', '.join(unknown_forge_flag)}", file=sys.stderr)
        return 2
    # #674 slice 3d: the same two coherence checks for MUST_ANSWER. "Both
    # backends must answer" contradicts "both backends must fail with X"
    # (BUDGET_EXHAUSTION_PROBES) and "the JIT may refuse this" (GAP_ALLOWLIST)
    # alike, and a name in neither PROBES nor the run guards nothing.
    contradictory_answer = sorted(
        set(MUST_ANSWER) & (set(BUDGET_EXHAUSTION_PROBES) | set(GAP_ALLOWLIST)))
    if contradictory_answer:
        print(f"jit_probes: error: in MUST_ANSWER and also in "
                f"BUDGET_EXHAUSTION_PROBES/GAP_ALLOWLIST: "
                f"{', '.join(contradictory_answer)}", file=sys.stderr)
        return 2
    unknown_answer = sorted(set(MUST_ANSWER) - set(PROBES))
    if unknown_answer:
        print(f"jit_probes: error: MUST_ANSWER names no such probe: "
                f"{', '.join(unknown_answer)}", file=sys.stderr)
        return 2
    unknown_exhaustion = sorted(set(BUDGET_EXHAUSTION_PROBES) - set(PROBES))
    if unknown_exhaustion:
        print(f"jit_probes: error: BUDGET_EXHAUSTION_PROBES names no such probe: "
                f"{', '.join(unknown_exhaustion)}", file=sys.stderr)
        return 2
    # #867 — same shape as FORGE_FLAG/BUDGET_EXHAUSTION_PROBES above: a typo'd
    # name would silently guard nothing (the both-fail branch below only ever
    # consults SAME_MESSAGE by membership).
    unknown_same_message = sorted(set(SAME_MESSAGE) - set(PROBES))
    if unknown_same_message:
        print(f"jit_probes: error: SAME_MESSAGE names no such probe: "
                f"{', '.join(unknown_same_message)}", file=sys.stderr)
        return 2

    diverged, new_gaps, ok, gaps = [], [], 0, 0
    refusals = 0
    both_fail = 0
    must_lower = []
    must_answer = []
    stale_allowlist = []
    for name, spec in PROBES.items():
        # A probe is (source, note), or (source, note, {sibling: source}) when
        # what it probes only exists across files.
        src, aux = spec[0], (spec[2] if len(spec) > 2 else None)
        extra_args = FORGE_FLAG.get(name)
        rs, rv, r_full = run(args.dmc, "run", src, args.timeout, aux, extra_args)
        js, jv, j_full = run(args.dmc, "jit", src, args.timeout, aux, extra_args)
        verdict = None
        if name in INTERP_REFUSAL:
            # The interpreter must refuse and the JIT must answer. Both halves
            # are asserted: a probe whose `dmc run` starts succeeding is no
            # longer probing what it was allowlisted for.
            if rs != "ok" and js == "ok":
                refusals += 1
                verdict = f"interp-refusal ({INTERP_REFUSAL[name]}): jit={jv}"
            else:
                diverged.append(name)
                verdict = (f"DIVERGE: allowlisted as an interpreter refusal, but "
                           f"run={rs}:{rv!r} jit={js}:{jv!r}")
        elif name in MUST_ANSWER and (rs != "ok" or js != "ok"):
            # #674 slice 3d: the positive gate. Placed AHEAD of the ok/ok arms
            # so it only ever fires on a failure; when both answer, the
            # ordinary value comparison below still judges the answers.
            must_answer.append((name, f"run={rs}:{rv[:60]!r} jit={js}:{jv[:60]!r}"))
            verdict = f"MUST-ANSWER: run={rs} jit={js}"
        elif rs == "ok" and js == "ok" and rv == jv:
            ok += 1
            verdict = f"OK ({rv})"
        elif rs == "ok" and js == "ok":
            diverged.append(name)
            verdict = f"DIVERGE: run={rv!r} jit={jv!r}"
        elif rs == "ok" and js != "ok" and name in MUST_LOWER:
            # The gate. `dmc run` answers and the JIT does not — refusal or
            # defect alike — on a construct this repo has committed to
            # lowering. Not a gap; a failure.
            must_lower.append((name, f"{js}: {jv[:70]}"))
            verdict = f"MUST-LOWER: jit={js}: {jv[:60]}"
        elif rs == "ok" and _UNSUPPORTED in jv:
            gaps += 1
            verdict = f"jit-gap: {jv[:60]}"
            if name not in GAP_ALLOWLIST:
                new_gaps.append((name, jv[:80]))
        elif rs == "ok" and js != "ok" and name in GAP_ALLOWLIST:
            # A jit FAILURE (not a refusal) on a program `dmc run` accepts is a
            # defect — but an explicitly tracked one is allowlistable, so a known
            # issue does not force the probe out of the battery (#480).
            gaps += 1
            verdict = f"jit-gap (tracked failure): {jv[:60]}"
        elif rs != "ok" and js != "ok" and name in SAME_MESSAGE and (
                _normalize_stderr(r_full) != _normalize_stderr(j_full)):
            # #867: a `both-fail` probe pinned to carry the SAME diagnostic on
            # both backends, but it doesn't today — the wording-divergence
            # blind spot #767/#865 found, made a gate. Counted like a
            # divergence (a failure), not folded into `both_fail`.
            diverged.append(name)
            verdict = (f"WORDING-DIVERGE: SAME_MESSAGE requires equal stderr "
                       f"(normalised); run={rs}:{_normalize_stderr(r_full)[:70]!r} "
                       f"jit={js}:{_normalize_stderr(j_full)[:70]!r}")
        elif rs != "ok" and js != "ok":
            want = BUDGET_EXHAUSTION_PROBES.get(name)
            if want is not None and (want not in rv or want not in jv):
                # #614: `both-fail` alone is too coarse for these four — it
                # is what "the file didn't even parse" would also score. Pin
                # the failure mode itself so a probe that broke for a
                # different reason does not stay green under the coarse verdict.
                # #674 slice 3d: the expected mode is per-probe now — `budget-
                # carve` / `budget-reap` for 0674's own refusals, `forge arena
                # exhausted` for PRE-A and for every pre-existing member.
                diverged.append(name)
                verdict = (f"DIVERGE: expected `{want}` on both "
                           f"backends, got run={rs}:{rv!r} jit={js}:{jv!r}")
            elif name in GAP_ALLOWLIST:
                # #771: every OTHER GAP_ALLOWLIST branch above is gated on
                # `rs == "ok"` — this is the one both-fail can reach, and
                # it is exactly the hole #771 found. GAP_ALLOWLIST's entry
                # describes "run answers, jit refuses"; a probe that has
                # regressed to "both refuse" (e.g. a checker fix that now
                # makes `dmc run` itself refuse first) leaves that entry
                # describing a state that no longer exists, and nothing
                # else ever consults GAP_ALLOWLIST once `rs != "ok"` — so a
                # dead entry stayed silently green until this check named
                # it. Same shape as `MUST_LOWER`/`MUST_ANSWER`: a gate, not
                # an informational count.
                stale_allowlist.append(
                    (name, f"GAP_ALLOWLIST says run=ok/jit=gap, but both refuse: "
                           f"run={rs}:{rv[:60]!r} jit={js}:{jv[:60]!r}"))
                verdict = f"STALE-ALLOWLIST: both-fail, run={rs} jit={js}"
            else:
                both_fail += 1
                verdict = f"both-fail (run={rs} jit={js})"
        else:
            # run-only-ok with a non-gap jit error, or jit-only-ok — suspicious.
            diverged.append(name)
            verdict = f"DIVERGE: run={rs}:{rv!r} jit={js}:{jv!r}"
        if (args.verbose or "DIVERGE" in verdict or verdict.startswith("MUST-LOWER")
                or verdict.startswith("STALE-ALLOWLIST")
                or (verdict.startswith("jit-gap") and name not in GAP_ALLOWLIST)):
            print(f"  {name:22} {verdict}")

    print(f"\njit_probes: {ok} ok, {gaps} jit-gap, {refusals} interp-refusal, "
            f"{both_fail} both-fail, "
            f"{len(diverged)} divergence(s), {len(new_gaps)} untracked gap(s), "
            f"{len(must_lower)} must-lower failure(s), "
            f"{len(must_answer)} must-answer failure(s), "
            f"{len(stale_allowlist)} stale-allowlist entrie(s)")
    for name, msg in new_gaps:
        print(f"  {name}: warning: untracked jit-gap — fix or add to GAP_ALLOWLIST: {msg}")
    for name, msg in must_lower:
        print(f"  {name}: error: MUST_LOWER, but the JIT did not lower it — {msg}")
    for name, msg in must_answer:
        print(f"  {name}: error: MUST_ANSWER ({MUST_ANSWER[name]}), but a backend "
                f"did not answer — {msg}")
    for name, msg in stale_allowlist:
        print(f"  {name}: error: GAP_ALLOWLIST entry is dead — {msg}")
    for name in diverged:
        print(f"  {name}: error: run/jit divergence")
    # #802: best-effort cleanup of the file_io Forge-charge probes' real
    # fixture — see FORGE_FILE_IO_FIXTURE's own comment for why it can't
    # live inside a probe's own `TemporaryDirectory` (self-deleting on `run()`
    # return, before the fixture could be reused across the below_min/
    # at_jit_min pair). Not fatal if it's already gone or was never created.
    try:
        os.remove(FORGE_FILE_IO_FIXTURE)
    except FileNotFoundError:
        pass
    return 1 if diverged or must_lower or must_answer or stale_allowlist else 0


if __name__ == "__main__":
    sys.exit(main())
