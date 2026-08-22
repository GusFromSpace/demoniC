//! #326 — GPU/Metal backend for the bf16 decode GEMV.
//!
//! This module is compiled only on macOS with the `gpu` feature enabled
//! (`cargo build --features gpu`). It provides two drop-in runtime builtins,
//! selected at lowering time when `--gpu` is passed:
//!
//!   * `dmc_matmul_bf16_gpu` — the single m==1 decode GEMV (same ABI as the
//!     CPU `dmc_matmul_bf16`).
//!   * `dmc_matmul_bf16_gpu_batched` — a whole batched projection
//!     (`[B,1,S,K] @ [H,K,N]`-style) in ONE command buffer with ONE dispatch:
//!     the JIT collapses its per-slice batch loop into a single call when the
//!     slice offsets are affine in the flat batch index (they are for all the
//!     decode projections). This is what makes the many small per-head
//!     attention GEMVs worth offloading at all.
//!
//! Design (validated by the phase-0/1 spike and the `gpu_dispatch_bench`
//! micro-benchmark in this file; measured on Apple M5):
//!   * Weights (the bf16 RHS) are uploaded once and **memoized** keyed on a
//!     content fingerprint, so per-token weight transfer is zero — the model's
//!     weights are constant across decode steps. A batched projection's whole
//!     weight span (all heads) is one resident buffer.
//!   * Activation (`l`) and output (`out`) are small (m==1) and cross per call.
//!   * Command-buffer round-trip overhead is ~16us — negligible against the
//!     300-800us kernels. The measured costs that actually matter are kernel
//!     bandwidth and a fixed per-dispatch ramp (~60-190us), which is why the
//!     batched entry packs a whole projection into one big dispatch instead of
//!     many small ones.
//!   * `wait_until_completed` beats spin-polling the status here: a spinning
//!     CPU core competes with the GPU for the shared power budget on a
//!     fanless machine (measured: spin made real kernels ~25% slower).
//!   * Kernels: `vec4cb` (bfloat4 coalesced loads, 8-way contiguous-chunk
//!     K-split, batch index on grid z) for the N % 4 == 0 shapes — all of the
//!     model's — and `tiled2d` as the any-N fallback.
//!
//! The kernels accumulate in f32 from in-register bf16 upconvert, matching the
//! CPU kernel's math; results agree within f32 reduction-order tolerance and
//! decode tokens are verified identical to the CPU path (#320, #326).

use metal::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::time::Instant;

/// Per-phase wall-clock accounting for the GPU path, enabled by
/// `DMC_GPU_TIME=1`. Accumulates across calls and prints one line per decode
/// step, delimited by the lm-head GEMV (the only n >= 100k dispatch, which
/// closes every step).
#[derive(Default)]
struct GpuStats {
    calls: u64,
    x_us: u64,
    fp_us: u64,
    enc_us: u64,
    wait_us: u64,
    rb_us: u64,
    /// GPU-timestamp view of the waits: sum over calls of the span from the
    /// first part's GPUStartTime to the last part's GPUEndTime, and the sum
    /// of per-part execution times (span < sum means the parts overlapped).
    gspan_us: u64,
    gsum_us: u64,
    /// gap anatomy: commit -> first GPUStartTime, and last GPUEndTime -> CPU
    /// wake (same CACurrentMediaTime domain via CLOCK_UPTIME_RAW)
    sched_us: u64,
    wake_us: u64,
    step_start: Option<Instant>,
    /// per-(k, n, batch) wait-time buckets for the step line
    by_shape: Vec<((usize, usize, usize), u64, u64)>,
}

impl GpuStats {
    fn step_line(&mut self) {
        let total = self.step_start.map(|t| t.elapsed().as_micros() as u64).unwrap_or(0);
        eprintln!(
            "[gpu] step: calls={} x={}us fp={}us enc={}us wait={}us (gpu span={}us sum={}us sched={}us wake={}us) rb={}us | span={}us",
            self.calls, self.x_us, self.fp_us, self.enc_us, self.wait_us,
            self.gspan_us, self.gsum_us, self.sched_us, self.wake_us, self.rb_us, total
        );
        for &((k, n, batch), cnt, us) in &self.by_shape {
            let bytes = cnt as f64 * (k * n * batch * 2) as f64;
            eprintln!("[gpu]   k={k} n={n} b={batch}: {cnt} calls {us}us  {:.1} GB/s",
                bytes / 1e9 / (us as f64 / 1e6));
        }
        *self = GpuStats::default();
    }

    fn bucket(&mut self, k: usize, n: usize, batch: usize, us: u64) {
        let key = (k, n, batch);
        if let Some(e) = self.by_shape.iter_mut().find(|e| e.0 == key) {
            e.1 += 1;
            e.2 += us;
        } else {
            self.by_shape.push((key, 1, us));
        }
    }
}

const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Batched GEMV, the workhorse: 4 output columns per thread via bfloat4 loads
// (a simdgroup covers 128 consecutive columns = 256 contiguous bytes per K
// row), threadgroup of (32 col4-groups x TK K-partitions), each K-partition
// walking a CONTIGUOUS row chunk, partials combined through threadgroup
// memory. grid.z is the batch slice: x advances LS floats, w advances RS
// bfloats, y advances N floats per slice (m == 1). Requires N % 4 == 0 and
// RS % 4 == 0 (bfloat4 / float4 alignment); the host checks.
//
// NOFF is the absolute starting column: the host splits one projection's
// columns (or its batch, via buffer offsets) across 2-3 command buffers so
// they pipeline — a lone synchronous command buffer pays a ~190us GPU
// fill/drain bubble that overlapping in-flight buffers hide (measured: one
// 50 MB GEMV/cb sync = 85 GB/s, two+ in flight = 120 GB/s).
kernel void gemv_bf16_vec4cb(
    device const float*  x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device float*        y [[buffer(2)]],
    constant uint& K       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    constant uint& LS      [[buffer(5)]],
    constant uint& RS      [[buffer(6)]],
    constant uint& NOFF    [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    const uint TK = 8u;
    device const float*  xz = x + gid.z * LS;
    device const bfloat* wz = w + gid.z * RS;
    device float*        yz = y + gid.z * N;
    uint n0 = NOFF + gid.x * 4u;
    uint lx = lid.x;
    uint ly = lid.y;
    threadgroup float4 red[32u * TK];
    float4 acc = float4(0.0f);
    uint chunk = (K + TK - 1u) / TK;
    uint k0 = ly * chunk;
    uint k1 = min(K, k0 + chunk);
    if (n0 + 3u < N) {
        for (uint k = k0; k < k1; ++k) {
            float xv = xz[k];
            bfloat4 wv = *((device const bfloat4*)(wz + k * N + n0));
            acc += xv * float4(wv);
        }
    }
    red[lx * TK + ly] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ly == 0u && n0 + 3u < N) {
        float4 s = float4(0.0f);
        for (uint j = 0; j < TK; ++j) s += red[lx * TK + j];
        *((device float4*)(yz + n0)) = s;
    }
}

// Any-N fallback GEMV: threadgroup of 32 columns x 8 K-partitions. The 32
// lanes walk consecutive columns (coalesced weight loads); the 8-deep dim
// splits the K reduction for parallelism, combined through threadgroup memory.
kernel void gemv_bf16_tiled2d(
    device const float*  x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device float*        y [[buffer(2)]],
    constant uint& K       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    const uint TK = 8u;
    uint n  = gid.x;
    uint lx = lid.x;
    uint ly = lid.y;
    threadgroup float red[32u * 8u];
    float acc = 0.0f;
    if (n < N) {
        for (uint k = ly; k < K; k += TK) acc += x[k] * float(w[k * N + n]);
    }
    red[lx * TK + ly] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ly == 0u && n < N) {
        float s = 0.0f;
        for (uint j = 0; j < TK; ++j) s += red[lx * TK + j];
        y[n] = s;
    }
}
"#;

/// Process-thread-local GPU context. Decode runs single-threaded (the JIT'd
/// `main` calls this builtin serially), mirroring the `FORGE` arena's
/// thread-local model, which also keeps the non-Send Metal handles off any
/// shared state.
struct GpuCtx {
    device: Device,
    queue: CommandQueue,
    pso_vec4cb: ComputePipelineState,
    pso_tiled2d: ComputePipelineState,
    /// Resident bf16 weight buffers, keyed on a content fingerprint rather than
    /// the pointer: the model re-materializes each weight view at a *fresh* forge
    /// address every decode step, but the bytes are identical, so a content key
    /// hits across steps where an address key never would (and would grow
    /// unbounded -> OOM). See `fingerprint`.
    weights: HashMap<(u64, u64), Buffer>,
    /// reused scratch for the activation vector and output, grown on demand.
    xbuf: Buffer,
    xcap: usize,
    ybuf: Buffer,
    ycap: usize,
    /// #326 dispatch pipelining — deferred (committed, un-awaited) calls:
    /// their command buffers, and their host readbacks as (scratch src, forge
    /// dst, f32 elems). Materialized by the next sync call or explicit flush;
    /// the JIT guarantees nothing reads a deferred output before then, and
    /// `overlaps_pending` double-checks the pointers on every call.
    pending_cbs: Vec<CommandBuffer>,
    pending_rb: Vec<(*const f32, *mut f32, usize)>,
    /// Scratch bump offsets (bytes) while deferred work is outstanding — each
    /// deferred call keeps its own x/y slices alive. Reset at flush.
    x_off: usize,
    y_off: usize,
    /// Scratch buffers replaced by growth while possibly still referenced by
    /// pending command buffers; freed at flush.
    retired: Vec<Buffer>,
    /// `DMC_GPU_TIME=1` per-phase timing (see `GpuStats`).
    timing: bool,
    stats: GpuStats,
}

impl GpuCtx {
    fn new() -> Option<GpuCtx> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();
        let lib = device
            .new_library_with_source(KERNEL_SRC, &CompileOptions::new())
            .ok()?;
        let mk = |name: &str| -> Option<ComputePipelineState> {
            let f = lib.get_function(name, None).ok()?;
            device.new_compute_pipeline_state_with_function(&f).ok()
        };
        let pso_vec4cb = mk("gemv_bf16_vec4cb")?;
        let pso_tiled2d = mk("gemv_bf16_tiled2d")?;
        // Untracked scratch (see `SCRATCH`): one call's 2-3 split command
        // buffers write disjoint slices of ybuf and must not be serialized by
        // whole-buffer hazard tracking; the CPU only touches the scratch
        // after waiting on all of them (and only between calls).
        let xbuf = device.new_buffer(64 * 1024, Self::SCRATCH);
        let ybuf = device.new_buffer(64 * 1024, Self::SCRATCH);
        Some(GpuCtx {
            device, queue, pso_vec4cb, pso_tiled2d,
            weights: HashMap::new(),
            xbuf, xcap: 64 * 1024,
            ybuf, ycap: 64 * 1024,
            pending_cbs: Vec::new(),
            pending_rb: Vec::new(),
            x_off: 0,
            y_off: 0,
            retired: Vec::new(),
            timing: std::env::var("DMC_GPU_TIME").is_ok(),
            stats: GpuStats::default(),
        })
    }

    /// Resident bf16 weight buffer for pointer `r` (`elems` u16 values),
    /// uploaded once on first sight and reused thereafter.
    fn weight(&mut self, r: usize, bytes: usize) -> &Buffer {
        let key = fingerprint(r as *const u8, bytes);
        // Safety valve: the size gate keeps only the model's big static weights
        // here, but guard against a pathological dynamic-big matmul filling
        // memory.
        if self.weights.len() > 2048 { self.weights.clear(); }
        let dev = &self.device;
        let n = self.weights.len();
        self.weights.entry(key).or_insert_with(|| {
            if std::env::var("DMC_GPU_DEBUG").is_ok() {
                eprintln!("[gpu] upload #{} {} MB", n + 1, bytes >> 20);
            }
            // Untracked like the scratch: written once here (CPU-side, before
            // any GPU use) and read-only forever after, and whole-buffer
            // hazard tracking would serialize the split command buffers that
            // read different slices of one projection's weights.
            dev.new_buffer_with_data(
                r as *const c_void, bytes as u64, GpuCtx::SCRATCH)
        })
    }

    const SCRATCH: MTLResourceOptions = MTLResourceOptions::from_bits_retain(
        MTLResourceOptions::StorageModeShared.bits()
            | MTLResourceOptions::HazardTrackingModeUntracked.bits());

    fn ensure_x(&mut self, bytes: usize) {
        if bytes > self.xcap {
            let old = std::mem::replace(
                &mut self.xbuf, self.device.new_buffer(bytes as u64, Self::SCRATCH));
            self.retired.push(old); // may still be referenced by pending cbs
            self.xcap = bytes;
        }
    }
    fn ensure_y(&mut self, bytes: usize) {
        if bytes > self.ycap {
            let old = std::mem::replace(
                &mut self.ybuf, self.device.new_buffer(bytes as u64, Self::SCRATCH));
            self.retired.push(old);
            self.ycap = bytes;
        }
    }

    /// Materialize all deferred work: wait for the outstanding command
    /// buffers, perform their host readbacks, release retired scratch, and
    /// reset the scratch bump offsets. No-op when nothing is pending.
    fn flush_pending(&mut self) {
        for cb in &self.pending_cbs { cb.wait_until_completed(); }
        for &(src, dst, elems) in &self.pending_rb {
            unsafe { std::ptr::copy_nonoverlapping(src, dst, elems); }
        }
        self.pending_cbs.clear();
        self.pending_rb.clear();
        self.retired.clear();
        self.x_off = 0;
        self.y_off = 0;
    }

    /// Does [p, p+bytes) intersect any deferred output's destination range?
    fn overlaps_pending(&self, p: usize, bytes: usize) -> bool {
        self.pending_rb.iter().any(|&(_, dst, elems)| {
            let d0 = dst as usize;
            p < d0 + elems * 4 && d0 < p + bytes
        })
    }
}

thread_local! {
    /// `Some(ctx)` once GPU init succeeds; `None` if it failed (then we fall
    /// back to the CPU kernel on every call). `RefCell` for interior mutability
    /// of the weight memo / scratch buffers.
    static GPU: RefCell<Option<Option<GpuCtx>>> = const { RefCell::new(None) };
}

/// Content fingerprint of a bf16 weight buffer: byte length plus a 256-point
/// strided sample, folded into a 128-bit key with two independent mixes. Stable
/// across decode steps (same bytes -> same key even at a new address) and, with
/// length + 256 samples, collision-free in practice across the model's ~few-
/// hundred distinct weights — including same-shape layers, which differ at every
/// sampled word. Cheap: ~256 word reads regardless of buffer size. A collision
/// would surface as a token divergence from the CPU path (which we test for).
#[inline]
fn fingerprint(ptr: *const u8, bytes: usize) -> (u64, u64) {
    let mut h1: u64 = 0xcbf2_9ce4_8422_2325 ^ bytes as u64;
    let mut h2: u64 = 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(bytes as u64 | 1);
    let words = bytes / 8;
    if words > 0 {
        let p = ptr as *const u64;
        let step = (words / 256).max(1);
        let mut i = 0usize;
        while i < words {
            let v = unsafe { p.add(i).read_unaligned() };
            h1 = (h1 ^ v).wrapping_mul(0x0000_0100_0000_01b3);
            h2 = (h2.wrapping_add(v)).rotate_left(27).wrapping_mul(0xff51_afd7_ed55_8ccd);
            i += step;
        }
    }
    (h1, h2)
}

/// f32-from-bf16 widen, identical to the CPU kernel's `bf16_bits_to_f32`.
#[inline]
fn bf16_to_f32(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }

/// CPU fallback bf16 GEMV (m rows), used when GPU init fails. Matches the
/// accumulation order of the JIT's `matmul_tile_bf16`.
unsafe fn cpu_bf16_gemm(a: *const f32, b: *const u16, c: *mut f32, m: usize, k: usize, n: usize) {
    for i in 0..m {
        let crow = c.add(i * n);
        for j in 0..n { *crow.add(j) = 0.0; }
        let arow = a.add(i * k);
        for kk in 0..k {
            let av = *arow.add(kk);
            let brow = b.add(kk * n);
            for j in 0..n {
                let cj = crow.add(j);
                *cj = av.mul_add(bf16_to_f32(*brow.add(j)), *cj);
            }
        }
    }
}

/// CPU fallback for the batched entry: per-slice `cpu_bf16_gemm` at the affine
/// stride offsets.
unsafe fn cpu_bf16_gemm_batched(
    a: *const f32, b: *const u16, c: *mut f32,
    k: usize, n: usize, batch: usize, ls: usize, rs: usize,
) {
    for z in 0..batch {
        cpu_bf16_gemm(a.add(z * ls), b.add(z * rs), c.add(z * n), 1, k, n);
    }
}

/// GPU bf16 GEMV — drop-in for the CPU `dmc_matmul_bf16`. `l`/`out` are f32,
/// `r` is bf16 (u16). Selected at lowering time only for the `m == 1` decode
/// GEMV; any other `m` (defensively) takes the CPU path here too.
///
/// # Safety
/// `l`, `r`, `out` must point to `m*k` f32, `k*n` u16, and `m*n` f32
/// respectively (the JIT guarantees this at the call site).
pub extern "C" fn dmc_matmul_bf16_gpu(l: i64, r: i64, out: i64, m: i64, k: i64, n: i64) {
    let (m, k, n) = (m as usize, k as usize, n as usize);
    if m == 0 || n == 0 { return; }
    // Only the m==1 decode GEMV is offloaded; anything else uses the CPU kernel.
    if m != 1 {
        unsafe { cpu_bf16_gemm(l as *const f32, r as *const u16, out as *mut f32, m, k, n); }
        return;
    }
    gpu_run(l as *const f32, r as *const u16, out as *mut f32, k, n, 1, 0, 0, false);
}

/// Batched GPU bf16 GEMV — one call for a whole broadcast-batched projection
/// (`[.., 1, K] @ [.., K, N]`): `batch` m==1 slices whose lhs/rhs offsets are
/// affine in the flat batch index (`ls` f32 elems and `rs` u16 elems per
/// slice; the output is contiguous at `n` f32 per slice). The JIT emits this
/// instead of a per-slice loop when the (compile-time) offset mapping is
/// affine, so the whole projection is one command buffer + one 3D dispatch.
///
/// # Safety
/// `l`, `r`, `out` must cover `(batch-1)*ls + k` f32, `(batch-1)*rs + k*n`
/// u16, and `batch*n` f32 respectively (the JIT guarantees this).
pub extern "C" fn dmc_matmul_bf16_gpu_batched(
    l: i64, r: i64, out: i64, m: i64, k: i64, n: i64, batch: i64, ls: i64, rs: i64,
) {
    let (m, k, n, batch) = (m as usize, k as usize, n as usize, batch as usize);
    if m == 0 || n == 0 || batch == 0 { return; }
    debug_assert!(m == 1 && ls >= 0 && rs >= 0, "batched GEMV: JIT gate violated");
    let (ls, rs) = (ls as usize, rs as usize);
    if m != 1 {
        unsafe {
            for z in 0..batch {
                cpu_bf16_gemm(
                    (l as *const f32).add(z * ls), (r as *const u16).add(z * rs),
                    (out as *mut f32).add(z * m * n), m, k, n);
            }
        }
        return;
    }
    gpu_run(l as *const f32, r as *const u16, out as *mut f32, k, n, batch, ls, rs, false);
}

/// Deferred batched GEMV — identical ABI to `dmc_matmul_bf16_gpu_batched`,
/// but commits without waiting and without writing `out`: the work is
/// materialized by the next sync batched call or by
/// `dmc_matmul_bf16_gpu_flush`. The JIT emits this only for a statement whose
/// immediate successor is guaranteed to do one of those before anything can
/// read `out` — keeping the GPU queue non-empty across back-to-back
/// projections (a lone synchronous dispatch pays ~165us of queue-kickoff +
/// wake latency per call; measured 26ms + 15ms per decode step).
pub extern "C" fn dmc_matmul_bf16_gpu_batched_deferred(
    l: i64, r: i64, out: i64, m: i64, k: i64, n: i64, batch: i64, ls: i64, rs: i64,
) {
    let (m, k, n, batch) = (m as usize, k as usize, n as usize, batch as usize);
    if m == 0 || n == 0 || batch == 0 { return; }
    debug_assert!(m == 1 && ls >= 0 && rs >= 0, "deferred GEMV: JIT gate violated");
    let (ls, rs) = (ls as usize, rs as usize);
    if m != 1 {
        // unreachable from the JIT gate; run it synchronously
        dmc_matmul_bf16_gpu_batched(
            l, r, out, m as i64, k as i64, n as i64, batch as i64, ls as i64, rs as i64);
        return;
    }
    gpu_run(l as *const f32, r as *const u16, out as *mut f32, k, n, batch, ls, rs, true);
}

/// Materialize all deferred GPU work (see the deferred entry above). The JIT
/// emits this on any matmul lowering path that reads memory directly while a
/// deferred call may be outstanding.
pub extern "C" fn dmc_matmul_bf16_gpu_flush() {
    GPU.with(|cell| {
        if let Some(Some(ctx)) = cell.borrow_mut().as_mut() {
            ctx.flush_pending();
        }
    });
}

/// Shared GPU path: upload the activation span, look up (or upload) the
/// resident weight span, encode ONE command buffer with ONE dispatch covering
/// all `batch` slices, wait, and copy the contiguous result back.
fn gpu_run(a: *const f32, b: *const u16, c: *mut f32, k: usize, n: usize,
           batch: usize, ls: usize, rs: usize, defer: bool) {
    GPU.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() { *slot = Some(GpuCtx::new()); }
        let ctx = match slot.as_mut().unwrap().as_mut() {
            Some(ctx) => ctx,
            None => { unsafe { cpu_bf16_gemm_batched(a, b, c, k, n, batch, ls, rs); } return; }
        };
        let x_elems = (batch - 1) * ls + k;
        let y_elems = batch * n;
        let w_elems = (batch - 1) * rs + k * n;
        // Safety net under the JIT's deferral gate: if this call touches any
        // deferred output — reads it as activation or weight, or writes over
        // it — materialize the deferred work first.
        if ctx.overlaps_pending(a as usize, x_elems * 4)
            || ctx.overlaps_pending(b as usize, w_elems * 2)
            || ctx.overlaps_pending(c as usize, y_elems * 4)
        {
            ctx.flush_pending();
        }
        // The vec4cb kernel needs float4/bfloat4-aligned slices; the JIT gate
        // guarantees this for the batched call, and tiled2d covers batch == 1
        // with odd n. Anything else (unreachable from the JIT) goes CPU.
        let vec4_ok = n % 4 == 0 && rs % 4 == 0;
        if !vec4_ok && batch != 1 {
            unsafe { cpu_bf16_gemm_batched(a, b, c, k, n, batch, ls, rs); }
            return;
        }

        let timing = ctx.timing;
        let mut t = if timing {
            if ctx.stats.step_start.is_none() { ctx.stats.step_start = Some(Instant::now()); }
            Some(Instant::now())
        } else { None };
        let mut lap = |slot: &mut u64| {
            if let Some(ref mut t0) = t {
                let now = Instant::now();
                *slot += now.duration_since(*t0).as_micros() as u64;
                *t0 = now;
            }
        };

        // upload the activation span into reused scratch, at this call's bump
        // offset (deferred neighbors keep their own slices alive; 256-byte
        // alignment keeps every float4/bfloat4 slice offset legal)
        let x_base = (ctx.x_off + 255) & !255;
        let y_base = (ctx.y_off + 255) & !255;
        ctx.ensure_x(x_base + x_elems * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(
                a, ctx.xbuf.contents().cast::<u8>().add(x_base) as *mut f32, x_elems);
        }
        ctx.ensure_y(y_base + y_elems * 4);
        lap(&mut ctx.stats.x_us);
        let wbuf = ctx.weight(b as usize, w_elems * 2) as *const Buffer;
        lap(&mut ctx.stats.fp_us);

        let (ku, nu) = (k as u32, n as u32);
        let mut cbs: Vec<&CommandBufferRef> = Vec::with_capacity(3);
        if vec4_ok {
            // Pipeline the projection across 2-3 command buffers over
            // disjoint output slices (batch slices when batch > 1, column
            // ranges otherwise), committing each as it's encoded. A single
            // synchronous command buffer leaves a ~190us GPU fill/drain
            // bubble per call; overlapping in-flight buffers hide it
            // (measured 85 -> 120 GB/s on the FFN GEMV). The scratch buffers
            // are hazard-untracked so the parts actually run concurrently.
            let (lsu, rsu) = (ls as u32, rs as u32);
            let width = (n as u64).div_ceil(128);
            // measured on M5: wide grids (FFN gate/up, lm-head) take 3-way
            // splits well; narrow ones (down, the batched projections) lose
            // per-part occupancy beyond 2
            let parts = std::env::var("DMC_GPU_PARTS").ok()
                .and_then(|v| v.parse::<usize>().ok()).filter(|&p| (1..=8).contains(&p))
                .unwrap_or(if batch == 1 && width >= 32 { 3 } else { 2 });
            let (queue, pso, xb, yb) = (&ctx.queue, &ctx.pso_vec4cb, &ctx.xbuf, &ctx.ybuf);
            let encode = move |xo: u64, wo: u64, yo: u64, noff: u32,
                               pw: u64, pb: u64| -> &CommandBufferRef {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(pso);
                enc.set_buffer(0, Some(xb), xo);
                enc.set_buffer(1, Some(unsafe { &*wbuf }), wo);
                enc.set_buffer(2, Some(yb), yo);
                enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &lsu as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &rsu as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &noff as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize { width: pw, height: 1, depth: pb },
                    MTLSize { width: 32, height: 8, depth: 1 });
                enc.end_encoding();
                cb.commit();
                cb
            };
            if batch > 1 {
                let parts = parts.min(batch);
                for i in 0..parts {
                    let z0 = batch * i / parts;
                    let z1 = batch * (i + 1) / parts;
                    cbs.push(encode(
                        (x_base + z0 * ls * 4) as u64, (z0 * rs * 2) as u64,
                        (y_base + z0 * n * 4) as u64,
                        0, width, (z1 - z0) as u64));
                }
            } else {
                let parts = (parts as u64).min(width);
                for i in 0..parts {
                    let g0 = width * i / parts;
                    let g1 = width * (i + 1) / parts;
                    cbs.push(encode(x_base as u64, 0, y_base as u64,
                        (g0 * 128) as u32, g1 - g0, 1));
                }
            }
        } else {
            let cb = ctx.queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&ctx.pso_tiled2d);
            enc.set_buffer(0, Some(&ctx.xbuf), x_base as u64);
            enc.set_buffer(1, Some(unsafe { &*wbuf }), 0);
            enc.set_buffer(2, Some(&ctx.ybuf), y_base as u64);
            enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
            enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
            enc.dispatch_thread_groups(
                MTLSize { width: (n as u64).div_ceil(32), height: 1, depth: 1 },
                MTLSize { width: 32, height: 8, depth: 1 });
            enc.end_encoding();
            cb.commit();
            cbs.push(cb);
        }
        lap(&mut ctx.stats.enc_us);
        let y_src = unsafe { ctx.ybuf.contents().cast::<u8>().add(y_base) as *const f32 };

        if defer {
            // committed, not awaited: record the readback and keep the
            // scratch slices reserved; the next sync call (or flush)
            // materializes everything.
            ctx.pending_cbs.extend(cbs.iter().map(|cb| (*cb).to_owned()));
            ctx.pending_rb.push((y_src, c, y_elems));
            ctx.x_off = x_base + x_elems * 4;
            ctx.y_off = y_base + y_elems * 4;
            if timing { ctx.stats.calls += 1; }
            return;
        }

        let t_commit = if timing { host_time() } else { 0.0 };
        for cb in &cbs { cb.wait_until_completed(); }
        for cb in &ctx.pending_cbs { cb.wait_until_completed(); }
        let t_wake = if timing { host_time() } else { 0.0 };
        let mut w_us = 0u64;
        lap(&mut w_us);
        ctx.stats.wait_us += w_us;
        if timing {
            let (mut t0, mut t1, mut sum) = (f64::MAX, 0f64, 0f64);
            for cb in &cbs {
                let (s, e) = gpu_window(cb);
                t0 = t0.min(s);
                t1 = t1.max(e);
                sum += e - s;
            }
            ctx.stats.gspan_us += ((t1 - t0).max(0.0) * 1e6) as u64;
            ctx.stats.gsum_us += (sum.max(0.0) * 1e6) as u64;
            ctx.stats.sched_us += ((t0 - t_commit).max(0.0) * 1e6) as u64;
            ctx.stats.wake_us += ((t_wake - t1).max(0.0) * 1e6) as u64;
        }

        // materialize any deferred neighbors, then this call's result
        ctx.flush_pending();
        unsafe {
            std::ptr::copy_nonoverlapping(y_src, c, y_elems);
        }
        lap(&mut ctx.stats.rb_us);
        if timing {
            ctx.stats.calls += 1;
            ctx.stats.bucket(k, n, batch, w_us);
            // the lm-head GEMV (only n >= 100k dispatch) closes every decode step
            if n >= 100_000 { ctx.stats.step_line(); }
        }
    });
}

/// Current host time in the `GPUStartTime` clock domain (CACurrentMediaTime
/// == mach_absolute_time == CLOCK_UPTIME_RAW), seconds. Timing diagnostics
/// only.
fn host_time() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, &mut ts); }
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

/// A command buffer's GPU execution window `(GPUStartTime, GPUEndTime)` in
/// seconds — not exposed by metal-rs 0.33, so sent via objc_msgSend directly
/// (arm64: f64 returns go through plain objc_msgSend). Timing diagnostics
/// only.
fn gpu_window(cb: &&CommandBufferRef) -> (f64, f64) {
    use metal::foreign_types::ForeignTypeRef;
    use metal::objc::runtime::Sel;
    extern "C" {
        fn objc_msgSend();
    }
    unsafe {
        let send: unsafe extern "C" fn(*const c_void, Sel) -> f64 =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        let p = (*cb).as_ptr() as *const c_void;
        (send(p, Sel::register("GPUStartTime")), send(p, Sel::register("GPUEndTime")))
    }
}

/// Dispatch-cost micro-benchmark (`cargo test --release --features gpu
/// -- --ignored --nocapture gpu_dispatch_bench`). Measures, through the real
/// production entry points, the per-call cost and effective bandwidth of the
/// decode shapes, plus the raw Metal round-trip floor. Numbers in the module
/// doc-comment come from here.
#[cfg(test)]
mod bench {
    use super::*;

    #[test]
    #[ignore]
    fn gpu_dispatch_bench() {
        // raw command-buffer round trip
        {
            let ctx = GpuCtx::new().expect("no Metal device");
            let t0 = Instant::now();
            let reps = 500;
            for _ in 0..reps {
                let cb = ctx.queue.new_command_buffer();
                cb.commit();
                cb.wait_until_completed();
            }
            eprintln!("empty cb round trip: {:.1} us",
                t0.elapsed().as_micros() as f64 / reps as f64);
        }

        // fixed-cost anatomy: 9 distinct-weight FFN GEMVs — one cb serial
        // encoder, one cb concurrent encoder, 9 cbs wait-last
        {
            let ctx = GpuCtx::new().expect("no Metal device");
            let (k, n) = (2560usize, 9728usize);
            let wbytes = k * n * 2;
            let host: Vec<u16> = vec![0x3f80u16; wbytes / 2];
            let mats: Vec<Buffer> = (0..9)
                .map(|_| ctx.device.new_buffer_with_data(
                    host.as_ptr() as *const c_void, wbytes as u64,
                    MTLResourceOptions::StorageModeShared))
                .collect();
            let xb = ctx.device.new_buffer((k * 4) as u64, MTLResourceOptions::StorageModeShared);
            let yb = ctx.device.new_buffer((9 * n * 4) as u64, MTLResourceOptions::StorageModeShared);
            let (ku, nu, z) = (k as u32, n as u32, 0u32);
            let encode = |enc: &ComputeCommandEncoderRef, i: usize| {
                enc.set_compute_pipeline_state(&ctx.pso_vec4cb);
                enc.set_buffer(0, Some(&xb), 0);
                enc.set_buffer(1, Some(&mats[i]), 0);
                enc.set_buffer(2, Some(&yb), (i * n * 4) as u64);
                enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
                enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
                enc.set_bytes(5, 4, &z as *const u32 as *const c_void);
                enc.set_bytes(6, 4, &z as *const u32 as *const c_void);
                enc.set_bytes(7, 4, &z as *const u32 as *const c_void);
                enc.dispatch_thread_groups(
                    MTLSize { width: (n as u64).div_ceil(128), height: 1, depth: 1 },
                    MTLSize { width: 32, height: 8, depth: 1 });
            };
            for (label, mode) in [("serial", 0u8), ("concurrent", 1), ("9 cbs wait-last", 2)] {
                let mut go = || {
                    match mode {
                        2 => {
                            let mut last = None;
                            for i in 0..9 {
                                let cb = ctx.queue.new_command_buffer();
                                let enc = cb.new_compute_command_encoder();
                                encode(enc, i);
                                enc.end_encoding();
                                cb.commit();
                                last = Some(cb);
                            }
                            last.unwrap().wait_until_completed();
                        }
                        m => {
                            let cb = ctx.queue.new_command_buffer();
                            let enc = if m == 1 {
                                cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
                            } else {
                                cb.new_compute_command_encoder()
                            };
                            for i in 0..9 { encode(enc, i); }
                            enc.end_encoding();
                            cb.commit();
                            cb.wait_until_completed();
                        }
                    }
                };
                go();
                let reps = 30;
                let t0 = Instant::now();
                for _ in 0..reps { go(); }
                let us = t0.elapsed().as_micros() as f64 / reps as f64 / 9.0;
                eprintln!("9x ffn gemv, {label}: {us:.1} us/gemv, {:.1} GB/s",
                    wbytes as f64 / 1e9 / (us / 1e6));
            }
        }

        // production-path bandwidth per decode shape: (label, k, n, batch, ls, rs)
        let shapes: &[(&str, usize, usize, usize, usize, usize)] = &[
            ("q-proj  [32,2560,128]", 2560, 128, 32, 0, 2560 * 128),
            ("kv-proj [8,2560,128]", 2560, 128, 8, 0, 2560 * 128),
            ("wo-proj [32,128,2560]", 128, 2560, 32, 128, 128 * 2560),
            ("ffn     [2560,9728]", 2560, 9728, 1, 0, 0),
            ("lm-head [2560,151936]", 2560, 151936, 1, 0, 0),
        ];
        for &(label, k, n, batch, ls, rs) in shapes {
            let w_elems = (batch - 1) * rs + k * n;
            let x_elems = (batch - 1) * ls + k;
            let w: Vec<u16> = (0..w_elems).map(|i| 0x3f00 | (i as u16 & 0x7f)).collect();
            let x: Vec<f32> = (0..x_elems).map(|i| (i % 13) as f32 * 0.1).collect();
            let mut y: Vec<f32> = vec![0.0; batch * n];
            let mut run = || dmc_matmul_bf16_gpu_batched(
                x.as_ptr() as i64, w.as_ptr() as i64, y.as_mut_ptr() as i64,
                1, k as i64, n as i64, batch as i64, ls as i64, rs as i64);
            run(); // warm: uploads the weight span
            let reps = (200_000_000 / w_elems).clamp(20, 300);
            let t0 = Instant::now();
            for _ in 0..reps { run(); }
            let us = t0.elapsed().as_micros() as f64 / reps as f64;
            let gbs = (w_elems * 2) as f64 / 1e9 / (us / 1e6);
            eprintln!("{label}: {us:.1} us/call, {gbs:.1} GB/s");
        }
    }
}
