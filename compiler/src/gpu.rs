//! #326 — GPU/Metal backend for the bf16 decode GEMV.
//!
//! This module is compiled only on macOS with the `gpu` feature enabled
//! (`cargo build --features gpu`). It provides `dmc_matmul_bf16_gpu`, a
//! drop-in replacement for the CPU `dmc_matmul_bf16` runtime builtin, selected
//! at lowering time for the `m == 1` decode GEMV when `--gpu` is passed.
//!
//! Design (validated by the phase-0/1 spike, see #326):
//!   * Weights (the bf16 RHS) are uploaded once and **memoized** keyed on their
//!     pointer, so per-token weight transfer is zero — the model's weights are
//!     constant across decode steps. (forge pointers aren't page-aligned, so we
//!     copy-once into a Metal buffer rather than alias; true zero-copy via a
//!     page-aligned forge is a later optimization.)
//!   * Activation (`l`) and output (`out`) are small (m==1) and cross per call.
//!   * Two shape-adaptive kernels: `vec4` (bfloat4 coalesced loads) for very
//!     large N (lm-head), `tiled2d` (coalesced columns + threadgroup-mem
//!     K-reduction) for everything else. Measured ~31× over CPU on Apple M5.
//!
//! The kernels accumulate in f32 from in-register bf16 upconvert, matching the
//! CPU kernel's math; results agree within f32 reduction-order tolerance (#320).

use metal::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;

const KERNEL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Very-large-N GEMV (lm-head): one thread per 4 output columns, weights read as
// 8-byte bfloat4 loads. y[n] = sum_k x[k] * w[k*N + n]. All N here are mult of 4.
kernel void gemv_bf16_vec4(
    device const float*  x [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device float*        y [[buffer(2)]],
    constant uint& K       [[buffer(3)]],
    constant uint& N       [[buffer(4)]],
    uint nq [[thread_position_in_grid]])
{
    uint n0 = nq * 4u;
    if (n0 >= N) return;
    float4 acc = float4(0.0f);
    for (uint k = 0; k < K; ++k) {
        float xv = x[k];
        bfloat4 wv = *((device const bfloat4*)(w + k * N + n0));
        acc += xv * float4(wv);
    }
    *((device float4*)(y + n0)) = acc;
}

// General GEMV: threadgroup of 32 columns x 8 K-partitions. The 32 lanes walk
// consecutive columns (coalesced weight loads); the 8-deep dim splits the K
// reduction for parallelism, combined through threadgroup memory. Best for the
// large-K / medium-N projections (down, wo) that starve a one-thread-per-column
// kernel.
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
    pso_vec4: ComputePipelineState,
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
        let pso_vec4 = mk("gemv_bf16_vec4")?;
        let pso_tiled2d = mk("gemv_bf16_tiled2d")?;
        let xbuf = device.new_buffer(64 * 1024, MTLResourceOptions::StorageModeShared);
        let ybuf = device.new_buffer(64 * 1024, MTLResourceOptions::StorageModeShared);
        Some(GpuCtx {
            device, queue, pso_vec4, pso_tiled2d,
            weights: HashMap::new(),
            xbuf, xcap: 64 * 1024,
            ybuf, ycap: 64 * 1024,
        })
    }

    /// Resident bf16 weight buffer for pointer `r` (k*n u16 values), uploaded
    /// once on first sight and reused thereafter.
    fn weight(&mut self, r: usize, bytes: usize) -> &Buffer {
        let key = fingerprint(r as *const u8, bytes);
        // Safety valve: the size gate keeps only the ~109 big static weights here,
        // but guard against a pathological dynamic-big matmul filling memory.
        if self.weights.len() > 2048 { self.weights.clear(); }
        let dev = &self.device;
        let n = self.weights.len();
        self.weights.entry(key).or_insert_with(|| {
            if std::env::var("DMC_GPU_DEBUG").is_ok() {
                eprintln!("[gpu] upload #{} {} MB", n + 1, bytes >> 20);
            }
            dev.new_buffer_with_data(
                r as *const c_void, bytes as u64, MTLResourceOptions::StorageModeShared)
        })
    }

    fn ensure_x(&mut self, bytes: usize) {
        if bytes > self.xcap {
            self.xbuf = self.device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            self.xcap = bytes;
        }
    }
    fn ensure_y(&mut self, bytes: usize) {
        if bytes > self.ycap {
            self.ybuf = self.device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
            self.ycap = bytes;
        }
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
    let a = l as *const f32;
    let b = r as *const u16;
    let c = out as *mut f32;

    // Only the m==1 decode GEMV is offloaded; anything else uses the CPU kernel.
    if m != 1 {
        unsafe { cpu_bf16_gemm(a, b, c, m, k, n); }
        return;
    }

    GPU.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() { *slot = Some(GpuCtx::new()); }
        let ctx = match slot.as_mut().unwrap().as_mut() {
            Some(ctx) => ctx,
            None => { unsafe { cpu_bf16_gemm(a, b, c, 1, k, n); } return; }
        };

        // upload activation (k f32) into reused scratch
        ctx.ensure_x(k * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(a, ctx.xbuf.contents() as *mut f32, k);
        }
        ctx.ensure_y(n * 4);
        let wbuf = ctx.weight(r as usize, k * n * 2) as *const Buffer;

        // shape-adaptive: vec4 for very large N (lm-head), else 2D-tiled.
        let use_vec4 = n >= 32768;
        let (ku, nu) = (k as u32, n as u32);
        let cb = ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(if use_vec4 { &ctx.pso_vec4 } else { &ctx.pso_tiled2d });
        enc.set_buffer(0, Some(&ctx.xbuf), 0);
        enc.set_buffer(1, Some(unsafe { &*wbuf }), 0);
        enc.set_buffer(2, Some(&ctx.ybuf), 0);
        enc.set_bytes(3, 4, &ku as *const u32 as *const c_void);
        enc.set_bytes(4, 4, &nu as *const u32 as *const c_void);
        if use_vec4 {
            let threads = (n as u64).div_ceil(4);
            let tg = 256u64;
            enc.dispatch_thread_groups(
                MTLSize { width: threads.div_ceil(tg), height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 });
        } else {
            enc.dispatch_thread_groups(
                MTLSize { width: (n as u64).div_ceil(32), height: 1, depth: 1 },
                MTLSize { width: 32, height: 8, depth: 1 });
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // copy result back to the forge `out` buffer
        unsafe {
            std::ptr::copy_nonoverlapping(ctx.ybuf.contents() as *const f32, c, n);
        }
    });
}
