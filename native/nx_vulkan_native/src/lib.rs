//! Rustler NIF for Nx.Vulkan.
//!
//! Three layers down from Elixir:
//!
//!   Elixir  →  Rust NIF  →  extern "C" shim  →  C++ spirit::vulkan
//!
//! v0.0.1 wires bootstrap (init, device_name, has_f64).
//! v0.0.2 adds tensor lifetime + upload/download via ResourceArc.

use rustler::{Binary, Encoder, Env, Error, NifResult, OwnedBinary, ResourceArc, Term};
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::sync::Mutex;

mod atoms {
    rustler::atoms! {
        ok,
        error,
        no_device,
        alloc_failed,
        upload_failed,
        download_failed,
        size_mismatch,
        dispatch_failed,
        bad_op,
    }
}

// extern "C" declarations matching c_src/nx_vulkan_shim.h.
unsafe extern "C" {
    fn nxv_init() -> i32;
    fn nxv_device_name() -> *const c_char;
    fn nxv_has_f64() -> i32;

    fn nxv_buf_alloc(n_bytes: u64) -> *mut c_void;
    fn nxv_buf_free(handle: *mut c_void);
    fn nxv_buf_upload(handle: *mut c_void, data: *const c_void, n_bytes: u64) -> i32;
    fn nxv_buf_download(handle: *mut c_void, data: *mut c_void, n_bytes: u64) -> i32;

    fn nxv_apply_binary(
        out: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        n: u32,
        op: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_apply_unary(
        out: *mut c_void,
        a: *mut c_void,
        n: u32,
        op: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_reduce(
        out_scalar: *mut f32,
        input: *mut c_void,
        n: u32,
        op: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_matmul(
        out: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_random(
        out: *mut c_void,
        n: u32,
        seed: u32,
        dist: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_transpose(
        out: *mut c_void,
        a: *mut c_void,
        m: u32,
        n: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_cast(
        out: *mut c_void,
        a: *mut c_void,
        n: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_reduce_axis(
        out: *mut c_void,
        a: *mut c_void,
        outer: u32,
        reduce_size: u32,
        inner: u32,
        op: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_pool_clear();
    fn nxv_pool_stats(
        hits: *mut u64,
        misses: *mut u64,
        freed: *mut u64,
        size_classes: *mut u64,
        total_pooled: *mut u64,
    );

    // f64 elementwise — same C shim, different .spv path. Caller computes
    // n_elems and out_bytes per element width.

    // f64 reduce_axis and broadcast use the existing C shims unchanged
    // (the C side is type-opaque). Out-buffer sizes scale with element width.

    // logsumexp uses the same shim as reduce_axis (same push layout) but
    // is f32-only — output 4 bytes/element.

    fn nxv_matmul_v(
        out: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        m: u32,
        n: u32,
        k: u32,
        tile_m: u32,
        tile_n: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_apply_binary_broadcast(
        out: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        op: u32,
        ndim: u32,
        out_shape: *const u32,
        a_strides: *const u32,
        b_strides: *const u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_fused_chain(
        out: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        n: u32,
        n_ops: u32,
        ops: *const u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_fused_chain_4(
        out: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        c: *mut c_void,
        d: *mut c_void,
        n: u32,
        n_ops: u32,
        ops: *const u32,
        buf_idx: *const u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_kinetic_energy(
        out: *mut c_void,
        p: *mut c_void,
        inv_mass: *mut c_void,
        n: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_normal_logpdf(
        out: *mut c_void,
        x: *mut c_void,
        mu: *mut c_void,
        sigma: *mut c_void,
        n: u32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_normal(
        q_new: *mut c_void,
        p_new: *mut c_void,
        q: *mut c_void,
        p: *mut c_void,
        inv_mass: *mut c_void,
        n: u32,
        eps: f32,
        mu: f32,
        sigma: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_normal(
        q_chain: *mut c_void,
        p_chain: *mut c_void,
        grad_chain: *mut c_void,
        logp_chain: *mut c_void,
        q_init: *mut c_void,
        p_init: *mut c_void,
        inv_mass: *mut c_void,
        n: u32,
        K: u32,
        eps: f32,
        mu: f32,
        sigma: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_normal_lg(
        q_chain: *mut c_void,
        p_chain: *mut c_void,
        grad_chain: *mut c_void,
        partial_logp: *mut c_void,
        q_init: *mut c_void,
        p_init: *mut c_void,
        inv_mass: *mut c_void,
        n: u32,
        K: u32,
        num_workgroups: u32,
        eps: f32,
        mu: f32,
        sigma: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_exponential(
        q_chain: *mut c_void,
        p_chain: *mut c_void,
        grad_chain: *mut c_void,
        logp_chain: *mut c_void,
        q_init: *mut c_void,
        p_init: *mut c_void,
        inv_mass: *mut c_void,
        n: u32,
        K: u32,
        eps: f32,
        lambda: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_studentt(
        q_chain: *mut c_void, p_chain: *mut c_void,
        grad_chain: *mut c_void, logp_chain: *mut c_void,
        q_init: *mut c_void, p_init: *mut c_void, inv_mass: *mut c_void,
        n: u32, K: u32,
        eps: f32, mu: f32, sigma: f32, nu: f32, logp_const: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_cauchy(
        q_chain: *mut c_void, p_chain: *mut c_void,
        grad_chain: *mut c_void, logp_chain: *mut c_void,
        q_init: *mut c_void, p_init: *mut c_void, inv_mass: *mut c_void,
        n: u32, K: u32,
        eps: f32, loc: f32, scale: f32, log_pi_scale: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_halfnormal(
        q_chain: *mut c_void, p_chain: *mut c_void,
        grad_chain: *mut c_void, logp_chain: *mut c_void,
        q_init: *mut c_void, p_init: *mut c_void, inv_mass: *mut c_void,
        n: u32, K: u32,
        eps: f32, sigma: f32, log_const: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_weibull(
        q_chain: *mut c_void, p_chain: *mut c_void,
        grad_chain: *mut c_void, logp_chain: *mut c_void,
        q_init: *mut c_void, p_init: *mut c_void, inv_mass: *mut c_void,
        n: u32, K: u32,
        eps: f32, k: f32, lambda: f32, logp_const: f32,
        spv_path: *const c_char,
    ) -> i32;

    fn nxv_leapfrog_chain_normal_f64(
        q_chain: *mut c_void, p_chain: *mut c_void,
        grad_chain: *mut c_void, logp_chain: *mut c_void,
        q_init: *mut c_void, p_init: *mut c_void, inv_mass: *mut c_void,
        n: u32, K: u32,
        eps: f64, mu: f64, sigma: f64,
        spv_path: *const c_char,
    ) -> i32;
}

// One-shot guard so Elixir can call init/0 idempotently. Vulkan's
// vk_init is itself idempotent at the spirit level (returns 0 if
// already inited) but tracking the state in Rust gives us cleaner
// error semantics on the Elixir side.
static INIT_STATE: Mutex<bool> = Mutex::new(false);

// Global submit serializer.
//
// Vulkan's VkQueue is "externally synchronized" — the spec says concurrent
// vkQueueSubmit calls from multiple host threads to the same VkQueue is
// undefined behaviour. Spirit's compute backend uses a single global
// queue, so any pair of NIF calls that submit (dispatch, upload, download,
// reduce, matmul, random) must NOT run on different threads at the same
// time.
//
// Without this lock, a stress test with 100 concurrent processes
// reproducibly triggers VK_ERROR_DEVICE_LOST within seconds.
//
// This lock serializes the entire submit-and-wait, which costs us
// concurrency on the GPU — but Spirit's submit_and_wait is itself
// blocking (no async dispatch), so we lose nothing real. Async dispatch
// + multiple queues is a v0.2 optimization.
static SUBMIT_LOCK: Mutex<()> = Mutex::new(());

// VulkanTensor owns a heap-allocated VkBuf via the C++ shim. When the
// Elixir reference is GC'd, ResourceArc drops this struct, which
// frees the GPU buffer through nxv_buf_free.
//
// SAFETY: the underlying handle is a void* pointer to a heap C++
// object. Send/Sync are unsafe-impl'd because BEAM may move the
// resource between schedulers; the C++ side serializes Vulkan calls
// through the global compute queue, so concurrent access from
// multiple NIF threads is bounded by Vulkan's own synchronization.
pub struct VulkanTensor {
    handle: *mut c_void,
    n_bytes: u64,
}

unsafe impl Send for VulkanTensor {}
unsafe impl Sync for VulkanTensor {}

impl Drop for VulkanTensor {
    fn drop(&mut self) {
        unsafe { nxv_buf_free(self.handle) };
    }
}

#[rustler::nif]
fn init<'a>(env: Env<'a>) -> NifResult<Term<'a>> {
    let mut state = INIT_STATE.lock().map_err(|_| Error::BadArg)?;

    if *state {
        return Ok((atoms::ok()).encode(env));
    }

    let rc = unsafe { nxv_init() };
    if rc == 0 {
        *state = true;
        Ok((atoms::ok()).encode(env))
    } else {
        Ok((atoms::error(), atoms::no_device()).encode(env))
    }
}

#[rustler::nif]
fn device_name<'a>(env: Env<'a>) -> NifResult<Term<'a>> {
    let state = INIT_STATE.lock().map_err(|_| Error::BadArg)?;
    if !*state {
        return Ok((rustler::types::atom::nil()).encode(env));
    }

    let ptr = unsafe { nxv_device_name() };
    if ptr.is_null() {
        return Ok((rustler::types::atom::nil()).encode(env));
    }

    let cstr = unsafe { CStr::from_ptr(ptr) };
    let s = cstr.to_string_lossy().to_string();
    Ok(s.encode(env))
}

#[rustler::nif]
fn has_f64<'a>(env: Env<'a>) -> NifResult<Term<'a>> {
    let rc = unsafe { nxv_has_f64() };
    Ok((rc != 0).encode(env))
}

/// Upload an Elixir binary (raw bytes — typically packed f32) to a
/// freshly-allocated GPU buffer. Returns a ResourceArc wrapping the
/// VulkanTensor; when the Elixir reference is GC'd, the buffer is
/// freed automatically.
#[rustler::nif]
fn upload_binary<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Term<'a>> {
    let n_bytes = data.len() as u64;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let handle = unsafe { nxv_buf_alloc(n_bytes) };
    if handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let rc = unsafe {
        nxv_buf_upload(handle, data.as_slice().as_ptr() as *const c_void, n_bytes)
    };
    if rc != 0 {
        unsafe { nxv_buf_free(handle) };
        return Ok((atoms::error(), atoms::upload_failed()).encode(env));
    }

    let tensor = VulkanTensor { handle, n_bytes };
    let resource = ResourceArc::new(tensor);
    Ok((atoms::ok(), resource).encode(env))
}

/// Download `n_bytes` from a GPU tensor back into an Elixir binary.
/// `n_bytes` must match the buffer's size.
#[rustler::nif]
fn download_binary<'a>(
    env: Env<'a>,
    tensor: ResourceArc<VulkanTensor>,
    n_bytes: u64,
) -> NifResult<Term<'a>> {
    if n_bytes != tensor.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let mut bin = OwnedBinary::new(n_bytes as usize)
        .ok_or_else(|| Error::Term(Box::new("could not allocate Elixir binary")))?;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let rc = unsafe {
        nxv_buf_download(
            tensor.handle,
            bin.as_mut_slice().as_mut_ptr() as *mut c_void,
            n_bytes,
        )
    };
    if rc != 0 {
        return Ok((atoms::error(), atoms::download_failed()).encode(env));
    }

    let term = bin.release(env).encode(env);
    Ok((atoms::ok(), term).encode(env))
}

/// Returns the byte size of the tensor.
#[rustler::nif]
fn byte_size<'a>(env: Env<'a>, tensor: ResourceArc<VulkanTensor>) -> NifResult<Term<'a>> {
    Ok((tensor.n_bytes).encode(env))
}

/// Apply an elementwise binary op to two GPU tensors. Allocates the
/// output buffer (same byte_size as inputs) and dispatches the
/// elementwise_binary shader. Returns a new ResourceArc.
///
/// op spec constant: 0=add, 1=mul, 2=sub, 3=div, 4=pow, 5=max, 6=min.
#[rustler::nif]
fn apply_binary<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    op: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if a.n_bytes != b.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    // Op range bumped 6→9 in v0.1 phase 1.1 — equal/less/greater
    // added to elementwise_binary.spv. Update spirit's .comp + .spv
    // when adding more.
    if op > 9 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_bytes = a.n_bytes;
    let n_elems = (n_bytes / 4) as u32;     // f32 elements

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_apply_binary(out_handle, a.handle, b.handle, n_elems, op, cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Apply an elementwise unary op. Allocates a fresh output buffer.
#[rustler::nif]
fn apply_unary<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    op: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 14 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_bytes = a.n_bytes;
    let n_elems = (n_bytes / 4) as u32;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe { nxv_apply_unary(out_handle, a.handle, n_elems, op, cstr.as_ptr()) };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Reduction (sum/min/max). Returns a host-side f32 scalar.
#[rustler::nif]
fn reduce_scalar<'a>(
    env: Env<'a>,
    input: ResourceArc<VulkanTensor>,
    op: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 2 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_elems = (input.n_bytes / 4) as u32;
    let mut out_scalar: f32 = 0.0;
    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let rc = unsafe {
        nxv_reduce(&mut out_scalar as *mut f32, input.handle, n_elems, op, cstr.as_ptr())
    };

    if rc != 0 {
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    Ok((atoms::ok(), out_scalar).encode(env))
}

/// Matmul C[M*N] = A[M*K] · B[K*N]. Allocates output.
#[rustler::nif]
fn matmul<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    m: u32,
    n: u32,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let expected_a = (m * k * 4) as u64;
    let expected_b = (k * n * 4) as u64;
    if a.n_bytes != expected_a || b.n_bytes != expected_b {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let out_bytes = (m * n * 4) as u64;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe { nxv_matmul(out_handle, a.handle, b.handle, m, n, k, cstr.as_ptr()) };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Random fill. Allocates an output buffer of `n` f32 elements.
/// dist: 0=uniform [0,1), 1=normal N(0,1).
#[rustler::nif]
fn random<'a>(
    env: Env<'a>,
    n: u32,
    seed: u32,
    dist: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if dist > 1 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_bytes = (n * 4) as u64;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe { nxv_random(out_handle, n, seed, dist, cstr.as_ptr()) };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// 2D transpose. Input M×N row-major; output N×M row-major.
/// Allocates the output buffer (same byte_size as input).
#[rustler::nif]
fn transpose<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    m: u32,
    n: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let expected = (m * n * 4) as u64;
    if a.n_bytes != expected {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(a.n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe { nxv_transpose(out_handle, a.handle, m, n, cstr.as_ptr()) };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: a.n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Cast f32↔f64. The .spv file determines direction; output buffer
/// width derives from the destination type. n is element count.
#[rustler::nif]
fn cast<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    n: u32,
    out_elem_bytes: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let out_bytes = (n as u64) * (out_elem_bytes as u64);

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe { nxv_cast(out_handle, a.handle, n, cstr.as_ptr()) };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Per-axis reduction over a virtual 3-D layout (outer, reduce, inner).
/// Output is (outer, inner) row-major, n_out = outer * inner * 4 bytes (f32).
#[rustler::nif]
fn reduce_axis<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    outer: u32,
    reduce_size: u32,
    inner: u32,
    op: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 2 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_out = (outer as u64) * (inner as u64);
    let out_bytes = n_out * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe { nxv_reduce_axis(out_handle, a.handle, outer, reduce_size, inner, op, cstr.as_ptr()) };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Broadcast elementwise binary op. `out_shape`, `a_strides`,
/// `b_strides` are length-4 vectors padded with 0. A stride of 0
/// on an axis means broadcast (any coord on that axis maps to index 0).
#[rustler::nif]
fn apply_binary_broadcast<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    op: u32,
    ndim: u32,
    out_shape: Vec<u32>,
    a_strides: Vec<u32>,
    b_strides: Vec<u32>,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 9 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }
    if ndim == 0 || ndim > 4 || out_shape.len() != 4
       || a_strides.len() != 4 || b_strides.len() != 4 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n: u64 = (0..ndim as usize)
        .map(|d| out_shape[d] as u64)
        .product();
    let out_bytes = n * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_apply_binary_broadcast(
            out_handle, a.handle, b.handle,
            op, ndim,
            out_shape.as_ptr(), a_strides.as_ptr(), b_strides.as_ptr(),
            cstr.as_ptr(),
        )
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Path A — fused elementwise chain. `ops` is a Vec<u32> of length ≤8;
/// shorter chains are padded with 255 (nop). Op codes:
///   0..6   binary (add/mul/sub/div/pow/max/min)
///   100..114 unary (exp..expm1)
#[rustler::nif]
fn fused_chain<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    ops: Vec<u32>,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if ops.is_empty() || ops.len() > 8 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }
    if a.n_bytes != b.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let n = (a.n_bytes / 4) as u32;
    let n_ops = ops.len() as u32;
    let mut padded: [u32; 8] = [255; 8];
    for (i, &c) in ops.iter().enumerate() { padded[i] = c; }

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(a.n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_fused_chain(out_handle, a.handle, b.handle, n, n_ops, padded.as_ptr(), cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: a.n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// f64 reduce_axis. Output is (outer*inner) f64 (8 bytes/element).
#[rustler::nif]
fn reduce_axis_f64<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    outer: u32,
    reduce_size: u32,
    inner: u32,
    op: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 2 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_out = (outer as u64) * (inner as u64);
    let out_bytes = n_out * 8;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_reduce_axis(out_handle, a.handle, outer, reduce_size, inner, op, cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// f64 broadcast elementwise binary. Same shim as f32 broadcast.
#[rustler::nif]
fn apply_binary_broadcast_f64<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    op: u32,
    ndim: u32,
    out_shape: Vec<u32>,
    a_strides: Vec<u32>,
    b_strides: Vec<u32>,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 9 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }
    if ndim == 0 || ndim > 4 || out_shape.len() != 4
       || a_strides.len() != 4 || b_strides.len() != 4 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n: u64 = (0..ndim as usize)
        .map(|d| out_shape[d] as u64)
        .product();
    let out_bytes = n * 8;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_apply_binary_broadcast(
            out_handle, a.handle, b.handle,
            op, ndim,
            out_shape.as_ptr(), a_strides.as_ptr(), b_strides.as_ptr(),
            cstr.as_ptr(),
        )
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// logsumexp: numerically-stable two-pass on a single reduced axis.
/// Reuses nxv_reduce_axis's shim (same push layout); op is unused but
/// passed as 0 for parity. f32 only.
#[rustler::nif]
fn logsumexp<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    outer: u32,
    reduce_size: u32,
    inner: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let n_out = (outer as u64) * (inner as u64);
    let out_bytes = n_out * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_reduce_axis(out_handle, a.handle, outer, reduce_size, inner, 0, cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// f64 elementwise binary. Same op codes 0..6 as the f32 path; the
/// shader's binding type makes the precision choice.
#[rustler::nif]
fn apply_binary_f64<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    op: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 6 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }
    if a.n_bytes != b.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let n_elems = (a.n_bytes / 8) as u32;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(a.n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_apply_binary(out_handle, a.handle, b.handle, n_elems, op, cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: a.n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// f64 elementwise unary.
#[rustler::nif]
fn apply_unary_f64<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    op: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if op > 14 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_elems = (a.n_bytes / 8) as u32;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(a.n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe { nxv_apply_unary(out_handle, a.handle, n_elems, op, cstr.as_ptr()) };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: a.n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Matmul variant — caller picks the .spv path and the workgroup
/// output tile size. Used by Nx.Vulkan auto-select to dispatch the
/// right shader for a given (M, N, K) shape.
#[rustler::nif]
fn matmul_v<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    m: u32,
    n: u32,
    k: u32,
    tile_m: u32,
    tile_n: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let expected_a = (m * k * 4) as u64;
    let expected_b = (k * n * 4) as u64;
    if a.n_bytes != expected_a || b.n_bytes != expected_b {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let out_bytes = (m * n * 4) as u64;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_matmul_v(out_handle, a.handle, b.handle, m, n, k, tile_m, tile_n, cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// kinetic_energy: 0.5 * sum(p² * inv_mass) per workgroup.
/// Output is `ceil(n / 256)` partial sums (4 bytes each) — caller
/// reduces them on host or via a follow-up reduce_axis dispatch.
#[rustler::nif]
fn kinetic_energy<'a>(
    env: Env<'a>,
    p: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if p.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let n = (p.n_bytes / 4) as u32;
    let n_groups: u64 = ((n + 255) / 256) as u64;
    let out_bytes = n_groups * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_kinetic_energy(out_handle, p.handle, inv_mass.handle, n, cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// normal_logpdf: -0.5*((x-mu)/sigma)² - log(sigma) - 0.5*log(2π).
/// Output shape matches x.
#[rustler::nif]
fn normal_logpdf<'a>(
    env: Env<'a>,
    x: ResourceArc<VulkanTensor>,
    mu: ResourceArc<VulkanTensor>,
    sigma: ResourceArc<VulkanTensor>,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if x.n_bytes != mu.n_bytes || x.n_bytes != sigma.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let n = (x.n_bytes / 4) as u32;
    let out_bytes = x.n_bytes;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_normal_logpdf(out_handle, x.handle, mu.handle, sigma.handle, n, cstr.as_ptr())
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// leapfrog_normal: fused NUTS leapfrog step for univariate Normal.
/// Returns {q_new, p_new}. mu, sigma, eps come in as f32 push constants.
/// q, p, inv_mass must all share byte size (n elements × 4 bytes).
#[rustler::nif]
fn leapfrog_normal<'a>(
    env: Env<'a>,
    q: ResourceArc<VulkanTensor>,
    p: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    eps: f64,
    mu: f64,
    sigma: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q.n_bytes != p.n_bytes || q.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let n = (q.n_bytes / 4) as u32;
    let out_bytes = q.n_bytes;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let q_new_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if q_new_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let p_new_handle = unsafe { nxv_buf_alloc(out_bytes) };
    if p_new_handle.is_null() {
        unsafe { nxv_buf_free(q_new_handle) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_normal(
            q_new_handle,
            p_new_handle,
            q.handle,
            p.handle,
            inv_mass.handle,
            n,
            eps as f32,
            mu as f32,
            sigma as f32,
            cstr.as_ptr(),
        )
    };

    if rc != 0 {
        unsafe { nxv_buf_free(q_new_handle) };
        unsafe { nxv_buf_free(p_new_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let q_new = VulkanTensor { handle: q_new_handle, n_bytes: out_bytes };
    let p_new = VulkanTensor { handle: p_new_handle, n_bytes: out_bytes };
    Ok((atoms::ok(), (ResourceArc::new(q_new), ResourceArc::new(p_new))).encode(env))
}

/// leapfrog_chain_normal: K-step fused leapfrog chain for univariate Normal.
/// Returns {q_chain, p_chain, grad_chain, logp_chain}. All four are
/// allocated by this NIF; q/p/grad chains are K*n*4 bytes each, logp_chain
/// is K*4 bytes. K must be a positive u32; n is derived from input byte size.
#[rustler::nif]
fn leapfrog_chain_normal<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64,
    mu: f64,
    sigma: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n = (q_init.n_bytes / 4) as u32;
    let chain_bytes = q_init.n_bytes * (k as u64);   // K * n * 4
    let logp_bytes  = (k as u64) * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let q_chain_handle = unsafe { nxv_buf_alloc(chain_bytes) };
    if q_chain_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let p_chain_handle = unsafe { nxv_buf_alloc(chain_bytes) };
    if p_chain_handle.is_null() {
        unsafe { nxv_buf_free(q_chain_handle) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let grad_chain_handle = unsafe { nxv_buf_alloc(chain_bytes) };
    if grad_chain_handle.is_null() {
        unsafe { nxv_buf_free(q_chain_handle) };
        unsafe { nxv_buf_free(p_chain_handle) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let logp_chain_handle = unsafe { nxv_buf_alloc(logp_bytes) };
    if logp_chain_handle.is_null() {
        unsafe { nxv_buf_free(q_chain_handle) };
        unsafe { nxv_buf_free(p_chain_handle) };
        unsafe { nxv_buf_free(grad_chain_handle) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_normal(
            q_chain_handle,
            p_chain_handle,
            grad_chain_handle,
            logp_chain_handle,
            q_init.handle,
            p_init.handle,
            inv_mass.handle,
            n,
            k,
            eps as f32,
            mu as f32,
            sigma as f32,
            cstr.as_ptr(),
        )
    };

    if rc != 0 {
        unsafe { nxv_buf_free(q_chain_handle) };
        unsafe { nxv_buf_free(p_chain_handle) };
        unsafe { nxv_buf_free(grad_chain_handle) };
        unsafe { nxv_buf_free(logp_chain_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let q_chain    = VulkanTensor { handle: q_chain_handle,    n_bytes: chain_bytes };
    let p_chain    = VulkanTensor { handle: p_chain_handle,    n_bytes: chain_bytes };
    let grad_chain = VulkanTensor { handle: grad_chain_handle, n_bytes: chain_bytes };
    let logp_chain = VulkanTensor { handle: logp_chain_handle, n_bytes: logp_bytes  };

    Ok((
        atoms::ok(),
        (
            ResourceArc::new(q_chain),
            ResourceArc::new(p_chain),
            ResourceArc::new(grad_chain),
            ResourceArc::new(logp_chain),
        ),
    )
        .encode(env))
}

/// leapfrog_chain_normal_lg: multi-workgroup K-step chain.
/// Returns {q_chain, p_chain, grad_chain, partial_logp}. partial_logp is
/// K * num_workgroups f32 (host sums num_workgroups partials per step).
#[rustler::nif]
fn leapfrog_chain_normal_lg<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64,
    mu: f64,
    sigma: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n = (q_init.n_bytes / 4) as u32;
    let num_workgroups = (n + 255) / 256;
    let chain_bytes   = q_init.n_bytes * (k as u64);
    let partial_bytes = (k as u64) * (num_workgroups as u64) * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let q_h = unsafe { nxv_buf_alloc(chain_bytes) };
    if q_h.is_null() { return Ok((atoms::error(), atoms::alloc_failed()).encode(env)); }
    let p_h = unsafe { nxv_buf_alloc(chain_bytes) };
    if p_h.is_null() {
        unsafe { nxv_buf_free(q_h) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let g_h = unsafe { nxv_buf_alloc(chain_bytes) };
    if g_h.is_null() {
        unsafe { nxv_buf_free(q_h); nxv_buf_free(p_h) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let pl_h = unsafe { nxv_buf_alloc(partial_bytes) };
    if pl_h.is_null() {
        unsafe { nxv_buf_free(q_h); nxv_buf_free(p_h); nxv_buf_free(g_h) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_normal_lg(
            q_h, p_h, g_h, pl_h,
            q_init.handle, p_init.handle, inv_mass.handle,
            n, k, num_workgroups,
            eps as f32, mu as f32, sigma as f32,
            cstr.as_ptr(),
        )
    };
    if rc != 0 {
        unsafe { nxv_buf_free(q_h); nxv_buf_free(p_h);
                 nxv_buf_free(g_h); nxv_buf_free(pl_h) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let q_c  = VulkanTensor { handle: q_h,  n_bytes: chain_bytes };
    let p_c  = VulkanTensor { handle: p_h,  n_bytes: chain_bytes };
    let g_c  = VulkanTensor { handle: g_h,  n_bytes: chain_bytes };
    let pl_c = VulkanTensor { handle: pl_h, n_bytes: partial_bytes };

    Ok((
        atoms::ok(),
        (
            ResourceArc::new(q_c),
            ResourceArc::new(p_c),
            ResourceArc::new(g_c),
            ResourceArc::new(pl_c),
        ),
    )
        .encode(env))
}

/// leapfrog_chain_exponential: K-step chain for Exp(lambda) on the
/// unconstrained line (log-transform). Same I/O shape as the Normal
/// chain (returns {q, p, grad, logp} 4-tuple).
#[rustler::nif]
fn leapfrog_chain_exponential<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64,
    lambda: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n = (q_init.n_bytes / 4) as u32;
    let chain_bytes = q_init.n_bytes * (k as u64);
    let logp_bytes  = (k as u64) * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let q_h = unsafe { nxv_buf_alloc(chain_bytes) };
    if q_h.is_null() { return Ok((atoms::error(), atoms::alloc_failed()).encode(env)); }
    let p_h = unsafe { nxv_buf_alloc(chain_bytes) };
    if p_h.is_null() {
        unsafe { nxv_buf_free(q_h) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let g_h = unsafe { nxv_buf_alloc(chain_bytes) };
    if g_h.is_null() {
        unsafe { nxv_buf_free(q_h); nxv_buf_free(p_h) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }
    let lc_h = unsafe { nxv_buf_alloc(logp_bytes) };
    if lc_h.is_null() {
        unsafe { nxv_buf_free(q_h); nxv_buf_free(p_h); nxv_buf_free(g_h) };
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_exponential(
            q_h, p_h, g_h, lc_h,
            q_init.handle, p_init.handle, inv_mass.handle,
            n, k,
            eps as f32, lambda as f32,
            cstr.as_ptr(),
        )
    };
    if rc != 0 {
        unsafe { nxv_buf_free(q_h); nxv_buf_free(p_h);
                 nxv_buf_free(g_h); nxv_buf_free(lc_h) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let q_c  = VulkanTensor { handle: q_h,  n_bytes: chain_bytes };
    let p_c  = VulkanTensor { handle: p_h,  n_bytes: chain_bytes };
    let g_c  = VulkanTensor { handle: g_h,  n_bytes: chain_bytes };
    let lc_c = VulkanTensor { handle: lc_h, n_bytes: logp_bytes };

    Ok((
        atoms::ok(),
        (
            ResourceArc::new(q_c),
            ResourceArc::new(p_c),
            ResourceArc::new(g_c),
            ResourceArc::new(lc_c),
        ),
    )
        .encode(env))
}

// --- Phase 2 chain NIFs (Student-t, Cauchy, HalfNormal) + f64 chain ---
//
// All four follow the same allocate-4-output-buffers / dispatch /
// return-4-tuple pattern as leapfrog_chain_exponential. The differences
// are the push-constant scalars and the underlying nxv_* dispatch.
// f64 chain uses 8 bytes per element instead of 4.

#[rustler::nif]
fn leapfrog_chain_studentt<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64, mu: f64, sigma: f64, nu: f64, logp_const: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 { return Ok((atoms::error(), atoms::bad_op()).encode(env)); }

    let n = (q_init.n_bytes / 4) as u32;
    let chain_bytes = q_init.n_bytes * (k as u64);
    let logp_bytes = (k as u64) * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;
    let (qh, ph, gh, lh) = match alloc_4(chain_bytes, chain_bytes, chain_bytes, logp_bytes) {
        Ok(t) => t,
        Err(_) => return Ok((atoms::error(), atoms::alloc_failed()).encode(env)),
    };
    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_studentt(
            qh, ph, gh, lh,
            q_init.handle, p_init.handle, inv_mass.handle,
            n, k,
            eps as f32, mu as f32, sigma as f32, nu as f32, logp_const as f32,
            cstr.as_ptr(),
        )
    };
    encode_chain_result(env, rc, qh, ph, gh, lh, chain_bytes, logp_bytes)
}

#[rustler::nif]
fn leapfrog_chain_cauchy<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64, loc: f64, scale: f64, log_pi_scale: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 { return Ok((atoms::error(), atoms::bad_op()).encode(env)); }

    let n = (q_init.n_bytes / 4) as u32;
    let chain_bytes = q_init.n_bytes * (k as u64);
    let logp_bytes = (k as u64) * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;
    let (qh, ph, gh, lh) = match alloc_4(chain_bytes, chain_bytes, chain_bytes, logp_bytes) {
        Ok(t) => t,
        Err(_) => return Ok((atoms::error(), atoms::alloc_failed()).encode(env)),
    };
    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_cauchy(
            qh, ph, gh, lh,
            q_init.handle, p_init.handle, inv_mass.handle,
            n, k,
            eps as f32, loc as f32, scale as f32, log_pi_scale as f32,
            cstr.as_ptr(),
        )
    };
    encode_chain_result(env, rc, qh, ph, gh, lh, chain_bytes, logp_bytes)
}

#[rustler::nif]
fn leapfrog_chain_halfnormal<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64, sigma: f64, log_const: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 { return Ok((atoms::error(), atoms::bad_op()).encode(env)); }

    let n = (q_init.n_bytes / 4) as u32;
    let chain_bytes = q_init.n_bytes * (k as u64);
    let logp_bytes = (k as u64) * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;
    let (qh, ph, gh, lh) = match alloc_4(chain_bytes, chain_bytes, chain_bytes, logp_bytes) {
        Ok(t) => t,
        Err(_) => return Ok((atoms::error(), atoms::alloc_failed()).encode(env)),
    };
    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_halfnormal(
            qh, ph, gh, lh,
            q_init.handle, p_init.handle, inv_mass.handle,
            n, k,
            eps as f32, sigma as f32, log_const as f32,
            cstr.as_ptr(),
        )
    };
    encode_chain_result(env, rc, qh, ph, gh, lh, chain_bytes, logp_bytes)
}

#[rustler::nif]
fn leapfrog_chain_weibull<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64, weibull_k: f64, lambda: f64, logp_const: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 { return Ok((atoms::error(), atoms::bad_op()).encode(env)); }

    let n = (q_init.n_bytes / 4) as u32;
    let chain_bytes = q_init.n_bytes * (k as u64);
    let logp_bytes = (k as u64) * 4;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;
    let (qh, ph, gh, lh) = match alloc_4(chain_bytes, chain_bytes, chain_bytes, logp_bytes) {
        Ok(t) => t,
        Err(_) => return Ok((atoms::error(), atoms::alloc_failed()).encode(env)),
    };
    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_weibull(
            qh, ph, gh, lh,
            q_init.handle, p_init.handle, inv_mass.handle,
            n, k,
            eps as f32, weibull_k as f32, lambda as f32, logp_const as f32,
            cstr.as_ptr(),
        )
    };
    encode_chain_result(env, rc, qh, ph, gh, lh, chain_bytes, logp_bytes)
}

#[rustler::nif]
fn leapfrog_chain_normal_f64<'a>(
    env: Env<'a>,
    q_init: ResourceArc<VulkanTensor>,
    p_init: ResourceArc<VulkanTensor>,
    inv_mass: ResourceArc<VulkanTensor>,
    k: u32,
    eps: f64, mu: f64, sigma: f64,
    spv_path: String,
) -> NifResult<Term<'a>> {
    // f64: 8 bytes per element. n derived from input byte size.
    if q_init.n_bytes != p_init.n_bytes || q_init.n_bytes != inv_mass.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 { return Ok((atoms::error(), atoms::bad_op()).encode(env)); }

    let n = (q_init.n_bytes / 8) as u32;
    let chain_bytes = q_init.n_bytes * (k as u64);
    let logp_bytes = (k as u64) * 8;

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;
    let (qh, ph, gh, lh) = match alloc_4(chain_bytes, chain_bytes, chain_bytes, logp_bytes) {
        Ok(t) => t,
        Err(_) => return Ok((atoms::error(), atoms::alloc_failed()).encode(env)),
    };
    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_leapfrog_chain_normal_f64(
            qh, ph, gh, lh,
            q_init.handle, p_init.handle, inv_mass.handle,
            n, k,
            eps, mu, sigma,
            cstr.as_ptr(),
        )
    };
    encode_chain_result(env, rc, qh, ph, gh, lh, chain_bytes, logp_bytes)
}

// Helper: allocate four output buffers, free everything if any alloc fails.
fn alloc_4(b1: u64, b2: u64, b3: u64, b4: u64)
    -> Result<(*mut c_void, *mut c_void, *mut c_void, *mut c_void), ()>
{
    let h1 = unsafe { nxv_buf_alloc(b1) };
    if h1.is_null() { return Err(()); }
    let h2 = unsafe { nxv_buf_alloc(b2) };
    if h2.is_null() { unsafe { nxv_buf_free(h1) }; return Err(()); }
    let h3 = unsafe { nxv_buf_alloc(b3) };
    if h3.is_null() { unsafe { nxv_buf_free(h1); nxv_buf_free(h2) }; return Err(()); }
    let h4 = unsafe { nxv_buf_alloc(b4) };
    if h4.is_null() { unsafe { nxv_buf_free(h1); nxv_buf_free(h2); nxv_buf_free(h3) }; return Err(()); }
    Ok((h1, h2, h3, h4))
}

// Helper: dispatch result → tuple-encode or free-and-error.
fn encode_chain_result<'a>(
    env: Env<'a>,
    rc: i32,
    qh: *mut c_void, ph: *mut c_void, gh: *mut c_void, lh: *mut c_void,
    chain_bytes: u64, logp_bytes: u64,
) -> NifResult<Term<'a>> {
    if rc != 0 {
        unsafe { nxv_buf_free(qh); nxv_buf_free(ph);
                 nxv_buf_free(gh); nxv_buf_free(lh); };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }
    let q_c = VulkanTensor { handle: qh, n_bytes: chain_bytes };
    let p_c = VulkanTensor { handle: ph, n_bytes: chain_bytes };
    let g_c = VulkanTensor { handle: gh, n_bytes: chain_bytes };
    let l_c = VulkanTensor { handle: lh, n_bytes: logp_bytes };
    Ok((
        atoms::ok(),
        (ResourceArc::new(q_c), ResourceArc::new(p_c),
         ResourceArc::new(g_c), ResourceArc::new(l_c)),
    ).encode(env))
}

/// 4-input fused chain. ops + buf_idx are length-≤8 vecs;
/// padded to 8 with [255, 1] respectively. All 4 input buffers must
/// be the same byte size (single output of that size).
#[rustler::nif]
fn fused_chain_4<'a>(
    env: Env<'a>,
    a: ResourceArc<VulkanTensor>,
    b: ResourceArc<VulkanTensor>,
    c: ResourceArc<VulkanTensor>,
    d: ResourceArc<VulkanTensor>,
    ops: Vec<u32>,
    buf_idx: Vec<u32>,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if ops.is_empty() || ops.len() > 8 || ops.len() != buf_idx.len() {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }
    if a.n_bytes != b.n_bytes || a.n_bytes != c.n_bytes || a.n_bytes != d.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let n = (a.n_bytes / 4) as u32;
    let n_ops = ops.len() as u32;

    let mut padded_ops: [u32; 8] = [255; 8];
    let mut padded_buf: [u32; 8] = [1; 8];
    for (i, (&op, &bi)) in ops.iter().zip(buf_idx.iter()).enumerate() {
        padded_ops[i] = op;
        padded_buf[i] = bi;
    }

    let _g = SUBMIT_LOCK.lock().map_err(|_| Error::BadArg)?;

    let out_handle = unsafe { nxv_buf_alloc(a.n_bytes) };
    if out_handle.is_null() {
        return Ok((atoms::error(), atoms::alloc_failed()).encode(env));
    }

    let cstr = std::ffi::CString::new(spv_path).map_err(|_| Error::BadArg)?;
    let rc = unsafe {
        nxv_fused_chain_4(
            out_handle, a.handle, b.handle, c.handle, d.handle,
            n, n_ops, padded_ops.as_ptr(), padded_buf.as_ptr(),
            cstr.as_ptr(),
        )
    };

    if rc != 0 {
        unsafe { nxv_buf_free(out_handle) };
        return Ok((atoms::error(), atoms::dispatch_failed()).encode(env));
    }

    let out = VulkanTensor { handle: out_handle, n_bytes: a.n_bytes };
    Ok((atoms::ok(), ResourceArc::new(out)).encode(env))
}

/// Release every pooled VkBuf back to the device. Call at idle time
/// to reclaim memory.
#[rustler::nif]
fn pool_clear<'a>(env: Env<'a>) -> NifResult<Term<'a>> {
    unsafe { nxv_pool_clear() };
    Ok(atoms::ok().encode(env))
}

/// Pool stats: returns {:ok, %{hits, misses, freed, size_classes, total_pooled}}.
#[rustler::nif]
fn pool_stats<'a>(env: Env<'a>) -> NifResult<Term<'a>> {
    let mut hits: u64 = 0;
    let mut misses: u64 = 0;
    let mut freed: u64 = 0;
    let mut size_classes: u64 = 0;
    let mut total_pooled: u64 = 0;

    unsafe {
        nxv_pool_stats(&mut hits, &mut misses, &mut freed,
                       &mut size_classes, &mut total_pooled);
    }

    let map = rustler::Term::map_from_pairs(
        env,
        &[
            (rustler::types::atom::Atom::from_str(env, "hits").unwrap().encode(env), hits.encode(env)),
            (rustler::types::atom::Atom::from_str(env, "misses").unwrap().encode(env), misses.encode(env)),
            (rustler::types::atom::Atom::from_str(env, "freed").unwrap().encode(env), freed.encode(env)),
            (rustler::types::atom::Atom::from_str(env, "size_classes").unwrap().encode(env), size_classes.encode(env)),
            (rustler::types::atom::Atom::from_str(env, "total_pooled").unwrap().encode(env), total_pooled.encode(env)),
        ],
    ).map_err(|_| Error::BadArg)?;

    Ok((atoms::ok(), map).encode(env))
}

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(VulkanTensor, env);
    true
}

// rustler 0.36 deprecated the second arg (functions are auto-discovered
// via the #[rustler::nif] attribute). One-arg form is the new shape.
rustler::init!("Elixir.Nx.Vulkan.Native", load = on_load);
