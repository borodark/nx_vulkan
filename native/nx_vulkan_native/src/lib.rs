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

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(VulkanTensor, env);
    true
}

// rustler 0.36 deprecated the second arg (functions are auto-discovered
// via the #[rustler::nif] attribute). One-arg form is the new shape.
rustler::init!("Elixir.Nx.Vulkan.Native", load = on_load);
