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
}

// One-shot guard so Elixir can call init/0 idempotently. Vulkan's
// vk_init is itself idempotent at the spirit level (returns 0 if
// already inited) but tracking the state in Rust gives us cleaner
// error semantics on the Elixir side.
static INIT_STATE: Mutex<bool> = Mutex::new(false);

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

    if op > 6 {
        return Ok((atoms::error(), atoms::bad_op()).encode(env));
    }

    let n_bytes = a.n_bytes;
    let n_elems = (n_bytes / 4) as u32;     // f32 elements

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

fn on_load(env: Env, _info: Term) -> bool {
    rustler::resource!(VulkanTensor, env);
    true
}

// rustler 0.36 deprecated the second arg (functions are auto-discovered
// via the #[rustler::nif] attribute). One-arg form is the new shape.
rustler::init!("Elixir.Nx.Vulkan.Native", load = on_load);
