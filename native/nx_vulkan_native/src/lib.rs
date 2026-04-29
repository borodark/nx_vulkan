//! Rustler NIF for Nx.Vulkan.
//!
//! Three layers down from Elixir:
//!
//!   Elixir  →  Rust NIF  →  extern "C" shim  →  C++ spirit::vulkan
//!
//! v0.0.1 only wires the bootstrap path: vk_init, device_name, has_f64.
//! Tensor allocation + dispatch land in v0.0.2.

use rustler::{Atom, Encoder, Env, Error, NifResult, Term};
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::Mutex;

mod atoms {
    rustler::atoms! {
        ok,
        error,
        no_device,
        already_initialized,
    }
}

// extern "C" declarations matching c_src/nx_vulkan_shim.h.
extern "C" {
    fn nxv_init() -> i32;
    fn nxv_destroy();
    fn nxv_device_name() -> *const c_char;
    fn nxv_has_f64() -> i32;
}

// One-shot guard so Elixir can call init/0 idempotently. Vulkan's
// vk_init is itself idempotent at the spirit level (returns 0 if
// already inited) but tracking the state in Rust gives us cleaner
// error semantics on the Elixir side.
static INIT_STATE: Mutex<bool> = Mutex::new(false);

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

rustler::init!("Elixir.Nx.Vulkan.Native", [init, device_name, has_f64]);
