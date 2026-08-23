//! nx_vulkan_vulkano — Pure-Rust Rustler NIF for synthesised chain
//! shader dispatch via vulkano.
//!
//! Sibling of `nx_vulkan_native` (the C++ shim + spirit Vulkan backend).
//! Resource lifetimes flow through `Arc<Buffer>` so stale `VkBuf*`
//! handles cannot escape — the bug class that surfaced in Mission II
//! R4 step 4 (Nx.Vulkan.Backend.to_binary ArgumentError on a freed
//! tensor reference) is structurally eliminated.
//!
//! Exposes one NIF for now:
//!
//!     leapfrog_chain_synth(q_bin, p_bin, extras_bin, push, k, spv_path)
//!         -> {:ok, {q_chain_bin, p_chain_bin, grad_chain_bin,
//!                   logp_chain_bin}}
//!         |  {:error, atom_or_string}
//!
//! All inputs and outputs are binaries; the NIF allocates fresh
//! Vulkan buffers per call (no persistent pool — that comes later
//! once the calling pattern is established).

use std::fs;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;

use rustler::{Binary, Encoder, Env, NewBinary, NifResult, ResourceArc, Term};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    shader::SpecializationConstant,
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
    VulkanLibrary,
};

mod atoms {
    rustler::atoms! {
        ok,
        error,
        size_mismatch,
        bad_input,
        spv_read_failed,
        vulkan_init_failed,
        dispatch_failed,
        upload_failed,
        download_failed,
    }
}

// -- Pipeline cache --------------------------------------------------------
//
// vulkano's StandardDescriptorSetAllocator (allocator.rs:448) creates a fresh
// DescriptorPool per unique layout identity. Per-call pipeline + layout
// creation produces a fresh layout every dispatch, so the allocator never
// recycles its 32-slot pool — it just keeps creating new pools, eventually
// exhausting driver-side limits (observed: ~5000 iterations on FreeBSD
// NVIDIA before `descriptor set: a non-validation error occurred`).
//
// Caching by (spv_path, op_code) means the same layout identity is used
// across calls; vulkano's allocator recycles slots within a single pool.
//
// op_code = -1 sentinel means "shader has no spec constant" (reduce_axis,
// transpose_2d, matmul, leapfrog_chain_synth).

#[derive(Clone)]
struct CachedPipeline {
    layout: Arc<PipelineLayout>,
    pipeline: Arc<ComputePipeline>,
}

static PIPELINE_CACHE: OnceLock<Mutex<std::collections::HashMap<(String, i32), CachedPipeline>>> =
    OnceLock::new();

fn pipeline_cache() -> &'static Mutex<std::collections::HashMap<(String, i32), CachedPipeline>> {
    PIPELINE_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn get_or_create_pipeline(
    spv_path: &str,
    op_code: Option<i32>,
) -> Result<CachedPipeline, String> {
    let key = (spv_path.to_string(), op_code.unwrap_or(-1));

    {
        let guard = pipeline_cache().lock().map_err(|_| "cache poisoned".to_string())?;
        if let Some(cached) = guard.get(&key) {
            return Ok(cached.clone());
        }
    }

    let context = ctx()?;
    let spv_bytes = fs::read(spv_path).map_err(|e| format!("read spv: {e}"))?;
    let spv_words = bytes_to_u32_words(&spv_bytes)?;

    let shader = unsafe {
        ShaderModule::new(context.device.clone(), ShaderModuleCreateInfo::new(&spv_words))
            .map_err(|e| format!("ShaderModule: {e}"))?
    };

    let entry = match op_code {
        Some(op) => {
            let mut spec: ahash::HashMap<u32, SpecializationConstant> =
                ahash::HashMap::default();
            spec.insert(0, SpecializationConstant::I32(op));
            let specialized = shader
                .specialize(spec)
                .map_err(|e| format!("specialize: {e}"))?;
            specialized
                .entry_point("main")
                .ok_or_else(|| "no main entry point".to_string())?
        }
        None => shader
            .entry_point("main")
            .ok_or_else(|| "no main entry point".to_string())?,
    };

    let stage = PipelineShaderStageCreateInfo::new(entry);
    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
        .into_pipeline_layout_create_info(context.device.clone())
        .map_err(|e| format!("layout info: {e}"))?;
    let layout = PipelineLayout::new(context.device.clone(), layout_info)
        .map_err(|e| format!("PipelineLayout: {e}"))?;

    let pipeline = ComputePipeline::new(
        context.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
    )
    .map_err(|e| format!("ComputePipeline: {e}"))?;

    let cached = CachedPipeline { layout, pipeline };

    pipeline_cache()
        .lock()
        .map_err(|_| "cache poisoned".to_string())?
        .insert(key, cached.clone());

    Ok(cached)
}

/// NIF resource: a Vulkan-backed buffer whose lifetime is owned by Rust.
/// When the Elixir-side reference is GC'd, Rustler runs the Drop, which
/// in turn drops the inner Subbuffer. The Subbuffer holds an Arc to the
/// underlying allocation; once all references go, vkDestroyBuffer +
/// vkFreeMemory run via vulkano's Drop chain. No raw VkBuf* escapes.
pub struct VulkanoTensor {
    buf: Subbuffer<[u8]>,
    n_bytes: u64,
}

/// Lazy-init Vulkan context: instance, device, queue, allocators.
/// Held across NIF calls to avoid per-dispatch instance creation.
struct VkContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    mem_allocator: Arc<StandardMemoryAllocator>,
    cmd_allocator: Arc<StandardCommandBufferAllocator>,
    set_allocator: Arc<StandardDescriptorSetAllocator>,
    device_name: String,
    device_type: String,
    supports_f64: bool,
}

static CTX: OnceLock<VkContext> = OnceLock::new();

fn ctx() -> Result<&'static VkContext, String> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }

    let library = VulkanLibrary::new().map_err(|e| format!("VulkanLibrary::new: {e}"))?;
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .map_err(|e| format!("Instance::new: {e}"))?;

    let (physical, queue_family_index) = instance
        .enumerate_physical_devices()
        .map_err(|e| format!("enumerate_physical_devices: {e}"))?
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .ok_or_else(|| "no compute-capable Vulkan device".to_string())?;

    let device_name = physical.properties().device_name.clone();
    let device_type = format!("{:?}", physical.properties().device_type);

    eprintln!("[nx_vulkan_vulkano] device: {device_name} ({device_type})");

    // Enable shaderFloat64 if the device supports it; required by the
    // _f64.spv shaders. Falls back gracefully on devices without it
    // (those will keep using the f32 paths + host fallback for f64).
    let supports_f64 = physical.supported_features().shader_float64;
    // robust_buffer_access makes out-of-bounds reads return 0 instead of
    // faulting — needed by the select shader, which reads a u8 `pred` buffer as
    // u32 words and may touch up to 3 bytes past the end on the tail word.
    let supports_robust = physical.supported_features().robust_buffer_access;

    let enabled_features = vulkano::device::Features {
        shader_float64: supports_f64,
        robust_buffer_access: supports_robust,
        ..Default::default()
    };

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_features,
            ..Default::default()
        },
    )
    .map_err(|e| format!("Device::new: {e}"))?;

    let queue = queues.next().ok_or_else(|| "no queue".to_string())?;

    let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    // Ampere DeviceLost @ 16 dispatches — hypothesis 1 from
    // HANDOFF_MAC248_AMPERE_DEVICELOST.md Thread 2. Default is 32 primary
    // buffers per pool; on Ampere, SimultaneousUse command buffers don't
    // get marked "returned" fast enough and we exhaust the pool at 16.
    // Bump to 128 — if crash moves from 16 to ~128, hyp 1 confirmed and
    // we pick a permanent value or switch to per-call allocator.
    let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo {
            primary_buffer_count: 128,
            secondary_buffer_count: 0,
            ..Default::default()
        },
    ));
    // Default 32-slot pool is fine *if* layouts are stable (which the
    // pipeline cache ensures). Bumping set_count regressed RTX 3060 Ti
    // perf 6× on small-matmul without helping FreeBSD's failure mode —
    // see r1 of the race bench (May 20 2026).
    let set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let ctx = VkContext {
        device,
        queue,
        mem_allocator,
        cmd_allocator,
        set_allocator,
        device_name,
        device_type,
        supports_f64,
    };

    let _ = CTX.set(ctx);
    Ok(CTX.get().unwrap())
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushBlock {
    k_steps: u32,
    n_obs: u32,
    d: u32,
    _pad: u32,
    eps: f32,
}

fn parse_push_block(bytes: &[u8]) -> Result<PushBlock, &'static str> {
    if bytes.len() < 20 {
        return Err("push block must be >= 20 bytes");
    }
    let u32_at = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let f32_at = |off: usize| {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    Ok(PushBlock {
        k_steps: u32_at(0),
        n_obs: u32_at(4),
        d: u32_at(8),
        _pad: u32_at(12),
        eps: f32_at(16),
    })
}

// Plan A* (Task #149): f64 variant of the push block for the boundary-cast
// double-precision synth chain shader. `eps` is f64 at byte offset 16
// (16 = 4×u32 is naturally 8-aligned for the f64 that follows). Total 24
// bytes header; prior-param doubles follow (host packs them as little-f64).
#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushBlockF64 {
    k_steps: u32,
    n_obs: u32,
    d: u32,
    _pad: u32,
    eps: f64,
}

// Task #154: batched multi-instrument push block. Layout matches Phase 1
// f32 batched shader template (see exmc multi_rv_custom_spec.ex
// @batched_template). 20-byte header. Prior-param floats follow.
#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushBlockBatch {
    k_steps: u32,
    n_obs: u32,
    d: u32,
    n_instances: u32,
    eps: f32,
}

fn parse_push_block_batch(bytes: &[u8]) -> Result<PushBlockBatch, &'static str> {
    if bytes.len() < 20 {
        return Err("push block (batch) must be >= 20 bytes");
    }
    let u32_at = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let f32_at = |off: usize| {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    Ok(PushBlockBatch {
        k_steps: u32_at(0),
        n_obs: u32_at(4),
        d: u32_at(8),
        n_instances: u32_at(12),
        eps: f32_at(16),
    })
}

fn parse_push_block_f64(bytes: &[u8]) -> Result<PushBlockF64, &'static str> {
    if bytes.len() < 24 {
        return Err("push block (f64) must be >= 24 bytes");
    }
    let u32_at = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let f64_at = |off: usize| {
        f64::from_le_bytes([
            bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3],
            bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7],
        ])
    };
    Ok(PushBlockF64 {
        k_steps: u32_at(0),
        n_obs: u32_at(4),
        d: u32_at(8),
        _pad: u32_at(12),
        eps: f64_at(16),
    })
}

fn bytes_to_u32_words(bytes: &[u8]) -> Result<Vec<u32>, &'static str> {
    if bytes.len() % 4 != 0 {
        return Err("SPV bytes must be u32-aligned");
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn upload_buffer(
    alloc: Arc<StandardMemoryAllocator>,
    bytes: &[u8],
    usage: BufferUsage,
) -> Result<Subbuffer<[u8]>, String> {
    Buffer::from_iter(
        alloc,
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        bytes.iter().copied(),
    )
    .map_err(|e| format!("upload buffer: {e}"))
}

fn alloc_buffer(
    alloc: Arc<StandardMemoryAllocator>,
    n_bytes: usize,
    usage: BufferUsage,
) -> Result<Subbuffer<[u8]>, String> {
    Buffer::from_iter(
        alloc,
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        std::iter::repeat(0u8).take(n_bytes),
    )
    .map_err(|e| format!("alloc buffer: {e}"))
}

fn download_buffer(buf: Subbuffer<[u8]>) -> Result<Vec<u8>, String> {
    let guard = buf.read().map_err(|e| format!("read buffer: {e}"))?;
    Ok(guard.to_vec())
}

/// Run a K-step leapfrog dispatch against the synthesised SPV.
///
/// Returns {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin}
/// as little-endian f32 binaries:
///   q/p/grad: K * d * 4 bytes
///   logp:    K * 4 bytes
#[rustler::nif(schedule = "DirtyIo")]
fn leapfrog_chain_synth<'a>(
    env: Env<'a>,
    q_init: Binary<'a>,
    p_init: Binary<'a>,
    extras: Binary<'a>,
    push: Binary<'a>,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.len() != p_init.len() {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }
    if push.len() == 0 || push.len() > 128 {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }

    let push_block = match parse_push_block(push.as_slice()) {
        Ok(p) => p,
        Err(_) => return Ok((atoms::error(), atoms::bad_input()).encode(env)),
    };

    let d = push_block.d as usize;
    let chain_bytes = (k as usize) * d * 4;
    let logp_bytes = (k as usize) * 4;

    let context = match ctx() {
        Ok(c) => c,
        Err(e) => {
            return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env));
        }
    };

    let result = (|| -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let layout = cached.layout.clone();
        let pipeline = cached.pipeline.clone();

        let q_buf = upload_buffer(
            context.mem_allocator.clone(),
            q_init.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;
        let p_buf = upload_buffer(
            context.mem_allocator.clone(),
            p_init.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;
        let extras_buf = upload_buffer(
            context.mem_allocator.clone(),
            extras.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;

        let q_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let p_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let grad_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let logp_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            logp_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, q_buf.clone()),
                WriteDescriptorSet::buffer(1, p_buf.clone()),
                WriteDescriptorSet::buffer(2, extras_buf.clone()),
                WriteDescriptorSet::buffer(3, q_chain_buf.clone()),
                WriteDescriptorSet::buffer(4, p_chain_buf.clone()),
                WriteDescriptorSet::buffer(5, grad_chain_buf.clone()),
                WriteDescriptorSet::buffer(6, logp_chain_buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let mut cmd = AutoCommandBufferBuilder::primary(
            &context.cmd_allocator,
            context.queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
        )
        .map_err(|e| format!("cmd builder: {e}"))?;

        cmd.bind_pipeline_compute(pipeline.clone())
            .map_err(|e| format!("bind pipeline: {e}"))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set.clone())
            .map_err(|e| format!("bind descriptor: {e}"))?
            .push_constants(layout.clone(), 0, push_block)
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([1, 1, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;

        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?;

        finish_and_disarm(context, future)?;

        Ok((
            download_buffer(q_chain_buf)?,
            download_buffer(p_chain_buf)?,
            download_buffer(grad_chain_buf)?,
            download_buffer(logp_chain_buf)?,
        ))
    })();

    match result {
        Ok((q, p, g, l)) => {
            let q_bin = bytes_to_nif_binary(env, &q);
            let p_bin = bytes_to_nif_binary(env, &p);
            let g_bin = bytes_to_nif_binary(env, &g);
            let l_bin = bytes_to_nif_binary(env, &l);
            Ok((atoms::ok(), (q_bin, p_bin, g_bin, l_bin)).encode(env))
        }
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

fn bytes_to_nif_binary<'a>(env: Env<'a>, bytes: &[u8]) -> Binary<'a> {
    let mut bin = NewBinary::new(env, bytes.len());
    bin.as_mut_slice().copy_from_slice(bytes);
    bin.into()
}

/// Plan A* — boundary-cast f64 variant of leapfrog_chain_synth.
///
/// Identical dispatch logic but assumes:
///   - q_init, p_init, extras are little-endian f64 binaries (8 bytes/elem)
///   - push block is 24+ bytes (eps is f64 at offset 16)
///   - SPV at spv_path is the f64 compute shader (uses GL_ARB_gpu_shader_fp64
///     for storage; transcendentals via host-side double(log(float(x))) wrappers)
///
/// Returns {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin}
/// as little-endian f64 binaries:
///   q/p/grad: K * d * 8 bytes
///   logp:    K * 8 bytes
///
/// Discovered while building Plan A: GLSL.std.450 §8.1-8.2 excludes f64
/// from log/exp/pow/etc. Boundary-cast pattern (compile-time wrapper in
/// the GLSL emitter) gets f64 storage benefit at the cost of ~7 decimal
/// digits per transcendental call. Option D analysis on 2026-05-25 showed
/// this precision loss is negligible (~1e-3 absolute in logp ≈ 500);
/// f64 storage IS the fix for the f32 intermediate-overflow that
/// destroyed RegimeModel sampling.
#[rustler::nif(schedule = "DirtyIo")]
fn leapfrog_chain_synth_f64<'a>(
    env: Env<'a>,
    q_init: Binary<'a>,
    p_init: Binary<'a>,
    extras: Binary<'a>,
    push: Binary<'a>,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.len() != p_init.len() {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }
    if push.len() == 0 || push.len() > 128 {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }

    let push_block = match parse_push_block_f64(push.as_slice()) {
        Ok(p) => p,
        Err(_) => return Ok((atoms::error(), atoms::bad_input()).encode(env)),
    };

    let d = push_block.d as usize;
    // f64 = 8 bytes per element (vs 4 for f32)
    let chain_bytes = (k as usize) * d * 8;
    let logp_bytes = (k as usize) * 8;

    let context = match ctx() {
        Ok(c) => c,
        Err(e) => {
            return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env));
        }
    };

    let result = (|| -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let layout = cached.layout.clone();
        let pipeline = cached.pipeline.clone();

        let q_buf = upload_buffer(
            context.mem_allocator.clone(),
            q_init.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;
        let p_buf = upload_buffer(
            context.mem_allocator.clone(),
            p_init.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;
        let extras_buf = upload_buffer(
            context.mem_allocator.clone(),
            extras.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;

        let q_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let p_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let grad_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let logp_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            logp_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, q_buf.clone()),
                WriteDescriptorSet::buffer(1, p_buf.clone()),
                WriteDescriptorSet::buffer(2, extras_buf.clone()),
                WriteDescriptorSet::buffer(3, q_chain_buf.clone()),
                WriteDescriptorSet::buffer(4, p_chain_buf.clone()),
                WriteDescriptorSet::buffer(5, grad_chain_buf.clone()),
                WriteDescriptorSet::buffer(6, logp_chain_buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let mut cmd = AutoCommandBufferBuilder::primary(
            &context.cmd_allocator,
            context.queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
        )
        .map_err(|e| format!("cmd builder: {e}"))?;

        cmd.bind_pipeline_compute(pipeline.clone())
            .map_err(|e| format!("bind pipeline: {e}"))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set.clone())
            .map_err(|e| format!("bind descriptor: {e}"))?
            .push_constants(layout.clone(), 0, push_block)
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([1, 1, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;

        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?;

        finish_and_disarm(context, future)?;

        Ok((
            download_buffer(q_chain_buf)?,
            download_buffer(p_chain_buf)?,
            download_buffer(grad_chain_buf)?,
            download_buffer(logp_chain_buf)?,
        ))
    })();

    match result {
        Ok((q, p, g, l)) => {
            let q_bin = bytes_to_nif_binary(env, &q);
            let p_bin = bytes_to_nif_binary(env, &p);
            let g_bin = bytes_to_nif_binary(env, &g);
            let l_bin = bytes_to_nif_binary(env, &l);
            Ok((atoms::ok(), (q_bin, p_bin, g_bin, l_bin)).encode(env))
        }
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Task #154 — batched multi-instrument leapfrog dispatch.
///
/// Mirrors leapfrog_chain_synth (f32 single-instance) but:
/// - Push block includes n_instances (PushBlockBatch — 20 bytes header)
/// - Input buffers carry N concatenated instances (qs: N*d*4, etc.)
/// - SPV must be the batched f32 shader (from MultiRvCustomSpec.render_batched/1)
/// - Dispatches with [n_instances, 1, 1] — one workgroup per instance
/// - Output buffers carry N concatenated chain outputs
///
/// Returns {q_chain_bin, p_chain_bin, grad_chain_bin, logp_chain_bin}
/// as little-endian f32 binaries:
///   q/p/grad: n_instances * K * d * 4 bytes
///   logp:    n_instances * K * 4 bytes
#[rustler::nif(schedule = "DirtyIo")]
fn leapfrog_chain_synth_batch<'a>(
    env: Env<'a>,
    q_init: Binary<'a>,
    p_init: Binary<'a>,
    extras: Binary<'a>,
    push: Binary<'a>,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if q_init.len() != p_init.len() {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    if k == 0 {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }
    if push.len() == 0 || push.len() > 128 {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }

    let push_block = match parse_push_block_batch(push.as_slice()) {
        Ok(p) => p,
        Err(_) => return Ok((atoms::error(), atoms::bad_input()).encode(env)),
    };

    let d = push_block.d as usize;
    let n_instances = push_block.n_instances as usize;
    if n_instances == 0 {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }
    // f32 = 4 bytes; per-instance chain: K * d * 4; total: n_instances * K * d * 4
    let chain_bytes = n_instances * (k as usize) * d * 4;
    let logp_bytes = n_instances * (k as usize) * 4;

    let context = match ctx() {
        Ok(c) => c,
        Err(e) => {
            return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env));
        }
    };

    let result = (|| -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let layout = cached.layout.clone();
        let pipeline = cached.pipeline.clone();

        let q_buf = upload_buffer(
            context.mem_allocator.clone(),
            q_init.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;
        let p_buf = upload_buffer(
            context.mem_allocator.clone(),
            p_init.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;
        let extras_buf = upload_buffer(
            context.mem_allocator.clone(),
            extras.as_slice(),
            BufferUsage::STORAGE_BUFFER,
        )?;

        let q_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let p_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let grad_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            chain_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;
        let logp_chain_buf = alloc_buffer(
            context.mem_allocator.clone(),
            logp_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
        )?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, q_buf.clone()),
                WriteDescriptorSet::buffer(1, p_buf.clone()),
                WriteDescriptorSet::buffer(2, extras_buf.clone()),
                WriteDescriptorSet::buffer(3, q_chain_buf.clone()),
                WriteDescriptorSet::buffer(4, p_chain_buf.clone()),
                WriteDescriptorSet::buffer(5, grad_chain_buf.clone()),
                WriteDescriptorSet::buffer(6, logp_chain_buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let mut cmd = AutoCommandBufferBuilder::primary(
            &context.cmd_allocator,
            context.queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
        )
        .map_err(|e| format!("cmd builder: {e}"))?;

        // KEY DIFFERENCE vs single-instance: dispatch [n_instances, 1, 1]
        // — each workgroup handles one instance, scaling compute with the
        // batch size while dispatch overhead stays constant.
        cmd.bind_pipeline_compute(pipeline.clone())
            .map_err(|e| format!("bind pipeline: {e}"))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set.clone())
            .map_err(|e| format!("bind descriptor: {e}"))?
            .push_constants(layout.clone(), 0, push_block)
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([n_instances as u32, 1, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;

        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?;

        finish_and_disarm(context, future)?;

        Ok((
            download_buffer(q_chain_buf)?,
            download_buffer(p_chain_buf)?,
            download_buffer(grad_chain_buf)?,
            download_buffer(logp_chain_buf)?,
        ))
    })();

    match result {
        Ok((q, p, g, l)) => {
            let q_bin = bytes_to_nif_binary(env, &q);
            let p_bin = bytes_to_nif_binary(env, &p);
            let g_bin = bytes_to_nif_binary(env, &g);
            let l_bin = bytes_to_nif_binary(env, &l);
            Ok((atoms::ok(), (q_bin, p_bin, g_bin, l_bin)).encode(env))
        }
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

// -- Buffer lifecycle NIFs ------------------------------------------------
//
// Sibling of the C++ shim's nxv_buf_* family, but every buffer is held
// behind a Rust Arc<Buffer> wrapped in a Subbuffer<[u8]> + ResourceArc.
// The stale-handle bug class is structurally absent: a Subbuffer cannot
// outlive its underlying Buffer because vulkano enforces it at the type
// level, and Rustler's ResourceArc Drop runs vulkano's Drop before any
// Elixir reference becomes dangling.

/// Allocate + upload a host binary into a fresh device buffer.
/// Returns `{:ok, resource}`.
#[rustler::nif(schedule = "DirtyIo")]
fn buf_upload<'a>(env: Env<'a>, data: Binary<'a>) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let buf = match upload_buffer(
        context.mem_allocator.clone(),
        data.as_slice(),
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
    ) {
        Ok(b) => b,
        Err(e) => return Ok((atoms::error(), atoms::upload_failed(), e).encode(env)),
    };

    let tensor = VulkanoTensor {
        buf,
        n_bytes: data.len() as u64,
    };
    Ok((atoms::ok(), ResourceArc::new(tensor)).encode(env))
}

/// Allocate a zero-initialised device buffer of `n_bytes`.
/// Returns `{:ok, resource}`.
#[rustler::nif(schedule = "DirtyIo")]
fn buf_alloc<'a>(env: Env<'a>, n_bytes: u64) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let buf = match alloc_buffer(
        context.mem_allocator.clone(),
        n_bytes as usize,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
    ) {
        Ok(b) => b,
        Err(e) => return Ok((atoms::error(), atoms::upload_failed(), e).encode(env)),
    };

    let tensor = VulkanoTensor { buf, n_bytes };
    Ok((atoms::ok(), ResourceArc::new(tensor)).encode(env))
}

/// Concatenate N device buffers into a single fresh buffer via
/// `vkCmdCopyBuffer`. No shader involved — pure DMA on the queue.
///
/// Tier 2 of SHAPE_C_PLAN.md step 1: the host-fallback `concatenate`
/// was 0.02× speedup vs BinaryBackend because BinaryBackend's
/// concatenate is essentially a binary append. The GPU-native path
/// here keeps the result on the device, avoiding the download +
/// re-upload round trip for downstream ops.
///
/// Returns `{:ok, output_tensor_ref}` with `n_bytes = Σ inputs[i].n_bytes`.
#[rustler::nif(schedule = "DirtyIo")]
fn concat_buffers<'a>(
    env: Env<'a>,
    inputs: Vec<ResourceArc<VulkanoTensor>>,
) -> NifResult<Term<'a>> {
    if inputs.is_empty() {
        return Ok((atoms::error(), atoms::bad_input()).encode(env));
    }

    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    // This NIF submits its own command buffer, so any dispatch that produced
    // one of the inputs has to have landed first.
    if let Err(e) = flush_pending() {
        return Ok((atoms::error(), atoms::dispatch_failed(), e).encode(env));
    }

    let total_bytes: u64 = inputs.iter().map(|t| t.n_bytes).sum();

    let dst = match alloc_buffer(
        context.mem_allocator.clone(),
        total_bytes as usize,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
    ) {
        Ok(b) => b,
        Err(e) => return Ok((atoms::error(), atoms::upload_failed(), e).encode(env)),
    };

    let mut cmd = match AutoCommandBufferBuilder::primary(
        &context.cmd_allocator,
        context.queue.queue_family_index(),
        CommandBufferUsage::SimultaneousUse,
    ) {
        Ok(c) => c,
        Err(e) => {
            return Ok(
                (atoms::error(), atoms::dispatch_failed(), format!("cmd builder: {e}")).encode(env),
            )
        }
    };

    let mut offset: u64 = 0;
    for input in &inputs {
        let len = input.n_bytes;
        let dst_slice = dst.clone().slice(offset..offset + len);

        if let Err(e) = cmd.copy_buffer(vulkano::command_buffer::CopyBufferInfo::buffers(
            input.buf.clone(),
            dst_slice,
        )) {
            return Ok(
                (atoms::error(), atoms::dispatch_failed(), format!("copy_buffer: {e}"))
                    .encode(env),
            );
        }

        offset += len;
    }

    let cmd_buf = match cmd.build() {
        Ok(b) => b,
        Err(e) => {
            return Ok(
                (atoms::error(), atoms::dispatch_failed(), format!("build cmd: {e}")).encode(env),
            )
        }
    };

    let future = match sync::now(context.device.clone())
        .then_execute(context.queue.clone(), cmd_buf)
    {
        Ok(f) => f,
        Err(e) => {
            return Ok(
                (atoms::error(), atoms::dispatch_failed(), format!("execute: {e}")).encode(env),
            )
        }
    };

    // Same trap as everywhere else, in a different shape: these used to be two
    // early  arms placed BEFORE signal_finished, so a
    // flush or wait_idle failure left the future armed and Drop panicked the
    // NIF instead of returning the error tuple that is right there.
    if let Err(e) = finish_and_disarm(context, future) {
        return Ok((atoms::error(), atoms::dispatch_failed(), e).encode(env));
    }

    let tensor = VulkanoTensor {
        buf: dst,
        n_bytes: total_bytes,
    };
    Ok((atoms::ok(), ResourceArc::new(tensor)).encode(env))
}

/// Download `tensor.n_bytes` bytes from a device buffer to the BEAM.
/// Returns `{:ok, binary}`.
#[rustler::nif(schedule = "DirtyIo")]
fn buf_download<'a>(env: Env<'a>, tensor: ResourceArc<VulkanoTensor>) -> NifResult<Term<'a>> {
    // The tensor's contents may still be sitting in the pending batch. This is
    // the boundary where deferred dispatch has to become real work.
    if let Err(e) = flush_pending() {
        return Ok((atoms::error(), atoms::dispatch_failed(), e).encode(env));
    }
    let bytes = match tensor.buf.read() {
        Ok(guard) => guard.to_vec(),
        Err(_) => return Ok((atoms::error(), atoms::download_failed()).encode(env)),
    };
    let bin = bytes_to_nif_binary(env, &bytes);
    Ok((atoms::ok(), bin).encode(env))
}

/// Tensor's buffer size in bytes.
#[rustler::nif]
fn buf_byte_size(tensor: ResourceArc<VulkanoTensor>) -> u64 {
    tensor.n_bytes
}

/// Overwrite an existing device buffer with new host data.
/// Returns `:ok` or `{:error, :size_mismatch}` when `data.len() != tensor.n_bytes`.
#[rustler::nif(schedule = "DirtyIo")]
fn buf_upload_into<'a>(
    env: Env<'a>,
    tensor: ResourceArc<VulkanoTensor>,
    data: Binary<'a>,
) -> NifResult<Term<'a>> {
    if data.len() as u64 != tensor.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }
    // A queued dispatch may read this buffer. Nothing has been submitted yet,
    // so vulkano's own in-use tracking would not catch the overwrite.
    if let Err(e) = flush_pending() {
        return Ok((atoms::error(), atoms::dispatch_failed(), e).encode(env));
    }
    let mut guard = match tensor.buf.write() {
        Ok(g) => g,
        Err(_) => return Ok((atoms::error(), atoms::upload_failed()).encode(env)),
    };
    guard.copy_from_slice(data.as_slice());
    Ok(rustler::types::atom::ok().encode(env))
}

// -- Compute NIFs ---------------------------------------------------------

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushN {
    n: u32,
}

/// Elementwise binary op. `op_code` selects:
///   0=add, 1=mul, 2=sub, 3=div, 4=pow, 5=max, 6=min
/// Bindings: a, b, out at 0, 1, 2. Push: uint n.
/// Workgroup: 256 threads, ceil(n/256) groups.
#[rustler::nif(schedule = "DirtyIo")]
fn apply_binary<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    b_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    op_code: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if a_ref.n_bytes != b_ref.n_bytes || a_ref.n_bytes != out_ref.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, Some(op_code as i32))?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, b_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushN { n }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushBcast {
    n: u32,
    rank: u32,
}

/// Broadcasting elementwise binary op. Bindings: a, b, out, params (shapes) at
/// 0..3. Push: {n, rank}. `op_code` spec constant selects the op. Keeps
/// broadcast ops (bias-add, scaling, relu) on the GPU instead of host-falling-
/// back. n = output element count.
#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn apply_binary_broadcast<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    b_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    op_code: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, Some(op_code as i32))?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, b_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBcast { n, rank }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Broadcasting comparison -> u8 (packed as u32). Bindings: a, b, out, params
/// at 0..3. Push: {n, rank}. `op_code` spec constant selects eq/ne/lt/le/gt/ge.
/// One thread per output u32 word (4 u8 results); dispatch ceil(n/4) threads.
#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn apply_compare<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    b_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    op_code: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, Some(op_code as i32))?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, b_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let nwords = n.div_ceil(4);
        enqueue_dispatch(context, &cached, set, PushBcast { n, rank }, [nwords.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Broadcasting select: out = pred ? t : f. Bindings: pred, t, f, out, params
/// at 0..4. Push: {n, rank}. pred is a u8 tensor read as u32 words in the shader
/// (needs robust_buffer_access for the tail). Keeps masking / where / relu-grad
/// on the GPU.
#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn apply_select<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    pred_ref: ResourceArc<VulkanoTensor>,
    t_ref: ResourceArc<VulkanoTensor>,
    f_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, pred_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, t_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, f_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(4, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBcast { n, rank }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// put_slice overlay (type-generic u32-word copy). Bindings: tensor, slice,
/// out, params at 0..3. Push: {n, rank} where n = output element count (== the
/// tensor's). Params carry element word count + tensor dims + slice dims +
/// clamped per-dim starts. Each output element reads the slice when it is
/// inside the window and the tensor otherwise, so the whole op is one dispatch
/// with no host round trip.
#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn apply_put_slice<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    in_ref: ResourceArc<VulkanoTensor>,
    slice_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, in_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, slice_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBcast { n, rank }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Strided slice (type-generic u32-word copy). Bindings: in, out, params at
/// 0..2. Push: {n, rank} where n = output element count. Params carry element
/// word count + source/output dims + start/stride. Keeps slice on the GPU.
#[rustler::nif(schedule = "DirtyIo")]
fn apply_slice<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    in_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, in_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBcast { n, rank }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Pad (type-generic copy). Bindings: in 0, out 1, params 2, pad-value 3. Push:
/// {n, rank} where n = output element count. Params carry element word count +
/// source/output dims + per-dim low + interior. Elements landing in an edge
/// pad, an interior gap, or outside the source get the pad value. Keeps pad on
/// the GPU.
#[rustler::nif(schedule = "DirtyIo")]
fn apply_pad<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    in_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    padval_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, in_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, padval_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBcast { n, rank }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Gather (leading-prefix axes). Bindings: in 0, out 1, indices 2, params 3.
/// Push: {n, K} where n = output element count, K = number of indexed leading
/// axes. Params carry element word count + index word count + inner block size
/// + per-leading-axis strides. Keeps gather on the GPU for the common case.
/// Scatter — indexed_put (op 0) and indexed_add (op 1), the inverse of
/// apply_gather. Bindings mirror the shader: updates 0, out 1, indices 2,
/// params 3.
///
/// `out_ref` is READ-WRITE here, unlike every other dispatch in this file. It
/// arrives pre-seeded with a copy of the target tensor (the Elixir side makes
/// that copy with `concat_buffers/1` on a single buffer, which waits before
/// returning), because a scatter only writes the elements the indices name and
/// everything else has to survive. op 1 then accumulates into it with an
/// integer `atomicAdd`.
#[rustler::nif(schedule = "DirtyIo")]
fn apply_scatter<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    upd_ref: ResourceArc<VulkanoTensor>,
    idx_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    k: u32,
    op_code: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, Some(op_code as i32))?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, upd_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, idx_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBcast { n, rank: k }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
fn apply_gather<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    in_ref: ResourceArc<VulkanoTensor>,
    idx_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, in_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, idx_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBcast { n, rank: k }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Generic JIT-shader dispatch (thrust 3 — the Defn fusion compiler). Runs a
/// runtime-generated shader whose layout is: input buffers at bindings
/// 0..k-1, output buffer at binding k, push constant {n = element count}.
/// `in_refs` is the ordered list of input buffers the codegen assigned to
/// bindings 0..k-1. One dispatch replaces a whole fused elementwise chain.
#[rustler::nif(schedule = "DirtyIo")]
fn dispatch_generated<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    in_refs: Vec<ResourceArc<VulkanoTensor>>,
    n: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let mut writes: Vec<WriteDescriptorSet> = in_refs
            .iter()
            .enumerate()
            .map(|(i, r)| WriteDescriptorSet::buffer(i as u32, r.buf.clone()))
            .collect();
        writes.push(WriteDescriptorSet::buffer(
            in_refs.len() as u32,
            out_ref.buf.clone(),
        ));

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            writes,
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushN { n }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Generic JIT fused-reduce dispatch (thrust 3). A runtime-generated shader
/// that fuses an elementwise chain into a reduction: inputs at bindings
/// 0..k-1, output at k, push {outer, reduce_size, inner, op} (the reduce op is
/// baked into the generated shader; `op` is passed 0 and ignored). One
/// invocation per output slot; dispatch ceil(outer*inner/256) workgroups.
#[rustler::nif(schedule = "DirtyIo")]
fn dispatch_generated_reduce<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    in_refs: Vec<ResourceArc<VulkanoTensor>>,
    outer: u32,
    reduce_size: u32,
    inner: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let mut writes: Vec<WriteDescriptorSet> = in_refs
            .iter()
            .enumerate()
            .map(|(i, r)| WriteDescriptorSet::buffer(i as u32, r.buf.clone()))
            .collect();
        writes.push(WriteDescriptorSet::buffer(
            in_refs.len() as u32,
            out_ref.buf.clone(),
        ));

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            writes,
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        // Parallel tree reduce: one workgroup per output slot (each workgroup's
        // 256 threads cooperatively reduce that slot's axis). The shader
        // grid-strides over slots, so cap the launch at 65535 workgroups
        // (maxComputeWorkGroupCount[0]) and let the loop cover any excess.
        let n_slots = outer * inner;
        let groups = n_slots.min(65535).max(1);
        enqueue_dispatch(
            context,
            &cached,
            set,
            PushReduceAxis {
                outer,
                reduce_size,
                inner,
                op: 0,
            },
            [groups, 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Elementwise dtype cast (e.g. f32<->f64). Bindings: in at 0, out at 1 (which
/// may have a different element size). Push: uint n (element count). The shader
/// determines the source/dest types; no op_code.
#[rustler::nif(schedule = "DirtyIo")]
fn cast<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushN { n }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Elementwise unary op. `op_code` selects:
///   0=exp 1=log 2=sqrt 3=abs 4=neg 5=sigmoid 6=tanh 7=relu
///   8=ceil 9=floor 10=sign 11=reciprocal 12=square
/// Bindings: a, out at 0, 1. Push: uint n. Workgroup: 256 threads.
#[rustler::nif(schedule = "DirtyIo")]
fn apply_unary<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    op_code: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    if a_ref.n_bytes != out_ref.n_bytes {
        return Ok((atoms::error(), atoms::size_mismatch()).encode(env));
    }

    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, Some(op_code as i32))?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushN { n }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushReduceAxis {
    outer: u32,
    reduce_size: u32,
    inner: u32,
    op: u32,
}

/// Per-axis reduction. `op`: 0=sum, 1=max, 2=min.
/// Bindings: a, out. Push: {outer, reduce_size, inner, op}.
/// dispatch ceil(outer*inner/256) workgroups.
#[rustler::nif(schedule = "DirtyIo")]
fn reduce_axis<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    outer: u32,
    reduce_size: u32,
    inner: u32,
    op_code: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let n_slots = outer * inner;

        enqueue_dispatch(
            context,
            &cached,
            set,
            PushReduceAxis {
                outer,
                reduce_size,
                inner,
                op: op_code,
            },
            [n_slots.div_ceil(256), 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushTranspose {
    m: u32,
    n: u32,
}

/// 2D transpose. Input A is M×N row-major; output is N×M row-major.
/// Bindings: a, out at 0, 1. Push: {m, n}. Workgroup 16×16.
#[rustler::nif(schedule = "DirtyIo")]
fn transpose_2d<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    m: u32,
    n: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let spv_bytes = fs::read(&spv_path).map_err(|e| format!("read spv: {e}"))?;
        let spv_words = bytes_to_u32_words(&spv_bytes)?;

        let shader = unsafe {
            ShaderModule::new(context.device.clone(), ShaderModuleCreateInfo::new(&spv_words))
                .map_err(|e| format!("ShaderModule: {e}"))?
        };

        let entry = shader
            .entry_point("main")
            .ok_or_else(|| "no main entry point".to_string())?;
        let stage = PipelineShaderStageCreateInfo::new(entry);

        let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(context.device.clone())
            .map_err(|e| format!("layout info: {e}"))?;
        let layout = PipelineLayout::new(context.device.clone(), layout_info)
            .map_err(|e| format!("PipelineLayout: {e}"))?;

        let pipeline = ComputePipeline::new(
            context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
        )
        .map_err(|e| format!("ComputePipeline: {e}"))?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let gx = n.div_ceil(16);
        let gy = m.div_ceil(16);

        // Wraps the per-call pipeline rather than going through
        // `get_or_create_pipeline`, purely so this NIF keeps its existing
        // pipeline strategy while joining the batch. Moving it (and `matmul`)
        // onto the cache is a separate, separately-measured change — building
        // a ShaderModule and ComputePipeline per dispatch is per-dispatch cost
        // too, and conflating the two would make neither measurable.
        let cached = CachedPipeline { layout, pipeline };

        enqueue_dispatch(context, &cached, set, PushTranspose { m, n }, [gx, gy, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Scatter source values to each window's argmax (pooling backward), rank <= 4,
/// NON-OVERLAPPING windows only. Bindings: a, src, init, out, params at 0..4.
/// Push: {n, rank}. Params: [rank, in[4], win_grid[4], win[4], strides[4]].
/// One thread per INPUT element — that inversion is what avoids float atomics,
/// since each element belongs to at most one window.
#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn window_scatter_max<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    src_ref: ResourceArc<VulkanoTensor>,
    init_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, src_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, init_ref.buf.clone()),
                WriteDescriptorSet::buffer(3, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(4, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(
            context,
            &cached,
            set,
            PushTransposeNd { n, rank },
            [n.div_ceil(256), 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Windowed max/min, rank <= 4. Bindings: a, out, params at 0..2.
/// Push: {n, rank}. Params: [rank, in[4], out[4], win[4], strides[4]] as i32.
/// `op_code` spec constant: 0=max, 1=min. One thread per output element, so
/// overlapping windows need no coordination.
#[rustler::nif(schedule = "DirtyIo")]
fn window_reduce<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    op_code: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, Some(op_code as i32))?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(
            context,
            &cached,
            set,
            PushTransposeNd { n, rank },
            [n.div_ceil(256), 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushBroadcastNd {
    n: u32,
}

/// Explicit broadcast, rank <= 4. Bindings: a, out, params at 0..2.
/// Push: {n}. Params: [out_rank, in_rank, out[4], in[4], axes[4]] as i32.
/// One thread per output element; no spec constant.
#[rustler::nif(schedule = "DirtyIo")]
fn broadcast_nd<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(context, &cached, set, PushBroadcastNd { n }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Reverse along a set of axes, rank <= 4. Bindings: a, out, params at 0..2.
/// Push: {n, rank}. Params: [rank, shape[4], rev[4]] as i32, `rev` a 0/1 flag
/// per axis. One thread per output element; no spec constant.
///
/// Keeps the conv input-gradient's kernel reversal on the GPU — on the host it
/// also stranded the tensor on Nx.BinaryBackend, taking everything downstream
/// with it.
#[rustler::nif(schedule = "DirtyIo")]
fn reverse_nd<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(
            context,
            &cached,
            set,
            PushTransposeNd { n, rank },
            [n.div_ceil(256), 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushTransposeNd {
    n: u32,
    rank: u32,
}

/// Generic permuted transpose for rank <= 4. Bindings: a, out, params at 0..2.
/// Push: {n, rank}. Params buffer: [rank, in[4], out[4], perm[4]] as i32.
/// One thread per output element; no spec constant.
///
/// This is what lets conv's BACKWARD pass stay on the GPU: Nx's conv gradient
/// emits convolutions with the first two axes swapped, which the conv fast path
/// cannot run directly, so the backend transposes into the native layout around
/// it rather than falling back to the host.
#[rustler::nif(schedule = "DirtyIo")]
fn transpose_nd<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(
            context,
            &cached,
            set,
            PushTransposeNd { n, rank },
            [n.div_ceil(256), 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}


#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushConcatNd {
    n: u32,
    rank: u32,
}

/// Concatenate one input tensor into a slab of an already-allocated output.
/// Bindings: a, out, params at 0..2. Push: {n, rank} where `n` is the element
/// count of THIS INPUT, not of the output. Params buffer:
/// [rank, ews, axis, offset, in[4], out[4]] as i32.
///
/// Called once per input tensor. Each call writes only the region
/// [offset, offset + in[axis]) along the concat axis, and those regions are
/// disjoint by construction, so the output accumulates across calls. The caller
/// (`VulkanoBackend.concatenate/3`) is responsible for allocating the output and
/// for the per-input offsets; axis-0 concat does not come here at all, because
/// `concat_buffers` handles it as a byte append.
#[rustler::nif(schedule = "DirtyIo")]
fn concat_nd<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    rank: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(
            context,
            &cached,
            set,
            PushConcatNd { n, rank },
            [n.div_ceil(256), 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushMatmul {
    m: u32,
    n: u32,
    k: u32,
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushMatmulBatched {
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
}

/// 2D matmul. C = A · B where A is M×K, B is K×N, C is M×N.
/// All row-major f32. Bindings: a, b, out at 0, 1, 2. Push {m, n, k}.
/// Workgroup 16×16, dispatch ceil(N/16)×ceil(M/16).
#[rustler::nif(schedule = "DirtyIo")]
fn matmul<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    b_ref: ResourceArc<VulkanoTensor>,
    m: u32,
    n: u32,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let spv_bytes = fs::read(&spv_path).map_err(|e| format!("read spv: {e}"))?;
        let spv_words = bytes_to_u32_words(&spv_bytes)?;

        let shader = unsafe {
            ShaderModule::new(context.device.clone(), ShaderModuleCreateInfo::new(&spv_words))
                .map_err(|e| format!("ShaderModule: {e}"))?
        };

        let entry = shader
            .entry_point("main")
            .ok_or_else(|| "no main entry point".to_string())?;
        let stage = PipelineShaderStageCreateInfo::new(entry);

        let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(context.device.clone())
            .map_err(|e| format!("layout info: {e}"))?;
        let layout = PipelineLayout::new(context.device.clone(), layout_info)
            .map_err(|e| format!("PipelineLayout: {e}"))?;

        let pipeline = ComputePipeline::new(
            context.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
        )
        .map_err(|e| format!("ComputePipeline: {e}"))?;

        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, b_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let gx = n.div_ceil(16);
        let gy = m.div_ceil(16);

        // Per-call pipeline wrapped so this NIF joins the batch without
        // changing its (legacy, uncached) pipeline strategy — see the same
        // note in `transpose_2d`.
        let cached = CachedPipeline { layout, pipeline };

        enqueue_dispatch(context, &cached, set, PushMatmul { m, n, k }, [gx, gy, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

// Register-blocked matmul dispatch: identical bindings/push to `matmul` but
// dispatches 32-wide output tiles (gx=ceil(N/32), gy=ceil(M/32)) for the
// *_rb32 shaders (16×16 workgroup computes a 32×32 tile, 2×2 per thread). Not
// wired as the backend default — the register-blocked kernels regressed on
// Kepler; this NIF exists so `examples/matmul_rb_race.exs` can benchmark them
// against the tiled default on other GPUs (e.g. Ampere). See F32_PLAN.md.
/// Batched matmul. C[b] = A[b] · B[b] for b in 0..batch, batches laid out
/// contiguously. Bindings match `matmul/7`; the push block gains `batch` and
/// the batch index rides the THIRD dispatch dimension rather than being looped
/// in the caller — dispatching once per matrix would pay the launch cost per
/// batch element, which is the overhead that made the vectorised `reduce/5`
/// fold lose to the host (bench/reduce_fold_vs_host.exs).
///
/// Workgroup 16×16×1, dispatch ceil(N/16) × ceil(M/16) × batch. Note /16 here
/// against `matmul32`'s /32: this shader uses a plain 16×16 output tile, not
/// the register-blocked one.
#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn matmul_batched<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    b_ref: ResourceArc<VulkanoTensor>,
    batch: u32,
    m: u32,
    n: u32,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, b_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        enqueue_dispatch(
            context,
            &cached,
            set,
            PushMatmulBatched { m, n, k, batch },
            [n.div_ceil(16), m.div_ceil(16), batch],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn matmul32<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    b_ref: ResourceArc<VulkanoTensor>,
    m: u32,
    n: u32,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, b_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let gx = (n + 31) / 32;
        let gy = (m + 31) / 32;
        enqueue_dispatch(context, &cached, set, PushMatmul { m, n, k }, [gx, gy, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

// -- FFT ------------------------------------------------------------------
//
// Radix-2 Cooley-Tukey (decimation-in-time), power-of-two length, over the
// last axis, batched. Two shaders: a bit-reversed complex load, then log2(n)
// in-place butterfly stages. Twiddles are computed here in f64 (GLSL fp64 has
// no sin/cos) and uploaded as a table of n/2 complex entries reused across
// stages. The whole transform is recorded into a single command buffer so
// vulkano's AutoCommandBufferBuilder inserts the compute-compute barriers
// between dependent stages automatically.

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushFftBitrev {
    n: u32,
    logn: u32,
    batch: u32,
    is_complex: u32,
    inverse: u32,
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushFftStage {
    n: u32,
    half_: u32,
    batch: u32,
    stride: u32,
}

#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn fft<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    in_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    logn: u32,
    batch: u32,
    is_complex: u32,
    inverse: u32,
    bitrev_spv: String,
    stage_spv: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        if n < 2 || (n & (n - 1)) != 0 {
            return Err(format!("fft length must be a power of two >= 2, got {n}"));
        }

        // This NIF records and submits its own multi-stage command buffer
        // rather than joining the global batch (its stages are in-place on
        // `out_ref` and it owns a scratch twiddle buffer), so anything that
        // produced `in_ref` has to have landed first.
        flush_pending()?;

        // Twiddle table: n/2 complex entries, tw[t] = exp(sgn*2*pi*i*t/n).
        // Forward DFT uses sgn = -1; inverse uses +1 (the 1/n scale is folded
        // into the bit-reversed load).
        let sgn = if inverse == 1 { 1.0f64 } else { -1.0f64 };
        let half = (n / 2) as usize;
        let mut tw_bytes: Vec<u8> = Vec::with_capacity(half * 16);
        for t in 0..half {
            let ang = sgn * std::f64::consts::TAU * (t as f64) / (n as f64);
            tw_bytes.extend_from_slice(&ang.cos().to_le_bytes());
            tw_bytes.extend_from_slice(&ang.sin().to_le_bytes());
        }
        let tw_buf = upload_buffer(
            context.mem_allocator.clone(),
            &tw_bytes,
            BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC | BufferUsage::TRANSFER_DST,
        )?;

        let bitrev = get_or_create_pipeline(&bitrev_spv, None)?;
        let stage = get_or_create_pipeline(&stage_spv, None)?;

        let mut cmd = AutoCommandBufferBuilder::primary(
            &context.cmd_allocator,
            context.queue.queue_family_index(),
            CommandBufferUsage::SimultaneousUse,
        )
        .map_err(|e| format!("cmd builder: {e}"))?;

        // Stage 0: bit-reversed load, in_ref -> out_ref (complex).
        let load_set = PersistentDescriptorSet::new(
            &context.set_allocator,
            bitrev.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, in_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("load descriptor set: {e}"))?;

        let load_groups = (batch * n).div_ceil(64);
        cmd.bind_pipeline_compute(bitrev.pipeline.clone())
            .map_err(|e| format!("bind pipeline: {e}"))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, bitrev.layout.clone(), 0, load_set)
            .map_err(|e| format!("bind descriptor: {e}"))?
            .push_constants(bitrev.layout.clone(), 0, PushFftBitrev { n, logn, batch, is_complex, inverse })
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([load_groups, 1, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        // Stages 1..=logn: in-place butterflies on out_ref.
        let stage_groups = (batch * (n / 2)).div_ceil(64);
        for s in 1..=logn {
            let stage_half = 1u32 << (s - 1);
            let m = 1u32 << s;
            let stride = n / m;

            let set = PersistentDescriptorSet::new(
                &context.set_allocator,
                stage.layout.set_layouts()[0].clone(),
                [
                    WriteDescriptorSet::buffer(0, out_ref.buf.clone()),
                    WriteDescriptorSet::buffer(1, tw_buf.clone()),
                ],
                [],
            )
            .map_err(|e| format!("stage descriptor set: {e}"))?;

            cmd.bind_pipeline_compute(stage.pipeline.clone())
                .map_err(|e| format!("bind pipeline: {e}"))?
                .bind_descriptor_sets(PipelineBindPoint::Compute, stage.layout.clone(), 0, set)
                .map_err(|e| format!("bind descriptor: {e}"))?
                .push_constants(stage.layout.clone(), 0, PushFftStage { n, half_: stage_half, batch, stride })
                .map_err(|e| format!("push_constants: {e}"))?
                .dispatch([stage_groups, 1, 1])
                .map_err(|e| format!("dispatch: {e}"))?;
        }

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;
        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?;
        finish_and_disarm(context, future)?;
        drop(tw_buf);
        Ok(())
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

// -- conv (im2col + GEMM) -------------------------------------------------
//
// Two shaders: conv_im2col unfolds the input into a column matrix A (M x K),
// then conv_gemm multiplies A by the flattened kernel and writes the output in
// canonical {N, Cout, O_total} layout. Covers spatial rank <= 3, feature and
// batch groups == 1, identity permutations; the Elixir side gates this and
// host-falls-back otherwise. Per-dim conv parameters ride in a 21-int params
// buffer (see conv_im2col_f64.comp); scalars go in push constants.

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushConvIm2col {
    n: u32,
    cin: u32,
    o_total: u32,
    k_total: u32,
    k: u32,
}

#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushConvGemm {
    n: u32,
    cout: u32,
    o_total: u32,
    k: u32,
}

#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn conv_im2col<'a>(
    env: Env<'a>,
    col_ref: ResourceArc<VulkanoTensor>,
    in_ref: ResourceArc<VulkanoTensor>,
    params_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    cin: u32,
    o_total: u32,
    k_total: u32,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, in_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, col_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, params_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        let groups = (n * o_total).saturating_mul(k).div_ceil(64);
        enqueue_dispatch(
            context,
            &cached,
            set,
            PushConvIm2col { n, cin, o_total, k_total, k },
            [groups, 1, 1],
        )
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

#[rustler::nif(schedule = "DirtyIo")]
#[allow(clippy::too_many_arguments)]
fn conv_gemm<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    col_ref: ResourceArc<VulkanoTensor>,
    kernel_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    cout: u32,
    o_total: u32,
    k: u32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, col_ref.buf.clone()),
                WriteDescriptorSet::buffer(1, kernel_ref.buf.clone()),
                WriteDescriptorSet::buffer(2, out_ref.buf.clone()),
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        // Tiled conv GEMM: C = A·Wᵀ over (M = N·O_total rows, Cout cols), 16×16
        // workgroups. global x over Cout, global y over M.
        let m = n.saturating_mul(o_total);
        let gx = cout.div_ceil(16);
        let gy = m.div_ceil(16);
        enqueue_dispatch(context, &cached, set, PushConvGemm { n, cout, o_total, k }, [gx, gy, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

// -- Deferred dispatch batching -------------------------------------------
//
// Every op used to be its own command buffer, its own vkQueueSubmit, and its
// own `queue.wait_idle()`. On a graph like the Axon MNIST MLP that is one
// full round trip per node, and the round trip — not the arithmetic —
// dominates: `bench_results/MNIST_EXLA_RACE.md` measures 14.1 ms per training
// step eager against EXLA's 0.715 ms on the same model, *and* measures that
// whole-graph fusion makes it worse (0.76×). An optimisation that removes
// work from the shaders cannot explain a gap that fusing the shaders widens,
// so the deficit is per-dispatch cost. This is T1 of PLAN_AFTER_BACKWARD_PASS.md.
//
// So a dispatch is now *recorded* into a pending queue and the queue is
// submitted as one command buffer with one fence wait. Two properties make
// this safe rather than a synchronisation minefield:
//
//  - vulkano's `AutoCommandBufferBuilder` tracks every resource a command
//    touches while recording and inserts the pipeline barriers between them
//    in `build()` (vulkano-0.34.2 `command_buffer/auto/builder.rs:272`). A
//    read-after-write between two batched dispatches is therefore
//    synchronised for us. Do not hand-roll barriers here; do not assume none
//    are needed either.
//  - The only way a value reaches the host is `buf_download`, and the only
//    ways a buffer is mutated behind the GPU's back are `buf_upload_into` and
//    the command buffers built by `concat_buffers` / the leapfrog synth NIFs.
//    Every one of those flushes first, so deferral is invisible to callers.
//
// The builder itself cannot be parked in the static between NIF calls:
// `StandardCommandBufferAllocator` deliberately does not implement `Send` for
// its builder (`command_buffer/allocator.rs:568`) because a command buffer
// may not migrate threads mid-recording, and consecutive NIFs land on
// whichever dirty scheduler is free. So we queue *closures* — `Send`, since
// `BufferContents: Send + Sync + 'static` — and replay them into a builder
// created on whichever thread ends up flushing.
//
// Cost of the deferral: a dispatch's *recording* errors (bind/push
// validation) now surface from the call that flushes rather than the call
// that queued it. Descriptor-set construction, which is where the failures
// actually happen in practice, still runs eagerly in each op's NIF.

type CmdBuilder = AutoCommandBufferBuilder<
    PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
    Arc<StandardCommandBufferAllocator>,
>;

type RecordFn = Box<dyn FnOnce(&mut CmdBuilder) -> Result<(), String> + Send>;

static PENDING: OnceLock<Mutex<Vec<RecordFn>>> = OnceLock::new();

fn pending() -> &'static Mutex<Vec<RecordFn>> {
    PENDING.get_or_init(|| Mutex::new(Vec::new()))
}

static BATCH_MAX: OnceLock<usize> = OnceLock::new();

/// Dispatches to record before forcing a submit. `NXV_BATCH_MAX=0` disables
/// batching entirely, restoring submit-per-dispatch — that is the A/B control
/// for every measurement of this change, and the escape hatch if a driver
/// turns out to dislike long command buffers.
///
/// The cap matters for more than latency: a batch holds all of its descriptor
/// sets alive at once, where the unbatched path dropped each one immediately.
/// vulkano's `StandardDescriptorSetAllocator` grows by allocating additional
/// pools when its 32-slot pool fills (`descriptor_set/allocator.rs:337`) and
/// recycles them through a reserve, so this is pool churn rather than a hard
/// limit — but it is hardware-sensitive, which is why the value is tunable
/// and gets raced across the fleet rather than picked.
fn batch_max() -> usize {
    *BATCH_MAX.get_or_init(|| {
        std::env::var("NXV_BATCH_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(64)
    })
}

/// Flush, wait for the queue to drain, and DISARM the future — on every path,
/// including the failing ones.
///
/// vulkano's `Drop for CommandBufferExecFuture` (`command_buffer/traits.rs:441`)
/// runs a fallback whenever `finished` is false:
///
/// ```ignore
/// self.flush().unwrap();
/// self.queue.with(|mut q| q.wait_idle()).unwrap();
/// ```
///
/// So the obvious sequence — `flush()?`, `wait_idle()?`, then
/// `signal_finished()` — has a trap in it. If either `?` fires, the function
/// returns with the future still armed, Drop re-runs the operation that just
/// failed, and `.unwrap()` converts a returnable error into a **panicked
/// NIF**. A panic in a dirty scheduler is not a failed call; it is a dead
/// BEAM thread and an `:erlang.nif_panicked` at the call site with no
/// indication of the real cause.
///
/// That is not hypothetical. The synthesised chain shader dispatches a single
/// workgroup whose runtime grows with K x n_obs; at n_obs = 600 it ran long
/// enough for the driver's watchdog to reset the device, `wait_idle` returned
/// `DEVICE_LOST`, and the caller saw `:nif_panicked` instead of an error it
/// could fall back from. See docs/TODO_CHAIN_SHADER_BUGS.md.
///
/// Disarming on the error path is safe in the only sense that matters here:
/// the submission has either completed or the device is gone, and Drop's
/// fallback can do nothing but re-run the same failing call. Leaving the
/// future armed cannot recover the device; it can only hide the reason.
fn finish_and_disarm<F: GpuFuture>(context: &VkContext, future: F) -> Result<(), String> {
    let flushed = future.flush().map_err(|e| format!("flush: {e}"));

    // Only wait if the submit itself went through. If flush failed there may
    // be nothing queued, and on a lost device wait_idle just fails again.
    let waited = if flushed.is_ok() {
        context
            .queue
            .with(|mut q| q.wait_idle())
            .map_err(|e| format!("wait_idle: {e}"))
    } else {
        Ok(())
    };

    // SAFETY: mirrors what every call site already asserted on the success
    // path — wait_idle above guarantees completion. On the error path this is
    // disarming a future whose device is lost, which is strictly better than
    // letting Drop panic. Must happen before the future is dropped.
    unsafe {
        future.signal_finished();
    }
    drop(future);

    flushed.and(waited)
}

/// Build, submit, and wait for one command buffer. The `flush` /
/// `wait_idle` / `signal_finished` / `drop` sequence is load-bearing: without
/// the final two the buffers stay marked in use and the next `buf.read()`
/// fails with "resource in use".
fn submit_and_wait(context: &VkContext, cmd: CmdBuilder) -> Result<(), String> {
    let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;
    let future = sync::now(context.device.clone())
        .then_execute(context.queue.clone(), cmd_buf)
        .map_err(|e| format!("then_execute: {e}"))?;
    finish_and_disarm(context, future)?;
    Ok(())
}

fn new_cmd_builder(context: &VkContext) -> Result<CmdBuilder, String> {
    AutoCommandBufferBuilder::primary(
        &context.cmd_allocator,
        context.queue.queue_family_index(),
        CommandBufferUsage::SimultaneousUse,
    )
    .map_err(|e| format!("cmd builder: {e}"))
}

/// Replay every queued dispatch into one command buffer and submit it. The
/// caller holds the pending lock for the whole submission, so two threads
/// cannot interleave halves of each other's batches onto the queue.
///
/// `drain` empties the queue even on the early return, which is deliberate: a
/// batch that failed to record is not retryable, and leaving its commands
/// queued would re-fail every subsequent flush.
fn flush_locked(context: &VkContext, queue: &mut Vec<RecordFn>) -> Result<(), String> {
    if queue.is_empty() {
        return Ok(());
    }

    let mut cmd = new_cmd_builder(context)?;
    for record in queue.drain(..) {
        record(&mut cmd)?;
    }
    submit_and_wait(context, cmd)
}

/// Submit any recorded-but-unsubmitted dispatches and wait for them. Call
/// before anything that reads or writes device memory outside the batch.
fn flush_pending() -> Result<(), String> {
    let context = ctx()?;
    let mut queue = pending().lock().map_err(|_| "pending queue poisoned".to_string())?;
    flush_locked(context, &mut queue)
}

// Shared dispatch helper: bind pipeline + descriptor set + push, then queue
// one dispatch for the next submit (or run it immediately when batching is
// off). Every compute NIF goes through here.
fn enqueue_dispatch<P: BufferContents>(
    context: &VkContext,
    cached: &CachedPipeline,
    set: Arc<PersistentDescriptorSet>,
    push: P,
    groups: [u32; 3],
) -> Result<(), String> {
    let pipeline = cached.pipeline.clone();
    let layout = cached.layout.clone();

    let record: RecordFn = Box::new(move |cmd: &mut CmdBuilder| {
        cmd.bind_pipeline_compute(pipeline)
            .map_err(|e| format!("bind pipeline: {e}"))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
            .map_err(|e| format!("bind descriptor: {e}"))?
            .push_constants(layout, 0, push)
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch(groups)
            .map_err(|e| format!("dispatch: {e}"))?;
        Ok(())
    });

    if batch_max() == 0 {
        let mut cmd = new_cmd_builder(context)?;
        record(&mut cmd)?;
        return submit_and_wait(context, cmd);
    }

    let mut queue = pending().lock().map_err(|_| "pending queue poisoned".to_string())?;
    queue.push(record);
    if queue.len() >= batch_max() {
        flush_locked(context, &mut queue)
    } else {
        Ok(())
    }
}

/// Force any pending dispatches to the GPU and wait for them. Exposed so a
/// benchmark can time the work rather than the recording of it — deferred
/// dispatch will otherwise charge a loop's whole cost to whichever iteration
/// happens to trip the batch cap.
#[rustler::nif(schedule = "DirtyIo")]
fn flush(env: Env) -> NifResult<Term> {
    match flush_pending() {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

/// Physical device name + type, for labelling benchmark/parity reports across
/// hosts (e.g. "NVIDIA GeForce GT 650M" vs "llvmpipe (...)").
#[rustler::nif]
fn device_name(env: Env) -> NifResult<Term> {
    match ctx() {
        Ok(c) => Ok((atoms::ok(), c.device_name.clone(), c.device_type.clone()).encode(env)),
        Err(e) => Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    }
}

/// Whether the physical device advertises `shaderFloat64`. The `_f64.spv`
/// shaders and any generated f64 kernel need it; without it, pipeline creation
/// for those fails at dispatch time, so callers must gate on this and take a
/// host fallback instead.
#[rustler::nif]
fn device_supports_f64(env: Env) -> NifResult<Term> {
    match ctx() {
        Ok(c) => Ok((atoms::ok(), c.supports_f64).encode(env)),
        Err(e) => Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    }
}

// Both lints fire inside the `rustler::resource!` expansion, not in code we
// write: it declares a non-local `impl Resource` and drops the registration
// result. Remove these when rustler emits a corrected macro (same upstream
// tracking as the rustc 1.85 pin in rust-toolchain.toml).
#[allow(unused_must_use)]
#[allow(non_local_definitions)]
fn load(env: rustler::Env, _info: rustler::Term) -> bool {
    rustler::resource!(VulkanoTensor, env);
    true
}

// rustler 0.36 ignores an explicit NIF list (it discovers #[nif] functions
// itself) and warns that passing one is deprecated.
rustler::init!("Elixir.Nx.Vulkan.NativeV", load = load);
