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
        AutoCommandBufferBuilder, CommandBufferUsage,
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

    eprintln!(
        "[nx_vulkan_vulkano] device: {} ({:?})",
        physical.properties().device_name,
        physical.properties().device_type
    );

    // Enable shaderFloat64 if the device supports it; required by the
    // _f64.spv shaders. Falls back gracefully on devices without it
    // (those will keep using the f32 paths + host fallback for f64).
    let supports_f64 = physical.supported_features().shader_float64;

    let enabled_features = vulkano::device::Features {
        shader_float64: supports_f64,
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
    let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
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

        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

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

        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

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

        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

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

    if let Err(e) = sync::now(context.device.clone())
        .then_execute(context.queue.clone(), cmd_buf)
        .map_err(|e| format!("execute: {e}"))
        .and_then(|f| f.flush().map_err(|e| format!("flush: {e}")))
    {
        return Ok(
            (atoms::error(), atoms::dispatch_failed(), e).encode(env),
        );
    }

    if let Err(e) = unsafe { context.device.wait_idle() } {
        return Ok(
            (atoms::error(), atoms::dispatch_failed(), format!("wait_idle: {e}")).encode(env),
        );
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
        let layout = cached.layout.clone();
        let pipeline = cached.pipeline.clone();

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

        let groups = (n + 255) / 256;

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
            .push_constants(layout.clone(), 0, PushN { n })
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([groups, 1, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;

        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

        Ok(())
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
        let layout = cached.layout.clone();
        let pipeline = cached.pipeline.clone();

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

        let groups = (n + 255) / 256;

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
            .push_constants(layout.clone(), 0, PushN { n })
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([groups, 1, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;
        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

        Ok(())
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
        let layout = cached.layout.clone();
        let pipeline = cached.pipeline.clone();

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

        let n_slots = outer * inner;
        let groups = (n_slots + 255) / 256;

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
            .push_constants(
                layout.clone(),
                0,
                PushReduceAxis {
                    outer,
                    reduce_size,
                    inner,
                    op: op_code,
                },
            )
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([groups, 1, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;
        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

        Ok(())
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

        let gx = (n + 15) / 16;
        let gy = (m + 15) / 16;

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
            .push_constants(layout.clone(), 0, PushTranspose { m, n })
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([gx, gy, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;
        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

        Ok(())
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

        let gx = (n + 15) / 16;
        let gy = (m + 15) / 16;

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
            .push_constants(layout.clone(), 0, PushMatmul { m, n, k })
            .map_err(|e| format!("push_constants: {e}"))?
            .dispatch([gx, gy, 1])
            .map_err(|e| format!("dispatch: {e}"))?;

        let cmd_buf = cmd.build().map_err(|e| format!("build cmd: {e}"))?;
        sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .flush()
            .map_err(|e| format!("flush: {e}"))?;

        // SAFETY: we just flushed the only pending command buffer on this
        // queue. device.wait_idle drains all queues but we only have one.
        unsafe { context.device.wait_idle() }.map_err(|e| format!("wait_idle: {e}"))?;

        Ok(())
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}

fn load(env: rustler::Env, _info: rustler::Term) -> bool {
    rustler::resource!(VulkanoTensor, env);
    true
}

rustler::init!(
    "Elixir.Nx.Vulkan.NativeV",
    [
        leapfrog_chain_synth,
        leapfrog_chain_synth_f64,
        leapfrog_chain_synth_batch,
        buf_upload,
        buf_alloc,
        buf_download,
        buf_byte_size,
        buf_upload_into,
        concat_buffers,
        apply_binary,
        apply_unary,
        reduce_axis,
        transpose_2d,
        matmul,
    ],
    load = load
);
