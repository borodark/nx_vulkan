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

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
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

    let spv_bytes = match fs::read(&spv_path) {
        Ok(b) => b,
        Err(_) => return Ok((atoms::error(), atoms::spv_read_failed()).encode(env)),
    };
    let spv_words = match bytes_to_u32_words(&spv_bytes) {
        Ok(w) => w,
        Err(_) => return Ok((atoms::error(), atoms::bad_input()).encode(env)),
    };

    let result = (|| -> Result<(Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>), String> {
        let shader = unsafe {
            ShaderModule::new(context.device.clone(), ShaderModuleCreateInfo::new(&spv_words))
                .map_err(|e| format!("ShaderModule: {e}"))?
        };

        let entry_point = shader
            .entry_point("main")
            .ok_or_else(|| "no main entry point".to_string())?;

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
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
            CommandBufferUsage::OneTimeSubmit,
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
            .map_err(|e| format!("then_execute: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("then_signal_fence_and_flush: {e}"))?;

        future.wait(None).map_err(|e| format!("wait: {e}"))?;

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
        let spv_bytes = fs::read(&spv_path).map_err(|e| format!("read spv: {e}"))?;
        let spv_words = bytes_to_u32_words(&spv_bytes)?;

        let shader = unsafe {
            ShaderModule::new(context.device.clone(), ShaderModuleCreateInfo::new(&spv_words))
                .map_err(|e| format!("ShaderModule: {e}"))?
        };

        // Specialize the shader module with op_code at constant ID 0.
        // vulkano expects ahash::HashMap, not std::HashMap.
        let mut spec_map: ahash::HashMap<u32, SpecializationConstant> =
            ahash::HashMap::default();
        spec_map.insert(0, SpecializationConstant::I32(op_code as i32));
        let specialized = shader
            .specialize(spec_map)
            .map_err(|e| format!("specialize: {e}"))?;

        let entry = specialized
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

        let groups = (n + 255) / 256;

        let mut cmd = AutoCommandBufferBuilder::primary(
            &context.cmd_allocator,
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
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

        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("fence: {e}"))?;
        future.wait(None).map_err(|e| format!("wait: {e}"))?;

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
        let spv_bytes = fs::read(&spv_path).map_err(|e| format!("read spv: {e}"))?;
        let spv_words = bytes_to_u32_words(&spv_bytes)?;

        let shader = unsafe {
            ShaderModule::new(context.device.clone(), ShaderModuleCreateInfo::new(&spv_words))
                .map_err(|e| format!("ShaderModule: {e}"))?
        };

        let mut spec_map: ahash::HashMap<u32, SpecializationConstant> =
            ahash::HashMap::default();
        spec_map.insert(0, SpecializationConstant::I32(op_code as i32));
        let specialized = shader
            .specialize(spec_map)
            .map_err(|e| format!("specialize: {e}"))?;

        let entry = specialized
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

        let groups = (n + 255) / 256;

        let mut cmd = AutoCommandBufferBuilder::primary(
            &context.cmd_allocator,
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
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
        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("fence: {e}"))?;
        future.wait(None).map_err(|e| format!("wait: {e}"))?;

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

        let n_slots = outer * inner;
        let groups = (n_slots + 255) / 256;

        let mut cmd = AutoCommandBufferBuilder::primary(
            &context.cmd_allocator,
            context.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
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
        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("fence: {e}"))?;
        future.wait(None).map_err(|e| format!("wait: {e}"))?;

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
            CommandBufferUsage::OneTimeSubmit,
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
        let future = sync::now(context.device.clone())
            .then_execute(context.queue.clone(), cmd_buf)
            .map_err(|e| format!("then_execute: {e}"))?
            .then_signal_fence_and_flush()
            .map_err(|e| format!("fence: {e}"))?;
        future.wait(None).map_err(|e| format!("wait: {e}"))?;

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
        buf_upload,
        buf_alloc,
        buf_download,
        buf_byte_size,
        buf_upload_into,
        apply_binary,
        apply_unary,
        reduce_axis,
        transpose_2d,
    ],
    load = load
);
