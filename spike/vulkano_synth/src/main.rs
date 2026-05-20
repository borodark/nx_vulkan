//! Mission II R5+ vulkano spike — dispatcher for the synthesised chain
//! shader. Loads a content-addressed SPV from disk, allocates 7 SSBOs
//! (3 read + 4 write), uploads the 3 input bytes, runs the compute
//! pipeline K leapfrog steps, downloads the 4 output bytes.
//!
//! CLI:
//!   vulkano_synth_dispatch \
//!     --spv     <path.spv> \
//!     --q-init  <q.f32.bin>    (d * 4 bytes) \
//!     --p-init  <p.f32.bin>    (d * 4 bytes) \
//!     --extras  <extras.f32.bin> ((n_obs + d) * 4 bytes) \
//!     --push    <push.bytes>   (≤128 bytes, opaque) \
//!     --k       <int> \
//!     --d       <int> \
//!     --out-q   <path>  \
//!     --out-p   <path>  \
//!     --out-grad <path> \
//!     --out-logp <path>
//!
//! Output binaries are f32 little-endian, sizes:
//!   q_chain, p_chain, grad_chain: K * d * 4 bytes
//!   logp_chain: K * 4 bytes

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

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
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    sync::{self, GpuFuture},
    VulkanLibrary,
};

fn parse_args() -> HashMap<String, String> {
    let mut out = HashMap::new();
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i + 1 < args.len() {
        let k = args[i].trim_start_matches("--").to_string();
        let v = args[i + 1].clone();
        out.insert(k, v);
        i += 2;
    }
    out
}

fn read_file_bytes(p: &str) -> Vec<u8> {
    fs::read(PathBuf::from(p)).expect("read input file")
}

/// Synth template's push block — matches the R2.2.4 fixed header.
#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushBlock {
    k_steps: u32,
    n_obs: u32,
    d: u32,
    _pad: u32,
    eps: f32,
}

fn parse_push_block(bytes: &[u8]) -> PushBlock {
    assert!(
        bytes.len() >= 20,
        "push block must be >= 20 bytes, got {}",
        bytes.len()
    );
    let u32_at = |off: usize| {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    let f32_at = |off: usize| {
        f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    };
    PushBlock {
        k_steps: u32_at(0),
        n_obs: u32_at(4),
        d: u32_at(8),
        _pad: u32_at(12),
        eps: f32_at(16),
    }
}

fn write_file_bytes(p: &str, bytes: &[u8]) {
    fs::write(PathBuf::from(p), bytes).expect("write output file");
}

fn main() {
    let args = parse_args();

    let spv_path = args.get("spv").expect("--spv required");
    let q_init_path = args.get("q-init").expect("--q-init required");
    let p_init_path = args.get("p-init").expect("--p-init required");
    let extras_path = args.get("extras").expect("--extras required");
    let push_path = args.get("push").expect("--push required");
    let k: u32 = args
        .get("k")
        .expect("--k required")
        .parse()
        .expect("k integer");
    let d: u32 = args
        .get("d")
        .expect("--d required")
        .parse()
        .expect("d integer");
    let out_q = args.get("out-q").expect("--out-q required");
    let out_p = args.get("out-p").expect("--out-p required");
    let out_grad = args.get("out-grad").expect("--out-grad required");
    let out_logp = args.get("out-logp").expect("--out-logp required");

    let spv_bytes = read_file_bytes(spv_path);
    let q_bytes = read_file_bytes(q_init_path);
    let p_bytes = read_file_bytes(p_init_path);
    let extras_bytes = read_file_bytes(extras_path);
    let push_bytes = read_file_bytes(push_path);

    eprintln!(
        "[vulkano-spike] inputs: spv={}b q={}b p={}b extras={}b push={}b K={} d={}",
        spv_bytes.len(),
        q_bytes.len(),
        p_bytes.len(),
        extras_bytes.len(),
        push_bytes.len(),
        k,
        d
    );

    let chain_bytes = (k as usize) * (d as usize) * 4;
    let logp_bytes = (k as usize) * 4;

    // --- Vulkano init ---
    let library = VulkanLibrary::new().expect("no Vulkan library");
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            ..Default::default()
        },
    )
    .expect("create Instance");

    let (physical, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
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
        .expect("no compute-capable device");

    eprintln!(
        "[vulkano-spike] device: {} ({:?})",
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
    .expect("create Device");

    let queue = queues.next().unwrap();

    // --- Allocators ---
    let mem_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let cmd_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));
    let set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    // --- Load SPV ---
    let words: Vec<u32> = bytemuck_cast(&spv_bytes);
    let shader = unsafe {
        ShaderModule::new(
            device.clone(),
            ShaderModuleCreateInfo::new(&words),
        )
    }
    .expect("ShaderModule from SPV");

    let entry_point = shader.entry_point("main").expect("'main' entry point");

    // Build pipeline layout via reflection from the shader stages.
    let stage = PipelineShaderStageCreateInfo::new(entry_point);

    let layout_info = PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
        .into_pipeline_layout_create_info(device.clone())
        .expect("infer layout from shader");

    let layout = PipelineLayout::new(device.clone(), layout_info).expect("PipelineLayout");

    let pipeline = ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout.clone()),
    )
    .expect("ComputePipeline");

    // --- Buffer allocation ---
    let q_buf = upload_buffer(mem_allocator.clone(), &q_bytes, BufferUsage::STORAGE_BUFFER);
    let p_buf = upload_buffer(mem_allocator.clone(), &p_bytes, BufferUsage::STORAGE_BUFFER);
    let extras_buf = upload_buffer(
        mem_allocator.clone(),
        &extras_bytes,
        BufferUsage::STORAGE_BUFFER,
    );

    let q_chain_buf = alloc_buffer(
        mem_allocator.clone(),
        chain_bytes,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
    );
    let p_chain_buf = alloc_buffer(
        mem_allocator.clone(),
        chain_bytes,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
    );
    let grad_chain_buf = alloc_buffer(
        mem_allocator.clone(),
        chain_bytes,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
    );
    let logp_chain_buf = alloc_buffer(
        mem_allocator.clone(),
        logp_bytes,
        BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_SRC,
    );

    let set = PersistentDescriptorSet::new(
        &set_allocator,
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
    .expect("descriptor set");

    // --- Command buffer ---
    let mut cmd = AutoCommandBufferBuilder::primary(
        &cmd_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let push_block = parse_push_block(&push_bytes);
    eprintln!(
        "[vulkano-spike] push: K={} n_obs={} d={} eps={}",
        push_block.k_steps, push_block.n_obs, push_block.d, push_block.eps
    );

    cmd.bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            layout.clone(),
            0,
            set.clone(),
        )
        .unwrap()
        .push_constants(layout.clone(), 0, push_block)
        .unwrap()
        .dispatch([1, 1, 1])
        .unwrap();

    let cmd_buf = cmd.build().unwrap();

    // --- Submit + wait ---
    let t_submit = Instant::now();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), cmd_buf)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
    let elapsed_us = t_submit.elapsed().as_micros();
    eprintln!("[vulkano-spike] dispatch wall: {} µs", elapsed_us);

    // --- Download outputs ---
    write_file_bytes(out_q, &download_buffer(q_chain_buf));
    write_file_bytes(out_p, &download_buffer(p_chain_buf));
    write_file_bytes(out_grad, &download_buffer(grad_chain_buf));
    write_file_bytes(out_logp, &download_buffer(logp_chain_buf));

    eprintln!("[vulkano-spike] wrote 4 output files");
}

fn upload_buffer(
    alloc: Arc<StandardMemoryAllocator>,
    bytes: &[u8],
    usage: BufferUsage,
) -> Subbuffer<[u8]> {
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
    .expect("upload buffer")
}

fn alloc_buffer(
    alloc: Arc<StandardMemoryAllocator>,
    n_bytes: usize,
    usage: BufferUsage,
) -> Subbuffer<[u8]> {
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
    .expect("alloc buffer")
}

fn download_buffer(buf: Subbuffer<[u8]>) -> Vec<u8> {
    let guard = buf.read().expect("read buffer");
    guard.to_vec()
}

// Minimal u8→u32 word cast (SPV is u32-aligned).
fn bytemuck_cast(bytes: &[u8]) -> Vec<u32> {
    assert_eq!(bytes.len() % 4, 0, "SPV must be u32-aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
