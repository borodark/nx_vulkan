# Plan: multi-device routing in nx_vulkan

Companion to `CONTEXT_LIFECYCLE_PLAN.md` (single-context tear-down /
rebuild) and roadmap open question #5 (dual-GPU on mac-247). This
doc captures the engineering needed to run nx_vulkan against more
than one Vulkan device in a single BEAM.

The motivating case is mac-247 — a 2013 Mac Pro with both an Intel
HD Graphics 4000 iGPU and an NVIDIA GeForce GT 650M discrete card
on the PCI bus. `pciconf -lv` confirms both:

```
vgapci1: Intel HD Graphics 4000      (0x8086 / 0x0166)   ← iGPU
vgapci0: NVIDIA GT 650M Mac Edition  (0x10de / 0x0fd5)   ← dGPU
```

But `vulkaninfo` only enumerates the NVIDIA card and `llvmpipe`
software. The Intel device exists on the bus but isn't surfaced to
Vulkan because Mesa's `anv` driver isn't currently loaded on
FreeBSD-side. That's the *driver* gap.

Even with both devices visible, the current nx_vulkan code picks
the first `DISCRETE_GPU` it finds and ignores everything else.
That's the *code* gap.

This doc plans both.

## What we'd gain

Honest performance ceiling, GT 650M vs HD 4000 in raw GFLOPS f32:

| Device | f32 GFLOPS | Bus | f64 supported? |
|--------|-----------|-----|----------------|
| GT 650M | ~691 | PCIe 3.0 | yes (1/24 rate) |
| HD 4000 | ~330 | on-CPU | no |

Adding the iGPU to a workload split:
- **Theoretical**: +48% peak compute on mac-247.
- **Realistic**: +20-30% on cleanly-partitioned workloads. Less on
  workloads with cross-device synchronization.
- **Negative case**: small ops can be *slower* with multi-device
  if the router pays per-op overhead exceeding the work.

This is not a transformative speedup. It's incremental, and on
**legacy hardware** that's not the production target. The reason
to do it is:

1. **Demonstration value.** "A 2013 Mac Pro with both GPUs lit up
   on FreeBSD" is a strong narrative for the dataalienist blog.
2. **Foundation work** for genuine multi-device on modern
   systems (super-io has only one RTX 3060 Ti, but Mission III
   could land on a multi-GPU box).
3. **Heterogeneous compute pattern** worth having in nx_vulkan
   regardless of which hardware lights up first.

## Three sub-plans

### Sub-plan 1: FreeBSD driver bring-up (~half a day on mac-247)

Outside of nx_vulkan proper; pure system administration.

```sh
# On mac-247:
pkg install drm-kmod mesa-libs
sysrc kld_list+=i915kms     # load on boot
service kld restart         # or reboot
vulkaninfo --summary        # confirm Intel device enumerated
```

Verify with `vulkaninfo --summary`:
```
GPU0: NVIDIA GeForce GT 650M
GPU1: Intel(R) HD Graphics 4000 (IVB GT2)    ← target
GPU2: llvmpipe (software)                     ← ignore
```

Risks specific to the 2013 MBP:
- **Apple MUX behavior**. The MBP has a hardware multiplexer that
  can route the active GPU to the display. When NVIDIA is active
  the Intel iGPU may be in a low-power or hidden state. Loading
  i915kms might fail with "device busy" or similar. Worst case,
  needs a BIOS/EFI variable poke to enable both simultaneously.
- **NVIDIA driver conflict**. NVIDIA's proprietary 470 driver
  manages the discrete card. i915kms is the in-tree Linux DRM
  driver ported to FreeBSD. They live in different kernel paths,
  but loading order can matter. Test on a non-trial host first.
- **Trial disruption**. The live trader is on mac-247. Don't
  load/unload kernel modules while it's running. Schedule a brief
  stop window.

If this step fails (Apple firmware refuses dual-GPU on FreeBSD),
the whole multi-device plan becomes moot for this hardware. Try
this first; it's the highest-risk and lowest-effort piece.

### Sub-plan 2: device-selectable `ctx()` in nx_vulkan (~3 days)

Refactor `Context` from a singleton to a per-device registry.
Depends on `CONTEXT_LIFECYCLE_PLAN.md`'s ArcSwap work — both go
through the same `OnceLock` → indexable structure transition.

**Step 2.1: device enumeration NIF.**

```rust
#[rustler::nif]
fn list_devices() -> NifResult<Vec<DeviceInfo>> {
    let library = vulkano::VulkanLibrary::new()?;
    let instance = Instance::new(library, InstanceCreateInfo::default())?;

    let devices = instance.enumerate_physical_devices()?
        .map(|pd| DeviceInfo {
            id: pd.properties().device_uuid,
            name: pd.properties().device_name.clone(),
            kind: format!("{:?}", pd.properties().device_type),
            vendor: pd.properties().vendor_id,
            f64: pd.supported_features().shader_float64,
        })
        .collect();

    Ok(devices)
}
```

Elixir side: `Nx.Vulkan.NativeV.list_devices/0` returns a list of
device descriptors. `Nx.Vulkan.devices/0` wraps it.

**Step 2.2: per-device context registry.**

```rust
static CONTEXTS: OnceLock<Mutex<HashMap<DeviceUuid, Arc<Context>>>>;

pub fn ctx_for(uuid: DeviceUuid) -> Result<Arc<Context>, String> {
    let map = CONTEXTS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = map.lock()?;

    if let Some(c) = guard.get(&uuid) {
        return Ok(c.clone());
    }

    let new_ctx = Arc::new(build_context_for(uuid)?);
    guard.insert(uuid, new_ctx.clone());
    Ok(new_ctx)
}

pub fn ctx() -> Result<Arc<Context>, String> {
    // Back-compat: default to whichever device was first registered.
    // Or: pick the most-capable DISCRETE_GPU at init.
    ctx_for(default_device_uuid())
}
```

All existing dispatch functions take an implicit "default device"
unless changed; this keeps the migration backward-compatible.

**Step 2.3: per-device buffer NIFs.**

```rust
#[rustler::nif]
fn buf_upload_on<'a>(env: Env<'a>, device_uuid: Binary<'a>, data: Binary<'a>)
    -> NifResult<ResourceArc<VulkanoTensor>> { ... }
```

For each existing `buf_upload`, `buf_alloc`, `apply_binary`,
`apply_unary`, `reduce_axis`, `matmul`, `leapfrog_chain_synth`,
add a `_on` variant that takes an explicit device UUID. The
default variants keep working for back-compat.

**Step 2.4: VulkanoBackend device tag.**

```elixir
defstruct [:ref, :shape, :type, :device]   # ← new field
```

Every tensor now carries which device it lives on. Operations
between tensors on the same device dispatch on that device;
operations between tensors on different devices either route to
one (configurable policy) or raise a clear error suggesting
explicit `Nx.backend_transfer(t, {VulkanoBackend, device: ...})`.

### Sub-plan 3: routing policy (~2 days)

How do we decide which device runs which op?

**Option A: explicit device tags only.** No router. Caller
chooses device via `Nx.tensor(..., backend: {VulkanoBackend, device: :intel})`
or `Nx.tensor(..., backend: {VulkanoBackend, device: :nvidia})`.
Cross-device ops error out. The user partitions the workload.

- **Pros**: Predictable. Easy to debug. No magic.
- **Cons**: Workload-author burden. Every Nx.tensor call needs
  thought.

**Option B: capability-based routing.** Each op declares
"compute-bound" vs "bandwidth-bound" vs "negligible." Compute-
bound ops route to the most-capable available device (NVIDIA);
bandwidth-bound ops route to the iGPU (which shares system memory
and avoids PCIe). Negligible ops stay where they are.

- **Pros**: Sensible default. Workload partitioning happens
  automatically.
- **Cons**: Magic. Hard to predict performance. Hard to debug
  why a tensor moved.

**Option C: hybrid.** Default routing per option B; explicit
device tag overrides. Best of both worlds — sensible default,
explicit override available.

**Recommended**: C. Default to option B's policy, but always
honor an explicit `device:` opt.

### Sub-plan 4: bench + validate (~1 day)

Extend `examples/vulkano_ops_bench.exs` to take a device parameter.
Run on mac-247 across `:nvidia`, `:intel`, and the hybrid router.
Produce three CSVs.

Look for:
- Compute-bound op crossover sizes per device (matmul on iGPU
  should win below some size where PCIe transfer cost beats
  raw FLOPs).
- Bandwidth-bound op behavior (iGPU's shared memory should be
  faster for memory-heavy ops).
- Hybrid router's wall-time vs naive single-device.

## Sequencing

```
   ┌──────────────────────────────────┐
   │ Sub-plan 1: FreeBSD anv driver   │ ←─ unblocks everything
   │ (half day, can fail on MBP MUX)  │
   └──────────┬───────────────────────┘
              ↓ (if successful)
   ┌──────────────────────────────────┐
   │ CONTEXT_LIFECYCLE_PLAN.md        │ ←─ ArcSwap refactor;
   │ (3 days)                         │     prerequisite
   └──────────┬───────────────────────┘
              ↓
   ┌──────────────────────────────────┐
   │ Sub-plan 2: device-selectable    │
   │ ctx() (3 days)                   │
   └──────────┬───────────────────────┘
              ↓
   ┌──────────────────────────────────┐
   │ Sub-plan 3: routing policy       │
   │ (2 days)                         │
   └──────────┬───────────────────────┘
              ↓
   ┌──────────────────────────────────┐
   │ Sub-plan 4: bench + blog (1 day) │
   └──────────────────────────────────┘
```

Total: about a week and a half elapsed, **assuming sub-plan 1
works**. If the Apple MUX blocks the Intel driver, the whole
plan dies before sub-plan 2 starts. Sub-plan 1 is the gate.

## Total effort and when to do it

- **Optimistic**: 1.5 weeks if everything goes smoothly.
- **Realistic**: 3 weeks if sub-plan 1 needs MUX/firmware investigation.
- **Pessimistic**: blocked indefinitely if Apple firmware refuses
  dual-GPU on FreeBSD; pivot to a different host (modern x86 box
  with NVIDIA + Intel iGPU on the same board).

Prerequisites:
- `CONTEXT_LIFECYCLE_PLAN.md` shipped (provides the ArcSwap
  foundation).
- A non-trial host available for sub-plan 1 driver experiments
  (don't disrupt the live trial on mac-247).
- A maintenance window on mac-247 once the driver path is known
  to work, to load i915kms on the production host.

Don't pick this up until:
1. The eXMC trial demonstrates a need for more compute that
   the GT 650M alone can't provide; OR
2. There's blog-worthy material in the demonstration (a
   "compute fabric from yesterday's hardware" post matching
   the existing dual-GPU narrative); OR
3. Mission III lands on a host with multiple modern GPUs.

Until then: park here, with the ArcSwap refactor (CONTEXT_LIFECYCLE_PLAN.md)
paving the way.

## Open questions to resolve before starting

1. **Default device policy.** When the user creates a tensor
   with `backend: VulkanoBackend` (no `device:` opt), which
   device gets it? First DiscreteGpu found? Always NVIDIA?
   Highest GFLOPS? Configurable via Application env? Pick the
   answer before sub-plan 2.

2. **Cross-device op error message.** What does
   `Nx.add(t_intel, t_nvidia)` raise? Routing-by-policy would
   transfer one tensor; option-A-pure would raise; option-C
   hybrid routes silently unless tagged. Pick the contract
   before sub-plan 3.

3. **Device identity persistence.** Vulkan exposes a
   `device_uuid` that's stable per device per driver
   installation. Use that, or use a higher-level
   `:intel | :nvidia | :virtual` atom? The atom is friendlier
   but breaks if the user has two NVIDIA cards.

4. **f64 fallback.** HD 4000 doesn't support f64 compute.
   What happens when a workload partitioned to the iGPU
   contains an f64 op? Auto-cast to f32 with a warning? Hard
   error? Migrate to the NVIDIA device?

These are design decisions, not engineering — best made before
the implementation work starts.
