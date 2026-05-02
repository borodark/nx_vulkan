# mac-248 — Closing Flavor A gaps (parallel work)

**Goal**: now that eXMC ships with `Nx.Vulkan` in `Exmc.JIT`'s
auto-detect priority (Flavor A merged on `pymc/main` at
`fa4739c97`), close two remaining gaps that the README now
explicitly advertises:

1. **FreeBSD bring-up.** mac-248 is the 2013 Mac Pro running
   FreeBSD 15 with a GT 750M — the same machine class that proved
   the Spirit Vulkan path on FreeBSD. nx_vulkan has never actually
   been built and run on FreeBSD; the test suite is green on Linux
   only. This is the milestone that turns the walkable-path post's
   strategic claim into a measurement.
2. **f64 full-axis reduce.** The only Vulkan op missing on the
   gradient hot path; currently falls back to BinaryBackend for
   f64 determinant and Welford accumulators on hosts that need
   double precision.

Branch: `feat/flavor-a-gaps` off current `main` on each repo
(`spirit` for the shader, `nx_vulkan` for the wiring).

Linux side will pick up your push and add the C++ shim, Rust
NIF, backend dispatch, and tests.

## Layout note

Mac-248 (FreeBSD 15, GT 750M) uses the flat layout: `~/spirit/`,
`~/nx_vulkan/`. Both checked out at current main. Pull before
branching:

```
cd ~/spirit && git pull origin main && git checkout -b feat/flavor-a-gaps
cd ~/nx_vulkan && git pull origin main && git checkout -b feat/flavor-a-gaps
```

## Two prototypes (independent — pick whichever interests you,
## or do them in sequence)

### A. FreeBSD bring-up smoke test on mac-248

Empirical proof that the *walkable-path* claim — *"the same
Elixir code that runs CPU-only on a host without a GPU runs
GPU-accelerated on a host with one, on FreeBSD via Vulkan"* —
is actually true. Today the test suite is green on Linux RTX
3060 Ti only. Mac-248's GT 750M on FreeBSD 15 is the right
hardware to close that gap.

Steps:

```
# 1. Install Vulkan loader, headers, validation tools
sudo pkg install vulkan-loader vulkan-headers vulkan-tools \
                 vulkan-validation-layers glslang shaderc

# 2. Confirm the GT 750M is visible
vulkaninfo --summary | head -30
# Should list NVIDIA GeForce GT 750M as a physical device.

# 3. Build the C++ shim + Rust NIF + Elixir
cd ~/nx_vulkan
mix deps.get
mix compile
# If the C++ shim doesn't find vulkan headers, set:
#   CPATH=/usr/local/include LIBRARY_PATH=/usr/local/lib mix compile
# If Rustler can't link libvulkan.so.1, set
#   RUSTFLAGS="-L /usr/local/lib"

# 4. Run the nx_vulkan test suite
mix test

# 5. Run the small leapfrog bench (proves end-to-end)
cd bench && mix run leapfrog_bench.exs
```

Report back:

- `pkg info | grep -i vulkan` — version of the loader actually installed
- Output of `Nx.Vulkan.Native.device_name()` — should name the GT 750M
- Output of `Nx.Vulkan.Native.has_f64()` — true or false (Kepler-class
  GPU; should be true, but the FreeBSD nvidia driver path may differ)
- Any panics, `DEVICE_LOST`, or load-time errors
- `mix test` summary line (NN tests, NN failures, NN excluded)
- `leapfrog_bench` µs/body number — comparable to RTX 3060 Ti's
  ~388 µs (slower expected; we just want a working number)

If everything is green, that's the proof — and the next
*walkable-path* TODO item flips from "promise" to "measurement."
If a specific shader fails validation under FreeBSD's nvidia
driver, note which `.spv` file and the validation error; Linux
side will pick a different lowering or guard the unsupported
feature.

If Rustler can't build (libvulkan version mismatch, missing
headers, etc.), capture the exact build error — Linux side will
adjust the build script to be FreeBSD-compatible.

What this DOES NOT need from you: any code changes upfront. Just
a build + run + report. The FreeBSD-specific Makefile / build
fixes (if any) come back through Linux side based on whatever the
build error says. Budget 30-60 minutes including the package
install and any build-script tweaking.

### B. `reduce_full_f64.spv` — f64 full-axis reduction shader

The only meaningful op on the eXMC gradient hot path that today
forces a BinaryBackend round-trip on f64 inputs. f32 has the
shader; f64 falls back. Closing this lets f64 mass-matrix Welford
accumulators and determinants stay on the GPU.

Mirror the existing `reduce.comp` (f32 full-axis reduce) but
operate on f64. Workgroup tree reduction; output one f64 per
workgroup; caller does host-side final combine over the small
workgroup-count vector. Op selector matches the f32 shader: 0 =
sum, 1 = max, 2 = min.

```glsl
#version 450
#extension GL_ARB_gpu_shader_fp64 : enable

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint n;
    uint op;   // 0 = sum, 1 = max, 2 = min
} pc;

layout (std430, binding = 0) readonly  buffer In_a   { double a[]; };
layout (std430, binding = 1) writeonly buffer Out_y  { double y[]; };

shared double partial[256];

double init_for(uint op) {
    if (op == 0u) return 0.0lf;
    if (op == 1u) return -1.0e308lf;  // -infinity sentinel
    return 1.0e308lf;                 // op == 2: +infinity sentinel
}

double combine(uint op, double a_val, double b_val) {
    if (op == 0u) return a_val + b_val;
    if (op == 1u) return max(a_val, b_val);
    return min(a_val, b_val);
}

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint i = gl_GlobalInvocationID.x;

    double local = init_for(pc.op);
    if (i < pc.n) local = a[i];

    partial[tid] = local;
    barrier();

    for (uint stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) {
            partial[tid] = combine(pc.op, partial[tid], partial[tid + stride]);
        }
        barrier();
    }

    if (tid == 0u) y[gl_WorkGroupID.x] = partial[0];
}
```

Output is `num_workgroups` doubles; Linux backend.ex does the
final combine on the host (cheap — at d=8 it's one or two
workgroups; at d=4096 it's sixteen).

Compile + push:

```
cd ~/spirit
glslangValidator -V shaders/reduce_full_f64.comp -o shaders/reduce_full_f64.spv
git add shaders/reduce_full_f64.{comp,spv}
git commit -m "shaders: reduce_full_f64 — f64 full-axis sum/max/min"
git push origin feat/flavor-a-gaps
```

Devices without f64 support (some integrated Intel parts, certain
mobile Vulkan implementations) will simply not load this shader;
backend.ex already guards on `has_f64()` before dispatching f64
paths, so absence is not a regression — just no gain on those
devices. The GT 750M is Kepler-class and reports f64 = true on
Linux; FreeBSD should be the same.

What this DOES NOT need from you: any C++ shim, Rust NIF, or
backend wiring. The Linux side has the existing `reduce.spv` path
and will mirror it for f64 once your shader lands.

## After your push

Linux side will:

1. Merge `feat/flavor-a-gaps` into `main` on both repos.
2. **(B only)** Add `nxv_reduce_full_f64` C++ shim, `reduce_full_f64`
   Rust NIF, and an f64 branch in `gpu_full_reduce/2`. Tests +
   benchmark.
3. **(A only)** Update the *walkable-path* blog and the nx_vulkan
   README to flip the FreeBSD bring-up TODO from "promise" to
   measurement, citing the device name + bench number you reported.
   If the build needed FreeBSD-specific Makefile changes, those
   land too — and become permanent FreeBSD support, not a one-shot
   patch.

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/RESEARCH_FAST_KERNELS.md` — the
  research note explaining the Fast-vs-IR-walker break-even and
  why named kernels aren't the universal answer (lessons learned).
- `~/projects/learn_erl/nx_vulkan/lib/nx_vulkan/backend.ex:381` —
  `gpu_full_reduce/2` — the dispatch point that needs an f64
  branch once your shader lands.
- `~/projects/learn_erl/pymc/exmc/lib/exmc/jit.ex` — Flavor A
  auto-detect priority (EXLA > EMLX > Nx.Vulkan > nil).
- `~/projects/learn_erl/pymc/www.dataalienist.com/blog-walkable-path.html`
  — the strategic framing this work serves.
