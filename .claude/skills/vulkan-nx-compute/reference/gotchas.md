# Gotchas (learned the hard way in this repo)

## Vulkano / NIF layer

- **Cache pipelines — descriptor-pool exhaustion.** vulkano's
  `StandardDescriptorSetAllocator` creates a fresh `DescriptorPool` per unique
  pipeline-layout identity. Building a fresh `PipelineLayout` every dispatch
  never recycles the pool → driver limits blow (~5000 dispatches on FreeBSD
  NVIDIA: "a non-validation error occurred"). Always go through
  `get_or_create_pipeline(spv_path, op_code)`, keyed by `(spv_path, op_code)` so
  the layout identity is stable. The legacy `matmul` NIF still builds per-call;
  do not copy that.
- **Spec-constant sentinel.** A shader with no `constant_id = 0` spec constant is
  cached with `op_code = None` (key `-1`). Passing `Some(op)` to a shader that
  has no spec constant (or vice-versa) mismatches the pipeline.
- **The `future` dance.** After dispatch: `flush()`, then
  `queue.with(|q| q.wait_idle())`, then `unsafe { future.signal_finished() }`,
  then `drop(future)`. Skipping the drop makes the next `buf.read()`/download see
  "resource in use". `run_single_dispatch` encapsulates this — reuse it.
- **DirtyIo.** All dispatch NIFs are `#[rustler::nif(schedule = "DirtyIo")]` —
  they block on `wait_idle`, so they must not run on a normal scheduler.
- **Ampere DeviceLost.** `StandardCommandBufferAllocator` `primary_buffer_count`
  was bumped to 128 (from 32) after Ampere crashed at 16 dispatches
  (SimultaneousUse buffers not returned fast enough). Do NOT bump the descriptor
  `set_count` — that regressed RTX 3060 Ti small-matmul 6x with no benefit.
- **Feature gating.** `ctx()` enables `shader_float64` and `robust_buffer_access`
  only if the physical device advertises them. f64 shaders silently need the
  former; the u8-mask-as-u32 select/compare shaders need the latter for the tail
  word. A device without f64 keeps f32 paths + host fallback.
- **Device selection** prefers Discrete > Integrated > Virtual > CPU
  (`min_by_key` on `device_type` in `ctx()`), so llvmpipe (CPU) is only chosen
  when nothing else exists.

## Numerical parity

- **f64 sum accumulator for f32 inputs** — `Nx.BinaryBackend` sums in f64; a
  naive f32 accumulator diverges. Every sum/mean kernel here uses `double acc`.
- **Matmul/conv accumulator policy** — default `:f64` accumulator (`*_f64acc`),
  matches an f64 reference to f32 round-off. `:f32` (`*_f32acc`) is opt-in, faster
  on f64-rate-limited GPUs, less accurate.
- **No f64 transcendentals in SPIR-V** (GLSL.std.450 §8) — log/exp/pow/sin/etc.
  Boundary-cast `double(log(float(x)))`; ~7 decimal digits lost per call
  (measured negligible for the leapfrog sampler). Otherwise host-fall-back.
- **Tie-breaking must match `Nx.BinaryBackend`, and clean test data hides it.**
  `window_scatter_max` routes a gradient to the argmax within each window; when
  two elements tie, which one gets it is observable. BinaryBackend's fold keeps
  the **last** maximum, so the shader uses `>=`, not `>`. Random f32 test inputs
  essentially never tie — the ties are in the real data (ReLU outputs are full
  of exact zeros, pooling over a padded border, one-hot labels). **Construct the
  tie deliberately in the test**; do not wait for it to show up.
- **Read the reference before matching it.** Parity means matching
  BinaryBackend's *actual* fold order, including where it is arguably arbitrary.
  Verify what it does rather than what the docs imply it does.
- **Codegen operand substitution** (JIT path, `codegen.ex`): use
  `String.replace(tmpl, ~r/\br\b/, operand)` — a word boundary. A plain
  `String.replace(_, "r", _)` clobbers the `r` in `sqrt`/`round`/`reciprocal`/`erf`.

## Backend layer

- **Always host-fall-back on the else branch.** Guard the GPU path on dtype AND
  shape AND `match?(%__MODULE__{}, tensor.data)`; anything unhandled transfers to
  `Nx.BinaryBackend`, computes, transfers back (`host_result`, the
  `*_host_fallback` helpers). Correctness never depends on the GPU path.
- **Defn params are thunks** (fusion compiler): a runtime arg arrives as a
  zero-arg function — `params[i].()` then `Nx.devectorize`, mirroring
  `Nx.Defn.Evaluator`. `__compile__` returns `fn [params] -> [outputs] end`
  (list-wrapped).
- **Stub arity must match the NIF** exactly or the whole `Nx.Vulkan.NativeV`
  module fails to load (silent-looking `:nif_not_loaded` at call time).
- **A narrow gate is a silent CPU path.** The host fallback returns a
  bit-identical result, so refusing the GPU costs performance and nothing else
  observable. Gate on what the kernel *can't* do, never on the shape you
  happened to test with — see SKILL.md §1b, which is the audit's central finding.
- **`host_result/2` is a macro, and that is deliberate.** It captures
  `__CALLER__.function` at compile time so the fallback counter can attribute
  the op. Erlang's tail-call optimisation discards the caller frame, so a
  runtime stacktrace walk names the wrong function. Shared helpers that are
  several calls deep pass attribution explicitly via `host_result/3`.
- **Nx containers are tuples and maps, never lists.** Passing a list of tensors
  to `Nx.Defn.jit_apply` fails for every arity; dispatch on
  `length(shapes)` and build the right closure.

## Environment

- **llvmpipe vs real GPU.** On a box without a loaded NVIDIA driver, Vulkan falls
  back to llvmpipe (software, CPU). Correct but not a perf signal; 2-3 `select`
  tests fail only on llvmpipe (pre-existing, pass on real NVIDIA). Check
  `Nx.Vulkan.NativeV.device_name()` first. On FreeBSD:
  `doas kldload nvidia nvidia-modeset` (not persistent across reboot).
- **Fleet perf validation.** Win/loss crossovers are hardware-specific (register
  blocking, many-slot reduce). Validate on weak (Kepler GT 650M/750M, or
  llvmpipe) AND strong (Ampere RTX 3060 Ti) — see the `gpu-fleet-and-f32` memory
  for the 247/248/249 hosts and SSH access. `Nx.Vulkan.Device.class/0` +
  `NXV_GPU_CLASS=weak|strong` gate device-dependent heuristics.
- **`git checkout <branch>` does not advance a pre-existing local branch.** On a
  fleet host that already has the branch, checkout silently leaves you on a
  stale commit and you benchmark the wrong code. Use
  `git fetch && git merge --ff-only origin/<branch>` and **verify the printed
  SHA** before trusting any number. This cost a full Kepler benchmark round.
- **Assert the loss is not NaN before believing any timing.** A diverged model
  runs fast. A 635× figure reported here came from a model producing NaN; every
  race row now carries its loss for exactly this reason.
- **A stale `.spv` is silent.** After switching branches, recompile the shaders
  (§3) and `mix compile` — `priv/shaders/*.spv` are committed artifacts and will
  not rebuild themselves.
