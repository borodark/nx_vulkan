# PARITY_TASK — bring VulkanoBackend to full Nx.Backend parity

**For:** the Claude instance running on **mac-247** (FreeBSD, GT 650M, Vulkan).
**Branch:** `parity-tier1` (you are on it). Commit here; publish with
`git push origin parity-tier1` — `origin` is the git server on **249**
(`git@192.168.0.249:/home/git/repos/nx_vulkan.git`), which is up. That's how
249 reviews your progress.
**Brief written by:** Claude on super-io (249). Updated 2026-07-27: **`conv` and
`fft`/`ifft` are now in scope as native shaders** (see the classification rule).

---

## Objective

Bring `Nx.Vulkan.VulkanoBackend` to full `Nx.Backend` parity except a small
permanent skip set — each op **verified against `BinaryBackend` on this host's
real Vulkan**. Downstream libraries (Scholar, Axon) should stop crashing.
eXMC itself uses none of these, so the trader is not at risk.

## The governing rule — classify every gap op before touching it

Sort each missing callback into exactly one bucket. The test is
**incidental vs. hot-kernel**, NOT "is the shader hard."

1. **Native shader (accelerate on the GPU).** The op *is* the workload's hot
   kernel, so a CPU fallback would be a silent performance cliff — the caller
   thinks they're on the GPU while the expensive part runs on the host. These
   get real f64 Vulkan compute shaders. The 24 existing ops + the fused leapfrog
   live here, and **`conv` + `fft`/`ifft` now join them** (Phase 2 below).

2. **Host fallback (correct-but-CPU, and that's fine).** The op is *incidental*
   — never the bottleneck: reductions, cumulative, shape, logic, sort, window,
   and small linalg (a `cholesky` on a mass matrix once per warmup). A shader
   wouldn't pay for itself; correct-but-CPU beats crashing. Phase 1 below.

3. **Skip (permanent).** Nothing to shader or no supported type:
   `from_pointer`/`to_pointer` (FFI handles — no computation; BinaryBackend
   itself raises) and `phase` (complex — the shader path is f64-*real*, it has
   no complex type to compute with).

A hard-but-hot op (`conv`) gets a shader; an easy-but-incidental op (`window_sum`)
gets a fallback. Do not skip an op just because its shader is hard.

## Step 0 — Baseline (commit before changing any op)

The old `docs/NX_PARITY_RESEARCH.md` / `docs/nx_parity_gap.csv` (2026-05-25) are
**stale** — ~75 callbacks and the host-fallback machinery already exist. Do NOT
work from that CSV.

1. `git checkout parity-tier1`
2. Regenerate the real missing set (path-independent, in `iex -S mix`):
   ```elixir
   impl = Nx.Vulkan.VulkanoBackend.__info__(:functions) |> Enum.map(&elem(&1, 0)) |> MapSet.new()
   cbs  = Nx.Backend.behaviour_info(:callbacks)          |> Enum.map(&elem(&1, 0)) |> MapSet.new()
   MapSet.difference(cbs, impl) |> Enum.sort()
   ```
   Write it to `docs/PARITY_STATUS.md` with today's date, each op tagged with its
   bucket (shader / fallback / skip).
3. Snapshot: `mix compile 2>&1 | grep -i warning` (note the dead `all/3`,
   `any/3`, `to_batched/3` clauses) and `mix test`. Record pass/fail.
4. Commit: `parity: baseline — regenerated gap + bucket classification`.

## Phase 1 — host-fallback the incidental gap (quick wins first)

Reuse the existing machinery in `lib/nx_vulkan/vulkano_backend.ex`:
`host_result/2`, `ensure_on_backend/1`, `binary_op_host_fallback/4`, the
`for op <- [:all, :any]` loop, and `block/4` (routes SVD/QR/LU through
BinaryBackend). Work easy → hard: reduction → cumulative → shape → logic →
sort → window → small linalg. Per callback:

```elixir
@impl true
def <cb>(out, tensor, opts) do
  bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
  host_result(out, apply(Nx, :<cb>, [bin, opts]))
end
```
Multi-tensor ops (`all_close`, `take_along_axis`, `solve`…): transfer each tensor
arg. linalg: route through BinaryBackend / `block/4`. Add a focused test vs a
BinaryBackend reference in **f64**, `Nx.all_close/3`; `mix test` green; commit
`parity: <cb> host-fallback (verified vs BinaryBackend)`.

**Cleanup (do early):** delete the dead duplicate `all/3`, `any/3`,
`to_batched/3` clauses the compiler flags, fix the `to_batched` unused var —
land at zero warnings.

## Phase 2 — native Vulkan shaders: `conv` + `fft`/`ifft` (in scope now)

These are **real f64 Vulkan compute shaders, not host fallbacks** — a fallback
here is the silent cliff from the rule above. Bigger effort than Phase 1 and it
extends into the native-op / Spirit shader layer; land Phase 1 first, and give
each its own series of commits.

- **`conv`** — pragmatic path: **im2col + the existing matmul shader.** Unfold
  the input patches into a matrix and reuse the native GEMM already on the GPU,
  rather than writing a bespoke direct-conv kernel first. Honour full `Nx.conv`
  semantics (strides, padding, input/kernel dilation, feature groups, batch).
  Correctness first; a fused direct-conv shader can come later for memory.
- **`fft`/`ifft`** — radix-2 Cooley–Tukey / Stockham butterfly in f64. Reference
  VkFFT's approach: bit-reversal permutation + log2(N) butterfly stages. Start
  with power-of-two 1-D, then generalise (mixed-radix, then `fft2`).
- **Verify twice:** correct vs BinaryBackend (`Nx.all_close`, f64) **and**
  confirm it actually dispatched on the GPU — i.e. it did *not* silently
  round-trip to the host. A quick way: assert no BinaryBackend transfer on the
  path, or check the dispatch log. GPU-correct AND on-GPU is the bar.

## Skip (permanent) — do NOT implement

`from_pointer`, `to_pointer` (FFI; nothing to compute), `phase` (complex; no
complex type in the f64-real shader ISA). Document in `LIMITATIONS.md` as
"falls back to EXLA / BinaryBackend."
(`fft2` — 2-D FFT — is in scope but comes *after* 1-D `fft` lands, on the same
machinery.)

## Definition of done

- Regenerated gap shows **only the permanent skips** remaining.
- Phase 1 ops: correct vs BinaryBackend; `mix test` green on FreeBSD Vulkan.
- Phase 2: `conv` + `fft`/`ifft` run **on the GPU** (verified not falling back)
  and correct vs BinaryBackend.
- Compile warning-free for these ops; `LIMITATIONS.md` lists the true skips.

## Guardrails

- Stay on `parity-tier1`; commit per op/family; `git push origin parity-tier1`
  (the git server on 249) so 249 can review.
- **Phase 1** — confine to `vulkano_backend.ex`, `test/`, docs (pure Elixir).
  **Phase 2** — expect to touch the native shader / Spirit layer for `conv`/`fft`;
  that's intended, but keep it on this branch.
- FreeBSD + Vulkan (this host) IS the reference the project is validated against.
  Always run the real `mix test` here — never a mock.
