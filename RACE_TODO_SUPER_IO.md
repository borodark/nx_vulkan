# RACE_TODO — run the f32-vs-f64 race on super-io (Ampere)

**For:** the Claude instance / operator on **super-io (192.168.0.249)**, which
has an **Ampere GPU** (consumer RTX → f64 rate-limited to ~1/32 of f32).
**Branch:** `f32-matmul-prototype` (on `origin`, the git server on 249).
**Written by:** Claude on mac-247 after racing on the local GT 650M, 2026-07-29.

## Why

We captured real-GPU numbers on the GT 650M (Kepler) and found that the **f64
accumulator negates the compute-bound f32 speedup**: `matmul_f32_f64acc` is
*slower* than f64 (0.55×), while a pure-f32 accumulator is 1.4–1.7× faster.
Bandwidth-bound ops (elementwise, reductions) win 1.8–4.1× regardless. See
`F32_PLAN.md` → "Real GPU results" and `bench_results/f32_race_mac_970cb1a.json`.

**Confirm the pattern on Ampere.** Consumer Ampere is even more f64-starved
(~1/32), so the hypothesis is: same shape — `matmul_f32_f64acc` ≤ f64, pure-f32
accumulator clearly faster, bandwidth-bound ops a large f32 win.

## Do this

1. On 249, in a working checkout of the repo (not the bare `/home/git/repos`):
   ```sh
   git fetch origin && git checkout f32-matmul-prototype && git pull
   ```
2. Make sure the NVIDIA driver + Vulkan ICD are active (`nvidia-smi` works and a
   Vulkan enumerate finds the Ampere device, not llvmpipe). On this branch a
   quick check is:
   ```sh
   mix run -e 'Application.ensure_all_started(:nx_vulkan); IO.inspect(Nx.Vulkan.NativeV.device_name())'
   # expect {:ok, "NVIDIA GeForce RTX ...", "DiscreteGpu"}, NOT llvmpipe
   ```
3. Run the two races:
   ```sh
   sh scripts/race.sh                              # all families -> bench_results/*.json
   mix run examples/matmul_accumulator_race.exs    # the 3-way accumulator race
   ```
4. Sanity: the whole suite should be green on the Ampere Vulkan —
   `mix test` (expect 171 tests, 0 failures). f64 shaders require
   `shaderFloat64`; Ampere supports it.

## Report back

- Commit the generated `bench_results/f32_race_<host>_<commit>.json` and push:
  ```sh
  git add bench_results/ && git commit -m "race: Ampere (super-io) f32 vs f64 report" && git push origin f32-matmul-prototype
  ```
- Paste the `matmul_accumulator_race.exs` table (it isn't auto-saved) into the
  commit message or a short note, so the f64acc-vs-f32acc-vs-f64 numbers are
  recorded for Ampere too.

## What we expect to learn

If Ampere confirms `matmul_f32_f64acc < f64 < matmul_f32_f32acc`, that settles
the design: land a per-op **accumulator policy** (default f64, opt-in f32 for
compute-bound kernels on rate-limited GPUs) rather than a single fixed
accumulator. If Ampere behaves differently (e.g. full-rate f64 on a workstation
card), record that too — it changes the default.
