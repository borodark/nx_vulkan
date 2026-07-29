# RACE_TODO — run the f32-vs-f64 race on super-io (Ampere)

**For:** the Claude instance / operator on **super-io (192.168.0.249)**, which
has an **Ampere GPU** (consumer RTX → f64 rate-limited to ~1/32 of f32).
**Branch:** `f32-matmul-prototype` (on `origin`, the git server on 249).
**Written by:** Claude on mac-247 after racing on the local GT 650M, 2026-07-29.
**Updated:** 2026-07-29 — the f32 **accumulator policy** is now implemented; this
run is to validate it on Ampere and decide the default.

> **STATUS: DONE (round 1); PLEASE RE-RACE round 2.** super-io ran this on an
> RTX 3060 Ti — see `bench_results/AMPERE_SUPER_IO_RESULTS.md`. Pattern confirmed
> (`:f64acc ≤ f64 ≤ :f32acc`); default kept `:f64`, `:f32` opt-in. conv-GEMM now
> shares the policy; the step-4 snippet is fixed; the JSON records the
> accumulator.
>
> **Round 2 ask — re-race the tiled kernels on the RTX 3060 Ti.** Since round 1,
> **every GEMM is now 16×16 shared-memory tiled**: both f32 matmul shaders, all
> three conv-GEMM shaders, and the f64 matmul. On the GT 650M this fixed the
> 1024³ `:f32acc` cliff (1.72×→**2.68×** at 512³, holds **2.66× at 1024³**, was
> ~1.1× on your card), gave conv `:f32acc` **1.4–2.0×** on larger convs, and made
> f64 matmul ~1.35–1.4× faster (still f64-exact). Please, on Ampere:
>
> 1. `sh scripts/race.sh` + `mix run examples/matmul_accumulator_race.exs` +
>    step-4 (below) — but **add n=1024 and n=2048** to the size sweep to confirm
>    the cliff is gone and see where (if anywhere) it reappears.
> 2. Race **conv** `:f64acc` vs `:f32acc` on a couple of real layers, e.g.
>    `{8,32,28,28}·{64,32,3,3}` and `{4,64,16,16}·{128,64,3,3}`.
> 3. Note the **f64 matmul** speedup from tiling (compare against round 1's f64
>    numbers in `AMPERE_SUPER_IO_RESULTS.md`).
> 4. Commit an `AMPERE_SUPER_IO_RESULTS_R2.md` with the tables + verdict.
>
> **The decision this reopens:** if tiled `:f32acc` is a size-stable ~2–2.5× on
> Ampere across 512–2048, is a device-aware default (`:f32` on detected
> f64-rate-limited cards) now worth it? And is a larger tile / register-blocked
> (e.g. 32×32 or 8×8-per-thread) kernel worth chasing for another step? Your call
> with the data.

## Why

Real GT 650M (Kepler) numbers showed the **f64 accumulator negates the
compute-bound f32 speedup**: f32 matmul with an f64 accumulator is *slower* than
f64, while a pure-f32 accumulator is faster. Bandwidth-bound ops (elementwise,
reductions) win regardless. On the GT 650M, via the real `Nx.dot` path, 512³:

```
f64 = 21.3ms   f32[:f64acc] = 38.5ms (0.55x)   f32[:f32acc] = 12.4ms (1.72x)
add 1M 4.14x   tanh 1.95x   sum 1.8-1.9x   conv 1.08-1.35x
```

Since then the branch shipped an **accumulator policy** (default `:f64`, opt-in
`:f32`) — see `F32_PLAN.md` → "Item 5". **Confirm the pattern and the policy on
Ampere** (even more f64-starved, ~1/32): expect `:f64acc ≤ f64 < :f32acc`, and a
large bandwidth-bound win.

## What's on the branch now (f32 surface)

- f32 GPU path for matmul/`dot`, conv, elementwise (unary+binary), reductions,
  2-D transpose — dispatched by tensor dtype (f64 stays default).
- **Accumulator policy** for f32 matmul:
  `Nx.Vulkan.VulkanoBackend.put_f32_matmul_accumulator(:f32 | :f64)` /
  `f32_matmul_accumulator/0`, or `config :nx_vulkan, :f32_matmul_accumulator`.
  `:f64` → `matmul_f32_f64acc.spv`, `:f32` → `matmul_f32_f32acc.spv`.
- `device_name` NIF labels reports; `scripts/race.sh` is the one-command trigger.

## Do this

1. Working checkout on 249 (not the bare `/home/git/repos`):
   ```sh
   git fetch origin && git checkout f32-matmul-prototype && git pull
   ```
2. Confirm Vulkan sees the Ampere device, not llvmpipe:
   ```sh
   mix run -e 'Application.ensure_all_started(:nx_vulkan); IO.inspect(Nx.Vulkan.NativeV.device_name())'
   # expect {:ok, "NVIDIA GeForce RTX ...", "DiscreteGpu"}
   ```
3. Run the races:
   ```sh
   sh scripts/race.sh                              # all families -> bench_results/*.json
   mix run examples/matmul_accumulator_race.exs    # 3-way: f64 vs f32/f64acc vs f32/f32acc
   ```
4. **Validate the policy through the real `Nx.dot` path** (this is the new bit).
   Use enough warm-up + iterations and **interleave** the three configs across
   rounds — a single-shot, config-ordered measurement is noise (the f64 path on
   these cards is jittery). This form allocates tensors once and averages 20
   timed iters after 5 warm-ups:
   ```sh
   mix run -e '
     Application.ensure_all_started(:nx_vulkan)
     alias Nx.Vulkan.VulkanoBackend, as: V
     rnd = fn n -> for i <- 1..n, do: :math.sin(i*0.01) end
     mk = fn ty -> a = Nx.tensor(rnd.(512*512), type: ty, backend: V) |> Nx.reshape({512,512})
                   b = Nx.tensor(rnd.(512*512), type: ty, backend: V) |> Nx.reshape({512,512}); {a,b} end
     time = fn {a,b} -> for _<-1..5, do: Nx.dot(a,b)
                        {us,_}=:timer.tc(fn -> for _<-1..20, do: Nx.dot(a,b) end); us/20/1000 end
     f = mk.({:f,64}); g = mk.({:f,32})
     for r <- 1..3 do
       f64 = time.(f)
       V.put_f32_matmul_accumulator(:f64); a64 = time.(g)
       V.put_f32_matmul_accumulator(:f32); a32 = time.(g)
       IO.puts("round #{r}: f64=#{Float.round(f64,3)}  :f64acc=#{Float.round(a64,3)} (#{Float.round(f64/a64,2)}x)  :f32acc=#{Float.round(a32,3)} (#{Float.round(f64/a32,2)}x)")
     end
   '
   ```
   Expect `:f32acc` ≈ 1.5–2× at 512³ and `:f64acc` < 1× (both differing ~3×,
   proving the policy is honoured). If a config-ordered single run shows a
   `:f32acc` regression, that's the measurement, not the shader.
5. Sanity: `mix test` should be green on Ampere Vulkan (expect **174 tests, 0
   failures**; f64 shaders need `shaderFloat64`, which Ampere supports).

## Report back

- Commit the generated `bench_results/f32_race_<host>_<commit>.json` and push:
  ```sh
  git add bench_results/ && git commit -m "race: Ampere (super-io) f32 vs f64 report" && git push origin f32-matmul-prototype
  ```
- Paste the `matmul_accumulator_race.exs` table and the step-4 policy line into
  the commit message or a note (they aren't auto-saved), so the Ampere
  f64 / :f64acc / :f32acc numbers are on record.

## The decision this run drives

- If Ampere confirms `:f64acc ≤ f64 < :f32acc` (expected), the accumulator policy
  is validated across two GPU generations. **Open question to answer with the
  data:** should the *default* accumulator stay `:f64` (accuracy-safe) or flip to
  `:f32` on detected f64-rate-limited devices? Record the ratio; if `:f32acc` is
  a large, consistent win with acceptable error on real workloads, a
  device-aware default becomes worth wiring.
- Next f32 step if confirmed: give conv's GEMM the same policy
  (`conv_gemm_f32_f32acc`), since it's the other compute-bound kernel.
- If Ampere behaves differently (e.g. a workstation card with full-rate f64),
  record that — it changes the default and the recommendation.
