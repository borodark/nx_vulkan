# PLAN — f64 chain shader coverage via vulkano synth (Option B)

Filed: 2026-07-05. Superseded 2026-07-09: strategic pivot from
"extend spirit C++ path" (original Surfaces 1-5) to "vulkano synth
subsumes single-family models under f64 default". Kept the file
name for continuity with existing cross-refs.

Cross-refs:
- `../248_TODO.md`
- `pymc/exmc/DECISIONS.md` D87 / D88
- `pymc/exmc/research/175_f64_leapfrog_plan.md`
- `pymc/exmc/research/PLAN_FREEBSD_FLEET.md`
- `pymc/exmc/research/BETA_GAMMA_SYNTH_REGRESSION.md`

## Strategic frame

D88 committed to a vulkano-first (Rust) compute path for the
FreeBSD fleet. The spirit C++ path (`Nx.Vulkan.Native.*`) is on the
retirement track — kept working for backward compat until the
fleet ships, then quietly deprecated. Every new f64 chain shader
we ship should live on vulkano.

Surface 7 (2026-07-09, commit `823e8a96c`) landed f64 support in
the **vulkano synth path**: any Custom-likelihood model — the
regime model, any hand-composed multi-RV logpdf — now samples at
f64 on Vulkan without the D87 f32-accumulation collapse. That
covers 100% of models exmc's `CustomSynth` codegen can emit for.

What's still on f32-only spirit: the six hand-written **family
fused chain shaders** (Normal, Exponential, StudentT, Cauchy,
Weibull, HalfNormal) at `nx_vulkan/priv/shaders/leapfrog_chain_*.spv`.
`ChainShaderCodegen.detect_meta/1` routes single-family models
there, so Vulkan sampling of a single Normal/Exponential/etc.
model still runs the D87 f32 collapse under D88 f64 default.

## Option B: route families through synth under f64

Instead of authoring five more f64 SPVs on the spirit path (the
original Surfaces 1-5), we make `ChainShaderCodegen` emit a synth
meta for single-family models when `Exmc.JIT.precision() == :f64`.
The vulkano synth codegen already handles arbitrary Nx expression
trees; a single-family `Normal(mu, sigma)` model is a subset of
what the regime model already exercises daily.

Payoffs:
1. **Zero new SPVs, NIFs, Elixir wrappers.** All the work is
   exmc-side codegen wiring — one function change in
   `ChainShaderCodegen.detect_meta/1`.
2. **Retires the spirit path faster.** Under D88 f64 default, no
   single-family model reaches spirit anymore. Spirit's family SPVs
   become f32-only fallback for `force_precision: :f32` mode; a
   later cleanup can delete them once no fleet host runs at f32.
3. **Multi-RV Normal at any d works on Vulkan.** Currently
   `SynthUnsupportedError` in the fair race — `detect_meta` fails
   for d>1 single-family models. Synth handles multi-RV natively
   (regime model has 8 free RVs).
4. **Correctness class fixed.** D87's silent-collapse pathology
   can't recur on any single-family model at f64 — same guarantee
   Surface 7 gave the synth-covered set.
5. **Beta / Gamma regression** flagged in
   `pymc/exmc/research/BETA_GAMMA_SYNTH_REGRESSION.md` is
   automatically covered — those cells already go through synth.

Costs:
- **Perf.** Synth generates one long fused kernel per unique
  (family, params) combination; the six hand-written spirit SPVs
  are pre-optimised for their families. Expect the synth path to
  be within 10-20% of the fused path per iteration (Surface 7's
  regime sampling on Kepler is competitive with the family SPVs
  at the same problem size). Not a correctness issue and not on
  the D88 Stage 2 critical path.
- **Codegen fragility.** Synth walks `Nx.Defn.grad` output; adding
  a new family requires no new codegen but does require the
  distribution's `logpdf` to be differentiable through Nx.Defn.
  All 6 current families already are — no new work.

## Scope — one exmc-side change, one nx_vulkan cleanup follow-up

### Surface A (exmc) — `ChainShaderCodegen.detect_meta/1` routes families to synth under f64

Location: `pymc/exmc/lib/exmc/nuts/chain_shader_codegen.ex` (module
name TBD if actual path differs). The current implementation
pattern-matches on IR shape:

- Single free RV, no data → detect family from `Exmc.Dist` module,
  return `{:normal, mu, sigma}` etc.
- Multi-RV or with Custom likelihood → fall through to
  `CustomSynth.synthesise/1`, which returns
  `{:synthesised, sha, layout, push_spec, spv_path, obs_bin}`.

Change: when `Exmc.JIT.precision() == :f64`, skip the single-family
detection and always route to `CustomSynth.synthesise/1`. Under
`:f32` (either default or `force_precision: :f32`), keep the
existing family fast path — it's faster and works.

Concretely:

```elixir
def detect_meta(ir) do
  case Exmc.JIT.precision() do
    :f64 -> route_to_synth(ir)
    :f32 -> detect_family(ir) || route_to_synth(ir)
  end
end
```

`route_to_synth` wraps `CustomSynth.synthesise(ir)` and returns
its `{:synthesised, ...}` tuple. `CustomSynth` was already the
"fallback" — this promotes it to primary at f64.

### Surface B (nx_vulkan, cleanup) — deprecate spirit family NIFs after fleet ships

Not now. After D88 Stage 2 passes (regime on FreeBSD, then per
`PLAN_FREEBSD_FLEET`), delete `nx_vulkan/priv/shaders/leapfrog_chain_
{normal,exponential,studentt,cauchy,weibull,halfnormal}*.spv` and
their `Native.leapfrog_chain_*` NIF stubs. The `_f64` Normal SPV
was already unused; it goes with them.

## Testing plan

- **Sampling parity** (`pymc/exmc/test/exmc/nuts/chain_shader_coverage_test.exs`):
  extend the harness to sample each of the 6 families at
  `force_precision: :f64` via the synth path and match posterior
  moments against BinaryBackend within reference SDs. Same fixture
  Surface 7 uses for regime.
- **Silent-collapse smoke test**: Normal with sigma prior scale
  `0.01`. Under f64 default: samples correctly via synth. Under
  `force_precision: :f32`: samples via family fused shader,
  collapses per D87. Both are the expected behaviour after this
  change.
- **Fair race delta**: rerun `bench/vulkan_only_race_nx_0_12.exs`
  after Surface A lands. Expect the 6 single-family cells to still
  work; expect multi-RV Normal at d=8/d=50 to leave SKIP status
  and produce real numbers. Compare wall-time regression vs
  pre-Surface-A (family-SPV) baseline; if within 20%, ship.

## Success criteria

1. `ChainShaderCodegen.detect_meta/1` routes single-family models
   to synth when `Exmc.JIT.precision() == :f64`.
2. All 6 family sampling tests pass at f64 via synth. Posterior
   moments within reference SDs of BinaryBackend.
3. Silent-collapse smoke test passes: sigma=0.01 Normal at f64
   samples correctly; the same model at `force_precision: :f32`
   collapses (assertion inverted).
4. Vulkan-only race post-Surface-A shows multi-RV Normal at d=8
   and d=50 producing real numbers, not SKIP.
5. `Exmc.NUTS.Vulkan.Dispatch.batch_chain_to_tensors/3`'s explicit
   `:f32` wire type stays — spirit family path is still reachable
   under `force_precision: :f32`. Only the routing changes.

## Non-goals

- **Extending spirit with f64 family SPVs / NIFs.** Explicitly
  rejected in favour of Option B. If mac-248 ever wants to ship
  the 5 files as a side project for parity's sake, fine — but not
  on the fleet critical path.
- **Deleting spirit family SPVs today.** Backward compat until
  D88 Stage 2 lands.
- **Batched f64 synth shader** (multi-instance Task #154 path).
  Same deferral as before.
- **f16 / int8.**

## Blocking dependencies

None. Surface 7 is on main. `CustomSynth.synthesise/1` already
handles single-family models (tested via prior-only regime
components). The change is a one-branch routing decision.

## Owner

Super-io side (single exmc-side change). mac-248's queue stays on
the smaller items: `atan2/3` straggler, BinaryBackend-vs-Vulkan
race, then Option B verification on Kepler once super-io ships
Surface A.

## Cross-references

- `bench/nx_0_12_race_results.md` — pre-Option-B race table.
  Post-Surface-A rerun replaces the multi-RV Normal SKIPs with
  real numbers.
- `pymc/exmc/research/BETA_GAMMA_SYNTH_REGRESSION.md` — Beta and
  Gamma cells already go through synth; Option B doesn't help or
  hurt them. Their regression is a separate profiling problem.
- `pymc/exmc/research/PLAN_FREEBSD_FLEET.md` — Stage 2 no longer
  needs family-f64. Regime model uses Custom likelihood → synth
  path, already handled.
