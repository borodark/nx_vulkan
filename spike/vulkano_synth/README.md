# vulkano_synth — Mission II R5+ spike

Standalone Rust binary (vulkano 0.34, no C++) that loads a content-
addressed SPV file and runs a K-step leapfrog dispatch against 7
SSBO bindings + a 20-byte push block. Same calling contract as the
`leapfrog_chain_synth` C++ NIF, but every Vulkan resource is owned
by `Arc<Buffer>` / `Arc<Pipeline>` etc., so the stale-handle class
of bugs (which surfaced in R4 step 4 as `Nx.Vulkan.Native.byte_size`
ArgumentError on a freed buffer) is structurally eliminated.

## Spike results (2026-05-19)

Inputs: the regime-model R3 cached SPV (8 free RVs, 200-obs softmax-
mixture custom likelihood), K=32, eps=0.05, d=8, n_obs=200.

| Run | Host | Device | Wall (μs) |
|---|---|---|---|
| vulkano (Rust) | super-io | RTX 3060 Ti | 17,815 |
| vulkano (Rust) | mac-247 | GT 650M Mac Edition | 66,256 |
| C++ spirit | mac-247 | GT 650M Mac Edition | ~60,000 (R3 bench) |

**Correctness:** vulkano's `q_chain`, `p_chain`, `grad_chain`,
`logp_chain` are **byte-identical** to the C++ spirit path's outputs
on the same inputs (K=32, regime model). `cmp -s` returns 0 on all
four files.

## Build

Tested on:
- Linux (super-io, Mesa NVIDIA 575.x): builds in ~30s
- FreeBSD 15.0 (mac-247, NVIDIA legacy driver): builds in ~3:18

```bash
cargo build --release
```

## Run

```bash
./target/release/vulkano_synth_dispatch \
  --spv     <path.spv> \
  --q-init  <q.f32.bin>           # d * 4 bytes \
  --p-init  <p.f32.bin>           # d * 4 bytes \
  --extras  <obs+inv_mass.f32.bin> # (n_obs + d) * 4 bytes \
  --push    <push.bytes>          # ≤128 bytes, opaque \
  --k       <int> \
  --d       <int> \
  --out-q   <path> \
  --out-p   <path> \
  --out-grad <path> \
  --out-logp <path>
```

## Implications for the migration

1. **vulkano can load any SPV the existing pipeline emits.** No
   shader changes needed — the content-addressed cache (the
   `~/.exmc/gpu_node/spv/synth_*.spv` files) is the same artefact
   both runtimes consume.
2. **The 7-SSBO + push-block calling convention maps cleanly** to
   vulkano's `WriteDescriptorSet` + `push_constants` APIs. No need
   to negotiate a new wire protocol.
3. **Performance parity within ~10%** on the bench target hardware
   suggests the C++ spirit path isn't doing anything vulkano can't
   match.
4. **The stale-handle bug class is gone** — Rust ownership manages
   buffer lifetimes; a `Subbuffer<[u8]>` cannot outlive its parent
   `Buffer`.

Next steps (separate session):
- Move from CLI binary to Rustler NIF (vulkano-backed sibling of the
  current `nx_vulkan_native` C++ NIF).
- Migrate the other `nxv_*` ops (apply_binary, reduce, matmul, etc.)
  to vulkano. Each is ~50–100 LOC of vulkano vs ~50 LOC of C++.
- Replace the persistent buffer pool (`ensure_persistent_bufs`,
  `ensure_extras_buf`) with vulkano's `StandardMemoryAllocator` +
  `SubbufferAllocator`.
- Cutover Mission II R4 against the vulkano backend instead of the
  C++ path — the stale-handle crash that blocked it goes away.
