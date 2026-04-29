# Nx.Vulkan — Elixir tensor backend over Spirit's Vulkan compute

**Status:** plan + bootstrap, April 2026.
**Predecessor:** Spirit's Vulkan backend (7 shaders, 47 tests, 5
production patterns benched on RTX 3060 Ti — see `RESULTS_RTX_3060_TI.md`
in the spirit repo). The compute is done; this is the Elixir-side
wrapper that exposes it as an `Nx.Backend`.
**Why:** Nx today has two GPU backends — EXLA (XLA, requires CUDA
or TPU) and EMLX (Apple Metal). On FreeBSD with NVIDIA, neither
works. Vulkan is the third backend, and the only one that runs on
the FreeBSD path Igor cares about.

---

## What we're building

```
Elixir application                  Rust NIF                        C++ Spirit Vulkan backend
─────────────────                  ────────                        ─────────────────────────
Nx.tensor(...)                      nx_vulkan_alloc       ─►       buf_alloc + upload
defn function                       nx_vulkan_dispatch    ─►       dispatch shader
Nx.Defn.jit                         nx_vulkan_download    ─►       download

backends:
  Nx.Vulkan.Backend                 Rustler resource:                  VkBuf (lifetime
  ── shape, type, buffer-ref         tensor_handle (Arc<VkBuf>)         tied to ResourceArc)
  ── nx_vulkan/lib/                  nx_vulkan/native/
```

Three layers. Elixir on top, Rust NIF in the middle, C ABI to the
existing C++ backend at the bottom. The Rust layer is mostly glue
— bind to Spirit's existing helpers, expose them as Erlang terms,
manage tensor-handle lifetimes via `ResourceArc`.

---

## Architecture decisions to lock

| # | Decision | Default lean |
|---|---|---|
| 1 | Repo layout | **New repo `nx_vulkan/`**, separate from `spirit/`. Spirit is a physics simulator; nx_vulkan is an Nx backend. Different audiences, different release cadences. nx_vulkan depends on Spirit's Vulkan backend as an external C++ dep (vendored or path dep). |
| 2 | NIF language | **Rust via Rustler.** Rustler handles ResourceArc lifetimes cleanly; Rust's FFI to C is straightforward; the Elixir community standard for serious NIFs is Rustler. C NIF is technically smaller but lifetime management of GPU buffers requires more bookkeeping than is worth it. |
| 3 | C++ → Rust ABI | Spirit's `Backend_par_vulkan.{hpp,cpp}` exposes a C++ API. New `Backend_par_vulkan_c.h` (extern "C") wrapper in Spirit. Rust binds via `bindgen` or hand-written `extern "C"` declarations. |
| 4 | Tensor representation | `%Nx.Vulkan.Backend{shape, type, ref}` where `ref` is a `Reference` to a Rustler-managed `ResourceArc<VulkanTensor>` holding a `VkBuf` + shape metadata. |
| 5 | Lifetime model | **GC-tied.** When the Elixir tensor goes out of scope and is GC'd, the Rust resource's `Drop` impl frees the `VkBuf`. No manual `tensor.free()` needed. |
| 6 | Dtype support | f32 first (matches every shader). f64 second (Spirit shaders support it via spec constant). i32 / u32 deferred (no shader for integer ops yet). |
| 7 | Operator coverage v0.1 | Elementwise (binary + unary), reductions, matmul. Broadcasting via materialize-then-elementwise (until v0.2 wires the broadcast shader). |
| 8 | Lazy vs eager | **Eager.** Each Nx op = one shader dispatch. `Nx.Defn` already builds a graph; we don't need to. |
| 9 | Fallback | Ops we don't yet support fall back to `Nx.BinaryBackend` (CPU). The backend's `__compatible__/2` flag this case so Nx auto-converts. |
| 10 | Out of scope (v0.1) | Autograd, mixed precision, fp16/bf16, sparse tensors, distributed compute, multi-device, kernel fusion, hot reload. Each is a separate iteration. |

---

## Milestones (v0.0 → v0.1)

| # | Milestone | Effort | Deliverable |
|---|---|---|---|
| 1 | **Bootstrap** — `mix new`, Rustler dep, NIF stub returning `:ok` from `vk_init`. | 0.5 d | Project compiles; `Nx.Vulkan.Native.init/0` returns `:ok` on RTX 3060 Ti |
| 2 | **C ABI shim** in spirit + bindgen + Rust wrapper for buf_alloc / upload / download. | 0.5 d | Round-trip `Nx.Vulkan.Native.upload_f32([1.0, 2.0])` returns identical list. |
| 3 | **Tensor type + Backend module skeleton** — `Nx.Vulkan.Backend.from_binary/3` + `to_binary/2`. | 1 d | `Nx.tensor([1.0, 2.0], backend: Nx.Vulkan.Backend)` round-trips through GPU memory. |
| 4 | **Elementwise binary** — wrap Spirit's `apply()` with the binary shader. | 1 d | `Nx.add(a, b)` + `Nx.multiply` etc. work on f32 tensors. |
| 5 | **Elementwise unary** — wrap with the unary shader. | 0.5 d | `Nx.exp/1`, `Nx.log/1`, etc. |
| 6 | **Reductions** — wrap Spirit's `reduce()`. | 0.5 d | `Nx.sum/1`, `Nx.mean/1`, `Nx.reduce_max/2`. |
| 7 | **Matmul** — wrap the naive matmul shader (tiled is a per-call swap). | 0.5 d | `Nx.dot(A, B)` with auto-shape-validation. |
| 8 | **Broadcasting** — start with materialize-then-elementwise; v0.2 swaps in the broadcast shader. | 0.5 d | `Nx.add([[1, 2]], [[10], [20], [30]])` works (3×2 broadcast). |
| 9 | **Nx.Defn integration** — register the backend; `Nx.Defn.jit` dispatches through it. | 1 d | `defn f(a, b) do Nx.add(a, b) end` runs on Vulkan. |
| 10 | **Test suite** — port a subset of Nx's `Nx.Backend` tests; assert parity with `Nx.BinaryBackend`. | 1 d | `mix test` shows 50+ green tests covering the v0.1 surface. |

Total: **~7 person-days** to v0.1. Critical path: 1 → 2 → 3 → 4
→ 9. The reduction / matmul / broadcasting milestones plug into
the same skeleton and parallelize.

---

## v0.1 scope — what works on the day of release

**Tensors:**
- f32 only.
- Up to 4D shapes (matches the broadcast shader's max).
- `Nx.tensor/2`, `Nx.from_binary/3`, `Nx.to_flat_list/1`, `Nx.shape/1`.

**Operations:**

| Family | Ops | Backed by shader |
|---|---|---|
| Elementwise binary | `add`, `subtract`, `multiply`, `divide`, `pow`, `max`, `min` | `elementwise_binary` (+ broadcast variant) |
| Elementwise unary | `exp`, `log`, `sqrt`, `abs`, `negate`, `sigmoid`, `tanh`, `relu`, `ceil`, `floor`, `sign`, `1/x`, `square` | `elementwise_unary` |
| Reductions | `sum`, `mean`, `min`, `max` (all-axis only in v0.1) | `reduce` |
| Linear algebra | `dot/2` (matrix × matrix, matrix × vector) | `matmul` (tiled if M ≥ 256) |
| Random | `Nx.Random.uniform`, `Nx.Random.normal` | `random_philox` |

**Defn integration:**
- `Nx.Defn.default_options(compiler: Nx.Defn.Evaluator, default_backend: Nx.Vulkan.Backend)`
- `defn` functions that use only v0.1 ops dispatch through Vulkan
- Anything else falls back to `Nx.BinaryBackend`

**Targets:**
- Linux + NVIDIA (RTX 3060 Ti baseline; should work on any Ampere/Ada)
- FreeBSD + NVIDIA (the GT 750M proves the path; real-hardware verification on the Mac Pro is post-bootstrap)
- macOS via MoltenVK (untested but the Vulkan layer translates to Metal; should "just work")
- AMD via Mesa RADV (untested but Vulkan-clean)

---

## Risks

1. **Rustler + C++ FFI mismatch.** `bindgen` doesn't always generate clean Rust types from C++ headers (templates, namespaces). Mitigated by writing a hand-rolled `Backend_par_vulkan_c.h` extern "C" shim that exposes a flat C ABI. ~30 LOC of glue.
2. **Buffer lifetime races.** GC-tied lifetime means a tensor used in a dispatch could be freed mid-flight if the Elixir reference is dropped. Vulkan's command-buffer fence-wait inside `dispatch()` blocks until done before returning, so the reference is alive across the dispatch. As long as we don't move to async dispatch, this is safe. (Async dispatch is post-v0.1.)
3. **Persistent buffers vs Nx's ephemeral tensors.** Nx tensors are immutable; every op produces a new one. If we naïvely allocate per-op, we're back to the "anti-pattern" that the persistent-buffer plan kills. Mitigation: **pool of size-classed buffers** at the NIF layer; reuse across short-lived tensors. v0.1 uses the simple direct-alloc path; v0.2 adds the pool.
4. **Persistent pipelines.** Same shape as the reductions API issue in Spirit — pipeline-create per call costs ~22 ms. v0.1 caches pipelines per (shader, spec-constant) pair on first use. ~50 LOC of HashMap in the NIF.
5. **Spirit dependency.** nx_vulkan needs Spirit's `Backend_par_vulkan.cpp` to compile. Two options: vendor it (copy), or path-dep against `~/projects/learn_erl/spirit/`. **Lean: path dep for v0.1**, vendor at v1.0 when Spirit's API stabilizes.

---

## Repo layout (target)

```
nx_vulkan/
  mix.exs                          - elixir_make + rustler deps
  README.md                        - install + first-run
  PLAN.md                          - this file
  Cargo.toml                       - workspace root for the NIF
  native/
    nx_vulkan_native/
      Cargo.toml
      src/
        lib.rs                     - rustler_export_nifs, ResourceArc<VulkanTensor>
        ffi.rs                     - extern "C" bindings to spirit's C ABI
        ops.rs                     - elementwise / reduction / matmul wrappers
  lib/
    nx_vulkan.ex                   - top-level Nx.Vulkan namespace
    nx_vulkan/
      backend.ex                   - implements Nx.Backend
      native.ex                    - Rustler load_nif boilerplate
      tensor.ex                    - %Nx.Vulkan.Backend{} struct
  test/
    nx_vulkan_test.exs             - parity tests vs Nx.BinaryBackend
  c_src/                           - vendored or path-included from spirit/
    Backend_par_vulkan_c.h         - extern "C" shim
    (Backend_par_vulkan.{hpp,cpp} consumed via path dep on first cut)
```

---

## What this DOES NOT cover (deferred from v0.1)

- **Autograd.** No `Nx.Defn.grad`. The forward path is what v0.1 proves.
- **Training loop integration** with Axon. Nx is the foundation; Axon sits on top and is unaffected by which backend Nx uses, but the integration tests require a working backend first.
- **Multi-GPU.** Single device. v0.1 picks the first Vulkan-capable physical device.
- **Hot tensor migration** between backends. v0.1 supports `Nx.backend_transfer/2` via download-then-upload (slow but correct). Direct device-to-device is a v0.2 optimization.
- **Mixed precision.** f32 only. f64 is a spec-constant flip in the shaders but the Nx-side type wiring takes work.
- **Nx 0.10 vs 0.7 compatibility.** Target Nx 0.7+ first (the stable line); 0.10 if it doesn't break the backend interface.

---

## Why this matters

Three concrete payoffs:

1. **Spirit's own Elixir bindings get a tensor type for free.** The Spirit team's plan for an Elixir frontend has been waiting on a GPU tensor library that runs on FreeBSD. nx_vulkan IS that.
2. **Igor's MCMC trader (`exmc`) gets GPU on FreeBSD.** Currently EXLA-on-CUDA on Linux only. With nx_vulkan, it runs on the Mac Pros without compromising on the deployment story.
3. **The blog has a third post.** *The GPU That Doesn't Need CUDA* (the Spirit retrospective) ended with "what this opens up" — Nx.Vulkan is the proof that what was opened up is actually shippable, not aspirational.

---

## Cross-references

- Spirit `feature/vulkan-backend` — the seven shaders this wraps.
- `~/projects/learn_erl/spirit/RESULTS_RTX_3060_TI.md` — the perf numbers.
- `~/projects/learn_erl/spirit/SHADERS_PLAN.md` — what shaders exist, what's deferred.
- `~/projects/learn_erl/spirit/PERSISTENT_BUFFERS_PLAN.md` — the optimization roadmap; the buffer pool in v0.2 is one of those iterations.
- `~/projects/learn_erl/pymc/www.dataalienist.com/blog-vulkan-on-freebsd.html` — the public framing.
- Nx documentation: <https://hexdocs.pm/nx/Nx.Backend.html>
- Rustler documentation: <https://hexdocs.pm/rustler/Rustler.html>
