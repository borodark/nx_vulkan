# Mission — nx_vulkan

**Written:** 2026-08-16, from `main` at `4017f9b`, on super-io (RTX 3060 Ti,
Linux). **Audience:** whoever picks this repo up next, including a session with
no memory of the one that wrote this.

This is the layer above [`PLAN_AFTER_BACKWARD_PASS.md`](PLAN_AFTER_BACKWARD_PASS.md).
That document is a worklist with a "done when" per item; this one says what the
project is *for*, what "finished" would mean, and which of the available work is
worth doing. Where the two disagree, this one is newer.

## Read these first, in this order

0. **[`NEXT.md`](NEXT.md)** — current state and what to do next. Shorter than
   this file and written later, so where the two disagree about *state*, NEXT
   wins. It also carries the two things that waste the most time if you do not
   know them: `origin` is the private server and `upstream` publishes, and
   `rm -rf _build/` is fine to run without ceremony.
1. **§9 of this document — "contradictory or stale".** Start there, not at the
   top. Seven items, and three of them are numbers you will otherwise trust:
   `README`, `ROADMAP` and `T11` circulate three *different* suite counts, all
   wrong (actual: **833 doctests, 871 tests**), and `ROADMAP.md` still carries
   the pre-reframing performance goal *above* the reach-not-speed paragraph, so
   stopping early gets you the old mission.
2. **This document**, §1–§8.
3. [`bench_results/MODEL_SCALING.md`](bench_results/MODEL_SCALING.md) — the
   measurement the mission rests on: EXLA on the *host CPU* beats this backend
   20–215× at every size tested. If you internalise one number, that is it.
4. [`PLAN_AFTER_BACKWARD_PASS.md`](PLAN_AFTER_BACKWARD_PASS.md) — the T-items
   the W-list below maps onto.
5. [`docs/BACKWARD_PASS_AUDIT.md`](docs/BACKWARD_PASS_AUDIT.md) §1b and
   [`.claude/skills/vulkan-nx-compute/SKILL.md`](.claude/skills/vulkan-nx-compute/SKILL.md)
   — the defect class this repo keeps producing (a gate written against the
   shapes one workload happens to emit). Nine instances so far. You will meet
   the tenth.

**Before you measure anything:** this GPU is shared. Check `uptime`; a
concurrent test suite has inflated an established figure by 43% here. And the
GT 750M has ±11–13% run-to-run spread against the GT 650M's ±2–4%
([`bench_results/CONCURRENT_DISPATCH.md`](bench_results/CONCURRENT_DISPATCH.md))
— a 15% effect measured once on that host is a coin flip. Race on the 650M, or
replicate.

**If you are editing this repo while an eXMC session is running:** both eXMC
repos path-depend on this one, so your half-finished `lib/` breaks their build.
Work in a git worktree. That is not hypothetical advice — it was necessary
twice on the day this was written.

Everything below is labelled **measured** (with the file that holds the numbers)
or **inferred** (a reading of the source, or an argument). The distinction is
load-bearing: this repo has twice built work on an inference that a measurement
later killed — the fallback explanation for fusion (§4), and the assumption that
the chain shader was earning its keep
([`bench_results/EXMC_PEROP_RACE.md`](bench_results/EXMC_PEROP_RACE.md) banner).

---

## 1. The mission

**nx_vulkan runs Nx on GPUs that no other Nx backend can reach — NVIDIA on
FreeBSD, decade-old Keplers, AMD and Intel parts, anything with a Vulkan driver
— and its goal is *completeness*: every Nx operation running correctly and
on-device on that hardware.** Reach is only worth having if the backend does not
fall off a cliff the first time a model touches an op nobody thought about, and
on that hardware the thing it falls back to is a pure-BEAM interpreter, so a gap
in coverage is a two-order-of-magnitude cliff rather than a rounding error.

### 1.1 Why completeness, and not speed

Because speed against the reference that matters is already won, and speed
against the reference that does not matter is unreachable. Both halves are
measured.

**Against `Nx.BinaryBackend` there is a real crossover, and it is large.**
A width sweep on the same gradient found the fallback-free per-op Vulkan path
overtaking the interpreter at roughly **10³ f64 elements** and reaching **410×
by 4×10⁵ elements**
([`bench_results/MODEL_SCALING.md`](bench_results/MODEL_SCALING.md), Result 1).
On a training step the same shape holds: a LeNet step went from 20.9 s to 84 ms
once its backward pass stopped leaving the device
([`docs/BACKWARD_PASS_AUDIT.md`](docs/BACKWARD_PASS_AUDIT.md) §3).

**Against a compiler there is no crossover anywhere reachable.** EXLA on
super-io's *host CPU* — not its GPU — computes the same gradient **20× faster
than this backend at small model sizes and 215× faster at 6×10⁶ elements, with
the gap widening** (`MODEL_SCALING.md`, Result 4). `exla_cuda` is
indistinguishable from `exla_host` on that workload even at 6×10⁷ elements, so
this is not a CUDA advantage — it is a compiled-versus-interpreted-dispatch
advantage, and nothing in this repo's roadmap closes two orders of magnitude.

So the honest case is **reach**: the FreeBSD Keplers cannot build EXLA at all
([`WHY.md`](WHY.md)), and there `BinaryBackend` genuinely is the alternative.
That is a real case. It is a portability case, and it should be argued as one.

### 1.2 Where speed matters, say against what

Three baselines, and every performance claim in this repo must name which one it
is using:

| baseline | what it is for | status |
|---|---|---|
| **`Nx.BinaryBackend`** | the only honest performance reference **on the Kepler fleet**, where it is what the user would otherwise run. Also the correctness reference everywhere, since it *is* the fallback path | won above ~10³ elements, measured |
| **this backend, previous commit** | regression control. Batched dispatch was accepted on this basis (1.45–1.71× across three hosts, [`bench_results/BATCHED_DISPATCH.md`](bench_results/BATCHED_DISPATCH.md)) | the standing bar for any perf change |
| **EXLA** | **not a target.** Where CUDA exists, use EXLA — the README says so above its own benchmark tables | conceded, measured |

A multiplier quoted against `BinaryBackend` on a CUDA-capable box is not a
performance result. It is a statement about a tree-walking interpreter, and it
flatters this project by one to two orders of magnitude
(`MODEL_SCALING.md`, "what this implies for where the effort goes", item 1).

### 1.3 What this does not mean

It does not mean performance work is forbidden. It means the *justification*
changed: a dispatch-cost improvement is worth doing because it makes the Kepler
fleet usable at smaller model sizes, not because it is a step toward EXLA
parity. The step toward EXLA parity does not exist.

---

## 2. What "complete" means, and how it is ratcheted

### 2.1 The metrics that do not work, and why

**`MapSet.difference(callbacks, impl)` is already empty.** nx 0.13 declares 115
`Nx.Backend` callbacks and `Nx.Vulkan.VulkanoBackend` implements all 115
(measured; the same check is recorded in
[`docs/PARITY_STATUS.md`](docs/PARITY_STATUS.md) for 2026-07-28 and still holds).
A metric that has read 100% since July while `Nx.add` on `{:s, 32}` computes on
the CPU is not measuring completeness. It measures that a module has a function
head — and every head is followed by a host fallback that returns the right
answer.

**A fallback census that "must only shrink" is the wrong shape.** The census is
a **lower bound**: once a fallback strands a tensor on `BinaryBackend`, every
downstream op dispatches there without reaching this backend, so a *rising*
count often means a fix worked and exposed what it was hiding
(`BACKWARD_PASS_AUDIT.md` §2, which records this happening four times). A
monotone-decreasing census would forbid the fixes.

**A `doctest Nx` pass rate measures the wrong property in the normal run.**
`doctest Nx` passes today (833 doctests, part of a green suite) *while* most of
it runs on the host, because a host fallback returns a bit-identical result. Pass
rate is an API-completeness signal, not a residency one.

### 2.2 The metric that does work: `doctest Nx` **under strict mode**

Combine the two: run Nx's own doctests with `host_fallback: :raise`. Then a
doctest fails unless it is both *correct* and *resident*, and Nx's doctests are
already the community-standard backend conformance suite, written by people with
no stake in this backend's gates.

**Measured today, on `main`** (this is now `sh scripts/doctest_residency.sh`,
which prints the rate and checks it; the raw form is kept here because it is
what the number means):

```
NXV_HOST_FALLBACK=raise NXV_DOCTEST_REGISTER=off \
    mix test test/nx_vulkan/nx_doctest_test.exs
#=> 833 doctests, 78 failures
```

**755 of 833 (90.6%) of Nx's own doctests run entirely on the GPU.** That is the
completeness number. It is blunt, it is reproducible in six seconds, it cannot
be gamed by adding function heads, and it moves only when an op actually reaches
the device.

Three properties make it the right ratchet:

- **It is honest about the biggest gap.** Nx's doctests are written in
  `{:s, 32}` — and integer dtypes were precisely where this backend had almost
  no GPU path (§3.1). The 524 failures were not noise; they were the gap,
  enumerated by someone else. That is what the ratchet was pointed at, and the
  gap is now mostly closed: s32 and u32 shaders across the elementwise, compare,
  reduce and argreduce families, narrow ints widening through s32, and **78
  failures left** where there were 524.
- **It composes with the existing machinery.** `sh scripts/strict_test.sh` and
  `.github/workflows/strict-fallback.yml` already exist (T3) and already have
  the enforcement primitive.
- **It exposed how narrow the ratchet was.** Before W2 the green strict run
  reported `843 doctests, 456 tests, 0 failures, **910 excluded**` — roughly 389
  of 1,299 assertions actually running under `:raise`, with the excluded
  majority being exactly the integer-typed surface where the gap was. A ratchet
  that excludes the gap is not a ratchet. It now reports **163 excluded**: the
  resident doctests — 755 of them now — moved into the run.

### 2.3 The definition of done — **DONE**, `main` @ W2

**Complete** = `sh scripts/strict_test.sh` is green *with `doctest Nx` no longer
excluded*, and every remaining exclusion is one line naming one op with a
reason. That is now the state of the tree. The three artefacts:

1. **`@moduletag :host_fallback_expected` on `nx_doctest_test.exs` is retired.**
   In its place, `test/nx_doctest_register.exs` names all 524 doctests that
   still leave the GPU, as 140 `{"Nx.fun/arity", [ordinals]}` lines in four
   reason-bucketed lists: `@integer_dtype` (409 — W5 retires it wholesale),
   `@f64_transcendental` (37 — GLSL.std.450 has no f64 `Sin`/`Log1p`/`Erf`),
   `@complex_and_fft` (45), `@float_residency_gap` (33 — the interesting ones,
   float ops on a float backend that still left the device; read this bucket
   before picking up W1 or W8).

   **One departure from the plan as written above.** It says "in exactly the
   shape the file already uses for `@rounding` / `@unsupported` / `@backlog`",
   i.e. `doctest :except`. That option is *function*-granularity: excepting the
   121 functions involved would also have dropped 154 doctests that are resident
   today — `add/2` has both `{:s, 32}` and `{:f, 32}` examples — and the run
   would have reported 165/843 instead of 319/843. So the register keys on
   ExUnit `:test`-name filters instead, which are per-doctest. The cost is that
   the ordinals renumber if the `:except` buckets or the `nx` dep change; that
   breaks loudly and the script prints the replacement list.

   A second consequence, and an improvement: register entries still **run** and
   still assert their values in the normal `mix test`. Only the residency claim
   is waived. An `:except` entry stops executing altogether, so correctness
   coverage would have dropped by 524 doctests. Prefer the register.

2. **A residency rate recorded in CI**, printed by the same job:
   `sh scripts/doctest_residency.sh` → `755 / 833 (90.6%)`. It fails in both
   directions — a doctest not in the register that falls back is a regression, a
   doctest in the register that no longer falls back is a stale entry
   understating the rate — and prints the exact doctest names either way, so the
   repair is a paste. That is what makes the number monotone by policy.
3. **`Nx.Vulkan.Fallback`'s allowlist stays the decision register** — 8 op
   entries plus 9 `{:block, Nx.Block.*}` entries today
   (`lib/nx_vulkan/fallback.ex:250-312`). Its length is the count of things this
   project has decided *not* to do. Growing it is allowed; growing it without a
   reason in the same line is not.

**Do not** replace the allowlist's per-op granularity with families. That is the
mistake that produced the narrow forward-pass gates in the first place
(`BACKWARD_PASS_AUDIT.md` §1), and the file's own moduledoc argues it at length.

---

## 3. The coverage gaps

Everything in this section was measured on `main` at `4017f9b` today, unless
marked otherwise. Method for the residency tables: transfer an input to the
device, assert `%Nx.Vulkan.VulkanoBackend{}` on `t.data`, then
`Nx.Vulkan.Fallback.count_total/1` around a single op. A first attempt at this
table reported everything as resident because the *setup* tensor had already
been stranded on `BinaryBackend` by an integer `add` — the lower-bound trap from
§2.1, live. Assert residency of the input or the table is fiction.

### 3.1 The largest gap: this is a float backend

`{2,2}` tensors, one op each, input asserted device-resident. `HOST` = the op
left the GPU.

| op | s32 | s64 | u8 | u32 | f32 | f64 |
|---|---|---|---|---|---|---|
| `add` | HOST | HOST | HOST | HOST | gpu | gpu |
| `multiply` | HOST | HOST | HOST | HOST | gpu | gpu |
| `abs` | HOST | HOST | gpu | gpu | gpu | gpu |
| `exp` | HOST | HOST | HOST | HOST | gpu | gpu |
| `greater` | HOST | HOST | HOST | HOST | gpu | gpu |
| `select` | HOST | HOST | HOST | HOST | gpu | gpu |
| `sum` (axis 0) | HOST | HOST | gpu | HOST | gpu | gpu |
| `reduce_max` | HOST | HOST | HOST | HOST | gpu | gpu |
| `transpose` | ~~HOST~~ gpu | ~~HOST~~ gpu | HOST | ~~HOST~~ gpu | gpu | gpu |
| `reverse` | ~~HOST~~ gpu | ~~HOST~~ gpu | HOST | ~~HOST~~ gpu | gpu | gpu |
| `broadcast` | ~~HOST~~ gpu | ~~HOST~~ gpu | HOST | ~~HOST~~ gpu | gpu | gpu |
| `slice` | gpu | gpu | HOST | gpu | gpu | gpu |
| `pad` | gpu | gpu | HOST | gpu | gpu | gpu |
| `concatenate` | gpu | gpu | gpu | gpu | gpu | gpu |
| `dot` (2×2) | HOST | HOST | HOST | HOST | gpu | gpu |
| `as_type → f32` | gpu | HOST | gpu | gpu | gpu | gpu |

**Measured before W1. 48 of 96 cells left the device, and every one of the 48
was an integer dtype** — the f32 and f64 columns are `gpu` in all sixteen rows. This is the
single largest completeness gap in the repo and it is not in
`PLAN_AFTER_BACKWARD_PASS.md`, because every item there was found by tracing a
*float* workload's gradient.

Two structurally different sub-gaps hide inside it, and they should not be
planned as one thing:

**(a) The pure-copy ops that refuse integers for no reason. — DONE (W1).**
`transpose_nd`, `reverse_nd` and `broadcast_nd` decompose an output index, map
it to an input index, and copy — no arithmetic — yet all three used to gate on
`{:f, 32} | {:f, 64}` and return `nil` otherwise. They are now one word-generic
shader each, gated on `word_copyable?/1`. The reason is only that their GLSL declares `buffer A { float a[]; }`.
The repo already has the fix pattern: `slice`, `pad`, `put_slice` and `gather`
are **type-generic word copies** gated on `rem(element_bytes, 4) == 0`, which is
why those rows read `gpu` for s32/s64/u32 above. T11 reached the same conclusion
from a different direction and called it "the duplication worth removing":
porting the three remap shaders to the word-copy form collapses six files to
three *and* gives integer support as a side effect.

**(b) The arithmetic ops that need genuinely new integer kernels.** `add`,
`multiply`, `max`, the six comparisons, `select`, `sum`, `reduce_max/min` need
`int`/`uint` variants of `elementwise_binary_*`, `compare_*`, `select_*` and
`reduce_axis_*`. This is real kernel work, but it is the most mechanical kind
this project does — the f32 shaders are the template, and the correctness test
is bit-equality against `BinaryBackend` on integers, which is exact.

**W1 struck through the nine cells above** (three ops × s32/s64/u32); u8 stays
HOST for the reason in §3.3.5. 39 of 96 cells now leave the device.

*Inferred, not measured:* (a) was estimated at a day or two — it came in well
under that. (b) is a week or two. Both
figures are estimates and neither is backed by a comparable finished piece of
work, so treat them the way `ROADMAP.md` treats its withdrawn SVD estimate.

### 3.2 Decisions — recorded, not oversights

These are on the allowlist with a reason, and the reason is still good. They are
listed here so that "still a fallback" reads as a decision.

| gap | why it stays |
|---|---|
| `sort` / `argsort` | no shader and no plan; a GPU sort is a project, the host path is correct. Note the coupling: it is also what `top_k` and several Scholar diagnostics want |
| `Nx.LinAlg` SVD / QR / LU / eigh / Cholesky / solve / determinant, and `triangular_solve/4` | iterative, convergence-sensitive, and awkward to make bit-reproducible across the fleet — which is a *documented feature* here (cross-Kepler bit-determinism). `ROADMAP.md` explicitly withdraws its old "2–4 weeks" estimate as unsupported |
| broadcasting `pow` — **f64 only** as of 2026-09-01 | `GLSL.std.450` has no f64 `pow`; the only fix is boundary-casting through f32, trading real precision for a nicer table. Equal-shape f32 **and** f64 `pow` already run on the GPU. **The f32 half of this row was wrong and is withdrawn**: GLSL.std.450 *does* have a native f32 `Pow`, so f32 broadcasting pow was excluded on an f64 limitation and cost precision nothing. It now runs on the GPU (`cf7b689`). The f64 half stands unchanged |
| overlapping pooling backward | needs `GL_EXT_shader_atomic_float`, not guaranteed on the Kepler fleet. The one-thread-per-input design is exactly what avoids atomics. Non-overlapping runs on the GPU |
| complex dtypes, `Nx.Block.Phase`, sub-byte (u2/u4/s4) | the shader ISA is real-valued and a non-byte-aligned bitstring cannot be uploaded to a buffer. There is nothing to implement, not merely nothing done |
| rank-5+ index-remap and rank-5+ broadcasting elementwise binary | mechanical (extend past the `rank <= 4` gates); no workload has ever produced one. Genuinely demand-driven |

### 3.3 Open gaps — no decision recorded

Ranked inside the section by value/effort. These are the ones where "it falls
back" is currently an accident.

1. **Integer dtypes**, §3.1 — the big one.
2. **The twelve unallowlisted `Nx.Block.*` structs.** `block/4` is now
   instrumented per struct (T13), and 12 of the 21 blocks nx 0.13 defines have
   no allowlist entry, so they raise under `:raise` with no decision on record:
   `LogicalNot`, `CumulativeSum/Product/Min/Max`, `Take`, `TakeAlongAxis`,
   `TopK`, `FFT2`, `IFFT2`, `RFFT`, `IRFFT`. Verified reaching `block/4` today:
   `Nx.cumulative_sum`, `Nx.top_k`, `Nx.take`, `Nx.logical_not` all record one
   `{:block, _}` fallback each. **Most of the value here is decisions, not
   shaders** — and two are probably cheap wins rather than decisions: `Take` /
   `TakeAlongAxis` are what the existing `gather` shader does, and `LogicalNot`
   is a compare against zero.
3. ~~**`Nx.LinAlg.solve/2` raises `ArithmeticError`.**~~ **FIXED (W3), and it
   was two bugs.** `encode_scalar/2` raised on the non-finite float ATOMS nx
   uses (`Nx.Constants.neg_infinity()` → `:neg_infinity`), which nx's LU pivot
   search emits. Fixing that exposed the second: `block/4` transferred its args
   to `BinaryBackend` but not the process default backend, and every `fun` it
   takes is a defn whose intermediates the evaluator materialises on that
   default — so `Nx.LinAlg.lu(Nx.eye(2))` returned `U = 0` for the identity and
   `solve` called it singular. Both fixed; `test/nx_vulkan/lin_alg_test.exs`
   pins them.

   **The lesson is worth more than the fix.** Bug 2 was reachable the whole
   time and returned confident garbage; nobody found it because bug 1 crashed
   first. An error path can hide a wrong-answer path indefinitely, and only
   fixing the error reveals it. Side effect: an `Nx.LinAlg` call's fallback
   census went from several hundred to **1**, identical whichever backend is
   default — those round trips were the bug's mechanism, not just its cost.
4. **`dot` outside rank-2 × rank-2.** The fast path requires both operands rank
   2, single contraction axes `[1]`/`[0]`, no batch axes
   (`vulkano_backend.ex:2186`); `dot_orient/6` rescues the other rank-2
   orientations by inserting a device transpose. So **`Nx.dot(vector, vector)`,
   matrix·vector, vector·matrix, every batched dot and every rank-3+ dot fall
   back** — measured today for v·v and m·v. Matrix·vector is ordinary inference
   at batch 1; batched dot is ordinary attention.
5. **The u8 mask family's remainder.** T12 closed softmax's backward pass, but
   `reduce_max`/`reduce_min` on u8 still fall back (Nx keeps their output at
   `{:u, 8}` and `reduce_spv/2` has no `{u,8} → {u,8}` entry), and middle-axis
   u8 `sum` still falls back. **The claim that §3.1(a) fixes the second one for
   free was wrong, and W1 disproved it.** `transpose_nd` is now a word copy
   handling every 4/8-byte dtype, and middle-axis u8 `sum` still falls back —
   because a word copy cannot address a byte, so the `rem(element_bytes, 4) == 0`
   gate excludes u8 exactly as it always did for `slice`/`pad`. What W1 *did*
   remove is the transpose being the binding constraint for s32/s64/u32; those
   are now gated only by `reduce_spv/2` having no integer entry (W5). Closing u8
   needs W10's byte-packed writer — a thread per output WORD gathering four
   bytes — which is a different kernel and a different decision.
6. **Sub-word dtypes generally** (u8/s8/u16/s16, f16/bf16). Every word-copy gate
   is `rem(element_bytes, 4) == 0`, which excludes all 1- and 2-byte dtypes by
   construction, and byte packing exists in exactly three hand-written places,
   all u8-specific. There is no general byte-packed writer. bf16/f8 have no
   scalar encoder at all. *This is a candidate for a §3.2-style recorded
   decision rather than work* — decide it, do not leave it open.
7. **Three in-use f64 SPIR-V blobs have no GLSL source.**
   `priv/shaders/elementwise_binary_f64.spv`, `elementwise_unary_f64.spv` and
   `reduce_axis_f64.spv` are referenced by module attributes
   (`vulkano_backend.ex:183`, `:310`, `:750`) and have no `.comp` in `glsl/`
   (52 `.comp`, 54 `.spv`). **The f64 core kernels cannot be regenerated or
   modified from source in this tree.** Any integer or dtype work that touches
   the elementwise/reduce families will hit this. Cheap to fix, and it is the
   kind of thing that is only cheap before you need it.

   > **RESOLVED 2026-08-16 in `ac509d2`, and the count above was wrong.** It was
   > **seven**, not three — this entry named only the three it had verified as
   > referenced by module attributes. The full set also included
   > `elementwise_binary_broadcast_f64`, `reduce_full_f64`,
   > `leapfrog_chain_normal_f64` and `transpose`.
   >
   > All seven are recovered and committed. They were never lost: they lived in
   > `~/spirit/shaders/` on **both** FreeBSD Keplers, a directory outside any
   > repository, and super-io did not have it. `scripts/build_and_test.sh`
   > pointed at that machine-local dir because it was the last thing that knew
   > where they were.
   >
   > **That script has since been deleted (2026-08-28).** Its pointer outlived
   > its purpose: the recovery it existed to support finished here, the
   > directory it named no longer exists on any box, and under `set -eu` the
   > script had been dying on its first unexpanded glob for long enough that
   > nobody noticed — which is itself the evidence it had stopped being run.
   > Compiling a shader is one line, documented in the skill:
   > `glslangValidator -V glsl/foo.comp -o priv/shaders/foo.spv`.
   >
   > Each was proven genuine rather than merely plausible: compiled with
   > `glslangValidator`, byte-compared against the committed `.spv`, **all seven
   > identical**. The tree now holds a checkable invariant — **every `.comp` ↔
   > its `.spv`, every blob regenerable** — and the `clean_all_build` skill
   > verifies it on every run instead of trusting a remembered list. It was 59
   > at the time of this entry and is **87** as of 2026-08-28, re-verified
   > byte-identical on both x86_64 and aarch64 (§1.4 of `NEXT.md`).

---

## 4. The fusion mystery (T2) — the one open question worth the name

**Measured:** `Nx.Vulkan.Compiler` is within noise of the per-op path on **13 of
13 cells** of a width sweep, with **zero host fallbacks**, on exactly the
elementwise-heavy graph it was designed for — ~40 elementwise ops and two
reductions, no `dot`, no conv (`MODEL_SCALING.md`, Result 2). On other shapes it
is a regression: 0.76× on a dense-only MLP, 0.98× on a conv CNN
([`bench_results/MNIST_EXLA_RACE.md`](bench_results/MNIST_EXLA_RACE.md)).

**The old explanation is dead.** `EXMC_PEROP_RACE.md` attributed the flat result
to the 137 host fallbacks happening *below* the compiler; T11 and T12 took that
census to 0 and the result did not move.

**Why it matters.** Whole-graph compilation is the only mechanism in this repo
that could recover the per-dispatch cost that Result 5 of `MODEL_SCALING.md`
makes visible — a model written as `d` scalar RVs costs ~15 ms *per additional
RV* on the GPU while the identical arithmetic vectorised costs a constant. It is
already built, and on its best-case graph it buys nothing. Until that is
understood, "the fusion compiler will fix per-dispatch cost" is not a claim this
repo can make, and several planning documents have made it.

**What to actually do**, in the order that each step can end the investigation:

1. Confirm the generated shader is running at all — check
   `priv/shader_cache/gen_*.spv` is hit, not regenerated per call, and that the
   stage schedule is the one you think it is.
2. Count dispatches, not milliseconds. If a fused graph submits the same number
   of dispatches as the per-op graph, the answer is "it is not fusing" and no
   timing is needed.
3. Look for boundary copies. A stage split that materialises intermediates costs
   exactly what fusion saved.

**The honest caveat.** Even if this is diagnosed and fixed, it does not change
§1: the gap to `exla_host` is 20–215× and fusion's *upper bound* is the fraction
of time spent on dispatch. Do this because "we built a compiler and cannot say
what it does" is an unacceptable state for the repo's largest single piece of
machinery — not because it opens a path to EXLA. **Timebox it.** A day to
answer "is it fusing", and a decision after that.

---

## 5. What this backend owes its two consumers

Two consumers depend on this repo: the OSS **eXMC** (Bayesian/NUTS sampling) and
the **live trader** built on it. Both are writing their own planning documents;
this section is only the coupling, so that nobody has to read three plans to
find the dependency.

**The coupling is two-sided.**

1. **Op coverage.** Both consume this backend through ordinary Nx, so every gap
   in §3 is theirs. The concrete instance: the eXMC per-op race counted 137 host
   fallbacks in a single `value_and_grad` of a d=8 model, and **108 of them were
   one guard** — `compare`/`select` refusing rank 0, a gate written against
   neural-network shapes meeting a probabilistic workload where every log-prob
   is built from scalar support checks (`EXMC_PEROP_RACE.md`; fixed in T11, the
   census is now 0). That is the shape this coupling takes: not a missing
   feature, a gate that never met this consumer's shapes.
2. **The chain-shader NIFs.** eXMC synthesises GLSL
   (`Exmc.NUTS.CustomSynth`) and dispatches it through *this* repo's
   `leapfrog_chain_synth*` NIFs. The GLSL is theirs; the dispatch is ours.

**The blocker.** [`docs/TODO_CHAIN_SHADER_BUGS.md`](docs/TODO_CHAIN_SHADER_BUGS.md)
records a **`:nif_panicked`** on that path when `n_obs` goes from 60 to 600 —
same IR, same generated shader, only the observation count changes
(`MODEL_SCALING.md`, Result 7). The trader's stated direction is *shorter tick
intervals and more data per sample*, i.e. increasing exactly that number, so the
path is not merely slow at the target workload, it is unavailable there. A NIF
panic takes down more than the caller.

**What this repo owes:** the panic is on the Rust side, so bounds/size
computation, so ours to diagnose even though the GLSL is generated elsewhere.
Either it dispatches correctly at `n_obs` in the thousands, or it refuses at
synth time the way `push_too_large` already does. **A graceful refusal is an
acceptable outcome; a panic is not.**

**What this repo does not owe:** making the chain shader fast. Measured, at the
width eXMC actually runs it, it is **3.2× slower than `BinaryBackend`**, its
real ceiling is **d ≤ 13** (a 128-byte push block — not the documented 256; d ≤ 6
with `Normal` priors), and synthesis costs up to two minutes inside that range.
The per-op path it exists to beat is now fallback-free and 6.7× faster than the
CPU at the same `n_obs`. Fix the panic because the code is shipped; do not
invest in the path.

*One more thing worth carrying across:* T10's second finding was that eXMC's
`Validator.run_exla/2` selected its reference backend by auto-detection, so on a
host without EXLA **the reference arm was the candidate**, and a fleet verdict
about which host was fit for numerical validation stood inverted for three weeks
over a live defect. Already fixed on the eXMC side. The transferable rule —
**a comparison harness must assert that its two arms are actually two** — applies
to every benchmark in this repo.

---

## 6. What not to do

Each of these is declined on a measurement or a recorded argument, not a
preference. Re-litigating one requires new evidence, not a new opinion.

| not doing | why |
|---|---|
| **GPU-node routing / owner-keyed pending queues / per-submission fences** | the shared `OnceLock<Mutex<Vec<RecordFn>>>` batch queue was raced at N ∈ {4,8,16,32} on both Keplers, five interleaved replicates per cell, and **batching wins at every N on both cards** — 1.33× at N=32 on the 650M. The contention these were designed against was looked for and not found ([`bench_results/CONCURRENT_DISPATCH.md`](bench_results/CONCURRENT_DISPATCH.md)) |
| **Multi-GPU device selection** | one process already under-feeds a single card (throughput roughly doubles from N=1 to N=8 before saturating, same report). Concurrency on one device is the unexploited axis; a second device is not |
| **Register-blocked GEMM (`*_rb32`)** | the shaders exist and regress on both Keplers. Benchmark-only. If revisited, it is behind `Nx.Vulkan.Device.class/0`, which exists for exactly this |
| **Cross-stage CSE** | built, raced, and found to **never** win on either device class — recompute is cheaper than the dispatch that avoids it. Ships default-off behind `NXV_CSE=1` ([`bench_results/CSE_SOFTMAX_RACE.md`](bench_results/CSE_SOFTMAX_RACE.md)) |
| **Chasing EXLA on performance** | §1.1. 20–215×, widening, on the host CPU. There is no plausible sequence of changes here that closes it |
| **Native GPU SVD / QR / LU / eigh** | §3.2. `ROADMAP.md` withdrew its estimate as unsupported and no one has prototyped one since |
| **Unifying the index-remap shader family behind a mode selector (old T8)** | **answered, and the answer is no.** The family is already seven shaders, so the "fourth member" trigger fired long ago unnoticed, and members differ in *arity and bindings*, not just mapping rule — a unified shader needs the union of all bindings plus dummy buffers. The useful refactor is the type-generic word-copy port instead (§3.1a) |
| **Optimising the synthesised chain-shader path** | §5. 3.2× slower than the CPU at its design width, d ≤ 13 ceiling, minutes of synthesis. Fix the panic, do not invest |
| **Making `NXV_BATCH_MAX` device-class-gated** | raced on all three hosts with no crossover; it ships on by default and needs no gate (T1) |

---

## 7. The plan, ranked by value over effort

"Value" here means completeness on the hardware this project exists for, per
§1. Effort figures are **inferred** unless a comparable finished item is cited.

| # | work | value | effort | why it is where it is |
|---|---|---|---|---|
| ~~**W1**~~ | ~~**Word-generic `transpose_nd` / `reverse_nd` / `broadcast_nd`** (§3.1a)~~ **DONE** — 6 shader files collapsed to 3, **12 cells** flipped (transpose r2/r3, reverse, broadcast × s32/s64/u32), 145/145 bit-identical to BinaryBackend on mac-247. Residency **319 → 347 of 843 (37.8% → 41.2%)** | high | low | **the u8 claim in the original row was wrong** — every word-copy gate is `rem(element_bytes, 4) == 0`, which excludes 1- and 2-byte dtypes by construction. Middle-axis u8 `sum` still falls back; it needs W10's byte-packed writer |
| ~~**W2**~~ | ~~**Turn the strict ratchet on `doctest Nx`** (§2.3)~~ **DONE** — `@moduletag` retired, `test/nx_doctest_register.exs` names the 524, `scripts/doctest_residency.sh` prints **319/843 (37.8%)** in CI and fails in both directions | high | low | **every other item is now measurable.** Run the script before and after your change; if the rate did not move, the op did not reach the device |
| ~~**W3**~~ | ~~**Fix or file `Nx.LinAlg.solve/2`'s `ArithmeticError`**~~ **DONE** — it was TWO bugs: `encode_scalar/2` raised on nx's non-finite float atoms, and behind that `block/4` leaked the default backend into the defn body, so `lu(eye(2))` returned a wrong matrix. Residency **347 → 355 (42.1%)**, SVD census **350+ → 1** | high | low | fixing the raise is what made the wrong answer reachable. A raise is a better failure than a plausible wrong matrix — see `test/nx_vulkan/lin_alg_test.exs` |
| **W4** | **Decide the twelve `Nx.Block.*`** (§3.3.2) | high | low–medium | mostly allowlist lines with reasons; `Take`/`TakeAlongAxis` likely route to the existing `gather` shader and `LogicalNot` to a compare. Converts twelve accidents into decisions |
| **W5** | **Integer elementwise / compare / select / reduce kernels** (§3.1b) | high | medium | the bulk of the 524 strict doctest failures. Mechanical against the f32 templates; exact bit-equality test. Do W7 first if it blocks |
| **W6** | **Chain-shader `:nif_panicked` at `n_obs` 600** (§5) | medium–high | medium | owed to the trader, blocks its stated direction, and a NIF panic is not a defect you ship. Graceful refusal counts as done |
| ~~**W7**~~ | ~~**Recover GLSL sources for the three f64 SPVs** (§3.3.7)~~ **DONE** `ac509d2` — it was seven, not three; all recovered from `~/spirit/shaders/` on the Keplers and verified byte-identical | medium | low | **W5 is no longer blocked on this.** The f64 elementwise/reduce families are now editable from source |
| **W8** | **`dot` beyond rank-2 × rank-2** (§3.3.4) | medium | medium | matrix·vector is batch-1 inference, batched dot is attention. Follow the `dot_orient/6` precedent: normalise into `(M,K)·(K,N)`, do not add kernels per shape |
| **W9** | **Diagnose fusion, timeboxed to a day** (§4) | medium | low–medium | the repo's largest machine and nobody can say what it does. Value is knowledge; §4's caveat is that fixing it changes little |
| **W10** | **`reduce_max`/`reduce_min` on u8** (§3.3.5) | low–medium | low | needs a byte-packed writer, which is also the prerequisite for any sub-word dtype work. Do it *with* a sub-word decision (§3.3.6), not before one |
| **W11** | **Fleet race automation** (old T9) | low–medium | low | one command producing a committed per-host race set. Must fast-forward the branch explicitly — `git checkout` alone silently benchmarks a stale commit, which has already happened once |
| **W12** | **Rank-5+ remap and rank-5+ broadcasting binary** | low | low | mechanical, and no workload in four months has produced one. Leave demand-driven |
| **W13** | **Upstream: `Nx.Helpers.check_grads!` into `Nx.Testing`; file the XLA gradient tiling bug** (old T5/T6) | low here, high elsewhere | low | neither changes this backend. Both are recorded in `PLAN_AFTER_BACKWARD_PASS.md` with reproducers; do them when blocked on something else |

**Sequencing note.** W2 came before W1 and W5, because W2 is what tells you
whether W1 and W5 worked. That gate is now open: the residency rate is the
acceptance test for the whole of §3, and W1 and W5 each have their acceptance
criterion already written down as a bucket in `test/nx_doctest_register.exs` —
W5 is "`@integer_dtype` empties, 409 doctests", W1 and W8 are lines in
`@float_residency_gap`. Next by ranking: **W1**.

---

## 8. How to verify you have not broken anything

**The suite.** Measured on super-io today at `4017f9b`:

```
mix test
#=> 833 doctests, 871 tests, 0 failures

sh scripts/strict_test.sh
#=> 833 doctests, 871 tests, 0 failures, 163 excluded
```

The doctest count is **833**, down from 851 by way of 843 — moving
`standard_deviation`, `covariance` and `variance` onto the GPU cost their
doctests to the `@rounding` bucket, because a native f32 divide lands 1 ULP from
a correctly-rounded one and the doctest compares `inspect` strings. **Expect this
to keep happening** — and it did, 843 to 833 since this was written; the
bucket in `test/nx_vulkan/nx_doctest_test.exs` is the place to watch, and
excepting a function drops *all* of its doctests, not just the one that drifted.

**The strict run is the one that matters** for anything in §3. It now excludes
591 assertions: 524 via `test/nx_doctest_register.exs` (the doctests that still
leave the GPU, each named with a reason — §2.3) and 67 via
`:host_fallback_expected` (the fallback-parity modules and a scatter of
individual cases) and `:host_fallback_open` (tracked debt — two tags in
`grad_test.exs`, both **stale**, see §9.9). None of them skips anything in a
normal `mix test`. If your change adds a `:host_fallback_open` tag or a register
line, that is a visible line in a diff — which is the point.

**Residency, per op:**

```elixir
t = Nx.backend_transfer(Nx.iota({2,2}, type: :s32, backend: Nx.BinaryBackend),
                        Nx.Vulkan.VulkanoBackend)
%Nx.Vulkan.VulkanoBackend{} = t.data          # or you are measuring nothing
Nx.Vulkan.Fallback.count(fn -> Nx.add(t, t) end)
#=> {#Nx.Tensor<...>, %{{:add, 3} => 1}}
```

Prefer `Nx.Vulkan.Fallback.strict/1` for a new assertion: it fires at the
*first* refused op, before the tensor leaves the device, so it names the cause
rather than the visible edge of a cascade.

**The fleet.** Every performance heuristic must be validated across it — the
`vulkan-nx-compute` skill requires this, and the reason is that win/loss
crossovers here are hardware-specific.

| host | GPU | OS | notes |
|---|---|---|---|
| super-io | RTX 3060 Ti (Ampere) | Linux | **shared.** A foreign `mix test` inflated an established measurement by 43% during one race and moved a CPU arm by 30% during another. Check the load average before believing a cell |
| mac-247 | GT 650M (Kepler) | FreeBSD, `192.168.0.247` | **the good timing host: ±2–4% across five replicates** |
| mac-248 | GT 750M (Kepler) | FreeBSD, `192.168.0.248` | **±11–13%.** A 15% "effect" measured here once is a coin flip |

Elixir on the FreeBSD hosts is at `/usr/local/elixir-1.18.4/bin`.

**Race on the GT 650M, or replicate.** This is not general caution. The first
pass of the concurrency race — one run per cell — produced a clean, monotone,
entirely convincing hardware crossover *with a plausible mechanism*, a
two-replicate check appeared to corroborate it, and five replicates erased the
whole thing (`CONCURRENT_DISPATCH.md`). The failure mode is not a wrong number;
it is a coherent false mechanism that survives the first attempt to check it.

**Two more traps, both of which have cost a day here:**

- **The timed path must read back only a scalar.** `buf_download` calls
  `flush_pending`, which submits every queued dispatch, so a scalar accounts for
  all the work. Transferring a gradient as well adds a large constant to every
  arm equally and flattens the effect you are measuring
  (`CONCURRENT_DISPATCH.md`, method notes).
- **Fast-forward the branch on each host explicitly.** `git checkout` alone
  leaves a stale local branch and silently benchmarks the wrong commit. This has
  happened.

---

## 9. Contradictions and stale claims in the evidence base

Found while writing this. None are fixed here; they are recorded so the next
session does not have to rediscover them.

1. **Three different suite counts are in circulation.** `README.md` says
   "851 doctests, 439 tests"; `ROADMAP.md` and `LIMITATIONS.md` say
   "851 doctests, 415 tests"; `PLAN_AFTER_BACKWARD_PASS.md` T11 says
   "843 doctests / 423 tests". Measured today: **833 doctests, 871 tests.** All
   three published figures are stale. *(And so was this line's own "measured
   today" figure, for about a month — a section about counts drifting is not
   exempt from counts drifting. Re-measured on four boxes 2026-08-29.)*
2. **`docs/PARITY_STATUS.md`'s central claim is true and useless.** "Every
   `Nx.Backend` callback is implemented" has been true since July while 42 of 96
   dtype × op cells run on the host (§3.1). The document is not wrong; it is
   measuring a property that does not distinguish this backend from a stub.
   `docs/NX_PARITY_RESEARCH.md` and `docs/nx_parity_gap.csv` (2026-05-25) are
   marked stale by that same document and should not be used at all — they
   describe a 33-of-71-callback world that nx 0.13 dissolved.
3. **`ROADMAP.md` still contains the pre-reframing performance section** ("6–12
   months to feature parity", "the remaining lever is GEMM quality") *above* the
   paragraph that reframes the project to reach-not-speed. Both are on the page.
   A reader who stops early gets the old goal.
4. **`bench_results/EXMC_PEROP_RACE.md` retracts two of its own conclusions in a
   banner** and its body still argues for them. That is the right way to handle
   it — the measurements stand, the reasoning does not — but the banner must be
   read first, and its "the chain shader is still earning its keep" line is
   quoted approvingly in at least one other place.
5. **`docs/TODO_CHAIN_SHADER_BUGS.md` documents a `d ≤ 256` cap that is really
   `d ≤ 13`**, and notes the wrong figure "has already misled planning". Check
   any planning document that quotes 256.
6. **`PLAN_AFTER_BACKWARD_PASS.md`'s housekeeping section is still open**: both
   Keplers are parked on `feat/conv-backward-on-gpu` rather than their own
   branches, `f32_race_*_c622757.json` reports exist per host and are
   uncommitted, and `mix hex.retire nx_vulkan 0.2.0` has not been run (hex.pm
   still reports `retirement: None`) because it needs the maintainer's
   interactive Hex password.
7. **T7's table in `PLAN_AFTER_BACKWARD_PASS.md` was corrected in place by T3's
   strict run** — three of its lines were wrong as written. It is right now, but
   it is the second version, and the first version circulated.
8. **`LIMITATIONS.md` carries a "partially superseded" banner from 2026-08-02**
   listing four things it is wrong about. It is still the document most likely
   to be read by a newcomer looking for "what doesn't work".
9. **T12's two `:host_fallback_open` tags are stale, and T12's own "done when"
   said to delete them.** `test/nx_vulkan/grad_test.exs:136` (`reduce_max`
   gradient) and `:160` (softmax) still carry the tag. Measured today:
   `NXV_HOST_FALLBACK=raise mix test test/nx_vulkan/grad_test.exs
   --include host_fallback_open` → **22 tests, 0 failures**. The fix landed,
   the tags did not get deleted, and they are currently excusing two tests that
   no longer need excusing. One-line cleanup, and the standing debt list is
   wrong until it is done.

---

## 10. One-paragraph version

This backend's reason to exist is reach, not speed: EXLA on a host CPU beats it
by 20–215× on the same gradient with the gap widening, so performance parity
with a compiler is not achievable and is not the goal — but the FreeBSD Keplers
cannot run EXLA at all, and there the alternative is a pure-BEAM interpreter this
backend beats by up to 410×. What makes that reach worth having is
**completeness**: every Nx op running correctly and on-device, so that a model
does not silently fall off a two-order-of-magnitude cliff the first time it uses
an op nobody thought about. Completeness is currently **90.6%** by the only
metric that resists gaming — Nx's own doctests under `host_fallback: :raise`,
755 of 833. The integer gap that dominated this number is largely closed: s32
and u32 shaders across the elementwise, compare, reduce and argreduce families,
with narrow ints widening through s32. What is left is 1- and 2-byte dtypes on
the word-copy paths, 64-bit integer arithmetic, and the host tail (`sort`,
`argsort`, linalg, `atan2`).
