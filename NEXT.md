# NEXT — nx_vulkan

**Written:** 2026-08-16, against `main` @ `40d3137` (the stale-figure sweep).
**Refreshed:** 2026-08-23, against `main` @ `00cfe3b`. **W5 is done and five
follow-on items have landed on top of it** — `stack/3` routing, `gather/4` axis
rotation, `bitcast/2`, the general contraction and the batched matmul — taking
residency to **83.9%** and closing `dot/7` entirely. All of it is
**verified across the whole fleet** (§1.4) — nine commits taking `doctest Nx` residency from **47.2% to 80.4%**.
W1–W5 are all closed. §1.2 is the write-up; §1.3 is what to do next and is
measured against the current tree rather than inherited from `MISSION.md` §7,
whose ranking W5's own census showed to be built on numbers that do not mean
what they look like.

**Everything is pushed and the tree is clean.** `origin/main` is at `00cfe3b`.
The 2026-08-17 reboot came and went without incident: the driver is matched at
**580.178.04** on both sides and `device_name()` is the 3060 Ti (§5).
**Read `MISSION.md` first** — this file assumes it and does not repeat it. This
one is only *what to do next and in what order*, plus the state as it actually
stands rather than as the mission planned it.

---

## 0. Two things to know before you touch anything

### `origin` is private. `upstream` publishes.

```
origin    git@localhost:/home/git/repos/nx_vulkan.git   # private server — working remote
upstream  git@github.com:borodark/nx_vulkan.git         # PUBLIC — pushing here is a release
```

The naming inverts the usual fork convention. From the FreeBSD Keplers the same
private server is `git@192.168.0.249:/home/git/repos/nx_vulkan.git` — one host,
two addresses. **Never push to `upstream` as the last step of a task.**

Current divergence:

| ref | sha | note |
|---|---|---|
| `HEAD` / local `main` | `00cfe3b` | W1–W5 + `concat_nd`, `scatter`, `argreduce`, `allany`, `stack`, `gather`, `bitcast`, tensordot |
| `origin/main` | `00cfe3b` | **level — nothing unbacked** |
| mac-247, mac-248 | `00cfe3b` | **level — both re-verified at `00cfe3b`, §1.4** |
| `upstream/main` | `6ab64ac` | **59 behind** |

Push with `git push origin main` — **not** `upstream`, which is the public
release remote and is deliberately behind. Publishing there is a release
decision and belongs to the operator (§2).

`../_exmc-things/exmc/mix.lock` still pins **`a25432f`**, i.e. the commit before
any of this. That is the right default — see §4 — but it now means the consumer
is ten working commits behind (fourteen counting doc refreshes), including a
`block/4` fix that changes what an `Nx.LinAlg` call costs there. Bumping it is a
deliberate act and should come with `bench/nuts_truth.exs` on both arms.

### `rm -rf _build/` — do it early, do not agonise

`_build/` regenerates from source and the lockfile. Nothing is lost. The `test`
env goes stale *independently* of `dev`, and a stale `_build/test/lib/<dep>` is
a first-class time sink. This bit hard in the consumer repo on 2026-08-16: 20
integration failures that looked like anything but a build artifact turned out
to be `_build/test/lib/nx_vulkan` sitting at version **0.1.0** against a
lockfile pinning `7067499`, with a NIF missing `device_supports_f64/0`.

Suspect `_build/test/lib/` **first** on: `UndefinedFunctionError` for a NIF, a
`:bad_lib` on_load warning, a loaded version disagreeing with `mix.lock`, or
"suddenly every test fails."

```sh
rm -rf _build/     # fine. do it.
```

**nx_vulkan-specific:** `_build` is not the only stale-artifact surface here.
`priv/shaders/*.spv` are committed and are **not** rebuilt by `mix compile` —
the sources are `glsl/*.comp`, and if you edit one you must re-run
`glslangValidator` by hand, or use the
`clean_all_build` skill, which does the whole set. `priv/shader_cache/` is
gitignored and safe to delete. And `~/.exmc/gpu_node/spv/` caches synthesised
shaders **keyed by a hash of the generated GLSL**, so it invalidates itself
correctly — but delete it if you suspect otherwise.

**The shader invariant: 73 `.comp` ↔ 73 `.spv`, every blob regenerable.**
Established in `ac509d2`, checked by the skill on every run, and holding as of
`9fc58f5` — W5 added 16 shaders and the count moved 57 → 73 with it. Until then,
seven `.spv` had no source in the tree and the only copies lived in
`~/spirit/shaders/` on the two Keplers, outside any repository. If the skill
ever reports `orphan (kept)`, look there before anything else.

---

## 1. W1–W5 are done. Next is picked from §1.3, not from `MISSION.md` §7

`MISSION.md` §7 ranks W1–W13 and its sequencing note put W2 first, because W2 is
what tells you whether everything after it worked. **That gate is open and has
now been used nine times.**

The ranking itself no longer survives contact: W5's own census showed that §7's
scores came from first-fallback counts, and those name the OP rather than the
GAP (§1.2). Pick the next item from **§1.3**, which is measured against the
current tree, not from §7's table.

| item | state |
|---|---|
| **W2** — strict ratchet on `doctest Nx` | **done** `6f8d406` |
| **W1** — word-generic remap family | **done** `912ce08` `578cf3a` |
| **W3** — `Nx.LinAlg.solve/2` | **done** `f614dd0`…`62b622e` |
| **W4** — decide the twelve `Nx.Block.*` | **done** `cc77b2a` `cae4dad` |
| **`concat_nd`** — axis > 0 concatenate | **done** `c9b1a31` — not a W item; W4's census found it |
| **W5** — integer kernels | **done** `828ae14`…`9fc58f5`, nine commits — see §1.2. 47.2% → 80.4% |
| **`scatter`, `argreduce`, `allany`** | **done** — not W items; W5's census found all three |
| **`stack`, `gather`, `bitcast`** | **done** `1feed9a` `580e2db` `ab0c761` — three gate widenings, no new shader between them |
| **tensordot** | **done** `fda7fc6` `00cfe3b` — general contraction, then batched matmul. **`dot/7` is entirely closed** |

```sh
sh scripts/doctest_residency.sh
#=> doctest Nx residency: 699 / 833 (83.9%) run with host fallbacks refused
#   (688 / 833, 82.6%, device-resident — see §1.2 on the 11-doctest gap)
```

`@moduletag :host_fallback_expected` is off `nx_doctest_test.exs`;
`test/nx_doctest_register.exs` names the 134 doctests that still leave the GPU,
in four reason-bucketed lists; `test_helper.exs` applies it only when fallbacks
are being refused, so a normal `mix test` still runs and asserts all 833. The
strict suite went from 910 excluded to 591 at W2, then 557 (W1, W3), 527 (W4),
518 (`concat_nd`), 237 across W5 and **208** after the follow-on items. CI runs the script as its own step. See `MISSION.md` §2.3 for what was built and the one departure
from the plan (ExUnit's `doctest :except` is function-granularity; using it
would have dropped 154 *resident* doctests and reported 165/843).

**The register is portable, and W5 is the hardest test it has passed.** It was
measured on super-io (Ampere/Linux) and reproduces byte-identically on mac-247
(Kepler GT 650M) and mac-248 (Kepler GT 750M), both FreeBSD — same 524 at W2,
496 at W1, 488 at W3, and **the same 670 / 833 (80.4%) on all three boxes at
`f0d9c96`**, with pass B failing exactly 163 and the script exiting 0 everywhere.
Sixteen new shaders, integer wrap semantics, `atomicAdd`, NaN ordering and 16×16
tiling all reproduce unchanged across two GPU generations and two operating
systems. The gates really are dtype/shape logic.

**Re-verified again at `00cfe3b`**, after `stack`, `gather`, `bitcast`, the
general contraction and the batched matmul: both Keplers rebuilt from scratch
report **699 / 833 (83.9%)**, 833/609/0, 208 excluded — identical to super-io
and to each other, with the script exiting 0 everywhere. That run carried the
first new NIF since `scatter` (`matmul_batched/8`), which is why it was rebuilt
rather than incrementally compiled. The one exception found so far
is **llvmpipe**, where
`Nx.sum` on `{:u, 8}` returns 0 and three doctests plus three `select` tests fail
on value; if a run reports one extra fallback, check `device_name()` before
touching the register.

**Every item since W2 has been measured with it, and it has worked every time.**
The rate moved 319 → 347 → 355 → 385 → 398 → 670 → 699, and every time the ratchet
failed the build on stale entries and named every one — including twice when the
ordinals renumbered wholesale and the repair was a paste rather than an
investigation. That is the loop this project was missing.

**W1 and W3 were briefly verified on mac-247 only — that gap is now closed.**
super-io's nvidia kernel module had gone version-mismatched against its userspace
mid-session (580.173.02 loaded, 580.178.04 installed), so Vulkan there fell
through to llvmpipe and every measurement moved to the Kepler. It has since been
rebooted; both sides now read **580.178.04**, `device_name()` returns the 3060
Ti, and all three commands were re-run there on 2026-08-16 at `a930157`:

| command | super-io (Ampere) | expected from mac-247 |
|---|---|---|
| `mix test` | 843 / 476 / 0 | 843 / 476 / 0 |
| `sh scripts/strict_test.sh` | 843 / 476 / 0, 557 excluded | same |
| `sh scripts/doctest_residency.sh` | 355 / 843 (42.1%) | same |

Exact on all three, and the script's pass B failed **488** — precisely the
register's 488 entries, no extras. The register is now confirmed portable at W3
on Ampere/Linux as well as Kepler/FreeBSD.

**Use it as the acceptance test for everything that follows.** Run the script
before and after; if the rate did not move, the op did not reach the device.
The buckets as they now stand — but read §1.3, not this table, for what to do
next: **the bucket names have stopped meaning what they say.**

| bucket | doctests | note |
|---|---:|---|
| `@integer_dtype` | 92 | **badly named now.** Most of what is left is shape- or capability-gated and is s32 only because Nx's doctests are. W5 took 265 out of it |
| `@float_residency_gap` | 16 | narrow-gate work on float ops. W5 closed most of this bucket as a side effect, because the gates it widened were never dtype-specific |
| `@f64_transcendental` | 37 | not work — GLSL.std.450 has no f64 `Sin`/`Log1p`/`Erf`. Same constraint that allowlists `pow/3` at float types |
| `@complex_and_fft` | 18 | not work under current dtype support. W4 allowlisted the four FFT blocks, which took 25 doctests out of this bucket without moving them onto the device |

### 1.1 `concat_nd` — the census cashed in

`glsl/concat_nd.comp` (`c9b1a31`). W4's census named three gaps; this closed one
of them and **all five ops that shared it went resident at once** — all four
`Nx.cumulative_*/2` and `Nx.take_along_axis/3`, zero fallbacks. Residency
385 → 398 (45.7% → 47.2%), strict 527 → 518 excluded, register 458 → 445.

**Unlike W4's own 30, this 13 needs no asterisk.** Every one is genuinely
device-resident, not merely permitted to leave. Refused-clean and
device-resident readings stood at 398/843 (47.2%) and 373/843 (44.2%) when this
landed; W5 has moved both since — see §1.2.

Axis 0 was never the problem — a row-major axis-0 concat is a byte append and
`concat_buffers/1` already did it. Axis > 0 is the kernel. It belongs to the
index-remap family but **inverts its direction**: `transpose_nd` / `reverse_nd`
/ `broadcast_nd` run one thread per *output* element because they have one
input, whereas concat has k inputs and k varies per call, so it cannot bind them
all. Instead it runs **one dispatch per input, one thread per input element**.
Each input owns `[offset, offset + in[axis])` on the concat axis, so the regions
are disjoint: the output accumulates with no races and no read-modify-write, and
traffic is O(output) rather than the O(k·output) that layering k `put_slice`
overlays would have cost.

**Two things worth knowing before touching it.**

*The skill says not to write this kernel.* `vulkan-nx-compute` §1 lists `concat`
under do-NOT-write-a-kernel. That advice is about compute cost, and concat has
none — it is pure data movement, i.e. the skill's own bandwidth-bound category
where the round trip costs more than the work, and the census showed it is not
rarely-hit either. **That line in the skill should be revised**; it has not been.

*The all-operands-resident gate looks too narrow and is not.* Requiring only one
resident operand and uploading the rest is the obvious §1b move, and it broke
four `Nx.mode/2` doctests. Promoting the operands makes the *result* resident,
and `Nx.take_along_axis/3` then hands that resident index tensor to `Nx.gather/3`
beside a host operand; nx resolves a multi-arg op to ONE backend, picks
`Nx.BinaryBackend.gather/3`, and it dies in `to_binary/1` with no clause. The
looser gate does not remove a mixed-backend pair, it moves one downstream where
this backend cannot fix it. §1b says gate on what the *kernel* cannot do — here
the kernel can and the *caller* cannot, which is still a real constraint. A test
in `test/nx_vulkan/concat_test.exs` pins that fallback so nobody "fixes" it again.

**Not done:** axis 0 still requires all operands resident too, and was left
alone rather than made consistent — it is the `stack/3` NUTS trace-building
shape and deserves its own measurement before being touched.

### 1.2 W5 is done — 47.2% to 80.4%, and what the census got wrong

**Landed 2026-08-17 → 2026-08-22, on `main` @ `9fc58f5`, pushed.** W5 as scoped
was three tiers; it ran to seven commits because the census kept being right
about the direction and wrong about the size.

| commit | what | refused-clean |
|---|---|---:|
| — | before | 398 / 843 (47.2%) |
| `828ae14` | **T1** integer elementwise, compare, select | 532 / 843 (63.1%) |
| `856132a` | `pow` allowlist correction | 529 / 843 (62.8%) |
| `d964fd0` | **T2** integer axis- and window-reduce | 570 / 835 (68.3%) |
| `ef084b3` | unary coercion + window padding gate | 606 / 833 (72.7%) |
| `ef2ca14` | `scatter` — indexed_put / indexed_add | 629 / 833 (75.5%) |
| `55a4495` | `argreduce` — argmax / argmin | 643 / 833 (77.2%) |
| `0fcd907` | `allany` — all / any | 653 / 833 (78.4%) |
| `66fdc96` | `reduce/5` allowlisted | 664 / 833 (79.7%) |
| `9fc58f5` | **T3** s32 matmul + rank-1 promotion | **670 / 833 (80.4%)** |

**Quote both numbers.** Refused-clean is **670/833 (80.4%)**; device-resident is
**659/833 (79.1%)**. The 11-doctest gap is `reduce/5`, allowlisted rather than
implemented — an allowlisted fallback is *permitted*, so it leaves the register
without running on the device. Same asterisk W4's 25 FFT doctests carried.

The denominator moved too: 843 → 833. `weighted_mean/3` (8) and `Nx.log/2` (2)
joined `@rounding` in `nx_doctest_test.exs` as their operands went resident and
their f32 arithmetic stopped matching BinaryBackend's *inspect string*. That
renumbered every register ordinal twice — the fragility the register's moduledoc
warns about, happening for real, and the ratchet printed the repair both times.

16 new shaders: **57 ↔ 57 → 73 ↔ 73.**

#### The census was directionally right and numerically wrong, three times

§1.2's original split — 195 dtype-gated / 138 no-path-at-any-dtype / 24
shape-gated — held up. What did not hold is **scoring work off first-fallback
counts**, because under `:raise` a doctest reports only where it stops first:

* **`window_sum` / `window_product`** were filed as dtype-gated. They had no GPU
  path at **any** dtype — the f32 cases fell back identically — so T2 had to add
  op codes and routing, not an integer variant.
* **`dot/7` showed 17** and T3 was scored at that. Only **four** were
  rank-2 × rank-2. The rest were rank-1, batched, multi-axis or higher-rank
  contractions that no dtype port touches. T3 was worth 4, plus 2 more from a
  gate widening that needed no shader.
* **`exp/2` appeared from nowhere at 9** after T2 — not a regression, but
  `logsumexp` doctests getting further and stopping somewhere new.

**The rule to carry forward: a first-fallback census names the OP, not the GAP.**
It is still the right tool for finding *what to look at* — it found `concat_nd`,
it found `indexed_put` — but the count next to a name is an upper bound on the
work, not an estimate of it.

#### What actually paid, and it was not the shaders

Four of the nine commits closed **narrow gates** rather than writing kernels, and
between them they were worth **65 doctests** — more than T2 and T3 combined:

| gate | doctests | what was wrong |
|---|---:|---|
| window padding / dilation | 23 | the `if` refused them; the shader needed a skip-out-of-bounds and no `inf` literals |
| unary coercion | 13 | `a_v.type == out.type` refused an integer operand, with `cast_s32_to_f32.spv` already in the tree |
| scatter operand coercion | — | `upd.type == type` refused what Nx *promotes*; this was what `Nx.LinAlg.invert/1` was actually hitting |
| rank-1 `dot` promotion | 2 | a length-1 axis is free in row-major; it helps floats too |

All four are the same species, and it is worth stating flatly:
**an exact-type-equality guard is almost always wrong where Nx promotes.**
Skill §1b said this about *gradients*; it is more general than that.

#### Six semantics traps, all measured against `BinaryBackend`, none recalled

The bar on integers is bit-equality — there is no eps to hide in. All six are
pinned in `test/nx_vulkan/integer_kernels_test.exs`:

| what | reference says | the trap |
|---|---|---|
| s32 `sum` / `dot` accumulate | `2e9 + 2e9 → -294967296` | **wraps**. The opposite of `reduce_axis_f32.comp`'s `double` accumulator — same rule, match the reference, opposite conclusion |
| `{:s, 8}` `multiply` | `100 * 100 → 16` | wraps at the **element** width, not 32 bits |
| `remainder` / `quotient` | `-7 rem 3 → -1`, `-7 / 3 → -2` | sign of the **dividend**, truncate toward zero; GLSL `%` and `/` are *undefined* for negatives |
| `count_leading_zeros(0)` | `32` | `findMSB` gives −1 for 0 *and* −1, and reports the top **zero** bit for negatives |
| `argmax`/`argmin` NaN | `argmax([nan,5,nan]) → 2` at `:tie_break :low` | NaN is **absorbing** and **last-NaN-wins**; `v > best` is *false* for any NaN operand, so IEEE gets both halves wrong |
| f32 `product` | `1e20 · 1e20 · 1e-20 → 1.00000002e20` | needs a **wide** accumulator — pure f32 overflows to `inf` on the first multiply |

Two of these were found by the suite going red, not by reading: integer `pow`
returning `0`, and `argmin([2.0, :nan, 4.0])`. In both cases the fix came from
reading `Nx.BinaryBackend`'s source for the exact rule rather than guessing at
a plausible one.

#### One correction to the record

`@rounding` in `nx_doctest_test.exs` claimed the GPU's f32 divide was "1 ULP off
a correctly-rounded one". **It is the other way round.** For 10/6 the GPU returns
`0x3FD55556` and BinaryBackend `0x3FD55555`; against the true 5/3 those sit
3.97e-8 and 1.59e-7 away. The GPU's is the correctly-rounded f32. The doctests
still need excepting — they assert BinaryBackend's string, and that is the
contract — but that bucket is **not** a list of GPU imprecisions.

#### `reduce/5` is a decision, and the numbers are committed

`Nx.reduce/4` takes an arbitrary user fun, so no shader expresses it. The obvious
workaround — vectorise the fold, one dispatch per step, fun evaluated on resident
tensors, which is exactly W4's block-routing move — was prototyped and **raced
against the host path it would replace**:

| `reduce_size` | on-device fold | host fallback |
|---:|---:|---:|
| 8 | 0.97 ms | 0.19 ms |
| 512 | 39.81 ms | 22.01 ms |
| 4096 | **440.62 ms** | **37.40 ms** |

Slower at every size, and the gap widens with the axis because the cost is
per-dispatch launch overhead. Nothing removes it without assuming the fun is
**associative** (a log2-step tree reduce), which `Nx.reduce` does not guarantee —
it is a left fold. Probing the fun to recognise `add` is unsound the way any
probe is. Trading +11 residency for a 12× regression is the 0.2.0 mistake in
miniature. `bench/reduce_fold_vs_host.exs` is committed so the next person
re-measures instead of re-deriving.

### 1.3 What is left, and what it is actually made of

134 doctests still refuse, and **the biggest single group is not work**:

| group | doctests | state |
|---|---:|---|
| f64 transcendentals + `atan2` | 41 | **decided** — GLSL.std.450 has no f64 `Sin`/`Log1p`/`Erf`/`Atan2` |
| complex / FFT | 18 | **decided** — the ISA is real-valued |
| ~~`dot/7`~~ | 0 | **closed** `fda7fc6` `00cfe3b` — general contraction, then batched matmul |
| `as_type/2` | 12 | **only 6 are closable** — sized below, and it is not "mechanical" |
| `select/4`, `concatenate/3` | 16 | mixed-backend operands — read the §1.1 note before touching these gates |
| `window_reduce/6` | 5 | arbitrary fun — the same argument as `reduce/5`, and probably the same answer |
| `argmax`/`argmin`, `indexed_add` at narrow or float dtypes | 11 | s64/u32 need Int64; float `indexed_add` needs `GL_EXT_shader_atomic_float` — both **decided** |
| ~~`bitcast/2`~~ | 0 | **closed** `ab0c761` — one line, exactly as sized |
| narrow dtypes (s8/s16/s64/u32) elsewhere | ~15 | needs Int64 or 8/16-bit storage; **check the Kepler fleet before assuming** |

#### `as_type/2` sized — 6 closable, not 12, and three traps

Measured at `580e2db`. The register's 12 split by what each actually needs:

| what | doctests | verdict |
|---|---:|---|
| float → `{:u, 8}` | 5 | **closable** |
| float → `{:s, 32}` | 1 | **closable** |
| complex, either direction | 3 | decided |
| `f32 → {:s, 64}` | 1 | needs `Int64` |
| `f32 → f16` / `bf16` | 2 | needs 16-bit float storage |

Current coverage is **8 of 20 pairs** across `{u8, s32, u32, f32, f64}`, and every
one of the eight goes *to* float — nothing converts *to* an integer:

```
from\to     u8      s32     u32     f32     f64
u8         same    host    host    GPU     GPU
s32        host    same    host    GPU     GPU
u32        host    host    same    GPU     GPU
f32        host    host    host    same    GPU
f64        host    host    host    GPU     same
```

Grouped by STORAGE class (packed-u8 / 32-bit word / f64) rather than by dtype,
filling the matrix is **4-6 new `.comp` files**; the six doctests alone need
**two**. The u8 output needs packed-byte writes, but `compare_*` and `allany_*`
already have that idiom.

**Three conversion rules, not one, and they disagree:**

| case | `Nx.BinaryBackend` | rule |
|---|---|---|
| `:infinity`/`:nan`/`:neg_infinity` → int | `255` / `0` / `0` at u8 | **saturate** to the destination's range, NaN to 0 |
| `300.0` / `-5.0` / `1.0e10` → u8 | `44` / `251` / `0` | **wrap** mod 2^width, after truncating toward zero |
| `300` / `-5` (s32) → u8 | `44` / `251` | **wrap**, same as above but from an integer source |

So a finite out-of-range float wraps while a non-finite one saturates, in the
same conversion. And **GLSL's `int(float)` is UNDEFINED out of range** — `int(1.0e10)`
is itself UB, so even the wrapping branch cannot be written as a plain cast.

That is why §1.3 used to call this "mechanical" and should not have. The work is
small; the trap surface is not. Write the differential test before the shader.

#### `bitcast/2` — 2 doctests for one line

Nx raises on mismatched bit widths, so this backend only ever sees a same-width
reinterpretation of the same bytes. That is metadata, exactly like `reshape/2`:

```elixir
def bitcast(%T{type: type} = out, %T{data: %__MODULE__{ref: ref}}),
  do: put_in(out.data, %__MODULE__{ref: ref, shape: out.shape, type: type})
```

Same species as `stack/3`: an op that never asked for a capability it already
had. Found while sizing `as_type`, which is the argument for sizing.

#### Ranked by value over effort

Both of the first two below have since been done — `bitcast` in one line as
sized, and tensordot in two commits (the general contraction needed no kernel at
all; only the batched half needed shaders and a NIF). What is left:

1. **`as_type/2` float→int (6)** — worth doing, but budget for the saturate/wrap
   split above rather than treating it as a port.
2. **`select/4`, `concatenate/3` (16)** — the largest group left, and the one
   with a known trap: read §1.1 before touching those gates.
3. **`window_reduce/6` (5)** — MEASURE first, see below.

`window_reduce/6` should be MEASURED the way `reduce/5` was —
`bench/reduce_fold_vs_host.exs` is committed for exactly this — and allowlisted
if it loses. Do not write that kernel on principle; the last arbitrary-fun op
lost to the host at every size and by 12x at `reduce_size` 4096.

**The pattern that has paid best is not writing kernels.** Of the ~110 doctests
closed after W5 T3, roughly two thirds came from widening a gate that was
narrower than the shader behind it — unary coercion, window padding, scatter
operand promotion, rank-1 `dot`, `stack` routing, `gather` axis rotation. The
recurring form is an exact-type-equality or exact-shape guard sitting in front
of a kernel that could always have done the work. Check for one before reaching
for GLSL.

### 1.4 The fleet re-verifications, and the one thing they caught

**Second run, 2026-08-23 at `00cfe3b`**, covering `stack`, `gather`, `bitcast`,
the general contraction and the batched matmul. Both Keplers rebuilt from
scratch — necessary rather than tidy, because this was the first new NIF since
`scatter` — and both read exactly what super-io does:

| | super-io (RTX 3060 Ti) | mac-247 (GT 650M) | mac-248 (GT 750M) |
|---|---|---|---|
| `mix test` | 833 / 609 / 0 | 833 / 609 / 0 | 833 / 609 / 0 |
| `strict_test.sh` | 0 failures, 208 excluded | same | same |
| `doctest_residency.sh` | 699 / 833 (83.9%) | same | same |
| `matmul_batched/8` exported | yes | yes | yes |

Nothing to report from it, which is the point: five items including a new NIF
and three new shaders, and the fleet agrees to the digit. Check
`function_exported?` for a new NIF rather than trusting a green compile — `mix`
will print "Generated nx_vulkan app" without rebuilding the crate.

**First run, 2026-08-22 at `f0d9c96`**, after W5 had gone nine commits on Ampere
alone.
Both Keplers were rebuilt from scratch (`rm -rf _build`, ~2–4 min for the Rust
crate) and read **exactly** what super-io does:

| | super-io (RTX 3060 Ti) | mac-247 (GT 650M) | mac-248 (GT 750M) |
|---|---|---|---|
| `mix test` | 833 / 589 / 0 | 833 / 589 / 0 | 833 / 589 / 0 |
| `strict_test.sh` | 0 failures, 237 excluded | same | same |
| `doctest_residency.sh` | 670 / 833 (80.4%) | same | same |

**It caught one defect, and it was in a test rather than a kernel.**
`integer_kernels_test.exs` compared every result to `Nx.BinaryBackend` with
exact list equality. `Nx.sqrt` of an s32 `9` is exactly `3.0` on Ampere and
`3.000000238418579` on both Keplers — and **neither is wrong**: Vulkan permits
`sqrt` up to 3 ULP of error, and Kepler spends that budget where Ampere does
not. The test was asserting a hardware property it had no business asserting.

Fixed in `f0d9c96` by splitting the helper along the line this file actually
tests against: **integers exact, floats within eps.** On integers there is no
eps — the GPU and BinaryBackend either agree bit-for-bit or one of them is
wrong, which is what makes every wrap and sign-convention trap checkable at all.
On floats the GPU is allowed to differ, and Vulkan says by how much.

Worth generalising, because it is the mirror image of the perf lesson in §5:
**a float assertion written on one box is a hardware claim until it has run on
another.** 62 tests passed on super-io, 61 on the Keplers, and the one that did
not was the suite's fault.

Note both Keplers agreed with each other exactly, including on the sqrt value —
so this is a *generation* difference, not per-card noise, unlike the ±11–13%
timing dispersion mac-248 shows on perf work.

### W4 is done — and it went by routing, not by allowlisting

All 21 `Nx.Block.*` structs in nx 0.13 are now decided. The twelve split
**8 routed / 4 allowlisted**, which is the opposite balance to what §3.3.2
anticipated, and the census is why.

`VulkanoBackend.@device_blocks` evaluates a block's body **on this backend**
instead of transferring wholesale. That is the right move exactly when the body
composes ops this backend already has — the inverse of the `Nx.LinAlg` case,
where a body composing into ~350 ops is why transferring once and noting once
is better. Measured on super-io, every result checked element-wise against
`Nx.BinaryBackend`:

| op | after routing |
|---|---|
| `Nx.logical_not/1` f32 | **0 fallbacks — resident** |
| `Nx.take/3` axis 0 | **0 fallbacks — resident** |
| `Nx.cumulative_*/2` axis 0 | **0 fallbacks — resident** |
| `Nx.logical_not/1` s32 | `equal/3` — W5's bucket |
| `Nx.take/3` axis 1 | `gather/4` — GPU path wants leading-prefix axes |
| `Nx.take_along_axis/3` | `concatenate/3` |
| `Nx.cumulative_*/2` axis 1 | `concatenate/3` ×2 |
| `Nx.top_k/2` | `argsort/3` — already an allowlisted decision |

**Twelve opaque blocks became three named gaps.** That is the argument against
the allowlist line: an entry saying "no scan shader" would have recorded a
decision about `cumulative_sum` when the thing actually missing is
`concatenate/3` — which four cumulative ops *and* `take_along_axis/3` all share.
**A concatenate shader was therefore the single highest-leverage missing
kernel**, and it did not appear anywhere in the W-ranking before this census.
**It is now written** — `glsl/concat_nd.comp`, axis > 0, and it moved all five
ops at once. Residency 385 -> 398 (45.7% -> 47.2%), and unlike W4's own 30 this
13 needs no asterisk: every one is genuinely device-resident. See §1.1.

Only `FFT2`/`IFFT2`/`RFFT`/`IRFFT` got allowlist lines. Routing them would
report `do_fft/4` — a rename of the same wall, since their bodies are
complex-valued and the ISA is real.

One trap worth knowing: routing must transfer args **up** to the device first.
nx dispatches a multi-arg op to one backend, so a body called with the operand
here and its indices on `BinaryBackend` resolves to `Nx.BinaryBackend.gather/3`
and hands it a Vulkano tensor, which dies in `to_binary/1` with no clause.
`Nx.take/3` reaches that state via `Nx.padding_with_index/2`. Ten doctests
caught it.

Acceptance, on super-io:

```
mix test                          # 843 / 476 / 0
sh scripts/strict_test.sh         # 843 / 476 / 0, 527 excluded  (was 557)
sh scripts/doctest_residency.sh   # 385 / 843 (45.7%)            (was 355 / 42.1%)
```

The ratchet earned its keep again: it failed the build on 30 stale register
entries and named every one. **Read that 30 carefully** — only **5** doctests
genuinely reached the device; the other **25 are FFT**, which stopped failing
because they are now *permitted* rather than *refused*. Device-resident-only,
W4 scores **360/843 (42.7%)**. The register's moduledoc carries the same
warning, because this is the first time the headline number has moved for a
reason other than work reaching the GPU.

### W4 as originally scoped, for reference

`MISSION.md` §3.3.2. `Nx.Vulkan.Fallback`'s allowlist carries **9**
`{:block, Nx.Block.*}` entries (SVD, QR, LU, Eigh, Cholesky, Solve,
Determinant, AllClose, Phase); the twelve *undecided* ones are the rest of the
family — `Take`, `TakeAlongAxis`, `TopK`, `LogicalNot`, `CumulativeSum/Product/
Min/Max`, `FFT2`, `IFFT2`, `RFFT`, `IRFFT`. Each needs one of two outcomes, and
both are cheap:

* **an allowlist line with a reason** — it is a decision, not an accident; or
* **a route to a shader that already exists**. §7 flags `Take` /
  `TakeAlongAxis` as likely `gather`, and `LogicalNot` as a compare.

W3 already showed why this is worth doing rather than deferring: `block/4` was
where a wrong *answer* hid, not just a slow path. Note also that
`Nx.LinAlg.invert/1` is **not** a block at all — it composes at the Nx level, so
`with_binary_backend/1` never sees it and it falls back at `indexed_put/5`.
Worth checking which other `Nx.LinAlg` entry points are in that position before
assuming the family is covered.

Then W5 (integer kernels — the 357-doctest bucket, and the only bucket that has
moved at all so far), and W8 (`dot` beyond rank-2 × rank-2, which
`@float_residency_gap` scores at four doctests and which W3's own tests walked
straight into at `dot/7`).

---

## 2. Housekeeping still open

From `MISSION.md` §9 and `PLAN_AFTER_BACKWARD_PASS.md`. The 2026-08-16 sweep
(`40d3137`) closed the stale-figure items — suite counts corrected in six files,
`ROADMAP.md`'s banner hoisted, `PARITY_STATUS.md` and `NX_PARITY_RESEARCH.md`
bannered, T12's two dead `:host_fallback_open` tags deleted after verifying
under `NXV_HOST_FALLBACK=raise`. What it did **not** close:

| item | state | who can do it |
|---|---|---|
| ~~Push to `origin`~~ | **done** — `origin/main` at `9fc58f5`, level with `HEAD` | |
| ~~Re-verify on super-io~~ | **done** — driver matched 580.178.04 both sides throughout W5, `device_name()` the 3060 Ti, all three figures exact (§5) | |
| ~~Re-verify on the Keplers~~ | **done twice** — 2026-08-22 at `f0d9c96` (found one defect, in a test) and again 2026-08-23 at `00cfe3b` covering the new NIF. Both boxes report 699/833 (83.9%), 833/609/0, 208 excluded, identical to super-io. §1.4 | |
| **`mix hex.retire nx_vulkan 0.2.0`** | hex.pm still reports `retirement: None` | **operator only** — needs an interactive Hex password |
| **`upstream/main` is 59 commits behind** | unpublished | **operator** — publishing decision |
| **Consumer pin is 19 commits behind** | `../_exmc-things/exmc/mix.lock` still on `a25432f` | anyone, but see §4 — bump it *with* `bench/nuts_truth.exs` on both arms |

The retirement command, for when someone has the password:

```sh
mix hex.retire nx_vulkan 0.2.0 deprecated \
  --message "Backward pass ran on the host: GPU training was ~250x slower than advertised. Results were correct; use 0.3.0 for training."
```

That message is worth keeping as written. It says what was wrong, that results
were still *correct*, and what to do instead — which is the whole job of a
retirement notice.

---

## 3. W6 got more urgent, and gained a sibling

**W6 — the chain-shader `:nif_panicked` at `n_obs` = 600** is still owed to the
trader and still blocks its stated direction (shorter ticks, more data per
sample). `docs/TODO_CHAIN_SHADER_BUGS.md` Bug 1 has the reproducer. Graceful
refusal — `{:unsupported, _}` the way `push_too_large` already does — counts as
done. A panic in a NIF takes down more than the caller.

**Do not mistake `1633073` for this.** That commit ("a failed dispatch panicked
the NIF instead of returning an error", branch `fix/nif-panic-on-dispatch-error`,
merged into `main`) hardened the *general* dispatch-error path. Bug 1's
`n_obs = 600` panic is a separate size/bounds computation and is still open.

**Bug 2 in that same document is now fixed downstream, and the fix confirms the
number.** The documented `d ≤ 256` cap really is `d ≤ 13`; measured with
`Push.pack/1` in the consumer repo: the header is 24 bytes, not the 16 the
docstring claimed, leaving 104 bytes = 13 f64 prior floats. `d ≤ 13` for
one-parameter priors, `d ≤ 6` for `Normal`, `d ≤ 3` for `TruncatedNormal`.
`docs/TODO_CHAIN_SHADER_BUGS.md` can be updated to say Bug 2 is closed in
eXMC 0.3.1 — but note the correction to its framing: the `d <= 256` guards are
**not** unreachable, 256 is the genuine `local_size_x` / `q_shared[256]`
thread-tile size. It simply is never the binding constraint.

### A new item, and it belongs near W6

The consumer found a defect that lives at the boundary this repo owns:
`compiler: :vulkan` returns a **frozen chain** for models with observations —
1 distinct value in 500 draws. Write-up in
`../_exmc-things/exmc/docs/OPEN_VULKAN_OBSERVED_MODEL.md` — note that
`_exmc-things/` is a **sibling** of this repo, not a directory inside it.

**The experiment has been run, it settled ownership, and the fix has landed
on the eXMC side** — `6c1589a` on `gate1/reconcile-core`, together with the
harness. The posterior now reads mean 3.966 / sd 0.539 against an analytic
3.99 / 0.577, with 469 of 500 draws distinct where there was 1. The fault was
eXMC's, not this repo's. Run on super-io 2026-08-16 against eXMC's own pinned
`a25432f` — legitimate, because `native/`, `native_v.ex`, `shader_template.ex`
and `synthesis.ex` are byte-identical between `a25432f` and `HEAD`, so no
lockfile bump was needed to measure the current dispatch path.

The four arrays diverge from **step 0**, not gradually:

| eps | first-step gpu grad | host grad | analytic |
|---|---|---|---|
| any | 31.495 | 10.495 | 10.495 |

The ratio is exact, and the generated GLSL explains it. For a model with three
*separate* scalar obs nodes, eXMC's emitter produces **one accumulator per
observed RV** (`_gacc0_0/_1/_2`, `_lpacc0/1/2`) and then gives **each one a full
loop over all `pc.n_obs` observations** — the identical body, three times, summed.
So the likelihood is counted `n_obs` times over:

```
grad_gpu = prior_grad + 3 × Σ_j (obs_j − q)      # should be prior + Σ_j
logp_gpu = prior_lp   + 3 × Σ_j log N(obs_j | q, 1)
```

Both predictions reproduce the measured GPU output to f32 constant precision
(`logp_chain[0]`: predicted −63.9595, measured −63.95952955528024). **The GPU
faithfully executes the shader it was handed; the shader is wrong before it
arrives.** `Nx.Vulkan.ShaderTemplate` is not even on this path — eXMC's
`MultiRvCustomSpec` renders its own GLSL.

Defect site: `_exmc-things/exmc/lib/exmc/nuts/custom_synth/multi_rv_custom_spec.ex`
— `compose_logp_defn/1` handed **every** observed node the whole `obs` vector
(`mod.logpdf(obs, resolved) |> Nx.sum()`), and `transform_reduce_sum/2` then
expanded each resulting marker into `for (j < pc.n_obs)`. Three distinct scalar
observations should each read their own slice. This is why the **vector** arm of
that test is correct and the **scalar** arm is not: the vector model emits
exactly one marker and one loop. The fix gives each marker its own
`{offset, count}` span.

**Worth carrying across to anything similar here:** attribution of markers to
nodes is positional, and the *gradient's* markers come out **mirrored** —
reverse-mode AD walks the forward left fold backwards. With identical
observations a mirrored assignment is bit-identical to the correct one, so the
first version of that fix passed a differential built on `Normal(mu, 1)` ×3 and
was still wrong. Distinct per-node sigmas are what caught it. A test whose
inputs are symmetric cannot see a permutation bug.

It also explains the freeze without needing the step-size lead. A likelihood
counted 3× is a posterior ~√3 narrower with 3× steeper gradients, so ε = 1.139
is far past stable and acceptance collapses to ~0.002. The "adapted ε identical
to sixteen digits" observation is a symptom of a saturated adaptation, not a
second bug.

Two smaller things the same run exposed, both eXMC-side and neither able to
explain the freeze: the obs buffer carries **f32-rounded** values on a nominally
f64 path (`3.8` arrives as `3.799999952316284`), and the priors are f32 tensors.

Harness: `../_exmc-things/exmc/bench/leapfrog_leaf_diff.exs` (~140 lines, left
**uncommitted** in that repo). It dispatches via
`Exmc.NUTS.Vulkan.Dispatch.chain/8`, references `Exmc.NUTS.Leapfrog.step/6`,
and cross-checks against a hand-derived analytic gradient so host and GPU
cannot both be wrong in the same direction. Run it with
`mix run bench/leapfrog_leaf_diff.exs` from the eXMC repo. The eXMC write-up
asks for this check to exist regardless of what it found; it is now the
regression test for the fix.

---

## 4. What this backend owes its consumers

`MISSION.md` §5 covers this; two additions from 2026-08-16.

**The pin is a feature, not friction.** `../_exmc-things/exmc/mix.lock` pins
`a25432f` as of this refresh — it tracked `7067499` when §0 was first written,
so it has been bumped once already. It is now **19 commits behind**, all of W5
among them. Bumping it is a deliberate act that should
come with a run of that
repo's `bench/nuts_truth.exs` on both arms, because a backend change that
alters numerics shows up in a posterior long before it shows up in a test that
compares two backends to each other.

**Do not assume a consumer's `_build` matches the pin.** It did not, for an
unknown length of time, and nothing detected it. If anything here changes a NIF
export, say so where a consumer will read it — a missing export surfaces as
`UndefinedFunctionError` at *runtime*, in whichever env is stale, not at compile
time.

---

## 5. Verification, unchanged but worth restating

`MISSION.md` §8 has the full procedure. The three that matter most:

```sh
# suite: 833 doctests, 609 tests, 0 failures.
# Measured at 00cfe3b on ALL THREE boxes — super-io (Ampere/Linux), mac-247 and
# mac-248 (Kepler/FreeBSD) — identical to the digit. See §1.4.
mix test

# strict — did the work stay on the GPU?
sh scripts/strict_test.sh            # 833/609/0, 208 excluded

# the number that actually means something
sh scripts/doctest_residency.sh      # 699 / 833 (83.9%), exits 0
                                     # 688 / 833 (82.6%) device-resident

# confirm the real GPU, not llvmpipe, before believing ANY figure — not just
# perf ones. llvmpipe is not merely slower here, it is WRONG: Nx.sum on {:u, 8}
# returns 0, which fails three doctests and three select tests. A red suite on
# super-io is worth a device check before it is worth debugging.
Nx.Vulkan.NativeV.device_name()      #=> {:ok, "NVIDIA GeForce RTX 3060 Ti", "DiscreteGpu"}
```

**Check the driver before anything else after any reboot.** This box
has already lost a session to it once: the nvidia kernel module went
version-mismatched against its userspace mid-session (580.173.02 loaded against
580.178.04 installed), Vulkan silently fell through to llvmpipe, and every
measurement had to move to the Kepler. A reboot is exactly when a pending driver
update lands, so it is exactly when this recurs.

```sh
cat /proc/driver/nvidia/version                        # loaded
ls /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.*     # installed
```

The two must match. They both read **580.178.04** throughout W5, which is the
state every figure in this file was measured under.

**Residency is not correctness, and a value assertion cannot see the
difference** — the host fallback *is* `Nx.BinaryBackend`, the reference every
test compares against, so a refused GPU gate returns a bit-identical result.
Count fallbacks (`Nx.Vulkan.Fallback.count/1`), or refuse them
(`NXV_HOST_FALLBACK=raise`); those are the only signals. And the count is a
**lower bound**: once a tensor lands on `BinaryBackend`, everything downstream
computes there unrecorded — which is why the ratchet *raises* on the first
refused op rather than tallying at the end.

**Validate perf heuristics across the fleet, never on one box.** Win/loss
crossovers here are hardware-specific — the many-slot fused reduce wins ~4.4× on
Kepler and *regresses* ~0.44× on Ampere. mac-247 (GT 650M) is the quiet box at
±2–4%; mac-248 (GT 750M) runs ±11–13% and has already produced one retracted
"hardware crossover" that was noise. Five replicates before believing a 15%
effect there.
