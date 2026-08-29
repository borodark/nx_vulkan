# NEXT — nx_vulkan

**Written:** 2026-08-16, against `main` @ `40d3137` (the stale-figure sweep).
**Refreshed:** 2026-08-28, against `main` @ `661477f`. W1–W5 are all closed and
**residency is 755 of 833 (90.6%)**, up from 714 (85.7%) five days earlier.
**There is no longer any dtype whose arithmetic leaves the device** — u32 was
the last (§1.3) — and the three "unexamined one-offs" that §1.3 used to list are
closed too, all three by widening a gate rather than writing a kernel.

Since then the work has turned from residency to health: `indexed_put`'s
duplicate-index write race is now deterministic (§1.5), and a sweep of dialyzer,
docs, coverage and format found one real bug and several traps (§1.6).

**Eight commits got there and NOT ONE of them was a new arithmetic kernel.**
Two new `.comp` files landed, both pure format conversion; everything else was a
gate that was narrower than the shader already behind it. §1.2a is the write-up.

| | doctests | what it actually was |
|---|---:|---|
| `classify_reduce_axes/2` | +8 | the middle axis of a reduce. Three clauses replaced by the one shape they were all special cases of |
| narrow ints (s8/u8/s16/u16) | +13 | widen → the EXISTING s32 kernel → truncate. No arithmetic shader |
| narrow `as_type` + reductions | +6 | the same pair, applied where narrow ints still left the device |
| integer `pow` | +3 | one arm, and a gate that reads the DATA |
| `round` + `remainder` | +2 | two missing float op codes, and three DEAD ones deleted |
| mixed narrow + `coerce_to` | +2 | operands need not share a type |
| narrow `broadcast` | +3 | a word copy cannot address a byte |

Closed outright along the way: `argmax`/`argmin`, `bitwise_not/1`,
`population_count/1`, `max/2`, `min/2`, `multiply/2`, `subtract/2`, `negate/1`,
`sum/2`, `product/2`, `all/2`, `linspace/3`, `pow/2`, `round/1`, `remainder/2`,
`divide/2`, `tril/2`, `triu/2`, `fill/3`.

**Verified on a fourth box** — the Jetson Nano, ARM/Tegra X1, Maxwell — which
matches super-io to the digit and reproduces all 80 `.spv` byte-for-byte (§1.4).

§1.3 is what to do next and is measured against the current tree rather than
inherited from `MISSION.md` §7, whose ranking W5's own census showed to be built
on numbers that do not mean what they look like.

**Everything is pushed and the tree is clean.** `origin/main` is at `661477f`.
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
| `HEAD` / local `main` | `661477f` | the above + u32, the one-offs, deterministic scatter, and the health sweep |
| `origin/main` | `661477f` | **level — nothing unbacked** |
| jetson (192.168.0.250) | `a950c9f` | re-verified there AND now the only box with EXLA (§1.4). **Two commits behind** |
| mac-247, mac-248 | `d548c85` | re-verified there, §1.4 — **two of four Kepler runs caught a real defect**. Seven commits behind; a run at `661477f` is in flight |
| jetson (192.168.0.250) | `d548c85` | re-verified there; the only box with EXLA (§1.4). Seven commits behind; a run at `661477f` is in flight |
| `upstream/main` | `6ab64ac` | **far behind** |

**All four boxes have now seen the ninety-percent run** — Ampere, Maxwell/ARM
and two Keplers — and they agree on every number. The Kepler run earned its
keep for the second time: it caught two mistagged tests that turned
`strict_test.sh` red (§1.4). When re-verifying, remember there is a NEW NIF
(`cast_spec/5`), so both Keplers needed a full crate rebuild — `rm -rf _build`,
and check `function_exported?` rather than trusting a green compile, because
`mix` will print "Generated nx_vulkan app" without rebuilding the crate.

**AND `function_exported?/3` LIES ON A COLD VM.** It returns `false` for a
module that has not been loaded yet, and `Application.ensure_all_started(
:nx_vulkan)` does NOT load `Nx.Vulkan.NativeV`. Reproduced on all three
architectures. So the check only means anything in this form:

```elixir
Code.ensure_loaded!(Nx.Vulkan.NativeV)
function_exported?(Nx.Vulkan.NativeV, :apply_scatter_ordered, 8)
```

Without the `ensure_loaded!` it reports a stale `.so` on a perfectly good build
— which is exactly the wrong failure, because it sends you off rebuilding a
crate that was already fine. Found on the 2026-08-28 fleet run, after this file
had been recommending the bare check for several rounds.

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

## 1. W1–W5 are done, and §1.3 is now EMPTY — next comes from §1.6, not §1.3

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
| **the ninety-percent run** | **done** `27bd82c`…`d84ed29`, eight commits — see §1.2a. 85.7% → 90.2%, and **not one new arithmetic kernel** |

```sh
sh scripts/doctest_residency.sh
#=> doctest Nx residency: 755 / 833 (90.6%) run with host fallbacks refused
#   (744 / 833, 89.3%, device-resident — see §1.2 on the 11-doctest gap)
```

`@moduletag :host_fallback_expected` is off `nx_doctest_test.exs`;
`test/nx_doctest_register.exs` names the 78 doctests that still leave the GPU,
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

### 1.2a The ninety-percent run — 714 to 751, and how it was found

Eight commits, `27bd82c` through `d84ed29`. **The thing worth carrying forward
is not the number; it is the method that produced it.**

#### Census the REFUSED OP, not the failing doctest

`doctest_residency.sh` prints the doctests that fall back. Every previous pass
over this list read it as a work queue — "argmax has 8, go write argmax" — and
NEXT.md has already recorded four times that this reads the OP and not the GAP.
This run stopped reading it that way. The strict error carries the refused
callback and the output shape/type:

```sh
awk '/host fallback refused:/{op=$0; sub(/.*refused: /,"",op);
     getline; getline; sig=$0; gsub(/^ +/,"",sig); print op"  "sig}' pass_b.log \
  | sort | uniq -c | sort -rn
```

The doctests said `argmax`. **The refusals said `argmax/3`, and the fix was in
`classify_reduce_axes/2`, which four op families share.** One predicate, +8
doctests, and `sum`/`reduce_max`/`reduce_min` and `all`/`any` came along
uncounted because their doctests happen not to reduce a middle axis. Run that
awk before writing anything.

#### Test the PREMISE of a "decided" verdict before believing it

§1.3 filed s8/s16/u8/u16 under "needs Int64 or 8/16-bit storage" and this file
called them decided. **Both halves of that were wrong**, and one probe found it:

```
s8 create   OK  resident=true  {:s, 8} [1, 2, -3, 4]
s8 add      FALLBACK: host fallback refused: add/3
```

Storage already worked. And a second probe — comparing `Nx.BinaryBackend`
against "widen to s32, compute, truncate" across every binary op — came back
`MATCHES` on all of them, because BinaryBackend computes narrow integers in full
precision and applies the width at the end. So the arithmetic never needed the
storage extension either. Thirteen doctests, no arithmetic shader.

**A "decided" verdict is a claim about the world, and claims can be tested. The
cheap test is worth more than the reasoning that produced the verdict.**

#### A pin that records a BELIEF will defend it

Three pinned tests broke this run, and the three are worth reading together:

  * `fallback_test.exs` asserted a middle-axis u8 sum falls back because the
    case "rotates kept axes to the front" and transpose has no u8 path. **There
    is no rotation.** The premise was false and the pin had been defending it
    since it was written.
  * `fallback_test.exs` asserted u8 `reduce_max`/`reduce_min` fall back because
    a `{:u, 8}` output "would need a byte-PACKED writer rather than a word one".
    True — and `cast_s32_to_narrow.comp` is now that writer. A correct pin whose
    premise got satisfied.
  * `strict_fallback_test.exs` pinned reduce ATTRIBUTION and had **already been
    re-pointed once**, from u8 `sum` to u8 `reduce_max`, with a comment naming
    the byte-packed writer it was now anchored to. It broke again for exactly
    the reason it had written down.

The rule: **a test about a MECHANISM must not be anchored to a GAP someone is
trying to close.** The attribution test is now anchored to `{:s, 64}`, which is
decided rather than merely unbuilt. And a pin should record a MEASURED limit,
never a belief about why the limit exists — the belief outlives its truth and
argues on its own behalf.

#### Dead code that disagrees with the live table is worse than missing code

`elementwise_binary_f64.comp` defined codes 7/8/9 as equal/less/greater, left
from before `compare_f64.comp` existed. Unreachable under
`binary_spv({:f, 64}, code) when code <= 6` — but `@binary_ops` says 8 is
`remainder`, so **widening that cap by one, which is exactly what adding
`remainder` does, would have returned a comparison mask.** Correct-looking,
silently wrong, invisible to a value assertion.

#### A gate may need to read the DATA

Integer `pow` is the one that could not be settled by type. `Nx.BinaryBackend`
RAISES on a negative integer exponent, and a shader cannot raise — it would
answer something plausible where the reference errors. So the exponent is proved
non-negative first: four bytes for a rank-0 exponent, one `reduce_min` plus
those four bytes otherwise. **Both are cheaper than the host fallback they
replace**, which moves both operands off the device and the result back.

The consequence for readers: `binary_spv/2` can no longer be read as the whole
gate for code 4.

#### What the two new shaders actually are

`cast_narrow_to_s32.comp` and `cast_s32_to_narrow.comp`. Neither computes
anything — they convert between packed sub-word storage and 32-bit words. Every
narrow-integer op in this run is `widen → an existing kernel → truncate`, which
is why one pair unlocked arithmetic, `as_type`, the reductions, mixed-type
operands, `coerce_to` and `broadcast` in turn. **Three dispatches against a host
round trip for the whole tensor is the trade, and it is the same one
`reduce_via_transpose/5` already made.**

### 1.3 What is left, and what it is actually made of

**78 doctests still refuse**, measured at `661477f`. Re-derived from the
REFUSED OP census (§1.2a) and grouped by REASON, not by op name — which is the
whole point of §1.2a and is why this table looks nothing like the one it
replaces.

**Every remaining group is decided.** This is the first refresh where nothing in
the table is open, and the honest reading is that doctest residency has reached
its ceiling: what is left needs a device capability this backend declines to
require (Int64, atomic float, 16-bit storage), a real-valued ISA it does not
have (complex/FFT), an f64 transcendental GLSL.std.450 does not define, or a GPU
sort that `MISSION.md` §3.2 declines as a project. **Further work should be
judged on residency in real workloads, not on this number.**

| group | doctests | state |
|---|---:|---|
| f64 transcendentals + `atan2` | 40 | **decided** — GLSL.std.450 has no f64 `Sin`/`Log1p`/`Erf`/`Atan2` |
| complex — `fft`/`ifft` (16), `conv` (1), `as_type` (2), `is_nan`/`is_infinity` (2) | 21 | **decided** — the ISA is real-valued |
| `concatenate/3` | 8 | **blocked behind an allowlisted `sort/3`**, not by the concat gate |
| s64 — `as_type` (2), `indexed_add`, `indexed_put`, `clz` | 5 | **decided** — needs Int64, a device capability this backend does not require |
| float `indexed_add` | 2 | **decided** — needs `GL_EXT_shader_atomic_float` |
| f16 / bf16 `as_type` | 2 | **decided** — needs 16-bit float storage |
| ~~u32 arithmetic~~ | 0 | **closed** `41a4ebf` — six `uint` shaders; it was 30 ops off the device, not one doctest |
| ~~`slice/5`, `indexed_put/5` at s32, `dot/7`~~ | 0 | **closed** `3a1357c` — three unrelated gates, no shader between them; see below |

Eighty percent of what is left is genuinely decided. The honest reading is that
**doctest residency is close to its ceiling** and further work should be judged
on residency in real workloads rather than on this number.

#### `u32` is closed — and it WAS understated by its doctest, by thirty to one

Done in `41a4ebf`. Recorded here because the sizing lesson outlives the work.

This entry used to say "exactly one doctest, but no u32 arithmetic runs on the
device at all". That was right, and still understated it: the census found
**thirty of thirty-four u32 operations** host-falling-back — every binary op,
every comparison, five of six unaries, all the reductions, `argmax`/`argmin`,
`select` and `dot`.

Six `uint` shaders: binary, broadcasting binary, unary, compare, `reduce_axis`,
`argreduce`. Three ops reuse the SIGNED kernel and each says why at the
selector — `select` copies words and does no arithmetic, `all`/`any` test
against zero, `matmul` is add and multiply which wrap identically. **No ordering
is involved in any of the three**, and ordering is the whole difference.

The per-code shortcut was available and was refused. `add`/`subtract`/
`multiply`/`pow`/bitwise/`left_shift` are bit-identical between s32 and u32, so
a selector split by code would have worked — and would have been one edit away
from silently wrong forever, because `max`, `min`, `quotient`, `remainder`,
`right_shift`, `sign` and every comparison are NOT. A signed kernel on u32 input
does not crash; it returns a different in-range plausible number.

Two things turned out simpler on the unsigned side, which is worth knowing if a
similar port ever comes up: `quotient`/`remainder` need no sign fixup (GLSL's
`/` and `%` are undefined for negative operands, and there are none), and `pow`
needs no data check (the gate exists to prove the exponent non-negative, and an
unsigned exponent cannot be otherwise).

#### `concatenate/3`'s 8 are still `sort/3`'s

Unchanged and re-confirmed. Six of the eight are `Nx.mode/2`, which sorts;
`sort/3` is allowlisted with no GPU sort and no plan for one (`MISSION.md` §3.2),
and everything downstream of a host fallback computes on the host. Loosening the
concat gate was re-tried on 2026-08-23 and still fails. **Treat as decided until
a GPU sort exists.**

#### The three one-offs — probed, and all three were gates

Closed in `3a1357c`. §1.3 said "probe before scoping", and **probing was the
whole job**: each was a gate narrower than the kernel behind it, none needed a
shader, and the three had nothing in common except that.

  * **`dot/7` at rank 5.** The `rank <= 4` cap exists because `transpose_nd`
    addresses four dims — but `dot_flatten/3` transposes ONLY when the
    permutation is not already the identity. `Nx.dot(rank_5, rank_1)` contracts
    a's last axis against b's only axis, so both permutations ARE the identity
    and nothing was going to be transposed. The cap sat on `resident?`, where it
    applied to operands needing no rotation at all.
  * **`indexed_put/5` over a non-prefix axis subset.** `gather/4` has rotated for
    this since `580e2db`; scatter had not. **Scatter also needs the rotation
    BACK** — a gather's result keeps the non-indexed dims in their original
    relative order, a scatter's carries the TARGET's shape.
  * **`slice/5` with a dynamic (tensor) start.** Four bytes, against a host round
    trip for the entire source tensor. **Clamping is what makes it more than a
    readback**: Nx clamps a dynamic start into `[0, dim - len]` rather than
    raising, and the shader would index past the buffer, where
    `robust_buffer_access` returns ZEROS.

#### Ranked by value over effort

**Nothing.** Every group in the table above is decided, and saying so is more
useful than leaving a backlog that looks actionable. There is no longer a dtype
whose arithmetic leaves the device, and no ungated op with a kernel behind it
that anyone has found.

That is not the same as "no work left" — it means the NEXT work is not doctest
residency. §1.6 is where the last two days went, and it found more per hour than
the last few residency items did.

**The pattern that has paid best is still not writing kernels.** Of the 37
doctests closed in the ninety-percent run, ZERO came from a new arithmetic
shader. u32 is the exception that proves it — six real kernels for ONE doctest,
justified by residency rather than by the register, and even there the doctest
only closed once a `coerce_to/2` gate was widened as well. Two `.comp` files landed and both are format conversion. The recurring
form is an exact-type-equality or exact-shape guard sitting in front of a kernel
that could always have done the work — and, twice this run, a comment or a
pinned test explaining why the guard had to be there.

### 1.4 The fleet re-verifications, and the one thing they caught

**Fifth run, 2026-08-28 at `d548c85`**, all four boxes, covering the
deterministic scatter and its new NIF.

| | super-io | jetson | mac-247 | mac-248 |
|---|---|---|---|---|
| `mix test --seed 0` | 833 / 837 / 0 | 833 / 837 / 0 | 833 / 837 / 0 | 833 / 837 / 0 |
| `strict_test.sh` | 0 fail, 153 excl | 0 fail, 153 excl | 0 fail, 153 excl | 0 fail, 153 excl |
| `doctest_residency.sh` | 755 / 833 (90.6%) | 755 / 833, 78==78 | 755 / 833, 78==78 | 755 / 833, 78==78 |
| `apply_scatter_ordered/8` | — | exported | exported | exported |

`0da264d` is the current head and changes only comments and this file; the
`d548c85` result stands for it.

**Three things this run established that the tests alone could not.**

  * **`[0, 50, 60, 0]` on every box**, 10/10 on each Kepler and 5/5 on the
    Jetson. The updates DESCEND, so a `atomicMax` over VALUES would have
    answered `[0, 90, 80, 0]`. The implementation really is max-over-ROW-INDEX.
  * **No flakiness in five runs per box.** The two dispatches and the zero-fill
    share one command buffer and rely on vulkano's `AutoCommandBufferBuilder`
    to insert the compute-compute barrier. If that were ineffective on Kepler
    or Tegra, pass 2 would read a half-written scratch buffer and the symptom
    would be flakiness, not a clean failure. None appeared. Honest limit: five
    runs of a 1.7s file is a modest sample for a race — "no flakiness observed",
    not "barrier proven".
  * **The full suite completes on the Jetson for the first time.** The
    `compiler_test.exs` many-slot reduce no longer builds 3 GB on the host, so
    the box that used to die mid-suite now finishes in 58s with 3.3 GB still
    free.

**And one thing only the fleet could show.** On both Keplers the RACING arm also
answers `[0, 50, 60, 0]`, so there the VALUE does not discriminate ordered from
racing — only the >2x timing gap does. The descending test discriminates the
IMPLEMENTATION, not which path executed. On Ampere it does both, because that is
the box whose race resolved the other way.

Two environment findings, both now in the runbook: `function_exported?/3`
returns false for an unloaded module (see §0), and on the Jetson **neither `mix`
nor `cargo` is on `PATH` in a non-login shell** — Rustler dies with `:enoent`
until `~/.cargo/env` is sourced alongside `~/.asdf/asdf.sh`. The Jetson's
checkout is also left on a DETACHED HEAD at `d548c85` with local `main` still at
`3a1357c`, so the next run there should `git checkout main` before pulling
rather than after.

**Fourth run, 2026-08-23 at `92d56cd`**, on both Keplers — and **it caught a
defect that `mix test` structurally cannot see.**

| | mac-247 (GT 650M) | mac-248 (GT 750M) | super-io |
|---|---|---|---|
| `mix test --seed 0` | 833 / 780 / 0 | 833 / 780 / 0 | same |
| `doctest_residency.sh` | 751 / 833 (90.2%) | 751 / 833 (90.2%) | same |
| register exact both ways | 82 == 82 | 82 == 82 | same |
| `cast_spec/5` exported | yes | yes | yes |
| **`strict_test.sh`** | **2 failures** | **2 failures** | **2 failures** |

**The two failures were mine, and they were in the tests rather than the
kernels.** `reduce_axes_test.exs` and `narrow_int_test.exs` each contain a test
whose SUBJECT is a host fallback — the non-contiguous `all`/`any` case, and the
float-source `as_type` refusal. Neither carried `@tag :host_fallback_expected`,
and `strict_test.sh` excludes only that tag, so under `NXV_HOST_FALLBACK=raise`
both raised.

**Why nothing else caught it.** `mix test` is green: outside strict mode a
fallback returns a bit-identical result, so the tests pass and assert exactly
what they mean to. `doctest_residency.sh` is green: it only reads
`nx_doctest_test.exs` and never sees these files. **Only `strict_test.sh` can
see this class of mistake, and it was the one check not re-run** — it went once,
eight commits earlier, and the rest of the run leaned on `mix test` plus
residency.

The rule that falls out, and it is the third instance of this shape today: **a
test that deliberately provokes a fallback must opt out of the strict run, and
the check that would tell you is not the one you are watching.** Run all three
scripts before calling a run clean, not the two that move.

Fixed in `235f99c`. Both boxes also confirmed every Kepler-specific risk —
signed overflow wrapping (`pow(2,32) = 0`, `pow(3,20) = -808182895`), the packed
sub-word tail, `int(b << 24) >> 24` as an arithmetic shift, `round` ties at
half-away-from-zero, and dividend-signed `remainder` at all four sign pairs.
**Three architectures now agree on all of it**: Ampere, Maxwell/ARM, Kepler.

The `sqrt` 3-ULP divergence is still there — `Nx.sqrt(9.0)` is
`3.000000238418579` on both Keplers — and is now harmless, because
`integer_kernels_test.exs` stopped asserting exact float equality in `f0d9c96`.

**Third run, 2026-08-23 at `221b8c1`**, on a FOURTH box — the Jetson Nano — and
it is the first ARM/Tegra verification this project has had.

| | super-io (RTX 3060 Ti, Ampere) | jetson (Tegra X1, Maxwell) |
|---|---|---|
| `mix test --seed 0` | 833 / 774 / 0 | 833 / 774 / 0 |
| `doctest_residency.sh` | 746 / 833 (89.6%) | 746 / 833 (89.6%) |
| register exact both ways | 87 == 87 | 87 == 87 |
| shaders reproducible | — | **80 / 80 byte-identical** |

Zero divergence, and the shader reproducibility check is new: all 80 `.spv` were
recompiled from `.comp` with a locally-built glslang 15.1.0 and came out
byte-identical to the committed binaries. That closes a question nobody had
asked out loud — whether the checked-in SPIR-V is a build artifact of one
machine.

**Three things this run pinned that only a second driver generation could.** The
commits it covered lean on behaviour the Vulkan/GLSL specs do not fully nail
down, and Maxwell agrees with Ampere on all of it:

  * **Signed integer overflow wraps mod 2^32.** `ipow` chains up to 31 `int`
    multiplies and must give `pow(2, 32) == 0` and `pow(3, 20) == -808182895`.
    It does, on both.
  * **`int(b << 24) >> 24` is an ARITHMETIC shift**, which the narrow-integer
    widening depends on: 200 as u8 must sign-extend to -56 under the signed
    variant. Confirmed on the Tegra driver.
  * **`round` at a tie.** GLSL's built-in `round()` is implementation-defined
    and may round to even; the shader spells out half-away-from-zero instead.
    Had it trusted the built-in and this driver picked round-to-even, the tie
    vector would read `[0, -0, 2, -2, 2, -2, 4, -4]` instead of
    `[1, -1, 2, -2, 3, -3, 4, -4]`. **This is the strongest evidence yet that
    pinning a formula beats trusting a built-in** — the divergence it guards
    against is exactly the kind that only appears on hardware you did not write
    the code on.

**Three caveats on the Jetson, all environmental rather than results.** The OTP
there is built `--disable-jit` (the default JIT build ICEs in
`erts/emulator/asmjit`, almost certainly `cc1plus` running out of memory), and
the Rust NIF is built with relaxed LTO (`CARGO_PROFILE_RELEASE_LTO=false`,
`CODEGEN_UNITS=4`) for the same reason.

The third was found later and explains both: **the box runs `nvpmodel` at 5W
with only two of its four cores online**, against 3.9 GB of shared DRAM plus
2×991 MB of zram. That is why compiles are slow and memory-tight, and it is a
third independent reason the box is **unsuitable for timings** — which was
already the standing verdict, now for a concrete reason rather than a general
one. `sudo nvpmodel -m 0` would restore all four cores; it needs root, which
nobody has there.

#### EXLA on the Jetson — INSTALLED, and the fleet finally has a second reference

Built 2026-08-24. `parity_test.exs` had self-skipped on every box since it was
written; the Jetson is now the first box in the project's history to run it.

**The blocker was the C++ compiler, not any of the three risks that were sized.**
EXLA 0.13 depends on `fine`, whose header needs CTAD to deduce an EMPTY
parameter pack from `return fine::Ok();`. GCC <= 8 cannot, and an explicit
deduction guide does not rescue it. Ubuntu 18.04 ships g++ 7.5 and the box also
had 8.4 — `g++-8` fixed all the absl/XLA header errors and left exactly that
one. **`g++-13` from `ppa:ubuntu-toolchain-r/test` fixed it outright** (that PPA
carries 9 through 16 for bionic/arm64; 13 was chosen because it is the version
super-io runs and was empirically confirmed to compile the failing construct
before anyone spent a `sudo` on it). Note `add-apt-repository` is broken on that
box — its `softwareproperties` Python module is missing — so the repo has to be
added by hand.

**Memory was never the problem, and the prediction about WHICH unit was wrong.**
Peak 3222 MB used / 1005 MB available, and zram was never touched. The worst
translation unit was `exla.cc` — the one that used to fail — not the
template-heavy `exla_mlir.cc`, which came third CHEAPEST. 11m20s at `-j1`.

**Two previously untested claims are now tested facts.** The prebuilt
`libxla_extension.so` needs `GLIBC_2.27` and the box has exactly 2.27 — zero
margin, and it loads. And the no-LSE-atomics reading of the disassembly is
confirmed by execution: an LSE instruction on this ARMv8.0 A57 would have
trapped in `Nx.dot`. Both were static arguments until something actually ran.

##### The parity result — 18 pass, 0 fail, `parity_score: 1.0`

**Seventeen of eighteen fixtures agree with XLA to the BIT** (`max_err` exactly
0.0): `sum`, `product`, all four `cumulative_*`, `sort`, `argsort`, `reverse`,
`determinant` at f64, the boolean family, and `dot` at f64 both 8x8 and
16x32 @ 32x8.

The single non-zero is **`dot` at f32, 8x8: `9.5367431640625e-7`** — exactly
2^-20, about 1 ULP at that magnitude, against a 1e-6 tolerance. It does NOT
appear in the f64 path, so it is FMA/accumulation-order in the f32 matmul and
not a defect. Same species as the Ampere/Kepler `sqrt` divergence in §1.4.

The two skips are `fft` and `conv`, both marked `expected_status: :skip`. `fft`
is the one genuine capability divergence: EXLA computes it, this backend raises,
which is the standing decision that the ISA is real-valued.

##### Enabling EXLA changes none of the numbers

All three acceptance scripts, both configurations, on the Jetson at `a950c9f`:

| | with `NXV_WITH_EXLA=1` | without |
|---|---|---|
| `mix test` | 833 / 780 / 0 | 833 / 780 / 0 |
| `strict_test.sh` | 0 failures, 156 excluded | 0 failures, 156 excluded |
| `doctest_residency.sh` | 751 / 833 (90.2%), 82 == 82 | 751 / 833 (90.2%), 82 == 82 |

##### TOGGLING THE FLAG POISONS `_build`, IN EITHER DIRECTION

The one real trap, and it is worth knowing before it costs someone an hour.
`mix` bakes the resolved dependency list into
`_build/<env>/lib/nx_vulkan/ebin/nx_vulkan.app` and does NOT regenerate it when
only the environment changed, because no SOURCE changed. A flagged build leaves
`exla` in that file's `applications` list, and the next unflagged run aborts
before a single test:

```
** (Mix) Could not start application exla: could not find application file: exla.app
```

```sh
rm -f _build/test/lib/nx_vulkan/ebin/nx_vulkan.app \
      _build/test/lib/nx_vulkan/.mix/compile.app_cache
```

No recompile, nothing tracked touched. `rm -rf _build` is equally safe and costs
nothing: the 11-minute NIF build lands in `deps/exla/cache/` and `_build` only
symlinks to it — a re-flag after clearing took 14.7s.

#### The original research, for reference — and where it was wrong

##### The pre-attempt research said GO, and was right about everything except the compiler

Researched 2026-08-23. `parity_test.exs` self-skips with "EXLA not available" on
every box in the fleet, so a second independent reference has never existed. On
the Jetson it could:

  * `xla` v0.10.0 ships a precompiled **`aarch64-linux-gnu-cpu`** asset, so
    there is NO Bazel megabuild — only EXLA's own NIF (6 C++ files) compiles.
  * The ABI fits **with zero margin**: the binary's highest requirement is
    `GLIBC_2.27` and the box has exactly 2.27. It contains **no LSE atomics**,
    so it is safe on the A57's ARMv8.0, and its dot-product/bf16 kernels sit
    behind runtime cpuinfo dispatch the A57 never selects.
  * The target auto-detects to `aarch64-linux-gnu-cpu` with no env vars —
    `infer_xla_target/0` matches only CUDA `release 12.`/`13.`, and the box has
    10.2.

**CUDA is dead three ways** and not for the reason one would guess. Compute
capability 5.3 is NOT the blocker for the cuda12 build. The blockers are that
the Nano is capped at JetPack 4.6.x / CUDA 10.2 permanently (CUDA 11 first
shipped in JetPack 5.0, which dropped this board), that the prebuilts come only
in cuda12/cuda13 flavours, and that cuDNN is 8.2.1 where XLA wants 9. cuda13
additionally raised the floor to SM 7.5.

**The one unverified risk is the NIF compile OOMing**, which is recoverable —
retry, or free memory — unlike a source build. `schedulers_online` is 2, so
EXLA's default job count computes to `-j1`: serial, slow, RAM-friendly.

Worth doing, because XLA's CPU backend is a genuinely INDEPENDENT implementation
(LLVM codegen, different accumulation orders, different fast-math choices)
rather than a second pure-Elixir reference — it catches a different bug class,
which is the whole point of a parity suite. Note this is the OPPOSITE conclusion
from the standing "EXLA is unbuildable" note, which is about the OSS `exmc` repo
and not about this hardware.

**The unified memory finding.** Tegra shares one physical DRAM between CPU and
GPU, and the memory types confirm it: type 2 is
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`. The load-bearing consequence is
the negative one — **there is no memory type that is HOST_VISIBLE but NOT
DEVICE_LOCAL.** An allocator hunting for a separate non-device-local staging
type finds nothing here. This backend already allocates with
`PREFER_DEVICE | HOST_SEQUENTIAL_WRITE` and uses no staging buffers, so nothing
needed changing — but a future staging path would have to notice. Also worth
knowing: type 3 is host-visible WITHOUT `HOST_COHERENT`, so anything that ever
selects it needs explicit flush/invalidate.


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
`function_exported?` for a new NIF — AFTER `Code.ensure_loaded!/1`, see §0 —
rather than trusting a green compile, because `mix`
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

### 1.4a The uninitialised allocator — and a prediction the fleet falsified

`084b937`. `alloc_buffer` zero-filled every output buffer through
`std::iter::repeat(0u8)` — a full host-side write, paid by every op that
produces an output, for bytes a shader was about to overwrite.
`Buffer::new_slice` skips it.

**The audit was the work, and the fleet turned it from reasoned into verified.**
All 87 shaders were checked for two hazards: accumulating into the output, or
writing fewer bytes than allocated. Exactly four are in the first class —
`allany_{f32,f64,s32,u8}`, which `atomicOr` one thread per slot into a packed u8
mask — and they use a new `buf_alloc_zeroed/1`.

What made it evidence rather than argument is what the Keplers did with it. A
clean run proves little here, because freshly mapped pages are OS-zeroed: a
shader wrongly depending on zeroed memory passes cold and fails once blocks
recycle. So they built a control first, **and their first poisoning scheme was
vacuous** — `buf_upload` buffers land in a different suballocator pool and came
back 0/24 dirty. Only after finding a scheme that produced 40/40 dirty
reallocations did they trust the result: ~4,275 poisoned operations, zero
defects, plus direct byte-level downloads of the PADDED buffers confirming every
tail lane reads zero.

The Jetson was then said to have repeated it on unified memory with 6 x 32 MiB
`0xFF` blocks churned before every case, and to be "also clean". **That leg was
vacuous corroboration, in the same paragraph that praises the Keplers for
catching exactly this.** The 6 x 32 MiB scheme produces **0/40 dirty
reallocations** on Tegra, and the reason is two subsections below: 32 MiB is
precisely the dedicated-`vkAllocateMemory` cliff. At and above it vulkano stops
suballocating, so the pages return to the driver on free and come back freshly
zeroed — it is the one size class that *cannot* poison. A sweep confirms
the boundary is the whole story:

    6 x 32 MiB (the recorded scheme)                    0/20 dirty
    16 x 8 MiB, 64 x 1 MiB, 512 x 64 KiB, 2048 x 4 KiB  20/20 dirty

Re-run under 16 x 8 MiB (40/40 dirty), the Jetson is genuinely clean, so the
conclusion stands. But it was *unestablished* while this section asserted it, and
the Kepler result was carrying the claim alone.

**And the cliff is not Tegra's.** The obvious reading of the above is that
unified memory puts the boundary where it is, so a discrete box would have it
somewhere else or not at all. Every box was asked, and the answer is a clean
binary split at exactly 32 MiB with no gradient anywhere:

    box                              6 x 32 MiB    sub-cliff schemes
    mac-247   GT 650M   Kepler          0/40            40/40
    mac-248   GT 750M   Kepler          0/40            40/40
    super-io  RTX 3060  Ampere          0/40            40/40
    jetson    Tegra X1  unified         0/40            20/20

Different silicon, different drivers, two operating systems, discrete PCIe and
unified LPDDR4 alike. **32 MiB is vulkano's dedicated-allocation threshold, not a
memory-architecture artifact** — which means the vacuous scheme was never
specific to the box that ran it, and any future poisoning attempt sized at the
cliff is vacuous everywhere. That is the argument for `scripts/poison_control.exs`
existing at all: the schemes lived in `/tmp` on three machines, so nobody could
compare them, and the one that could not work looked exactly like the ones that
could.

**A third vacuity, found three times independently.** The lesson from the concat
fix was to build test operands by COMPUTING them, because `Nx.tensor/2` uploads
exact-sized buffers and only kernel-allocated ones carry padding. That lesson
does not compose with `{:s,64}`: there is no s64 add shader, so `Nx.add` returns
a host tensor, `all_vulkano?/1` is then false, and `concatenate` declines —
correctly. All three re-verifying boxes wrote a concat residency assertion the
assuming way, all three got a false failure on s64, and all three tracked it to
a missing integer kernel one step upstream rather than to the guard. The
standing trap: **any future concat test using computed s64 or u64 operands is
silently vacuous for those types**, because the operands never reach the device.
`poison_control.exs` had the same flaw and now measures operand residency
instead of assuming it (`ca74518`).

#### The prediction was wrong, and the reason is the interesting part

I wrote that the Jetson would show the win most, having the slowest memory in
the fleet. **It shows it least** — 3.61 ms saved at 16 MiB (37x) against
Ampere's 5.07 ms (635x), smaller both absolutely and proportionally.

The control disproves the reasoning cleanly. `buf_upload` IS slower on Tegra
(6.0 ms vs Ampere's 3.6 at 16 MiB), so the memory really is slow. But the
ZERO-FILL is FASTER there (3.71 vs 5.08). On Tegra the buffer is
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`, so writing a constant into LPDDR4
costs less than Ampere's host write across PCIe into BAR memory.

**Unified memory made the OLD path cheap, so there was less to reclaim.** The
bottleneck this change removes was never memory speed — it was the bus. Which is
also why the unified-memory optimisation I originally went looking for did not
exist: there was no staging copy to eliminate, because `HOST_SEQUENTIAL_WRITE`
already writes straight into mapped memory.

#### And a cliff a coarse grid hides

Confirmed on two architectures by a fine sweep. Below 32 MiB the new path is
flat and O(1); at and above, vulkano stops suballocating from a pooled block and
issues a dedicated `vkAllocateMemory`, after which the driver commits and zeroes
pages itself at ~0.83 ms/MiB on Tegra — outside this code's control.

    Tegra X1, MiB:    8      16     24     31     32       48     64
    buf_alloc  ms:    0.09   0.12   0.14   0.12   26.98    39.69  52.92
    old path   ms:    2.94   3.71   5.38   6.70   55.87    83.27  110.58

Still ~2x above the cliff, but **"O(1) allocation" is only true below 32 MiB**,
and a 1/4/16/64 benchmark grid straddles it invisibly. Both boxes that measured
it found it independently.

### 1.5 `indexed_put`'s duplicate-index race — and why "conforming" was not enough

Closed in `d548c85`. Worth its own section because **the decision was harder than
the fix**, and I got it wrong first.

`Nx.indexed_put/4` documents duplicate indices as non-deterministic:

> "In case of repeating indices, the result is non-deterministic, since the
> operation happens in parallel when running on devices such as the GPU."

So the racing single-dispatch write was **conforming**, and `scatter.comp`'s own
comment said so with that quote in it. **I recommended fixing it as a bug
without reading the shader I was proposing to change.**

What made determinism worth buying anyway is not "it disagrees with the host" —
that is permitted — but the fleet table:

| | result | stability |
|---|---|---|
| `Nx.BinaryBackend` | `[30, 0, 0]` last update | defined |
| RTX 3060 Ti (Ampere) | `[10, 0, 0]` **FIRST** row | stable, 5/5 |
| GT 650M, GT 750M (Kepler) | `[30, 0, 0]` **LAST** row | stable, 10/10 |
| Tegra X1 (Maxwell) | `[30, 0, 0]` **LAST** row | stable, 25/25 |

Every box individually deterministic, boxes disagreeing with each other, and
**three of four agreeing with the host BY ACCIDENT**. The bug was invisible on
three quarters of the fleet, and a stable-per-box answer that differs across
boxes is harder to debug than one that is obviously random.

`scatter_ordered.comp` runs two passes over the same index arithmetic sharing
one descriptor set: `atomicMax(winner[dst], row + 1)`, then write where
`winner[dst] == row + 1`. `row + 1` because the scratch buffer is zero-filled
and 0 must mean unclaimed; `atomicMax` yields the highest row, which is Nx's
last update. The zero-fill and both dispatches record into ONE command buffer —
vulkano's `AutoCommandBufferBuilder` inserts the compute-compute barrier, the
same mechanism the FFT's multi-stage transform relies on.

**Cost, measured on the two IDLE Keplers** (load 0.00–0.18 before and during),
racing → ordered, median ms:

| case | GT 650M | GT 750M |
|---|---|---|
| tiny target(8), 4 updates | 0.266 → 0.290 | 0.144 → 0.189 |
| 1M target, 4 updates | 1.181 → 1.981 | 1.053 → 1.860 |
| 4M target, 100k updates | 4.830 → 10.502 | 4.519 → 10.672 |

The overhead tracks the **target** size, not the update count — one u32 per
target element, zero-filled — so scattering four values into a large tensor pays
most, proportionally. `indexed_add` is untouched: atomicAdd is commutative, so
ordering never could reach its answer. **`NXV_SCATTER_ORDERED=0`** restores the
racing write.

**An Ampere timing table previously stood here and was withdrawn.** It was taken
without checking super-io's load. The fleet caught it by cross-comparison — my
table had a GT 650M beating an RTX 3060 Ti 2:1 — and a re-run at load 4.65 with
another `beam.smp` at 337% CPU reported the ordered path as 0.179 ms FASTER than
racing on one row. Noise, not a result. **Check `uptime` before AND during any
timing** is a standing rule here and the original run did not.

### 1.6 The health sweep — dialyzer, docs, coverage, format

Two days on non-residency work. It found **one real bug** and several traps, and
per hour it outproduced the last few residency items.

#### The bug: an f32 constant in an f64 chain — and how I oversold it

`Nx.Vulkan.Fast.normal_logpdf/3` computed at **f32 precision on f64 input**.
`@log_sqrt_2pi` was a bare Elixir float, so `Nx.tensor/1` materialised it at
`{:f, 32}`: `0.91893853320467274` became `0.9189385175704956`, losing eight
significant digits. Subtracting that from an f64 tensor yields an f64 RESULT
TYPE carrying an f32-precision VALUE — `normal_logpdf(0, 0, 1)` answered
`-0.9189385175704956` where the exact value is `-0.9189385332046727`.

**I first wrote that it "feeds a Metropolis acceptance ratio across thousands of
leapfrog steps." That is the one use where the bug has ZERO effect**, and I took
it from the module's own statement of intent instead of checking. The error is a
CONSTANT — the same additive term on every element of every call — and a
Metropolis ratio is `logp(proposed) - logp(current)`, so it cancels exactly:

```
exact ratio       : -0.06000000000000005
both shifted by e : -0.06000000000000005     identical
```

Where it does bite is a SUMMED absolute log-likelihood, accumulating as
N x 1.56e-8: 0.016 at N = 1e6, 1.56 at N = 1e8. That is the range where model
comparison (WAIC, LOO, Bayes factors) argues over ~0.1 in log-evidence.

**And it was one instance, not a class** — the only float literal in `lib/` with
more than seven significant digits outside a GLSL string. Reproduced on
`BinaryBackend` alone first, so it was never a GPU issue.

**Why it survived four months is the interesting part.** It was not an
oversight: `Nx.Vulkan.Fast`'s fused path was **f32-only** (every backend callback
gated on `all_f32?/1`), so an f32 constant was the correct width for the world
the module was written in. It became wrong when the project went f64-first and
nothing re-examined the module. Correct-for-its-era code left behind by a change
of era, which is a different failure mode from carelessness and needs a
different defence — the one that works is running the checks, since the module
had no tests and no callers to notice.

**The module has since been deleted** (see the CHANGELOG): its dispatch
mechanism was removed by Nx 0.12 and its target dtype was abandoned by this
backend, so it had no mechanism, no target and no callers. The fix is recorded
here because the LESSON outlives the code — materialise a float constant at the
computation's type, because `Nx.tensor/1` defaults to f32 and will silently
degrade an f64 chain while leaving the type signature intact.

#### What each check was actually worth

  * **Dialyzer: no bugs, and that was the finding.** `mix dialyzer` was already
    clean — but clean at DEFAULT strictness, and dialyxir enables no extra
    warning flags at all, so `:unmatched_returns` had never run. It is the one
    that finds an unhandled `{:error, _}`. Enabled now, with the four
    best-effort returns it reported written `_ =` at the site rather than
    filtered. `:underspecs`/`:overspecs` are deliberately NOT on: 13 findings,
    every one a spec intentionally narrower or wider than the inferred typing,
    which is what a hand-written spec is for.
  * **Coverage: 87.20% → 90.78%, and the threshold now means something.**
    `NativeV` is 426 lines of Rustler stubs whose bodies are replaced at load
    time; it can never exceed ~2%, so including it made the repo's own 90%
    threshold **unreachable by construction**. A threshold that can only fail
    stops being read. Excluded, and the genuinely untested code found behind it:
    the three `*_push` builders pack C structs a shader reads BY OFFSET (a
    field in the wrong order returns plausible numbers, not an error), and
    `ShaderTemplate.derive_grad_n/1` never runs in production because all three
    shipped specs supply their own `grad_block_n`. (`Nx.Vulkan.Fast` was in that
    list at 0% and has since been deleted rather than tested — the coverage
    number was what sent someone to look at it.)
  * **Docs: `mix docs` was emitting warnings.** `ROADMAP.md` linked `MISSION.md`,
    which is not in the hex package **and carries internal LAN addresses** —
    link removed rather than repointed, because a public roadmap should not send
    readers to an internal planning doc at all.
  * **Doctests: the suite ran `doctest Nx` and not one of its own.** Adding them
    found `Nx.Vulkan.Synthesis`'s "Usage" example had NEVER compiled — five
    undefined variables, because it was an illustrative sketch written as an
    `iex>` block. Now a fenced code block.
  * **Format: 29 files unformatted, 20 of them untouched by that session.**
    Long-standing drift. Committed separately (`1eec6ec`) so a 675-line
    mechanical diff did not bury behaviour.

#### And one script deleted rather than repaired

`scripts/build_and_test.sh` pointed at `/home/io/spirit/shaders`, a machine-local
directory. That was never a build system: per `MISSION.md` it was a RECOVERY AID
for seven shader sources that had gone missing from the tree, and it pointed
there because it was "the last thing that knew where they were." That recovery
finished in `ac509d2`. The directory no longer exists on any box, so under
`set -eu` the script had been dying on its first unexpanded glob — long enough
for nobody to notice, which is the evidence it had stopped being run.

**The substance was not the deletion.** It was that the glslang version was
pinned nowhere: SPIR-V output is a function of the compiler, so every
reproducibility claim this project has made is relative to an unstated version.
`SKILL.md` now names it — **glslang 15.1.0, SPIR-V 1.6, generator 11** — with
the warning that a mismatch yields a working shader and a useless comparison.

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
| **Re-verify the fleet at `661477f`** | done five times before, most recently at `d548c85` (§1.4). **Three of the five found a defect** — twice in a test, once a compile warning — which is the argument for running it every time rather than before releases only. A run at `661477f` is in flight; it is the first to carry a whole-tree `mix format`, a deleted script and an untracked `glsl/*.spv`, so a stale working tree there is the thing to watch | anyone |
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
# RUN THIS EVERY TIME, not just at the end. It is the ONLY check that can see a
# test which provokes a fallback without carrying @tag :host_fallback_expected —
# `mix test` passes (a fallback is bit-identical) and doctest_residency.sh never
# reads these files. That exact mistake shipped twice and was caught by the
# fleet, not here. See §1.4.
sh scripts/strict_test.sh            # 834/855/0, excluded count moves with the register

# the number that actually means something
sh scripts/doctest_residency.sh      # 755 / 833 (90.6%), exits 0
                                     # 744 / 833 (89.3%) device-resident

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
