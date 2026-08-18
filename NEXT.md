# NEXT — nx_vulkan

**Written:** 2026-08-16, against `main` @ `40d3137` (the stale-figure sweep).
**Refreshed:** 2026-08-17, against `main` @ `c9b1a31`, with **W4 and the
`concat_nd` shader both landed** on top of W2/W1/W3. The ranking is unchanged;
the super-io re-verification is closed, the leapfrog ownership question is
settled *and fixed downstream*, and the work item W4's census surfaced — an
axis > 0 `concatenate` shader — is **written and green**. Next is W5, and it is
now scoped and priced: **§1.2**.

**The reboot happened, and both worries came to nothing.** The driver came back
matched at **580.178.04** on both sides, `device_name()` is the 3060 Ti, and all
three figures reproduce exactly (§5). The five commits are **pushed** —
`origin/main` is at `9167899`, level with `HEAD` (§0).
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
| `HEAD` / local `main` | `9167899` | W2 + W1 + W3 + W4 + `concat_nd` |
| `origin/main` | `9167899` | **level — the W4/concat work is pushed** |
| `upstream/main` | `6ab64ac` | **48 behind** |

The five commits that were the only copy of W4 and the concat shader are on the
private server as of 2026-08-17. Nothing here is unbacked any more:

```
9167899  docs(NEXT): refresh against c9b1a31, before the super-io reboot
c9b1a31  concat_nd: an axis > 0 concatenate shader, and it moves five ops at once
70117d5  docs(NEXT): the §0/§1 ledger, refreshed against the merge
73c57f0  docs(NEXT): the leapfrog defect is fixed upstream, and why it nearly wasn't
cae4dad  W4: fold the 30 doctests it moved, and mark it done
cc77b2a  W4: route eight Nx.Block.* on-device, allowlist the four FFT ones
```

Push with `git push origin main` — **not** `upstream`, which is the public
release remote and is deliberately 48 behind.

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

**The shader invariant: 57 `.comp` ↔ 57 `.spv`, every blob regenerable.**
Established in `ac509d2`, checked by the skill on every run, and holding as of
`a25432f`. Until then,
seven `.spv` had no source in the tree and the only copies lived in
`~/spirit/shaders/` on the two Keplers, outside any repository. If the skill
ever reports `orphan (kept)`, look there before anything else.

---

## 1. W2, W1, W3, W4 and `concat_nd` are done. Next is W5

`MISSION.md` §7 ranks W1–W13 and nothing since has changed the ranking. Its
sequencing note put W2 first, because W2 is what tells you whether W1 and W5
worked. **That gate is open, and it has now been used four times.**

| item | state |
|---|---|
| **W2** — strict ratchet on `doctest Nx` | **done** `6f8d406` |
| **W1** — word-generic remap family | **done** `912ce08` `578cf3a` |
| **W3** — `Nx.LinAlg.solve/2` | **done** `f614dd0`…`62b622e` |
| **W4** — decide the twelve `Nx.Block.*` | **done** `cc77b2a` `cae4dad` |
| **`concat_nd`** — axis > 0 concatenate | **done** `c9b1a31` — not a W item; W4's census found it |
| **W5** — integer kernels | the 357-doctest bucket; the big one, and now the clear next. **Scoped in §1.2** — worth 47.2% → 76.0% |

```sh
sh scripts/doctest_residency.sh
#=> doctest Nx residency: 398 / 843 (47.2%) run with host fallbacks refused
```

`@moduletag :host_fallback_expected` is off `nx_doctest_test.exs`;
`test/nx_doctest_register.exs` names the 445 doctests that still leave the GPU,
131 lines in four reason-bucketed lists; `test_helper.exs` applies it only when
fallbacks are being refused, so a normal `mix test` still runs and asserts all
843. The strict suite went from 910 excluded to 591 at W2, then 557 as W1 and
W3 moved doctests onto the device, 527 at W4 and 518 after `concat_nd`. CI runs the script as its own step. See `MISSION.md` §2.3 for what was built and the one departure
from the plan (ExUnit's `doctest :except` is function-granularity; using it
would have dropped 154 *resident* doctests and reported 165/843).

**The register is portable.** It was measured on super-io (Ampere/Linux) and
reproduces byte-identically on mac-247 (Kepler/FreeBSD) — same 524 at W2, same
496 at W1, same 488 at W3. The gates really are dtype/shape logic. The one exception found so
far is **llvmpipe**, where `Nx.sum` on `{:u, 8}` returns 0 and three doctests
plus three `select` tests fail on value; if a run reports one extra fallback,
check `device_name()` before touching the register.

**W1, W3 and W4 were all measured with it, and it worked every time.** The
rate moved 319 → 347 → 355 → 385 → 398, and every time the ratchet failed the
build on stale entries and named every one. That is the loop this project was missing.

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
The buckets are scored as work items:

| bucket | doctests | item |
|---|---:|---|
| `@integer_dtype` | 357 | **W5** — but it does *not* empty this bucket wholesale; §1.2 has the census that says so, and the simulation that prices it at 47.2% → 76.0%. W1 took 28, W3 took 8 |
| `@float_residency_gap` | 32 | **W8** and the rest of the narrow-gate work — float ops that still left a float backend. Rank-0 `dot`/`product`/`reduce`/`divide`, `dot` at `{1,1,2,2}`, rank-3 windows, and `Nx.log2`/`log10`/`log/2` refusing at f32 while `Nx.log/1` runs natively |
| `@f64_transcendental` | 37 | not work — GLSL.std.450 has no f64 `Sin`/`Log1p`/`Erf`. Same constraint that allowlists `pow/3` |
| `@complex_and_fft` | 20 | not work under current dtype support. W4 allowlisted the four FFT blocks, which took 25 doctests out of this bucket without moving them onto the device |

### 1.1 `concat_nd` — the census cashed in

`glsl/concat_nd.comp` (`c9b1a31`). W4's census named three gaps; this closed one
of them and **all five ops that shared it went resident at once** — all four
`Nx.cumulative_*/2` and `Nx.take_along_axis/3`, zero fallbacks. Residency
385 → 398 (45.7% → 47.2%), strict 527 → 518 excluded, register 458 → 445.

**Unlike W4's own 30, this 13 needs no asterisk.** Every one is genuinely
device-resident, not merely permitted to leave. Refused-clean and
device-resident readings are now 398/843 (47.2%) and 373/843 (44.2%).

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

### 1.2 W5, scoped — the census, and what it is actually worth

Measured on super-io at `1c57eb7`, 2026-08-17, by the same method W4 used: run
the 843 `doctest Nx` under `NXV_HOST_FALLBACK=raise` with the register off, and
read the op and dtype out of every `Nx.Vulkan.HostFallbackError`. All 445
failures parse; the 357 in `@integer_dtype` partition exactly.

**The headline first, because it is the argument for doing W5 at all.** The
allowlist can be used as a simulator: excuse a family of callbacks and re-run,
and the residency you get back is the residency that family's shaders would
buy. Excusing every op W5 would write:

| tier | what is excused | residency |
|---|---|---:|
| — | today | 398 / 843 (47.2%) |
| **T1** | integer elementwise binary + unary + compare + select | **535 / 843 (63.5%)** |
| **T2** | T1 + integer axis-reduce and window-reduce | **624 / 843 (74.0%)** |
| **T3** | T2 + integer `dot` | **641 / 843 (76.0%)** |

**+243 doctests, 47.2% → 76.0%.** Nothing else on the board is within an order
of magnitude of that. The figure errs in both directions and roughly cancels:
it over-counts by ~14, because excusing `window_sum/4` excuses its f32
instances too and those belong to `@float_residency_gap`; and it under-counts,
because an *allowlisted* op still computes on the host and returns a host
tensor, so every residency-gated op downstream of one still reports. That
second effect is visible and large — see the residual below.

#### The 357 is three different gaps, not one

`MISSION.md` §7 scopes W5 as "integer elementwise / compare / select / reduce
kernels" and the register scores it at 357. **Those are not the same number.**
Classified by what is actually in the way:

| class | doctests | what it is |
|---|---:|---|
| **A — dtype-gated** | **195** | a shader exists for f32/f64 and the selector returns `nil` for integers. `binary_spv/1`, `unary_spv/1`, `compare_spv/1`, `select_spv/1`, `reduce_spv/2`, `window_reduce_spv/1` — six functions, each ending in `defp …(_), do: nil`. This is W5 as written |
| **B — no path at any dtype** | **138** | the callback is in `@host_fallback_unary_ops` / `@host_fallback_binary_ops`, or transfers unconditionally. f32 falls back here too. Writing an integer shader does *not* close these; each also needs an op code and a route |
| **C — shape/residency-gated** | **24** | dtype is not the gate. `gather/4` off-prefix axes, `concatenate/3` with a host operand, int→int `as_type/2`. They sit in `@integer_dtype` only because Nx's doctests are `{:s, 32}` |

Class B splits again, and the split is the plan. **67 of the 138 ride the same
shaders as class A** — `quotient`, `remainder`, the three bitwise binaries, both
shifts, the three logicals, `bitwise_not`, `population_count`,
`count_leading_zeros`, `is_nan`, `is_infinity`, `product` — they need an op code
in a kernel W5 is writing anyway, plus deletion from the host-fallback list. The
other **71** are separate work items and W5 should not claim them:
`indexed_put/5` (20) and `indexed_add/5` (4) have no scatter shader for any
dtype, `argmax`/`argmin` (22) no shader at all, `reduce/5` (10) takes an
arbitrary fun, `all`/`any` (10) are u8 boolean reductions, `stack/3` (5)
transfers unconditionally and could simply route to `concatenate/3`.

#### It is a 32-bit job

Of the 195 class-A doctests: **168 are `{:s, 32}`**, 23 are `{:u, 8}` *outputs*
of comparisons whose *inputs* are `s32` — the compare shader already writes
packed u8, so those need an `int` input variant and nothing more — and **4 are
`{:s, 8}`**. Across the whole 445 there are five `{:s, 64}` rows, four
`{:u, 32}`, two `{:s, 16}`.

So `int`/`uint` in core GLSL 450 covers 191 of the 195. **No `Int64`
capability, no `GL_EXT_shader_8bit_storage`, no `16bit_storage`** — nothing that
would need checking against the Kepler fleet. The four s8 rows are a documented
tail, not a scope item.

**And no Rust.** Every dispatch entry point already takes the SPIR-V path as a
parameter — `apply_binary(out, a, b, n, op_code, spv_path)`,
`apply_unary/5`, `apply_compare/7`, `apply_select/8`, `reduce_axis/7`,
`window_reduce/7`, `matmul32/7`. W5 is `.comp` files, `glslangValidator`, and
one clause per selector. The `native/` tree does not move.

Seven new shaders, taking the invariant from 57 ↔ 57 to 64 ↔ 64:
`elementwise_binary_s32`, `elementwise_binary_bcast_s32`,
`elementwise_unary_s32`, `compare_s32`, `select_s32`, `reduce_axis_s32`,
`window_reduce_s32` — plus `matmul_s32` for T3.

#### Six semantics traps, each measured against `Nx.BinaryBackend`

The correctness test for W5 is bit-equality against `BinaryBackend` on integers,
which is *exact* — there is no tolerance to hide in. These are the places where
the obvious GLSL gives a different answer. All six were run on the host at
`1c57eb7`:

| what | `BinaryBackend` says | the trap |
|---|---|---|
| `sum` of `{:s, 32}` overflowing | `2e9 + 2e9 → -294967296` | it **wraps**. The f32 reduce shader accumulates in `double` to match; an s32 reduce must **not** widen, or it disagrees exactly where it matters |
| `sum` of `{:s, 8}` | type `{:s, 32}` | reductions **widen**, like the existing `{:u, 8} → {:u, 32}` entry. `reduce_spv/2` is keyed on the (in, out) pair for precisely this reason — keep it that way |
| `multiply` on `{:s, 8}` | `100 * 100 → 16` | elementwise ops wrap **at the element width**, not at 32 bits |
| `remainder` | `-7 rem 3 → -1`, `7 rem -3 → 1` | sign of the **dividend**. GLSL's `%` is *undefined* for negative operands — write `x - (x/y)*y`, do not use `%` |
| `quotient` | `-7 / 3 → -2` | truncates toward zero. Same GLSL caveat |
| `count_leading_zeros` | `0 → 32`, `1 → 31`, `-1 → 0` | `findMSB` returns `-1` for zero; the zero case needs its own branch |

One thing that is *not* a trap: `Nx.divide` on two integers returns `{:f, 32}`,
so the integer binary shader needs no divide op code at all.

#### Read the demand histogram, not just the first-fallback count

Under `:raise` each doctest reports once — its *first* fallback. Under
`:warn` all 843 run to completion and report **759** fallbacks. The two
orderings disagree sharply, and the second is the one that predicts how much
GPU work W5 keeps resident:

| op | first-fallback | all fallbacks |
|---|---:|---:|
| `max/3` | 13 | **89** |
| `add/3` | 15 | **70** |
| `concatenate/3` | 8 | **57** |
| `subtract/3` | 23 | 47 |
| `sum/3` | 24 | 31 |

Integer `max` is the single most-called missing kernel in the whole API, and it
ranks thirteenth by the first-fallback count. `Nx.clip/3`, `Nx.mode/2` and the
`cumulative_*` family all lean on it.

#### What is left at 76%, and why some of it is an artifact

The residual after the full simulation is 202 doctests. The honest reading:

```
34  concatenate/3   s32     ← artifact: see below
22  indexed_put/5           no scatter shader, any dtype
16  do_fft/4        c64     decided (complex)
12  as_type/2               int→int and f16/bf16 casts
11  gather/4        s32     off-prefix axes
11  argmax/3        s32     no shader
11  argmin/3        s32     no shader
11  reduce/5        s32     arbitrary-fun reduce
40  f64 transcendentals     decided (GLSL.std.450)
```

**`concatenate/3` going 7 → 34 is the simulation lying, not a real gap.** An
allowlisted op still computes on the host, so its result comes back on
`BinaryBackend`, and `concat_nd`'s `all_vulkano?` gate — the one §1.1 explains
at length and pins a test on — then refuses. With real integer shaders those
operands stay resident and most of the 34 close for free. The same applies to
`gather/4` (11) and `stack/3` (5). Do not plan work against those rows.

**`indexed_put/5` at 22 is the real find.** It is an unconditional host
fallback for *every* dtype, it is the largest single non-decided residual after
W5, and `MISSION.md` §3.3 already noted that `Nx.LinAlg.invert/1` dies there.
It deserves a W-number of its own; a scatter shader is the natural sibling of
`gather.comp`, which has existed all along.

#### Suggested order

T1 first, and stop to measure. It is 137 doctests on its own, it is the tier
whose shaders every later tier reuses, and it is where all six semantics traps
live — get wrapping and sign conventions bit-exact against `BinaryBackend` on
30 lines of GLSL before there are 200. Then T2, then T3. Run
`sh scripts/doctest_residency.sh` before and after each; if the rate does not
move, the op did not reach the device.

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
| ~~Push to `origin`~~ | **done** — `origin/main` at `9167899`, level with `HEAD` | |
| ~~Re-verify on super-io~~ | **done twice** — once at `a930157`, again after the 2026-08-17 reboot at `9167899`: driver matched 580.178.04 both sides, `device_name()` the 3060 Ti, all three figures exact (§5) | |
| **Re-verify on mac-247** | not run since W4; W4 and `concat_nd` are super-io-only measurements | anyone with the Kepler |
| **`mix hex.retire nx_vulkan 0.2.0`** | hex.pm still reports `retirement: None` | **operator only** — needs an interactive Hex password |
| **`upstream/main` is 47 commits behind** | unpublished | **operator** — publishing decision |
| **Consumer pin is 10 commits behind** | `../_exmc-things/exmc/mix.lock` still on `a25432f` | anyone, but see §4 — bump it *with* `bench/nuts_truth.exs` on both arms |

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
so it has been bumped once already. Bumping it is a deliberate act that should
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
# suite: 843 doctests, 526 tests, 0 failures.
# Last measured on super-io at 9167899, after the reboot. mac-247 has NOT been
# re-run since W4 — see §1 and the reboot note below.
mix test

# strict — did the work stay on the GPU?
sh scripts/strict_test.sh            # 843/526/0, 518 excluded

# the number that actually means something
sh scripts/doctest_residency.sh      # 398 / 843 (47.2%), exits 0

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

The two must match. They both read **580.178.04** at `c9b1a31` and still do
after the 2026-08-17 reboot at `9167899`, which is the state every figure in
this file was measured under.

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
