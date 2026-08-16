# NEXT — nx_vulkan

**Written:** 2026-08-16, against `main` @ `40d3137` (the stale-figure sweep).
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
| `HEAD` / local `main` | `40d3137` | |
| `origin/main` | `7067499` | **1 behind** — the stale-figure sweep is unpushed |
| `upstream/main` | `6ab64ac` | **30 behind** |

`7067499` is also the sha that `_exmc-things/exmc/mix.lock` pins, so pushing
`40d3137` to `origin` does not move either consumer until someone bumps the pin
deliberately. That is the right default; see §4.

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
if you edit a `.comp`, you must re-run `glslangValidator` by hand (see the
skill, §3). `priv/shader_cache/` is gitignored and safe to delete. And
`~/.exmc/gpu_node/spv/` caches synthesised shaders **keyed by a hash of the
generated GLSL**, so it invalidates itself correctly — but delete it if you
suspect otherwise.

---

## 1. The plan is unchanged: W2 first

`MISSION.md` §7 ranks W1–W13 and nothing since has changed the ranking. The
sequencing note there is the important part and bears repeating:

> **W2 before W1 and W5**, because W2 is what tells you whether W1 and W5
> worked. The residency rate is the acceptance test for the whole of §3.

**W2 — turn the strict ratchet on `doctest Nx`.** Baseline **319 of 843 (38%)**
run entirely on the GPU. The work is retiring one `@moduletag` in favour of an
except list and printing the rate in CI. Until that number is in CI, every other
item is unmeasurable, and "unmeasurable" is how this project's two worst bugs
survived.

Then W1 (word-generic remap family, best ratio available), W3, W4, W5.

---

## 2. Housekeeping still open

From `MISSION.md` §9 and `PLAN_AFTER_BACKWARD_PASS.md`. The 2026-08-16 sweep
(`40d3137`) closed the stale-figure items — suite counts corrected in six files,
`ROADMAP.md`'s banner hoisted, `PARITY_STATUS.md` and `NX_PARITY_RESEARCH.md`
bannered, T12's two dead `:host_fallback_open` tags deleted after verifying
under `NXV_HOST_FALLBACK=raise`. What it did **not** close:

| item | state | who can do it |
|---|---|---|
| **Push `40d3137` to `origin`** | 1 commit unpushed | anyone |
| **`mix hex.retire nx_vulkan 0.2.0`** | hex.pm still reports `retirement: None` | **operator only** — needs an interactive Hex password |
| **`upstream/main` is 30 commits behind** | unpublished | **operator** — publishing decision |

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
`_exmc-things/exmc/docs/OPEN_VULKAN_OBSERVED_MODEL.md`.

It is not yet known which side of the NIF the fault is on, and the experiment
that decides it is one this repo is better placed to run:

> Fix `q0`, `p0`, `eps`, `inv_mass`, `K = 32`. Dispatch
> `leapfrog_chain_synth_f64`. Read back `q_chain`, `p_chain`, `grad_chain`,
> `logp_chain`. Run the same K leapfrog steps on the host. Compare all four
> element-wise.

If they agree, the fault is in how eXMC consumes the arrays and this repo is
clear. If they diverge, the step index at which they first diverge names the
bug. **This is a half-day and it settles ownership** — worth doing before either
side spends longer guessing. The strongest lead recorded so far is that the
adapted step size is bit-identical across a change that alters every log-density
in the trajectory, which points at the host side.

---

## 4. What this backend owes its consumers

`MISSION.md` §5 covers this; two additions from 2026-08-16.

**The pin is a feature, not friction.** `_exmc-things/exmc/mix.lock` pins
`7067499`. Bumping it is a deliberate act that should come with a run of that
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
# suite (super-io, at 40d3137): 843 doctests, 456 tests, 0 failures
mix test

# strict — the number that actually means something
NXV_HOST_FALLBACK=raise mix test     # 843/456/0, 910 excluded

# confirm the real GPU, not llvmpipe, before believing any perf figure
Nx.Vulkan.NativeV.device_name()      #=> {:ok, "NVIDIA GeForce RTX 3060 Ti", "DiscreteGpu"}
```

**Residency is not correctness, and a value assertion cannot see the
difference** — the host fallback *is* `Nx.BinaryBackend`, the reference every
test compares against, so a refused GPU gate returns a bit-identical result.
Count fallbacks (`Nx.Vulkan.Fallback.count/1`); it is the only signal. And the
count is a **lower bound**: once a tensor lands on `BinaryBackend`, everything
downstream computes there unrecorded.

**Validate perf heuristics across the fleet, never on one box.** Win/loss
crossovers here are hardware-specific — the many-slot fused reduce wins ~4.4× on
Kepler and *regresses* ~0.44× on Ampere. mac-247 (GT 650M) is the quiet box at
±2–4%; mac-248 (GT 750M) runs ±11–13% and has already produced one retracted
"hardware crossover" that was noise. Five replicates before believing a 15%
effect there.
