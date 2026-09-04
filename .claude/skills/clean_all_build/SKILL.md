---
name: clean_all_build
description: Fully rebuild an Elixir project in this tree from scratch — _build, deps, Rust NIFs, GLSL shaders and the GPU shader/pipeline caches. Use when a failure smells like a stale artifact rather than a bug: a NIF UndefinedFunctionError, a :bad_lib on_load warning, a loaded version disagreeing with mix.lock, shader changes that do not take effect, or "suddenly every test fails". Applies to nx_vulkan and to the eXMC repos that depend on it.
---

# Clean rebuild (nx_vulkan and its consumers)

Use this when a fresh start is cheaper than the diagnosis. It usually is:
`_build/` regenerates from source and the lockfile, so nothing is lost, and a
stale artifact can burn hours looking like a real bug.

**One rule, read it before running anything:**

> ## Recompile `priv/shaders/*.spv` in place. Never clean-and-rebuild them.

As of `ac509d2` the tree holds a clean invariant — **59 `.comp` ↔ 59 `.spv`,
every blob regenerable from a source in `glsl/`** — so deleting them is no
longer *unrecoverable*. It is still the wrong move, and the script still
refuses to do it, because the invariant is worth checking rather than
assuming: if a `.spv` ever loses its source again, you want the script to
**tell you** rather than silently delete the last copy.

It happened once already. Until `ac509d2` there were 54 `.spv` and 52 `.comp`,
and **seven** blobs had no source in the tree — six of them in-use GPU kernels.
The sources were not lost, but the only copies were in `~/spirit/shaders/` on
the two FreeBSD Keplers, outside any repository, on machines nobody thought of
as holding source. `MISSION.md` §3.3.7 had the count wrong too, at three.

Everything else in the list below is safe to delete freely.

## 1. When to reach for this

Symptoms that mean "stale artifact", not "bug":

- `UndefinedFunctionError` for a NIF function that plainly exists in the source
- `The on_load function for module X returned: {:error, {:bad_lib, ...}}`
- a loaded dependency version that disagrees with `mix.lock`
- **every** test in a repo fails, including ones unrelated to your change
- a `.comp` edit that has no effect (shaders are **not** rebuilt by `mix compile`)
- `mix test` and `mix run` behaving differently for no reason you can name —
  the `test` env goes stale *independently* of `dev`

Real case, 2026-08-16: 20 integration failures in `_exmc-things/exmc` that
looked like anything but a build artifact. `_build/test/lib/nx_vulkan` was
version **0.1.0** while `mix.lock` pinned `7067499`; its NIF was missing
`device_supports_f64/0` and `leapfrog_chain_synth_f64/6`. The `dev` env was
fine the whole time. Suspect `_build/test/lib/` **first**.

## 2. What is safe to delete, and what is not

| path | safe? | why |
|---|---|---|
| `_build/` | **yes, freely** | regenerated from source + lockfile |
| `deps/` | yes | re-fetched from `mix.lock`; costs network + a long Rust build |
| `priv/shader_cache/` | yes | gitignored; JIT-fused kernels, regenerated on demand |
| `~/.exmc/gpu_node/spv/` | yes | synthesised chain shaders, **keyed by a hash of the generated GLSL**, so it already self-invalidates |
| `~/.exmc/gpu_node/pipeline_cache` | yes | Vulkan pipeline cache; a cold start is slower, nothing else |
| `native/*/target/` | yes | cargo output; rustler also builds into `_build` |
| **`priv/shaders/*.spv`** | **no** | recompiled in place; the script reports any that lost their source rather than removing it |

### If the script reports `orphan (kept)`

That means a `.spv` has no `glsl/*.comp` and the 59↔59 invariant has broken.
**Do not delete it and do not shrug.** Before `ac509d2` the answer was
`~/spirit/shaders/` on mac-247 / mac-248 — check there first:

```sh
ssh 192.168.0.247 'ls ~/spirit/shaders/*.comp'
```

Then prove any candidate is the real source rather than a lookalike, by
compiling it and comparing bytes to the committed blob:

```sh
glslangValidator -V recovered.comp -o /tmp/check.spv && cmp /tmp/check.spv priv/shaders/<name>.spv
```

A byte match is proof. That is how all seven were verified in `ac509d2`.

Never conclude a shader is unused just because grep finds no reference:
`Nx.Vulkan.shader_path/1` (`lib/nx_vulkan.ex:37`) resolves a shader by **name at
runtime**.

## 3. Run it

```sh
sh .claude/skills/clean_all_build/clean_all_build.sh            # this repo, dev+test
sh .claude/skills/clean_all_build/clean_all_build.sh --deps     # also re-fetch deps/
sh .claude/skills/clean_all_build/clean_all_build.sh --shaders-only
sh .claude/skills/clean_all_build/clean_all_build.sh --dry-run
```

From a consumer repo, point it at itself — it detects whether `glsl/` exists and
skips the shader stage if not:

```sh
cd ~/projects/learn_erl/_exmc-things/exmc
sh ~/projects/learn_erl/nx_vulkan/.claude/skills/clean_all_build/clean_all_build.sh
```

The script is deliberately not clever. Read it before running it in a tree you
care about; it is 60 lines.

## 4. What it does, in order

1. **Guard.** Refuses to run outside a directory containing `mix.exs`. Prints
   the tree, the git sha, and what it is about to delete.
2. **Caches.** `_build/`, `priv/shader_cache/`, `native/*/target/`, and the
   `~/.exmc/gpu_node/` caches. `deps/` only with `--deps`.
3. **Shaders.** For each `glsl/*.comp`, runs
   `glslangValidator -V glsl/<n>.comp -o priv/shaders/<n>.spv`. **Recompiles in
   place; deletes nothing.** Reports each as `ok` / `FAILED`, and reports any
   `.spv` with no source as `orphan (kept)`.
4. **Elixir + Rust.** `mix deps.get` (with `--deps`), then `mix compile` for
   **both** `dev` and `test`, because that is the whole point.
5. **Verify.** Prints `Nx.Vulkan.NativeV.device_name()` so you can confirm a
   real GPU and not llvmpipe, then `git status --short priv/shaders/` so any
   shader whose bytes changed is visible rather than silent.

Expect the Rust build to dominate — `nx_vulkan_vulkano` against vulkano 0.34 is
roughly two minutes on super-io, longer on the Keplers.

## 5. Afterwards

```sh
mix test                                # nx_vulkan: 833 doctests, 903 tests, 0 failures
sh scripts/strict_test.sh               # strict: 0 failures, 163 excluded
sh scripts/doctest_residency.sh         # residency: 755 / 833 (90.6%)
```

If `git status --short priv/shaders/` shows a modified `.spv` you did not
intend, that is a real signal: either a `.comp` changed under you, or the
installed `glslangValidator` emits different bytes than the one that produced
the committed blob. **Do not commit it reflexively** — check which.

## 6. The exception this cannot fix

In `_exmc-things/exmc`, `exla` is a CUDA build whose NIF cannot load
(`libnvshmem_host.so.3` is absent machine-wide) and it does **not** rebuild from
source either — `mix deps.compile exla` fails in `runtime_callback_cuda.o`
against the installed g++. Deleting `_build/` re-fetches the same broken thing,
and because `:exla` is a dependency application Mix starts it anyway, so **every**
test in that repo fails at startup.

To run that suite, move it aside and put it back:

```sh
mv _build/test/lib/exla /tmp/exla_save
mix test --no-deps-check <files>
mv /tmp/exla_save _build/test/lib/exla     # and VERIFY it is back
```

Verify the restore. Leaving it out silently changes what later runs measure,
which is its own class of bug. The real fixes — install a CPU EXLA, or stop the
test env requiring an optional dep to be startable — are the operator's call and
are recorded in that repo's `MISSION.md` §7.0.
