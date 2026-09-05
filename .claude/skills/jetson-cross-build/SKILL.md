---
name: jetson-cross-build
description: Cross-compile this repo's Rust NIF for the Jetson (Tegra X1, aarch64, Ubuntu 18.04) in a container on super-io, instead of compiling natively on that 2-core board. Use when the Jetson needs a rebuilt libnx_vulkan_vulkano.so, when a native build there is too slow or the box is contended, or when checking whether an artifact is ABI-safe for it.
---

# Cross-building the nx_vulkan NIF for the Jetson

The Jetson (`jake-desktop`, `192.168.0.250`, user `io`) is a 2-core-online 5W
Tegra X1. Compiling the Rust NIF there is slow enough to be the main cost of any
verification round — a full `rm -rf _build` runs ~47 min, because the dev and
test `_build` trees do not share a cargo target dir and the crate compiles
twice. super-io does the same crate in **under two minutes**.

This is a real cross-compile: an **amd64** container running at native speed. No
qemu, no binfmt, no root on the host.

## Why bionic, and why that is the whole trick

The Jetson has **glibc 2.27 with zero margin**. A NIF built on a modern host
requires `GLIBC_2.34` (verified — super-io's own x86_64 artifact does), and such
a library fails to load there as a `:bad_lib` / `on_load` error that looks like
a stale artifact rather than an ABI mismatch. That misreading is the trap this
skill exists to prevent: reach for `clean_all_build` and you will rebuild for an
hour and still fail.

`ubuntu:18.04` is chosen for its glibc, not its age — 2.27, exactly the target's
— and its `gcc-aarch64-linux-gnu` cross toolchain targets the same. The
resulting artifact needs at most `GLIBC_2.25`.

`archive.ubuntu.com` still serves bionic. Do NOT rewrite sources to
`old-releases.ubuntu.com`; it 404s for bionic and only breaks the build.

## What the NIF actually links

Nothing exotic. `libdl`, `libgcc_s`, `libpthread`, `libm`, `libc`.

**Vulkan is not in `NEEDED`** — vulkano `dlopen`s the loader at runtime — so the
sysroot needs no Vulkan headers or libraries at all. The Jetson resolves it to
its own Tegra ICD (`/usr/lib/aarch64-linux-gnu/tegra/libvulkan.so.1.2.141`).
The crate deps (`rustler`, `vulkano`, `ahash`) are pure Rust; no `*-sys` crate
needs a target-side C library.

## Build

    cd .claude/skills/jetson-cross-build
    nerdctl build -t nxv-jetson-cross:1.85.0 .

    nerdctl run --rm \
      -v $PWD/native/nx_vulkan_vulkano:/src \
      -v /some/scratch/target-aarch64:/target \
      -v /some/scratch/cargo-registry:/opt/cargo/registry \
      -e CARGO_TARGET_DIR=/target \
      nxv-jetson-cross:1.85.0 \
      cargo build --release --locked --target aarch64-unknown-linux-gnu

Artifact lands at
`/target/aarch64-unknown-linux-gnu/release/libnx_vulkan_vulkano.so`.

**Always set `CARGO_TARGET_DIR` outside the repo.** Sharing
`native/nx_vulkan_vulkano/target` with the host's x86_64 build invites exactly
the stale-artifact confusion this repo has been bitten by before.

Rust is pinned to **1.85.0**, the Jetson's own toolchain version, so the only
variable between a cross build and a native one is the target triple.

Two gotchas that cost a build each:

* The image needs **host `gcc` as well as the cross gcc**. Build scripts
  (`libc`, `zerocopy`, `serde`, `ash`, ...) compile for the HOST; without
  `/usr/bin/cc` cargo dies with ``linker `cc` not found`` on a dozen crates and
  the error says nothing about cross-compiling.
* Verify aarch64 binaries with **`aarch64-linux-gnu-objdump`**, not the host's.
  Host binutils on super-io cannot disassemble aarch64 and fails with
  `can't disassemble for architecture UNKNOWN!` — piped into `grep -c` that
  reads as a clean `0`, which will tell you a binary is free of instructions you
  never actually looked for.

## Verifying an artifact before it goes near the box

    nerdctl run --rm -v /some/scratch/target-aarch64:/target nxv-jetson-cross:1.85.0 bash -c '
      SO=/target/aarch64-unknown-linux-gnu/release/libnx_vulkan_vulkano.so
      aarch64-linux-gnu-readelf -h "$SO" | grep -E "Class|Machine|Type"
      aarch64-linux-gnu-readelf -d "$SO" | grep NEEDED
      aarch64-linux-gnu-readelf -V "$SO" | grep -o "GLIBC_[0-9.]*" | sort -uV | tail -3'

Bars: `ELF64` / `AArch64` / `DYN`; max `GLIBC_2.27`; `NEEDED` confined to the
five libraries above.

`file` is NOT installed in the image — an earlier version of this snippet used
it and returned `file: command not found`, which is easy to skim past as noise
when the lines below it succeed. `readelf -h` is the check that actually runs.

Counting the outline-atomics helpers needs plain `nm`, not `nm -D`: they are
local symbols, so the dynamic table shows zero and a `-D` count will tell you
they are absent when there are 22 of them.

### The LSE atomics question, settled

The A57 is ARMv8.0 and has no LSE atomics, so "does this binary contain
`cas`/`swp`/`ldadd`?" looks like the right check. **It is the wrong bar**, and
insisting on zero will send you chasing a flag that cannot deliver it.

Every stock Rust `aarch64-unknown-linux-gnu` binary contains them, inside
`compiler_builtins`' outline-atomics helpers (`__aarch64_cas*`, `__aarch64_swp*`,
`__aarch64_ldadd*`), guarded at runtime by `__aarch64_have_lse_atomics`, which is
set from HWCAP at startup. The Jetson's `/proc/cpuinfo` has no `atomics` flag, so
the guard is 0 and the LDXR/STXR fallback executes. The LSE instructions are
never reached.

`-C target-feature=-outline-atomics` does **not** remove them: `compiler_builtins`
ships precompiled in rust-std, so only `-Z build-std` on nightly would rebuild
it. Not worth it.

The measurement that settles it — the Jetson's own working native build has
MORE of them than a cross build does:

    artifact                       outline syms   LSE insns   max GLIBC
    Jetson native (rustc 1.85.0)        19            20        2.25
    this container's cross build        13            12        2.25

So the correct bar is "no *unguarded* LSE", and both satisfy it. A cross-built
artifact is strictly more conservative than what that box already runs.

## Deploying

**VALIDATED end to end on 2026-08-31** at `d7b5f08`: built, verified, deployed,
loaded, and passed the box's full suite (833 doctests, 871 tests, 0 failures).
Artifact `dfed921a...`, 1m55s to build against ~47 min native.

The `.so` goes to `~/nx_vulkan/priv/native/libnx_vulkan_vulkano.so`. Back up the
existing one first — it is the known-good fallback.

**`priv` is a symlink** in both `_build/dev/lib/nx_vulkan` and
`_build/test/lib/nx_vulkan` (`-> ../../../../priv`), so overwriting that single
file covers every environment. Confirm the symlink rather than assuming it.

### The overwrite problem, and the actual solution

Rustler rebuilds the NIF on the next `mix compile` and silently replaces the
copy. A green suite that used a natively rebuilt `.so` proves nothing about the
cross artifact.

Two cases, and the second used to be a dead end.

**Rust-only commit — use `--no-compile`.** Sound whenever no Elixir source
changed since the box's last build:

    git diff --name-only <box_HEAD> <target_commit> | grep -E "^(lib|test)/"

Empty means `_build` is current and only the NIF differs.

**Mixed Elixir+Rust commit — use `NXV_SKIP_NIF_BUILD=1`.** The box needs a real
`mix compile` for the Elixir side, and that normally triggers Rustler, which
rebuilds the crate natively and overwrites the artifact you just shipped. This
section used to say the skill therefore "buys you nothing" for such commits.
That was wrong: `config/config.exs` now sets Rustler's `skip_compilation?` when
that variable is set, so the Elixir side compiles and `priv/native` is left
alone.

    NXV_SKIP_NIF_BUILD=1 mix compile        # Elixir only, .so untouched
    NXV_SKIP_NIF_BUILD=1 mix test

It prints a warning to stderr when active, because a stale or wrong-architecture
`.so` under this flag gives a green suite that says nothing about the code you
just compiled.

**The flag is sticky, and this bit me within the hour.** Rustler reads it via
`Application.compile_env`, so the value is baked into the compiled module and
Elixir refuses to boot when the runtime value differs:

    ** (Mix) the application :nx_vulkan has a different value set for key
       Nx.Vulkan.NativeV during runtime compared to compile time.
       Compile time value was set to: [skip_compilation?: true]
       Runtime value was not set

So set it for `mix compile` AND for every `mix run` / `mix test` afterwards —
which is why both lines above carry it. To return to ordinary builds, unset it
and force the NIF module to recompile: **a plain `mix compile` will not clear
it**, but deleting
`_build/<env>/lib/nx_vulkan/ebin/Elixir.Nx.Vulkan.NativeV.beam` will. Note the
shape — the same "a rebuild does not necessarily rebuild" problem as the
`.so` section below, one layer up.

Do not run plain `--no-compile` on a mixed commit to keep the fast path. Seven
changed Elixir sources against a stale `_build` will run the OLD tests against
the OLD lib and report green.

### `mix compile` does not necessarily refresh `priv/native`

Found 2026-09-02 and it applies to EVERY workflow here, not just this skill.
Replacing `priv/native/libnx_vulkan_vulkano.so` with 25 bytes of text and
running a plain `mix compile` — no flags, an Elixir source touched — left the
corrupted file in place and produced 1622 test failures. Cargo saw no change to
the Rust sources, reported the crate up to date, and Rustler never re-copied.

So **a swapped, stale or wrong-architecture `.so` is not fixed by recompiling.**
To force it, touch a Rust source or remove the crate's `target/` directory. This
is why the checksum discipline below is not optional and is not specific to the
skip flag: a benchmark `.so` swap left behind on another checkout will survive
an ordinary rebuild and silently misattribute every result after it.

**Checksum before AND after the run.** That is the only proof the artifact under
test is the one that executed:

    sha256sum priv/native/libnx_vulkan_vulkano.so   # before
    mix test --no-compile < /dev/null
    sha256sum priv/native/libnx_vulkan_vulkano.so   # must be unchanged

**COMPARE ONLY WITHIN ONE `MIX_ENV`.** The artifact embeds its own absolute
build path, so `_build/dev/...` and `_build/test/...` produce DIFFERENT bytes
from identical source. Measured on super-io 2026-09-05, same commit:

    mix compile                -> 71cf018235
    MIX_ENV=test mix compile   -> fd64164779     (identical source)
    mix compile                -> fd64164779     (dev did NOT take it back)

And because `priv` is a symlink from both build trees, the two environments
share ONE `.so` and the last writer keeps it — a `mix test` run can be
executing the dev-env build, or vice versa. Same code, so this is a
hash-reasoning hazard rather than a correctness one, but a cross-env comparison
will report "the artifact changed" when nothing did, and that reads exactly
like the swap this discipline exists to catch.

Cargo itself is reproducible here: three consecutive forced rebuilds of
identical source in one `MIX_ENV` gave one hash. A moving hash within a fixed
env is real; across envs it is expected.

### `mix` swallows heredoc stdin

`mix test` and `mix run` read stdin. Inside `ssh host 'bash -s'` with a heredoc
they consume the rest of the script, so trailing verification lines — including
the after-checksum — never execute, and their absence looks like a truncated
transcript rather than a bug. **Redirect: `< /dev/null`.**

### First check is the load, not the suite

    mix run --no-compile -e 'IO.inspect Nx.Vulkan.NativeV.device_name()' < /dev/null

An ABI mismatch fails here, in seconds, instead of somewhere inside an
85-second suite. Expect:

    [nx_vulkan_vulkano] device: NVIDIA Tegra X1 (nvgpu) (IntegratedGpu)
    [nx_vulkan_vulkano] unified memory: true (staging path: OFF)

That second line is also the assertion that the box took the `unified` branch of
`alloc_buffer` — worth reading, not skipping, since several code paths are
no-ops only on that branch.

Gate any deploy on the box's own correctness suite: **833 doctests, 931 tests,
0 failures** (871 before the 2026-09 property-test tier; 903 before the
allowlist-integrity and strict-mode-validation tests). Note the suite prints a `GenServer terminating ** (RuntimeError)
boom` trace from `node_test.exs` — that is an intentional test, not a failure.

## Jetson environment traps (they will bite the test run, not the build)

From the fleet notes, all paid for:

* Neither `mix` nor `cargo` is on `PATH` in a non-login shell. Source
  `~/.asdf/asdf.sh` AND `~/.cargo/env`.
* `~/.asdf/asdf.sh` cannot be sourced from `/bin/sh` (dash) — use bash, or
  export `ASDF_DIR=$HOME/.asdf` first.
* `~/.local/bin` must be on PATH; `glslangValidator` lives there, and without it
  `SynthesisTest` fails 6 tests on `:enoent` that look like real breakage.
* Set `ERL_CRASH_DUMP_SECONDS=0` — a crash there writes 2.8 GB dumps.
* **Check contention first, and again DURING.** The box hosts other work —
  exmc fleet verifications run there for ~110 min at a stretch and hold a core
  at ~90%. `uptime` at 2.28 on two online cores means someone else has it.
  Check with `pgrep -x beam.smp`, never `pgrep -f "beam|cargo"`, which matches
  the shell running it and produces a wait loop that never fires.
  Correctness survives contention; timings do not. A race run there voided at
  177.8% estimator divergence with a foreign job present, against 30.7% on a
  quiet box. If you must time, sample load every 10s for the whole run and
  report the samples, not just the endpoints.
