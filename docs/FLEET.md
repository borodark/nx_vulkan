# The fleet — four boxes, and why each one is there

**Scope:** the verification hardware. Every correctness claim this project makes
is green on all four; every performance heuristic is raced on all four before it
ships. The fleet is not a nice-to-have — it is the reason several defaults in
this repo are the opposite of what a single machine would have chosen.

| host | GPU | arch | year | OS | CPU |
|---|---|---|---|---|---|
| super-io | GeForce RTX 3060 Ti | Ampere | 2021 | Linux | x86_64 |
| mac-248 | GeForce GT 750M | Kepler | 2013 | FreeBSD | x86_64 |
| mac-247 | GeForce GT 650M | Kepler | 2012 | FreeBSD | x86_64 |
| jake-desktop | Tegra X1 (Jetson Nano) | Maxwell | 2015 | Ubuntu 18.04 | aarch64 |

**833 doctests, 903 tests, 0 failures** on every row. Same source, same shaders,
same SPIR-V.

---

## What each box is for

**super-io** is the fast one and the only box that can run EXLA, so it is where
every "are we actually competitive" question gets answered honestly. It is also
where the flattering numbers come from, which is exactly why nothing ships on
its evidence alone.

**mac-247 and mac-248** are 2012 and 2013 laptop GPUs running FreeBSD, where
neither CUDA nor Metal has ever existed. They are the *only* path case — for
these machines this backend is not the second-best option, it is the option.
They also serve as the weak-GPU class: the many-slot fused reduce wins ~4.4× here
and **regresses ~0.44× on Ampere**, which is why that optimisation is
device-class-gated rather than global.

**jake-desktop** is a Jetson Nano, and it earns its slot by being different from
the other three in a way that keeps finding bugs.

## Why the Jetson matters

It is the *only* board in the fleet with **unified memory**. On Tegra, every
Vulkan memory type is `DEVICE_LOCAL` — the host-visible-but-not-device-local
type that a discrete-GPU allocator is built to navigate simply does not exist.
That makes it a natural control arm: any code path that only matters because
there is a PCIe bus is a no-op there, and the difference shows up immediately.

Three things it caught that a discrete-only fleet could not have:

- **It disproved an optimisation by winning least.** Removing a redundant
  zero-fill from every output buffer gave Ampere 635× and the Jetson only 37×.
  The unified board gained least because writing a constant into shared LPDDR4
  was already cheap — the advantage of unified memory was never fast memory, it
  is *no bus*, so an optimisation that removes a bus cost has the least to
  remove there.
- **It explained why four failed experiments had looked good.** The Jetson was
  flattering a series of memory designs for one reason: it was the only machine
  not paying the staging tax the others paid. Once that was understood, the
  actual defect — output buffers living in system RAM behind a
  `PREFER_DEVICE | HOST_RANDOM_ACCESS` filter, with every shader store crossing
  PCIe at a measured 10.8 GB/s — became visible on the boxes that did pay it.
- **It is the second CPU architecture.** aarch64 means byte-identity claims are
  not an x86 coincidence, and it means the shader pipeline is exercised against
  a locally built `glslangValidator` on a different toolchain.

It is also, pointedly, hardware its vendor walked away from: CUDA support for
this board stopped at 10.2. Its Vulkan driver did not stop. That is the whole
thesis of this project compressed into one 5-watt board — see
[`SECOND_CHANCES_THESIS.md`](https://github.com/borodark/nx_vulkan/blob/main/docs/SECOND_CHANCES_THESIS.md).

**Practical notes.** The board has no sudo, 4 GB of DRAM shared between CPU and
GPU, an OTP built `--disable-jit` (the JIT build ICEs in `asmjit`), and a NIF
built with relaxed LTO for the same reason. A full `rm -rf _build` costs ~47
minutes there because the dev and test trees do not share a cargo target dir and
the Rust crate compiles twice. Cross-compiling the NIF on a Linux box instead
takes under two minutes — the recipe, including the glibc 2.27 ABI trap that
makes a modern host's artifact fail to load in a way that looks like a stale
build, is in the
[`jetson-cross-build`](https://github.com/borodark/nx_vulkan/blob/main/.claude/skills/jetson-cross-build/SKILL.md) skill.

---

## The rule the fleet exists to enforce

**Win/loss crossovers are hardware-specific.** Three optimisations in this repo
were built, raced, and given three different fates purely on fleet evidence:

| optimisation | outcome | ships as |
|---|---|---|
| batched command submission | 1.45–1.71×, wins on **all** hosts, no crossover | on by default, no gate |
| many-slot fused reduce | 4.4× on Kepler, **0.44×** on Ampere | gated on `Nx.Vulkan.Device.class/0` |
| cross-stage CSE | never wins on either class | default-off, `NXV_CSE=1` |
| 32×32 register-blocked GEMM | wins on Ampere, regresses **both** Keplers | benchmark-only variant |

A single-box benchmark would have shipped two of those four as global defaults,
and two of the four would have been wrong for most of the hardware this backend
exists to serve.

Correctness survives contention; timings do not. The boxes host other work, so
any timing run samples load throughout and reports the samples — a race on a
noisy host does not merely fail to resolve a small effect, it can manufacture a
large one (a real 1.3% was once measured as 17%).
