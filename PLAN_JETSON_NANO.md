# Plan — Jetson Nano as a fourth verification host

> **SUPERSEDED, 2026-08-26. THE BRING-UP IS DONE.** The box is a working fourth
> verification host and has EXLA installed — the only box in the fleet that
> does. **`NEXT.md` §1.4 is the record; this file is kept only for its §0
> hardware survey**, which is still accurate and was not cheap to gather.
>
> Every `[ ]` below is stale, and four of the plan's predictions were wrong in
> ways worth keeping:
>
> | the plan said | what actually happened |
> |---|---|
> | "nothing has been installed or changed there yet" | everything in §1 is done |
> | J3 rustup `[ ]` open, "no `rustc`, no `~/.cargo`" | rustup 1.85.0 was **already installed** |
> | J4 OTP "should be mechanical", "hours-long" | ICE'd twice in `erts/emulator/asmjit`; needs **`--disable-jit`**, and took ~90 min. The plan never mentions the JIT, nor that the resulting OTP is JIT-less |
> | J5 "cmake is the most likely place to get stuck" | cmake was the easy half. **g++ 7.5 has no `<filesystem>`** — J5 is impossible without `g++-8`, plus `-DCMAKE_CXX_STANDARD_LIBRARIES=-lstdc++fs`. Neither is mentioned |
>
> A fifth, found later and not a prediction at all: the box runs `nvpmodel` at
> **5W with two of four cores parked**, which is the underlying reason for both
> the OTP ICE and the relaxed-LTO NIF build. It is why this box is for
> **correctness only, never timings**.
>
> J1 (owner sign-off) and J2 (real swap) remain genuinely open and both need a
> human. The hostname is still `jake-desktop` and `/home/jake` still exists.

Bring-up plan for `192.168.0.250` (Jetson Nano, arm64, Tegra X1) as a parity
host alongside super-io, mac-247 and mac-248. Written from a read-only survey of
the box on 2026-08-22, when nothing had yet been installed or changed there.

Each item states **why it is on the list**, **done when** (how you know it
worked), and the risk.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` deliberately
not doing.

---

## Verdict up front

**Viable, and worth doing — but it is a from-scratch toolchain build on a
2-core / 4 GB box, not a clone-and-run.**

The GPU side is a clean pass. The toolchain side is empty, and Ubuntu 18.04's
archives cannot supply any of the four missing pieces.

The one capability that could have killed it did not:

```
	shaderFloat64                           = 1
	shaderInt16                             = 1
	shaderInt64                             = 1
```

So this box would exercise the same f64 kernel set as the rest of the fleet, on
a third microarchitecture (Maxwell GM20B) and the **first non-x86 CPU**. That is
the axis that caught the `sqrt` 3-ULP Kepler bug.

---

## §0 — What the box actually is

Surveyed read-only over `ssh -o BatchMode=yes io@192.168.0.250` (key auth
already in place, no password needed).

| | |
|---|---|
| model | `NVIDIA Jetson Nano Developer Kit` |
| hostname | **`jake-desktop`** — not a fleet-convention name, see §4 |
| OS | Ubuntu 18.04.6 LTS (bionic), glibc 2.27, OpenSSL 1.1.1 |
| kernel | `4.9.337-tegra`, `aarch64` |
| L4T | `R32 (release), REVISION: 7.6` = JetPack 4.6.6 (final Nano release) |
| CPU / RAM | 2× Cortex-A57, 3.9 GB shared with GPU |
| swap | 2× zram (`zram0`/`zram1`, 991 MB each) — **no real swap**, see J2 |
| disk | 98 GB free of 118 GB on `/dev/mmcblk0p1` |
| load | idle: `load average: 0.45, 0.36, 0.54`, 830 MB used |
| present | gcc/g++ 7.5.0, make 4.1, cmake 3.10.2, git 2.17.1, python3 3.9.18 |
| network | outbound HTTPS to github.com and static.rust-lang.org works |

### Vulkan — one device, real GPU, no software fallback

```
Vulkan Instance Version: 1.2.70
GPU0
	apiVersion     = 0x402083  (1.2.131)
	driverVersion  = 134332800 (0x801c180)
	vendorID       = 0x10de
	deviceID       = 0x92ba03d7
	deviceType     = INTEGRATED_GPU
	deviceName     = NVIDIA Tegra X1 (nvgpu)
```

There is **no llvmpipe/lavapipe ICD anywhere on the box**, so unlike super-io
there is no software rasteriser to accidentally measure against.

The ICD is **not** at `/usr/share/vulkan/icd.d/` — that path does not exist on
L4T. It is `/etc/vulkan/icd.d/nvidia_icd.json` →
`/usr/lib/aarch64-linux-gnu/tegra/nvidia_icd.json`, `library_path:
libGLX_nvidia.so.0` (present), `api_version: 1.2.131`. `libvulkan.so.1` →
`tegra/libvulkan.so.1.2.141`.

Enumeration works headless (`DISPLAY` unset; vulkaninfo skipped surfaces and
still found GPU0). One queue family: `queueFlags = GRAPHICS | COMPUTE |
TRANSFER | SPARSE`, `queueCount = 16`.

Limits — **both code assumptions hold**:

```
		maxComputeWorkGroupCount[0]             = 2147483647
		maxComputeWorkGroupCount[1]             = 65535
		maxComputeWorkGroupCount[2]             = 65535
		maxComputeWorkGroupInvocations          = 1536
		maxComputeSharedMemorySize              = 0xc000   (48 KiB)
		maxStorageBufferRange                   = 0xffffffff
```

65535 workgroups on y/z (exactly, no headroom — same everywhere) and 1536
invocations against the assumed 256.

`lib.rs:217` reads `physical.supported_features().shader_float64` and enables it
conditionally, so f64 lights up on its own with no gate change.

### What is missing

| needed | present? |
|---|---|
| Elixir (`~> 1.17` per mix.exs; fleet runs 1.18.4) | **no** — `command not found` |
| Erlang/OTP (fleet runs 27) | **no** — `command not found` |
| Rust 1.85.0 (pinned in `rust-toolchain.toml`) | **no** — no `rustc`, no `~/.cargo`, no `~/.rustup` |
| `glslangValidator` | **no** |
| repo checkout | **no** — no `~/nx_vulkan`; stock Ubuntu desktop `$HOME` |
| C toolchain | yes |

bionic cannot help. `apt-cache policy` gives `erlang: 1:20.2.2` (22.0.7 in
backports) and `elixir: 1.3.3`, and **`glslang-tools`, `vulkan-tools` and
`spirv-tools` return nothing at all** — those packages do not exist in bionic.
Every missing tool comes from rustup / asdf / source.

---

## §1 — Bring-up steps

### J1 — Confirm the box is ours to take, and get sudo `[ ]`

**Why.** The hostname is `jake-desktop` and `$HOME` is a stock Ubuntu desktop
profile — this looks like someone's existing machine, not a spare. Separately,
`sudo -n tegrastats` returns `sudo: a password is required`, so nothing
automated can collect telemetry.

**Done when.** Owner has agreed the box can be dedicated to fleet use, and
either a sudoers entry for `tegrastats` exists or we have accepted that
telemetry is manual.

**Risk.** Low technically, but this gates everything below — do not start
installing on someone else's desktop.

### J2 — Give it real swap `[ ]`

**Why.** 3.9 GB shared with the GPU, and the only swap is two zram devices.
**zram compresses into the same RAM — it adds no genuine headroom.** The NIF
release profile is `lto = "fat"`, `codegen-units = 1`, `opt-level = 3`; linking
vulkano + rustler under that on 4 GB is a plausible OOM. Disk is not a
constraint (98 GB free).

**Done when.** A swapfile of at least 4 GB (8 GB preferred) is active and
persists across reboot; `swapon --show` lists it alongside the zram devices.

**Risk.** SD-card swap is slow and writes wear the card. Accept it — this is a
correctness host, not a timing host (§2).

### J3 — rustup, pinned to 1.85.0 `[ ]`

**Why.** `rust-toolchain.toml` pins `channel = "1.85.0"` (rustler 0.36/0.37 hit
a rustler-sys borrow signature mismatch on 1.90).

**Done when.** `rustc --version` reports 1.85.0 for
`aarch64-unknown-linux-gnu`.

**Risk.** Low. Tier-1 target, glibc 2.27 is well above the floor. `profile =
"minimal"` keeps the download small.

### J4 — Elixir 1.18.4 / OTP 27 from source `[ ]`

**Why.** Match the rest of the fleet exactly; bionic's packages (OTP 20, Elixir
1.3.3) are unusable.

**Done when.** `elixir --version` reports 1.18.4 on OTP 27, matching super-io
and the Macs.

**Risk.** Slow — compiling OTP on 2× A57 is an hours-long job. OpenSSL 1.1.1 and
the needed `-dev` packages are available, so it should be mechanical. Run it
under `nohup`/tmux; do not babysit an SSH session for hours.

### J5 — Build `glslangValidator` from source `[ ]`

**Why — this is the sharp one.** No bionic package, and Khronos ships **no
aarch64 release binaries**, so it must be compiled. And it is not merely a
build-time tool: `lib/nx_vulkan/synthesis.ex:55` and
`lib/nx_vulkan/codegen.ex:571` shell out to `glslangValidator` **at dispatch
time**, so the synthesis/JIT paths need it on `PATH` at runtime.

The 78 static `.comp` shaders have their 78 `.spv` committed to git, so the
non-JIT suite would survive without it — but a partial run is a much weaker
verification target and would not be comparable to a full fleet run.

**Done when.** `glslangValidator -V` compiles one of the repo's `.comp` files to
a `.spv` that is byte-identical to the committed one, and `synthesis_test.exs`
passes.

**Risk.** Local cmake is 3.10.2; recent glslang wants ≥ 3.17.2. Either check out
an older glslang tag that builds under 3.10, or install a newer cmake first.
Budget real time for this step — it is the most likely place to get stuck.

### J6 — Clone, build the NIF, run the suite `[ ]`

**Why.** The actual goal. Nothing is checked out on the box today.

**Done when.** Full `mix test` result recorded, and the box added to the fleet
table in `NEXT.md` (§1.4) with its commit SHA, the way mac-247/248 are tracked.

**Risk.** First arm64 build of the NIF — expect the unexpected in vulkano/ash
codegen on aarch64. If J2's swap proves insufficient, fall back to relaxing
`lto`/`codegen-units` **for this host only**, and say so in the record: a NIF
built with a different profile is still valid for parity, but note it.

---

## §2 — What this buys, and what it does not

**Buys:** first arm64 host, first Tegra/integrated GPU, third microarchitecture,
full f64 coverage. Genuine new failure surface of exactly the kind that has paid
off before.

**Does not buy — do not use this box for timings.** Tegra X1 is Maxwell with
**FP64 at 1/32 rate**, on 2 Cortex-A57 cores with SD-card I/O. `MISSION.md`
§ fleet table already treats mac-248 as untrustworthy for timing at ±11–13%;
this box will be worse and far slower. Use it for **parity/correctness only**
and keep the timing story on super-io (Ampere) and mac-247 (the good timing
host).

---

## §3 — Notes for whoever does this

- `vulkaninfo` on the box is Ubuntu's ancient `vulkan-utils 1.1.70`, which is
  why the instance line reads `1.2.70` while the ICD advertises `1.2.131`. The
  loader actually in use is the Tegra `libvulkan.so.1.2.141`, so vulkano 0.34's
  1.1+ entry points are there. But that binary is too old to print
  `VkPhysicalDeviceVulkan12Features` — if richer capability output is ever
  needed, query it from the NIF instead of trusting `vulkaninfo`.
- The ICD path differs from every other Linux box in the fleet
  (`/etc/vulkan/icd.d/`, not `/usr/share/vulkan/icd.d/`). Any tooling or doc
  that hardcodes the `/usr/share` path will need a branch.
- L4T 32.7.6 is the **last** Nano release; NVIDIA ships nothing further. The
  driver is frozen at `1.2.131` forever, which is arguably a feature for a
  verification host — it will not drift underneath us.

## §4 — Open question — J1 answered, and it needs a human decision

**Surveyed 2026-08-23.** `fleet-hosts.md` does **not** list `192.168.0.250`; the
recorded fleet is `.247`, `.248`, super-io, and nas at `.33`. So this box has
never been a fleet host.

It is also actively shared. There is a `/home/jake`, and `last` shows `jake` on
the **physical console** the same day:

```
io       pts/2        192.168.0.249    Sun Aug 23 18:58   still logged in
jake     pts/2        127.0.0.1        Sun Aug 23 18:34 - 18:37  (00:02)
jake     :0           :0               Sun Aug 23 18:07 - 18:37  (00:30)
```

This is not only etiquette. J4 pins both cores for hours compiling OTP, and with
J2 skipped there is no real swap — so a bring-up run makes jake's desktop
unusable while it lasts.

**Constraints adopted in consequence** (until someone clears the box for fleet
use):

- Everything installs under `~io` only — `rm -rf ~/.asdf ~/.cargo ~/.rustup
  ~/glslang ~/nx_vulkan` reverts it completely, leaving no trace for jake.
- **J2 is not done.** It is the only step changing global system state, and it
  needs root. Use the documented fallback instead: relax the release profile via
  `CARGO_PROFILE_RELEASE_LTO=false` and `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=4`,
  which overrides `Cargo.toml` with no file edit.
- Every build runs under `nice`, `-j1` when jake is at the console.
- No `sudo` anywhere. Note `sudo pip3 install --user` is doubly wrong: `pip3`
  does not exist (use `python3 -m pip`, pip 23.0.1 is present for python3.9),
  and `sudo` + `--user` installs into root's home, defeating the containment
  above.

**Worth weighing:** the Nano's value is its GPU (`shaderFloat64 = 1` on a real
Tegra X1), and that value does not require it to also be the machine that
compiles the toolchain. Cross-building or copying a toolchain in is a legitimate
alternative to hours of on-box compilation on someone else's desktop.

### Toolchain versions to match

super-io runs **glslang 15.1.0** (`Glslang Version: 11:15.1.0`, SPIR-V 0x00010600),
so J5 should build tag `15.1.0` for a meaningful `.spv` comparison. That version
requires cmake >= 3.17.2, against the box's system cmake 3.10.2 — hence the
no-sudo `python3 -m pip install --user cmake` into `~/.local`.
