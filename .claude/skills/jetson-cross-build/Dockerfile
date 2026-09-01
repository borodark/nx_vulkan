# Cross-build the nx_vulkan Rust NIF for the Jetson (Tegra X1, aarch64).
#
# bionic is chosen for its glibc, not its age: 2.27 is EXACTLY what the Jetson
# has, with zero margin. Building on a modern host instead produces a .so
# needing GLIBC_2.34 (verified: the x86_64 artifact already does), which fails
# to load there and presents as a :bad_lib / on_load error rather than an ABI
# error. The image is amd64 and CROSS-compiles; no qemu, no binfmt, no root.
FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
      gcc libc6-dev \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
# Pin to the Jetson's own toolchain version so the only variable is the target.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.85.0 --profile minimal \
                    --target aarch64-unknown-linux-gnu

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
    CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
    AR_aarch64_unknown_linux_gnu=aarch64-linux-gnu-ar
# The A57 is ARMv8.0 and has NO LSE atomics; never let a newer target-cpu leak in.
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C target-cpu=cortex-a57"
WORKDIR /src
