#!/bin/sh
# setup_freebsd_gpu_node.sh — Bootstrap a FreeBSD host for Zed GPU agent
#
# Run as your regular user (uses doas for pkg install):
#   sh setup_freebsd_gpu_node.sh
#
# Idempotent — safe to re-run. Soft-errors if packages already installed.
# Tested on FreeBSD 15.0-RELEASE with NVIDIA Kepler GPUs (GT 650M, GT 750M).

set -e

echo "=== Zed GPU Node Setup for FreeBSD ==="
echo ""

# ----------------------------------------------------------------
# 1. System packages
# ----------------------------------------------------------------

install_pkg() {
    if pkg info -e "$1" >/dev/null 2>&1; then
        echo "[ok] $1 already installed"
    else
        echo "[install] $1"
        doas pkg install -y "$1" || echo "[warn] failed to install $1 — continuing"
    fi
}

echo "--- System packages ---"
# NVIDIA driver (adjust version if your GPU needs a different branch)
install_pkg nvidia-driver-470
install_pkg nvidia-kmod-470

# Vulkan stack
install_pkg vulkan-loader
install_pkg vulkan-headers
install_pkg vulkan-tools

# Shader compiler
install_pkg glslang

# Erlang + Elixir (OTP 27 required for NIF ABI compatibility)
install_pkg erlang
install_pkg erlang-runtime27
install_pkg elixir

# Build Elixir 1.18 against OTP 27 if not already done
if [ ! -f /tmp/elixir-1.18.4/bin/elixir ]; then
    echo "[build] Building Elixir 1.18.4 against OTP 27..."
    cd /tmp
    wget -q https://github.com/elixir-lang/elixir/archive/refs/tags/v1.18.4.tar.gz 2>/dev/null
    tar xf v1.18.4.tar.gz 2>/dev/null
    cd elixir-1.18.4
    PATH=/usr/local/lib/erlang27/bin:$PATH gmake clean compile 2>/dev/null
    echo "[ok] Elixir 1.18.4 built at /tmp/elixir-1.18.4/bin/"
else
    echo "[ok] Elixir 1.18.4 already built"
fi

# Set PATH for OTP 27 + Elixir 1.18
export PATH=/tmp/elixir-1.18.4/bin:/usr/local/lib/erlang27/bin:$PATH

# Persist in shell profile
if ! grep -q "erlang27" ~/.profile 2>/dev/null; then
    echo 'export PATH=/tmp/elixir-1.18.4/bin:/usr/local/lib/erlang27/bin:$PATH' >> ~/.profile
    echo "[ok] PATH added to ~/.profile"
fi

# Rust (for Rustler NIF compilation)
install_pkg rust

# Build tools
install_pkg git
install_pkg gmake
install_pkg cmake
install_pkg pkgconf
install_pkg gcc13
install_pkg wget
install_pkg curl

# doas (should already be there, but just in case)
install_pkg doas

echo ""

# ----------------------------------------------------------------
# 2. NVIDIA kernel module
# ----------------------------------------------------------------

echo "--- NVIDIA kernel module ---"
if kldstat | grep -q nvidia; then
    echo "[ok] nvidia.ko already loaded"
else
    echo "[load] loading nvidia.ko"
    doas kldload nvidia || echo "[warn] nvidia.ko failed to load — may need reboot"
fi

# Ensure it loads at boot
if grep -q 'nvidia_load' /boot/loader.conf 2>/dev/null; then
    echo "[ok] nvidia_load already in /boot/loader.conf"
else
    echo "[config] adding nvidia_load to /boot/loader.conf"
    doas sh -c 'echo "nvidia_load=\"YES\"" >> /boot/loader.conf'
fi

echo ""

# ----------------------------------------------------------------
# 3. Verify Vulkan sees the GPU
# ----------------------------------------------------------------

echo "--- Vulkan device check ---"
if command -v vulkaninfo >/dev/null 2>&1; then
    DEVICE=$(vulkaninfo --summary 2>&1 | grep deviceName | head -1 | sed 's/.*= //')
    if [ -n "$DEVICE" ]; then
        echo "[ok] Vulkan device: $DEVICE"
    else
        echo "[warn] vulkaninfo ran but no device found — check nvidia driver"
    fi
else
    echo "[warn] vulkaninfo not found — vulkan-tools not installed?"
fi

echo ""

# ----------------------------------------------------------------
# 4. Clone repositories
# ----------------------------------------------------------------

GIT_HOST="192.168.0.33"
GIT_BASE="/mnt/jeff/home/git/repos"

clone_or_pull() {
    REPO=$1
    DIR=$2
    BRANCH=${3:-main}

    if [ -d "$DIR" ]; then
        echo "[ok] $DIR exists — pulling"
        cd "$DIR" && git checkout "$BRANCH" 2>/dev/null; git pull 2>/dev/null || true
        cd ~
    else
        echo "[clone] $REPO -> $DIR"
        git clone "git@${GIT_HOST}:${GIT_BASE}/${REPO}.git" "$DIR" || \
        git clone "${GIT_HOST}:${GIT_BASE}/${REPO}.git" "$DIR" || \
        echo "[warn] failed to clone $REPO — set up SSH keys to $GIT_HOST"
    fi
}

echo "--- Repositories ---"
clone_or_pull "spirit.git" "$HOME/spirit" "feature/vulkan-backend"
clone_or_pull "nx_vulkan.git" "$HOME/nx_vulkan" "main"
clone_or_pull "zed.git" "$HOME/zed" "feat/demo-cluster"

echo ""

# ----------------------------------------------------------------
# 5. Build nx_vulkan
# ----------------------------------------------------------------

echo "--- Build nx_vulkan ---"
if [ -d "$HOME/nx_vulkan" ]; then
    cd "$HOME/nx_vulkan"
    export NX_VULKAN_PATH="$HOME/nx_vulkan"

    echo "[deps] mix deps.get"
    mix deps.get 2>/dev/null || echo "[warn] deps.get failed"

    echo "[compile] mix compile"
    mix compile 2>&1 | tail -3

    echo "[test] mix test"
    mix test 2>&1 | tail -3
else
    echo "[skip] nx_vulkan not cloned"
fi

echo ""

# ----------------------------------------------------------------
# 6. Build Zed with GPU support
# ----------------------------------------------------------------

echo "--- Build Zed ---"
if [ -d "$HOME/zed" ]; then
    cd "$HOME/zed"
    export NX_VULKAN_PATH="$HOME/nx_vulkan"

    echo "[deps] mix deps.get"
    mix deps.get 2>/dev/null || echo "[warn] deps.get failed"

    echo "[compile] mix compile"
    mix compile 2>&1 | tail -3

    echo "[test] mix test (base)"
    mix test 2>&1 | tail -3
else
    echo "[skip] zed not cloned"
fi

echo ""

# ----------------------------------------------------------------
# 7. Smoke test GPU agent
# ----------------------------------------------------------------

echo "--- GPU Agent smoke test ---"
if [ -d "$HOME/zed" ] && [ -d "$HOME/nx_vulkan" ]; then
    cd "$HOME/zed"
    ZED_GPU=1 NX_VULKAN_PATH="$HOME/nx_vulkan" \
      mix run -e '
        {:ok, info} = Zed.GPU.info()
        IO.puts("GPU Agent: #{info.device} (f64=#{info.has_f64})")
        IO.puts("Setup complete!")
      ' 2>&1 | grep -E "GPU Agent:|Setup|ready"
else
    echo "[skip] repos not available"
fi

echo ""

# ----------------------------------------------------------------
# 8. Print summary
# ----------------------------------------------------------------

echo "=== Setup Complete ==="
echo ""
echo "To start the GPU node:"
echo ""
echo "  cd ~/zed"
echo "  ZED_GPU=1 NX_VULKAN_PATH=\$HOME/nx_vulkan \\"
echo "    iex --name gpu2@\$(hostname -I | awk '{print \$1}') \\"
echo "    --cookie zed_gpu_demo -S mix"
echo ""
echo "Then from mac-248:"
echo "  Node.connect(:\"gpu2@<this-ip>\")"
echo "  Zed.GPU.info(:\"gpu2@<this-ip>\")"
echo ""
