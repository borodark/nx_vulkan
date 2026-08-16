#!/bin/sh
# Full clean rebuild: _build, caches, Rust NIFs, GLSL shaders, dev + test envs.
#
# HARD RULE: never delete priv/shaders/*.spv. Seven of the 54 blobs in
# nx_vulkan have no .comp source and cannot be regenerated. Shaders are
# RECOMPILED IN PLACE from whatever sources exist; nothing is removed.
#
# Usage: clean_all_build.sh [--deps] [--shaders-only] [--dry-run]

set -eu

DEPS=0
SHADERS_ONLY=0
DRY=0

for arg in "$@"; do
  case "$arg" in
    --deps)         DEPS=1 ;;
    --shaders-only) SHADERS_ONLY=1 ;;
    --dry-run)      DRY=1 ;;
    -h|--help)      sed -n '2,9p' "$0"; exit 0 ;;
    *)              echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

[ -f mix.exs ] || { echo "ERROR: no mix.exs here. cd to the project root first." >&2; exit 1; }

run() { if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else eval "$@"; fi }

echo "=== clean_all_build ==="
echo "tree : $(pwd)"
echo "git  : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "mode : deps=$DEPS shaders_only=$SHADERS_ONLY dry_run=$DRY"
echo

# --- 1. caches -------------------------------------------------------------
if [ "$SHADERS_ONLY" = 0 ]; then
  echo "--- removing build artifacts and caches ---"
  run "rm -rf _build"
  run "rm -rf priv/shader_cache"
  for t in native/*/target; do [ -d "$t" ] && run "rm -rf '$t'"; done
  # Synthesised chain shaders are hash-keyed so they self-invalidate, but a
  # cold cache is the point of a clean rebuild.
  [ -d "$HOME/.exmc/gpu_node/spv" ] && run "rm -rf '$HOME/.exmc/gpu_node/spv'"
  [ -d "$HOME/.exmc/gpu_node/pipeline_cache" ] && run "rm -rf '$HOME/.exmc/gpu_node/pipeline_cache'"
  [ "$DEPS" = 1 ] && run "rm -rf deps"
  echo
fi

# --- 2. shaders ------------------------------------------------------------
if [ -d glsl ]; then
  echo "--- recompiling shaders (in place; nothing deleted) ---"
  if ! command -v glslangValidator >/dev/null 2>&1; then
    echo "  ERROR: glslangValidator not on PATH — install the Vulkan SDK." >&2
    exit 1
  fi
  mkdir -p priv/shaders
  ok=0; failed=0
  for comp in glsl/*.comp; do
    [ -e "$comp" ] || continue
    name=$(basename "$comp" .comp)
    if [ "$DRY" = 1 ]; then
      echo "  [dry-run] glslangValidator -V $comp -o priv/shaders/$name.spv"
    elif out=$(glslangValidator -V "$comp" -o "priv/shaders/$name.spv" 2>&1); then
      ok=$((ok + 1))
    else
      failed=$((failed + 1))
      echo "  FAILED $name"
      echo "$out" | sed 's/^/      /'
    fi
  done
  echo "  compiled: $ok ok, $failed failed"

  # Invariant: every .spv has a .comp. Report, NEVER remove — a .spv whose
  # source vanished may be the last copy. Seven were in exactly that state
  # until ac509d2; the sources turned out to be in ~/spirit/shaders/ on the
  # Keplers, outside any repo. See SKILL.md.
  orphans=0
  for spv in priv/shaders/*.spv; do
    [ -e "$spv" ] || continue
    name=$(basename "$spv" .spv)
    if [ ! -f "glsl/$name.comp" ]; then
      echo "  orphan (kept, no .comp source): $name"
      orphans=$((orphans + 1))
    fi
  done
  if [ "$orphans" -gt 0 ]; then
    echo "  WARNING: $orphans .spv have no source. The 59<->59 invariant is broken."
    echo "           Try: ssh 192.168.0.247 'ls ~/spirit/shaders/*.comp'"
    echo "           Verify a candidate by compiling it and cmp-ing against the blob."
  else
    echo "  invariant ok: every .spv has a .comp source"
  fi
  echo
  [ "$failed" -gt 0 ] && { echo "ERROR: $failed shader(s) failed to compile." >&2; exit 1; }
else
  echo "--- no glsl/ dir; skipping shader stage ---"
  echo
fi

[ "$SHADERS_ONLY" = 1 ] && { echo "done (shaders only)."; exit 0; }

# --- 3. elixir + rust, BOTH envs ------------------------------------------
echo "--- compiling (this rebuilds the Rust NIFs; expect minutes) ---"
[ "$DEPS" = 1 ] && run "mix deps.get"
echo "  [dev]"
run "mix compile"
echo "  [test]  <- the env that goes stale unnoticed"
run "env MIX_ENV=test mix compile"
echo

# --- 4. verify -------------------------------------------------------------
echo "--- verify ---"
if [ "$DRY" = 0 ] && grep -q 'defmodule Nx.Vulkan' lib/nx_vulkan.ex 2>/dev/null; then
  mix run -e 'IO.puts("  device: #{inspect(Nx.Vulkan.NativeV.device_name())}")' 2>/dev/null \
    || echo "  device: could not query (see output above)"
fi

if [ -d priv/shaders ] && [ "$DRY" = 0 ]; then
  changed=$(git status --short priv/shaders/ 2>/dev/null || true)
  if [ -n "$changed" ]; then
    echo "  shader bytes CHANGED — check why before committing:"
    echo "$changed" | sed 's/^/    /'
  else
    echo "  shaders: byte-identical to git"
  fi
fi

echo
echo "done. Next: mix test   and   NXV_HOST_FALLBACK=raise mix test"
