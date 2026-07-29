#!/bin/sh
# Trigger the f32-vs-f64 GPU race on this host and emit a labelled report.
#
# Run on any Vulkan box (GT 650M / 750M / RTX / llvmpipe):
#
#   sh scripts/race.sh
#
# Fetches deps, (re)builds the native NIF + shaders, runs the race, and writes
# bench_results/f32_race_<host>_<commit>.json (also printed as a table). Commit
# or paste that file to compare across hosts. The real GPU story lives on the
# f64-rate-limited devices — this is how you capture it.
set -e

cd "$(dirname "$0")/.."

echo "==> mix deps.get"
mix deps.get >/dev/null

echo "==> mix compile (native NIF + shaders)"
MIX_ENV=${MIX_ENV:-dev} mix compile

echo "==> running race"
mix run examples/f32_vs_f64_race.exs

echo "==> done. Report(s):"
ls -1 bench_results/f32_race_*.json 2>/dev/null || echo "  (none written)"
