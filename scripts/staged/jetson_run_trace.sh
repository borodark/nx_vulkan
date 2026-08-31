#!/bin/bash
export ASDF_DIR=$HOME/.asdf
source $HOME/.asdf/asdf.sh
source $HOME/.cargo/env
export PATH=$HOME/.local/bin:$PATH
export ERL_CRASH_DUMP_SECONDS=0
cd $HOME/nx_vulkan || exit 1

# refuse to measure if the box is not quiet
for i in $(seq 1 6); do
  if pgrep -f "mix test" >/dev/null 2>&1 || pgrep -f "rustc --crate-name" >/dev/null 2>&1; then
    echo "ABORT: box became busy before measuring"; exit 9
  fi
  sleep 2
done

echo "=== commit: $(git log -1 --format=%H) ==="
echo "=== uptime before ==="; uptime
echo "=== free before ==="; free -m | head -2

# devfreq sampler at 250ms, builtin reads, epoch-stamped
( while true; do
    read -r f < /sys/class/devfreq/57000000.gpu/cur_freq
    read -r l < /sys/devices/gpu.0/load
    printf "%s clk=%s load=%s\n" "$(date +%s.%N)" "$f" "$l"
    sleep 0.25
  done ) > /tmp/clkT.log 2>&1 &
SAMP=$!

echo "=== RUN START $(date) ==="
# stamp every output line so Race 1c's per-F windows can be isolated
mix run examples/unified_vs_discrete_race.exs 2>&1 \
  | while IFS= read -r line; do printf '%s %s\n' "$(date +%s.%N)" "$line"; done > /tmp/traceT.txt
echo "=== RUN END $(date) ==="
kill $SAMP 2>/dev/null

echo "=== uptime after ==="; uptime
echo "=== contention check after ==="
pgrep -f "mix test" >/dev/null 2>&1 && echo "WARNING: exmc mix test was running at end" || echo "clean: no exmc test at end"
echo "DONE_MARKER"
