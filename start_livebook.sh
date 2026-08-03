#!/usr/bin/env bash
# Start Livebook locally with the nx_vulkan notebooks.
#
# Usage:
#   ./start_livebook.sh
#   ./start_livebook.sh 8888
#   ./start_livebook.sh --port 8888
#
# Opens http://localhost:8080 (or custom port) with livebooks/ available.
# Requires: mix (Elixir), livebook escript installed via:
#   mix escript.install hex livebook
#
# The notebooks install nx_vulkan as a path dependency on this checkout, so
# the first cell compiles the vulkano NIF (~2 min cold, cached afterwards)
# and every later cell runs against your working tree — edit lib/, re-run the
# setup cell, and the notebook picks the change up.

set -euo pipefail

PORT=8080
case "${1:-}" in
  --port) PORT="${2:-8080}" ;;
  "") ;;
  *) PORT="$1" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Install livebook escript if missing
if ! command -v livebook &>/dev/null && ! [ -f "$HOME/.mix/escripts/livebook" ]; then
  echo "Installing Livebook escript..."
  mix escript.install hex livebook --force
fi

LIVEBOOK_BIN="livebook"
if ! command -v livebook &>/dev/null; then
  LIVEBOOK_BIN="$HOME/.mix/escripts/livebook"
fi

# Local dev server: no auth token, bound to loopback so skipping the token
# does not expose the machine. Drop --ip to reach it from another host, but
# then leave the token enabled.
export LIVEBOOK_TOKEN_ENABLED=false

echo "Livebook on http://localhost:$PORT — notebooks in $SCRIPT_DIR/livebooks"

exec "$LIVEBOOK_BIN" server \
  --port "$PORT" \
  --ip 127.0.0.1 \
  --home "$SCRIPT_DIR" \
  "$SCRIPT_DIR/livebooks"
