#!/bin/sh
# build_and_test.sh — clean build of nx_vulkan + copy shaders + test
# Run from the nx_vulkan project root:
#   sh scripts/build_and_test.sh

set -eu

SPIRIT_SHADERS="/home/io/spirit/shaders"
PRIV_SHADERS="priv/shaders"

echo "==> Recompiling all Spirit shaders"
for f in "$SPIRIT_SHADERS"/*.comp; do
    spv="${f%.comp}.spv"
    echo "  $(basename "$f")"
    glslangValidator -V "$f" -o "$spv"
done

echo "==> Copying .spv files to $PRIV_SHADERS"
mkdir -p "$PRIV_SHADERS"
cp "$SPIRIT_SHADERS"/*.spv "$PRIV_SHADERS/"
ls "$PRIV_SHADERS"/*.spv | xargs -I{} basename {}

echo "==> Clean build"
mix deps.get --only test 2>/dev/null
rm -rf _build/test/lib/nx_vulkan
MIX_ENV=test mix compile --force

echo "==> Running tests"
mix test
