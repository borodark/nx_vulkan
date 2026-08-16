#!/bin/sh
# strict_test.sh — run the suite with host fallbacks REFUSED.
#
#   sh scripts/strict_test.sh            # the ratchet
#   sh scripts/strict_test.sh --trace    # extra args pass through to mix test
#
# `config :nx_vulkan, host_fallback: :raise` makes the first host fallback that
# is not on Nx.Vulkan.Fallback's allowlist raise Nx.Vulkan.HostFallbackError.
# A fallback is bit-identical to the GPU path, so no assertion on values can
# see one; this is the only thing that fails a build for it.
#
# WHY RAISE AND NOT COUNT. Nx.Vulkan.Fallback's census is a lower bound: once a
# fallback strands a tensor on BinaryBackend, everything downstream computes
# there unrecorded, so the count names the visible edge of a cascade. Raising
# fires on the FIRST refused op, before the tensor leaves the device, so a red
# strict run names the cause.
#
# TWO EXCLUSIONS, both enumerated in the tree (`grep -rn` them):
#
#   :host_fallback_expected — the test's SUBJECT is the fallback path. It
#       asserts a host fallback returns the right answer, so refusing fallbacks
#       there is a category error. Covers the two fallback-parity modules and
#       `doctest Nx` (upstream's own examples, written in {:s, 32} across the
#       whole Nx API — an API-completeness suite, not a residency one).
#
#   :host_fallback_open — a REAL fallback nobody knew about until strict mode
#       found it, tracked and not waived. Every one of these is debt with a
#       plan item. Adding a tag is a visible line in a diff; that is the point.
#
# Neither tag skips anything in the normal `mix test` run.

set -eu

echo "==> mix test with host_fallback: :raise"
echo "    excluding :host_fallback_expected (fallback is the test's subject)"
echo "    excluding :host_fallback_open (tracked, open — see PLAN_AFTER_BACKWARD_PASS.md T12)"
echo

NXV_HOST_FALLBACK=raise exec mix test \
    --exclude host_fallback_expected \
    --exclude host_fallback_open \
    "$@"
