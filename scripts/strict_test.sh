#!/bin/sh
# strict_test.sh — run the suite with host fallbacks REFUSED.
#
#   sh scripts/strict_test.sh            # the ratchet
#   sh scripts/strict_test.sh --trace    # extra args pass through to mix test
#
# The ExUnit timeout defaults to 600000ms here rather than ExUnit's 60s; pass
# your own --timeout to override it. See the note above the invocation.
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
# TWO TAGS AND ONE REGISTER, all enumerated in the tree (`grep -rn` them):
#
#   :host_fallback_expected — the test's SUBJECT is the fallback path. It
#       asserts a host fallback returns the right answer, so refusing fallbacks
#       there is a category error. Covers the two fallback-parity modules and a
#       scatter of individual cases in cast/fft/pad/slice/gather/conv/select.
#
#   :host_fallback_open — a REAL fallback nobody knew about until strict mode
#       found it, tracked and not waived. Every one of these is debt with a
#       plan item. Adding a tag is a visible line in a diff; that is the point.
#
#   test/nx_doctest_register.exs — `doctest Nx` USED to be under the first tag,
#       843 doctests behind one line. It is not any more (W2). The register
#       names the 488 that still leave the GPU, one line per op with a reason,
#       and test_helper.exs applies it only when fallbacks are being refused.
#       The other 355 run here like everything else. `sh
#       scripts/doctest_residency.sh` prints the rate and checks the register
#       against reality in both directions.
#
# None of the three skips anything in the normal `mix test` run.

set -eu

echo "==> mix test with host_fallback: :raise"
echo "    excluding :host_fallback_expected (fallback is the test's subject)"
echo "    excluding :host_fallback_open (tracked, open — see PLAN_AFTER_BACKWARD_PASS.md T12)"
echo "    doctest Nx is IN, minus test/nx_doctest_register.exs (355 of 843 resident)"
echo

# Default the ExUnit timeout, and let the caller still override it.
#
# ExUnit's 60s default is not enough on the slower fleet boxes, and the test it
# trips is not even a GPU one: `NXV_FUSE_REDUCE=1 ... many-slot reduce` builds a
# {100_000, 128} tensor with `Nx.BinaryBackend.iota/3` in its SETUP, on the
# host. On the Jetson's two 5W A57 cores that alone exceeds a minute, so a bare
# `sh scripts/strict_test.sh` reported 1 failure that had nothing to do with
# strictness — an ExUnit.TimeoutError dressed up as a strict-mode result.
#
# That is the worst shape of false negative for this script: it is the ONE check
# that can see a mistagged fallback test, so a spurious red here trains people
# to discount it. Found on the 2026-08-26 fleet run.
case " $* " in
    *" --timeout "*) TIMEOUT_ARG="" ;;
    *)               TIMEOUT_ARG="--timeout 600000" ;;
esac

# shellcheck disable=SC2086
NXV_HOST_FALLBACK=raise exec mix test \
    --exclude host_fallback_expected \
    --exclude host_fallback_open \
    $TIMEOUT_ARG \
    "$@"
