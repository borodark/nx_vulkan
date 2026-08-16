#!/bin/sh
# doctest_residency.sh — what share of Nx's own doctests runs entirely on the GPU?
#
#   sh scripts/doctest_residency.sh
#
# Prints one number and defends it. That number is the acceptance test for the
# whole coverage effort: a host fallback returns a BIT-IDENTICAL result — it *is*
# Nx.BinaryBackend, the reference every doctest compares against — so no
# assertion on values can see one. Residency has to be measured separately or it
# is not measured at all, and "not measured" is how this project's two most
# expensive bugs survived (docs/BACKWARD_PASS_AUDIT.md).
#
# HOW IT WORKS. Two passes over test/nx_vulkan/nx_doctest_test.exs, both with
# host fallbacks refused:
#
#   pass A  register applied — the 524 doctests named in
#           test/nx_doctest_register.exs are excluded. Everything else must stay
#           on the GPU. A failure here is a RESIDENCY REGRESSION: an op that was
#           resident has started falling back.
#
#   pass B  NXV_DOCTEST_REGISTER=off — nothing excluded, so the failures are
#           exactly the doctests that fall back today. That count is the truth
#           the register is checked against.
#
# Pass A's excluded count and pass B's failure count must be EQUAL. Higher
# excluded means the register names doctests that no longer fall back — stale
# entries, quietly understating the rate. This is what makes the number monotone
# by policy rather than by hope: it can only move when someone edits the
# register on purpose.
#
# THE ORDINALS ARE FRAGILE, ON PURPOSE. The register keys on ExUnit's doctest
# ordinals, which renumber if nx_doctest_test.exs's :except buckets change or
# the `nx` dep is bumped. That breaks this script loudly and prints the correct
# list, so the repair is a paste. See the register's moduledoc.

set -eu

FILE=test/nx_vulkan/nx_doctest_test.exs
OUT="${TMPDIR:-/tmp}/nxv_residency.$$"
trap 'rm -f "$OUT.a" "$OUT.b"' EXIT

# `N doctests, N failures[, N excluded]` — pull one field out of ExUnit's tally.
tally() { grep -oE "[0-9]+ $2" "$1" | tail -1 | cut -d' ' -f1; }
names() { sed -n 's/^ *[0-9][0-9]*) doctest \(.*\) (Nx\.Vulkan\.NxDoctestTest)$/  \1/p' "$1"; }

echo "==> pass A: strict, register applied"
NXV_HOST_FALLBACK=raise mix test "$FILE" --seed 0 >"$OUT.a" 2>&1 || true
tail -1 "$OUT.a"

echo "==> pass B: strict, register OFF (measuring the truth)"
NXV_HOST_FALLBACK=raise NXV_DOCTEST_REGISTER=off mix test "$FILE" --seed 0 >"$OUT.b" 2>&1 || true
tail -1 "$OUT.b"

total=$(tally "$OUT.a" "doctests?")
failed_a=$(tally "$OUT.a" "failures?")
excluded=$(tally "$OUT.a" excluded)
failed_b=$(tally "$OUT.b" "failures?")
: "${excluded:=0}"

if [ -z "$total" ] || [ -z "$failed_a" ] || [ -z "$failed_b" ]; then
    echo
    echo "FAIL: could not read a tally out of ExUnit's output. Full log:"
    cat "$OUT.a"
    exit 2
fi

resident=$((total - failed_b))
echo
echo "-------------------------------------------------------------------"
awk -v r="$resident" -v t="$total" \
    'BEGIN { printf "doctest Nx residency: %d / %d (%.1f%%) run with host fallbacks refused\n", r, t, 100 * r / t }'
echo "-------------------------------------------------------------------"

status=0

if [ "$failed_a" -ne 0 ]; then
    echo
    echo "FAIL: $failed_a doctest(s) fell back that the register does not excuse."
    echo "      A residency REGRESSION — these used to stay on the GPU:"
    names "$OUT.a"
    echo
    echo "      Fix the gate, or add them to test/nx_doctest_register.exs with a"
    echo "      reason. Adding is allowed; adding without a reason is not."
    status=1
fi

if [ "$excluded" -ne "$failed_b" ]; then
    echo
    echo "FAIL: the register excuses $excluded doctest(s), but $failed_b actually fall back."
    if [ "$excluded" -gt "$failed_b" ]; then
        echo "      STALE entries — the rate above is understating the truth."
    else
        echo "      The register is short, or the ordinals have renumbered."
    fi
    echo "      These are the doctests that fall back today; the register must"
    echo "      name exactly this set:"
    names "$OUT.b"
    status=1
fi

exit $status
