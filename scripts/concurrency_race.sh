#!/bin/sh
# Race batched dispatch under CONCURRENCY, sweeping the batch cap.
#
#   sh scripts/concurrency_race.sh
#   PROCS=1,2,4,8,16,32 REPS=30 sh scripts/concurrency_race.sh
#   CAPS="0 64" sh scripts/concurrency_race.sh          # just control vs default
#
# Why a shell loop instead of one Elixir script: `NXV_BATCH_MAX` is read into a
# `OnceLock` in the NIF on first dispatch and is then fixed for the life of the
# OS process. The cap sweep therefore CANNOT happen inside one `mix run`; each
# cap needs its own BEAM. The process-count sweep does happen inside one run,
# which is why that dimension is a comma list and this one is a loop.
#
# Writes bench_results/concurrency_<device>_cap<N>.json per cap.
#
# BEFORE YOU TRUST THE OUTPUT
#
#   - The box must be QUIET. This benchmark measures contention; anything else
#     using the GPU is measured as if it were part of the workload. That
#     includes a test suite in another worktree, another agent's benchmark, or
#     a compositor doing something ambitious. Check first.
#   - `git checkout` alone leaves a pre-existing local branch stale and
#     silently benchmarks the wrong tree. That has cost a full round in this
#     project before, which is why the SHA guard below exists.
set -e

cd "$(dirname "$0")/.."

CAPS=${CAPS:-"0 4 16 64 256"}
PROCS=${PROCS:-"1,2,4,8,16"}
REPS=${REPS:-20}

# SHA guard: refuse to benchmark a tree that is not what the operator thinks it
# is. Set EXPECT_SHA to the commit you intend to measure.
HEAD_SHA=$(git rev-parse --short HEAD)
if [ -n "$EXPECT_SHA" ] && [ "$HEAD_SHA" != "$EXPECT_SHA" ]; then
  echo "ABORT: HEAD is $HEAD_SHA, expected $EXPECT_SHA." >&2
  echo "       Fast-forward the branch explicitly; 'git checkout' alone is not enough." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "WARNING: working tree is dirty — the report will be labelled $HEAD_SHA," >&2
  echo "         which is not what is being measured." >&2
fi

echo "==> commit   : $HEAD_SHA"
echo "==> caps     : $CAPS"
echo "==> procs    : $PROCS"
echo "==> reps     : $REPS"
echo

echo "==> mix compile (native NIF + shaders)"
MIX_ENV=${MIX_ENV:-dev} mix compile

for cap in $CAPS; do
  echo
  echo "############################################################"
  echo "# NXV_BATCH_MAX=$cap"
  echo "############################################################"
  NXV_BATCH_MAX="$cap" PROCS="$PROCS" REPS="$REPS" \
    mix run examples/concurrent_dispatch_bench.exs
done

echo
echo "==> done. Reports:"
ls -1 bench_results/concurrency_*.json 2>/dev/null || echo "  (none written)"
echo
echo "Read the cap=0 arm as the control: it is submit-per-dispatch, so it has no"
echo "shared bucket to contend for. If batching's advantage over it shrinks as N"
echo "rises, the single global pending queue is the cause and the GPU-node work"
echo "(T1 follow-ups) has a measured motivation. If the advantage holds, it does not."
