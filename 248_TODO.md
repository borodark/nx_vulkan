# mac-248 — Quick TODO: commit `transpose.spv`

**Why**: `shaders/transpose.comp` is in spirit, but `transpose.spv` was never
committed. Both Macs have it locally (from an earlier build) so their tests
pass; Linux has no `glslangValidator` and excludes the test. One commit
unblocks all three hosts.

## Steps

```
cd ~/spirit
git pull
git checkout -b chore/commit-transpose-spv

# transpose.spv should already be on disk from earlier work.
# Recompile to be safe — deterministic output:
glslangValidator -V shaders/transpose.comp -o shaders/transpose.spv

git add shaders/transpose.spv
git commit -m "shaders: commit transpose.spv (was on disk but untracked)"
git push origin chore/commit-transpose-spv
```

## After your push

Linux side will:
1. Merge `chore/commit-transpose-spv` to `feature/vulkan-backend` in spirit.
2. `cd ~/projects/learn_erl/nx_vulkan` and rerun `mix test` — the
   `transpose.spv` will copy into nx_vulkan's priv dir on next build.
3. Drop the `--exclude needs_transpose_shader` flag from default test runs;
   re-tag the 1 remaining test (`mass-matrix-style: cholesky → solve
   composition`) to no longer require it.
4. Three-host parity: 104 tests, 0 failures, 5 excluded everywhere
   (only `:needs_compare_shader` left, which is also obsolete since
   `elementwise_binary.spv` already includes the compare ops — that
   exclusion can probably be dropped too, but separate cleanup).

30 seconds of work; biggest impact is dropping the Linux test-exclude flag.
