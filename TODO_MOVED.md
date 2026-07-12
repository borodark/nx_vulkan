# TODO files moved to exmc/research/ — 2026-07-12

The three handoff files that used to live here have been folded into
the phd repo, under `exmc/research/`. On Mac hosts that's
`~/exmc/exmc/research/`; on super-io it's
`~/projects/learn_erl/pymc/exmc/research/`. They're now git-tracked
in one place so both Kepler hosts stop drifting apart at each host's
pull cadence.

| Was (this repo) | Now (phd repo, `exmc/research/`) |
|---|---|
| `nx_vulkan/248_TODO.md` | `HANDOFF_MAC248_AMPERE_DEVICELOST.md` |
| `nx_vulkan/247_TODO.md` | `HANDOFF_MAC247_CAST_SHADER.md` |
| `nx_vulkan/TODO_Hypothesis_test.md` | `HYPOTHESIS_LINUX_VULKAN_NUTS_OVERHEAD.md` |

The live thread — Ampere DeviceLost @ 16 dispatches (merge blocker
for `feat/nx-0.12` → main) — is in
`HANDOFF_MAC248_AMPERE_DEVICELOST.md` Thread 2.

Other current handoffs in the same directory:

- `D91_VERIFICATION_HANDOFF.md` — verify D91 Option C fix
- `D91_MAC248_OPTION_C_PLAN.md` — the K=1-during-warmup plan
- `247_TODO.md` (in `exmc/`, not `exmc/research/`) — synth coverage
  extension P1-P4 for mac-247
