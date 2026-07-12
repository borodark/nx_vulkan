# TODO files moved to exmc repo — 2026-07-12

The three handoff files that used to live here have been folded into
`pymc/exmc/research/` so both Kepler hosts and super-io share one
canonical git-tracked location (they previously drifted between
hosts because each pulled the nx_vulkan repo at its own cadence).

| Was | Now |
|---|---|
| `nx_vulkan/248_TODO.md` | `pymc/exmc/research/HANDOFF_MAC248_AMPERE_DEVICELOST.md` |
| `nx_vulkan/247_TODO.md` | `pymc/exmc/research/HANDOFF_MAC247_CAST_SHADER.md` |
| `nx_vulkan/TODO_Hypothesis_test.md` | `pymc/exmc/research/HYPOTHESIS_LINUX_VULKAN_NUTS_OVERHEAD.md` |

The live thread — Ampere DeviceLost @ 16 dispatches (merge blocker
for `feat/nx-0.12` → main) — is in
`HANDOFF_MAC248_AMPERE_DEVICELOST.md` Thread 2.

Related handoffs in the same `exmc/research/` directory:
- `D91_VERIFICATION_HANDOFF.md` — verify D91 Option C fix
- `D91_MAC248_OPTION_C_PLAN.md` — the K=1-during-warmup plan
- `247_TODO.md` (in `exmc/`, not `exmc/research/`) — synth coverage
  extension P1-P4 for mac-247
