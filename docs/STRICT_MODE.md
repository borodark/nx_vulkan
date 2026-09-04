# Strict mode — making a silent host fallback impossible to miss

**Scope:** the fallback counter, `host_fallback: :raise`, the allowlist
discipline, and the doctest residency ratchet.

## Why this exists

Every op this backend cannot run natively transfers to `Nx.BinaryBackend`,
computes, and transfers back. The result is **bit-identical** — the fallback
*is* the reference implementation — so no assertion on values can ever tell
"ran on the GPU" from "silently didn't." That is not hypothetical: conv's
entire backward pass ran on the CPU for the whole life of the conv shaders,
with a green suite and green doctests, until somebody counted.

`Nx.Vulkan.Fallback.count/1` makes a fallback detectable. Strict mode makes it
impossible to miss:

```elixir
config :nx_vulkan, host_fallback: :allow   # default — correct, just slower
config :nx_vulkan, host_fallback: :warn    # log every non-allowlisted fallback
config :nx_vulkan, host_fallback: :raise   # raise Nx.Vulkan.HostFallbackError
```

`:allow` is the default and is meant to stay the default. A library that
raised on a correct-but-slow path would give up the property that makes this
backend usable at all: *every* `Nx` op works on it.

Scope it to a block instead of the whole VM. This is per-process, so one strict
test cannot poison an `async: true` suite:

```elixir
Nx.Vulkan.Fallback.strict(fn -> Nx.Defn.jit_apply(train_step, params) end)
Nx.Vulkan.Fallback.strict(:warn, fn -> ... end)
Nx.Vulkan.Fallback.strict(:allow, fn -> Nx.LinAlg.svd(a) end)   # escape hatch
```

**Why raise rather than count.** The counter is documented as a *lower bound*:
once a fallback strands a tensor on `Nx.BinaryBackend`, Nx dispatches
everything downstream there without this backend ever seeing it, so a census
shows the visible edge of a cascade. `:raise` fires on the **first** refused
op, before the tensor has left the device — so a strict failure names the
cause, not the symptom.

**The allowlist is the whole risk surface.** `@allowlist` in
`lib/nx_vulkan/fallback.ex` is one line per exemption, each naming a single
`{fun, arity}` and the reason it is permitted. There are no wildcards and no
"this op family may fall back": `{:transpose, 3}` is not exempt — only
`{:transpose, 3}` *at rank ≥ 5* is, because the rank-4 case has a shader and a
rank-4 transpose leaving the GPU is a bug. An allowlist that grows loosely is
how `:raise` comes to mean nothing, which is the same way the original gates
got too wide.

Run the whole suite that way:

```sh
sh scripts/strict_test.sh
```

Two tag-based exclusions, both enumerated in the tree and both printed by the
CI job: `:host_fallback_expected` for tests whose *subject* is the fallback
path, and `:host_fallback_open` for real fallbacks that are tracked and open.
Neither is skipped by a normal `mix test`.

Nx's own doctests run under the ratchet too, minus the ones named in
`test/nx_doctest_register.exs` — 78 of 833 that still leave the GPU for at
least one op, each line carrying its reason. That makes the share of upstream
Nx this backend actually computes on-device a number you can print:

```sh
sh scripts/doctest_residency.sh
#=> doctest Nx residency: 755 / 833 (90.6%) run with host fallbacks refused
```

It fails if a doctest not in the register falls back (a regression) *or* if one
in the register stops falling back (a stale entry holding the number down), so
the rate moves only when someone edits the register deliberately.
