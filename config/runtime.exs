import Config

# Host-fallback strictness. See `Nx.Vulkan.Fallback`.
#
#   :allow  (default) — a fallback is correct, just slower. This is the
#                       behaviour that makes "every Nx op works" true, and it
#                       must stay the default.
#   :warn             — log every fallback that is not on the allowlist
#   :raise            — raise Nx.Vulkan.HostFallbackError on the first one
#
# Runtime (not compile-time) config so the CI ratchet can flip it without a
# rebuild: `NXV_HOST_FALLBACK=raise mix test`, which is what
# `scripts/strict_test.sh` and the strict-fallback CI job run.
#
# Note this file configures *this* project only. A library's config is never
# evaluated for its consumers, so depending on :nx_vulkan does not inherit it —
# an application that wants strict mode sets `config :nx_vulkan, host_fallback:`
# itself, or scopes it with `Nx.Vulkan.Fallback.strict/2`.
host_fallback =
  case System.get_env("NXV_HOST_FALLBACK") do
    nil -> :allow
    "allow" -> :allow
    "warn" -> :warn
    "raise" -> :raise
    other -> raise "NXV_HOST_FALLBACK must be allow | warn | raise, got: #{inspect(other)}"
  end

config :nx_vulkan, host_fallback: host_fallback
