import Config

# Opt-in escape hatch for cross-compiled deploys.
#
# With NXV_SKIP_NIF_BUILD=1, Rustler does NOT build the crate and mix links
# whatever `.so` is already in priv/native. That is what lets a cross-built
# artifact survive a `mix compile` the Elixir side needs.
#
# Why it exists: `.claude/skills/jetson-cross-build` builds the Jetson's aarch64
# NIF on super-io in ~2 minutes instead of ~47 native. Until now that only
# helped for RUST-ONLY commits, because any Elixir change forces a `mix compile`
# which triggers Rustler, which rebuilds the crate natively and overwrites the
# artifact you just shipped. With this, the cross-build applies to every commit:
#
#     NXV_SKIP_NIF_BUILD=1 mix compile      # Elixir only, .so untouched
#     NXV_SKIP_NIF_BUILD=1 mix test
#
# DANGER, and it is the whole reason this is an env var rather than a setting:
# with this on, a stale or wrong-architecture `.so` produces a green suite that
# says nothing about the code you just compiled. ALWAYS checksum
# priv/native/libnx_vulkan_vulkano.so before and after. The skill documents the
# procedure; this comment exists so the trap is visible from the config too.
if System.get_env("NXV_SKIP_NIF_BUILD") == "1" do
  IO.puts(:stderr, """
  [nx_vulkan] NXV_SKIP_NIF_BUILD=1 — Rustler will NOT build the crate;
              priv/native/*.so is used as-is. CHECKSUM IT.

              This is Application.compile_env, so the value is BAKED INTO the
              compiled module and Elixir refuses to boot when the runtime value
              differs. Set this variable for `mix compile` AND every `mix run`
              or `mix test` after it. To go back to normal builds, unset it and
              force a recompile of the NIF module — deleting
              _build/<env>/lib/nx_vulkan/ebin/Elixir.Nx.Vulkan.NativeV.beam is
              enough; a plain `mix compile` will NOT clear it.
  """)

  config :nx_vulkan, Nx.Vulkan.NativeV, skip_compilation?: true
end
