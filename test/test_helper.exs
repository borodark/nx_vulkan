ExUnit.start()

# The doctest residency ratchet (W2). Nx.Vulkan.NxDoctestTest runs Nx's own 843
# doctests; 524 of them still leave the GPU for at least one op, and each is
# named with a reason in the register below. Under `NXV_HOST_FALLBACK=raise`
# those 524 step aside so the other 319 can assert that they stayed resident —
# a host fallback is bit-identical to the GPU path, so refusing it is the only
# thing that can detect one.
#
# This has to happen HERE, not in the test file. `mix test` merges
# `Application.get_env(:ex_unit, :exclude)` with the command-line excludes right
# after requiring test_helper.exs, then hands that snapshot to
# `ExUnit.async_run/0` before any test file loads — an `ExUnit.configure/1` from
# inside a test file lands after the runner has already taken its copy and is
# silently ignored.
#
# NXV_DOCTEST_REGISTER=off runs strict with the register NOT applied; that is
# how scripts/doctest_residency.sh measures the truth to check the register
# against.
Code.require_file("nx_doctest_register.exs", __DIR__)

if Nx.Vulkan.Fallback.mode() == :raise and System.get_env("NXV_DOCTEST_REGISTER") != "off" do
  ExUnit.configure(
    exclude: ExUnit.configuration()[:exclude] ++ Nx.Vulkan.NxDoctestRegister.filters()
  )
end
