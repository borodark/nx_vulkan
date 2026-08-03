[
  # Nx.Backend declares a tensor's `data` field as %{__struct__: atom()}, and
  # every callback here narrows it to %Nx.Vulkan.VulkanoBackend{}. Narrowing to
  # a subtype is what a backend *is*, so dialyzer reports a mismatch on each
  # one. Structural to the behaviour — every Nx backend, EXLA included, has
  # these.
  #
  # One member of this family looks alarming and is not: put_slice/4. Nx's
  # @callback declares `put_slice(out, tensor, tensor, list)`, but Nx actually
  # calls `put_slice(out, tensor, start_indices, slice)` (see Nx.put_slice/3 in
  # nx.ex) — the last two are transposed in the spec, not in the call. Our
  # implementation matches the call, which is why it works at runtime and is
  # covered by test/nx_vulkan/vulkano_backend_host_fallback_test.exs. Do NOT
  # "fix" the implementation to match the spec; that would break it.
  {"lib/nx_vulkan/vulkano_backend.ex", :callback_arg_type_mismatch},
  {"lib/nx_vulkan/vulkano_backend.ex", :callback_type_mismatch},

  # from_pointer/5 and to_pointer/2 delegate to Nx.BinaryBackend, which raises
  # ArgumentError("does not support pointer manipulation") unconditionally.
  # Never returning is the intended behaviour, so no_return is accurate rather
  # than a defect.
  #
  # This is file-scoped rather than line-pinned: dialyxir did not match the
  # {file, type, line} form here, and a stale line number would be worse than
  # useless. The cost is that a genuine "no local return" introduced elsewhere
  # in this file would also be suppressed — so if you add a function to this
  # module that dialyzer says never returns, check it by hand.
  {"lib/nx_vulkan/vulkano_backend.ex", :no_return}
]
