[
  # Nx.Backend callbacks use %{__struct__: atom()} for the data field,
  # but we correctly narrow to %Nx.Vulkan.Backend{} in our clauses.
  # Every Nx backend has these — they're structural, not bugs.
  {"lib/nx_vulkan/backend.ex", :callback_arg_type_mismatch},
  {"lib/nx_vulkan/backend.ex", :callback_type_mismatch}
]
