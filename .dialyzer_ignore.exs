[
  # Nx.Backend callbacks use %{__struct__: atom()} for the data field,
  # but we correctly narrow to %Nx.Vulkan.Backend{} in our clauses.
  # Every Nx backend has these — they're structural, not bugs.
  {"lib/nx_vulkan/backend.ex", :callback_arg_type_mismatch},
  {"lib/nx_vulkan/backend.ex", :callback_type_mismatch},
  # Fast kernel specs are correct at runtime (Nx.Tensor.t()) but
  # dialyzer sees Nx.Defn.Expr structs during defn tracing.
  # Expr.optional/3 returns an Expr, not a Tensor — structural mismatch.
  {"lib/nx_vulkan/fast.ex", :invalid_contract}
]
