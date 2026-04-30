defmodule Nx.Vulkan.MixProject do
  use Mix.Project

  @version "0.0.1"
  @source_url "https://github.com/borodark/nx_vulkan"

  def project do
    [
      app: :nx_vulkan,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      dialyzer: [plt_add_apps: [:nx], ignore_warnings: ".dialyzer_ignore.exs"],
      description: "Nx tensor backend on Vulkan compute. Works on FreeBSD where CUDA does not.",
      package: package(),
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.7"},
      # The NIF — bound to spirit's Vulkan compute backend via a small
      # extern "C" shim. Rustler manages NIF resource lifetimes so
      # tensor handles get freed when their Elixir reference is GC'd.
      # Pin to 0.36; 0.37.3 has a rustler-sys signature mismatch with
      # rustc 1.90 (`&self.as_c_arg()` where `self.as_c_arg()` is wanted).
      {:rustler, "~> 0.36.0"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib native mix.exs README.md PLAN.md .formatter.exs)
    ]
  end
end
