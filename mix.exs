defmodule Nx.Vulkan.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/borodark/nx_vulkan"

  def project do
    [
      app: :nx_vulkan,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      dialyzer: [
        plt_add_apps: [:nx, :ex_unit, :mix],
        plt_file: {:no_warn, "priv/plts/dialyzer.plt"},
        ignore_warnings: ".dialyzer_ignore.exs"
      ],
      description: description(),
      package: package(),
      source_url: @source_url,
      docs: docs()
    ]
  end

  defp description do
    "GPU tensor backend for Nx via Vulkan compute (vulkano/Rust, f64-only compute). Validated on Axon training, Scholar linear regression, and the eXMC NUTS sampler. Works on Linux + FreeBSD NVIDIA where CUDA does not exist."
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "WHY.md",
        "livebooks/intro_10min.livemd",
        "CHANGELOG.md",
        "ROADMAP.md",
        "docs/VULKANO_BACKEND_ROADMAP.md"
      ]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.13"},
      # Pure-Rust vulkano NIF. Rustler manages NIF resource lifetimes so
      # tensor handles get freed when their Elixir reference is GC'd.
      # Pin to 0.36; 0.37.3 has a rustler-sys signature mismatch with
      # rustc 1.90 (`&self.as_c_arg()` where `self.as_c_arg()` is wanted).
      {:rustler, "~> 0.36.0"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      # Already a transitive dep via rustler; declared here so the parity
      # test suite's JSON-report output is documented + version-pinned.
      {:jason, "~> 1.4"}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      maintainers: ["Igor Ostaptchenko"],
      links: %{
        "GitHub" => @source_url,
        "Blog: The Backend That Didn't Need to Know" =>
          "http://www.dataalienist.com/blog-backend-didnt-need-to-know.html"
      },
      files:
        ~w(
          lib
          native/nx_vulkan_vulkano/Cargo.toml
          native/nx_vulkan_vulkano/src
          priv/shaders
          docs
          examples
          livebooks
          CHANGELOG.md
          README.md
          LICENSE
          mix.exs
          .formatter.exs
        )
    ]
  end
end
