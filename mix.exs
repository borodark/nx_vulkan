defmodule Nx.Vulkan.MixProject do
  use Mix.Project

  @version "0.3.0"
  @source_url "https://github.com/borodark/nx_vulkan"

  def project do
    [
      app: :nx_vulkan,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      dialyzer: [
        # :unmatched_returns is the one that finds real bugs — an ignored
        # {:error, _} from a NIF or a File call. It is OFF by default in
        # dialyxir, so it had never run here; when first enabled it reported
        # four ignored returns, all genuinely best-effort, now written `_ =`
        # rather than silenced.
        #
        # :error_handling reports functions that only terminate by raising.
        # There is one deliberate case (`__shard_jit__/6`), filtered.
        #
        # NOT enabled: :underspecs, :overspecs, :specdiffs. Those reported 13
        # findings and every one was a spec deliberately narrower or wider than
        # the inferred success typing, which is what a hand-written spec is FOR.
        # Turning them on would trade a useful signal for noise.
        flags: [:unmatched_returns, :error_handling],
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
    "GPU tensor backend for Nx via Vulkan compute (vulkano/Rust, native f32 and f64), forward and backward pass on-device. Validated on Axon training, Scholar linear regression, and the eXMC NUTS sampler. Works on Linux + FreeBSD NVIDIA where CUDA does not exist."
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
        "docs/VULKANO_BACKEND_ROADMAP.md",
        # The CHANGELOG links to these; without them here the links 404 on
        # hexdocs, since `bench_results/` is not in `package/0`'s file list.
        # They are the evidence behind this release's performance claims.
        #
        # NOT extended to MODEL_SCALING.md, which ROADMAP.md also links.
        # Shipping it CASCADES: it in turn links model_scaling/raw_grad.txt,
        # raw_exla.txt, EXMC_PEROP_RACE.md and CONCURRENT_DISPATCH.md, all of
        # which exist in the repo and none of which are extras, so `mix docs`
        # simply trades one warning for four. ROADMAP.md links it by absolute
        # GitHub URL instead — a pattern that file already used elsewhere.
        "bench_results/BATCHED_DISPATCH.md",
        "bench_results/MNIST_EXLA_RACE.md",
        "bench_results/CSE_SOFTMAX_RACE.md"
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
    ] ++ exla_dep()
  end

  # EXLA is OPT-IN, and it has to be.
  #
  # `parity_test.exs` runs a second, independent reference when EXLA is present
  # and self-skips when it is not — the check is `Code.ensure_loaded?/1` at
  # runtime, so nothing in the test code needs a flag. Only the DEPENDENCY does.
  #
  # Declaring it unconditionally would break the FreeBSD half of the fleet.
  # `exla` pulls `xla`, which ships precompiled binaries for a fixed matrix of
  # targets — linux and darwin, x86_64 and aarch64 — and **there is no FreeBSD
  # build in it at all**. Both Keplers would fail `mix deps.get`, which takes
  # `mix test` down repo-wide. That is not hypothetical: it is the standing
  # failure mode in the sibling `exmc` repo, and it is the one thing
  # `rm -rf _build` does not fix.
  #
  # So: off by default, on where it is known to work.
  #
  #     NXV_WITH_EXLA=1 mix deps.get
  #     NXV_WITH_EXLA=1 mix test
  #
  # The flag is needed for BOTH — `deps/0` is evaluated every time, so a build
  # without it simply does not see the dependency.
  #
  # TOGGLING THE FLAG POISONS `_build`, IN EITHER DIRECTION. `mix` bakes the
  # resolved dependency list into `_build/<env>/lib/nx_vulkan/ebin/nx_vulkan.app`
  # and does not regenerate it when only the environment changed, because no
  # SOURCE changed. So a flagged build leaves `exla` in that file's
  # `applications` list, and the next unflagged run aborts before a single test:
  #
  #     ** (Mix) Could not start application exla:
  #        could not find application file: exla.app
  #
  # Clear the two stale artifacts — no recompile, and nothing tracked is
  # touched:
  #
  #     rm -f _build/test/lib/nx_vulkan/ebin/nx_vulkan.app \
  #           _build/test/lib/nx_vulkan/.mix/compile.app_cache
  #
  # `rm -rf _build` is equally safe and costs nothing here: the 11-minute NIF
  # build lands in `deps/exla/cache/`, and `_build` only symlinks to it.
  #
  # Known-good targets: super-io (linux x86_64) and the Jetson
  # (linux aarch64, CPU only — its CUDA is 10.2 and permanently so, while the
  # prebuilts start at CUDA 12). See NEXT.md §1.4.
  defp exla_dep do
    if System.get_env("NXV_WITH_EXLA") in ["1", "true"] do
      [{:exla, "~> 0.13", only: [:dev, :test]}]
    else
      []
    end
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
      # `docs` was a blanket glob, which shipped four blog drafts (published
      # on the website anyway) and several May-era internal planning
      # documents to every Hex user — including docs/NX_PARITY_RESEARCH.md,
      # which docs/PARITY_STATUS.md explicitly labels "stale, do not use".
      # Shipping a document another shipped document tells you not to read is
      # not a docs directory, it is a working tree. Explicit list instead:
      # things a *consumer* of the package has reason to open.
      files: ~w(
          lib
          native/nx_vulkan_vulkano/Cargo.toml
          native/nx_vulkan_vulkano/src
          priv/shaders
          docs/BACKWARD_PASS_AUDIT.md
          docs/BACKEND_VERIFICATION_GAP.md
          docs/PARITY_STATUS.md
          docs/VULKANO_BACKEND_ROADMAP.md
          docs/SECOND_CHANCES_THESIS.md
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
