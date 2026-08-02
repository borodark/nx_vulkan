# Second Chances for Stranded Silicon

*A sourced assessment of the hypothesis that rising AI-hardware prices give
discarded 5–10-year-old hardware a second life — and where `nx_vulkan` sits in
it.*

**Date:** 2026-08-02 · **Status:** research memo (living document) ·
**Companion blog:** *Second Chances for Stranded Silicon* on dataalienist.com

---

## The hypothesis under test

> Current upward trends in the pricing of hardware involved with model
> training/inference infrastructure open the opportunity for second-hand,
> discarded, 5–10-year-old hardware to be given a second chance — provided
> software libraries push that weak hardware to its maximum, given the
> well-researched boundaries of what is possible for a specific GPU / RAM / CPU
> combination. `nx_vulkan` is offered as one such library.

**Verdict in one line.** Substantially true, with two corrections to the
framing. The economic pull is real; the software mechanism is real and
well-founded; but the viable band of "old" hardware is **narrower and newer**
than "5–10 years," and software repeals **none** of the physical ceilings
(memory bandwidth, VRAM capacity, perf/watt). It relocates the frontier of what
is *worth doing* on hardware you already own. It does not move the wall.

The reframing this whole memo turns on:

> **Software doesn't move the wall; it moves the line of what's worth doing on
> your side of it.** Rising new-silicon cost and structural scarcity widen the
> band of workloads for which *slower, cheaper, already-owned, correctly-utilized
> old hardware* is the rational choice — and vendor-neutral, roofline-disciplined
> software like `nx_vulkan` is what converts stranded silicon from e-waste back
> into a compute asset.

Confidence tags below: **[P]** primary / peer-reviewed · **[S]** secondary
aggregator, cross-checked · **[A]** anecdote / single report · **[C]** contested.

---

## Claim 1 — Hardware prices are trending upward

**Verdict: true in the parts that matter, but genuinely two-sided.** Absolute
and aggregate cost rise; per-unit cost-per-FLOP falls. Jevons' paradox
reconciles the two.

### Rising (directionally unambiguous)

- **HBM / DRAM is the clearest "up" signal.** SK Hynix and Micron report their
  **entire 2026 HBM output sold out** under fixed-price contracts; **~20% HBM3E
  price hikes** planned for 2026; Micron expects shortage "well beyond 2027."
  Wafer reallocation to HBM pushed **conventional DRAM contract prices ~90–95%
  QoQ in Q1 2026.** HBM is now the largest accelerator cost line — **~20% of BOM
  on A100 → ~52–58% on B200.** [P/S] TrendForce (2025-12-24); Micron earnings;
  SK Hynix Q2 2026 (+257% YoY revenue, record 76% op margin); Epoch AI B200
  breakdown.
- **Power / grid costs are rising sharply.** Datacenter electricity demand grew
  **+17% in 2025** (IEA, 2026-04-16); the PJM capacity auction went **$28.92 →
  $329.17/MW-day (~+1,053%)**; datacenter buildout cost rose **$7.7M/MW (2020) →
  $10.7M/MW (2025)**; US utilities requested **>$29B in rate increases in H1
  2025, double H1 2024.** [P] IEA; Utility Dive; EESI; JLL.
- **Per-frontier-run training cost is growing ~2.4×/yr**, projected **>$1B by
  2027** (GPT-4 ~$78M; Gemini Ultra ~$191M). [P] Epoch AI.
- **New-silicon scarcity is structural, not merely cyclical.** Blackwell
  **B200/GB200 sold out through mid-2026**, ~3.6M-unit backlog; NVIDIA secured
  ~70% of TSMC CoWoS-L capacity; lead times ~36–52 weeks. [S] financialcontent;
  gpuaas.

### Falling (the honest counter-evidence)

- **Cost-per-FLOP falls ~30–40%/yr** (hardware FLOP/$ doubles every ~2.3–2.5
  yr). [P] Epoch AI; Stanford AI Index 2025.
- **Inference $/token fell ~280× in 18 months** (GPT-3.5-level: $20 → $0.07 /M
  tokens, Nov 2022 → Oct 2024). [P] Stanford AI Index 2025.
- **H100 rental collapsed ~64% in 2025** (~$8 → ~$2–3/hr; AWS cut P5 ~44% in
  Jun 2025; 300+ neoclouds entered) — **but SemiAnalysis's primary index shows a
  ~40% rebound into 2026** ($1.70 → $2.35/hr, 1-yr contract), suggesting the dip
  was oversupply, not a trend reversal. [P/S] introl; Silicon Data; SemiAnalysis
  GPU index.
- **Used prior-gen prices fell 50–70%** (A100 80GB SXM → ~$5–9K). [S]
  hashrateindex.

### The reconciling mechanism

**Jevons' paradox** (Satya Nadella invoked it by name, Jan 2025): cheaper
*effective* FLOPs drive demand faster than supply expands, so **per-unit cost
falls while aggregate cost and scarcity rise simultaneously.** The hypothesis
leans on the second half — which is real. But the falling cost-per-FLOP is the
same force that makes the *newest* hardware attractive; that is the counter-
pressure the thesis must survive.

*Caveat:* NVIDIA publishes **no** official rack list prices — all GB200
(~$3M) / GB300 (~$3.7–4M) / Rubin (~$8.8M rumored) per-rack figures are
third-party estimates. Vendor efficiency multipliers (NVIDIA's 25–35× cost/token
claims) are best-case; independent per-GPU gains are ~2–2.5×.

---

## Claim 2 — Old hardware gets a second chance

**Verdict: true, but the sweet spot is narrower and newer than "5–10 years."**

### The supply pipeline is real and flowing

- **Depreciation schedules stretched to 5–6 years** release older gear
  downstream: Microsoft 4→6 yr (2022, ~$3.7B FY23 benefit); Google →6 yr (2023,
  −$3.9B depreciation). **Amazon reversed 6→5 yr in Jan 2025 "citing the
  increased pace of AI/ML," plus a $920M accelerated-depreciation charge** — the
  clearest signal that AI gear is being cycled *faster*, not slower. [P] 10-K /
  8-K filings.
- **The controversy [C]:** Michael Burry (Nov 2025) accused hyperscalers of
  understating depreciation ~$176B cumulatively 2026–28, arguing frontier GPUs
  last ~2–3 yr, not 5–6, given NVIDIA's annual cadence. **Contested — no public
  GPU failure-rate data substantiates "2–3 years."**
- **Fate of old gear:** GPUs cascade *down tiers* (training → inference → dev →
  edge) rather than being scrapped; refurb enterprise gear resells at **40–60%
  discounts**; ITAD market ~$17.5–28B (2025). [S] rcrtech; Grand View.
- **E-waste + policy tailwind:** 62 Mt e-waste in 2022 (only 22.3% recycled, UN
  2024); AI-specific e-waste projected up to ~2.5 Mt/yr by 2030 (Nature Comp.
  Sci. 2024); extending server life one year cuts embodied carbon ~16%; EU
  Right-to-Repair in force Jul 2024. [P]

### The prices are trivial, and they price VRAM, not speed

Tesla **P100 16GB HBM2 ~$80** · **P40 24GB ~$150–260** · **M40 24GB ~$150** ·
**K80 24GB ~$60** (but 2×12GB usable) · **V100 16GB ~$180–389** · consumer
**GTX 1080 Ti 11GB ~$140**. [S/P-sold, gpudojo / eBay, Aug 2026]

The market's revealed preference is stark: the P40 (24GB single pool) trades at
~$240–400 while the compute-similar M40 sits at ~$150 and the K80 at ~$60. **The
premium is the LLM community bidding for a *usable* 24GB — capacity, not FLOPs.**

### Real, reproducible workloads run on it

- **P100 16GB:** Llama2-7B **~50 tok/s**, Llama3.1-8B ~33 tok/s (Ollama
  benchmark). [P-bench]
- **4×P40 rig (~$2,500, ~800W):** Llama3.1-8B ~48 tok/s, **gpt-oss 120B MoE
  ~28 tok/s**. [P-bench]
- **MoE + QLoRA widen the envelope:** gpt-oss 120B activates only ~5.1B
  params/token, so it hosts on cheap multi-GPU rigs; QLoRA fits a 7B fine-tune
  in ~6GB. [P/S]

### The two corrections the framing needs

1. **The CUDA-13 cutoff (2025) is a hard, dated obsolescence line.** NVIDIA
   removed **Maxwell, Pascal, AND Volta** offline-compile / library support in
   CUDA 13.0; the minimum target is now **Turing (sm_75)**. PyTorch dropped
   Maxwell/Pascal; FlashAttention-2 needs Ampere; BF16 needs Ampere. Everything
   Volta-and-older is off the modern *CUDA* toolchain. [P] NVIDIA CUDA 13.0
   release notes.
2. **The community's actual "best value" is not the 10-year-old card — it's the
   ~2020 used RTX 3090** (Ampere, 24GB, ~$600–900): real tensor cores, BF16,
   FlashAttention, ~4–5× a P40. The genuinely ancient Kepler/Maxwell gear is
   cheap *because it's stranded.* [S]

So the defensible band is **~4–8-year-old hardware**, and its *upper* end
(Pascal/Kepler) is viable **only on a non-CUDA path** — which is exactly where
Claim 3 and `nx_vulkan` come in.

---

## Claim 3 — Software unlocks it via well-researched boundaries

**Verdict: the best-supported part of the thesis.** Three findings make it
rigorous rather than hand-wavy.

### The "boundaries" are a real, published model

The **Roofline model** (Williams, Waterman & Patterson, *Communications of the
ACM* 52(4):65–76, April 2009, DOI 10.1145/1498765.1498785): attainable
performance = **min(peak compute, peak bandwidth × arithmetic intensity)**,
where arithmetic intensity = FLOPs per byte moved. Two device constants fix a
**ridge point** that separates memory-bound from compute-bound and names the
binding resource for any given kernel. [P]

Modern LLM work uses it directly (arXiv:2402.16363, Feb 2024): **prefill is
compute-bound (GEMM, high AI); token-generation is memory-bandwidth-bound
(matrix-vector, AI ≈ 1).** That is *why* quantization — cutting bytes moved — is
the dominant lever on weak hardware, and why a decade-old memory subsystem, not
the ALUs, is usually the wall. [P] NVIDIA Nsight Compute auto-plots each
kernel's achieved AI against the roofs.

### Vulkan / SPIR-V is the structural rescue for CUDA-abandoned silicon

Because it runs against the graphics/compute driver, **not the CUDA toolchain,
Vulkan compute keeps working after CUDA drops a GPU.** The llama.cpp Vulkan
scoreboard shows a decade-old **AMD RX 470 still doing 207 tok/s** (Llama-2 7B
Q4_0). One SPIR-V binary targets NVIDIA / AMD / Intel / Apple-via-MoltenVK /
Android. The Linux Foundation hosts **Kompute** for exactly this. [P/S]
llama.cpp #10879; lei.chat GPGPU-ML-Vulkan.

**The measured cost of that portability: ~20–30% slower than CUDA on identical
NVIDIA hardware** (A100, Llama-8B pp512: 4,462 vs 2,972 t/s). Real tradeoff, not
free. [P] llama.cpp #17273.

### Software gives a second chance; it does not repeal limits

The same roofline names the ceilings no kernel can lift:

- **Memory bandwidth** — LLM decode sits below the compute roofline on
  essentially every GPU; faster kernels can't beat an old memory subsystem. [P]
- **VRAM capacity** — 70B at Q4 ≈ 35–40GB won't fit a 24GB card without slow CPU
  offload. [P, arithmetic]
- **Missing hardware features** — no tensor cores pre-Volta; consumer Pascal
  runs FP16 at **1/64 of FP32**; FP8 only on Hopper+; FlashAttention needs
  Turing+. [P] NVIDIA Pascal tuning guide.
- **Perf/watt** — H100 delivers ~3× A100's inference perf/watt; old cards cost
  multiples more energy per token. Over a 5-yr cluster life, power+cooling is
  24–34% of TCO — which inverts the economics for a low-compute-per-watt card
  running 24/7. [S] introl TCO model.

---

## Where `nx_vulkan` sits

`nx_vulkan` is a **genuine, and unusually pure, instance of the software half of
this thesis** — for a *specific* slice: f64/f32 general Nx numerical compute
(Axon, Scholar, the eXMC NUTS sampler), not quantized LLM inference. The repo's
own data corroborates the fit better than most projects could:

- **It is validated on the stranded hardware in question.** The test fleet is a
  **2012 GT 650M (Kepler)** and a **GT 750M** — cards CUDA dropped years ago —
  with a 2021 RTX 3060 Ti (Ampere) as the modern control. Same suite, **863
  doctests / 361 tests / 0 failures on all three.** The thesis, running in a lab.
- **It rides exactly the rescue path Claim 3 identifies.** Vulkan/SPIR-V, not
  CUDA — and it is *the only GPU compute path for FreeBSD + NVIDIA*, a platform
  CUDA never served. The "vendor-neutral API rescues stranded silicon" argument
  at its purest.
- **Its engineering method *is* roofline-per-device-class, empirically.** It
  doesn't assume; it *races the boundary* on real hardware. 16×16 tiling fixed
  the 1024³ cliff; the `:f32` accumulator wins **1.8–3× compute-bound / ~2–5×
  bandwidth-bound**; **register-blocking helps Ampere but regresses both Kepler
  cards**, so it's kept off; the fusion compiler's cross-stage CSE was built,
  raced, and found to **never win** — on a GPU, recompute is cheaper than the
  dispatch it saves. That last result is the thesis's own discipline turned on
  itself: knowing the device's boundary proved a textbook optimization worthless
  here. (See `bench_results/CSE_SOFTMAX_RACE.md`.)

**Placed honestly:** `nx_vulkan` shares every unliftable ceiling above — the
~20–30% Vulkan tax, bandwidth limits, and the perf/watt penalty that makes "old
hardware on a metered 24/7 grid" a TCO loser. Its honest home is exactly where
the second-hand case is strongest: **intermittent, VRAM/correctness-bound,
capex-sensitive, already-sunk-power** deployments — and CUDA-orphaned platforms
where it is the *only* option, not merely a cheap one.

---

## Bottom line

| Sub-claim | Verdict | Strongest anchor |
|---|---|---|
| Hardware prices trending up | **True in aggregate** (HBM, power, per-run, scarcity); cost-per-FLOP falling — Jevons reconciles | TrendForce HBM sell-out; IEA power +17%; Epoch $/FLOP −30–40%/yr |
| Old hardware gets a second chance | **True, narrower/newer band** (~4–8 yr; upper end CUDA-only-via-Vulkan) | CUDA-13 Turing floor; used P40/P100/3090 prices + benchmarks |
| Software unlocks it via known boundaries | **Best-supported**; Roofline + Vulkan portability are the mechanism; ceilings remain | CACM 2009 Roofline; llama.cpp Vulkan scoreboard; arXiv:2402.16363 |
| `nx_vulkan` is such a library | **Yes — a pure instance of a specific (numerical, CUDA-orphaned) slice** | 3-GPU fleet incl. 2012 Kepler; roofline-raced heuristics; FreeBSD-NVIDIA-only path |

The thesis's power is **not** "old hardware is secretly as good as new" — the
perf/watt and CUDA-EOL evidence kills that. Its power is that **rising
new-silicon cost + structural scarcity widen the band of workloads for which
slower, cheaper, already-owned, correctly-utilized old hardware is the rational
choice** — and vendor-neutral, roofline-disciplined software is what converts
that stranded silicon from e-waste back into a compute asset. Software doesn't
move the wall; it moves the line of what's worth doing on your side of it.

---

## Primary anchors (highest-trust sources)

- **Roofline model** — Williams, Waterman, Patterson, *CACM* 52(4), Apr 2009,
  DOI 10.1145/1498765.1498785.
- **LLM roofline** — "LLM Inference Unveiled: Survey and Roofline Model
  Insights," arXiv:2402.16363, Feb 2024.
- **CUDA 13.0 release notes** (Maxwell/Pascal/Volta removal; Turing floor) —
  docs.nvidia.com/cuda/archive/13.0.3/cuda-toolkit-release-notes/.
- **HBM/DRAM** — TrendForce (2025-12-24); Micron / SK Hynix earnings; Epoch AI
  B200 cost breakdown.
- **Power/grid** — IEA "Data-centre electricity use surged in 2025" (2026-04-16);
  PJM capacity auction (Utility Dive); JLL data-center outlook.
- **Cost-per-FLOP / $/token** — Epoch AI GPU price-performance; Stanford AI
  Index 2025.
- **Rental** — SemiAnalysis GPU index; Silicon Data H100 rental series.
- **Used market** — gpudojo trackers; eBay sold listings (Aug 2026).
- **Depreciation** — Microsoft/Google/Amazon/Meta 10-K & 8-K filings.
- **E-waste** — UN/ITU Global E-waste Monitor 2024; Nature Comp. Sci. (Oct 2024).
- **Vulkan portability + tax** — llama.cpp #10879 (scoreboard), #17273 (CUDA gap);
  Linux Foundation Kompute.

*Full URLs and per-claim dates are retained in the research working notes; this
memo carries the load-bearing anchors. Treat as claim-level: single-blog tok/s
figures, Burry's $176B / "GPUs last 2–3 years," and analyst TCO percentage
splits.*
