# Midpoint Check-in Draft: Optimizing `circuit_tracer` for Exact Tracing on Limited Hardware

## 1. Project overview and change in direction

This is formally a team project, but up to this midpoint the implementation, debugging, benchmarking, and analysis reported here have been done almost entirely by me. I think it is worth stating that explicitly in the writeup, because the current project state reflects work I completed personally rather than a fully parallelized team workflow. At the same time, the project is now in a better state for parallel work than it was earlier in the semester: now that exact tracing is at least workable, there are clearer subproblems that can be split across teammates.

My original proposal focused on a scientific question: whether the temporal stability of attribution circuits during autoregressive generation predicts answer correctness on GSM8K math problems. The intended workflow was to trace exact attribution graphs over multiple generation steps, evaluate each completion for correctness, and then compare temporal stability metrics between correct and incorrect answers.

After beginning implementation, the main bottleneck was not the analysis itself but the cost of obtaining the traces. Exact tracing on realistic GSM8K prompts was far more expensive than expected in both GPU memory and host RAM. In the original setup, longer prompts could require more than 80 GB of VRAM, which is beyond the hardware that is consistently available to me on OSC. In practice, this meant that the core scientific experiment could not be run at useful scale without first solving the systems problem.

Part of the reason I underestimated this at the proposal stage is that some of the papers I was using as motivation were operating at a much higher hardware budget than I initially realized. For example, work I cited in the proposal used **H200** GPUs, and GraphGhost reports full-sample GSM8K tracing times on the order of roughly **0.5 to 3 hours per sample**, depending on the prompt. Other related work such as CRV was done in a much more resource-rich environment, including training custom transcoders on **8×H200**. By comparison, the best GPU resource I can reliably target on the Ohio Supercomputer Center is an **H100**, and Ascend only provides **A100 40GB** nodes. So the gap between the naive experimental plan and the actually available hardware was much larger than I originally thought.

There is also a mismatch between the public framing of the library and the workloads I ultimately care about. The `circuit_tracer` library advertises that it can run even in environments like Google Colab, but the example prompts used there are very short. My later measurements show that short prompts produce much smaller feature counts, and therefore much smaller RAM/VRAM usage. GSM8K-style prompts are long enough that the same method enters a very different resource regime.

Because of that, I changed the direction of the project. The new central question is:

> How can I optimize `circuit_tracer` enough that exact tracing becomes practical on smaller OSC hardware, especially for longer prompts where feature counts become large?

I still use GSM8K, Gemma-3-1B-IT, and GemmaScope-2 cross-layer transcoders as the motivating application, but the contribution of the project is now primarily **systems-oriented** rather than purely **hypothesis-testing**. The goal is to characterize runtime/memory bottlenecks, identify feasible operating regions, and document which combinations of batching, chunking, and caching make exact tracing practical on the available cluster hardware.

I think this is a reasonable pivot rather than a retreat from the original idea. The optimization work is what now makes the downstream scientific experiment possible at all.

---

## 2. Current system and evaluation setup

The current pipeline is built around the local fork of `circuit_tracer` with exact chunked decoder tracing. The benchmark/evaluation setup now has three pieces:

1. **Feature-distribution analysis (Phase 0 only).**
   This experiment measures how prompt length affects the number of active features during sparse encoding / reconstruction. It is useful for establishing a baseline scaling law before exact tracing.

2. **Wave 1 exact benchmark sweep.**
   This is a factorial benchmark over attribution batch size and decoder chunk size on three representative prompts (GSM8K indices 94, 361, and 828) across the two OSC clusters I used:
   - **Ascend**: A100 40GB
   - **Cardinal**: H100 80GB

   The purpose of Wave 1 is to map out the runtime / memory tradeoff surface and identify feasible exact configurations.

3. **Wave 2 cache sweeps.**
   This focuses on the harder prompt 361 in both a base form and a late-prefix form, and sweeps decoder cache budgets to evaluate whether caching can make long-prompt exact tracing more practical.

To support this evaluation, I also wrote extractor scripts that normalize the experiment outputs into flat tables. These scripts parse:

- `scenario.json` and `result.json` for configuration and status,
- `completion.json` for artifact summaries and resource snapshots,
- `run.log` for per-phase timings and CUDA OOM information,
- SLURM `.err` / `.out` files for host-memory OOMs and job metadata.

This extraction step matters because it let me verify that some earlier log-derived memory measurements were underreporting peak VRAM. For successful runs, the best GPU memory field is the final resource snapshot (`resource_snapshot_cuda_peak_reserved_gib`) from `completion.json`, not the sparse memory lines printed in `run.log`.

So at this midpoint I do not just have code that runs; I also have a more trustworthy evaluation/measurement pipeline for the optimization study.

---

## 3. Related work (section to expand in final report)

This section needs fuller citations in the final version, but the relevant areas are now clearer than they were in the proposal.

1. **Mechanistic interpretability / circuit tracing.**
   The project is directly motivated by work on attribution graphs, sparse feature circuits, and mechanistic explanations of transformer behavior. My system uses `circuit_tracer` as the main tracing framework.

2. **Sparse features and GemmaScope-style transcoders.**
   The scaling bottleneck in this project is closely tied to the number of active sparse features selected during tracing. The GemmaScope-2 cross-layer transcoder setup gives a concrete, high-dimensional sparse-feature space in which exact tracing becomes expensive.

3. **Systems optimization for interpretability tooling.**
   A lot of interpretability work reports algorithmic insights but does not focus on the engineering question of how to make exact methods practical on constrained hardware. My current project direction fits here: chunking, batching, and decoder-cache reuse are all systems-level techniques aimed at making exact tracing usable in a broader compute setting.

4. **Prior work assumes substantially more compute than my target environment.**
   This has become an important part of the story of the project. Related work such as GraphGhost and CRV demonstrates that exact or near-exact tracing can produce useful analysis, but those projects also rely on much stronger hardware assumptions than what I can treat as normal on OSC. So one way to interpret the current project is as an attempt to close part of that gap: not by changing the downstream scientific question, but by making the tracing system itself more practical under tighter resource constraints.

In the final report I should cite the specific `circuit_tracer` work/repository, the GemmaScope-2 / transcoder work, and a small amount of prior systems-oriented interpretability tooling where available. For the midpoint, the main point is that I now have a much clearer picture of where this project sits: it is no longer just a downstream application of tracing, but a study of the tracing system itself.

---

## 4. Preliminary results

### 4.1 Phase-0 feature-distribution baseline

I first ran a dedicated feature-distribution analysis on **360 GSM8K prompts** to measure how prompt length affects the number of active features before doing full exact tracing.

The main result is that prompt length is an extremely strong predictor of active feature count.

- Correlation between prompt token count and total active features: **0.998**
- Linear slope: about **43.9k active features per token**
- Mean active features per token: about **41.6k**

This is shown in **Figure `phase0_feature_distribution_scaling`**.

![Phase-0 feature distribution scaling](experiments/figures/optimization_benchmarks/phase0_feature_distribution_scaling.png)

*Figure: Phase-0 feature-distribution scaling. Left: prompt length vs active features. Middle: active features vs Phase-0 runtime. Right: active features vs peak VRAM during the Phase-0 analysis runs.*

This baseline is important because it validates a central systems intuition: **longer prompts activate more sparse features almost linearly**, so longer prompts are not just semantically harder but structurally more expensive to trace.

This experiment did **not** explain the later Phase-4 anomaly I saw on prompt 94, but it did provide a useful baseline scaling law and an early sanity check that the Phase-0 instrumentation is working correctly.

---

### 4.2 Exact tracing scales strongly with feature count

Using the exact benchmark runs, I then looked at how overall exact tracing cost scales with feature count.

Across successful exact runs (excluding the special prompt-94 anomaly rerun):

- Feature count vs total runtime correlation: **0.859**
- Feature count vs host RAM correlation: **0.999**
- Feature count vs peak VRAM correlation: positive but weaker than host RAM

Approximate fitted slopes:

- **~754 seconds per additional 1M active features** for total runtime
- **~61.3 GiB per additional 1M active features** for host RAM

These results are shown in **Figure `exact_feature_scaling`**.

![Exact tracing feature scaling](experiments/figures/optimization_benchmarks/exact_feature_scaling.png)

*Figure: Exact tracing scaling on successful runs (excluding the prompt-94 special case). The plots show how active feature count relates to total runtime, host RAM, and peak VRAM.*

The strongest conclusion here is that **host RAM is the dominant scaling bottleneck**. VRAM still matters for feasibility, but once the graph gets large, the host-side graph/materialization cost grows very steeply with the number of active features.

This is also visible in one of the most informative paired comparisons from Wave 2:

- `361_base`, conservative config (`b128/c2048/cache12g`):
  - **5.22M** active features
  - **329.5 GiB** final RSS
  - **69.9 min** total runtime
  - **25.0 GiB** peak VRAM

- `361_late`, same config family:
  - **10.61M** active features
  - **664.3 GiB** final RSS
  - **122.9 min** total runtime
  - **27.7 GiB** peak VRAM

So doubling the feature count roughly doubled the host RAM and runtime, while peak VRAM increased only moderately.

---

### 4.3 Wave 1: batch size / chunk size tradeoffs

The main benchmark suite contains **66 exact scenarios**, of which **60 succeeded** and **6 failed with OOM**. The broad purpose of Wave 1 was to characterize the tradeoff surface over:

- attribution batch size,
- decoder chunk size,
- cluster/hardware,
- prompt difficulty / prompt length.

The cleanest summary is:

1. **Higher batch size usually reduces runtime.**
2. **Higher batch size usually increases peak VRAM.**
3. **Chunk size matters, but less strongly than batch size.**
4. **Host RAM is mostly determined by prompt/feature count, not by batch/chunk configuration.**

These trends are shown in:

- **Figure `wave1_runtime_by_config`**
- **Figure `wave1_peak_vram_by_config`**
- **Figure `wave1_runtime_vram_tradeoff`**

![Wave-1 runtime by config](experiments/figures/optimization_benchmarks/wave1_runtime_by_config.png)

*Figure: Wave-1 runtime by configuration, overlaid by prompt. Color indicates batch size, while line style / marker distinguishes Ascend and Cardinal.*

![Wave-1 peak VRAM by config](experiments/figures/optimization_benchmarks/wave1_peak_vram_by_config.png)

*Figure: Wave-1 peak VRAM by configuration, overlaid by prompt. This makes the runtime / memory tradeoff easier to compare across clusters on the same prompts.*

![Wave-1 runtime/VRAM tradeoff](experiments/figures/optimization_benchmarks/wave1_runtime_vram_tradeoff.png)

*Figure: Wave-1 runtime vs peak VRAM tradeoff. Each panel is one prompt, with color indicating batch size and marker indicating cluster.*

For example, on Ascend for prompt 828, increasing batch size from 128 to 256 improved runtime noticeably while increasing VRAM by a few GiB. On Cardinal, the same general pattern holds, but the larger GPU budget allows exploration of more aggressive batch/chunk combinations.

One important qualitative result from this sweep is that **batch size is the primary runtime/VRAM knob**, while chunk size looks more like a secondary efficiency knob. That is useful for practical deployment because it suggests a simple tuning strategy: first choose the largest stable batch size for the available GPU, then tune chunk size within that feasible region.

---

### 4.4 Wave 2: cache sweeps on long-prompt stress cases

Wave 2 focuses on prompt 361 in two forms:

- `361_base`
- `361_late` (a late-prefix stress case with much larger initial input length)

This is shown in **Figure `wave2_cache_sweeps`**.

![Wave-2 cache sweeps](experiments/figures/optimization_benchmarks/wave2_cache_sweeps.png)

*Figure: Wave-2 cache sweeps on prompt 361 for the base and late-prefix variants. The top row shows runtime vs cache budget and the bottom row shows peak VRAM vs cache budget; X markers denote OOM runs.*

The results show two useful things:

1. **Decoder cache can reduce runtime substantially**, especially on the harder prompt variants.
2. **The more aggressive throughput-oriented configuration (`b256/c2048`) is less robust on the late-prefix case**, where some cache settings still OOM.

At this point my practical conclusion is:

- **`b128/c2048`** is the safer exact configuration for difficult long-prompt cases.
- **`b256/c2048`** can be faster when it works, but is less reliable.

This is exactly the kind of result I needed from the optimization study: not just “is tracing possible,” but **which configuration should I actually trust when I want a run to finish**.

---

### 4.5 Cluster comparison and hardware effects

The cluster comparison is one of the most interesting results.

Using matched successful Wave-1 configurations, I compared Ascend and Cardinal directly. The comparison is shown in **Figure `cluster_shared_config_comparison`**.

![Cluster comparison on matched configs](experiments/figures/optimization_benchmarks/cluster_shared_config_comparison.png)

*Figure: Direct Ascend vs Cardinal comparison on matched successful Wave-1 configurations. The top row compares runtime and the bottom row compares peak VRAM for the same batch/chunk settings.*

For prompts **361** and **828**, Cardinal is typically faster than Ascend on matched configurations, while using broadly comparable peak VRAM. The exact ratios vary by prompt and configuration, but the pattern is fairly consistent.

There is also a second effect captured in **Figure `cardinal_headroom_gain`**: because Cardinal has an 80 GB H100, it can run some larger configurations that are not available on Ascend’s 40 GB A100. However, the runtime gain from these Cardinal-only larger configs is **real but modest** (roughly low single-digit to mid single-digit percent in the best cases I measured). That suggests that the biggest difference is not just “more VRAM allows everything,” but also the underlying hardware / backend throughput.

![Cardinal headroom gain](experiments/figures/optimization_benchmarks/cardinal_headroom_gain.png)

*Figure: Runtime gain on Cardinal from configurations that are unavailable on Ascend. The gains are real, but modest, which suggests that raw hardware throughput matters in addition to the larger VRAM budget.*

The exception is **prompt 94**, which is a true anomaly. For prompt 94, Cardinal is much slower in Phase 4 even when the configuration is nominally identical. I now treat this as a separate debugging/investigation case rather than a general benchmark result. It likely reflects some deeper difference in Phase-4 feature selection or backend behavior rather than the normal cluster trend.

I already have a useful preliminary diagnosis for this case. I reran `prompt94_compare` specifically so that I could save comparable graph artifacts and inspect the intermediate stages directly. So far, the anomaly does **not** seem to come from Phase 0. The Phase-0 feature counts are nearly identical across clusters (about **3.71M** on Ascend vs **3.70M** on Cardinal, with very large overall overlap on the order of **3.5–3.6M** shared features). In other words, the two runs begin from almost the same sparse feature state.

The large divergence appears later, in **Phase 4**. There, the two runs select almost completely different feature sets. On Ascend, the selected features are concentrated more in shallow layers, while on Cardinal they are concentrated more in deep layers. The resulting graphs have very little overlap, even when comparing the corresponding selected layers. That is consistent with the large runtime difference I observe, because the Phase-4 work being done is no longer effectively the same computation.

At the moment, my best tentative explanation is that this may be caused by **floating-point sensitivity interacting with the current iterative feature-selection algorithm in Phase 4**. I do not want to overclaim this at midpoint, because I have not fully isolated the root cause yet. But the current evidence already narrows the problem substantially: it looks much more like a Phase-4 selection instability than a generic prompt-level or Phase-0 difference.

This anomaly is actually useful for the project: it motivated the feature-distribution investigation and highlighted that exact-tracing optimization is not just about simple scaling, but also about stability and reproducibility across hardware/software environments.

---

## 5. What progress has been made since the proposal?

I have made substantial progress relative to the proposal, even though the central question changed.

Compared to the proposal stage, I now have:

- a working exact chunked tracing pipeline on OSC,
- benchmark suites for Phase-0 scaling, Wave 1 config sweeps, and Wave 2 cache sweeps,
- scripts for extracting structured runtime/memory/failure data from logs and JSON outputs,
- a set of report-ready figures summarizing the optimization results,
- preliminary empirical conclusions about which configurations are feasible and where the main bottlenecks are.

In terms of team contribution, the current midpoint progress described above is mostly the result of my own work. I mention that not to overemphasize authorship in the midpoint report, but to explain why the current results are concentrated around tracing infrastructure, benchmarking, and systems analysis: those were the pieces I had to build first in order for the rest of the project to become runnable.

So this is not a resubmitted proposal. The project has shifted, but in a way that is justified by what I learned from implementation: the optimization problem turned out to be the real research bottleneck.

---

## 6. Limitations at midpoint

There are still several limitations.

1. **The downstream temporal-stability hypothesis is not yet tested at meaningful scale.**
   The optimization work came first because it had to.

2. **The cluster anomaly on prompt 94 is unresolved.**
   I now know that it is stable across reruns, that Phase 0 is nearly identical across clusters for this prompt, and that the major divergence appears during Phase 4 feature selection. However, I do not yet know the exact root cause. A leading hypothesis is floating-point sensitivity inside the iterative feature-selection procedure, but that still needs direct validation.

3. **The current results are strongest for benchmarking and feasibility, not yet for packaging a fully general optimizer.**
   At this stage I have solid empirical guidance and infrastructure, but the final report should probably emphasize the practical operating envelope rather than claiming a complete general solution.

---

## 7. Plan for the rest of the semester

My plan for the remaining three weeks is:

1. **Finalize the benchmark analysis.**
   Clean up figure selection and write the final experimental narrative around scaling, feasibility, and tradeoffs.

2. **Continue the optimization work inside the tracing algorithm itself.**
   The current benchmark results point to several concrete directions:
   - reduce host RAM usage by avoiding unnecessary materialization or duplication of large tensors/structures,
   - improve cleanup so that temporary objects do not stay alive longer than needed,
   - revisit GPU-side scheduling in Phase 4, for example with layer-aware chunking so that shallow and deep layers are not mixed in a way that creates poorly balanced batches,
   - explore automatic scaling heuristics that try to use most of the available VRAM without crossing the limit (similar in spirit to a target-VRAM policy used in systems like vLLM).

3. **Investigate the prompt-94 anomaly more deeply.**
   In particular, compare the Phase-4 feature sets / graph overlap between Ascend and Cardinal to determine why nominally identical runs diverge so strongly. Right now prompt 94 looks like a stable cross-cluster anomaly rather than random noise, so understanding it is important both for performance and for trustworthiness of the tracing procedure.

4. **Explore cross-run caching for repeated-prefix traces.**
   For continuous traces or repeated traces that share a long prefix, it may be possible to cache at least the forward pass on the prefix and potentially some carefully selected backward-pass information. If this works, it could reduce repeated work substantially on workloads where many runs share the same initial context.

5. **Test early sparsification after Phases 0-2.**
   Since Phases 3 and 4 dominate runtime in nearly every configuration, an attractive direction is to cut down the total number of features before those phases. The problem is that early sparsification can reduce trace quality, so this would require a more rigorous evaluation of how much pruning is acceptable and what mathematical/empirical guarantees I can give.

6. **Consolidate practical recommendations for exact tracing on OSC.**
   The final report should include a clear summary of which settings are recommended for short prompts, long prompts, and late-prefix stress cases.

7. **Optionally reconnect to the original motivation.**
   If time permits, I would like to run at least a small downstream tracing experiment using the optimized settings, to show that the systems work directly enables the interpretability application that originally motivated the project.

Because this is a team project, I also now have a clearer way to divide the remaining work:

- **I** will continue to own the core tracing pipeline, benchmark analysis, and the prompt-94 investigation.
- **Jay** can take on cross-run caching experiments, especially repeated-prefix reuse ideas where forward-pass information may be reusable across related traces.
- **Jayden** can take on sparsification experiments and the evaluation side of that work, since early sparsification is only useful if we can quantify the quality/performance tradeoff carefully.

Earlier in the semester, that division was hard to make concrete because the underlying tracing setup was still unstable. Now that I have the benchmark/extraction pipeline working and can reliably generate exact traces in at least part of the configuration space, these remaining tasks are much more realistic to parallelize across the team.

The main risk is that very large exact runs are still expensive in both time and host RAM. But compared to the proposal stage, I now have a much better understanding of where the bottlenecks are and what configurations are realistic.

---

## 8. Takeaway

The midpoint result of this project is not yet a final scientific claim about GSM8K reasoning behavior. Instead, it is a systems result:

> exact circuit tracing is feasible on OSC-scale hardware, but only within a carefully chosen operating regime, and the main determinants of feasibility are active feature count, host RAM, batch size, chunk size, and decoder-cache policy.

I think this is a meaningful and defensible midpoint contribution. It explains the project pivot, shows concrete progress, provides preliminary empirical results, and gives a clear plan for what I will do next.
