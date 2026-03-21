# Project Implementation Plan (Temporal Circuit Stability)

We have about **6 weeks left after the last homework**, so the goal is roughly:

- **Weeks 1–2:** pipeline + graph extraction + temporal metrics
- **Weeks 3–4:** data collection + verifier experiments

message.txt
6 KB
﻿
# Project Implementation Plan (Temporal Circuit Stability)

We have about **6 weeks left after the last homework**, so the goal is roughly:

- **Weeks 1–2:** pipeline + graph extraction + temporal metrics
- **Weeks 3–4:** data collection + verifier experiments
- **Week 5:** analysis + figures
- **Week 6:** report + backup time

The main risk early is **getting circuit-tracer working**, so the first phase is focused on establishing a working pipeline on a very small number of examples.

---

# Week 1 — Pipeline Bootstrapping

### Goal

Make sure we can:

1. load GSM8K
2. generate answers with the model
3. extract attribution graphs with circuit-tracer

We only need this to work on **1–2 examples** at first.

---

## Andrei

Focus: **interpretability tooling + graph extraction**

Tasks:

- Install and test **circuit-tracer**
- Load **Gemma-3-1B-IT**
- Integrate **Gemma Scope transcoders**
- Run circuit-tracer on **1–2 GSM8K prompts**
- Save **per-token attribution graphs**

Also define the **artifact format** we will use for saved graphs.

Example output structure:

```
experiments/
   traces/
      prompt_001/
         completion_001/
            step_000.json
            step_001.json
            ...
```

Deliverable by end of week:

- successful graph extraction for at least **one full generation**

---

## Jay

Focus: **dataset + inference pipeline**

Tasks:

- Implement **GSM8K loader**
- Sample dataset splits:
    - 300 train questions
    - 60 dev
    - 50 test
- Create **prompt template**
- Implement **inference script**

The inference pipeline should output:

```
{
  prompt_id
  prompt
  completion_text
  parsed_final_answer
  token_logprobs
}
```

This pipeline should run **without circuit-tracer first**, so we can test everything easily.

Once Andrei's tracer pipeline is ready, this will be extended to generate traced runs.

Deliverable by end of week:

- script that generates completions for GSM8K prompts

---

## Jayden

Focus: **evaluation harness**

Tasks:

- Implement **answer parsing**
- Implement **correctness evaluation**
- Build **evaluation script**

Input:

```
generations.jsonl
```

Output:

```
{
  prompt_id
  completion_id
  predicted_answer
  ground_truth
  correct
}
```

Also implement basic metric utilities:

- AUROC
- AUPRC

Deliverable:

- script that evaluates correctness on generated completions

---

# Week 2 — Temporal Graph Metrics

### Goal

Convert raw graphs into **temporal stability features**.

---

## Andrei

Implement graph processing:

1. **Graph sparsification**
    - cumulative edge-mass threshold (α ≈ 0.95)
2. **Temporal metrics**
    - step-to-step overlap
    - weighted Jaccard
    - churn = 1 − overlap
3. **Stable-core detection**

Output per completion:

```
{
  avg_overlap
  avg_churn
  stable_core_size
  stable_core_mass
}
```

Deliverable:

script that converts raw graphs → **feature table**

---

## Jay

Tasks:

- run generation pipeline for **larger batch**
- collect completions for ~50–100 prompts
- integrate tracer pipeline once stable
- manage experiment runs

Maintain experiment folders like:

```
experiments/
   run_01/
      generations.jsonl
      traces/
```

Deliverable:

first **medium-size traced dataset**

---

## Jayden

Tasks:

- load temporal features
- merge with correctness labels
- perform **initial analysis**

Example checks:

- churn distribution (correct vs incorrect)
- overlap distribution
- simple scatter plots

Deliverable:

initial exploratory plots

---

# Week 3 — Verifier Training

### Goal

Train models that predict correctness from features.

---

Tasks:

Train three models:

1️⃣ **Black-box features**

- avg logprob
- entropy
- sequence length

2️⃣ **Temporal features**

- overlap
- churn
- stable-core metrics

3️⃣ **Combined features**

Models to try:

- Logistic regression
- Gradient boosting

Evaluate on:

- dev set
- held-out test set

Metrics:

- AUROC
- AUPRC

---

# Week 4 — Extended Experiments

Possible extensions:

- prompt-level aggregation
- cross-run stability metrics
- stability trajectory visualization
- deeper error analysis

Example:

identify cases that are **stable but wrong**

Deliverable:

final experiment results.

---

# Week 5 — Analysis + Figures

Tasks:

- generate plots for paper
- visualize temporal graph evolution
- analyze interesting examples

Possible figures:

- churn vs correctness
- stable-core size vs correctness
- temporal stability curves

Also begin writing report sections:

- methods
- experiments
- results

---

# Week 6 — Report + Buffer

Tasks:

- finalize report
- finalize figures
- polish evaluation
- backup time if experiments break

Everyone contributes to writing.

---

# Weekly Meeting Plan

Short weekly sync (~30 minutes):

1. progress updates
2. blockers
3. experiment results
4. next tasks
