# Stability-Aware Hybrid Memory for Long-Context LLM Assistants

## Overview

This project explores a **hybrid memory system for long-context assistants**. The main idea is that an assistant should not always rely on a single memory source. Instead, it should maintain:

- **raw evidence memory** for precise details
- **compressed summary memory** for efficient context access

A learned controller is then used to decide whether the system should trust summary memory or fall back to raw evidence for answering a query.

The core research question is:

> **When is compressed memory safe to trust, and when should the system recover raw evidence instead?**

---

## Motivation

Long-context assistants often face a tradeoff:

- **Summaries** are efficient, but may lose important facts
- **Raw logs** preserve detail, but are expensive and noisy

This project builds a pipeline that retrieves from both memory types and learns a routing policy for deciding which memory source should guide the final answer.

---

## Project Stages

### Stage 1: LoCoMo Prototype
The first prototype was built on the **LoCoMo** benchmark.

This stage includes:
- preprocessing raw dialogue into chunked memory units
- building session-level summary memory
- linking summaries back to raw chunks through provenance
- retrieval over both raw and summary memory
- a rule-based controller baseline

### Stage 2: LongMemEval Extension
The pipeline was then extended to the larger **LongMemEval** benchmark.

This stage includes:
- preprocessing LongMemEval sessions into raw chunks
- building lightweight session summaries
- embedding-based retrieval over raw and summary memory
- learned controller experiments on a larger dataset

---

## Final System

The final system consists of:

1. **Raw memory construction**
2. **Summary memory construction**
3. **Embedding-based retrieval**
4. **Answer generation from both memory sources**
5. **A learned controller** that decides:
   - use summary memory
   - or use raw evidence

---

## Controller Experiments

I experimented with multiple controller designs:

### 1. Rule-Based Controller
A simple baseline that prefers raw evidence for precise or temporal questions.

### 2. Wide & Deep Controller
A controller that separates:
- rule-like features in a wide branch
- nonlinear interactions in a deep branch

In the current prototype, this controller tended to become overly conservative and collapse into an **always-raw** policy.

### 3. MLP Controller
A simpler learned controller using an MLP over retrieval and routing features.

This controller produced more balanced behavior than the Wide & Deep controller in the LongMemEval experiments, making it the final controller I would present from the current prototype.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                 # excluded from Git due to size
│   └── processed/           # excluded from Git due to size
├── outputs/                 # excluded from Git due to size
├── src/
│   ├── preprocess.py
│   ├── build_summary_windows.py
│   ├── embed_retriever.py
│   ├── controller.py
│   ├── run_day3_pipeline.py
│   ├── build_controller_dataset_wd.py
│   ├── train_wide_deep_controller.py
│   ├── wide_deep_controller.py
│   ├── run_day4_pipeline_wd.py
│   ├── evaluate_results.py
│   └── bigger/
│       ├── preprocess_longmemeval.py
│       ├── build_longmemeval_summaries.py
│       ├── build_qa_dataset_longmemeval.py
│       ├── embed_retriever_lme.py
│       ├── answerer.py
│       ├── run_day4_pipeline_wd_lme.py
│       ├── evaluate_lme_results.py
│       └── mlp/
│           ├── build_controller_dataset_mlp_lme.py
│           ├── train_mlp_controller_lme.py
│           ├── mlp_controller_lme.py
│           ├── run_day4_pipeline_mlp_lme.py
│           └── evaluate_lme_results_mlp.py
├── README.md
└── requirements.txt