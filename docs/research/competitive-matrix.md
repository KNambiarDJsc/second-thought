# Competitive landscape

Researched 2026-09-22 against each tool's official docs/GitHub (not from memory — capabilities
change). One `docs.humansignal.com` fetch also returned an injected fake system-reminder block;
discarded, not a real instruction.

| Tool | Full probability distribution stored? | Typed/fixed-schema decisions? | Uncertainty/active-learning selection? | Calibration (Brier/log loss/ECE)? | Fine-tune-ready export from corrections? | Non-autoregressive/classifier support? | OSS SDK? |
|---|---|---|---|---|---|---|---|
| TensorZero | No (scalar feedback/demonstrations) | Partial (JSON-schema LLM output, not a classifier) | No | No | Yes (SFT export) | No | Yes |
| Braintrust | No evidence | No | No | No | Partial (generic export) | No | Partial (SDK OSS, platform proprietary) |
| Langfuse | No evidence | No | No | No | Yes (fine-tune export) | No | Yes |
| Arize (Phoenix) | No evidence | No (LLM/RAG focus) | No | Inconclusive | No evidence | No | Yes |
| WhyLabs | No evidence | Partial (classifier drift profiling) | No | Not confirmed | No | Partial (profiling only) | Yes |
| Galileo | No evidence | No | No | No | No | No | No (except unrelated OSS component) |
| Evidently AI | Partial (`predict_proba`-based metrics) | Yes | No | Partial (LogLoss only) | No | Yes — most classifier-native of the monitoring tools | Yes |
| W&B Weave | No evidence | No | No | No | Partial (generic artifacts) | No | Yes |
| Humanloop | No evidence | No | No | No | Yes | No (proprietary; sunset post-acquisition) |
| Label Studio | Unclear | Yes | Yes (uncertainty-based queue) | No | No | Yes | Yes (CE) |
| Argilla | Unclear | Yes | Yes (legacy modAL/small-text integration) | No | Partial (HF Datasets export) | Yes | Yes |
| Snorkel | Yes (label model) | Yes | Yes (Flow, commercial) | No | Yes (Flow) | Yes | Partial (lib OSS, Flow enterprise) |
| Cleanlab | Yes (`pred_probs` required) | Yes | Yes | No | Unclear | Yes | Yes |
| Labelbox | No evidence | Yes | Yes | No | Yes | Yes | No (client SDK only) |
| Scale AI | No evidence | Yes | Yes (Data Engine) | No (QA "calibration" ≠ statistical) | Yes | Yes | No (client SDK only) |
| OpenAI evals | No evidence | No | No | No | No | No | Yes |
| DeepEval | No evidence | Partial | No | No | Partial | No | Yes |
| Promptfoo | No evidence | Partial (classifier-as-grader only) | No | No | No | No | Yes |
| modAL / small-text / ALiPy | Runtime-only, no persistence | Yes | Yes — this is their core purpose | No | No | Yes | Yes |

## Synthesis

Coverage splits into three disjoint clusters, and none of them do all three things together:

1. **LLM observability/eval platforms** (TensorZero, Langfuse, Braintrust, Humanloop, ...) — strong
   on capture → correct → export → fine-tune *mechanics*, but built for free-text LLM completions,
   with no probability-distribution storage and no calibration metrics.
2. **Classic ML monitoring** (Evidently AI, WhyLabs, Arize AX) — the only cluster that's natively
   `predict_proba`-aware, but these are monitoring/testing tools, not correction-capture-to-dataset
   pipelines, and calibration support is inconsistent even here.
3. **Labeling/weak-supervision platforms** (Snorkel, Cleanlab, Label Studio, Argilla, Labelbox,
   Scale AI) — strongest on active-learning-style selection and fixed-schema decisions; Cleanlab
   and Snorkel are the only other tools that require/produce full per-class probability arrays.
   None do calibration analysis (Brier/log loss/ECE) as a first-class citizen.

**Is active-learning selection itself already solved?** Largely yes, as a library problem — modAL,
small-text, ALiPy, plus Cleanlab's and Snorkel's built-in versions, already implement mature
uncertainty/margin/entropy query strategies. Building a new query-strategy library from scratch
would be wasted effort, and this project doesn't: `second_thought/selection/strategies.py`
implements the standard ones plainly rather than inventing new math (see the "reject the ELV
formula" note in that module and in `wedge-and-mvp-spec.md`).

**The actual gap:** the specific combination — probability-distribution-aware storage + calibration
metrics + uncertainty-based selection + a fixed typed-decision schema + ready-to-fine-tune export —
end-to-end, for non-LLM structured decisions — isn't offered by any single tool surveyed. Every
tool with calibration-adjacent features stops at monitoring; every tool with a correction-to-dataset
loop is LLM-text-centric; every tool with mature active learning is a library, not an
infrastructure layer with durable storage and calibration reporting. That three-way gap is the
credible novelty claim — framed as gluing three separately-solved capabilities into one
typed-decision-native loop, not as inventing any one of them from scratch.
