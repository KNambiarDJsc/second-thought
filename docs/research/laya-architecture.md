# Laya architecture (reverse-engineered from the actual code, not the README)

Source: `github.com/NandhaKishorM/laya`, cloned and inspected 2026-09-22. Apache-2.0. HF
checkpoints `convaiinnovations/laya` and `convaiinnovations/laya-multilingual` (ungated).

## Provenance note — read before depending on this

The GitHub repo shows `created_at: 2026-09-18` but already had 13,317 stars and 1,087 forks four
days later — not a plausible organic curve for a brand-new repo, and it benchmarks itself by name
against TypeSafe Jev in its own assets (`assets/laya_vs_jev.png`) and fine-tuning notebook. Taken
alone that's a red flag. Corroborating checks softened it considerably: the `convaiinnovations`
Hugging Face org has a genuine multi-year publishing history unrelated to Laya (38 models back to
2023, including medical-imaging and RL projects), and independent third-party coverage exists (AI
Weekly, the eesel AI blog, a separate community Node.js/ONNX port at `receptron/laya`, a dedicated
landing page). Working conclusion: treat it as a real, unusually fast viral launch rather than a
fabricated project — but pin the `laya` version, don't execute its training notebook unreviewed,
and don't repeat its self-reported benchmark numbers (7.8× faster than Jev, 0.766 vs. 0.735
accuracy, etc.) as independently verified anywhere in this project.

## Pipeline, as implemented

```
questions dict + state
  -> Agent._to_internal          (laya/agent.py:254)   normalizes question defs
  -> build_sequence               (laya/common.py:49)   tokenizes:
       [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
  -> DecisionModel.forward        (laya/common.py:105)  shared transformer encoder
                                                          + optional TransformerEncoder head
                                                          -> one logit per option, gathered
                                                             at each [MASK] position
  -> softmax / temperature bucket (laya/common.py:219)  per (qtype, option-count) bucket,
                                                          clamped to [0.5, 5.0] (common.py:232)
  -> typed answer dict            (laya/agent.py:337)
```

## The real native API

- `laya.load(model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None) -> Agent` — `laya/agent.py:376`
- `Agent.system_one(state, questions) -> dict` (aliased `predict`; `RLAgent = Agent`) — `laya/agent.py:266,370,373`
- `Router.route(state, questions=None, model=None, task=None, lang=None) -> RouteDecision` and `Router.predict(...)` (aliased `system_one`) — `laya/router.py:257,315`
- `predict_shortlist(agent, state, questions, embed_fn, k=20, **kwargs) -> dict`, `embed_fn_from_agent(agent, ...)` — `laya/shortlist.py:51,111`

## Output schema — exact fields, not paraphrased

Top level: `{"model": "laya-rl-agent", "answers": {...}, "usage": {"input_tokens": n, "output_tokens": 0}}`
(`laya/agent.py:364`). Per question id (`laya/agent.py:337-362`):

- `choice`: `{"type":"choice","choice":<key>,"probabilities":{key:prob,...},"confidence":float,"action":{"act_probability":float}}`
- `score`: `{"type":"score","score":<expected value>,"legend":{...},"probabilities":{"0":p,...},"confidence":float,"action":{...}}`
- `noul`: `{"type":"noul","noul":p_true,"confidence":max(p,1-p),"action":{...}}`

This is the same `choice`/`score`/`noul` vocabulary Jev's documented API uses, and the same
top-level `{"model", "answers", "usage"}` envelope — which is why
`second_thought.capture.normalize()` can handle both providers with one function
(`second_thought/capture.py`) rather than one normalizer per provider.

## One trap: `result["model"]` doesn't tell you which checkpoint answered

It's a hardcoded literal, `"laya-rl-agent"`, regardless of which of the three checkpoints
(root/English, `laya-multilingual`, `laya-typed-decisions`) actually produced the answer. The
`LayaProvider` adapter (`second_thought/adapters/laya.py`) captures the `model_id_or_path` you
actually loaded as `DecisionEvent.model_version` instead, since that's the field worth trusting.

## Fine-tuning: not a supported CLI, only a Kaggle notebook

There is no `train.py` or console-script entry point anywhere in `pyproject.toml`. The only
training artifact is `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`, which writes and
executes its own `train_ddp.py` inline. The underlying pieces (`DecisionModel`, `build_sequence`,
`proper_reward`, `collate_items`) are importable and reusable, but there's no documented, supported
`laya train ...` workflow to wrap — which is exactly why this project's Laya fine-tuning story
(`examples/laya_customer_service/`) is a documented reproduction guide, not a one-command wrapper
around functionality Laya itself doesn't expose as a stable API.

## Calibration internals

Confidence = `1 − normalized_entropy(probabilities)` (`confidence_from_probs`,
`laya/common.py:210`) — the same formula this project's `calibration.normalized_entropy`
reimplements independently, so it applies uniformly regardless of which provider produced a
decision. Laya fits a per-`(question_type, option_count)` temperature bucket
(`laya/common.py:219`), clamped to `[0.5, 5.0]` because the shipped `choice:11+` bucket
over-sharpens (documented in a code comment). `ece_score` (15-bin ECE) ships as a diagnostic in
`laya/common.py`.

## What to log post-hoc, without touching Laya internals

Since `system_one`/`predict`/`Router.predict` are plain synchronous calls, wrap the call site: log
the exact `state`/`questions` passed in, the full returned dict, `usage.input_tokens`, the
`routing` dict if using `Router`, the `shortlist` dict if using `predict_shortlist`, wall-clock
latency, and the actual `model_id_or_path`/checkpoint loaded (not `result["model"]`). This is
exactly what `second_thought.capture.normalize()` + `LayaProvider` do.
