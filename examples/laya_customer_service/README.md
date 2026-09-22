# Reproducing the core loop against real Laya

**Steps 1–3 and 5 below have now been run for real** (2026-09-22, one RTX 4090, `laya==0.3.5`,
`convaiinnovations/laya-typed-decisions`) — see [`results/report.md`](results/report.md) and the
runnable script, [`run.py`](run.py). 87.5% accuracy on 24 live-inferred tickets; uncertainty-based
`select()` caught 3/3 real mispredictions within the first third of the review queue; and a real,
non-synthetic finding — the run's ECE (0.62) corroborates a load-time warning Laya itself emits
about out-of-range calibration temperatures. Step 4, actual fine-tuning, is **not** covered by
`run.py` and remains unexecuted: Laya ships no supported fine-tuning CLI, only a Kaggle notebook
(see step 4 below), which isn't a turnkey automatable pod script the way steps 1–3+5 are. If you
run step 4, please open a PR with the same rigor — real hardware, real Laya version, real numbers,
not hand-edited ones — that `results/report.md` already follows.

## 1. Install

```
pip install "second-thought[laya]"
```

## 2. Capture decisions from Laya on your own data

```python
from second_thought import Store, capture
from second_thought.adapters import LayaProvider

provider = LayaProvider("convaiinnovations/laya-typed-decisions")
store = Store(".secondthought/store.db")

questions = {
    "action": {
        "type": "choice",
        "instructions": "What should we do with this support ticket?",
        "criteria": ["refund", "replace", "escalate", "ignore"],
    }
}

for ticket in your_tickets:  # bring your own labeled-or-unlabeled data
    result = provider.predict(ticket.text, questions)
    capture(store, result, provider="laya", state=ticket.text, questions=questions,
            workflow="customer_service", model_version=provider.model_id_or_path)
```

## 3. Select and correct

```
secondthought select --budget 100 --workflow customer_service
secondthought review <event_id> action correct --value refund --reviewer you@example.com
```

Or script it with `second_thought.select()` / `second_thought.correct()` directly if you're
reviewing programmatically against historical ground truth rather than by hand.

## 4. Export and fine-tune

```
secondthought export --out dataset/ --provider laya
```

Laya itself does not ship a supported fine-tuning CLI (see
`docs/research/laya-architecture.md` — only a Kaggle notebook exists in the Laya repo). Use
`dataset/corrections.train.jsonl` as the input to that notebook's data-loading cell, substituting
it for whatever dataset the notebook expects, and follow Laya's own fine-tuning instructions from
there.

## 5. Evaluate v1 vs v2

```
secondthought evaluate --provider laya   # before, against corrections.test.jsonl-equivalent decisions
# ... fine-tune, get a new checkpoint, re-run the same held-out set through the new checkpoint ...
secondthought evaluate --provider laya   # after
```

Compare accuracy / Brier / log loss / ECE between runs. Report both numbers, not just the
improvement — a Second Thought report that hides the baseline isn't trustworthy.
