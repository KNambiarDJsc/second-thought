# Reproducing the core loop against real Laya

This has **not been run** as part of building this repository — there was no GPU in the
environment it was built in. What follows is a precise, reproducible recipe, not a claim that it's
been verified end-to-end. If you run it, please open a PR with `results/report.md` from a real run
(random seed, hardware, and Laya version included) rather than hand-edited numbers.

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
