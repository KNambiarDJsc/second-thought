"""Reproduce the real capture -> select -> correct -> export -> evaluate loop against
real Laya, on a GPU.

    pip install "second-thought[laya]"
    python examples/laya_customer_service/run.py

This is steps 1-3 and 5 of this directory's README, executed for real (not fine-tuning
-- see the README for why step 4, actual fine-tuning, isn't a turnkey automated script:
Laya has no supported fine-tuning CLI, only a Kaggle notebook). First real run:
results/report.md, 2026-09-22, RTX 4090 (RunPod), laya==0.3.5,
convaiinnovations/laya-typed-decisions.

The 24 tickets below are synthetic example text (same spirit as
examples/generic_provider/data.py) -- not a real production queue. What's real: live
Laya inference, real capture into the actual Store, real uncertainty-based selection,
real human-judgment corrections, a real dataset export, and real calibration metrics
computed by the actual shipped evaluate() -- against the live model, not synthetic
sklearn data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from second_thought import Store, correct, export, select
from second_thought.adapters import LayaProvider
from second_thought.capture import capture
from second_thought.evaluation import evaluate

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "What should we do with this support ticket?",
        "criteria": ["refund", "replace", "escalate", "ignore"],
    }
}

# (ticket text, ground-truth judgment used as the reviewer's correction)
TICKETS: list[tuple[str, str]] = [
    ("My order arrived broken, the screen is cracked. I want a replacement, not a refund.", "replace"),
    ("This is the third time this month my package has been late. I'm done, give me my money back.", "refund"),
    ("Your product literally set off sparks when I plugged it in. This is a fire hazard, I need this escalated immediately.", "escalate"),
    ("hey just wanted to say thanks for the quick shipping!! no issues at all :)", "ignore"),
    ("I ordered a size medium but received a size small. Please send the correct size.", "replace"),
    ("I've been charged twice for the same order. Please refund the duplicate charge.", "refund"),
    ("Your customer service rep was incredibly rude to me on the phone yesterday. I want to speak to a manager.", "escalate"),
    ("Just leaving a review, love the product, 5 stars!", "ignore"),
    ("The item I received doesn't match the description at all - I ordered blue and got red.", "replace"),
    ("I cancelled my subscription two weeks ago and you're still billing me. Refund immediately.", "refund"),
    ("I'm a lawyer and I will be pursuing legal action if this data breach isn't addressed within 24 hours.", "escalate"),
    ("No message, just an empty ticket submitted by accident.", "ignore"),
    ("The blender motor burned out after one use. It's clearly defective, I need a new one sent.", "replace"),
    ("I never received my order and tracking shows it was delivered to the wrong address. I want my money back.", "refund"),
    ("This is the fifth email I've sent with no response. I am extremely frustrated and considering a chargeback and a BBB complaint.", "escalate"),
    ("Just confirming my address update went through, thanks!", "ignore"),
    ("The replacement part you sent last week doesn't fit either. This is the second wrong part in a row.", "replace"),
    ("I was overcharged shipping fees that weren't disclosed at checkout. Please refund the difference.", "refund"),
    ("Your app crashed and deleted all my saved data with no backup option. I need this bumped to engineering right away.", "escalate"),
    ("Quick question - do you ship to Canada? Not a complaint, just curious.", "ignore"),
    ("The charger cable that came with my laptop is frayed and unsafe to use. Please send a working one.", "replace"),
    ("I was double-billed on my last invoice, this needs to be refunded this week.", "refund"),
    ("Someone at your company posted my private information publicly. This needs to go to legal/security right now.", "escalate"),
    ("Loved the unboxing experience, no issues to report.", "ignore"),
]


def main() -> None:
    t_load = time.time()
    provider = LayaProvider("convaiinnovations/laya-typed-decisions", device="cuda")
    load_s = time.time() - t_load

    store = Store("/tmp/laya_live.db")
    latencies = []
    rows = []
    for text, gt in TICKETS:
        t0 = time.time()
        raw = provider.predict(text, QUESTIONS)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)
        event = capture(
            store, raw, provider="laya", state=text, questions=QUESTIONS,
            workflow="customer_service_live", model_version=provider.model_id_or_path,
            latency_ms=latency_ms, strict=True,
        )
        assert event is not None
        correct(store, event.id, "action", gt, reviewer="karthik", note="ground truth for live eval")
        pred = event.predictions["action"]
        rows.append({
            "text": text, "laya_choice": pred.choice, "laya_confidence": pred.confidence,
            "ground_truth": gt, "correct": pred.choice == gt, "latency_ms": latency_ms,
        })

    all_events = list(store.query(workflow="customer_service_live"))
    report = evaluate(all_events)

    # Selection, on fresh uncorrected captures, to measure what Second Thought would
    # have surfaced to a human first -- before any corrections existed.
    store2 = Store("/tmp/laya_live_selection.db")
    for text, _gt in TICKETS:
        raw = provider.predict(text, QUESTIONS)
        capture(store2, raw, provider="laya", state=text, questions=QUESTIONS,
                workflow="customer_service_live", strict=True)
    budget = 8
    candidates = select(list(store2.query()), budget=budget)
    selected_texts = {c.event.input_state for c in candidates}
    total_wrong = sum(1 for r in rows if not r["correct"])
    wrong_caught = sum(1 for r in rows if not r["correct"] and r["text"] in selected_texts)

    export_report = export(list(store.query()), "/tmp/laya_live_dataset", dataset_name="laya_live")

    out = {
        "hardware": "NVIDIA GeForce RTX 4090 (RunPod, secure cloud)",
        "laya_package_version": __import__("laya").__version__,
        "laya_checkpoint": provider.model_id_or_path,
        "model_load_seconds": load_s,
        "n_tickets": len(TICKETS),
        "accuracy": report.accuracy,
        "mean_brier": report.mean_brier,
        "mean_log_loss": report.mean_log_loss,
        "ece": report.ece,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "selection_budget": budget,
        "total_mispredictions": total_wrong,
        "mispredictions_caught_by_selection": wrong_caught,
        "export": {
            "written_records": export_report.written_records,
            "split_counts": export_report.split_counts,
        },
        "rows": rows,
    }

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
