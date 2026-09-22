"""Second Thought is not a Laya-specific tool. This proves it with a model that
is neither Laya nor Jev: a plain TF-IDF + logistic regression text classifier,
wrapped as a System One provider in about ten lines via `FunctionProvider`.

    python examples/generic_provider/run.py

No GPU, no download, no dependency on either named adapter. If your model can
answer a typed question with a probability distribution, it can use this SDK.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from data import TICKETS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from second_thought import Store, accept, capture, correct, evaluate, export, select
from second_thought.adapters import FunctionProvider
from second_thought.correction import predicted_value

CATEGORIES = ["billing", "technical", "account"]
QUESTIONS = {
    "category": {
        "type": "choice",
        "instructions": "What team should triage this support ticket?",
        "criteria": CATEGORIES,
    }
}


def make_provider(vectorizer: TfidfVectorizer, clf: LogisticRegression) -> FunctionProvider:
    def predict_fn(state: str, questions: dict) -> dict:
        x = vectorizer.transform([state])
        proba = clf.predict_proba(x)[0]
        probabilities = dict(zip(clf.classes_, proba.tolist(), strict=True))
        top = max(probabilities, key=lambda k: probabilities[k])
        return {
            "model": "demo-ticket-classifier",
            "answers": {
                "category": {
                    "type": "choice",
                    "choice": top,
                    "probabilities": probabilities,
                    "confidence": max(probabilities.values()),
                }
            },
            "usage": {"input_tokens": len(state.split())},
        }

    return FunctionProvider(
        "demo-ticket-classifier", predict_fn, model_info={"vectorizer": "tfidf", "classes": CATEGORIES}
    )


def main() -> None:
    texts = [t for t, _ in TICKETS]
    labels = [c for _, c in TICKETS]

    seed_texts, pool_texts, seed_labels, pool_labels = train_test_split(
        texts, labels, train_size=15, stratify=labels, random_state=0
    )

    vectorizer = TfidfVectorizer()
    x_seed = vectorizer.fit_transform(seed_texts)
    clf = LogisticRegression(max_iter=1000).fit(x_seed, seed_labels)

    provider = make_provider(vectorizer, clf)

    db_path = Path(tempfile.mkdtemp()) / "store.db"
    store = Store(db_path)

    for text in pool_texts:
        result = provider.predict(text, QUESTIONS)
        capture(
            store,
            result,
            provider=provider.name,
            state=text,
            questions=QUESTIONS,
            workflow="ticket_triage",
            model_version=provider.model_info()["vectorizer"],
        )

    events = list(store.query())
    budget = 8
    candidates = select(events, budget=budget)

    print(f"Top {budget} tickets by uncertainty (out of {len(events)} in the pool):\n")
    wrong_in_selection = 0
    for c in candidates:
        true_label = pool_labels[pool_texts.index(c.event.input_state)]
        predicted = predicted_value(c.event.predictions[c.question_id])
        was_wrong = predicted != true_label
        wrong_in_selection += was_wrong
        correct(store, c.event.id, c.question_id, true_label, reviewer="demo-reviewer")
        marker = "WRONG" if was_wrong else "right"
        print(
            f"  [{marker}] score={c.score:.3f} predicted={predicted!r} truth={true_label!r} "
            f"-- {c.event.input_state[:60]}..."
        )

    # Accept the model's own answer on a random sample of the *rest* of the
    # pool as a baseline correction rate for comparison.
    remaining = [e for e in events if not e.has_correction("category")]
    wrong_in_random = 0
    for event in remaining[:budget]:
        true_label = pool_labels[pool_texts.index(event.input_state)]
        predicted = predicted_value(event.predictions["category"])
        wrong_in_random += predicted != true_label
        correct(store, event.id, "category", true_label, reviewer="demo-reviewer")
    for event in remaining[budget:]:
        accept(store, event.id, "category", reviewer="demo-reviewer")

    print(
        f"\nModel was wrong on {wrong_in_selection}/{budget} of the uncertainty-selected "
        f"tickets, vs {wrong_in_random}/{min(budget, len(remaining))} of the next "
        f"{min(budget, len(remaining))} tickets in pool order (a stand-in random sample)."
    )

    out_dir = Path(tempfile.mkdtemp())
    report = export(list(store.query()), out_dir)
    print(f"\nExported {report.written_records} corrected records to {out_dir}")

    eval_report = evaluate(list(store.query()))
    print(f"\nEvaluation over all {eval_report.n} corrected tickets: {eval_report}")


if __name__ == "__main__":
    main()
