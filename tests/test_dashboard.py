"""Tests for the dashboard's HTTP layer, in-process (no live server, no network).

Every assertion here is really testing that the thin HTTP layer correctly
wires request/response JSON onto second_thought's own already-tested
functions (Store, select, correct/accept/abstain, evaluate, drift) — not
re-testing that logic itself.
"""

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from second_thought import Store
from second_thought.capture import capture
from second_thought.dashboard.app import create_app

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "What should we do?",
        "criteria": ["refund", "replace"],
    }
}


def _seeded_store(path):
    store = Store(path)
    for i in range(5):
        raw = {
            "model": "m",
            "answers": {
                "action": {
                    "type": "choice", "choice": "refund",
                    "probabilities": {"refund": 0.6, "replace": 0.4}, "confidence": 0.2,
                }
            },
        }
        capture(store, raw, provider="laya", state=f"ticket {i}", questions=QUESTIONS,
                workflow="support", strict=True)
    store.close()


def _client(tmp_path):
    db = str(tmp_path / "s.db")
    _seeded_store(db)
    app = create_app(db)
    return TestClient(app)


def test_summary_reports_totals_and_breakdowns(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_events"] == 5
    assert data["corrected_events"] == 0
    assert data["by_workflow"] == {"support": 5}
    assert data["by_provider"] == {"laya": 5}


def test_queue_returns_uncorrected_candidates_ranked_by_score(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/queue?budget=3")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 3
    assert all(not i["corrected"] for i in items)
    assert all(i["question_id"] == "action" for i in items)
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True)


def test_correct_then_disappears_from_queue(tmp_path):
    client = _client(tmp_path)
    items = client.get("/api/queue?budget=5").json()
    target = items[0]

    resp = client.post("/api/correct", json={
        "event_id": target["event_id"], "question_id": "action",
        "value": "replace", "reviewer": "tester",
    })
    assert resp.status_code == 200
    assert resp.json()["corrected"] is True

    remaining = client.get("/api/queue?budget=5").json()
    assert target["event_id"] not in {i["event_id"] for i in remaining}


def test_accept_records_the_predicted_value(tmp_path):
    client = _client(tmp_path)
    items = client.get("/api/queue?budget=1").json()
    target = items[0]
    resp = client.post("/api/accept", json={"event_id": target["event_id"], "question_id": "action"})
    assert resp.status_code == 200
    assert resp.json()["corrected"] is True


def test_correct_unknown_event_returns_404(tmp_path):
    client = _client(tmp_path)
    resp = client.post("/api/correct", json={
        "event_id": "does-not-exist", "question_id": "action", "value": "refund",
    })
    assert resp.status_code == 404


def test_correct_unknown_question_returns_404_not_500(tmp_path):
    client = _client(tmp_path)
    items = client.get("/api/queue?budget=1").json()
    target = items[0]
    resp = client.post("/api/correct", json={
        "event_id": target["event_id"], "question_id": "no-such-question", "value": "x",
    })
    assert resp.status_code == 404


def test_drift_endpoint_returns_a_report_shape(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/drift?metric=ece")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metric"] == "ece"
    assert "points" in data


def test_drift_endpoint_rejects_unknown_metric_with_400_not_500(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/api/drift?metric=not_a_real_metric")
    assert resp.status_code == 400


def test_index_page_is_served(tmp_path):
    client = _client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Second Thought" in resp.text


def test_create_app_without_fastapi_raises_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "fastapi":
            raise ImportError("no fastapi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="dashboard"):
        create_app(":memory:")
