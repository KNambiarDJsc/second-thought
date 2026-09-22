"""Tests for the open typed-decision wire-shape conformance checker.

Response shapes below are drawn from real inspection, not invented: the
`choice` example matches the actual live Laya response captured in
examples/laya_customer_service/results/report.json; the `noul` and `score`
shapes match second_thought/schema.py's own field definitions and
docs/research/system-one-technical-report.md's documented field table.
"""

from second_thought.protocol import json_schema, validate_response

VALID_CHOICE_RESPONSE = {
    "model": "laya-rl-agent",
    "answers": {
        "action": {
            "type": "choice",
            "choice": "replace",
            "probabilities": {"refund": 0.1339, "replace": 0.6002, "escalate": 0.216, "ignore": 0.05},
            "confidence": 0.238,
        }
    },
    "usage": {"input_tokens": 44, "output_tokens": 0},
}

VALID_NOUL_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {"is_spam": {"type": "noul", "noul": 0.92}},
}

VALID_SCORE_RESPONSE = {
    "model": "some-model",
    "answers": {
        "urgency": {
            "type": "score",
            "score": 3.4,
            "probabilities": {"1": 0.05, "2": 0.1, "3": 0.3, "4": 0.4, "5": 0.15},
            "confidence": 0.6,
            "legend": {"1": "not urgent", "5": "critical"},
        }
    },
}


def test_valid_choice_response_conforms():
    result = validate_response(VALID_CHOICE_RESPONSE)
    assert result.ok
    assert result.errors == []
    assert result.warnings == []


def test_valid_noul_response_conforms():
    result = validate_response(VALID_NOUL_RESPONSE)
    assert result.ok
    assert result.errors == []


def test_valid_score_response_conforms():
    result = validate_response(VALID_SCORE_RESPONSE)
    assert result.ok
    assert result.errors == []


def test_missing_model_field_is_an_error():
    result = validate_response({"answers": {}})
    assert not result.ok
    assert any("model" in e for e in result.errors)


def test_choice_answer_missing_probabilities_is_an_error():
    raw = {
        "model": "m",
        "answers": {"q1": {"type": "choice", "choice": "a"}},
    }
    result = validate_response(raw)
    assert not result.ok
    assert any("probabilities" in e for e in result.errors)


def test_noul_answer_missing_the_noul_field_is_an_error():
    raw = {"model": "m", "answers": {"q1": {"type": "noul"}}}
    result = validate_response(raw)
    assert not result.ok
    assert any("noul" in e for e in result.errors)


def test_noul_with_confidence_is_a_warning_not_an_error():
    raw = {"model": "m", "answers": {"q1": {"type": "noul", "noul": 0.5, "confidence": 0.9}}}
    result = validate_response(raw)
    assert result.ok
    assert any("confidence" in w for w in result.warnings)


def test_probabilities_not_summing_to_one_is_a_warning():
    raw = {
        "model": "m",
        "answers": {"q1": {"type": "choice", "choice": "a", "probabilities": {"a": 0.1, "b": 0.1}}},
    }
    result = validate_response(raw)
    assert result.ok
    assert any("sum to" in w for w in result.warnings)


def test_unknown_extra_fields_are_allowed_not_rejected():
    """OpenJev documents extensions (images, steps, samples, think) beyond the
    base shape -- a conformant response with vendor extras must still pass."""
    raw = {
        "model": "m",
        "answers": {"q1": {"type": "noul", "noul": 0.5, "think": "reasoning trace"}},
        "steps": 4,
    }
    result = validate_response(raw)
    assert result.ok


def test_json_schema_is_generated_and_has_the_three_question_types():
    schema = json_schema()
    assert "$defs" in schema or "properties" in schema
    schema_str = str(schema)
    assert "choice" in schema_str
    assert "score" in schema_str
    assert "noul" in schema_str
