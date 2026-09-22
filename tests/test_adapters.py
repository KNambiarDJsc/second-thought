from second_thought.adapters.custom import FunctionProvider
from second_thought.adapters.jev import JevProvider
from second_thought.capture import normalize


def test_jev_provider_name_matches_policy_blocklist():
    from second_thought.datasets.policy import TRAINING_EXPORT_BLOCKED_PROVIDERS

    assert JevProvider.name in TRAINING_EXPORT_BLOCKED_PROVIDERS


def test_jev_capture_response_is_a_pure_passthrough():
    provider = JevProvider()
    raw = {"model": "jev-1.13.0", "answers": {"q1": {"type": "noul", "noul": 0.5}}}
    assert provider.capture_response(raw) is raw


def test_jev_model_info():
    provider = JevProvider(model_version="jev-1.13.0")
    assert provider.model_info() == {"provider": "jev", "model_version": "jev-1.13.0"}


def test_function_provider_wraps_an_arbitrary_callable():
    def predict_fn(state, questions):
        return {
            "model": "my-classifier-v1",
            "answers": {"q1": {"type": "noul", "noul": 0.7, "confidence": 0.4}},
        }

    provider = FunctionProvider("my-classifier", predict_fn, model_info={"framework": "sklearn"})
    result = provider.predict("some input", {"q1": {"type": "noul", "instructions": "spam?"}})

    assert provider.name == "my-classifier"
    assert provider.model_info() == {"framework": "sklearn"}
    assert result["answers"]["q1"]["noul"] == 0.7

    # and it feeds straight into the same normalizer Laya/Jev use — no special-casing
    event = normalize(
        result,
        provider=provider.name,
        state="some input",
        questions={"q1": {"type": "noul", "instructions": "spam?"}},
    )
    assert event.predictions["q1"].noul == 0.7


def test_function_provider_default_model_info():
    provider = FunctionProvider("bare-classifier", lambda s, q: {"model": "x", "answers": {}})
    assert provider.model_info() == {"provider": "bare-classifier"}


def test_laya_import_error_is_actionable_when_not_installed():
    import importlib
    import sys

    laya_backup = sys.modules.pop("laya", None)
    sys.modules["laya"] = None  # forces ImportError on `import laya`
    try:
        module = importlib.import_module("second_thought.adapters.laya")
        importlib.reload(module)
        try:
            module.LayaProvider()
            raise AssertionError("expected ImportError")
        except ImportError as exc:
            assert "pip install" in str(exc)
    finally:
        del sys.modules["laya"]
        if laya_backup is not None:
            sys.modules["laya"] = laya_backup
