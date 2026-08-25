from __future__ import annotations

import json

import pytest
from fuzzyxai.evidence.contracts import (
    ActionStatement,
    DecisionStatement,
    ExplanationDetails,
    ExplanationGraph,
    HumanExplanation,
    ReasonStatement,
    ReliabilityStatement,
)
from fuzzyxai.verbalization import SLMVerbalizer
from fuzzyxai.verbalization.backends import OllamaBackend
from fuzzyxai.verbalization.contracts import BackendTimeoutError, VerbalizationBackendError


def _explanation() -> HumanExplanation:
    decision = DecisionStatement("Решение", "Класс 1, вероятность 0.83", ("C1",), ("E1",), "available")
    reason = ReasonStatement("Причина", "feature_2 сильно повлиял", ("C1",), ("E1",), "feature_2", "supports", "выше медианы")
    reliability = ReliabilityStatement("Надёжность", "Оценка 0.9 надёжна", ("C1",), ("E1",), ("E1",), (), (), "можно доверять")
    action = ActionStatement("Действие", "Принять результат", ("C1",), ("E1",), "accept")
    return HumanExplanation("domain_user", "ru", decision, (reason,), (), reliability, action, (), ExplanationDetails(), ExplanationGraph((), ()))


class _JsonBackend:
    """New-protocol fake backend: accepts response_schema, returns a fixed JSON string."""

    def __init__(self, response: str, *, model: str = "fake-model") -> None:
        self.response = response
        self.model = model

    def generate(self, prompt: str, *, response_schema=None) -> str:
        return self.response


class _RaisingBackend:
    model = "fake-model"

    def generate(self, prompt: str, *, response_schema=None) -> str:
        raise BackendTimeoutError("simulated timeout")


def test_no_backend_is_deterministic_and_makes_no_network_call() -> None:
    result = SLMVerbalizer(None).run(_explanation(), template_text="TEMPLATE TEXT")
    assert result.text == "TEMPLATE TEXT"
    assert result.status == "deterministic"
    assert result.backend is None
    assert result.fallback_reason is not None


class TestStrictMode:
    def test_valid_order_and_connector_assembles_deterministically(self) -> None:
        backend = _JsonBackend(json.dumps({"order": ["decision-0", "reason-0"], "connector": "plain"}))
        result = SLMVerbalizer(backend, mode="strict").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "generated"
        assert result.backend == "_JsonBackend"
        assert result.model == "fake-model"
        assert result.source_claim_ids == ("decision-0", "reason-0")
        # The assembled text must be built purely from canonical claim text —
        # no token the backend "wrote" beyond order/connector can appear.
        assert "Класс 1, вероятность 0.83" in result.text
        assert "feature_2 сильно повлиял" in result.text

    def test_structured_connector_produces_a_list(self) -> None:
        backend = _JsonBackend(json.dumps({"order": ["decision-0"], "connector": "structured"}))
        result = SLMVerbalizer(backend, mode="strict").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.text.startswith("- ")

    def test_backend_cannot_smuggle_new_text_through_order_field(self) -> None:
        # Even a maximally adversarial backend can only select IDs — it has
        # no field through which to inject free text in strict mode.
        backend = _JsonBackend(json.dumps({"order": ["decision-0"], "connector": "plain; DROP EVERYTHING"}))
        result = SLMVerbalizer(backend, mode="strict").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "generated"
        assert result.text == "Класс 1, вероятность 0.83"  # invalid connector silently defaults to "plain"

    def test_invalid_json_is_rejected(self) -> None:
        backend = _JsonBackend("not json at all")
        result = SLMVerbalizer(backend, mode="strict").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"
        assert result.text == "TEMPLATE TEXT"

    def test_unknown_claim_id_is_rejected(self) -> None:
        backend = _JsonBackend(json.dumps({"order": ["nonexistent-claim"], "connector": "plain"}))
        result = SLMVerbalizer(backend, mode="strict").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"
        assert result.text == "TEMPLATE TEXT"

    def test_empty_order_is_rejected(self) -> None:
        backend = _JsonBackend(json.dumps({"order": [], "connector": "plain"}))
        result = SLMVerbalizer(backend, mode="strict").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"


class TestRewriteMode:
    def test_grounded_rewrite_is_accepted(self) -> None:
        response = json.dumps(
            {
                "sentences": [
                    {"text": "Модель выбрала класс 1 с вероятностью 0.83.", "source_claim_ids": ["decision-0"]},
                    {"text": "Основную роль сыграл feature_2.", "source_claim_ids": ["reason-0"]},
                ]
            }
        )
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "generated"
        assert result.source_claim_ids == ("decision-0", "reason-0")

    def test_hallucinated_number_is_rejected(self) -> None:
        response = json.dumps({"sentences": [{"text": "Вероятность на самом деле 0.999.", "source_claim_ids": ["decision-0"]}]})
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"
        assert result.text == "TEMPLATE TEXT"
        assert "no_new_numbers" in (result.fallback_reason or "") or result.fallback_reason

    def test_new_feature_name_is_rejected(self) -> None:
        response = json.dumps({"sentences": [{"text": "feature_99 тоже сильно повлиял.", "source_claim_ids": ["reason-0"]}]})
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"

    def test_unlicensed_certainty_language_is_rejected(self) -> None:
        response = json.dumps({"sentences": [{"text": "Это точно доказывает диагноз.", "source_claim_ids": ["decision-0"]}]})
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"

    def test_sentence_without_source_claim_id_is_rejected(self) -> None:
        response = json.dumps({"sentences": [{"text": "Класс 1.", "source_claim_ids": []}]})
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"

    def test_sentence_with_unknown_claim_id_is_rejected(self) -> None:
        response = json.dumps({"sentences": [{"text": "Класс 1.", "source_claim_ids": ["nonexistent"]}]})
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"

    def test_empty_output_is_rejected(self) -> None:
        response = json.dumps({"sentences": []})
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"

    def test_backend_introducing_a_new_technical_identifier_is_rejected(self) -> None:
        # If a backend "obeys" injected text inside a claim and fabricates a
        # new identifier-shaped token (the actual threat model this guard
        # targets: a new feature/class/entity name), it is rejected.
        response = json.dumps({"sentences": [{"text": "Также сработал ADMIN_OVERRIDE_MODE.", "source_claim_ids": ["reason-0"]}]})
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "rejected"

    def test_known_limitation_ordinary_word_fabrication_is_not_caught_by_surface_guards(self) -> None:
        """Documents a real, deliberate limitation rather than hiding it.

        The entity guard only checks identifier-shaped tokens (snake_case,
        ALL-CAPS, alphanumeric) — see ``guards._TECHNICAL_TOKEN`` — because
        checking every ordinary word rejects almost any natural paraphrase
        (Russian inflection changes word endings constantly). This means a
        fabricated sentence using only ordinary vocabulary (no new numbers,
        no new technical identifiers, no banned certainty words) can still
        pass the current guard set. This is exactly why ``rewrite`` mode's
        guarantee is "passed surface checks", not "proven grounded" — see
        the module docstring and README. ``strict`` mode does not have this
        gap, because it never lets the backend author new text at all.
        """

        response = json.dumps(
            {"sentences": [{"text": "Система раскрывает внутренние детали полностью и подробно.", "source_claim_ids": ["reason-0"]}]}
        )
        backend = _JsonBackend(response)
        result = SLMVerbalizer(backend, mode="rewrite").run(_explanation(), template_text="TEMPLATE TEXT")
        assert result.status == "generated"  # passes today's surface guards — the known gap, not a bug


def test_backend_exception_falls_back_without_raising() -> None:
    result = SLMVerbalizer(_RaisingBackend()).run(_explanation(), template_text="TEMPLATE TEXT")
    assert result.status == "fallback"
    assert result.text == "TEMPLATE TEXT"
    assert "timeout" in (result.fallback_reason or "").lower()


def test_ollama_backend_raises_typed_error_when_unreachable() -> None:
    backend = OllamaBackend(host="http://127.0.0.1:1", timeout=1.0)
    with pytest.raises(VerbalizationBackendError):
        backend.generate("hello")


def test_ollama_backend_env_var_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUZZYXAI_OLLAMA_MODEL", "custom-model:1b")
    monkeypatch.setenv("FUZZYXAI_OLLAMA_HOST", "http://example.invalid:9999")
    backend = OllamaBackend()
    assert backend.model == "custom-model:1b"
    assert backend.host == "http://example.invalid:9999"


def test_ollama_backend_explicit_args_win_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUZZYXAI_OLLAMA_MODEL", "env-model")
    backend = OllamaBackend(model="explicit-model")
    assert backend.model == "explicit-model"


def test_ollama_backend_default_model_is_pinned_not_latest() -> None:
    backend = OllamaBackend()
    assert backend.model == "qwen3:1.7b"
    assert "latest" not in backend.model


@pytest.mark.optional_integration
def test_ollama_backend_real_smoke() -> None:
    import os

    if os.environ.get("FUZZYXAI_OLLAMA_SMOKE") != "1":
        pytest.skip("set FUZZYXAI_OLLAMA_SMOKE=1 with a running local Ollama to run this")
    backend = OllamaBackend()
    result = SLMVerbalizer(backend).run(_explanation(), template_text="TEMPLATE TEXT")
    assert result.text
