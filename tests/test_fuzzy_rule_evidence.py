"""P11: generic fuzzy/rule adapter contract.

Any model can supply real rule-activation evidence (rule id, antecedent
terms with membership degrees, activation strength, conclusion) through a
plain-dict ``activated_rules`` channel on ``extract_local_evidence`` — not
tied to any specific ANFIS library. This must be rendered with genuine rule
semantics (activation strength, membership degrees), never collapsed into a
feature_contribution-style "+0.42" template.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pytest
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters.contracts_v2 import ExplanationContext, LocalModelEvidence
from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.model_v2 import ModelAdapterV2
from fuzzyxai.evidence import FuzzyRuleActivation, collect_fuzzy_rule_activations


def _gaussian(x: float, mean: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2)


class SimpleANFIS:
    TERMS: ClassVar[dict[str, dict[str, tuple[float, float]]]] = {
        "temperature": {"low": (20.0, 8.0), "high": (35.0, 8.0)},
        "pressure": {"low": (1.0, 0.4), "high": (2.0, 0.4)},
    }
    RULES: ClassVar[list[tuple[str, dict[str, str], int]]] = [
        ("R1", {"temperature": "low", "pressure": "low"}, 0),
        ("R2", {"temperature": "low", "pressure": "high"}, 0),
        ("R3", {"temperature": "high", "pressure": "low"}, 1),
        ("R4", {"temperature": "high", "pressure": "high"}, 1),
    ]

    def _activations(self, temperature: float, pressure: float) -> list[dict[str, Any]]:
        values = {"temperature": temperature, "pressure": pressure}
        memberships = {f: {t: _gaussian(values[f], m, s) for t, (m, s) in terms.items()} for f, terms in self.TERMS.items()}
        activations = []
        for rule_id, antecedents, consequent in self.RULES:
            strength = 1.0
            terms = []
            for feature, term in antecedents.items():
                degree = memberships[feature][term]
                strength *= degree
                terms.append({"feature": feature, "term": term, "membership_degree": degree, "feature_value": values[feature]})
            activations.append({"rule_id": rule_id, "terms": terms, "activation_strength": strength, "conclusion": str(consequent)})
        return activations

    def predict_one(self, temperature: float, pressure: float) -> tuple[int, list[dict[str, Any]]]:
        activations = self._activations(temperature, pressure)
        total = sum(a["activation_strength"] for a in activations) or 1e-9
        weighted = sum(a["activation_strength"] * float(a["conclusion"]) for a in activations) / total
        return round(weighted), activations

    def predict(self, X: Any) -> np.ndarray:
        rows = np.asarray(X)
        return np.array([self.predict_one(row[0], row[1])[0] for row in rows])


class ANFISLikeAdapter(ModelAdapterV2):
    adapter_id = "test_anfis_like_adapter"
    model_family = "fuzzy_rule_system"

    def predict(self, inputs: Any) -> ModelPrediction:
        rows = np.atleast_2d(np.asarray(inputs, dtype=float))
        predictions = self.model.predict(rows)
        return ModelPrediction(predictions=predictions.tolist(), probabilities=None, model_type="SimpleANFIS", adapter_id=self.adapter_id, metadata={"task_type": self.task_type.value})

    def extract_local_evidence(self, inputs: Any, prediction: ModelPrediction, context: ExplanationContext) -> LocalModelEvidence:
        del prediction, context
        rows = np.atleast_2d(np.asarray(inputs, dtype=float))
        _, activations = self.model.predict_one(rows[0][0], rows[0][1])
        return LocalModelEvidence(channels={"activated_rules": activations})


def _wrapped(temperature: float = 33.0, pressure: float = 1.9):
    model = SimpleANFIS()
    adapter = ANFISLikeAdapter(model, task="classification")
    fx = FuzzyXAI.wrap(model, adapter=adapter)
    native_prediction, _ = model.predict_one(temperature, pressure)
    result = fx.explain_one([temperature, pressure], feature_names=["temperature", "pressure"])
    return model, result, native_prediction


def test_native_prediction_matches() -> None:
    _, result, native_prediction = _wrapped()
    assert result.prediction.predictions[0] == native_prediction


def test_all_activated_rules_are_preserved_as_claims() -> None:
    _, result, _ = _wrapped()
    rule_claims = {c.subject_id for c in result.claims if c.claim_type == "fuzzy_rule"}
    assert rule_claims == {"R1", "R2", "R3", "R4"}


def test_membership_degrees_and_activation_strength_are_preserved() -> None:
    model, result, _ = _wrapped()
    _, native_activations = model.predict_one(33.0, 1.9)
    native_by_id = {a["rule_id"]: a for a in native_activations}
    for claim in result.claims:
        if claim.claim_type != "fuzzy_rule":
            continue
        native = native_by_id[claim.subject_id]
        assert claim.strength == pytest.approx(native["activation_strength"])
        assert claim.metric_value == pytest.approx(native["activation_strength"])
        for term in native["terms"]:
            assert f"{term['membership_degree']:.2f}" in claim.statement


def test_provenance_traces_to_rule_evidence_uniquely_per_rule() -> None:
    _, result, _ = _wrapped()
    rule_claims = [c for c in result.claims if c.claim_type == "fuzzy_rule"]
    refs = [c.evidence_refs for c in rule_claims]
    assert len(refs) == len(set(refs)), "each rule claim must have a distinct evidence_ref, not a shared/duplicated one"
    for claim in rule_claims:
        assert claim.evidence_refs == (f"fuzzy_rule:object_0:{claim.subject_id}",)


def test_summary_reflects_rule_semantics_not_a_feature_contribution_template() -> None:
    _, result, _ = _wrapped()
    text = result.summary()
    assert "активировано со степенью" in text
    assert "соответствует терму" in text
    # Must not use the linear feature_contribution wording for this evidence.
    assert "измеренный коэффициент" not in text
    # Bare technical rule ids (R1..R4) must not leak into domain_user text —
    # this collided with the existing TECHNICAL_TERMS \bR\d+\b guard and was
    # a real bug caught while building this feature.
    import re

    assert not re.search(r"\bR\d\b", text)


def test_strict_verbalizer_preserves_rule_terms_without_inventing_content() -> None:
    import json
    import re

    _, result, _ = _wrapped()

    class FakeStrictBackend:
        model = "test-fake"

        def generate(self, prompt: str, *, response_schema=None) -> str:
            del response_schema
            claim_ids = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
            return json.dumps({"order": claim_ids, "connector": "structured"})

    detailed = result.verbalize_detailed(backend=FakeStrictBackend())
    assert detailed.status == "generated"
    assert "активировано со степенью" in detailed.text


def test_compact_and_audit_agree_on_the_same_rule_evidence() -> None:
    """compact's decision_evidence and audit's full claim list must reference
    the same underlying claims (by id) — compact just uses short_statement
    instead of the full statement text, not different evidence."""

    _, result, _ = _wrapped()
    compact = result.to_dict(detail="compact")
    audit = result.to_dict(detail="audit")
    compact_rule_claim_ids = {item["claim_id"] for item in compact["supporting_evidence"] if "Правило" in item["statement"]}
    audit_favorable_rule_claim_ids = {c["claim_id"] for c in audit["claims"] if c["claim_type"] == "fuzzy_rule" and c["effect"] == "favorable"}
    assert compact_rule_claim_ids
    assert compact_rule_claim_ids <= audit_favorable_rule_claim_ids
    audit_claims_by_id = {c["claim_id"]: c for c in audit["claims"]}
    for claim_id in compact_rule_claim_ids:
        assert audit_claims_by_id[claim_id]["metric_value"] is not None


def test_works_through_canonical_wrap_api_end_to_end() -> None:
    model = SimpleANFIS()
    adapter = ANFISLikeAdapter(model, task="classification")
    fx = FuzzyXAI.wrap(model, adapter=adapter)
    result = fx.explain_one([33.0, 1.9], feature_names=["temperature", "pressure"])
    assert result.action in {"accept", "review", "block", "insufficient_evidence"}
    assert result.object_representation is not None  # tabular fallback at minimum


def test_collect_fuzzy_rule_activations_drops_malformed_entries_honestly() -> None:
    raw = [
        {"rule_id": "R1", "terms": [{"feature": "a", "term": "high", "membership_degree": 0.5}], "activation_strength": 0.7, "conclusion": "1"},
        {"rule_id": "", "terms": [{"feature": "a", "term": "high", "membership_degree": 0.5}], "activation_strength": 0.7, "conclusion": "1"},  # no id
        {"rule_id": "R3", "terms": [], "activation_strength": 0.7, "conclusion": "1"},  # no terms
        {"rule_id": "R4", "terms": [{"feature": "a", "term": "high", "membership_degree": 0.5}], "activation_strength": 1.7, "conclusion": "1"},  # out of range
    ]
    activations = collect_fuzzy_rule_activations(raw, object_id="p0")
    assert len(activations) == 1
    assert activations[0].rule_id == "R1"
    assert isinstance(activations[0], FuzzyRuleActivation)


def test_fuzzy_term_membership_degree_must_be_in_unit_interval() -> None:
    from fuzzyxai.evidence import FuzzyTermMembership

    with pytest.raises(ValueError):
        FuzzyTermMembership(feature="a", term="high", membership_degree=1.5)
