"""P14: integration consistency closure.

Five real integration gaps found by independently re-inspecting the P11/P12
wheel and final_validation bundle — the new fuzzy_rule/image_region evidence
types were correct in isolation but not fully wired into the older
graph/explanation-level/compact-export/strict-verbalizer layers that predate
them. Each test below reproduces the exact defect that was found.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, ClassVar

import numpy as np
from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters.contracts_v2 import ExplanationContext, LocalModelEvidence
from fuzzyxai.adapters.model import ModelPrediction
from fuzzyxai.adapters.model_v2 import ModelAdapterV2
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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
        out = []
        for rule_id, antecedents, consequent in self.RULES:
            strength = 1.0
            terms = []
            for feature, term in antecedents.items():
                degree = memberships[feature][term]
                strength *= degree
                terms.append({"feature": feature, "term": term, "membership_degree": degree, "feature_value": values[feature]})
            out.append({"rule_id": rule_id, "terms": terms, "activation_strength": strength, "conclusion": str(consequent)})
        return out

    def predict_one(self, temperature: float, pressure: float) -> tuple[int, list[dict[str, Any]]]:
        activations = self._activations(temperature, pressure)
        total = sum(a["activation_strength"] for a in activations) or 1e-9
        weighted = sum(a["activation_strength"] * float(a["conclusion"]) for a in activations) / total
        return round(weighted), activations

    def predict(self, X: Any) -> np.ndarray:
        rows = np.asarray(X)
        return np.array([self.predict_one(row[0], row[1])[0] for row in rows])


class ANFISLikeAdapter(ModelAdapterV2):
    adapter_id = "p14_test_anfis_adapter"
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


def _fuzzy_result(temperature: float = 33.0, pressure: float = 1.9):
    model = SimpleANFIS()
    fx = FuzzyXAI.wrap(model, adapter=ANFISLikeAdapter(model, task="classification"))
    return fx.explain_one([temperature, pressure], feature_names=["temperature", "pressure"])


def _image_result():
    rng = np.random.default_rng(0)
    X = rng.random((40, 30))
    y = (X[:, 0] > 0.5).astype(int)
    model = LogisticRegression(max_iter=500).fit(X, y)
    image = rng.random((5, 6))
    mask_a = np.zeros((5, 6), dtype=bool)
    mask_a[0:2, 0:2] = True
    mask_b = np.zeros((5, 6), dtype=bool)
    mask_b[3:5, 3:6] = True
    fx = FuzzyXAI.wrap(model)
    return fx.explain_one(
        image.flatten(),
        raw_object=image,
        region_masks={"roi_a": mask_a, "roi_b": mask_b},
        evidence={"contributions": {"roi_a": 2.0, "roi_b": -1.5}},
    )


def _tabular_result():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    return fx.explain_one(X_test[0], object_id="p0")


# --- 1. graph provenance for fuzzy_rule / image_region -----------------------


def test_fuzzy_rule_claim_reaches_its_evidence_node_via_inspect() -> None:
    result = _fuzzy_result()
    rule_claim = next(c for c in result.claims if c.claim_type == "fuzzy_rule")
    inspection = result.inspect(f"claim:{rule_claim.claim_id}")
    node_types = {node.node_type for node in inspection.related_nodes}
    assert "fuzzy_rule" in node_types
    assert "data" in node_types
    assert "prediction" in node_types
    relations = {(edge.source, edge.target, edge.relation) for edge in inspection.related_edges}
    assert any(edge[2] == "supports_claim" for edge in relations)


def test_image_region_claim_reaches_its_evidence_node_via_inspect() -> None:
    result = _image_result()
    region_claim = next(c for c in result.claims if c.claim_type == "image_region")
    inspection = result.inspect(f"claim:{region_claim.claim_id}")
    node_types = {node.node_type for node in inspection.related_nodes}
    assert "image_region" in node_types


def test_graph_reachability_has_no_new_errors() -> None:
    for result in (_fuzzy_result(), _image_result()):
        assert result.explanation_graph.validate_reachability() == ()


# --- 2. fuzzy evidence must not simultaneously exist and be "missing" -------


def test_fuzzy_case_does_not_claim_model_rules_are_missing() -> None:
    result = _fuzzy_result()
    assert "model_rules_or_concepts" not in result.view_model.trace["missing_evidence"]


def test_fuzzy_case_reaches_at_least_e2_with_fuzzy_channel_available() -> None:
    result = _fuzzy_result()
    assert result.explanation_level in {"E2", "E3", "E4", "E5"}
    assert "fuzzy_rule_activations" in result.available_channels


def test_story_knowledge_stage_reflects_activated_fuzzy_rules() -> None:
    result = _fuzzy_result()
    story_text = result.story()
    knowledge_section = story_text.split("## Знания модели")[1].split("## Решение")[0]
    assert "Evidence для этапа отсутствует" not in knowledge_section


# --- 3. adverse fuzzy rule wording must not be self-contradictory ----------


def test_adverse_fuzzy_rule_wording_blames_the_prediction_not_itself() -> None:
    result = _fuzzy_result()
    adverse_claim_id = next(c.claim_id for c in result.claims if c.claim_type == "fuzzy_rule" and c.effect == "adverse")
    # "auditor" has an effectively unlimited concern cap (max_concerns=1000)
    # so every adverse claim is guaranteed to be present, not truncated.
    human = result.explain_for(audience="auditor")
    adverse_statement = next(s for s in human.concerns if adverse_claim_id in s.claim_refs)
    # Must not say the rule's conclusion contradicts *itself* (the old bug):
    # "Заключение правила противоречит результату «класс 0»" where class 0
    # IS the rule's own conclusion.
    assert "что противоречит текущему прогнозу" in adverse_statement.explanation
    assert "класс 1" in adverse_statement.explanation  # names the actual prediction, not just the rule's own class


def test_all_adverse_fuzzy_rules_name_the_actual_prediction() -> None:
    result = _fuzzy_result()
    predicted = str(result.prediction.predictions[0])
    human = result.explain_for(audience="auditor")
    adverse_claim_ids = {c.claim_id for c in result.claims if c.claim_type == "fuzzy_rule" and c.effect == "adverse"}
    assert adverse_claim_ids
    for claim_id in adverse_claim_ids:
        statement = next(s for s in human.concerns if claim_id in s.claim_refs)
        assert f"класс {predicted}" in statement.explanation


# --- 4. compact export must classify adverse directional evidence correctly -


def test_compact_contradicting_evidence_includes_adverse_fuzzy_rules() -> None:
    result = _fuzzy_result()
    compact = result.to_dict(detail="compact")
    contradicting_subjects = {item["claim_id"] for item in compact["contradicting_evidence"]}
    adverse_fuzzy_claim_ids = {c.claim_id for c in result.claims if c.claim_type == "fuzzy_rule" and c.effect == "adverse"}
    assert adverse_fuzzy_claim_ids
    assert adverse_fuzzy_claim_ids <= contradicting_subjects


def test_compact_contradicting_evidence_includes_adverse_image_regions() -> None:
    result = _image_result()
    compact = result.to_dict(detail="compact")
    contradicting_ids = {item["claim_id"] for item in compact["contradicting_evidence"]}
    adverse_region_ids = {c.claim_id for c in result.claims if c.claim_type == "image_region" and c.effect == "adverse"}
    assert adverse_region_ids
    assert adverse_region_ids <= contradicting_ids


def test_compact_supporting_evidence_includes_real_feature_contributions_not_only_surrogate_rules() -> None:
    result = _tabular_result()
    compact = result.to_dict(detail="compact")
    supporting_claim_ids = {item["claim_id"] for item in compact["supporting_evidence"]}
    claims_by_id = {c.claim_id: c for c in result.claims}
    assert any(claims_by_id[cid].claim_type == "feature_contribution" for cid in supporting_claim_ids if cid in claims_by_id)


def test_compact_and_summary_agree_on_which_fuzzy_rules_contradict() -> None:
    """summary()'s '## Что вызывает сомнение' and compact's
    contradicting_evidence must reference the same underlying claims."""

    result = _fuzzy_result()
    human = result.explain_for(audience="auditor")
    concern_claim_ids = {ref for statement in human.concerns for ref in statement.claim_refs}
    compact = result.to_dict(detail="compact")
    compact_claim_ids = {item["claim_id"] for item in compact["contradicting_evidence"]}
    adverse_fuzzy_ids = {c.claim_id for c in result.claims if c.claim_type == "fuzzy_rule" and c.effect == "adverse"}
    assert adverse_fuzzy_ids <= concern_claim_ids
    assert adverse_fuzzy_ids <= compact_claim_ids


# --- 5. strict verbalizer must be able to select similarity evidence -------


class _OrderPreservingStrictBackend:
    model = "p14-fake-backend"

    def generate(self, prompt: str, *, response_schema=None) -> str:
        del response_schema
        claim_ids = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
        return json.dumps({"order": claim_ids, "connector": "structured"})


def test_strict_verbalizer_includes_similar_case_exemplars() -> None:
    result = _tabular_result()
    detailed = result.verbalize_detailed(backend=_OrderPreservingStrictBackend())
    assert detailed.status == "generated"
    assert "train_" in detailed.text
    assert "Похожий" in detailed.text or "похож" in detailed.text.lower()


def test_atomic_claims_include_a_similar_case_reason_with_traceable_source() -> None:
    from fuzzyxai.verbalization import extract_atomic_claims

    result = _tabular_result()
    explanation = result.explain_for(audience="domain_user")
    atomic_claims = extract_atomic_claims(explanation)
    similar_case_claim_ids = {ref for statement in explanation.details.similar_cases for ref in statement.claim_refs}
    covered = {ref for claim in atomic_claims for ref in claim.source_claim_ids}
    assert similar_case_claim_ids <= covered
