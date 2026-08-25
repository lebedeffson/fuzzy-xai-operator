"""P2: compact/standard/audit export projections.

``to_dict()``/``export_json()`` used to only ever produce one very large
payload (~800KB+ for a typical tabular case) — everything, always. That is
fine for an audit trail but heavy for a normal application that just wants
the prediction and its top evidence. These three tiers are read-only
projections of the *same already-computed* ``ModelExplanationResult`` — none
of them re-runs ``explain()`` — so prediction/claims/similar_cases/action are
guaranteed identical across tiers by construction, not by convention.
"""

from __future__ import annotations

import json

import pytest
from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _tabular_result():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    return fx.explain_one(X_test[0], object_id="p0")


def _text_result():
    documents = [
        "the router keeps dropping the wifi connection at night",
        "wifi signal is unstable and drops every evening",
        "invoice payment was declined by the billing system",
        "billing system rejected the invoice payment again",
    ]
    labels = ["network", "network", "billing", "billing"]
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(documents).toarray()
    model = LogisticRegression(max_iter=2000).fit(features, labels)
    names = list(vectorizer.get_feature_names_out())
    query_text = "wifi connection drops constantly every night"
    query_vector = vectorizer.transform([query_text]).toarray()[0]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(query_vector, object_id="text-0", feature_names=names, raw_object=query_text)
    return result, query_text


def test_default_detail_is_audit_and_unchanged() -> None:
    """Backward compatibility: no detail= argument means today's full payload."""

    result = _tabular_result()
    default_payload = result.to_dict()
    explicit_audit = result.to_dict(detail="audit")
    assert default_payload == explicit_audit
    assert set(default_payload.keys()) >= {"model", "claims", "explanation_graph", "human_explanations", "visual_spec", "trace"}


def test_unsupported_detail_level_raises() -> None:
    result = _tabular_result()
    with pytest.raises(ValueError):
        result.to_dict(detail="bogus")


def test_prediction_identical_across_all_detail_levels() -> None:
    result = _tabular_result()
    compact = result.to_dict(detail="compact")
    standard = result.to_dict(detail="standard")
    audit = result.to_dict(detail="audit")
    assert compact["prediction"]["value"] == standard["prediction"]["value"]
    assert standard["prediction"]["value"] == audit["model"]["predictions"]
    assert compact["prediction"]["score"] == standard["prediction"]["score"] == audit["model"]["score"]


def test_action_identical_across_all_detail_levels() -> None:
    result = _tabular_result()
    compact = result.to_dict(detail="compact")
    standard = result.to_dict(detail="standard")
    audit = result.to_dict(detail="audit")
    assert compact["action"] == standard["action"] == audit["risk"]["action"] == result.action


def test_similar_cases_identical_across_compact_and_standard() -> None:
    result = _tabular_result()
    compact = result.to_dict(detail="compact")
    standard = result.to_dict(detail="standard")
    assert compact["similar_cases"] == standard["similar_cases"] == list(result.similar_cases)


def test_claim_values_identical_between_standard_and_result_claims() -> None:
    result = _tabular_result()
    standard = result.to_dict(detail="standard")
    assert [claim.to_dict() for claim in result.claims] == standard["claims"]


def test_compact_is_materially_smaller_than_audit() -> None:
    result = _tabular_result()
    compact_size = len(json.dumps(result.to_dict(detail="compact"), ensure_ascii=False))
    standard_size = len(json.dumps(result.to_dict(detail="standard"), ensure_ascii=False))
    audit_size = len(json.dumps(result.to_dict(detail="audit"), ensure_ascii=False))
    assert compact_size < standard_size < audit_size
    assert compact_size < audit_size * 0.1  # compact must be at least an order of magnitude smaller


def test_include_raw_false_strips_raw_text_at_every_detail_level() -> None:
    result, query_text = _text_result()
    for detail in ("compact", "standard", "audit"):
        payload = result.to_dict(detail=detail, include_raw=False)
        assert query_text not in json.dumps(payload, ensure_ascii=False)


def test_include_raw_true_restores_raw_text_at_every_detail_level() -> None:
    result, query_text = _text_result()
    for detail in ("compact", "standard", "audit"):
        payload = result.to_dict(detail=detail, include_raw=True)
        assert query_text in json.dumps(payload, ensure_ascii=False)


def test_audit_retains_full_provenance_graph_and_all_audiences() -> None:
    result = _tabular_result()
    audit = result.to_dict(detail="audit")
    assert audit["explanation_graph"]["nodes"]
    assert set(audit["human_explanations"].keys()) >= {"domain_user", "ml_engineer", "researcher", "auditor"}
    assert audit["trace"]["adapter_id"]


def test_standard_selects_one_audience_but_can_be_overridden() -> None:
    result = _tabular_result()
    default_standard = result.to_dict(detail="standard")
    assert default_standard["human_explanation"]["audience"] == "domain_user"
    expert_standard = result.to_dict(detail="standard", audience="ml_engineer")
    assert expert_standard["human_explanation"]["audience"] == "ml_engineer"


def test_export_does_not_mutate_result_and_is_deterministic() -> None:
    result = _tabular_result()
    action_before = result.action
    first = json.dumps(result.to_dict(detail="compact"), sort_keys=True, ensure_ascii=False)
    second = json.dumps(result.to_dict(detail="compact"), sort_keys=True, ensure_ascii=False)
    assert first == second
    assert result.action == action_before


def test_export_json_file_matches_to_dict(tmp_path) -> None:
    result = _tabular_result()
    for detail in ("compact", "standard", "audit"):
        path = result.export_json(tmp_path / f"{detail}.json", detail=detail)
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        # Round-trip through JSON turns tuples into lists; normalize both
        # sides through the same encoder before comparing structural content.
        expected = json.loads(json.dumps(result.to_dict(detail=detail), ensure_ascii=False, default=list))
        assert loaded == expected


def test_compact_carries_minimal_provenance_not_full_graph() -> None:
    result = _tabular_result()
    compact = result.to_dict(detail="compact")
    assert "explanation_graph" not in compact
    assert "human_explanations" not in compact
    assert compact["provenance"]["adapter_id"]
    assert compact["provenance"]["model_fingerprint"]
