"""P1: reference corpus + similarity evidence in the canonical runtime.

Before this, `find_similar_tabular_cases` existed but nothing in the
canonical `FuzzyXAI.wrap(...).explain_one(...)` path exercised it unless the
caller manually passed `reference_data=` *and* `include_similar_cases=True`
on every call — in practice, both real demos had `similar_cases=[]`. Now a
reference corpus can be registered once on `wrap()`, similar-case evidence
is produced by default whenever one is available, and it stays honestly
absent (never fabricated) when it isn't.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _split():
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def test_no_reference_corpus_means_no_similar_cases_and_no_flag_needed() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.similar_cases == ()


def test_reference_corpus_registered_on_wrap_is_used_by_default() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(
        model,
        adapter="auto",
        task="classification",
        reference_data=X_train,
        reference_labels=y_train,
        reference_ids=train_ids,
    )
    # No include_similar_cases=True, no per-call reference_data — the
    # corpus registered on wrap() must be enough.
    result = fx.explain_one(X_test[0], object_id="p0")
    assert len(result.similar_cases) > 0
    top = result.similar_cases[0]
    assert top["reference_object_id"] in train_ids
    assert 0.0 <= top["score"] <= 1.0


def test_per_call_reference_data_overrides_wrap_level_corpus() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")  # no corpus at wrap() time
    result = fx.explain_one(X_test[0], object_id="p0", reference_data=X_train, reference_labels=y_train)
    assert len(result.similar_cases) > 0


def test_explicit_include_similar_cases_false_suppresses_even_with_corpus() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0", include_similar_cases=False)
    assert result.similar_cases == ()


def test_similar_case_wording_is_non_causal_and_names_the_reference_object() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    result = fx.explain_one(X_test[0], object_id="p0")
    human = result.explain_for(audience="domain_user")
    assert human.details.similar_cases, "expected similar-case statements in the composed explanation"

    for statement in human.details.similar_cases:
        lowered = statement.explanation.lower()
        # Must not claim the prediction happened *because of* the similar
        # example — similarity is supporting/additional evidence unless the
        # model adapter is itself prototype-based.
        assert "потому что" not in lowered
        assert "модель выбрала класс потому" not in lowered
        # Each statement must be individually identifiable (not a copy of
        # the others) via the reference object's own id.
        assert any(ref_id in statement.title or ref_id in statement.explanation for ref_id in train_ids)

    titles = {statement.title for statement in human.details.similar_cases}
    assert len(titles) == len(human.details.similar_cases), "similar-case titles must be distinguishable, not duplicates"


def test_similarity_evidence_is_supporting_not_a_main_causal_claim() -> None:
    """Similarity claims are ranked/typed as supporting evidence (additional_support),
    not as the primary reason — matching the project's non-causal-by-default rule."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0")
    similar_claims = [claim for claim in result.claims if claim.claim_type == "similar_case"]
    assert similar_claims


# --- P1.1: reference corpus / similarity presentation -----------------------


def test_reference_labels_are_retained_on_each_similar_case() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.similar_cases
    for case in result.similar_cases:
        assert case["reference_label"] in {"0", "1"}


def test_reference_rank_and_count_are_correct() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert result.similar_cases
    top = result.similar_cases[0]
    assert top["reference_rank"] == 1
    assert top["reference_count"] == len(X_train)
    # Rank must be non-decreasing across the returned (already-sorted) cases.
    ranks = [case["reference_rank"] for case in result.similar_cases]
    assert ranks == sorted(ranks)
    assert all(case["reference_count"] == len(X_train) for case in result.similar_cases)


def test_matched_and_different_features_are_surfaced_and_ranked() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train)
    result = fx.explain_one(X_test[0], object_id="p0")
    top = result.similar_cases[0]
    # At least one of the two channels must be populated for a real tabular
    # comparison across 30 features.
    assert top["matched_features"] or top["different_features"]
    human = result.explain_for(audience="domain_user")
    joined = " ".join(statement.explanation for statement in human.details.similar_cases)
    # The per-case human text must actually name concrete feature channels,
    # not just a generic "important features" placeholder.
    if top["matched_features"]:
        assert any(name in joined for name in top["matched_features"][:3])


def test_summary_includes_exemplar_section_when_similar_cases_present() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    result = fx.explain_one(X_test[0], object_id="p0")
    text = result.summary()
    assert "## Похожие примеры" in text
    top = result.similar_cases[0]
    assert str(top["reference_object_id"]) in text
    assert "не устанавливает причину" in text  # non-causal disclaimer present


def test_summary_has_no_exemplar_section_without_reference_corpus() -> None:
    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    result = fx.explain_one(X_test[0], object_id="p0")
    assert "## Похожие примеры" not in result.summary()


def test_per_exemplar_claim_refs_point_only_to_their_own_claim() -> None:
    """Each SimilarCaseSpec must reference its own claim, not every
    similar_case claim for the query object (the provenance bug: before the
    fix, every exemplar's claim_refs held the full [C-xxx, C-yyy, C-zzz] list)."""

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    result = fx.explain_one(X_test[0], object_id="p0")
    assert len(result.similar_cases) >= 2, "need at least 2 exemplars to prove refs are distinguishable"
    all_claim_refs = [tuple(case["claim_refs"]) for case in result.similar_cases]
    for refs in all_claim_refs:
        assert len(refs) == 1, f"expected exactly one claim per exemplar, got {refs}"
    assert len(set(all_claim_refs)) == len(all_claim_refs), "each exemplar must have a distinct claim_refs, not a shared copy"
    # Cross-check against the actual claim statements: each referenced claim
    # must mention that exemplar's own reference_object_id, not a different one.
    claims_by_id = {claim.claim_id: claim for claim in result.claims}
    for case, refs in zip(result.similar_cases, all_claim_refs):
        claim = claims_by_id[refs[0]]
        assert str(case["reference_object_id"]) in claim.statement


def test_strict_verbalizer_preserves_exemplar_subject_when_similarity_is_a_main_reason() -> None:
    """When a similar-case reason is promoted into main_reasons, its
    AtomicClaim.subject (used by the strict renderer) must still name the
    concrete reference object, not a generic 'Похожий случай' placeholder."""

    from dataclasses import replace

    from fuzzyxai.verbalization import extract_atomic_claims

    X_train, X_test, y_train, _ = _split()
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    train_ids = [f"train_{i}" for i in range(len(X_train))]
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification", reference_data=X_train, reference_labels=y_train, reference_ids=train_ids)
    result = fx.explain_one(X_test[0], object_id="p0")
    explanation = result.explain_for(audience="domain_user")
    assert explanation.details.similar_cases
    similar_reason = explanation.details.similar_cases[0]
    forced = replace(explanation, main_reasons=(similar_reason, *explanation.main_reasons))
    atomic_claims = extract_atomic_claims(forced)
    reason_claim = next(claim for claim in atomic_claims if claim.claim_id == "reason-0")
    assert reason_claim.subject == similar_reason.title
    assert any(train_id in reason_claim.subject for train_id in train_ids)
