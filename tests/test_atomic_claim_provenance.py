"""P6: AtomicClaim carries direction + source_claim_ids (spec 12.2 minimum contract).

Before this, AtomicClaim had claim_id/kind/subject/canonical_text/
allowed_numbers/allowed_entities but no explicit direction or link back to
the ExplanationClaim(s) it was built from — a verbalizer backend had no
structured signal for "supports vs opposes" beyond parsing Russian prose,
and a consumer couldn't trace a rendered sentence back to its evidence.
Both are read straight off the already-existing HumanStatement fields
(effect_direction, claim_refs) — no new computation, no redesign of the
verbalization architecture.
"""

from __future__ import annotations

from fuzzyxai import FuzzyXAI
from fuzzyxai.verbalization import extract_atomic_claims
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _result():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    return fx.explain_one(X_test[0], object_id="p0")


def test_reason_claims_carry_a_non_neutral_direction() -> None:
    result = _result()
    explanation = result.explain_for(audience="domain_user")
    claims = extract_atomic_claims(explanation)
    reason_claims = [claim for claim in claims if claim.kind == "reason"]
    assert reason_claims
    for claim in reason_claims:
        assert claim.direction in {"supports", "opposes", "mixed", "additional_support"}


def test_every_atomic_claim_traces_back_to_a_real_explanation_claim_id() -> None:
    result = _result()
    explanation = result.explain_for(audience="domain_user")
    claims = extract_atomic_claims(explanation)
    real_claim_ids = {claim.claim_id for claim in result.claims}
    for atomic in claims:
        assert atomic.source_claim_ids, f"{atomic.claim_id} has no source_claim_ids"
        for ref in atomic.source_claim_ids:
            assert ref in real_claim_ids, f"{atomic.claim_id} references unknown claim {ref}"


def test_decision_and_action_claims_default_to_neutral_direction() -> None:
    result = _result()
    explanation = result.explain_for(audience="domain_user")
    claims = extract_atomic_claims(explanation)
    by_id = {claim.claim_id: claim for claim in claims}
    assert by_id["decision-0"].direction == "neutral"
    assert by_id["action-0"].direction == "neutral"
