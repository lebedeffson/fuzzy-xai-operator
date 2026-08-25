"""P0.3: HumanExplanation quality fixes found via manual demo review.

1. `rank_human_claims` used `sorted(..., reverse=True)` on a (score,
   claim_id) tuple, which reverses the tie-break too — when several
   feature_contribution claims scored identically (because `strength` was
   clamped to 1.0 for almost every real contribution — see fix in
   claims.py), they sorted by *descending* claim_id instead of actual
   importance. `summary()`'s "main reasons" showed the wrong features.
2. Per-feature explanations phrased a degenerate (<2 object) reference
   comparison as "only one similar case is known" — which reads as
   similarity evidence that was never actually produced (`similar_cases`
   stays empty; this is an unrelated reference-profile percentile
   calculation). Removed in favor of the existing generic fallback wording.
3. The strict verbalizer lost each reason's subject/feature name — every
   sentence read identically ("supports the prediction") with no way to
   tell which feature it was about. Now `AtomicClaim.subject` threads
   through and strict-mode output is prefixed the same way the
   deterministic summary already is.
"""

from __future__ import annotations

import json

from fuzzyxai import FuzzyXAI
from fuzzyxai.evidence.human import rank_human_claims
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class _StrictOrderBackend:
    model = "fake-strict"

    def generate(self, prompt: str, *, response_schema=None) -> str:
        import re

        claim_ids = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
        return json.dumps({"order": claim_ids, "connector": "structured"})


def _wrapped_tabular_result():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = LogisticRegression(max_iter=2000).fit(X_train, y_train)
    fx = FuzzyXAI.wrap(model, adapter="auto", task="classification")
    return fx.explain_one(X_test[0], object_id="p0")


def test_feature_contribution_strength_is_not_uniformly_clamped_to_one() -> None:
    result = _wrapped_tabular_result()
    strengths = {claim.subject_id: claim.strength for claim in result.claims if claim.claim_type == "feature_contribution"}
    assert len({round(value, 3) for value in strengths.values() if value is not None}) > 1, (
        "every feature_contribution claim had the same strength — the clamp-to-1.0 bug is back"
    )
    # The single strongest contribution for this object must get strength 1.0.
    assert max(strengths.values()) == 1.0


def test_rank_human_claims_orders_by_actual_contribution_magnitude() -> None:
    result = _wrapped_tabular_result()
    fc_claims = [claim for claim in result.claims if claim.claim_type == "feature_contribution"]
    ranked = rank_human_claims(fc_claims)
    ranked_abs_values = [abs(claim.metric_value) for claim in ranked if claim.metric_value is not None]
    assert ranked_abs_values == sorted(ranked_abs_values, reverse=True), "ranked claims are not in descending |contribution| order"


def test_tie_break_does_not_reverse_claim_id_order() -> None:
    # Regression for the literal bug: sorted(claims, key=score, reverse=True)
    # reversed the (score, claim_id) tie-break along with the score.
    result = _wrapped_tabular_result()
    fc_claims = [claim for claim in result.claims if claim.claim_type == "feature_contribution"]
    ranked = rank_human_claims(fc_claims)
    # The top-ranked claim must have the single largest |contribution| — not
    # simply the highest claim_id, which is what the bug produced.
    top = ranked[0]
    assert abs(top.metric_value) == max(abs(claim.metric_value) for claim in fc_claims if claim.metric_value is not None)


def test_summary_main_reasons_match_the_strongest_contributions() -> None:
    result = _wrapped_tabular_result()
    contributions = result.view_model.model["contributions"]
    favorable_sorted = sorted(((name, value) for name, value in contributions.items() if value >= 0), key=lambda item: -item[1])
    expected_top_3 = {name for name, _ in favorable_sorted[:3]}

    human = result.explain_for(audience="domain_user")
    shown = {reason.subject_label.replace(" ", "_") for reason in human.main_reasons}
    assert shown == expected_top_3, f"main_reasons {shown} do not match the 3 strongest contributions {expected_top_3}"


def test_degenerate_reference_comparison_not_phrased_as_similar_case() -> None:
    result = _wrapped_tabular_result()
    human = result.explain_for(audience="domain_user")
    for reason in human.main_reasons:
        assert "похож" not in reason.explanation.lower(), (
            f"reason for {reason.subject_label!r} reads as similarity evidence "
            "that was never produced (similar_cases is empty): " + reason.explanation
        )


def test_strict_verbalizer_preserves_subject_labels() -> None:
    result = _wrapped_tabular_result()
    detailed = result.verbalize_detailed(backend=_StrictOrderBackend())
    assert detailed.status == "generated"
    human = result.explain_for(audience="domain_user")
    for reason in human.main_reasons:
        assert reason.subject_label.capitalize() in detailed.text or reason.title in detailed.text, (
            f"strict output lost the subject for reason {reason.subject_label!r}:\n{detailed.text}"
        )


def test_deterministic_summary_titles_distinguish_multiple_reasons() -> None:
    result = _wrapped_tabular_result()
    text = result.summary()
    human = result.explain_for(audience="domain_user")
    assert len(human.main_reasons) > 1
    for reason in human.main_reasons:
        assert f"**{reason.title}.**" in text
