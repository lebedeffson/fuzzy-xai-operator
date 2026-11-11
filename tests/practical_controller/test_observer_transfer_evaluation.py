"""Transfer to unexamined data/model pairs and honest acceptance limitations."""

from __future__ import annotations

from experiments.observer_transfer_evaluation import run_transfer


def test_transfer_clean_alternative_backgrounds_and_tolerance_blind_spot() -> None:
    report = run_transfer(per_task=2)
    assert report["n_distinct_test_objects"] == 6
    assert report["n_dependent_variants"] == 6 * 9
    assert report["largest_clean_native_vs_oracle_error"] < 1e-7
    by_family = report["by_family"]
    assert by_family["clean_primary"]["certified"] == 6
    assert by_family["clean_alternate_background"]["certified"] == 6
    assert by_family["zero_sum_0_05"]["incompatible"] == 6
    assert by_family["zero_sum_0_05"]["baseline_certified"] == 6
    assert by_family["zero_sum_subthreshold_1e_8"]["certified"] == 6
    assert by_family["zero_sum_subthreshold_1e_8"]["baseline_certified"] == 6
    assert by_family["wrong_background_sha"]["incompatible"] == 6
    assert by_family["wrong_score_channel"]["incompatible"] == 6
    assert by_family["approximation_protocol"]["insufficient_evidence"] == 6
    assert by_family["missing_background_sha"]["insufficient_evidence"] == 6
    # Donor-shift is not guaranteed to create detectable errors; report any
    # misses instead of selecting favorable donors or weakening thresholds.
    assert sum(by_family["donor_attribution"][k] for k in (
        "certified", "incompatible", "insufficient_evidence"
    )) == 6
    assert report["claim_allowed"] is False
