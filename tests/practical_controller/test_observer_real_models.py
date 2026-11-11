"""End-to-end seeded observer exercise with fitted models and held-out inputs."""
from __future__ import annotations

from experiments.observer_local_evidence_benchmark import run_benchmark


def test_observer_contract_detects_declared_faults_on_three_fitted_models() -> None:
    report = run_benchmark(n_per_dataset=5, seed=42)
    assert report["n_rows"] == 3 * 5 * 9
    assert report["maximum_clean_logit_reconstruction_error"] < 1e-8

    families = report["by_family"]
    assert families["clean"]["accepted_structurally"] == 15
    assert families["clean"]["blocked"] == 0
    assert families["clean"]["review_required"] == 0

    for name in (
        "object_substitution",
        "target_substitution",
        "feature_order_substitution",
        "flattened_multiclass_attribution",
        "global_attribution_as_local",
        "source_digest_substitution",
    ):
        assert families[name]["blocked"] == 15, name
        assert families[name]["accepted_structurally"] == 0, name

    assert families["missing_target"]["review_required"] == 15
    # This negative result is intentional: a shape/identity contract cannot
    # certify the semantics of the attribution values it was handed.
    assert families["silent_value_permutation"]["accepted_structurally"] == 15
    assert report["claim_allowed"] is False
