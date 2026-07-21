from __future__ import annotations

from fuzzyxai.q1_final.hypotheses import evaluate_h1, evaluate_h2, evaluate_h3, evaluate_h4, evaluate_h5


def _payload(modality: str) -> dict[str, object]:
    rows = []
    for index in range(40):
        confidence = 0.55 + 0.01 * (index % 40)
        rows.append(
            {
                "seed": 4201,
                "model_id": "model_1",
                "family": "linear",
                "object_id": index,
                "true_class": index % 3,
                "predicted_class": index % 3 if index % 7 else (index + 1) % 3,
                "confidence": confidence,
                "correct": bool(index % 7),
                "class_probability": confidence,
                "rare_class": index % 3 == 2,
                "low_confidence": confidence <= 0.65,
            }
        )
    return {
        "status": "PASS",
        "dataset": {
            "dataset_id": f"dataset-{modality}",
            "native_class_count": 3,
            "raw_sha256": "a" * 64,
        },
        "seeds": [4201],
        "models": [{"status": "measured", "model_id": "model_1", "family": "linear"}],
        "object_predictions": rows,
        "evaluation_object_ids": list(range(40)),
    }


def _payloads() -> dict[str, dict[str, object]]:
    return {modality: _payload(modality) for modality in ("tabular", "image", "text", "timeseries")}


def test_h1_does_not_promote_missing_real_pairs() -> None:
    result = evaluate_h1(_payloads())
    assert result["status"] == "inconclusive"
    assert result["claim_allowed"] is False


def test_real_artifact_missingness_and_structural_diagnostics_are_separate() -> None:
    h2 = evaluate_h2(_payloads(), removals_per_modality=20)
    h5 = evaluate_h5(_payloads(), faults_per_modality=25)
    assert h2["n_removals"] == 80
    assert h2["missingness_f1"] == 1.0
    assert h5["structural"]["n_faults"] == 100
    assert h5["structural"]["status"] == "supported"
    assert h5["predictive"]["status"] == "not_supported"
    assert h5["predictive"]["incremental_auprc"] == 0.0


def test_hard_case_and_uncertainty_results_are_explicitly_scoped() -> None:
    h3 = evaluate_h3(_payloads())
    h4 = evaluate_h4(_payloads())
    assert h3["full_population_status"] in {"supported", "not_supported"}
    assert h3["hard_case_status"] in {"supported", "not_supported"}
    assert h3["test_outcomes_not_used_to_define_hard_cases"] is True
    assert h4["adaptive_fml_fraction"] <= 1.0
    assert len(h4["results"]) == 6
