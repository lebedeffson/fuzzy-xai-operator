"""Developmental evidence audit after replacing a genuinely fitted estimator.

Reuses Certified Route v8 and the equal-evidence imperative comparator from
certified_route_benchmark. Stages are controlled scalar receipts, NOT SHAP/LIME.
No independent incident labels, clinical claims, or trusted production signer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
from certified_route_benchmark import BETA, SEEDS, TEST_KEY, _datasets, _input_hash, _manual_equal_evidence, _stage
from fuzzyxai.core.certified_route import SignedPrediction, certify_route, sign_prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROTOCOL_PATH = Path(__file__).with_name("model_replacement_revalidation_protocol_v1.json")
SCENARIOS = ("original_clean", "stale_route_after_real_model_replacement",
             "recomputed_route_after_model_replacement", "missing_prediction_witness")
EXPECTED = ("certified", "rejected", "certified", "insufficient_evidence")


def model_digest(model):
    # Identifies this environment's fitted pickle bytes, NOT a portable semantic identity.
    return hashlib.sha256(pickle.dumps(model, protocol=4)).hexdigest()


def make_receipt(dataset, seed, row_id, score, model, features, family):
    object_id = f"{dataset}:{int(row_id)}"
    model_version = f"{dataset}:seed{seed}:{family}:v1"
    return sign_prediction(SignedPrediction(
        object_id=object_id, event_id="positive", model_version=model_version,
        unit="probability", score=float(score),
        source_sha256=_input_hash(features), model_sha256=model_digest(model),
        trace_checksum=hashlib.sha256((model_version + object_id).encode()).hexdigest(),
        complement_event_id="negative",
    ), secret=TEST_KEY)


def run(output_dir: Path) -> dict:
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes)
    assert protocol["random_seeds"] == list(SEEDS)
    assert protocol["scenarios"] == list(SCENARIOS)
    assert protocol["expected_statuses"] == list(EXPECTED)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, model_rows = [], []
    for dataset, x, y in _datasets():
        for seed in SEEDS:
            training, testing = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y
            )
            old = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1200, C=1.0, random_state=seed)
            )
            new = make_pipeline(
                StandardScaler(), RandomForestClassifier(
                    n_estimators=64, max_depth=8, random_state=seed, n_jobs=1
                )
            )
            old.fit(x[training], y[training])
            new.fit(x[training], y[training])
            old_p = old.predict_proba(x[testing])[:, 1]
            new_p = new.predict_proba(x[testing])[:, 1]
            old_hash, new_hash = model_digest(old), model_digest(new)
            assert old_hash != new_hash
            model_rows.append({
                "dataset": dataset, "seed": seed, "train_n": len(training),
                "test_n": len(testing), "old_model_sha256": old_hash,
                "new_model_sha256": new_hash,
                "mean_absolute_probability_shift": float(np.mean(np.abs(new_p - old_p))),
            })
            for i, row_index in enumerate(testing):
                old_ref = make_receipt(dataset, seed, row_index, old_p[i], old, x[row_index], "logreg")
                new_ref = make_receipt(dataset, seed, row_index, new_p[i], new, x[row_index], "rf")
                old_stages = [_stage(old_ref, j) for j in range(3)]
                new_stages = [_stage(new_ref, j) for j in range(3)]
                inputs = (
                    (old_ref, old_stages),
                    (new_ref, old_stages),
                    (new_ref, new_stages),
                    (None, new_stages),
                )
                for scenario, expected, (reference, stages) in zip(
                    SCENARIOS, EXPECTED, inputs, strict=True
                ):
                    certified = certify_route(
                        stages, signed_prediction=reference,
                        verification_secret=TEST_KEY, beta=BETA, gamma_limit=1.0
                    )
                    control = _manual_equal_evidence(reference, stages)
                    assert certified.status.value == expected
                    assert control == certified.status.value
                    first_code = certified.findings[0].code if certified.findings else ""
                    first_stage = certified.findings[0].stage if certified.findings else ""
                    if scenario == SCENARIOS[1]:
                        assert (first_code, first_stage) == ("REFERENCE_MISMATCH", 0)
                    if scenario == SCENARIOS[3]:
                        assert first_code == "SOURCE_REFERENCE_UNVERIFIED"
                    rows.append({
                        "dataset": dataset, "seed": seed, "row_index": int(row_index),
                        "scenario": scenario, "expected": expected,
                        "fuzzyxai_status": certified.status.value,
                        "same_evidence_direct": control, "first_finding": first_code,
                        "first_finding_stage": first_stage,
                        "old_model_sha256": old_hash, "new_model_sha256": new_hash,
                        "old_probability": float(old_p[i]),
                        "new_probability": float(new_p[i]),
                        "certificate_fingerprint": certified.route_fingerprint or "",
                    })
    assert len(model_rows) == 9
    assert len(rows) == 6120
    for filename, records in (("model_replacement_cases.csv", rows), ("model_pairs.csv", model_rows)):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    summary = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "status": "DEVELOPMENT_ONLY_NONBLINDED",
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__},
        "n_fitted_pairs": len(model_rows),
        "n_fitted_estimators": 2 * len(model_rows),
        "n_distinct_test_predictions": len(rows) // len(SCENARIOS),
        "n_status_records": len(rows),
        "by_scenario": {
            name: {
                "n": len(group),
                "statuses": dict(Counter(r["fuzzyxai_status"] for r in group)),
                "same_evidence_equal": sum(r["fuzzyxai_status"] == r["same_evidence_direct"] for r in group),
                "first_findings": dict(Counter(r["first_finding"] for r in group)),
            }
            for name in SCENARIOS
            for group in [[r for r in rows if r["scenario"] == name]]
        },
        "model_pairs": model_rows,
        "limitations": protocol["limitations"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("artifacts/model_replacement_revalidation"))
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps({"n_records": summary["n_status_records"],
                      "by_scenario": summary["by_scenario"]}, indent=2))


if __name__ == "__main__":
    main()
