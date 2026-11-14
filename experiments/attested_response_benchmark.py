"""Second-attestation controlled benchmark for linear logit attribution.

Uses fitted sklearn models. Synthetic, nonblinded faults are correlated by
test prediction. Equal-evidence comparator is designed to tie the operator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import sklearn
from certified_route_benchmark import BETA, SEEDS, _datasets, _model_hash, _stage
from fuzzyxai.core.certified_route import (
    SignedPrediction,
    certify_route_with_attested_local_effects,
    certify_route_with_local_effects,
    sign_local_effect_witness,
    sign_prediction,
)
from fuzzyxai.selective_observer.evidence_identity import LocalExplanationEvidence
from fuzzyxai.selective_observer.local_effects import (
    SUPPORTED_PROTOCOL,
    collect_linear_reference_effects_batched,
    hash_numeric_input,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_KEY = b"development-prediction-signing-key-not-production"
PROBE_KEY = b"development-probe-signing-key-not-production"
CASES = (
    "clean", "two_sided_coherent_probe_spoof", "unsigned_probe",
    "forged_probe_signature", "wrong_prediction_binding",
    "attribution_only_spoof",
)
EFFECT_FIELDS = (
    "protocol", "object_id", "model_version", "target_class", "feature_names",
    "effect_on_score", "reference_score", "explained_score",
    "input_sha256", "reference_sha256",
)


def _manual_equal_evidence(local, probed, witness, origin):
    """Separate rules for the locked positive-route cases.

    Does not consult the Certified Route result or intermediate weak outcome.
    It is limited to this experimental protocol, not a general route auditor.
    """
    if (local.object_id != origin.object_id
            or probed.object_id != origin.object_id
            or local.model_version != origin.model_version
            or probed.model_version != origin.model_version
            or local.target_class != origin.event_id
            or probed.target_class != origin.event_id
            or local.scope != "local"
            or probed.protocol != SUPPORTED_PROTOCOL
            or local.feature_names != probed.feature_names
            or local.input_sha256 != origin.source_sha256
            or probed.input_sha256 != origin.source_sha256
            or local.reference_sha256 != probed.reference_sha256):
        return "rejected"
    local_effects = local.contributions
    observed_effects = probed.effect_on_score
    if (local_effects is None or observed_effects is None
            or len(local_effects) != len(observed_effects)
            or len(local_effects) != len(local.feature_names)):
        return "insufficient_evidence"
    if (any(abs(float(a) - float(b)) > 1e-6
            for a, b in zip(local_effects, observed_effects, strict=True))
            or abs(sum(local_effects) -
                   (probed.explained_score - probed.reference_score)) > 1e-6):
        return "rejected"
    if witness is None:
        return "insufficient_evidence"
    if (witness.source_signature != origin.signature
            or witness.producer_id != "independent-sklearn-prober"):
        return "rejected"
    if not isinstance(witness.signature, str) or len(witness.signature) != 64:
        return "rejected"
    payload = json.dumps(
        witness.payload(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode()
    expected = hmac.new(PROBE_KEY, payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(witness.signature, expected):
        return "rejected"
    response = {key: getattr(probed, key) for key in EFFECT_FIELDS}
    data = json.dumps(
        response, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode()
    digest = hashlib.sha256(data).hexdigest()
    return "certified" if hmac.compare_digest(digest, witness.effects_sha256) else "rejected"

def _fault(case, local, response, witness):
    if case == "two_sided_coherent_probe_spoof":
        changed = list(response.effect_on_score)
        changed[0] += 0.05
        changed[1] -= 0.05
        response = replace(response, effect_on_score=tuple(changed))
        local = replace(local, contributions=tuple(changed))
    elif case == "unsigned_probe":
        witness = None
    elif case == "forged_probe_signature":
        witness = replace(witness, producer_id="attacker")
    elif case == "wrong_prediction_binding":
        witness = sign_local_effect_witness(
            response, source_signature="e" * 64,
            producer_id="independent-sklearn-prober", secret=PROBE_KEY,
        )
    elif case == "attribution_only_spoof":
        changed = list(local.contributions)
        changed[0] += 0.05
        changed[1] -= 0.05
        local = replace(local, contributions=tuple(changed))
    elif case != "clean":
        raise ValueError(case)
    return local, response, witness


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    n_predictions = 0
    for dataset, features, labels in _datasets():
        for seed in SEEDS:
            train_idx, test_idx = train_test_split(
                np.arange(len(labels)), test_size=0.2, random_state=seed,
                stratify=labels,
            )
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1200, C=1.0, random_state=seed),
            )
            model.fit(features[train_idx], labels[train_idx])
            model_sha = _model_hash(model)
            reference = tuple(float(v) for v in features[train_idx].mean(axis=0))
            names = tuple(f"feature_{i}" for i in range(features.shape[1]))
            ref_hash = hash_numeric_input(names, reference)
            predictions = model.predict_proba(features[test_idx])[:, 1]
            for ridx, pred in zip(test_idx, predictions, strict=True):
                n_predictions += 1
                raw_row = tuple(float(v) for v in features[int(ridx)])
                version = f"{dataset}-seed{seed}"
                object_id = f"{dataset}:{int(ridx)}"
                input_hash = hash_numeric_input(names, raw_row)
                origin = sign_prediction(
                    SignedPrediction(
                        object_id=object_id, event_id="positive",
                        unit="probability", model_version=version,
                        score=float(pred), source_sha256=input_hash,
                        model_sha256=model_sha,
                        trace_checksum=hashlib.sha256(
                            (version + object_id).encode()
                        ).hexdigest(),
                        complement_event_id="negative",
                    ),
                    secret=MODEL_KEY,
                )
                stages = [_stage(origin, i) for i in (0, 1)]
                effects = collect_linear_reference_effects_batched(
                    object_id=object_id, model_version=version,
                    target_class="positive", feature_names=names,
                    input_values=raw_row, reference_values=reference,
                    target_logits=lambda batch, model=model: tuple(
                        float(value) for value in model.decision_function(
                            np.asarray(batch, dtype=float)
                        )
                    ),
                )
                assert effects.input_sha256 == input_hash
                assert effects.reference_sha256 == ref_hash
                scaler = model.named_steps["standardscaler"]
                classifier = model.named_steps["logisticregression"]
                contributions = tuple(
                    float((x_i - r_i) / s_i * c_i)
                    for x_i, r_i, s_i, c_i in zip(
                        raw_row, reference, scaler.scale_, classifier.coef_[0],
                        strict=True,
                    )
                )
                explanation_bytes = json.dumps(
                    {
                        "model_sha256": model_sha, "input_sha256": input_hash,
                        "reference_sha256": ref_hash,
                        "contributions": contributions, "feature_names": names,
                    },
                    sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode()
                local = LocalExplanationEvidence(
                    object_id=object_id, model_version=version,
                    target_class="positive", scope="local", feature_names=names,
                    contributions=contributions,
                    source_sha256=hashlib.sha256(explanation_bytes).hexdigest(),
                    input_sha256=input_hash, reference_sha256=ref_hash,
                    attribution_protocol=SUPPORTED_PROTOCOL,
                    score_channel="target_logit",
                )
                witness = sign_local_effect_witness(
                    effects, source_signature=origin.signature,
                    producer_id="independent-sklearn-prober", secret=PROBE_KEY,
                )
                for case in CASES:
                    loc, response, receipt = _fault(case, local, effects, witness)
                    weak = certify_route_with_local_effects(
                        stages, signed_prediction=origin,
                        verification_secret=MODEL_KEY, beta=BETA,
                        local_evidence=loc, source_model_effects=response,
                    )
                    strong = certify_route_with_attested_local_effects(
                        stages, signed_prediction=origin,
                        verification_secret=MODEL_KEY, beta=BETA,
                        local_evidence=loc, source_model_effects=response,
                        signed_effect_witness=receipt,
                        effects_verification_secret=PROBE_KEY,
                        expected_effect_producer_id="independent-sklearn-prober",
                    )
                    manual = _manual_equal_evidence(
                        loc, response, receipt, origin,
                    )
                    expected = (
                        "certified" if case == "clean"
                        else "insufficient_evidence" if case == "unsigned_probe"
                        else "rejected"
                    )
                    assert strong.status.value == manual == expected, (
                        dataset, seed, ridx, case, weak.to_dict(),
                        strong.to_dict(), manual,
                    )
                    rows.append({
                        "dataset": dataset, "seed": seed,
                        "object_id": object_id, "case": case, "expected": expected,
                        "weak_unauthenticated": weak.status.value,
                        "dual_attested": strong.status.value,
                        "manual_equal_evidence": manual,
                        "reason": strong.findings[0].code if strong.findings else "OK",
                    })
    with (output_dir / "case_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_case = []
    for case in CASES:
        group = [row for row in rows if row["case"] == case]
        by_case.append({
            "case": case, "n": len(group),
            "weak_certified": sum(
                row["weak_unauthenticated"] == "certified" for row in group
            ),
            "strong_certified": sum(
                row["dual_attested"] == "certified" for row in group
            ),
            "strong_rejected": sum(
                row["dual_attested"] == "rejected" for row in group
            ),
            "strong_insufficient": sum(
                row["dual_attested"] == "insufficient_evidence" for row in group
            ),
            "equal_evidence_matches": sum(
                row["dual_attested"] == row["manual_equal_evidence"]
                for row in group
            ),
        })
    report = {
        "protocol_id": "dual-attested-linear-response-development-v1",
        "status": "DEVELOPMENT_ONLY_NONBLINDED",
        "models": 9, "n_predictions": n_predictions,
        "n_correlated_trials": len(rows),
        "cases": list(CASES), "by_case": by_case,
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "limits": [
            "All attacks are deliberately synthesized on fitted sklearn models",
            "Attestation is the changed evidence source; equal-evidence baseline ties by design",
            "Demo signs both receipts in one process: independent key custody is not established",
            "Only linear additive target-logit responses are under test, not generic SHAP/LIME",
            "Cases from the same test prediction are correlated, not independent samples",
            "Real-world predictor correctness, causality, and human usefulness are not certified",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                        default=Path("certified_route_attested_results"))
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps({
        key: summary[key]
        for key in ("models", "n_predictions", "n_correlated_trials", "status")
    }, indent=2))
