"""Developmental full-cycle audit: actual logistic retraining, local logit effects,
diagnosis and selective evidence regeneration against a full replay baseline.

Controlled defects and same-process test signers are NOT independent validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from experiments.certified_route_benchmark import (
    BETA, SEEDS, TEST_KEY, _datasets, _input_hash, _manual_equal_evidence, _stage,
)
from fuzzyxai.core.certified_route import (
    RouteStatus, SignedPrediction, certify_route_with_attested_local_effects,
    sign_local_effect_witness, sign_prediction, verify_local_effect_witness, verify_prediction,
)
from fuzzyxai.selective_observer.evidence_identity import LocalExplanationEvidence
from fuzzyxai.selective_observer.local_effects import (
    SUPPORTED_PROTOCOL, collect_linear_reference_effects_batched, hash_numeric_input,
    verify_additive_local_effects,
)

EFFECT_KEY = b"controlled-effect-producer-separate-development-key"
PRODUCER = "development-model-prober"
SCENARIOS = ("stale_model", "stale_local_explanation", "missing_effect_receipt", "changed_input")
EXPECTED = ("rejected", "rejected", "insufficient_evidence", "rejected")


@dataclass
class ModelSource:
    scaler: StandardScaler
    logistic: LogisticRegression
    name: str
    seed: int
    family: str
    score_evaluations: int = 0

    @property
    def identity(self) -> str:
        payload = {
            "name": self.name, "seed": self.seed, "family": self.family,
            "coef": self.logistic.coef_.tolist(), "intercept": self.logistic.intercept_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(), "scaler_scale": self.scaler.scale_.tolist(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def probability(self, row: np.ndarray) -> float:
        self.score_evaluations += 1
        z = self.scaler.transform(np.asarray(row, dtype=float).reshape(1, -1))
        return float(self.logistic.predict_proba(z)[0, 1])

    def target_logits(self, rows: tuple[tuple[float, ...], ...]) -> tuple[float, ...]:
        self.score_evaluations += len(rows)
        return tuple(float(v) for v in self.logistic.decision_function(
            self.scaler.transform(np.asarray(rows, dtype=float))
        ))


def snapshot(source: ModelSource, row: np.ndarray, row_id: int):
    row = np.asarray(row, dtype=float)
    feature_names = tuple(f"feature_{j}" for j in range(len(row)))
    object_id = f"{source.name}:{row_id}"
    model_version = f"{source.name}:seed{source.seed}:{source.family}"
    score = source.probability(row)
    anchor = sign_prediction(SignedPrediction(
        object_id=object_id, event_id="positive", model_version=model_version,
        unit="probability", score=score, source_sha256=_input_hash(row),
        model_sha256=source.identity,
        trace_checksum=hashlib.sha256((model_version + object_id).encode()).hexdigest(),
        complement_event_id="negative",
    ), secret=TEST_KEY)
    stages = tuple(_stage(anchor, j) for j in range(3))
    reference = np.asarray(source.scaler.mean_, dtype=float)
    effects = collect_linear_reference_effects_batched(
        object_id=object_id, model_version=model_version, target_class="positive",
        feature_names=feature_names,
        input_values=tuple(map(float, row)), reference_values=tuple(map(float, reference)),
        target_logits=source.target_logits,
    )
    assert effects.input_sha256 == anchor.source_sha256 == hash_numeric_input(
        feature_names, tuple(map(float, row))
    )
    # Real signed target-logit attribution in original input coordinates.
    weights = source.logistic.coef_[0] / source.scaler.scale_
    contributions = tuple(float(v) for v in weights * (row - reference))
    local = LocalExplanationEvidence(
        object_id=object_id, model_version=model_version, target_class="positive",
        scope="local", feature_names=feature_names, contributions=contributions,
        source_sha256=anchor.model_sha256, input_sha256=anchor.source_sha256,
        reference_sha256=effects.reference_sha256,
        attribution_protocol=SUPPORTED_PROTOCOL, score_channel="target_logit",
    )
    assert max(abs(a - b) for a, b in zip(contributions, effects.effect_on_score, strict=True)) < 1e-8
    witness = sign_local_effect_witness(
        effects, source_signature=anchor.signature, producer_id=PRODUCER, secret=EFFECT_KEY,
    )
    return (anchor, stages, local, effects, witness)


def certified(parts):
    anchor, stages, local, effects, witness = parts
    return certify_route_with_attested_local_effects(
        stages, signed_prediction=anchor, verification_secret=TEST_KEY, beta=BETA,
        local_evidence=local, source_model_effects=effects,
        signed_effect_witness=witness, effects_verification_secret=EFFECT_KEY,
        expected_effect_producer_id=PRODUCER, gamma_limit=1.0,
    )


def equal_evidence_direct(parts):
    """Same witness fields and checks; NOT an independently sourced oracle."""
    anchor, stages, local, effects, witness = parts
    status = _manual_equal_evidence(anchor, stages)
    if status != RouteStatus.CERTIFIED.value:
        return status
    if local is None or effects is None:
        return RouteStatus.INSUFFICIENT.value
    if any((
        local.object_id != anchor.object_id, local.model_version != anchor.model_version,
        local.target_class != anchor.event_id, local.scope != "local",
        local.input_sha256 != anchor.source_sha256,
        local.reference_sha256 != effects.reference_sha256,
        effects.input_sha256 != anchor.source_sha256,
    )):
        return RouteStatus.REJECTED.value
    if verify_additive_local_effects(local, effects).status.value != "certified":
        return RouteStatus.REJECTED.value
    witness_status = verify_local_effect_witness(
        witness, source_model_effects=effects, source_signature=anchor.signature,
        secret=EFFECT_KEY, expected_producer_id=PRODUCER,
    )
    return witness_status.value


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    groups = []
    for dataset, x, y in _datasets():
        for seed in SEEDS:
            train, test = train_test_split(
                np.arange(len(y)), test_size=0.2, random_state=seed, stratify=y
            )
            scaler = StandardScaler().fit(x[train])
            z_train = scaler.transform(x[train])
            original = LogisticRegression(max_iter=1200, C=1.0, random_state=seed).fit(z_train, y[train])
            replacement = LogisticRegression(max_iter=1200, C=0.2, random_state=seed).fit(z_train, y[train])
            old_source = ModelSource(scaler, original, dataset, seed, "logreg_C1")
            new_source = ModelSource(scaler, replacement, dataset, seed, "logreg_C0_2")
            assert old_source.identity != new_source.identity
            group_count = 0
            for row_id in test[:12]:
                row_id = int(row_id)
                row = np.asarray(x[row_id], dtype=float)
                old = snapshot(old_source, row, row_id)
                fresh = snapshot(new_source, row, row_id)
                assert certified(old).status is RouteStatus.CERTIFIED
                assert certified(fresh).status is RouteStatus.CERTIFIED
                old_anchor, old_stages, old_local, old_effects, old_witness = old
                new_anchor, new_stages, new_local, new_effects, new_witness = fresh

                changed = row.copy()
                changed[0] += max(0.1, float(scaler.scale_[0]) * 0.25)
                input_changed = snapshot(new_source, changed, row_id)
                changed_anchor = input_changed[0]

                # These are fault observations, not the post-repair outputs.
                observed = (
                    (new_anchor, old_stages, old_local, old_effects, old_witness),
                    (new_anchor, new_stages, replace(
                        old_local, object_id=new_anchor.object_id,
                        model_version=new_anchor.model_version,
                        source_sha256=new_anchor.model_sha256,
                    ), new_effects, new_witness),
                    (new_anchor, new_stages, new_local, new_effects, None),
                    (changed_anchor, new_stages, new_local, new_effects, new_witness),
                )
                expected_fixed = (fresh, fresh, fresh, input_changed)
                for scenario, expected, faulty, repaired in zip(
                    SCENARIOS, EXPECTED, observed, expected_fixed, strict=True
                ):
                    verdict = certified(faulty)
                    after = certified(repaired)
                    assert verdict.status.value == expected, (dataset, seed, row_id, scenario, verdict.to_dict())
                    assert after.status is RouteStatus.CERTIFIED
                    assert equal_evidence_direct(faulty) == verdict.status.value
                    assert equal_evidence_direct(repaired) == after.status.value

                    # Count *additional* source-model score evaluations required after the
                    # existing signed/source-effect receipts have been collected.
                    dimension = len(row)
                    targeted_score_evals = dimension + 2 if scenario in (SCENARIOS[0], SCENARIOS[3]) else 0
                    full_score_evals = dimension + 3  # reference+perturbations and fresh prediction
                    records.append({
                        "dataset": dataset, "seed": seed, "row_index": row_id,
                        "features": dimension, "scenario": scenario,
                        "detected_status": verdict.status.value,
                        "finding": verdict.findings[0].code if verdict.findings else "",
                        "finding_stage": verdict.findings[0].stage if verdict.findings else "",
                        "recovered_status": after.status.value,
                        "equal_evidence_baseline": equal_evidence_direct(faulty),
                        "targeted_additional_source_scores": targeted_score_evals,
                        "full_replay_source_scores": full_score_evals,
                        "model_version_before": old_source.identity,
                        "model_version_after": new_source.identity,
                    })
                group_count += 1
            groups.append({"dataset": dataset, "seed": seed, "observations": group_count})
    assert len(groups) == 9
    assert len(records) == 9 * 12 * len(SCENARIOS)
    with (output_dir / "recovery_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    scenario_summary = {}
    for name in SCENARIOS:
        rows = [record for record in records if record["scenario"] == name]
        scenario_summary[name] = {
            "cases": len(rows),
            "fault_statuses": dict(Counter(r["detected_status"] for r in rows)),
            "recovered_certified": sum(r["recovered_status"] == "certified" for r in rows),
            "equal_evidence_ties": sum(r["equal_evidence_baseline"] == r["detected_status"] for r in rows),
            "targeted_additional_score_evaluations": sum(r["targeted_additional_source_scores"] for r in rows),
            "full_replay_score_evaluations": sum(r["full_replay_source_scores"] for r in rows),
            "first_findings": dict(Counter(r["finding"] for r in rows)),
        }
    summary = {
        "protocol": "controlled-linear-source-recovery-v1",
        "not_independent_blinded_validation": True,
        "n_train_splits": len(groups), "n_test_observations": sum(g["observations"] for g in groups),
        "n_faults": len(records),
        "result": scenario_summary,
        "limitations": [
            "All faults and repair records were created by the same experimental process",
            "Reference effects use actual fitted sklearn LogisticRegression calls and linear target-logit contributions; no SHAP/LIME generalization",
            "The same-information direct checker uses the same primitive witness verification and ties the operator",
            "The calculated score-evaluation counts are protocol accounting, not measured timing or an automatic repair implementation",
            "Signed source and effect witnesses are created with distinct but collocated development keys",
            "Certified routes are synthetic scalar route stages, and correctness is restricted to declared identity and linear effects",
            "Training, data ingestion, and human review costs are excluded",
        ],
        "groups": groups,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/full_cycle_linear_recovery"))
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({"n_faults": result["n_faults"], "result": result["result"]}, indent=2))
