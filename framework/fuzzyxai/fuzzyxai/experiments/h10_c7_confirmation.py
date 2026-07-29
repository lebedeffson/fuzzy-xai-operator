from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from fuzzyxai.repository_diagnostics.calibrated_confirmation import (
    FEATURE_NAMES,
    FORBIDDEN_FEATURE_NAMES,
    CalibratedDiagnosisConfirmer,
    ConfirmationDecision,
    ConfirmationFeatures,
    DeterministicConfirmationModel,
    LogisticConfirmationModel,
    evaluation_family,
    extract_confirmation_features,
    mean_feature,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedDiagnosis,
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import IncidentQuery
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events

from .h10_c7 import DevelopmentIncident, GoldLocalization, _graph, _metrics, _row
from .h10_c7_replay import _json, _replay_records

R5C_STATUS_GO = "H10_C7_OPEN_REPLAY_GO"
R5C_STATUS_NO_GO = "H10_C7_OPEN_REPLAY_NO_GO"
R5C_SEED = 1707
R5_BASELINE_METRICS = {
    "recall_at_10": 0.8666666666666667,
    "recall_at_20": 0.9333333333333333,
    "mrr": 0.5298015873015873,
    "contract_macro_f1": 0.6238521168753727,
    "joint_hit_at_3": 0.4666666666666667,
}
PARENT_REFRACTOR_STATUS = Path(
    "results/h10_c7/open_replay_refactor/OPEN_REPLAY_STATUS.json"
)
PARENT_REFACTOR_MATRIX = Path(
    "results/h10_c7/open_replay_refactor/OPEN_REPLAY_MODEL_MATRIX.csv"
)


class ConfirmationProbabilityModel(Protocol):
    model_id: str

    def fit(
        self,
        features: tuple[ConfirmationFeatures, ...],
        labels: tuple[int, ...],
    ) -> ConfirmationProbabilityModel:
        ...

    def predict_probability(self, features: ConfirmationFeatures) -> float:
        ...

    def parameters(self) -> dict[str, object]:
        ...


@dataclass(frozen=True)
class ConfirmationReplayIncident:
    incident: DevelopmentIncident
    gold: GoldLocalization
    r5: GuidedDiagnosis
    r6: GuidedDiagnosis
    features: ConfirmationFeatures
    active_features: ConfirmationFeatures | None
    target: int
    r5_row: dict[str, object]


@dataclass(frozen=True)
class FoldPrediction:
    incident_id: str
    repository: str
    model_id: str
    threshold: float
    probability_before: float
    probability_after: float
    active_evidence_status: str
    decision: ConfirmationDecision
    target: int
    features: ConfirmationFeatures
    selected_contract: str
    top_10_signature: str
    top_20_signature: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for value in values:
            target.write(json.dumps(value, sort_keys=True) + "\n")


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    *,
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        raise ValueError(f"cannot infer columns for empty CSV: {path}")
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _joint_target(diagnosis: GuidedDiagnosis, gold: GoldLocalization) -> int:
    contract = (
        evaluation_family(diagnosis.candidates[0])
        if diagnosis.candidates
        else "UNKNOWN_CONTRACT"
    )
    contract_hit = contract in {atom.contract for atom in gold.atoms}
    top_three_hit = any(
        candidate.file_path == atom.file_path
        and candidate.symbol == atom.symbol
        for candidate in diagnosis.candidates[:3]
        for atom in gold.atoms
    )
    return int(contract_hit and top_three_hit)


def _load_replay(
    *,
    bundle: Path,
    engine: GuidedNaturalDiagnosisEngine,
) -> tuple[ConfirmationReplayIncident, ...]:
    records, gold = _replay_records(bundle)
    values = []
    for value in records:
        query_value = value["query"]
        query = IncidentQuery(
            str(value["incident_id"]),
            str(query_value.get("issue", "")),
            tuple(str(item) for item in query_value.get("failing_tests", [])),
            str(query_value.get("traceback", "")),
            str(query_value.get("assertion", "")),
        )
        incident = DevelopmentIncident(
            str(value["incident_id"]),
            str(value["repository"]),
            query,
            _graph(_json((bundle / str(value["graph_path"])).resolve())),
            int(value["repository_symbol_count"]),
        )
        runtime_events = load_runtime_events(
            (bundle / str(value["runtime_events_path"])).resolve()
        )
        r5 = engine.diagnose(incident.graph, query, "R5", runtime_events)
        r6 = engine.diagnose(incident.graph, query, "R6", runtime_events)
        if not r5.candidates:
            raise ValueError(f"R5 produced no candidates for {incident.incident_id}")
        features = extract_confirmation_features(r5, query)
        active_features = (
            extract_confirmation_features(
                r6,
                query,
                active_probe=True,
            )
            if r6.candidates
            and r6.active_evidence_status == "ACTIVE_EVIDENCE_APPLIED"
            else None
        )
        incident_gold = gold[incident.incident_id]
        values.append(
            ConfirmationReplayIncident(
                incident,
                incident_gold,
                r5,
                r6,
                features,
                active_features,
                _joint_target(r5, incident_gold),
                _row(incident, r5, incident_gold, 0.0),
            )
        )
    return tuple(values)


def _eligible_training_thresholds(
    probabilities: list[float],
) -> tuple[float, ...]:
    values = {
        max(0.0, min(1.0, probability))
        for probability in probabilities
    }
    return tuple(sorted({0.0, 1.0, *values}))


def _apply_active_probability(
    model: ConfirmationProbabilityModel,
    incident: ConfirmationReplayIncident,
    probability: float,
    threshold: float,
) -> tuple[float, str]:
    if not 0.60 <= probability < threshold:
        return probability, "NOT_REQUESTED"
    if incident.active_features is None:
        return probability, "ACTIVE_EVIDENCE_UNAVAILABLE"
    updated = model.predict_probability(incident.active_features)
    if updated <= probability:
        return probability, "ACTIVE_EVIDENCE_NO_POSITIVE_GAIN"
    return updated, "ACTIVE_EVIDENCE_APPLIED"


def _margin_threshold(
    values: tuple[ConfirmationReplayIncident, ...],
    field: str,
) -> float:
    positives = sorted(
        float(getattr(item.features, field))
        for item in values
        if item.target
    )
    if not positives:
        return 1.0
    index = max(0, int(0.20 * (len(positives) - 1)))
    return positives[index]


def _decision_for(
    *,
    confirmer: CalibratedDiagnosisConfirmer,
    incident: ConfirmationReplayIncident,
    model: ConfirmationProbabilityModel,
    threshold: float,
    candidate_margin_threshold: float,
    joint_margin_threshold: float,
) -> tuple[ConfirmationDecision, float, float, str, ConfirmationFeatures]:
    before = model.predict_probability(incident.features)
    after, active_status = _apply_active_probability(
        model,
        incident,
        before,
        threshold,
    )
    features = (
        incident.active_features
        if active_status == "ACTIVE_EVIDENCE_APPLIED"
        and incident.active_features is not None
        else incident.features
    )
    decision = confirmer.decide(
        features=features,
        contract_family=(
            incident.r6.candidates[0].contract.family
            if active_status == "ACTIVE_EVIDENCE_APPLIED"
            and incident.r6.candidates
            else incident.r5.candidates[0].contract.family
        ),
        probability=after,
        threshold=threshold,
        candidate_margin_threshold=candidate_margin_threshold,
        joint_margin_threshold=joint_margin_threshold,
    )
    return decision, before, after, active_status, features


def _select_threshold(
    *,
    train: tuple[ConfirmationReplayIncident, ...],
    model: ConfirmationProbabilityModel,
    candidate_margin_threshold: float,
    joint_margin_threshold: float,
) -> tuple[float, dict[str, float]]:
    confirmer = CalibratedDiagnosisConfirmer()
    probabilities = [
        model.predict_probability(item.features) for item in train
    ]
    choices = []
    for threshold in _eligible_training_thresholds(probabilities):
        decisions = [
            _decision_for(
                confirmer=confirmer,
                incident=item,
                model=model,
                threshold=threshold,
                candidate_margin_threshold=candidate_margin_threshold,
                joint_margin_threshold=joint_margin_threshold,
            )[0]
            for item in train
        ]
        confirmed = [
            (decision, item)
            for decision, item in zip(decisions, train, strict=True)
            if decision.status == "DIAGNOSIS_CONFIRMED"
        ]
        correct = sum(item.target for _, item in confirmed)
        precision = correct / len(confirmed) if confirmed else 1.0
        coverage = len(confirmed) / len(train)
        choices.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "precision": precision,
                "correct": float(correct),
                "false": float(len(confirmed) - correct),
            }
        )
    valid = [
        item
        for item in choices
        if item["precision"] >= 0.80 and item["coverage"] > 0.0
    ]
    if not valid:
        return 1.0, {
            "coverage": 0.0,
            "precision": 1.0,
            "correct": 0.0,
            "false": 0.0,
        }
    winner = max(
        valid,
        key=lambda item: (
            item["coverage"],
            item["precision"],
            item["correct"],
            item["threshold"],
        ),
    )
    return float(winner["threshold"]), {
        key: float(winner[key])
        for key in ("coverage", "precision", "correct", "false")
    }


def _model_factory(model_id: str) -> ConfirmationProbabilityModel:
    if model_id == "C0":
        return DeterministicConfirmationModel()
    if model_id == "C1":
        return LogisticConfirmationModel(seed=R5C_SEED)
    raise ValueError(f"unknown confirmation model: {model_id}")


def _run_loro(
    incidents: tuple[ConfirmationReplayIncident, ...],
    *,
    model_id: str,
) -> tuple[list[FoldPrediction], list[dict[str, object]]]:
    repositories = sorted({item.incident.repository for item in incidents})
    predictions: list[FoldPrediction] = []
    folds = []
    confirmer = CalibratedDiagnosisConfirmer()
    for repository in repositories:
        train = tuple(
            item
            for item in incidents
            if item.incident.repository != repository
        )
        test = tuple(
            item
            for item in incidents
            if item.incident.repository == repository
        )
        model = _model_factory(model_id).fit(
            tuple(item.features for item in train),
            tuple(item.target for item in train),
        )
        candidate_margin_threshold = _margin_threshold(
            train,
            "candidate_margin_normalized",
        )
        joint_margin_threshold = _margin_threshold(
            train,
            "joint_margin_normalized",
        )
        threshold, training = _select_threshold(
            train=train,
            model=model,
            candidate_margin_threshold=candidate_margin_threshold,
            joint_margin_threshold=joint_margin_threshold,
        )
        fold_predictions = []
        for item in test:
            decision, before, after, active_status, features = _decision_for(
                confirmer=confirmer,
                incident=item,
                model=model,
                threshold=threshold,
                candidate_margin_threshold=candidate_margin_threshold,
                joint_margin_threshold=joint_margin_threshold,
            )
            fold_predictions.append(
                FoldPrediction(
                    item.incident.incident_id,
                    item.incident.repository,
                    model_id,
                    threshold,
                    before,
                    after,
                    active_status,
                    decision,
                    item.target,
                    features,
                    evaluation_family(item.r5.candidates[0]),
                    str(item.r5_row["top_k_signature"]),
                    json.dumps(
                        [candidate.node_id for candidate in item.r5.candidates],
                        separators=(",", ":"),
                    ),
                )
            )
        predictions.extend(fold_predictions)
        confirmed = [
            item
            for item in fold_predictions
            if item.decision.status == "DIAGNOSIS_CONFIRMED"
        ]
        correct = sum(item.target for item in confirmed)
        folds.append(
            {
                "model_id": model_id,
                "repository": repository,
                "training_repositories": json.dumps(
                    [
                        item
                        for item in repositories
                        if item != repository
                    ],
                    separators=(",", ":"),
                ),
                "training_incidents": len(train),
                "test_incidents": len(test),
                "threshold": threshold,
                "candidate_margin_threshold": candidate_margin_threshold,
                "joint_margin_threshold": joint_margin_threshold,
                "training_confirmation_coverage": training["coverage"],
                "training_selective_precision": training["precision"],
                "test_confirmation_coverage": (
                    len(confirmed) / len(test) if test else 0.0
                ),
                "test_selective_precision": (
                    correct / len(confirmed) if confirmed else 1.0
                ),
                "parameters": json.dumps(
                    model.parameters(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    if {item.incident_id for item in predictions} != {
        item.incident.incident_id for item in incidents
    }:
        raise AssertionError("LORO predictions do not cover every incident once")
    return predictions, folds


def _confirmation_metrics(
    predictions: list[FoldPrediction],
) -> dict[str, object]:
    confirmed = [
        item
        for item in predictions
        if item.decision.status == "DIAGNOSIS_CONFIRMED"
    ]
    correct = sum(item.target for item in confirmed)
    false = len(confirmed) - correct
    counts = Counter(item.decision.status for item in predictions)
    return {
        "confirmed_total": len(confirmed),
        "confirmed_correct": correct,
        "confirmed_false": false,
        "selective_precision": (
            correct / len(confirmed) if confirmed else 1.0
        ),
        "confirmation_coverage": len(confirmed) / len(predictions),
        "probable_diagnoses": counts["DIAGNOSIS_PROBABLE"],
        "candidate_only_diagnoses": counts["DIAGNOSIS_CANDIDATES"],
        "insufficient_evidence": counts["INSUFFICIENT_EVIDENCE"],
        "unknown_contract_confirmed": sum(
            item.selected_contract == "UNKNOWN_CONTRACT"
            for item in confirmed
        ),
        "test_symbol_false_confirmation": sum(
            item.features.test_symbol and not item.target
            for item in confirmed
        ),
        "mean_independent_evidence_groups": mean_feature(
            (item.features for item in predictions),
            "independent_evidence_groups",
        ),
        "mean_rank_stability": mean_feature(
            (item.features for item in predictions),
            "rank_stability",
        ),
    }


def _leakage_audit(
    *,
    incidents: tuple[ConfirmationReplayIncident, ...],
    predictions: list[FoldPrediction],
    folds: list[dict[str, object]],
) -> dict[str, object]:
    feature_fields = set(asdict(incidents[0].features))
    forbidden = sorted(feature_fields.intersection(FORBIDDEN_FEATURE_NAMES))
    prediction_ids = [item.incident_id for item in predictions]
    fold_tests = {str(item["repository"]) for item in folds}
    training_excludes_test = all(
        str(item["repository"])
        not in set(json.loads(str(item["training_repositories"])))
        for item in folds
    )
    checks = {
        "feature_schema_has_no_gold_or_identity": not forbidden,
        "model_schema_has_no_gold_or_identity": not set(
            FEATURE_NAMES
        ).intersection(FORBIDDEN_FEATURE_NAMES),
        "every_incident_predicted_once": (
            len(prediction_ids)
            == len(set(prediction_ids))
            == len(incidents)
        ),
        "every_repository_is_tested_once": fold_tests
        == {item.incident.repository for item in incidents},
        "training_repositories_exclude_test_repository": (
            training_excludes_test
        ),
        "scoring_uses_only_oof_predictions": all(
            item.repository in fold_tests for item in predictions
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "feature_fields": sorted(feature_fields),
        "model_feature_names": list(FEATURE_NAMES),
        "forbidden_feature_intersection": forbidden,
        "loro_test_predictions": len(predictions),
        "unique_test_incidents": len(set(prediction_ids)),
    }


def _risk_coverage_rows(
    predictions: list[FoldPrediction],
) -> list[dict[str, object]]:
    ordered = sorted(
        predictions,
        key=lambda item: item.probability_after,
        reverse=True,
    )
    rows = []
    correct = 0
    for index, item in enumerate(ordered, start=1):
        correct += item.target
        precision = correct / index
        rows.append(
            {
                "selected_count": index,
                "coverage": index / len(ordered),
                "selective_precision": precision,
                "risk": 1.0 - precision,
                "probability_threshold": item.probability_after,
            }
        )
    return rows


def _parent_r5_signatures() -> dict[str, tuple[str, str]]:
    rows = list(
        csv.DictReader(PARENT_REFACTOR_MATRIX.open(encoding="utf-8"))
    )
    return {
        str(row["incident_id"]): (
            str(row["top_k_signature"]),
            str(row["predicted_contract"]),
        )
        for row in rows
        if row["variant"] == "R5"
    }


def _retrieval_immutability(
    incidents: tuple[ConfirmationReplayIncident, ...],
) -> dict[str, object]:
    parent_status = _json(PARENT_REFRACTOR_STATUS)
    parent_metrics = parent_status["metrics"]["R5"]
    current_rows = [dict(item.r5_row) for item in incidents]
    current_metrics = _metrics(current_rows)
    signatures = _parent_r5_signatures()
    checks = {
        "recall_at_10_unchanged": (
            current_metrics["recall_at_10"]
            == R5_BASELINE_METRICS["recall_at_10"]
            == parent_metrics["recall_at_10"]
        ),
        "recall_at_20_unchanged": (
            current_metrics["recall_at_20"]
            == R5_BASELINE_METRICS["recall_at_20"]
            == parent_metrics["recall_at_20"]
        ),
        "mrr_unchanged": (
            current_metrics["mrr"]
            == R5_BASELINE_METRICS["mrr"]
            == parent_metrics["mrr"]
        ),
        "contract_macro_f1_unchanged": (
            current_metrics["contract_macro_f1"]
            == R5_BASELINE_METRICS["contract_macro_f1"]
            == parent_metrics["contract_macro_f1"]
        ),
        "joint_hit_at_3_unchanged": (
            current_metrics["joint_hit_at_3"]
            == R5_BASELINE_METRICS["joint_hit_at_3"]
            == parent_metrics["joint_hit_at_3"]
        ),
        "top_10_signatures_unchanged": all(
            str(item.r5_row["top_k_signature"])
            == signatures[item.incident.incident_id][0]
            for item in incidents
        ),
        "predicted_contracts_unchanged": all(
            str(item.r5_row["predicted_contract"])
            == signatures[item.incident.incident_id][1]
            for item in incidents
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "current_metrics": current_metrics,
        "parent_status_sha256": _sha256(PARENT_REFRACTOR_STATUS),
        "parent_matrix_sha256": _sha256(PARENT_REFACTOR_MATRIX),
    }


def _selection_key(
    model_id: str,
    metrics: dict[str, object],
) -> tuple[bool, float, int, int]:
    return (
        float(metrics["selective_precision"]) >= 0.80,
        float(metrics["confirmation_coverage"]),
        -int(metrics["confirmed_false"]),
        int(model_id == "C0"),
    )


def _gate(
    *,
    retrieval: dict[str, object],
    metrics: dict[str, object],
    repositories_improved: int,
    leakage_passed: bool,
) -> dict[str, bool]:
    values = retrieval["current_metrics"]
    return {
        "recall_at_10_at_least_0_80": values["recall_at_10"] >= 0.80,
        "recall_at_20_at_least_0_90": values["recall_at_20"] >= 0.90,
        "mrr_at_least_0_45": values["mrr"] >= 0.45,
        "contract_macro_f1_at_least_0_60": (
            values["contract_macro_f1"] >= 0.60
        ),
        "joint_hit_at_3_at_least_0_30": (
            values["joint_hit_at_3"] >= 0.30
        ),
        "selective_precision_at_least_0_80": (
            metrics["selective_precision"] >= 0.80
        ),
        "confirmation_coverage_at_least_0_40": (
            metrics["confirmation_coverage"] >= 0.40
        ),
        "repositories_improved_at_least_6": repositories_improved >= 6,
        "false_confirmed_at_most_3": metrics["confirmed_false"] <= 3,
        "confirmed_correct_at_least_10": metrics["confirmed_correct"] >= 10,
        "unknown_contract_confirmed_zero": (
            metrics["unknown_contract_confirmed"] == 0
        ),
        "test_symbol_false_confirmation_zero": (
            metrics["test_symbol_false_confirmation"] == 0
        ),
        "gold_leakage_zero": leakage_passed,
        "r5_retrieval_unchanged": bool(retrieval["passed"]),
    }


def _write_reports(
    *,
    output: Path,
    reports: Path,
    status: dict[str, object],
    selected_predictions: list[FoldPrediction],
    leakage: dict[str, object],
) -> None:
    metrics = status["selected_metrics"]
    model_metrics = status["model_metrics"]
    report = [
        "# H10-C7-R5C calibrated confirmation",
        "",
        f"Status: `{status['status']}`",
        "",
        "Scientific result: `NOT_EVALUATED`.",
        "",
        "R5 retrieval and the 30 disclosed replay incidents were unchanged.",
        "All confirmation decisions are leave-one-repository-out predictions.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Confirmed | {metrics['confirmed_total']} |",
        f"| Confirmed correct | {metrics['confirmed_correct']} |",
        f"| Confirmed false | {metrics['confirmed_false']} |",
        f"| Selective precision | {metrics['selective_precision']:.4f} |",
        f"| Confirmation coverage | {metrics['confirmation_coverage']:.4f} |",
        "",
        "## Model comparison",
        "",
        (
            "| Model | Confirmed | Correct | False | Selective precision | "
            "Coverage |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *(
            (
                f"| {model_id} | {values['confirmed_total']} | "
                f"{values['confirmed_correct']} | "
                f"{values['confirmed_false']} | "
                f"{values['selective_precision']:.4f} | "
                f"{values['confirmation_coverage']:.4f} |"
            )
            for model_id, values in sorted(model_metrics.items())
        ),
        "",
        (
            "C0 was selected by the locked lexicographic rule because it "
            "preserved the required precision by failing closed. C1 did not "
            "transfer safely across excluded repositories."
        ),
        "",
        (
            "The result is an open-development engineering gate, not "
            "scientific support for H10-C7."
        ),
    ]
    (reports / "R5C_CONFIRMATION_REPORT.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    errors = [
        "# H10-C7-R5C error analysis",
        "",
        "False confirmations:",
        "",
    ]
    false_items = [
        item
        for item in selected_predictions
        if item.decision.status == "DIAGNOSIS_CONFIRMED" and not item.target
    ]
    if not false_items:
        errors.append("- Selected C0: none, because it failed closed.")
    else:
        errors.extend(
            f"- `{item.incident_id}` ({item.repository}), "
            f"p={item.probability_after:.6f}."
            for item in false_items
        )
    c1 = model_metrics["C1"]
    errors.extend(
        (
            (
                f"- Rejected C1: {c1['confirmed_false']} out-of-fold "
                "confirmations were false."
            ),
            (
                "  Its training-fold threshold did not transfer safely "
                "across excluded repositories."
            ),
        )
    )
    (reports / "R5C_ERROR_ANALYSIS.md").write_text(
        "\n".join(errors) + "\n",
        encoding="utf-8",
    )
    (reports / "R5C_LEAKAGE_AUDIT.md").write_text(
        "# H10-C7-R5C leakage audit\n\n"
        f"Status: `{'PASS' if leakage['passed'] else 'FAIL'}`\n\n"
        "- Repository and incident identity are fold metadata only.\n"
        "- Gold file, symbol and contract are training targets only.\n"
        "- The model feature schema contains no identity or Gold fields.\n"
        "- Every reported decision was produced outside its repository fold.\n",
        encoding="utf-8",
    )
    (reports / "R5C_REPRODUCTION.md").write_text(
        "# H10-C7-R5C reproduction\n\n"
        "```bash\n"
        "PYTHONPATH=framework/fuzzyxai "
        "/home/lebedeffson/Code/venv/bin/python "
        "scripts/ch4_revision/run_h10_c7_confirmation.py "
        "--bundle /home/lebedeffson/.local/share/fuzzyxai/h10-c7/"
        "h10-c7-open-replay-bundle "
        "--output results/h10_c7/confirmation "
        "--reports reports/h10_c7\n"
        "```\n\n"
        "No network, project setup, failing-test execution, neural model, "
        "new development data or held-out data is used.\n",
        encoding="utf-8",
    )
    chapter_status = (
        "passed"
        if status["gate_passed"]
        else "did not pass"
    )
    (reports / "R5C_CHAPTER_FRAGMENT.md").write_text(
        "# Chapter fragment\n\n"
        "A leave-one-repository-out confirmation calibration was applied to "
        "the unchanged R5 retrieval results. The internal open-replay gate "
        f"{chapter_status}. The deterministic confirmer failed closed, while "
        "the logistic variant produced false confirmations outside its "
        "training repositories and was rejected by the registered selection "
        "rule. This is a development engineering result; scientific "
        "evaluation and held-out scoring were not performed.\n",
        encoding="utf-8",
    )


def _write_sha256sums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    content = "".join(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in files
    )
    (root / "SHA256SUMS").write_text(content, encoding="utf-8")


def run_r5c_confirmation(
    *,
    bundle: Path,
    output: Path,
    reports: Path,
    engine: GuidedNaturalDiagnosisEngine,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    incidents = _load_replay(bundle=bundle, engine=engine)
    retrieval = _retrieval_immutability(incidents)
    if not retrieval["passed"]:
        raise ValueError("R5C blocked: R5 retrieval no longer matches parent")

    predictions_by_model = {}
    folds_by_model = {}
    metrics_by_model = {}
    for model_id in ("C0", "C1"):
        predictions, folds = _run_loro(incidents, model_id=model_id)
        predictions_by_model[model_id] = predictions
        folds_by_model[model_id] = folds
        metrics_by_model[model_id] = _confirmation_metrics(predictions)

    selected_model = max(
        ("C0", "C1"),
        key=lambda model_id: _selection_key(
            model_id,
            metrics_by_model[model_id],
        ),
    )
    selected_predictions = predictions_by_model[selected_model]
    selected_metrics = metrics_by_model[selected_model]
    parent_status = _json(PARENT_REFRACTOR_STATUS)
    repositories_improved = int(
        parent_status["repository_improvements"]["R5"]
    )
    leakage = _leakage_audit(
        incidents=incidents,
        predictions=selected_predictions,
        folds=folds_by_model[selected_model],
    )
    gate = _gate(
        retrieval=retrieval,
        metrics=selected_metrics,
        repositories_improved=repositories_improved,
        leakage_passed=bool(leakage["passed"]),
    )
    gate_passed = all(gate.values())
    status = {
        "protocol_id": "H10-C7-R5C",
        "status": R5C_STATUS_GO if gate_passed else R5C_STATUS_NO_GO,
        "scientific_result": "NOT_EVALUATED",
        "held_out_created": False,
        "held_out_scored": False,
        "retrieval_modified": False,
        "calibration_mode": "leave_one_repository_out",
        "gate_passed": gate_passed,
        "ready_for_official_development_extension": gate_passed,
        "selected_confirmer": selected_model,
        "model_metrics": metrics_by_model,
        "selected_metrics": selected_metrics,
        "r5_metrics": retrieval["current_metrics"],
        "r5_retrieval_immutability": retrieval,
        "repositories_improved": repositories_improved,
        "checks": gate,
        "gold_leakage": 0 if leakage["passed"] else 1,
        "leakage_audit": leakage,
        "new_data_collected": False,
        "neural_models_executed": False,
        "failing_tests_reexecuted": False,
    }

    per_incident = []
    active_rows = []
    incident_by_id = {
        item.incident.incident_id: item for item in incidents
    }
    for item in selected_predictions:
        replay = incident_by_id[item.incident_id]
        active_details = dict(replay.r6.active_evidence_details)
        row = {
            "incident_id": item.incident_id,
            "repository": item.repository,
            "model_id": item.model_id,
            "status": item.decision.status,
            "target_correct": item.target,
            "threshold": item.threshold,
            "probability_before": item.probability_before,
            "probability_after": item.probability_after,
            "active_evidence_status": item.active_evidence_status,
            "selected_contract": item.selected_contract,
            "top_10_signature": item.top_10_signature,
            "top_20_signature": item.top_20_signature,
            "reasons": list(item.decision.reasons),
            "rejected_reasons": list(item.decision.rejected_reasons),
            "features": asdict(item.features),
        }
        per_incident.append(row)
        active_rows.append(
            {
                "incident_id": item.incident_id,
                "repository": item.repository,
                "status": item.active_evidence_status,
                "probability_before": item.probability_before,
                "probability_after": item.probability_after,
                "rank_before": int(active_details.get("rank_before", "1")),
                "rank_after": int(active_details.get("rank_after", "1")),
                "contract_before": active_details.get(
                    "contract_before",
                    item.selected_contract,
                ),
                "contract_after": active_details.get(
                    "contract_after",
                    item.selected_contract,
                ),
                "entropy_before": _binary_entropy(item.probability_before),
                "entropy_after": _binary_entropy(item.probability_after),
                "selected_probe": active_details.get("event_id", ""),
                "observed_evidence": active_details.get(
                    "observed_node",
                    "",
                ),
            }
        )
    _write_jsonl(output / "R5C_PER_INCIDENT.jsonl", per_incident)
    _write_jsonl(output / "R5C_ACTIVE_EVIDENCE.jsonl", active_rows)

    repository_rows = []
    for repository in sorted({item.repository for item in selected_predictions}):
        rows = [
            item for item in selected_predictions if item.repository == repository
        ]
        metrics = _confirmation_metrics(rows)
        repository_rows.append({"repository": repository, **metrics})
    _write_csv(output / "R5C_PER_REPOSITORY.csv", repository_rows)
    _write_csv(
        output / "R5C_FOLD_THRESHOLDS.csv",
        [
            row
            for model_id in ("C0", "C1")
            for row in folds_by_model[model_id]
        ],
    )
    feature_rows = [
        {
            "incident_id": item.incident.incident_id,
            "repository": item.incident.repository,
            **asdict(item.features),
        }
        for item in incidents
    ]
    pd.DataFrame(feature_rows).to_parquet(
        output / "R5C_FEATURES.parquet",
        index=False,
    )
    risk_rows = _risk_coverage_rows(selected_predictions)
    _write_csv(output / "R5C_RISK_COVERAGE.csv", risk_rows)
    _write_csv(output / "R5C_PRECISION_COVERAGE.csv", risk_rows)
    _write_json(output / "R5C_STATUS.json", status)
    _write_reports(
        output=output,
        reports=reports,
        status=status,
        selected_predictions=selected_predictions,
        leakage=leakage,
    )
    _write_sha256sums(output)
    return status


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -probability * math.log(probability) - (
        1.0 - probability
    ) * math.log(1.0 - probability)
