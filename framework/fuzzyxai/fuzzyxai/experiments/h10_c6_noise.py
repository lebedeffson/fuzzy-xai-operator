from __future__ import annotations

import hashlib
import io
import json
import statistics
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fuzzyxai import FuzzyXAI
from fuzzyxai.adapters import get_adapter
from fuzzyxai.diagnostics import RepairExecutionContext, RouteGraph
from fuzzyxai.diagnostics.contracts import canonical_sha256

DATASET_PATH = Path("data/h10_c6_noise/bank_marketing_uci_222.zip")
LOCK_PATH = Path("results/h10_c6_noise/H10_C6_N_PROTOCOL_LOCK.json")
OBJECT_IDS_PATH = Path("results/h10_c6_noise/H10_C6_N_OBJECT_IDS.json")
NOISE_LEVELS = (0.01, 0.05, 0.10)
NUMERIC_COLUMNS = (
    "age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
)
SEED = 6101
MODEL_VERSION = "bank-logistic-v1"
EXPLAINER_VERSION = "linear-coefficient-v1"
DEFECT_FAMILY = "MODEL_EXPLAINER_VERSION"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_dataset(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as outer:
        nested = outer.read("bank-additional.zip")
    with zipfile.ZipFile(io.BytesIO(nested)) as inner:
        raw = inner.read("bank-additional/bank-additional-full.csv")
    frame = pd.read_csv(io.BytesIO(raw), sep=";")
    if len(frame) != 41188 or "y" not in frame:
        raise ValueError("unexpected UCI Bank Marketing payload")
    return frame


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    indexes = np.arange(len(frame))
    train, test = train_test_split(
        indexes, test_size=0.30, random_state=SEED, stratify=frame["y"]
    )
    return frame.iloc[train].copy(), frame.iloc[test].copy()


def _object_id(index: int, row: pd.Series) -> str:
    values = {
        str(key): value.item() if hasattr(value, "item") else value
        for key, value in row.items()
    }
    return hashlib.sha256(
        json.dumps(
            {"source_row": int(index), "values": values},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def prepare_noise_protocol(root: Path) -> dict[str, object]:
    dataset = root / DATASET_PATH
    frame = _load_dataset(dataset)
    train, test = _split(frame)
    selected = sorted(
        (_object_id(int(index), row), int(index))
        for index, row in test.iterrows()
    )[:1000]
    objects = {
        "protocol_id": "h10-c6-n-feature-noise-v1",
        "selection": "first 1000 test objects by SHA256(content and source row)",
        "selection_uses_outcome": False,
        "random_seed": SEED,
        "object_count": len(selected),
        "objects": [
            {"object_id": object_id, "source_row": source_row}
            for object_id, source_row in selected
        ],
    }
    object_path = root / OBJECT_IDS_PATH
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_text(
        json.dumps(objects, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    scales = {
        column: {
            "standard_deviation": float(train[column].std(ddof=0)),
            "minimum": float(train[column].min()),
            "maximum": float(train[column].max()),
        }
        for column in NUMERIC_COLUMNS
    }
    lock = {
        "protocol_id": "h10-c6-n-feature-noise-v1",
        "status": "LOCKED_BEFORE_EXECUTION",
        "dataset": "UCI Bank Marketing dataset 222, bank-additional-full",
        "dataset_relative_path": DATASET_PATH.as_posix(),
        "dataset_sha256": _sha256(dataset),
        "dataset_rows": len(frame),
        "training_rows": len(train),
        "test_rows": len(test),
        "object_count": 1000,
        "object_ids_sha256": _sha256(object_path),
        "selection_uses_outcome": False,
        "split_random_seed": SEED,
        "noise_random_seed": SEED,
        "noise_levels": list(NOISE_LEVELS),
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_and_binary_noise": "disabled",
        "feature_scales_source": "training split only",
        "feature_scales": scales,
        "feature_scales_sha256": canonical_sha256(scales),
        "model": "scikit-learn LogisticRegression with train-only preprocessing",
        "model_version": MODEL_VERSION,
        "scikit_learn_version": sklearn.__version__,
        "explainer": "linear coefficient contribution in transformed feature space",
        "explainer_version": EXPLAINER_VERSION,
        "registered_defect_family": DEFECT_FAMILY,
        "cost_model": "diagnostic default registered cost",
        "contract_registry_version": "diagnostic-schema-1.0",
        "primary_endpoint": "median_jaccard_sigma_0.05",
        "primary_threshold": 0.80,
        "empty_base_cut_is_evidence": False,
        "allowed_statuses": [
            "H10_C6_N_SUPPORTED", "H10_C6_N_NOT_SUPPORTED", "H10_C6_N_BLOCKED"
        ],
        "parent_h10_c6_modified": False,
        "human_or_industrial_claim_permitted": False,
    }
    (root / LOCK_PATH).write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return lock


def _validate_lock(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    lock = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    objects = json.loads((root / OBJECT_IDS_PATH).read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_EXECUTION":
        raise ValueError("H10-C6-N protocol is not locked")
    if lock.get("dataset_sha256") != _sha256(root / DATASET_PATH):
        raise ValueError("H10-C6-N dataset hash mismatch")
    if lock.get("object_ids_sha256") != _sha256(root / OBJECT_IDS_PATH):
        raise ValueError("H10-C6-N object selection hash mismatch")
    if objects.get("object_count") != 1000:
        raise ValueError("H10-C6-N requires exactly 1000 selected objects")
    return lock, objects


def _fit_model(train: pd.DataFrame) -> Pipeline:
    features = [column for column in train if column != "y"]
    categorical = [column for column in features if column not in NUMERIC_COLUMNS]
    preprocessing = ColumnTransformer(
        (
            ("numeric", StandardScaler(), list(NUMERIC_COLUMNS)),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        )
    )
    model = Pipeline(
        (
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
        )
    )
    model.fit(train[features], (train["y"] == "yes").astype(int))
    return model


def _explain(model: Pipeline, row: pd.DataFrame) -> dict[str, float]:
    transformed = model.named_steps["preprocessing"].transform(row)
    values = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    coefficients = model.named_steps["model"].coef_[0]
    names = model.named_steps["preprocessing"].get_feature_names_out()
    ranked = sorted(
        zip(names, values * coefficients, strict=True),
        key=lambda item: (-abs(float(item[1])), str(item[0])),
    )[:20]
    return {str(name): float(value) for name, value in ranked}


def _diagnostic_route(object_id: str, input_digest: str, explanation_digest: str) -> dict[str, object]:
    return {
        "route_id": f"h10-c6-n:{object_id}",
        "metadata": {
            "dataset_id": "uci-bank-marketing-222",
            "object_id": object_id,
            "input_digest": input_digest,
            "explanation_digest": explanation_digest,
        },
        "nodes": [
            {
                "node_id": "adapter", "node_type": "preprocessing",
                "component_version": "bank-schema-v1",
                "registered_attributes": {"schema": "bank-additional-v1"},
                "observed_attributes": {"schema": "bank-additional-v1"},
                "mandatory": True, "repairable": True,
                "evidence_refs": [f"object:{object_id}:adapter"],
            },
            {
                "node_id": "model", "node_type": "model",
                "component_version": MODEL_VERSION,
                "registered_attributes": {"version": MODEL_VERSION},
                "observed_attributes": {"version": MODEL_VERSION},
                "mandatory": True, "repairable": True,
                "evidence_refs": [f"object:{object_id}:model"],
            },
            {
                "node_id": "explainer", "node_type": "explainer",
                "component_version": EXPLAINER_VERSION,
                "registered_attributes": {"version": EXPLAINER_VERSION},
                "observed_attributes": {"version": "linear-coefficient-v0"},
                "mandatory": True, "repairable": True,
                "evidence_refs": [f"object:{object_id}:explainer"],
            },
        ],
        "edges": [
            {
                "edge_id": "adapter-to-model", "source": "adapter", "target": "model",
                "relation": "transforms", "relation_status": "known_valid",
                "mandatory": True, "registered_contract": {"compatible": True},
                "observed_contract": {"compatible": True}, "repairable": True,
                "evidence_refs": [f"object:{object_id}:adapter-model"],
            },
            {
                "edge_id": "model-to-explainer", "source": "model", "target": "explainer",
                "relation": "explains", "relation_status": "known_valid",
                "mandatory": True, "registered_contract": {"model_version": MODEL_VERSION},
                "observed_contract": {"model_version": MODEL_VERSION}, "repairable": True,
                "evidence_refs": [f"object:{object_id}:model-explainer"],
            },
        ],
    }


def _execute_route(route: dict[str, object]) -> object:
    planned = FuzzyXAI().diagnose(route=route, repair_mode="plan")
    if planned.repair_plan is None or planned.minimal_cut is None:
        return planned

    def restore(graph: RouteGraph, step: object) -> RouteGraph:
        nodes = tuple(
            replace(node, observed_attributes=dict(node.registered_attributes))
            if node.node_id == step.target.subject_id else node
            for node in graph.nodes
        )
        return replace(graph, nodes=nodes)

    context = RepairExecutionContext(
        handlers={step.operation: restore for step in planned.repair_plan.steps},
        approved_step_ids=frozenset(step.step_id for step in planned.repair_plan.steps),
        allow_external_changes=True,
    )
    return FuzzyXAI().diagnose(route=route, repair_mode="execute", repair_context=context)


def _cut(report: object) -> tuple[str, ...]:
    return report.minimal_cut.atom_keys if report.minimal_cut is not None else ()


def _covered(report: object) -> tuple[str, ...]:
    return report.minimal_cut.covered_obligations if report.minimal_cut is not None else ()


def _jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    a, b = set(left), set(right)
    if not a:
        raise ValueError("empty baseline cut is excluded from H10-C6-N")
    return len(a & b) / len(a | b) if a | b else 0.0


def _recertified(report: object) -> bool:
    recertification = report.recertification
    return bool(
        recertification is not None
        and recertification.status == "full_success"
        and not recertification.remaining_critical_issues
        and not recertification.new_critical_issues
        and recertification.all_required_postconditions_verified
    )


def _payload(row: pd.DataFrame, probability: float, prediction: int, explanation: dict[str, float]) -> dict[str, object]:
    values = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in row.iloc[0].items()
    }
    return {
        "scenario_id": "external_wine_classification",
        "source_type": "registered_h10_c6_n_bank_route",
        "model_name": MODEL_VERSION,
        "dataset_name": "UCI Bank Marketing 222",
        "predicted_class": prediction,
        "class_probability": probability,
        "feature_values": values,
        "feature_importance": explanation,
        "quality_metrics": {"missing_rate": 0.0, "feature_range_violation": 0.0},
    }


def _run_object(
    frame: pd.DataFrame,
    model: Pipeline,
    adapter: object,
    selected: dict[str, object],
    scales: dict[str, dict[str, float]],
    ordinal: int,
) -> tuple[list[dict[str, object]], dict[str, str] | None]:
    object_id, source_row = str(selected["object_id"]), int(selected["source_row"])
    features = [column for column in frame if column != "y"]
    base = frame.loc[[source_row], features].copy()
    base_probability = float(model.predict_proba(base)[0, 1])
    base_prediction = int(base_probability >= 0.5)
    base_explanation = _explain(model, base)
    base_operator = FuzzyXAI().run_payload(
        _payload(base, base_probability, base_prediction, base_explanation), adapter
    )
    base_report = _execute_route(
        _diagnostic_route(
            object_id, canonical_sha256(base.iloc[0].to_dict()), canonical_sha256(base_explanation)
        )
    )
    base_cut = _cut(base_report)
    if not base_cut:
        return [], {"object_id": object_id, "reason": "EMPTY_BASELINE_DIAGNOSTIC_CUT"}
    rows: list[dict[str, object]] = []
    for sigma in NOISE_LEVELS:
        rng = np.random.default_rng(SEED + ordinal * 1009 + int(sigma * 1000))
        changed = base.copy()
        for column in NUMERIC_COLUMNS:
            scale = scales[column]
            value = float(changed.iloc[0][column])
            value += sigma * scale["standard_deviation"] * float(rng.normal())
            changed.loc[changed.index[0], column] = np.clip(value, scale["minimum"], scale["maximum"])
        probability = float(model.predict_proba(changed)[0, 1])
        prediction = int(probability >= 0.5)
        explanation = _explain(model, changed)
        operator = FuzzyXAI().run_payload(_payload(changed, probability, prediction, explanation), adapter)
        report = _execute_route(
            _diagnostic_route(
                object_id, canonical_sha256(changed.iloc[0].to_dict()), canonical_sha256(explanation)
            )
        )
        recertification = report.recertification
        rows.append(
            {
                "object_id": object_id,
                "source_row": source_row,
                "sigma": sigma,
                "base_cut": json.dumps(base_cut),
                "perturbed_cut": json.dumps(_cut(report)),
                "cut_jaccard": _jaccard(base_cut, _cut(report)),
                "coverage_jaccard": _jaccard(_covered(base_report), _covered(report)),
                "recertification_success": _recertified(report),
                "new_critical_violation_count": len(recertification.new_critical_issues) if recertification else 1,
                "prediction_unchanged": prediction == base_prediction,
                "representation_class_unchanged": operator.computed_result.get("representation_class") == base_operator.computed_result.get("representation_class"),
                "diagnostic_status_unchanged": report.route_status == base_report.route_status,
                "base_probability": base_probability,
                "perturbed_probability": probability,
                "structural_defect_family": DEFECT_FAMILY,
            }
        )
    return rows, None


def run_noise_experiment(root: Path) -> dict[str, object]:
    lock, object_manifest = _validate_lock(root)
    frame = _load_dataset(root / DATASET_PATH)
    train, _ = _split(frame)
    model = _fit_model(train)
    adapter = get_adapter("tabular_classification")
    rows: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    scales = lock["feature_scales"]
    selected = object_manifest["objects"]
    for ordinal, item in enumerate(selected):
        object_rows, exclusion = _run_object(frame, model, adapter, item, scales, ordinal)
        rows.extend(object_rows)
        if exclusion:
            excluded.append(exclusion)
    output = root / "results/h10_c6_noise"
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "H10_C6_N_RUNS.parquet"
    pd.DataFrame(rows).to_parquet(run_path, index=False)
    summaries = []
    for sigma in NOISE_LEVELS:
        subset = [row for row in rows if row["sigma"] == sigma]
        summaries.append(
            {
                "noise_level": sigma,
                "N": len(subset),
                "median_jaccard": statistics.median(float(row["cut_jaccard"]) for row in subset),
                "mean_coverage_stability": statistics.fmean(float(row["coverage_jaccard"]) for row in subset),
                "recertification_success": statistics.fmean(float(row["recertification_success"]) for row in subset),
                "new_critical_violations": sum(int(row["new_critical_violation_count"]) for row in subset),
                "prediction_unchanged": statistics.fmean(float(row["prediction_unchanged"]) for row in subset),
                "representation_class_unchanged": statistics.fmean(float(row["representation_class_unchanged"]) for row in subset),
                "diagnostic_status_unchanged": statistics.fmean(float(row["diagnostic_status_unchanged"]) for row in subset),
            }
        )
    summary_path = output / "H10_C6_N_SUMMARY.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    primary = next(row for row in summaries if row["noise_level"] == 0.05)
    complete = len(selected) == 1000 and not excluded and len(rows) == 3000
    new_critical = sum(int(row["new_critical_violation_count"]) for row in rows)
    supported = complete and float(primary["median_jaccard"]) >= 0.80 and new_critical == 0
    status = "H10_C6_N_SUPPORTED" if supported else "H10_C6_N_NOT_SUPPORTED" if complete else "H10_C6_N_BLOCKED"
    final = {
        "protocol_id": lock["protocol_id"],
        "status": status,
        "dataset": lock["dataset"],
        "objects_registered": len(selected),
        "objects_analyzed": len(selected) - len(excluded),
        "excluded_objects": excluded,
        "noise_levels": list(NOISE_LEVELS),
        "run_count": len(rows),
        "median_jaccard_sigma_0.05": primary["median_jaccard"],
        "coverage_stability_sigma_0.05": primary["mean_coverage_stability"],
        "recertification_sigma_0.05": primary["recertification_success"],
        "new_critical_violations": new_critical,
        "feature_scale_leakage": False,
        "parent_h10_c6_modified": False,
        "scientific_scope": "diagnostic-cut stability under registered numeric feature noise with a fixed structural defect",
        "human_or_industrial_claim_permitted": False,
        "artifacts": {
            "runs_sha256": _sha256(run_path),
            "summary_sha256": _sha256(summary_path),
            "lock_sha256": _sha256(root / LOCK_PATH),
            "object_ids_sha256": _sha256(root / OBJECT_IDS_PATH),
        },
    }
    (output / "H10_C6_N_FINAL_STATUS.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_reports(root, final, summaries)
    return final


def _write_reports(root: Path, final: dict[str, object], summaries: list[dict[str, object]]) -> None:
    report_root = root / "reports/h10_c6_noise"
    report_root.mkdir(parents=True, exist_ok=True)
    table = [
        "| Noise | N | Median Jaccard | Coverage | Recertification | New critical |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        table.append(
            "| {noise_level:.2f} | {N} | {median_jaccard:.6f} | {mean_coverage_stability:.6f} | {recertification_success:.6f} | {new_critical_violations} |".format(**row)
        )
    report = [
        "# H10-C6-N Feature Noise Robustness", "",
        f"- Status: `{final['status']}`",
        f"- Dataset: `{final['dataset']}`",
        f"- Registered objects: `{final['objects_registered']}`",
        f"- Executed comparisons: `{final['run_count']}`",
        "- Scope: diagnostic-cut stability under numeric feature noise with one fixed registered structural defect.",
        "- This is not a general predictor-robustness, human-time, or industrial-performance claim.",
        "", *table, "",
    ]
    (report_root / "H10_C6_N_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    (report_root / "H10_C6_N_TABLE_FOR_CHAPTER.md").write_text("\n".join(table) + "\n", encoding="utf-8")
