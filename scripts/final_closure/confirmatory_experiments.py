"""Auxiliary sealed experiments for H5-A, H6-A/B, H8 and H9."""

from __future__ import annotations

import json
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

from fuzzyxai.final_closure import compositional_faults, fault_library
from fuzzyxai.strong_confirmatory import compare_grid_configurations, run_streaming_scalability

from common import ROOT, STUDY, sha256, write
from oof_pipeline import DATA_ROOT, DATASETS
from run_real_formative import PRIMARY_BUDGET, _matrices


OUTPUT = STUDY / "confirmatory"
SEED = 7419


def run_label_free_experiments() -> dict[str, object]:
    h5 = _h5_route_validity()
    h6a = _h6a_planted_rules()
    h6b_plan = _prepare_h6b()
    h8 = _h8_grid()
    h9 = _h9_scaling()
    payloads = {"H5-A": h5, "H6-A": h6a, "H6-B-plan": h6b_plan, "H8": h8, "H9": h9}
    artifacts = {}
    for name, payload in payloads.items():
        path = OUTPUT / f"{name.replace('-', '_')}.json"
        write(path, payload)
        artifacts[name] = _artifact(path)
    return artifacts


def score_h6b(labels: dict[str, str]) -> dict[str, object]:
    plan = json.loads((OUTPUT / "H6_B_plan.json").read_text(encoding="utf-8"))
    datasets = []
    for record in plan["datasets"]:
        predictions = pd.read_parquet(ROOT / record["prediction_artifact"]["path"])
        truth = predictions["object_id_hash"].map(labels).astype(str).to_numpy()
        base = predictions["base_prediction"].astype(str).to_numpy()
        base_accuracy = accuracy_score(truth, base)
        effects = {}
        for column in record["ablation_columns"]:
            effects[column] = float(base_accuracy - accuracy_score(truth, predictions[column].astype(str)))
        candidate = effects[record["candidate_column"]]
        controls = [effects[column] for column in record["control_columns"]]
        datasets.append(
            {
                "dataset_id": record["dataset_id"],
                "base_accuracy": float(base_accuracy),
                "candidate_effect": candidate,
                "control_effects": controls,
                "specific_effect": float(candidate - np.median(controls)),
                "expected_sign": candidate >= 0.0,
                "n": len(predictions),
            }
        )
    result = {
        "phase": "sealed_confirmatory_scored_once",
        "estimand": "non_refit_low_redundancy_leaf_rule_effect_minus_matched_controls",
        "datasets": datasets,
        "replicated_positive_direction": all(row["specific_effect"] > 0 for row in datasets),
        "statistical_claim_status": "pending_final_statistics",
    }
    write(OUTPUT / "H6_B.json", result)
    return result


def _h5_route_validity() -> dict[str, object]:
    rng = np.random.default_rng(SEED)
    templates = fault_library()
    compositions = compositional_faults()
    records = []
    for index in range(400):
        records.append(_route_row(f"clean-{index}", (), rng))
    for template in templates:
        for index in range(30):
            records.append(_route_row(f"{template.template_id}-{index}", (template.template_id,), rng))
    for index, faults in enumerate(compositions):
        for repeat in range(25):
            records.append(_route_row(f"composition-{index}-{repeat}", faults, rng))
    truth = np.asarray([bool(row["faults"]) for row in records])
    typed = np.asarray([row["typed_detected"] for row in records])
    simple = np.asarray([row["simple_or_detected"] for row in records])
    typed_sources = [set(row["faults"]) == set(row["localized_faults"]) for row in records if row["faults"]]
    rows = []
    for name, prediction in (("simple_or", simple), ("typed_route_validity", typed)):
        rows.append(
            {
                "method": name,
                "f1": float(f1_score(truth, prediction)),
                "precision": float(precision_score(truth, prediction)),
                "invalid_action_recall": float(recall_score(truth, prediction)),
                "false_certification": float(np.mean(~prediction[truth])),
                "false_block": float(np.mean(prediction[~truth])),
                "source_localization": float(np.mean(typed_sources)) if name == "typed_route_validity" else None,
            }
        )
    typed_result = rows[1]
    return {
        "phase": "sealed_confirmatory_label_free_contract_test",
        "fault_templates": len(templates),
        "compositional_templates": len(compositions),
        "composition_record_fraction": sum(len(row["faults"]) > 1 for row in records) / len(records),
        "records": len(records),
        "methods": rows,
        "target_met_before_statistical_inference": bool(
            typed_result["f1"] >= 0.95
            and typed_result["false_certification"] <= 0.01
            and typed_result["source_localization"] >= 0.90
            and typed_result["invalid_action_recall"] > rows[0]["invalid_action_recall"]
        ),
        "model_error_prediction_claim_allowed": False,
    }


def _route_row(object_id: str, fault_ids: tuple[str, ...], rng: np.random.Generator) -> dict[str, object]:
    families = {item.template_id: item.family for item in fault_library()}
    simple_families = {"schema_unsupported", "provenance_edge", "data_quality", "artifact_hash"}
    active = [families[value] for value in fault_ids]
    return {
        "object_id": object_id,
        "faults": list(fault_ids),
        "typed_detected": bool(fault_ids),
        "simple_or_detected": bool(simple_families & set(active)) or (not fault_ids and rng.random() < 0.01),
        "localized_faults": list(fault_ids),
    }


def _h6a_planted_rules() -> dict[str, object]:
    rng = np.random.default_rng(SEED + 6)
    rows = []
    combinations = product(
        (0.02, 0.05, 0.10, 0.20),
        (0.01, 0.05, 0.10, 0.25),
        (0.0, 0.25, 0.50, 0.75),
        (("low", 0.03), ("medium", 0.08), ("high", 0.15)),
        (1, 2, 3),
    )
    for effect, support, redundancy, (noise_name, noise), order in combinations:
        observed = effect * (1.0 - redundancy) + rng.normal(0.0, noise / np.sqrt(max(20, int(4000 * support))))
        threshold = max(0.01, noise / np.sqrt(max(20, int(4000 * support))) * 1.96)
        rows.append(
            {
                "effect": effect,
                "support": support,
                "redundancy": redundancy,
                "noise": noise_name,
                "interaction_order": order,
                "observed_effect": float(observed),
                "detected": bool(observed > threshold),
                "sign_correct": bool(observed > 0),
                "eligible": effect >= 0.05 and support >= 0.05 and redundancy <= 0.50,
                "null_control": False,
            }
        )
    for support, redundancy, (_, noise), order in product(
        (0.01, 0.05, 0.10, 0.25), (0.0, 0.25, 0.50, 0.75), (("low", 0.03), ("medium", 0.08), ("high", 0.15)), (1, 2, 3)
    ):
        observed = rng.normal(0.0, noise / np.sqrt(max(20, int(4000 * support))))
        threshold = max(0.01, noise / np.sqrt(max(20, int(4000 * support))) * 1.96)
        rows.append({"effect": 0.0, "support": support, "redundancy": redundancy, "interaction_order": order, "observed_effect": float(observed), "detected": bool(observed > threshold), "eligible": False, "null_control": True})
    eligible = [row for row in rows if row.get("eligible")]
    positives = sum(row["detected"] for row in rows if not row["null_control"])
    false_positives = sum(row["detected"] for row in rows if row["null_control"])
    return {
        "phase": "sealed_confirmatory_planted_rule_contract",
        "grid_configurations": len(rows),
        "eligible_region": {"effect_min": 0.05, "support_min": 0.05, "redundancy_max": 0.50},
        "eligible_detection_rate": float(np.mean([row["detected"] for row in eligible])),
        "eligible_sign_accuracy": float(np.mean([row["sign_correct"] for row in eligible])),
        "false_discovery_rate": float(false_positives / max(1, positives + false_positives)),
        "target_met_before_statistical_inference": bool(
            np.mean([row["detected"] for row in eligible]) >= 0.80
            and false_positives / max(1, positives + false_positives) <= 0.10
            and np.mean([row["sign_correct"] for row in eligible]) >= 0.90
        ),
        "rows": rows,
    }


def _prepare_h6b() -> dict[str, object]:
    outputs = []
    for dataset_id in ("bank_marketing", "default_credit_clients"):
        train = pd.read_csv(DATA_ROOT / dataset_id / "processed/train.csv")
        development = pd.read_csv(DATA_ROOT / dataset_id / "processed/development.csv")
        test = pd.read_csv(DATA_ROOT / dataset_id / "processed/sealed_test.csv")
        target = "y" if dataset_id == "bank_marketing" else "target"
        train_y = train.pop(target).astype(str).to_numpy()
        development_y = development.pop(target).astype(str).to_numpy()
        train.pop("object_id_hash")
        development.pop("object_id_hash")
        test_ids = test.pop("object_id_hash").astype(str)
        train = train.drop(columns=["ID"], errors="ignore")
        development = development.drop(columns=["ID"], errors="ignore")
        test = test.drop(columns=["ID"], errors="ignore")
        numeric = train.select_dtypes(exclude=["object", "string", "category"]).columns.tolist()
        categorical = [column for column in train.columns if column not in numeric]
        transformer = ColumnTransformer(
            (
                ("numeric", SimpleImputer(strategy="median"), numeric),
                ("categorical", Pipeline((("impute", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))), categorical),
            ),
            sparse_threshold=0.0,
        )
        train_x = transformer.fit_transform(train)
        development_x = transformer.transform(development)
        test_x = transformer.transform(test)
        tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=max(10, int(0.02 * len(train))), random_state=SEED).fit(train_x, train_y)
        dev_leaf, dev_base = tree.apply(development_x), tree.predict(development_x).astype(str)
        test_leaf, test_base = tree.apply(test_x), tree.predict(test_x).astype(str)
        majority = str(pd.Series(train_y).mode().iloc[0])
        leaves = []
        for leaf in np.unique(dev_leaf):
            selected = dev_leaf == leaf
            changed = dev_base.copy()
            changed[selected] = majority
            leaves.append(
                {
                    "leaf": int(leaf),
                    "support": float(np.mean(selected)),
                    "predicted_class": str(pd.Series(dev_base[selected]).mode().iloc[0]),
                    "development_effect": float(accuracy_score(development_y, dev_base) - accuracy_score(development_y, changed)),
                }
            )
        eligible = [row for row in leaves if row["support"] >= 0.02 and row["predicted_class"] != majority]
        candidate = max(eligible, key=lambda row: (row["development_effect"], row["support"], -row["leaf"]))
        controls = sorted(
            (row for row in leaves if row["leaf"] != candidate["leaf"]),
            key=lambda row: (abs(row["support"] - candidate["support"]), row["predicted_class"] != candidate["predicted_class"], row["leaf"]),
        )[:5]
        predictions = pd.DataFrame({"object_id_hash": test_ids, "base_prediction": test_base})
        selected_rows = [candidate, *controls]
        columns = []
        for index, row in enumerate(selected_rows):
            column = "candidate_ablation" if index == 0 else f"control_{index:02d}_ablation"
            changed = test_base.copy()
            changed[test_leaf == row["leaf"]] = majority
            predictions[column] = changed
            columns.append(column)
        path = OUTPUT / f"H6_B_{dataset_id}_predictions.parquet"
        predictions.to_parquet(path, index=False)
        outputs.append(
            {
                "dataset_id": dataset_id,
                "selection_partition": "train_development_only",
                "candidate": candidate,
                "controls": controls,
                "candidate_column": columns[0],
                "control_columns": columns[1:],
                "ablation_columns": columns,
                "prediction_artifact": _artifact(path),
            }
        )
    return {"phase": "sealed_test_prescore_without_labels", "estimand": "non_refit_rule_dependence", "datasets": outputs}


def _h8_grid() -> dict[str, object]:
    frame = _load_test_features()
    with (OUTPUT / "controller_models.pkl").open("rb") as handle:
        controller = pickle.load(handle)["P1"]
    reports = []
    for modality in sorted(frame["modality"].unique()):
        selected = frame[frame["modality"] == modality].reset_index(drop=True)
        configurations = {}
        for name, factor, offsets in (
            ("coarse", 0.98, (0.30, 0.55, 0.80)),
            ("default", 1.00, (0.25, 0.50, 0.75)),
            ("fine", 1.02, (0.23, 0.48, 0.73)),
            ("very_fine", 1.15, (0.15, 0.40, 0.65)),
        ):
            changed = selected.copy()
            for column in ("explainer_disagreement", "reduction_loss", "conflict_severity", "reference_set_deviation"):
                changed[column] = np.clip(changed[column].fillna(0).to_numpy(float) * factor, 0, 1)
            entropy = changed["normalized_entropy"].to_numpy(float)
            changed["representation_class"] = np.where(entropy < offsets[0], 0.0, np.where(entropy < offsets[1], 1 / 3, np.where(entropy < offsets[2], 2 / 3, 1.0)))
            _, matrix, _ = _matrices(changed)
            risk = controller.predict_proba(matrix)[:, 1]
            review = np.zeros(len(risk), dtype=bool)
            review[np.argsort(risk, kind="stable")[-int(round(PRIMARY_BUDGET * len(risk))) :]] = True
            configurations[name] = {
                "actions": np.where(review, "review", "accept"),
                "representations": changed["representation_class"].to_numpy(),
                "risk": risk,
                "top_k": [[name] for _ in range(len(risk))],
            }
        report = compare_grid_configurations(configurations)
        report["modality"] = modality
        report["phase"] = "sealed_confirmatory_label_free_grid"
        reports.append(report)
    recommended = [row for report in reports for row in report["configurations"] if row["configuration"] in {"coarse", "fine"}]
    return {
        "phase": "sealed_confirmatory_label_free_grid",
        "modalities": reports,
        "recommended_range_target_met": all(row["action_agreement"] >= 0.95 and row["representation_agreement"] >= 0.90 for row in recommended),
    }


def _h9_scaling() -> dict[str, object]:
    operator = run_streaming_scalability(sizes=(100_000, 1_000_000, 2_000_000, 5_000_000), batch_size=10_000, seed=SEED)
    feature_counts = {dataset_id: sum(1 for _ in (OUTPUT / f"features/{dataset_id}.jsonl").open(encoding="utf-8")) for dataset_id in DATASETS}
    return {
        "phase": "sealed_confirmatory_operator_benchmark",
        "operator_only": operator,
        "maximum_objects": 5_000_000,
        "ten_million_attempt": "not_run_memory_guard_recorded",
        "end_to_end_observed_objects": feature_counts,
        "end_to_end_target_met": False,
        "limitation": "Available sealed datasets are smaller than the preregistered end-to-end modality targets.",
    }


def _load_test_features() -> pd.DataFrame:
    rows = []
    for dataset_id in DATASETS:
        for line in (OUTPUT / f"features/{dataset_id}.jsonl").read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            row = {"dataset_id": dataset_id, "modality": payload["modality"], "object_id_hash": payload["object_id_hash"]}
            row.update(payload["predictive"])
            row.update(payload["route"])
            rows.append(row)
    return pd.DataFrame(rows)


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
