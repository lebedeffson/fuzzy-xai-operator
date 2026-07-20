#!/usr/bin/env python3
"""Freeze 360 anonymized source cases from measured Q1 modality artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "release_evidence/q1_final/real_jobs"
OUTPUT = ROOT / "study/ai_pre_review/source_case_evidence.jsonl"
MODALITIES = ("tabular", "image", "text", "timeseries")
SOURCE_COMMIT = "e34e52fb8ae62ee1be043d6d5b26a0c9214a0572"


def main() -> None:
    records: list[dict[str, object]] = []
    case_index = 1
    for modality in MODALITIES:
        payload = load(INPUT / f"{modality}.json")
        explainers = load(INPUT / f"{modality}_explainers.json")
        selected = select_rows(payload, explainers, count=90)
        for local_index, row in enumerate(selected):
            split = "formative" if local_index < 60 else "confirmatory"
            condition = controlled_condition(local_index)
            strata = base_strata(row)
            if condition:
                strata.append(condition)
            strata.append("missing_provenance" if condition == "missing_provenance" else "complete_provenance")
            strata.append("structural_rupture" if condition == "controlled_structural_rupture" else "no_structural_rupture")
            object_hash = digest(f"{modality}:{row['object_id']}")
            records.append(
                {
                    "schema_version": "1.0",
                    "case_id": f"case_{case_index:06d}",
                    "object_id_hash": object_hash,
                    "dataset_id": payload["dataset"]["dataset_id"],
                    "dataset_sha256": payload["dataset"]["raw_sha256"],
                    "modality": modality,
                    "task": "classification",
                    "model_family": row["family"],
                    "model_version_hash": digest(f"{SOURCE_COMMIT}:{modality}:{row['model_id']}:{row['seed']}"),
                    "split": split,
                    "stratum": strata,
                    "prediction": {
                        "display_label": f"class_{row['predicted_class']}",
                        "score": row["confidence"],
                        "is_correct": row["correct"],
                        "true_label": f"class_{row['true_class']}",
                    },
                    "rare_class": row["rare_class"],
                    "low_confidence": row["low_confidence"],
                    "cross_model_conflict": row["cross_model_conflict"],
                    "explainer_disagreement": row["explainer_disagreement"],
                    "explainer_evidence": row["explainer_evidence"],
                    "controlled_condition": condition,
                    "controlled_condition_disclosure": (
                        "controlled diagnostic condition derived from the frozen route/provenance protocol" if condition else None
                    ),
                    "source_commit": SOURCE_COMMIT,
                    "source_artifact": f"release_evidence/q1_final/real_jobs/{modality}.json",
                }
            )
            case_index += 1
    validate(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    print(f"PASS: ai_pre_review_source_cases cases={len(records)} sha256={sha256(OUTPUT)}")


def select_rows(payload: dict[str, object], explainers: dict[str, object], *, count: int) -> list[dict[str, object]]:
    first_seed = int(payload["seeds"][0])
    first_model = str(payload["models"][0]["model_id"])
    evaluation_ids = {str(value) for value in payload["evaluation_object_ids"]}
    all_rows = list(payload["object_predictions"])
    conflict: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in all_rows:
        conflict[(int(row["seed"]), str(row["object_id"]))].add(int(row["predicted_class"]))
    explainer_by_object: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in explainers.get("pairs", []):
        explainer_by_object[str(row["object_id"])].append(row)
    candidates = []
    seen = set()
    for raw in all_rows:
        object_id = str(raw["object_id"])
        if int(raw["seed"]) != first_seed or str(raw["model_id"]) != first_model or object_id not in evaluation_ids or object_id in seen:
            continue
        seen.add(object_id)
        evidence = explainer_by_object.get(object_id, [])
        fidelities = [float(item["base_fidelity"]) for item in evidence]
        disagreement = standard_deviation(fidelities)
        row = dict(raw)
        row["cross_model_conflict"] = len(conflict[(first_seed, object_id)]) > 1
        row["explainer_disagreement"] = disagreement
        row["explainer_evidence"] = [
            {
                "source": f"local_source_{index + 1}",
                "fidelity": float(item["base_fidelity"]),
                "rank_agreement": float(item["rank_agreement"]),
                "attribution_sha256": item["base_attribution_sha256"],
            }
            for index, item in enumerate(evidence)
        ]
        candidates.append(row)
    candidates.sort(key=lambda row: digest(str(row["object_id"])))
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        level = "high" if float(row["confidence"]) >= 0.70 else "low"
        buckets[f"{'correct' if row['correct'] else 'wrong'}_{level}"].append(row)
    selected: list[dict[str, object]] = []
    while len(selected) < count and any(buckets.values()):
        for name in ("correct_high", "correct_low", "wrong_high", "wrong_low"):
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].pop(0))
    if len(selected) != count:
        raise RuntimeError(f"could not select {count} stratified cases")
    return selected


def base_strata(row: dict[str, object]) -> list[str]:
    strata = ["correct_prediction" if row["correct"] else "wrong_prediction"]
    strata.append("high_confidence" if float(row["confidence"]) >= 0.70 else "low_confidence")
    strata.append("rare_class" if row["rare_class"] else "common_class")
    if row["cross_model_conflict"]:
        strata.append("model_conflict")
    if float(row["explainer_disagreement"]) >= 0.10:
        strata.append("unstable_explanation")
        strata.append("explainer_conflict")
    return strata


def controlled_condition(index: int) -> str | None:
    return (
        None,
        "missing_provenance",
        "controlled_distribution_shift",
        "controlled_structural_rupture",
        "controlled_attribution_instability",
        None,
    )[index % 6]


def validate(records: list[dict[str, object]]) -> None:
    if len(records) != 360 or len({row["object_id_hash"] for row in records}) != 360:
        raise RuntimeError("source snapshot must contain 360 unique objects")
    for modality in MODALITIES:
        rows = [row for row in records if row["modality"] == modality]
        if len([row for row in rows if row["split"] == "formative"]) != 60:
            raise RuntimeError(f"{modality} formative count mismatch")
        if len([row for row in rows if row["split"] == "confirmatory"]) != 30:
            raise RuntimeError(f"{modality} confirmatory count mismatch")
        required = {
            "correct_prediction",
            "wrong_prediction",
            "high_confidence",
            "low_confidence",
            "complete_provenance",
            "missing_provenance",
            "structural_rupture",
            "no_structural_rupture",
        }
        observed = {item for row in rows for item in row["stratum"]}
        if not required.issubset(observed):
            raise RuntimeError(f"{modality} lacks required prediction strata")


def load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise RuntimeError(f"missing frozen Q1 artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def standard_deviation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
