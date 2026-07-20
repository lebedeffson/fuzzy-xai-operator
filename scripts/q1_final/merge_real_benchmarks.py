#!/usr/bin/env python3
"""Merge final native jobs and fail closed on missing models or explainers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_evidence/q1_final/real_benchmarks/combined_status.json"
MODALITIES = ("tabular", "image", "text", "timeseries")
EXPECTED_CLASSES = {"tabular": 7, "image": 10, "text": 20, "timeseries": 7}
MIN_EXPLANATIONS = {"tabular": 1000, "image": 500, "text": 500, "timeseries": 500}
REQUIRED_MODELS = {
    "tabular": {"linear", "tree", "boosting", "neural_mlp"},
    "image": {"CNN", "ONNX"},
    "text": {"linear", "linear_svc", "sequence"},
    "timeseries": {"tree", "boosting", "CNN", "sequence", "TCN"},
}
REQUIRED_EXPLAINERS = {
    "tabular": {"SHAP", "LIME", "Anchors", "RuleFit"},
    "image": {"Grad-CAM", "Integrated Gradients"},
    "text": {"token masking"},
    "timeseries": {"window masking"},
}


def _one(input_dir: Path, pattern: str) -> dict[str, object] | None:
    candidates = list(input_dir.rglob(pattern))
    if len(candidates) > 1:
        raise RuntimeError(f"expected at most one {pattern}, got {candidates}")
    return json.loads(candidates[0].read_text(encoding="utf-8")) if candidates else None


def merge(input_dir: Path, *, allow_incomplete: bool) -> dict[str, object]:
    rows = {}
    checks: dict[str, bool] = {}
    failures = []
    for modality in MODALITIES:
        classical = _one(input_dir, f"{modality}.json")
        neural = _one(input_dir, f"{modality}_neural.json") if modality != "tabular" else None
        explainers = _one(input_dir, f"{modality}_explainers.json")
        if classical is None:
            failures.append(f"missing_{modality}_classical")
            continue
        dataset = classical["dataset"]
        model_rows = list(classical["models"])
        if neural:
            model_rows.extend(neural["models"])
        families = {str(row["family"]) for row in model_rows if row.get("status") == "measured"}
        if neural and neural.get("onnx", {}).get("status") == "verified":
            families.add("ONNX")
        methods = {
            str(row["method"])
            for row in (explainers or {}).get("methods", [])
            if row.get("status") == "measured"
        }
        evaluation_ids = list(classical.get("evaluation_object_ids", []))
        explainer_ids = list((explainers or {}).get("evaluation_object_ids", []))
        modality_checks = {
            "objects_10k": int(dataset["n_objects"]) >= 10_000,
            "native_classes": int(dataset["native_class_count"]) == EXPECTED_CLASSES[modality],
            "five_seeds": len(set(classical["seeds"])) >= 5,
            "required_models": REQUIRED_MODELS[modality].issubset(families),
            "explanation_sample": len(evaluation_ids) >= MIN_EXPLANATIONS[modality],
            "same_explanation_ids": bool(explainers) and evaluation_ids == explainer_ids,
            "required_explainers": REQUIRED_EXPLAINERS[modality].issubset(methods),
            "licenses_and_hashes": bool(dataset.get("license") and dataset.get("raw_sha256")),
        }
        for name, passed in modality_checks.items():
            checks[f"{modality}_{name}"] = passed
        rows[modality] = {
            "dataset": dataset,
            "models": model_rows,
            "model_families": sorted(families),
            "explainers": (explainers or {}).get("methods", []),
            "evaluation_object_count": len(evaluation_ids),
            "checks": modality_checks,
        }
    failures.extend(name for name, passed in checks.items() if not passed)
    status = "PASS" if not failures and len(rows) == 4 else "INCOMPLETE" if allow_incomplete else "FAIL"
    payload = {
        "schema_version": "2.0",
        "status": status,
        "modalities": rows,
        "checks": checks,
        "failures": sorted(set(failures)),
        "stable_claims_allowed": status == "PASS",
        "limitations": [
            "benchmark performance does not establish deployment validity",
            "external human and domain gates are evaluated separately",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if status == "FAIL":
        raise RuntimeError(f"final real benchmark merge failed: {payload['failures']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    payload = merge(args.input_dir, allow_incomplete=args.allow_incomplete)
    print(f"PASS: q1_final_real_merge status={payload['status']} modalities={len(payload['modalities'])}")


if __name__ == "__main__":
    main()
