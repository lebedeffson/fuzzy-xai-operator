#!/usr/bin/env python3
"""Merge modality jobs and fail closed on missing methods or model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_remediation"


def main(input_dir: Path) -> None:
    modalities = ("tabular", "image", "text", "timeseries")
    payloads = {}
    for modality in modalities:
        candidates = list(input_dir.rglob(f"{modality}.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one {modality}.json, got {candidates}")
        payloads[modality] = json.loads(candidates[0].read_text(encoding="utf-8"))
    datasets = [payloads[item]["dataset"] for item in modalities]
    models = [row for payload in payloads.values() for row in payload["models"]]
    explainers = [row for payload in payloads.values() for row in payload["explainers"]]
    measured_families = {row["family"] for row in models if row["status"] == "measured"}
    measured_methods = {row["method"] for row in explainers if row["status"] == "measured"}
    h6 = json.loads((EVIDENCE / "rule_ablation/h6_rule_ablation.json").read_text(encoding="utf-8"))
    checks = {
        "tabular_10k": payloads["tabular"]["dataset"]["n_objects"] >= 10_000,
        "image_10k": payloads["image"]["dataset"]["n_objects"] >= 10_000,
        "text_10k": payloads["text"]["dataset"]["n_objects"] >= 10_000,
        "timeseries_10k": payloads["timeseries"]["dataset"]["n_objects"] >= 10_000,
        "real_and_controlled": (EVIDENCE / "controlled_summary.json").is_file(),
        "licenses_and_hashes": all(row["license"] and row["raw_sha256"] and row["processed_sha256"] for row in datasets),
        "linear": "linear" in measured_families,
        "tree": "tree" in measured_families,
        "boosting": "boosting" in measured_families,
        "cnn": "CNN" in measured_families,
        "sequence": "sequence" in measured_families,
        "onnx": "ONNX" in measured_families,
        "shap": "SHAP" in measured_methods,
        "lime": "LIME" in measured_methods,
        "anchors": "Anchors" in measured_methods,
        "rulefit": "RuleFit" in measured_methods,
        "gradcam": "Grad-CAM" in measured_methods,
        "integrated_gradients": "Integrated Gradients" in measured_methods,
        "text_timeseries_attribution": {"token_masking", "window_masking"}.issubset(measured_methods),
        "rule_conditional_model": h6["conditional_model"]["status"].startswith("measured"),
    }
    failed = [key for key, passed in checks.items() if not passed]
    payload = {
        "schema_version": "1.0",
        "datasets": datasets,
        "models": models,
        "explainers": explainers,
        "checks": checks,
        "status": "PASS" if not failed else "FAIL",
        "failures": failed,
        "limitations": [
            "benchmark performance does not establish deployment validity",
            "attributions are associational rather than causal",
            "external human and domain gates remain open",
        ],
    }
    output = EVIDENCE / "real_benchmarks/combined_status.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"real benchmark merge failures: {failed}")
    print(f"PASS: q1_real_benchmark_merge datasets={len(datasets)} models={len(models)} explainers={len(explainers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.input_dir)
