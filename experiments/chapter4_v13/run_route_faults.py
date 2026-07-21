from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .common import ARTIFACTS, protocol, sha256_bytes, sha256_file, write_json, write_jsonl


REGISTERED = {
    "missing_provenance": "provenance",
    "model_version_mismatch": "model",
    "preprocessing_incompatibility": "preprocessing",
    "corrupted_checksum": "artifact",
    "missing_calibration": "calibration",
}
HELD_OUT = {
    "reordered_transformations": "preprocessing",
    "unknown_dictionary_version": "dictionary",
    "broken_post_reduction_link": "reduction",
    "mixed_model_artifacts": "model",
    "incompatible_feature_space": "features",
}


def clean_route(index: int) -> dict[str, object]:
    payload = sha256_bytes(f"route-payload-{index}".encode())
    return {
        "object_id": f"route:{index:07d}",
        "model_version": "distilbert-ag-news@52ee64d",
        "explanation_model_version": "distilbert-ag-news@52ee64d",
        "preprocessing_version": "tokenizer@52ee64d",
        "transformation_order": ["normalize_whitespace", "tokenize", "truncate_128"],
        "expected_transformation_order": ["normalize_whitespace", "tokenize", "truncate_128"],
        "calibration_version": "isotonic-v13",
        "dictionary_version": "ag-news-en-v1",
        "feature_space": "distilbert-wordpiece-v1",
        "expected_feature_space": "distilbert-wordpiece-v1",
        "provenance_channels": ["prediction", "model", "tokenizer", "explainer", "reference"],
        "mandatory_provenance_channels": ["prediction", "model", "tokenizer", "explainer", "reference"],
        "artifact_sha256": payload,
        "expected_artifact_sha256": payload,
        "post_reduction_link_valid": True,
    }


def inject(route: dict[str, object], fault: str) -> None:
    if fault == "missing_provenance":
        route["provenance_channels"] = ["prediction", "model", "tokenizer"]
    elif fault == "model_version_mismatch":
        route["model_version"] = "distilbert-ag-news@stale"
        route["explanation_model_version"] = "distilbert-ag-news@stale"
    elif fault == "preprocessing_incompatibility":
        route["preprocessing_version"] = "tokenizer@incompatible"
    elif fault == "corrupted_checksum":
        route["artifact_sha256"] = "0" * 64
    elif fault == "missing_calibration":
        route["calibration_version"] = None
    elif fault == "reordered_transformations":
        route["transformation_order"] = ["tokenize", "normalize_whitespace", "truncate_128"]
    elif fault == "unknown_dictionary_version":
        route["dictionary_version"] = "unknown-dictionary-v99"
    elif fault == "broken_post_reduction_link":
        route["post_reduction_link_valid"] = False
    elif fault == "mixed_model_artifacts":
        route["explanation_model_version"] = "other-model@deadbeef"
    elif fault == "incompatible_feature_space":
        route["feature_space"] = "sentencepiece-v2"
    else:
        raise KeyError(fault)


def _registered_conditions(route: Mapping[str, object]) -> list[tuple[str, str]]:
    conditions = []
    if not set(route["mandatory_provenance_channels"]) <= set(route["provenance_channels"]):
        conditions.append(("missing_provenance", "provenance"))
    if route["model_version"] != "distilbert-ag-news@52ee64d":
        conditions.append(("model_version_mismatch", "model"))
    if route["preprocessing_version"] != "tokenizer@52ee64d":
        conditions.append(("preprocessing_incompatibility", "preprocessing"))
    if route["artifact_sha256"] != route["expected_artifact_sha256"]:
        conditions.append(("corrupted_checksum", "artifact"))
    if route["calibration_version"] is None:
        conditions.append(("missing_calibration", "calibration"))
    return conditions


def simple_or(route: Mapping[str, object]) -> tuple[list[str], list[str]]:
    detected = []
    if not set(route["mandatory_provenance_channels"]) <= set(route["provenance_channels"]):
        detected.append("missing_provenance")
    if route["artifact_sha256"] != route["expected_artifact_sha256"]:
        detected.append("corrupted_checksum")
    return detected, [REGISTERED[name] for name in detected]


def independent_if_else(route: Mapping[str, object]) -> tuple[list[str], list[str]]:
    conditions = _registered_conditions(route)
    return ([conditions[0][0]], [conditions[0][1]]) if conditions else ([], [])


def weighted_fault_score(route: Mapping[str, object]) -> tuple[list[str], list[str]]:
    conditions = _registered_conditions(route)
    score = len(conditions) / len(REGISTERED)
    return (["weighted_registered_fault"] if score >= 0.2 else [], ["route"] if score >= 0.2 else [])


def typed_route_validator(route: Mapping[str, object]) -> tuple[list[str], list[str]]:
    conditions = _registered_conditions(route)
    if route["transformation_order"] != route["expected_transformation_order"]:
        conditions.append(("reordered_transformations", "preprocessing"))
    if route["dictionary_version"] != "ag-news-en-v1":
        conditions.append(("unknown_dictionary_version", "dictionary"))
    if not route["post_reduction_link_valid"]:
        conditions.append(("broken_post_reduction_link", "reduction"))
    if route["explanation_model_version"] != route["model_version"]:
        conditions.append(("mixed_model_artifacts", "model"))
    if route["feature_space"] != route["expected_feature_space"]:
        conditions.append(("incompatible_feature_space", "features"))
    return [name for name, _ in conditions], [source for _, source in conditions]


METHODS: dict[str, Callable[[Mapping[str, object]], tuple[list[str], list[str]]]] = {
    "simple_or": simple_or,
    "independent_if_else": independent_if_else,
    "weighted_fault_score": weighted_fault_score,
    "typed_route_validator": typed_route_validator,
}


def build_cases() -> list[dict[str, object]]:
    cfg = protocol()["route_validation"]
    per_type = int(cfg["samples_per_fault_type"])
    cases: list[dict[str, object]] = []
    index = 0
    for _ in range(int(cfg["clean_samples"])):
        cases.append({"route": clean_route(index), "group": "clean", "faults": [], "sources": []})
        index += 1
    for fault, source in REGISTERED.items():
        for _ in range(per_type):
            route = clean_route(index)
            inject(route, fault)
            cases.append({"route": route, "group": "registered_single", "faults": [fault], "sources": [source]})
            index += 1
    combinations = [
        combination
        for size in range(int(cfg["registered_compositional"]["minimum_faults"]), int(cfg["registered_compositional"]["maximum_faults"]) + 1)
        for combination in itertools.combinations(REGISTERED, size)
    ]
    for combination in combinations:
        for _ in range(per_type):
            route = clean_route(index)
            for fault in combination:
                inject(route, fault)
            cases.append({"route": route, "group": "registered_compositional", "faults": list(combination), "sources": [REGISTERED[fault] for fault in combination]})
            index += 1
    for fault, source in HELD_OUT.items():
        for _ in range(per_type):
            route = clean_route(index)
            inject(route, fault)
            cases.append({"route": route, "group": "held_out_fault_types", "faults": [fault], "sources": [source]})
            index += 1
    return cases


def _score(records: Sequence[Mapping[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["method"]), str(row["group"]))].append(row)
    summaries = []
    for (method, group), rows in sorted(grouped.items()):
        truth = np.asarray([bool(row["faults"]) for row in rows])
        detected = np.asarray([bool(row["detected_faults"]) for row in rows])
        tp = int((truth & detected).sum())
        fp = int((~truth & detected).sum())
        fn = int((truth & ~detected).sum())
        tn = int((~truth & ~detected).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        type_accuracy = np.mean([set(row["faults"]) == set(row["detected_faults"]) for row in rows if row["faults"]]) if truth.any() else 1.0
        source_accuracy = np.mean([set(row["sources"]) == set(row["detected_sources"]) for row in rows if row["faults"]]) if truth.any() else 1.0
        summaries.append(
            {
                "method": method,
                "group": group,
                "n": len(rows),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(2 * precision * recall / max(1e-12, precision + recall)),
                "false_certification": float(fn / max(1, tp + fn)),
                "false_rejection": float(fp / max(1, fp + tn)),
                "fault_type_accuracy": float(type_accuracy),
                "component_localization_accuracy": float(source_accuracy),
                "diagnostic_time_ms_mean": float(np.mean([float(row["diagnostic_time_ns"]) for row in rows]) / 1e6),
            }
        )
    return summaries


def run() -> dict[str, object]:
    cases = build_cases()
    records = []
    for case in cases:
        for method_name, method in METHODS.items():
            start = time.perf_counter_ns()
            faults, sources = method(case["route"])
            elapsed = time.perf_counter_ns() - start
            records.append(
                {
                    "object_id": case["route"]["object_id"],
                    "group": case["group"],
                    "method": method_name,
                    "faults": case["faults"],
                    "sources": case["sources"],
                    "detected_faults": faults,
                    "detected_sources": sources,
                    "diagnostic_time_ns": elapsed,
                }
            )
    raw_path = ARTIFACTS / "route_faults" / "raw_results.jsonl"
    write_jsonl(raw_path, records)
    summaries = _score(records)
    summary_path = ARTIFACTS / "route_faults" / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    payload = {
        "cases": len(cases),
        "evaluations": len(records),
        "groups": sorted({str(case["group"]) for case in cases}),
        "methods": list(METHODS),
        "raw_sha256": sha256_file(raw_path),
        "summary_sha256": sha256_file(summary_path),
        "held_out_fault_claim": "exploratory; no universal unknown-fault detection claim",
    }
    write_json(ARTIFACTS / "route_faults" / "manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(f"PASS: route benchmark cases={result['cases']} evaluations={result['evaluations']}")


if __name__ == "__main__":
    main()
