from __future__ import annotations

import argparse
import math
from numbers import Number
from pathlib import Path

import pandas as pd

from .common import ARTIFACTS, git_commit, protocol, read_json, sha256_file, write_json


def _write_table(name: str, frame: pd.DataFrame) -> Path:
    path = ARTIFACTS / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build() -> dict[str, object]:
    cfg = protocol()
    dataset_manifest = read_json(ARTIFACTS / "manifests" / "dataset_manifest.json")
    explanations = read_json(ARTIFACTS / "explanations" / "sealed_test_summary.json")
    quality = read_json(ARTIFACTS / "policies" / "test_quality.json")
    policy = pd.read_csv(ARTIFACTS / "policies" / "policy_results.csv")
    route = pd.read_csv(ARTIFACTS / "route_faults" / "summary.csv")
    runtime = pd.read_csv(ARTIFACTS / "runtime" / "summary.csv")
    case_timing = read_json(ARTIFACTS / "end_to_end_case" / "stage_timings.json")
    statistics = read_json(ARTIFACTS / "policies" / "statistical_tests.json")
    primary = next(row for row in statistics["comparisons"] if row["primary"])

    tables: dict[str, dict[str, object]] = {}
    split_files = dataset_manifest["processed_files"]
    train_rows = int(split_files["train"]["rows"])
    validation_rows = int(split_files["validation"]["rows"])
    test_rows = int(split_files["sealed_test"]["rows"])
    modern = pd.DataFrame(
        [
            {
                "dataset": "AG News",
                "objects_total": train_rows + validation_rows + test_rows,
                "train": train_rows,
                "validation": validation_rows,
                "sealed_test": test_rows,
                "test_accuracy": quality["accuracy"],
                "local_explanations": explanations["objects"],
                "canonical_hash_preservation": explanations["canonical_hash_preservation_rate"],
            }
        ]
    )
    tables["modern_contour"] = {"path": _write_table("modern_contour", modern), "sources": [ARTIFACTS / "manifests" / "dataset_manifest.json", ARTIFACTS / "manifests" / "model_manifest.json", ARTIFACTS / "explanations" / "sealed_test_summary.json", ARTIFACTS / "policies" / "test_quality.json"], "status": "confirmatory"}

    policies = policy[(policy["review_budget"] == 0.20) & (policy["cost_profile"] == "balanced")].copy()
    policies = policies[["policy", "automatic_coverage", "wrong_automatic_actions", "selective_risk", "manual_review_load", "false_blocks", "total_cost", "risk_auroc", "risk_auprc", "expected_calibration_error", "brier_score"]]
    tables["policies_budget_20"] = {"path": _write_table("policies_budget_20", policies), "sources": [ARTIFACTS / "policies" / "policy_results.csv", ARTIFACTS / "policies" / "statistical_tests.json"], "status": "confirmatory"}

    selected_baseline = str(quality["best_simple_selected_on_validation"])
    budget_rows = []
    for comparison in statistics["comparisons"]:
        if comparison["baseline"] != selected_baseline:
            continue
        budget = float(comparison["review_budget"])
        baseline_row = policy[(policy["policy"] == selected_baseline) & (policy["review_budget"] == budget) & (policy["cost_profile"] == "balanced")].iloc[0]
        fuzzy_row = policy[(policy["policy"] == "full_fuzzyxai") & (policy["review_budget"] == budget) & (policy["cost_profile"] == "balanced")].iloc[0]
        budget_rows.append(
            {
                "review_budget": budget,
                "selected_baseline": selected_baseline,
                "automatic_coverage": float(fuzzy_row["automatic_coverage"]),
                "baseline_wrong_actions": int(comparison["baseline_wrong"]),
                "fuzzyxai_wrong_actions": int(comparison["full_wrong"]),
                "baseline_error_rate": float(baseline_row["accepted_error_rate"]),
                "fuzzyxai_error_rate": float(fuzzy_row["accepted_error_rate"]),
                "baseline_minus_fuzzyxai": float(comparison["absolute_rate_reduction"]),
                "ci_lower": float(comparison["ci_lower"]),
                "ci_upper": float(comparison["ci_upper"]),
                "holm_adjusted_p": float(comparison["holm_adjusted_p"]),
            }
        )
    all_budgets = pd.DataFrame(budget_rows).sort_values("review_budget")
    tables["policies_all_budgets"] = {
        "path": _write_table("policies_all_budgets", all_budgets),
        "sources": [ARTIFACTS / "policies" / "policy_results.csv", ARTIFACTS / "policies" / "statistical_tests.json", ARTIFACTS / "policies" / "validation_selection.json"],
        "status": "confirmatory_primary_20_secondary_other_budgets",
    }

    route_table = route[["group", "method", "n", "precision", "recall", "f1", "false_certification", "false_rejection", "fault_type_accuracy", "component_localization_accuracy", "diagnostic_time_ms_mean"]]
    tables["route_validator"] = {"path": _write_table("route_validator", route_table), "sources": [ARTIFACTS / "route_faults" / "raw_results.jsonl", ARTIFACTS / "route_faults" / "summary.csv"], "status": "confirmatory_and_exploratory_held_out"}
    held_out = route[route["group"] == "held_out_fault_types"].copy()
    held_out["scope"] = "predetermined held-out registered types; exploratory; not arbitrary open-set faults"
    tables["route_held_out_status"] = {
        "path": _write_table("route_held_out_status", held_out),
        "sources": [ARTIFACTS / "route_faults" / "raw_results.jsonl", ARTIFACTS / "route_faults" / "summary.csv", ARTIFACTS / "route_faults" / "manifest.json"],
        "status": "exploratory_held_out_registered_not_open_set",
    }

    runtime_columns = [
        "modality",
        "model",
        "explainer",
        "n",
        "repetitions",
        "model_seconds_median",
        "explainer_seconds_median",
        "fuzzyxai_seconds_median",
        "serialization_seconds_median",
        "total_seconds_median",
        "objects_per_second_median",
        "peak_rss_bytes_median",
        "peak_vram_bytes_median",
        "fuzzyxai_time_fraction",
        "explainer_time_fraction",
    ]
    tables["end_to_end_runtime"] = {"path": _write_table("end_to_end_runtime", runtime[runtime_columns]), "sources": [ARTIFACTS / "runtime" / "raw_results.csv", ARTIFACTS / "runtime" / "summary.csv", ARTIFACTS / "runtime" / "manifest.json", ARTIFACTS / "runtime" / "environment_snapshots" / "shared_gpu_during_benchmark.txt"], "status": "descriptive"}
    tables["runtime_full_statistics"] = {
        "path": _write_table("runtime_full_statistics", runtime),
        "sources": [ARTIFACTS / "runtime" / "raw_results.csv", ARTIFACTS / "runtime" / "summary.csv", ARTIFACTS / "runtime" / "manifest.json", ARTIFACTS / "runtime" / "environment_snapshots" / "shared_gpu_during_benchmark.txt"],
        "status": "descriptive_shared_gpu_environment",
    }

    case = pd.DataFrame([{"object_id": read_json(ARTIFACTS / "end_to_end_case" / "input_reference.json")["object_id"], **case_timing}])
    tables["end_to_end_case"] = {"path": _write_table("end_to_end_case", case), "sources": [ARTIFACTS / "end_to_end_case" / "stage_timings.json", ARTIFACTS / "end_to_end_case" / "action.json"], "status": "descriptive"}

    h1_supported = explanations["canonical_hash_preservation_rate"] == 1.0
    h2_positive = primary["absolute_rate_reduction"] > 0 and primary["ci_lower"] > 0 and primary["holm_adjusted_p"] < 0.05
    typed_registered = route[
        (route["method"] == "typed_route_validator") & route["group"].isin(["registered_single", "registered_compositional"])
    ]
    h3_supported = bool((typed_registered["f1"] == 1.0).all() and (typed_registered["component_localization_accuracy"] == 1.0).all())
    hypotheses = pd.DataFrame(
        [
            {"hypothesis": "H3-original", "result": "not_supported", "scope": "frozen v1.3.0"},
            {"hypothesis": "H5-P-original", "result": "not_supported", "scope": "frozen v1.3.0"},
            {"hypothesis": "H6-general", "result": "not_supported", "scope": "frozen v1.3.0"},
            {"hypothesis": "V13-H1", "result": "supported" if h1_supported else "not_supported", "scope": "2000 AG News explanations"},
            {"hypothesis": "V13-H2", "result": "positive_effect" if h2_positive else "no_confirmatory_advantage", "scope": "matched coverage sealed test"},
            {"hypothesis": "V13-H3", "result": "supported_registered_library" if h3_supported else "not_supported", "scope": "registered single and compositional route faults"},
            {"hypothesis": "V13-H4", "result": "descriptive", "scope": "end-to-end N=1..10000"},
            {"hypothesis": "V13-H5", "result": "exploratory", "scope": "held-out fault types"},
        ]
    )
    tables["hypothesis_status"] = {"path": _write_table("hypothesis_status", hypotheses), "sources": [ARTIFACTS / "policies" / "statistical_tests.json", ARTIFACTS / "explanations" / "sealed_test_summary.json", ARTIFACTS / "route_faults" / "summary.csv"], "status": "mixed"}

    evidence_entries = []
    table_manifest = {}
    counter = 1
    for table_name, metadata in tables.items():
        table_path = metadata["path"]
        frame = pd.read_csv(table_path)
        sources = [Path(path) for path in metadata["sources"]]
        table_manifest[table_name] = {
            "path": str(table_path.relative_to(ARTIFACTS)),
            "sha256": sha256_file(table_path),
            "rows": len(frame),
            "sources": [{"path": str(path.relative_to(ARTIFACTS)), "sha256": sha256_file(path)} for path in sources],
            "status": metadata["status"],
        }
        for row_index, row in frame.iterrows():
            for column, value in row.items():
                if isinstance(value, Number) and not (isinstance(value, float) and math.isnan(value)):
                    evidence_entries.append(
                        {
                            "evidence_id": f"V13-E{counter:05d}",
                            "table": table_name,
                            "row": int(row_index),
                            "column": column,
                            "value": float(value),
                            "table_path": str(table_path.relative_to(ARTIFACTS)),
                            "table_sha256": sha256_file(table_path),
                            "source_artifacts": table_manifest[table_name]["sources"],
                            "builder": "experiments/chapter4_v13/build_tables.py",
                            "config": "config/chapter4_v13_protocol.yaml",
                            "commit": git_commit(),
                            "status": metadata["status"],
                        }
                    )
                    counter += 1
    write_json(ARTIFACTS / "manifests" / "tables_manifest.json", table_manifest)
    evidence_map = {
        "schema_version": "chapter4-v13-evidence-map-1.0",
        "protocol": cfg["protocol"],
        "commit": git_commit(),
        "entries": evidence_entries,
        "table_manifest_sha256": sha256_file(ARTIFACTS / "manifests" / "tables_manifest.json"),
    }
    write_json(ARTIFACTS / "evidence_map.json", evidence_map)
    return {"tables": len(tables), "numeric_entries": len(evidence_entries), "primary_comparison": primary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = build()
    print(f"PASS: tables={result['tables']} evidence_entries={result['numeric_entries']}")


if __name__ == "__main__":
    main()
