from __future__ import annotations

import argparse

import pandas as pd

from .common import ARTIFACTS, git_commit, protocol, read_json, read_jsonl, sha256_file, verify_protocol_hash, write_json


REQUIRED = (
    "manifests/dataset_manifest.json",
    "manifests/model_manifest.json",
    "explanations/sealed_test.jsonl",
    "explanations/sealed_test_summary.json",
    "policies/pre_score_lock.json",
    "policies/policy_results.csv",
    "policies/statistical_tests.json",
    "route_faults/raw_results.jsonl",
    "route_faults/summary.csv",
    "runtime/raw_results.csv",
    "runtime/summary.csv",
    "end_to_end_case/SHA256SUMS",
    "manifests/tables_manifest.json",
    "manifests/figures_manifest.json",
    "evidence_map.json",
    "leakage_audit.json",
)


def validate() -> dict[str, object]:
    errors: list[str] = []
    protocol_hash = verify_protocol_hash()
    for relative in REQUIRED:
        if not (ARTIFACTS / relative).exists():
            errors.append(f"missing:{relative}")
    if errors:
        return {"passed": False, "errors": errors}

    leakage = read_json(ARTIFACTS / "leakage_audit.json")
    if not leakage.get("passed"):
        errors.append("leakage_audit_failed")
    score_rows = list(read_jsonl(ARTIFACTS / "policies" / "test_policy_scores.jsonl"))
    forbidden = {"label", "true_label", "is_correct", "ground_truth", "expected_action"}
    for row in score_rows:
        if forbidden & set(row):
            errors.append("test_outcome_in_policy_feature_rows")
            break
    lock = read_json(ARTIFACTS / "policies" / "pre_score_lock.json")
    if lock["policy_scores_sha256"] != sha256_file(ARTIFACTS / "policies" / "test_policy_scores.jsonl"):
        errors.append("pre_score_hash_mismatch")
    if lock.get("test_labels_loaded") is not False:
        errors.append("pre_score_opened_labels")

    evidence = read_json(ARTIFACTS / "evidence_map.json")
    tables = read_json(ARTIFACTS / "manifests" / "tables_manifest.json")
    mapped = {(entry["table"], int(entry["row"]), entry["column"]) for entry in evidence["entries"]}
    for name, metadata in tables.items():
        path = ARTIFACTS / metadata["path"]
        if sha256_file(path) != metadata["sha256"]:
            errors.append(f"table_hash_mismatch:{name}")
        frame = pd.read_csv(path)
        for row_index, row in frame.iterrows():
            for column, value in row.items():
                if pd.api.types.is_number(value) and pd.notna(value) and (name, int(row_index), column) not in mapped:
                    errors.append(f"unmapped_numeric_cell:{name}:{row_index}:{column}")

    case_dir = ARTIFACTS / "end_to_end_case"
    for line in (case_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        if sha256_file(case_dir / filename) != expected:
            errors.append(f"case_hash_mismatch:{filename}")

    results = {
        "passed": not errors,
        "errors": errors,
        "protocol_sha256": protocol_hash,
        "commit": git_commit(),
        "required_artifacts": len(REQUIRED),
        "evidence_entries": len(evidence["entries"]),
        "test_policy_rows": len(score_rows),
        "test_outcomes_absent_from_policy_features": not any(error == "test_outcome_in_policy_feature_rows" for error in errors),
        "stable_negative_claims": protocol()["scope"]["frozen_negative_claims"],
    }
    write_json(ARTIFACTS / "manifests" / "validation.json", results)
    report = ARTIFACTS / "validation_report.md"
    report.write_text(
        "# Chapter 4 v13 validation report\n\n"
        f"- status: `{'PASS' if results['passed'] else 'FAIL'}`\n"
        f"- protocol SHA256: `{protocol_hash}`\n"
        f"- commit: `{results['commit']}`\n"
        f"- required artifacts: `{results['required_artifacts']}`\n"
        f"- evidence-map entries: `{results['evidence_entries']}`\n"
        f"- policy rows checked before scoring: `{results['test_policy_rows']}`\n"
        f"- test outcomes absent from policy features: `{results['test_outcomes_absent_from_policy_features']}`\n"
        f"- errors: `{errors}`\n\n"
        "H3-original, H5-P-original and H6-general remain not supported. No user-comprehension or domain-safety result is claimed.\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = validate()
    print(f"PASS: evidence entries={result['evidence_entries']} leakage-safe={result['test_outcomes_absent_from_policy_features']}")


if __name__ == "__main__":
    main()
