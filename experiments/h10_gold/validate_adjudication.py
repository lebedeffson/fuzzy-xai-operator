from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .common import ARTIFACT_ROOT, PRIVATE_ROOT, ROOT, load_config, read_jsonl, write_json
from .metrics import best_set_scores, set_scores


def _load_reviewer(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing real reviewer file: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            case_id = row["case_id"]
            if not row["source_elements_json"] or not row["optimal_cuts_json"] or not row["repair_actions_json"]:
                raise RuntimeError(f"incomplete adjudication row: {path.name}:{case_id}")
            rows[case_id] = {
                "source": json.loads(row["source_elements_json"]),
                "cuts": json.loads(row["optimal_cuts_json"]),
                "repair": json.loads(row["repair_actions_json"]),
                "ambiguous": str(row["ambiguous"]).strip().lower() in {"1", "true", "yes"},
            }
    if set(rows) != expected_ids:
        raise RuntimeError(f"reviewer identities differ from blind sample: {path.name}")
    return rows


def validate(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    blind = read_jsonl(ARTIFACT_ROOT / "adjudication" / "blind_cases.jsonl")
    expected_ids = {row["case_id"] for row in blind}
    reviewer_1 = _load_reviewer(ARTIFACT_ROOT / "adjudication" / "reviewer_1.csv", expected_ids)
    reviewer_2 = _load_reviewer(ARTIFACT_ROOT / "adjudication" / "reviewer_2.csv", expected_ids)
    truth_by_id = {row["case_id"]: row for row in read_jsonl(PRIVATE_ROOT / "protocol_validation_truth.jsonl")}
    inter_source: list[float] = []
    inter_repair: list[float] = []
    oracle_source: dict[str, list[float]] = {"reviewer_1": [], "reviewer_2": []}
    oracle_repair: dict[str, list[float]] = {"reviewer_1": [], "reviewer_2": []}
    cut_scores: dict[str, list[float]] = {"reviewer_1": [], "reviewer_2": []}
    disagreement_cases: list[str] = []
    for case_id in sorted(expected_ids):
        first, second = reviewer_1[case_id], reviewer_2[case_id]
        truth = truth_by_id[case_id]
        inter_source.append(set_scores(first["source"], second["source"])[2])
        inter_repair.append(set_scores(first["repair"], second["repair"])[2])
        for name, response in (("reviewer_1", first), ("reviewer_2", second)):
            oracle_source[name].append(set_scores(truth["source_truth"], response["source"])[2])
            oracle_repair[name].append(set_scores(truth["repair_truth"], response["repair"])[2])
            reviewer_cuts = response["cuts"] if response["cuts"] and isinstance(response["cuts"][0], list) else [response["cuts"]]
            cut_scores[name].append(max(best_set_scores(truth["optimal_cuts"], cut)[2] for cut in reviewer_cuts))
        if inter_source[-1] < 0.5 or inter_repair[-1] < 0.5:
            disagreement_cases.append(case_id)
    def mean(values: list[float]) -> float:
        return sum(values) / len(values)
    criteria = config["adjudication"]
    result = {
        "status": "PASS",
        "cases": len(expected_ids),
        "reviewers": 2,
        "inter_reviewer_source_f1": mean(inter_source),
        "inter_reviewer_repair_f1": mean(inter_repair),
        "oracle_source_f1": {name: mean(values) for name, values in oracle_source.items()},
        "oracle_repair_f1": {name: mean(values) for name, values in oracle_repair.items()},
        "oracle_cut_f1": {name: mean(values) for name, values in cut_scores.items()},
        "systematic_disagreement_rate": len(disagreement_cases) / len(expected_ids),
        "disagreement_case_ids": disagreement_cases,
        "responses_are_human_supplied": True,
        "responses_generated_by_pipeline": False,
    }
    if result["inter_reviewer_source_f1"] < float(criteria["minimum_inter_reviewer_source_f1"]):
        result["status"] = "BLOCKED_INTER_REVIEWER_DISAGREEMENT"
    if min(result["oracle_source_f1"].values()) < float(criteria["minimum_oracle_source_f1"]):
        result["status"] = "BLOCKED_ORACLE_SOURCE_DISAGREEMENT"
    if result["systematic_disagreement_rate"] > float(criteria["maximum_systematic_disagreement_rate"]):
        result["status"] = "BLOCKED_SYSTEMATIC_DISAGREEMENT"
    write_json(ARTIFACT_ROOT / "adjudication" / "adjudication_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_final_gold_protocol.yaml")
    args = parser.parse_args()
    result = validate(args.config)
    if result["status"] != "PASS":
        raise SystemExit(result["status"])


if __name__ == "__main__":
    main()
