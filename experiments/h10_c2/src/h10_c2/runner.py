from __future__ import annotations

import csv
import json
from pathlib import Path

from .baselines import METHODS
from .data import generate_cases, write_split
from .hashing import read_json, write_json
from .methods import run as run_fuzzyxai
from .metrics import score_cut, score_repair
from .oracle import derive_gold
from .paths import ARTIFACT_ROOT
from .repair import execute_plan
from .statistics.aggregation import select_baseline
from .statistics.paired_analysis import analyze_primary


def generate_from_design(split: str, *, seed_offset: int = 0) -> dict:
    design = read_json(ARTIFACT_ROOT / "power" / "recommended_design.json")
    if split == "sealed":
        if design["status"] != "power_target_reached":
            raise PermissionError("BLOCKED_POWER: sealed generation requires an adequately powered design")
        if not (ARTIFACT_ROOT / "lock" / "protocol.lock.json").exists():
            raise PermissionError("BLOCKED_PROTOCOL: sealed generation requires a protocol lock")
    total = int(design["h10_c2a"]["recommended_total_cases"])
    fractions = {"development": 0.35, "protocol_validation": 0.15, "sealed": 0.50}
    count = max(60, round(total * fractions[split]))
    return write_split(split, count, 221003 + seed_offset, include_private_gold=True)


def _method_functions() -> dict:
    return {**METHODS, "fuzzyxai_v21": run_fuzzyxai}


def run_split(split: str) -> Path:
    manifest = read_json(ARTIFACT_ROOT / "data" / split / "manifest.json")
    seeds = {"development": 1, "protocol_validation": 2, "sealed": 3}
    cases = generate_cases(split, int(manifest["case_count"]), seed=221003 + seeds[split])
    rows = []
    for case in cases:
        gold = derive_gold(case)
        method_case = case.method_view()
        for name, method in _method_functions().items():
            result = method(method_case)
            cut = score_cut(result, gold, case.public_obligations)
            execution = execute_plan(case.observed_route, case.clean_route, result.repair_actions)
            repair = score_repair(execution, result.predicted_cost, gold.optimal_cost)
            rows.append(
                {
                    "case_id": case.case_id,
                    "pipeline": case.pipeline,
                    "modality": case.modality,
                    "split": split,
                    "case_type": case.case_type,
                    "gold_status": gold.gold_status,
                    "repairable": gold.repairable,
                    "method": name,
                    "predicted_cut": json.dumps(result.predicted_cut),
                    "predicted_cost": result.predicted_cost,
                    "runtime_ms": result.runtime_ms,
                    **cut,
                    **repair,
                }
            )
    output = ARTIFACT_ROOT / "results" / f"{split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if split == "development":
        baseline = select_baseline(rows)
        write_json(
            ARTIFACT_ROOT / "lock" / "baseline_selection.json",
            {"split": "development", "metric": "optimal_cut_set_membership", "selected_baseline": baseline},
        )
    return output


def load_result_rows(split: str) -> list[dict]:
    with (ARTIFACT_ROOT / "results" / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_nonconfirmatory_statistics(split: str) -> list[dict]:
    rows = load_result_rows(split)
    baseline = read_json(ARTIFACT_ROOT / "lock" / "baseline_selection.json")["selected_baseline"]
    numeric = []
    for row in rows:
        converted = dict(row)
        for key in ("optimal_cut_set_membership", "full_recertification_success"):
            converted[key] = float(row[key] == "True")
        numeric.append(converted)
    repetitions = 10000 if split == "sealed" else 1000
    results = analyze_primary(numeric, baseline, repetitions=repetitions, seed=221020)
    for item in results:
        if split == "sealed":
            item["status"] = "confirmatory_scored"
            item["confirmatory"] = True
        else:
            item["status"] = "development_only" if split == "development" else "protocol_validation_only"
            item["confirmatory"] = False
    write_json(ARTIFACT_ROOT / "results" / f"{split}_statistics.json", results)
    return results
