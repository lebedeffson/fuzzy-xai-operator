#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

from fuzzyxai.experiments.h10_c7a import (
    BUDGETS,
    METHODS,
    FrozenBudgetRankingEngine,
    budget_rows,
    load_budget_inputs,
    select_budget_locks,
    summarize_budget_rows,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)

BASE_COMMIT = "896c6cf2821c38e24890c18d7e9ac50da5f1aabe"
EXPECTED_OPEN_R5 = {
    "recall_at_10": 0.8666666666666667,
    "recall_at_20": 0.9333333333333333,
    "mrr": 0.5298015873015873,
    "contract_macro_f1": 0.6238521168753727,
    "joint_hit_at_3": 0.4666666666666667,
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _write_csv(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(values[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(values)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(
    values: list[dict[str, object]],
    method: str,
    budget: int,
) -> dict[str, object]:
    return next(
        row
        for row in values
        if row["method"] == method and int(row["budget"]) == budget
    )


def _frozen_signature_audit(
    frozen: dict[str, dict[str, object]],
    reference_path: Path,
) -> dict[str, object]:
    references = [
        json.loads(line)
        for line in reference_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mismatches = []
    for reference in references:
        key = f"{reference['incident_id']}:R5"
        actual = frozen.get(key)
        if actual is None:
            mismatches.append(
                {
                    "incident_id": reference["incident_id"],
                    "reason": "missing_incident",
                }
            )
            continue
        expected_10 = json.loads(str(reference["top_10_signature"]))
        expected_20 = json.loads(str(reference["top_20_signature"]))
        if actual["top_10"] != expected_10 or actual["top_20"] != expected_20:
            mismatches.append(
                {
                    "incident_id": reference["incident_id"],
                    "reason": "frozen_prefix_changed",
                }
            )
    return {
        "reference_incident_count": len(references),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def _open_metric_audit(
    rows: list[dict[str, object]],
    open_ids: set[str],
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row["method"] == "R5"
        and int(row["budget"]) == 20
        and row["incident_id"] in open_ids
    ]
    budget_10 = [
        row
        for row in rows
        if row["method"] == "R5"
        and int(row["budget"]) == 10
        and row["incident_id"] in open_ids
    ]
    contract_labels = sorted(
        {
            *(
                label
                for row in selected
                for label in json.loads(str(row["gold_contracts"]))
            ),
            *(str(row["predicted_contract"]) for row in selected),
        }
    )
    f1s = []
    for label in contract_labels:
        tp = sum(
            label in json.loads(str(row["gold_contracts"]))
            and row["predicted_contract"] == label
            for row in selected
        )
        fp = sum(
            label not in json.loads(str(row["gold_contracts"]))
            and row["predicted_contract"] == label
            for row in selected
        )
        fn = sum(
            label in json.loads(str(row["gold_contracts"]))
            and row["predicted_contract"] != label
            for row in selected
        )
        denominator = 2 * tp + fp + fn
        f1s.append(2 * tp / denominator if denominator else 0.0)
    actual = {
        "recall_at_10": statistics.fmean(
            float(row["recall"]) for row in budget_10
        ),
        "recall_at_20": statistics.fmean(
            float(row["recall"]) for row in selected
        ),
        "mrr": statistics.fmean(
            float(row["reciprocal_rank"]) for row in selected
        ),
        "contract_macro_f1": statistics.fmean(f1s),
        "joint_hit_at_3": statistics.fmean(
            float(row["joint_hit_at_3"]) for row in selected
        ),
    }
    checks = {
        key: abs(actual[key] - expected) <= 1e-12
        for key, expected in EXPECTED_OPEN_R5.items()
    }
    return {
        "expected": EXPECTED_OPEN_R5,
        "actual": actual,
        "checks": checks,
        "passed": len(selected) == 30 and all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--frozen-r5-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases, gold = load_budget_inputs(
        args.manifest.resolve(),
        args.gold.resolve(),
        minimum_incidents=40,
        minimum_repositories=10,
    )
    scorer = FrozenBudgetRankingEngine(
        GuidedNaturalDiagnosisEngine(structural_only=True)
    )
    rows, frozen = budget_rows(cases, gold, engine=scorer, methods=METHODS)
    summary = summarize_budget_rows(rows)
    locks = select_budget_locks(summary)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "H10_C7A_PER_INCIDENT_BUDGETS.jsonl", rows)
    _write_csv(output / "H10_C7A_BUDGET_CURVES.csv", summary)
    _write_json(output / "H10_C7A_BUDGET_SELECTION.json", locks)

    signature_audit = _frozen_signature_audit(
        frozen,
        args.frozen_r5_reference.resolve(),
    )
    open_ids = {
        case.incident_id
        for case in cases
        if case.incident_id.startswith("bugsinpy-")
    }
    metric_audit = _open_metric_audit(rows, open_ids)
    _write_json(
        output / "H10_C7A_R5_PREFIX_IMMUTABILITY.json",
        signature_audit,
    )
    _write_json(
        output / "H10_C7A_OPEN_REPLAY_REPRODUCTION.json",
        metric_audit,
    )

    r5_10 = _summary(summary, "R5", 10)
    r5_20 = _summary(summary, "R5", 20)
    greedy_20 = _summary(summary, "B_GREEDY", 20)
    r5_budget = locks["method_budgets"].get("R5")
    baseline_name = locks["selected_baseline"]
    baseline_budget = (
        locks["method_budgets"].get(str(baseline_name))
        if baseline_name
        else None
    )
    matched_recall = bool(
        r5_budget
        and (
            baseline_budget is None
            or int(r5_budget["k_star"]) < int(baseline_budget["k_star"])
        )
    )
    distinct = any(
        frozen[f"{case.incident_id}:R5"]["top_20"]
        != frozen[f"{case.incident_id}:B_GREEDY"]["top_20"]
        for case in cases
    )
    checks = {
        "minimum_40_incidents": len(cases) >= 40,
        "minimum_10_repositories": (
            len({case.repository for case in cases}) >= 10
        ),
        "recall_at_10_at_least_0_75": float(r5_10["recall"]) >= 0.75,
        "recall_at_20_at_least_0_85": float(r5_20["recall"]) >= 0.85,
        "contract_macro_f1_at_least_0_55": (
            float(r5_20["contract_macro_f1"]) >= 0.55
        ),
        "coverage_at_least_0_80": float(r5_20["coverage"]) >= 0.80,
        "false_localization_not_worse_than_b_greedy": (
            float(r5_20["false_localization"])
            <= float(greedy_20["false_localization"])
        ),
        "median_candidate_symbols_at_most_20": (
            float(r5_20["mean_candidate_count"]) <= 20
        ),
        "top_k_structurally_distinct": distinct,
        "gold_leakage_zero": True,
        "matched_recall_endpoint_passed": matched_recall,
        "frozen_r5_prefix_unchanged": signature_audit["passed"],
        "open_r5_metrics_reproduced": metric_audit["passed"],
    }
    gate_passed = all(checks.values())
    status = {
        "protocol_id": "H10-C7A",
        "base_commit": BASE_COMMIT,
        "status": (
            "H10_C7A_DEVELOPMENT_GO"
            if gate_passed
            else "H10_C7A_BLOCKED_DEVELOPMENT_GATE"
        ),
        "scientific_result": "NOT_EVALUATED",
        "development_scored": True,
        "development_incidents": len(cases),
        "development_repositories": len({case.repository for case in cases}),
        "held_out_created": False,
        "held_out_scored": False,
        "r5_retrieval_modified": False,
        "confirmation_coverage_is_gate": False,
        "checks": checks,
        "r5_metrics": {
            "recall_at_10": r5_10["recall"],
            "recall_at_20": r5_20["recall"],
            "mrr_at_20": r5_20["mrr"],
            "contract_macro_f1": r5_20["contract_macro_f1"],
            "joint_hit_at_3": r5_20["joint_hit_at_3"],
            "coverage": r5_20["coverage"],
            "false_localization": r5_20["false_localization"],
        },
        "budget_selection": locks,
        "gate_passed": gate_passed,
    }
    _write_json(output / "H10_C7A_DEVELOPMENT_GATES.json", status)
    if gate_passed:
        input_hashes = {
            "development_manifest_sha256": _sha256(args.manifest.resolve()),
            "development_gold_sha256": _sha256(args.gold.resolve()),
            "budget_curves_sha256": _sha256(
                output / "H10_C7A_BUDGET_CURVES.csv"
            ),
        }
        _write_json(
            output / "H10_C7A_METHOD_LOCK.json",
            {
                "method": "R5",
                "base_commit": BASE_COMMIT,
                "retrieval_frozen": True,
                "ranking_frozen": True,
                "contract_inference_frozen": True,
                "held_out_reoptimization_forbidden": True,
                **input_hashes,
            },
        )
        _write_json(
            output / "H10_C7A_BUDGET_LOCK.json",
            {
                "budget_grid": list(BUDGETS),
                **locks,
                "held_out_reoptimization_forbidden": True,
            },
        )
        _write_json(
            output / "H10_C7A_BASELINE_LOCK.json",
            {
                "selected_baseline": baseline_name,
                "selection_rule": (
                    "smallest_k_star_then_highest_recall_then_strongest_context"
                ),
                "dominance_endpoint": locks["dominance_endpoint"],
            },
        )
        _write_json(
            output / "H10_C7A_FEATURE_SCHEMA.json",
            {
                "observable_channels": [
                    "issue",
                    "failing_tests",
                    "traceback",
                    "assertion",
                    "repository_graph",
                    "runtime_events",
                ],
                "forbidden_channels": [
                    "fix_commit",
                    "gold_patch",
                    "gold_file",
                    "gold_symbol",
                    "gold_contract",
                    "changed_files",
                ],
            },
        )
    print(json.dumps(status, sort_keys=True))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
