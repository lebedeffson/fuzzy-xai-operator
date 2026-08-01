#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from fuzzyxai.experiments.h10_c7r import load_held_out_inputs
from fuzzyxai.experiments.h10_c7r_r9 import (
    AVAILABLE_STRUCTURAL_VARIANTS,
    build_feature_cases,
    load_published_v1_rows,
    score_loro_lambdamart,
    selected_summary,
    summarize,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _csv(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(values[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclusion-lock", type=Path, required=True)
    parser.add_argument("--v1-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()

    published = load_published_v1_rows(args.v1_results)
    inputs = load_held_out_inputs(
        args.manifest,
        args.gold,
        args.exclusion_lock,
    )
    engine = GuidedNaturalDiagnosisEngine(structural_only=True)
    feature_cases = build_feature_cases(inputs, engine=engine)
    rows, selected, folds = score_loro_lambdamart(
        feature_cases,
        compressor=engine.r9_compressor,
    )
    variant_summaries = [
        asdict(summarize(rows, variant=variant))
        for variant in AVAILABLE_STRUCTURAL_VARIANTS
    ]
    loro = selected_summary(selected)
    gates = {
        "recall_at_20_at_least_0_82": loro["recall_at_20"] >= 0.82,
        "repository_lower_quartile_at_least_0_75": (
            loro["repository_recall_lower_quartile"] >= 0.75
        ),
        "schema_gap_zero": loro["schema_gap_count"] == 0,
        "contract_does_not_reorder": loro["contract_reordering_count"] == 0,
        "exact_candidates_at_most_500": (
            loro["maximum_exact_candidates"] <= 500
        ),
    }
    gate_passed = all(gates.values())
    status = {
        "protocol_id": "H10-C7R-R9-development-v1",
        "status": (
            "H10_C7R_R9_DEVELOPMENT_GO"
            if gate_passed
            else "H10_C7R_R9_DEVELOPMENT_NO_GO"
        ),
        "scientific_result": "NOT_EVALUATED",
        "h10_c7r_v1_status_preserved": "H10_C7R_NOT_SUPPORTED",
        "published_v1_rows_read": len(published),
        "published_v1_recalculated": False,
        "container_execution": False,
        "runtime_recollection": False,
        "neural_models_executed": False,
        "ranker": "lightgbm_lambdamart",
        "ranker_version": "4.6.0",
        "ranker_training_scope": "leave_one_repository_out",
        "test_repository_used_for_training": False,
        "loro": loro,
        "gates": gates,
        "gate_passed": gate_passed,
        "ready_for_new_held_out": gate_passed,
        "new_held_out_created": False,
        "new_held_out_scored": False,
    }
    output = args.output.resolve()
    reports = args.reports.resolve()
    _jsonl(output / "R9_PER_VARIANT.jsonl", rows)
    _jsonl(output / "R9_LORO_SELECTED.jsonl", selected)
    _csv(output / "R9_LORO_FOLDS.csv", folds)
    _json(
        output / "R9_LORO_MODEL_LOCKS.json",
        [
            {
                "held_repository": fold["held_repository"],
                "selected_variant": fold["selected_variant"],
                "training_repositories": str(
                    fold["training_repositories"]
                ).split("|"),
                "ranker": fold["ranker"],
                "model_sha256": fold["model_sha256"],
                "feature_importances": json.loads(
                    str(fold["feature_importances"])
                ),
            }
            for fold in folds
        ],
    )
    _json(output / "R9_VARIANT_SUMMARIES.json", variant_summaries)
    _json(output / "R9_DEVELOPMENT_STATUS.json", status)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "R9_DEVELOPMENT_REPORT.md").write_text(
        _report(status, variant_summaries),
        encoding="utf-8",
    )
    print(json.dumps(status, sort_keys=True))
    return 0


def _report(
    status: dict[str, object],
    variants: list[dict[str, object]],
) -> str:
    loro = status["loro"]
    rows = "\n".join(
        f"| {item['variant']} | {item['recall_at_20']:.4f} | "
        f"{item['repository_recall_lower_quartile']:.4f} |"
        for item in variants
    )
    return f"""# H10-C7R R9 development report

The disclosed H10-C7R-v1 held-out set was used only as development data.
Published v1 rows were read, not recalculated. No containers, runtime
collection, neural models, or new held-out scoring were executed.

| Variant | Recall@20 | Repository recall Q1 |
|---|---:|---:|
{rows}

LORO-selected Recall@20: `{loro['recall_at_20']:.4f}`.
Repository recall lower quartile:
`{loro['repository_recall_lower_quartile']:.4f}`.

Status: `{status['status']}`.
Scientific result: `NOT_EVALUATED`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
