#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuzzyxai.experiments.h10_c7r import load_held_out_inputs
from fuzzyxai.experiments.h10_c7r_r10 import (
    development_gates,
    score_r10_variants,
    select_loro_variant,
    summarize_r10,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repository-separated R10 development scoring"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclusion-lock", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    if not readiness.get("summary", {}).get("all_incidents_ready", False):
        raise RuntimeError(
            "R10 development scoring requires all causal runtime incidents ready"
        )

    inputs = load_held_out_inputs(
        args.manifest,
        args.gold,
        args.exclusion_lock,
    )
    engine = GuidedNaturalDiagnosisEngine(structural_only=True)
    variants = ("R10A",)
    rows = score_r10_variants(inputs, engine=engine, variants=variants)
    selected, folds = select_loro_variant(rows, variants=variants)
    summary = summarize_r10(selected)
    gates = development_gates(summary)
    gate_passed = all(gates.values())
    status = {
        "protocol_id": "H10-C7R-R10-development-v1",
        "status": (
            "H10_C7R_R10_DEVELOPMENT_GO"
            if gate_passed
            else "H10_C7R_R10_DEVELOPMENT_NO_GO"
        ),
        "scientific_result": "NOT_EVALUATED",
        "variants_executed": list(variants),
        "source_aware_model_executed": False,
        "runtime_readiness": "PASS",
        "loro": summary,
        "gates": gates,
        "gate_passed": gate_passed,
        "ready_for_model_lock": gate_passed,
        "new_held_out_created": False,
        "new_held_out_scored": False,
    }
    output = args.output.resolve()
    _write_jsonl(output / "R10_PER_VARIANT.jsonl", rows)
    _write_jsonl(output / "R10_LORO_SELECTED.jsonl", selected)
    _write_json(output / "R10_LORO_FOLDS.json", folds)
    _write_json(output / "R10_DEVELOPMENT_STATUS.json", status)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
