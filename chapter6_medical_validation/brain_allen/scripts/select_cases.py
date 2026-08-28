from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def normalized_l1_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    """Diagnostic map distance; deliberately not the operator Gamma."""

    vectors = []
    for value in (left, right):
        flat = np.maximum(np.asarray(value, dtype=float), 0.0).reshape(-1)
        total = float(flat.sum())
        vectors.append(flat / total if total > 0 else flat)
    return float(np.abs(vectors[0] - vectors[1]).sum() / 2.0)


def choose(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    correct_hpf = [row for row in rows if row["correct"] and row["label"] == 1]
    correct_other = [row for row in rows if row["correct"] and row["label"] == 0]
    errors = [row for row in rows if not row["correct"]]
    selected: dict[str, dict[str, object]] = {}
    selected["BRAIN_A"] = max(correct_hpf, key=lambda row: float(row["confidence"])) if correct_hpf else {"status": "not_available", "reason": "no correct HPF test object"}
    selected["BRAIN_B"] = max(correct_other, key=lambda row: float(row["confidence"])) if correct_other else {"status": "not_available", "reason": "no correct OTHER test object"}
    selected["BRAIN_C"] = min(rows, key=lambda row: float(row["top1_top2_margin"]))
    selected["BRAIN_D"] = max(errors, key=lambda row: float(row["confidence"])) if errors else {"status": "not_available", "reason": "canonical test predictions contain no errors"}
    selected["BRAIN_E"] = max(rows, key=lambda row: float(row["xai_diagnostic_disagreement"]))
    selected["BRAIN_F"] = min(rows, key=lambda row: float(row["technical_quality_score"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    cases_root = ROOT / args.output_root / "cases"
    rows = json.loads((cases_root / "case_summaries.json").read_text(encoding="utf-8"))
    for row in rows:
        case_dir = cases_root / str(row["object_id"])
        grad_cam = np.load(case_dir / "grad_cam_raw.npy")
        integrated = np.load(case_dir / "integrated_gradients_signed.npy")
        positive_ig = np.maximum(integrated, 0.0).sum(axis=0)
        row["xai_diagnostic_disagreement"] = normalized_l1_disagreement(grad_cam, positive_ig)
        row["xai_disagreement_semantics"] = "diagnostic normalized-L1 map distance; not FuzzyXAI Gamma"
    (cases_root / "case_summaries.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    payload = {"selection_policy": "deterministic over frozen canonical test predictions", "cases": choose(rows)}
    (cases_root / "selected_cases.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(cases_root / "selected_cases.json")


if __name__ == "__main__":
    main()
