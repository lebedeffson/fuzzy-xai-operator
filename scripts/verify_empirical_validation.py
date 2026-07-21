"""Verify the measured Chapter 4 package and preserve external-gate blockers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EMPIRICAL = ROOT / "release_evidence/empirical_experiments/breast_cancer_checkpoint"
CHAPTER = ROOT / "release_evidence/chapter4_empirical_validation"
CONTROLLED = ROOT / "release_evidence/controlled_fixtures/object_85_controlled_story_fixture/origin.json"
PILOT = ROOT / "release_evidence/user_study/comprehension_pilot/scoring_report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(root: Path, manifest_path: Path) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in payload["files"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"manifest mismatch: {path}")


def main() -> None:
    summary = json.loads((EMPIRICAL / "empirical_summary.json").read_text(encoding="utf-8"))
    controlled = json.loads(CONTROLLED.read_text(encoding="utf-8"))
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    cross_model = json.loads((EMPIRICAL / "cross_model_matrix.json").read_text(encoding="utf-8"))
    if summary["checkpoints"] < 15 or summary["checkpoint_hashes_unique"] < 15:
        raise RuntimeError("real training requires at least 15 unique checkpoint states")
    if not summary["selected_case"]["forgetting_events"]:
        raise RuntimeError("forgetting case was not detected")
    if summary["selected_case"]["public_id"] != "case_real_001":
        raise RuntimeError("empirical case must not reuse object 85")
    if controlled["source_type"] != "controlled" or summary["result_origin"] != "measured":
        raise RuntimeError("controlled and measured origins are mixed")
    roles = set(summary["similar_case_roles"])
    if roles != {"support", "counterexample"}:
        raise RuntimeError("similar cases require one support and one counterexample")
    if summary["counterfactual_modes"] != ["sensitivity_analysis"]:
        raise RuntimeError("unreviewed intervention must remain sensitivity analysis")
    by_model = {item["model"]: item for item in cross_model}
    if by_model["black_box_callable"]["native_rule_count"] != 0:
        raise RuntimeError("black-box callable received fabricated native rules")
    if by_model["decision_tree"]["native_rule_count"] <= 0:
        raise RuntimeError("decision tree did not expose native paths")
    if by_model["sugeno_native_rules"]["native_rule_count"] <= 0:
        raise RuntimeError("rule model did not expose native rules")
    if pilot["status"] != "planned_not_run" or pilot["claim_allowed"]:
        raise RuntimeError("pilot status must remain blocked until independent responses exist")
    verify_manifest(EMPIRICAL, EMPIRICAL / "manifest_sha256.json")
    verify_manifest(CHAPTER, CHAPTER / "manifest_sha256.json")
    print("PASS: controlled_empirical_separation")
    print(f"PASS: measured_checkpoints {summary['checkpoints']}")
    print(f"PASS: automatic_forgetting_case {summary['selected_case']['public_id']}")
    print("PASS: measured_rule_ablation")
    print(f"PASS: cross_model_matrix {len(cross_model)}")
    print("PASS: chapter4_empirical_manifest")
    print("BLOCKED: comprehension_pilot planned_not_run")
    print("BLOCKED: release_tag")


if __name__ == "__main__":
    main()
