"""Verify the v1.1 claim, visual, and golden-evidence release surface."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from fuzzyxai.runtime import ModelExplanationResult
from fuzzyxai.schemas import validate_payload


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "framework/fuzzyxai/explanation_experience_manifest.json"
GOLDEN = ROOT / "release_evidence/explanation_experience"


def resolve(path: str):
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for contract in manifest["contracts"]:
        resolve(contract)
    for method in manifest["result_api"]:
        if not callable(getattr(ModelExplanationResult, method, None)):
            raise RuntimeError(f"missing result API method: {method}")
    for relative in manifest["golden_explanations"]:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"missing golden explanation: {relative}")
    hashes = json.loads((GOLDEN / "manifest_sha256.json").read_text(encoding="utf-8"))["files"]
    for name, expected in hashes.items():
        actual = hashlib.sha256((GOLDEN / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"checksum mismatch: {name}")
    object_85 = json.loads((GOLDEN / "object_85_explanation.json").read_text(encoding="utf-8"))
    if object_85["explanation_level"]["level"] != "E5":
        raise RuntimeError("object 85 must exercise the controlled E5 route")
    if len(object_85["visual_spec"]["training_timeline"][0]["points"]) < 12:
        raise RuntimeError("object 85 requires at least 12 observed checkpoints")
    visual_validation = validate_payload(object_85["visual_spec"], "explanation_visual_spec")
    if not visual_validation.valid or not object_85["visual_spec"]["audit"]["graph_valid"]:
        raise RuntimeError(f"VisualSpec/graph validation failed: {visual_validation.errors}")
    medical = json.loads((GOLDEN / "medical_research_similarity_explanation.json").read_text(encoding="utf-8"))
    if medical["clinical_claims"] or not all(claim["limitations"] for claim in medical["claims"]):
        raise RuntimeError("medical fixture must remain research-only and limitation-backed")
    if medical.get("counterexample_count", 0) < 2 or not medical.get("media_artifacts"):
        raise RuntimeError("medical fixture requires media artifacts and two counterexamples")
    matrix = json.loads((GOLDEN / "cross_model/cross_model_matrix.json").read_text(encoding="utf-8"))
    if not all(item["graph_valid"] for item in matrix.values()):
        raise RuntimeError("cross-model graph validation failed")
    if "rules" not in matrix["sklearn_linear"]["surrogate_channels"]:
        raise RuntimeError("linear rules must remain labelled surrogate")
    if "rules" not in matrix["tree_native_paths"]["native_channels"]:
        raise RuntimeError("tree decision paths must remain native")
    if manifest["human_validation"]["status"] != "planned_not_run":
        raise RuntimeError("human validation status changed without a reviewed study artifact")
    print("PASS: explanation_claim_contract")
    print("PASS: explanation_levels_E0_E5")
    print("PASS: explanation_visual_spec")
    print("PASS: golden_explanations")
    print("PASS: golden_checksums")
    print("PASS: cross_model_explanations")
    print("PASS: object85_12_checkpoint_trace")
    print("PASS: comprehension_study_planned_not_run")


if __name__ == "__main__":
    main()
