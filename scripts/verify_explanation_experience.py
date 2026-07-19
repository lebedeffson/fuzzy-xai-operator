"""Verify the v1.2 claim, human, visual, and golden-evidence release surface."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from pathlib import Path

from fuzzyxai.runtime import ModelExplanationResult
from fuzzyxai.schemas import validate_payload


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "framework/fuzzyxai/explanation_experience_manifest.json"
GOLDEN = ROOT / "release_evidence/explanation_experience"
FORBIDDEN_USER_TERMS = re.compile(
    r"\b(?:R\d+|S\d+|E[0-5]|gamma|delta|rho|claim_id|defer_to_human|audit_report)\b|\[C-",
    re.IGNORECASE,
)


def resolve(path: str):
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def verify_human_explanation(payload: dict, *, expected_audience: str = "domain_user") -> None:
    validation = validate_payload(payload, "human_explanation")
    if not validation.valid:
        raise RuntimeError(f"HumanExplanation validation failed: {validation.errors}")
    if payload["audience"] != expected_audience:
        raise RuntimeError(f"unexpected human audience: {payload['audience']}")
    if len(payload["main_reasons"]) > 3 or len(payload["concerns"]) > 2:
        raise RuntimeError("domain-user explanation exceeds the card limits")
    fragments = [
        payload["decision"],
        *payload["main_reasons"],
        *payload["concerns"],
        payload["reliability"],
        payload["recommended_action"],
        *payload["what_would_change_result"],
    ]
    for fragment in fragments:
        if not fragment.get("claim_refs") or not fragment.get("evidence_refs"):
            raise RuntimeError("human fragment is not grounded in claims and evidence")
    user_text = str(payload.get("summary", ""))
    if FORBIDDEN_USER_TERMS.search(user_text):
        raise RuntimeError("technical term leaked into domain-user explanation")


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
    object_85_human = json.loads((GOLDEN / "object_85_human_explanation.json").read_text(encoding="utf-8"))
    verify_human_explanation(object_85_human)
    if "16-го этапа" not in object_85_human["summary"]:
        raise RuntimeError("object 85 human explanation must identify the observed epoch-16 forgetting transition")
    medical = json.loads((GOLDEN / "medical_research_similarity_explanation.json").read_text(encoding="utf-8"))
    if medical["clinical_claims"] or not all(claim["limitations"] for claim in medical["claims"]):
        raise RuntimeError("medical fixture must remain research-only and limitation-backed")
    if medical.get("counterexample_count", 0) < 2 or not medical.get("media_artifacts"):
        raise RuntimeError("medical fixture requires media artifacts and two counterexamples")
    medical_human = json.loads((GOLDEN / "medical_research_human_explanation.json").read_text(encoding="utf-8"))
    verify_human_explanation(medical_human)
    if "геометрии выделенных областей" not in medical_human["summary"] or "не является вероятностью" not in medical_human["summary"]:
        raise RuntimeError("medical user text must define the scope and limitation of mask similarity")
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
    print("PASS: human_explanation_layer")
    print("PASS: object85_human_cards")
    print("PASS: medical_similarity_semantics")
    print("PASS: comprehension_study_planned_not_run")


if __name__ == "__main__":
    main()
