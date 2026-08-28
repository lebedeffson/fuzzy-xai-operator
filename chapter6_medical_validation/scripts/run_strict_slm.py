"""Run the same pinned local SLM over public human layers only.

The script never loads raw images, ECGs, model internals, or system formulas.
It reconstructs the already serialized public HumanExplanation and uses strict
mode, where the model selects claim IDs but cannot author factual text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from fuzzyxai.evidence.contracts import (
    ActionStatement,
    ConcernStatement,
    DecisionStatement,
    ExplanationDetails,
    ExplanationGraph,
    HumanExplanation,
    ReasonStatement,
    ReliabilityStatement,
)
from fuzzyxai.verbalization import SLMVerbalizer
from fuzzyxai.verbalization.atomic_claims import extract_atomic_claims
from fuzzyxai.verbalization.verbalizer import _strict_prompt

from chapter6_medical_validation.shared.transformers_verbalizer import LocalTransformersBackend

PINNED_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PINNED_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def default_model_path() -> Path:
    """Locate the pinned experiment-side snapshot under the declared data root."""

    data_root = os.environ.get("FUZZYXAI_CH6_DATA_ROOT")
    if not data_root:
        raise FileNotFoundError(
            "FUZZYXAI_CH6_DATA_ROOT is required to locate the pinned local SLM snapshot"
        )
    return (
        Path(data_root)
        / "slm-cache"
        / "hub"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / PINNED_REVISION
    )


def _statement(payload: dict[str, object], cls=ConcernStatement):
    return cls(str(payload["title"]), str(payload["explanation"]), tuple(payload["claim_refs"]), tuple(payload["evidence_refs"]))


def _human(payload: dict[str, object]) -> HumanExplanation:
    decision_raw = dict(payload["decision"])
    decision = DecisionStatement(str(decision_raw["title"]), str(decision_raw["explanation"]), tuple(decision_raw["claim_refs"]), tuple(decision_raw["evidence_refs"]), str(decision_raw["domain_language_status"]))
    reasons = tuple(
        ReasonStatement(str(item["title"]), str(item["explanation"]), tuple(item["claim_refs"]), tuple(item["evidence_refs"]), str(item.get("subject_label", item["title"])), str(item.get("effect_direction", "mixed")), str(item.get("comparison_text", "not stated")))
        for item in payload.get("main_reasons", [])
    )
    concerns = tuple(_statement(dict(item)) for item in payload.get("concerns", []))
    reliability_raw = dict(payload["reliability"])
    reliability = ReliabilityStatement(str(reliability_raw["title"]), str(reliability_raw["explanation"]), tuple(reliability_raw["claim_refs"]), tuple(reliability_raw["evidence_refs"]), tuple(reliability_raw["supported_by"]), tuple(reliability_raw["limited_by"]), tuple(reliability_raw["missing_evidence"]), str(reliability_raw["conclusion"]))
    action_raw = dict(payload["recommended_action"])
    action = ActionStatement(str(action_raw["title"]), str(action_raw["explanation"]), tuple(action_raw["claim_refs"]), tuple(action_raw["evidence_refs"]), str(action_raw["action"]))
    return HumanExplanation(str(payload["audience"]), str(payload["language"]), decision, reasons, concerns, reliability, action, (), ExplanationDetails(), ExplanationGraph((), ()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audience", required=True)
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result_json.read_text(encoding="utf-8"))
    # P19 public serialization is audience-indexed.  Retain the legacy field
    # only for frozen earlier artifacts; no model evidence is reconstructed.
    human_payload = payload.get("human_explanation")
    if human_payload is None:
        human_layers = payload.get("human_explanations", {})
        if not isinstance(human_layers, dict) or not isinstance(human_layers.get("domain_user"), dict):
            raise KeyError("public result has neither human_explanation nor human_explanations.domain_user")
        human_payload = human_layers["domain_user"]
    explanation = _human(dict(human_payload))
    model_path = args.model_path or default_model_path()
    backend = LocalTransformersBackend(model_path, model_id=PINNED_MODEL_ID, revision=PINNED_REVISION)
    result = SLMVerbalizer(backend, mode="strict").run(explanation, template_text=explanation.user_text, audience=args.audience)
    accepted = result.status == "generated"
    source_ids = set(result.source_claim_ids)
    metrics = {
        "P_fact": 1.0 if accepted else 0.0,
        "H": 0 if accepted else None,
        "P_num": 1.0 if accepted else 0.0,
        "P_action": 1.0 if "action-0" in source_ids else 0.0,
        "P_lim": 1.0 if any(value.startswith("concern-") or value == "reliability-0" for value in source_ids) else 0.0,
    }
    output = {
        "mode": "strict",
        "model_id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "generation": {"do_sample": False, "max_new_tokens": 256},
        "prompt_profile_sha256": hashlib.sha256(
            _strict_prompt(extract_atomic_claims(explanation), args.audience).encode("utf-8")
        ).hexdigest(),
        "source_result": str(args.result_json),
        "status": result.status,
        "guard_checks": list(result.guard_checks),
        "fallback_reason": result.fallback_reason,
        "source_claim_ids": list(result.source_claim_ids),
        "metrics": metrics,
        "text": result.text,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": result.status, "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
