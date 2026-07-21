from __future__ import annotations

import argparse

from fuzzyxai.practical_controller import (
    CostProfileName,
    DeploymentContext,
    ExplanationArtifact,
    PracticalPolicy,
    PredictionArtifact,
    ReviewBudget,
    RouteArtifacts,
    assess_action,
    cost_profile,
    verify_replay,
)

from .common import canonical_bytes, sha256_bytes, verify_protocol_hash
from .run_route_faults import clean_route, inject, simple_or, typed_route_validator


def run() -> dict[str, object]:
    verify_protocol_hash()
    payload = {"object_id": "smoke:1", "tokens": ["market", "report"], "scores": [0.7, -0.2]}
    digest = sha256_bytes(canonical_bytes(payload))
    prediction = PredictionArtifact("smoke:1", "Business", 0.8, (0.05, 0.05, 0.8, 0.1), "smoke-model", entropy=0.4, prediction_margin=0.7)
    explanation = ExplanationArtifact(digest, "smoke-explainer", "smoke-model", "v13", "v13-dictionary", ("prediction", "model", "explainer"))
    route = RouteArtifacts("smoke-preprocessing", "smoke-calibration", "smoke-reference", "v13", digest, ("prediction", "model", "explainer"))
    context = DeploymentContext(
        "smoke-model",
        "smoke-preprocessing",
        "smoke-explainer",
        "smoke-calibration",
        "smoke-reference",
        "v13",
        "v13",
        "v13-dictionary",
        digest,
        ("prediction", "model", "explainer"),
        0.25,
        "smoke-policy",
    )
    policy = PracticalPolicy("1.0", "smoke-policy", (0.1,) * 8, -1.0, (0.1,) * 10, -1.0, 0.25, 0.5, 0.8, "platt", (1.0, 0.0), "2" * 64, True)
    assessment = assess_action(prediction, explanation, route, context, ReviewBudget(1.0), cost_profile(CostProfileName.BALANCED), policy=policy)
    held_out = clean_route(1)
    inject(held_out, "mixed_model_artifacts")
    if simple_or(held_out)[0] or typed_route_validator(held_out)[0] != ["mixed_model_artifacts"]:
        raise RuntimeError("route smoke did not distinguish typed validation from simple OR")
    if not verify_replay(assessment):
        raise RuntimeError("action replay hash is not deterministic")
    return {"action": assessment.action.value, "trace_id": assessment.trace_id, "canonical_sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = run()
    print(f"PASS: chapter4-v13 smoke action={result['action']} trace={str(result['trace_id'])[:12]}")


if __name__ == "__main__":
    main()
