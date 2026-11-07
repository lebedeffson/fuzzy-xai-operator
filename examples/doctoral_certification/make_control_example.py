"""Generate a deliberately synthetic, fully attributed control input.

The example only checks operator consistency, NEVER a trained predictor.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HEX = lambda msg: hashlib.sha256(msg.encode()).hexdigest()


def build() -> dict:
    weights = {"rho_p": 0.30, "u_M": 0.20, "one_minus_I_pre": 0.20, "Delta": 0.20, "chi_R": 0.10}
    comps = {"rho_p": 0.10, "u_M": 0.15, "one_minus_I_pre": 0.10, "Delta": 0.05, "chi_R": 0}
    rho = sum(weights[k] * comps[k] for k in comps)
    return {
        "trace": {
            "adapter_id": "control/artificial-rule-adapter", "input_sha256": HEX("synthetic-input"),
            "explain_plan_sha256": HEX("synthetic-explain-plan"), "model_fingerprint": HEX("artificial-model"),
            "dataset_version": "control-fixture-v1", "object_ids": ["synthetic-1"],
            "generated_at": "2026-11-07T16:00:00+00:00", "result_origin": "controlled_fixture"
        },
        "explanation_graph": {
            "nodes": [
                {"node_id": "dataset:root", "node_type": "dataset", "payload": {"dataset_version": "control-fixture-v1"}},
                {"node_id": "model_artifact:root", "node_type": "model_artifact", "payload": {"model_fingerprint": HEX("artificial-model")}},
                {"node_id": "data:synthetic-1", "node_type": "data", "evidence_refs": []},
                {"node_id": "prediction", "node_type": "prediction", "evidence_refs": []},
                {"node_id": "rule:R1", "node_type": "rule", "evidence_refs": ["source:R1"]},
                {"node_id": "claim:C1", "node_type": "claim", "evidence_refs": []},
                {"node_id": "action", "node_type": "action", "payload": {"action": "accept"}, "evidence_refs": []},
            ],
            "edges": [
                {"source": "dataset:root", "target": "data:synthetic-1", "relation": "derived_from"},
                {"source": "model_artifact:root", "target": "prediction", "relation": "produces"},
                {"source": "data:synthetic-1", "target": "prediction", "relation": "derived_from"},
                {"source": "prediction", "target": "rule:R1", "relation": "derived_from"},
                {"source": "rule:R1", "target": "claim:C1", "relation": "explains"},
                {"source": "claim:C1", "target": "action", "relation": "derived_from"},
            ],
            "missing_evidence": []
        },
        "claims": [{"claim_id": "C1", "claim_type": "rule", "evidence_status": "supported",
                    "statement": "Rule R1 activated on a synthetic control input",
                    "evidence_refs": ["rule:R1"]}],
        "layers": {"rules": [{"rule_id": "R1", "source_objects": ["synthetic-1"], "native": True}],
                   "fuzzy_rule_activations": [{"object_id": "synthetic-1", "rule_id": "R1", "activation_strength": 0.8,
                      "terms": [{"feature": "risk", "term": "high", "membership_degree": 0.8, "feature_value": 0.9},
                                {"feature": "context", "term": "valid", "membership_degree": 0.9, "feature_value": 0.1}]}]},
        "fuzzy": {
            "selected_representation": "F_int",
            "memberships": {"synthetic-1": {"high": 0.8}},
            "membership_policies": {"risk": {"origin": "expert", "version": "v1",
              "terms": [{"label": "high", "function": "triangular", "parameters": [0.5, 1.0, 1.0]}]},
              "context": {"origin": "expert", "version": "v1",
              "terms": [{"label": "valid", "function": "triangular", "parameters": [0.0, 0.0, 1.0]}]}}
        },
        "disagreement": {"alignment_policy": {"applicable": True}, "alignment_status": "measured",
                          "components": {"rule": 0.10}, "beta": {"rule": 1.0}, "gamma": 0.10, "gamma_max": 0.30,
                          "reduction_policy": {"applicable": True}, "reduction_status": "measured", "delta": 0.05,
                          "pre_interpretability": 0.90},
        "risk": {"components": comps, "risk_weights": weights, "rho": rho, "chi_r_crit": 0,
                 "status": "complete", "action": "accept",
                 "risk_thresholds": {"theta_1": 0.20, "theta_2": 0.40, "theta_3": 0.60, "theta_4": 0.80}},
        "route": [{"node_id": "representation_selection", "operator_type": "representation_selection",
                   "value_source": "computed", "input_refs": ["rule:R1"],
                   "output_values": {"selected_class": "F_int"}}],
        "diagnostics": [], "explanation_level": {"available_channels": ["native_rule"]}
    }


def run() -> None:
    ROOT.joinpath("view_model.json").write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ROOT.joinpath("taxonomy.json").write_text(json.dumps({"method_id": "synthetic_rule_control",
        "source": "intrinsic", "object": "logic", "audience": "auditor",
        "canonical_configuration": "single transparent rule on artificial input"}, indent=2), encoding="utf-8")
    ROOT.joinpath("representation.json").write_text(json.dumps({"profile": ["u_num", "u_ling", "u_int"],
        "selected_class": "F_int", "measured_delta": 0.05}, indent=2), encoding="utf-8")
    ROOT.joinpath("policy.json").write_text(json.dumps({"allow_controlled_fixture": True}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
