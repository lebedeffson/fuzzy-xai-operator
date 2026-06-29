from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fuzzyxai.audit.common import ROOT


SCENARIOS: dict[str, dict[str, Any]] = {
    "hybrid_xiris": {
        "dataset_id": "hybrid_xiris_control",
        "artifact_type": "iris_image",
        "status": "control_demo_artifact",
        "evidence_level": "full_control_run",
        "model_id": "iris_quality_model",
        "model_status": "demo_control_model",
        "not_a_claim": "Not an industrial biometric validation.",
        "input_values": {
            "image_quality": 0.31,
            "segmentation_quality": 0.27,
            "model_match_signal": 0.88,
            "alpha_accept": 0.82,
            "alpha_block": 0.91,
        },
        "expected": {
            "gamma": 0.351,
            "delta": 0.106811,
            "r_delta": 0.3225,
            "rho": 0.800,
            "action": "block",
            "diagnostic": "D_quality_source_conflict",
        },
    },
    "medical_ecg_signal": {
        "dataset_id": "ecg_control_signal",
        "artifact_type": "ecg_signal",
        "status": "control_demo_artifact",
        "evidence_level": "operator_control_example",
        "model_id": "ecg_quality_model",
        "model_status": "demo_control_model",
        "not_a_claim": "Not a clinical diagnostic validation.",
        "input_values": {
            "sampling_rate": 250,
            "quality_score": 0.58,
            "noise_segments": 1,
            "missing_fragments": 2,
            "baseline_instability": 0.21,
        },
        "expected": {
            "action": "defer_to_human",
            "diagnostic": "D_signal_quality",
        },
    },
    "gd_anfis_shap": {
        "dataset_id": "gd_anfis_shap_control",
        "artifact_type": "tabular_rules",
        "status": "control_demo_artifact",
        "evidence_level": "operator_control_example",
        "model_id": "gd_anfis_shap_rule_model",
        "model_status": "demo_control_model",
        "not_a_claim": "Operator-control example, not a production GD model.",
        "input_values": {
            "x1": 0.88,
            "x2": 0.22,
            "x1_term": "high",
            "x2_term": "low",
            "alpha_rule": 0.82,
            "shap_x1": 0.45,
            "shap_x2": -0.30,
        },
        "expected": {
            "gamma_rule_shap": 0.685,
            "diagnostic": "D_rule_attribution_conflict",
            "action": "audit",
        },
    },
    "beacon_xai": {
        "dataset_id": "beacon_counterevidence_control",
        "artifact_type": "time_series",
        "status": "control_demo_artifact",
        "evidence_level": "route_demonstration",
        "model_id": "beacon_counterevidence_model",
        "model_status": "demo_control_model",
        "not_a_claim": "BEACON check reduction is an external mechanism demonstration.",
        "input_values": {
            "single_object_fragments": 100,
            "support_fragments": 70,
            "counter_fragments": 30,
            "batch_objects": 100,
            "objects_with_counterevidence": 83,
            "audit_reports": 12,
            "checks_full": 64,
            "checks_reduced": 11,
        },
        "expected": {
            "diagnostic": "D_counterevidence_conflict",
            "action": "audit",
        },
    },
    "gis_integro": {
        "dataset_id": "gis_integro_control",
        "artifact_type": "geo_layer",
        "status": "control_demo_artifact",
        "evidence_level": "operator_control_example",
        "model_id": "gis_route_model",
        "model_status": "demo_control_model",
        "not_a_claim": "Control geolayer, not a production GIS validation.",
        "input_values": {
            "p": 0.67,
            "alpha_mean": 0.72,
            "s": 0.47,
        },
        "expected": {
            "gamma_route": 0.20,
            "delta": 0.08,
            "formula": "max(|p - alpha_mean|, |p - s|)",
            "action": "audit_report",
        },
    },
}


RAW_FILES = {
    "hybrid_xiris": ROOT / "data/raw/iris/input_hybrid_xiris_eye_sample.json",
    "medical_ecg_signal": ROOT / "data/raw/ecg/input_ecg_sample_signal.csv",
    "gd_anfis_shap": ROOT / "data/raw/tabular/gd_anfis_shap_sample.csv",
    "beacon_xai": ROOT / "data/raw/timeseries/beacon_timeseries_sample.csv",
    "gis_integro": ROOT / "data/raw/gis/gis_layer_sample.json",
}


def stable_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(data: Any) -> str:
    return hashlib.sha256(stable_json(data).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

