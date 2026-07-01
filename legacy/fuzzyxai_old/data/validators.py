from __future__ import annotations

from pathlib import Path
from typing import Any

from fuzzyxai.audit.common import ROOT
from fuzzyxai.data.loaders import load_csv_rows, load_json


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def raw_file(entry: dict[str, Any], filename: str) -> Path:
    return ROOT / entry["raw_path"] / filename


def validate_common(scenario_id: str, entry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in [
        "dataset_id",
        "scenario_id",
        "artifact_type",
        "status",
        "source",
        "license",
        "raw_path",
        "processed_path",
        "required_files",
        "required_fields",
        "evidence_level",
        "not_a_claim",
    ]:
        require(field in entry and entry[field] not in (None, ""), f"{scenario_id}: missing {field}", errors)
    require(entry.get("scenario_id") == scenario_id, f"{scenario_id}: scenario_id mismatch", errors)
    require(entry.get("status") in {"real_public_artifact", "control_demo_artifact", "synthetic_control_artifact", "extension_area"}, f"{scenario_id}: bad status", errors)
    for file_name in entry.get("required_files", []):
        path = raw_file(entry, file_name)
        require(path.exists(), f"{scenario_id}: missing raw file {path}", errors)
        if path.exists():
            require(path.stat().st_size > 0, f"{scenario_id}: empty raw file {path}", errors)
    return errors


def validate_hybrid(entry: dict[str, Any]) -> list[str]:
    errors = validate_common("hybrid_xiris", entry)
    data = load_json(raw_file(entry, "input_hybrid_xiris_eye_sample.json"))
    require(data.get("image_quality") == 0.31, "hybrid_xiris: image_quality != 0.31", errors)
    require(data.get("segmentation_quality") == 0.27, "hybrid_xiris: segmentation_quality != 0.27", errors)
    require(data.get("model_match_signal") == 0.88, "hybrid_xiris: model_match_signal != 0.88", errors)
    require(data.get("alpha_accept") == 0.82, "hybrid_xiris: alpha_accept != 0.82", errors)
    require(data.get("alpha_block") == 0.91, "hybrid_xiris: alpha_block != 0.91", errors)
    require(entry.get("control_demo_artifact") is True, "hybrid_xiris: control_demo_artifact not true", errors)
    require(entry.get("not_industrial_biometric_validation") is True, "hybrid_xiris: industrial validation disclaimer missing", errors)
    return errors


def validate_ecg(entry: dict[str, Any]) -> list[str]:
    errors = validate_common("medical_ecg_signal", entry)
    rows = load_csv_rows(raw_file(entry, "input_ecg_sample_signal.csv"))
    require(bool(rows), "medical_ecg_signal: empty ECG rows", errors)
    require(rows[0].get("sampling_rate") == "250", "medical_ecg_signal: sampling_rate != 250", errors)
    require(rows[0].get("quality_score") == "0.58", "medical_ecg_signal: quality_score != 0.58", errors)
    require(sum(1 for r in rows if r.get("missing_mask") == "1") == 2, "medical_ecg_signal: missing_fragments != 2", errors)
    require(entry.get("not_clinical_diagnosis") is True, "medical_ecg_signal: clinical disclaimer missing", errors)
    return errors


def validate_gd(entry: dict[str, Any]) -> list[str]:
    errors = validate_common("gd_anfis_shap", entry)
    rows = load_csv_rows(raw_file(entry, "gd_anfis_shap_sample.csv"))
    by_feature = {r["feature"]: r for r in rows}
    require(by_feature.get("alpha_rule", {}).get("value") == "0.82", "gd_anfis_shap: alpha_rule != 0.82", errors)
    require(by_feature.get("X1", {}).get("shap") == "0.45", "gd_anfis_shap: shap_x1 != +0.45", errors)
    require(by_feature.get("X2", {}).get("shap") == "-0.30", "gd_anfis_shap: shap_x2 != -0.30", errors)
    return errors


def validate_beacon(entry: dict[str, Any]) -> list[str]:
    errors = validate_common("beacon_xai", entry)
    rows = load_csv_rows(raw_file(entry, "beacon_timeseries_sample.csv"))
    support = sum(1 for r in rows if r.get("support") == "1")
    counter = sum(1 for r in rows if r.get("counter") == "1")
    require(len(rows) == 100, "beacon_xai: single_object_fragments != 100", errors)
    require(support == 70, "beacon_xai: support_fragments != 70", errors)
    require(counter == 30, "beacon_xai: counter_fragments != 30", errors)
    require(entry.get("external_beacon_mechanism") is True, "beacon_xai: BEACON external mechanism flag missing", errors)
    return errors


def validate_gis(entry: dict[str, Any]) -> list[str]:
    errors = validate_common("gis_integro", entry)
    data = load_json(raw_file(entry, "gis_layer_sample.json"))
    p, alpha_mean, s = data.get("p"), data.get("alpha_mean"), data.get("s")
    gamma_route = max(abs(p - alpha_mean), abs(p - s))
    require(round(gamma_route, 2) == 0.20, "gis_integro: gamma_route != 0.20", errors)
    require(data.get("formula") == "max(|p - alpha_mean|, |p - s|)", "gis_integro: wrong gamma_route formula", errors)
    require("alpha_mean-s" not in data.get("formula", ""), "gis_integro: forbidden pairwise formula", errors)
    return errors

