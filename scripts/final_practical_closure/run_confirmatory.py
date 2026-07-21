#!/usr/bin/env python3
"""Import and validate sealed confirmatory measurements after protocol lock."""

from __future__ import annotations

from common import CONFIRMATORY, LOCK, ROOT, STUDY, load_json, sha256, write_json


REQUIRED_EXPERIMENTS = {
    "H3-P1",
    "H3-P2",
    "H3-P3",
    "H3-P4",
    "H5-A",
    "H6-A",
    "H6-B",
    "H7-A",
    "H7-B",
    "H8",
    "H9",
}


def main() -> None:
    if not LOCK.is_file():
        raise SystemExit("BLOCKED: confirmatory protocol is not locked")
    lock = load_json(LOCK)
    if lock.get("status") != "locked":
        raise SystemExit("FAIL: invalid confirmatory protocol lock")
    result_manifest_path = STUDY / "confirmatory_result_manifest.json"
    if not result_manifest_path.is_file():
        raise SystemExit("BLOCKED: sealed confirmatory result manifest is not available")
    manifest = load_json(result_manifest_path)
    if manifest.get("protocol_lock_sha256") != sha256(LOCK):
        raise SystemExit("FAIL: confirmatory results do not match the frozen protocol lock")
    if manifest.get("tuning_after_test_open") is not False:
        raise SystemExit("FAIL: confirmatory manifest does not prohibit post-open tuning")
    records = manifest.get("experiments", [])
    seen: set[str] = set()
    imported: list[dict[str, object]] = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            raise SystemExit("FAIL: malformed confirmatory experiment entry")
        experiment_id = str(record.get("experiment_id", ""))
        source = ROOT / str(record.get("artifact_path", ""))
        if experiment_id in seen or experiment_id not in REQUIRED_EXPERIMENTS:
            raise SystemExit(f"FAIL: unexpected or duplicate experiment {experiment_id!r}")
        if not source.is_file() or record.get("sha256") != sha256(source):
            raise SystemExit(f"FAIL: invalid confirmatory artifact for {experiment_id}")
        payload = load_json(source)
        _validate_measurement(experiment_id, payload)
        seen.add(experiment_id)
        imported.append(
            {
                "experiment_id": experiment_id,
                "artifact_path": source.relative_to(ROOT).as_posix(),
                "sha256": sha256(source),
                "status": _status(experiment_id, payload),
                "measurement": payload,
            }
        )
    missing = REQUIRED_EXPERIMENTS - seen
    if missing:
        raise SystemExit(f"BLOCKED: missing confirmatory experiments {sorted(missing)}")
    output = {
        "schema_version": "1.0",
        "phase": "sealed_confirmatory",
        "confirmatory_run_completed": True,
        "post_open_tuning": False,
        "protocol_lock_sha256": sha256(LOCK),
        "result_manifest_sha256": sha256(result_manifest_path),
        "experiments": imported,
    }
    write_json(CONFIRMATORY / "summary.json", output)
    write_json(STUDY / "confirmatory_opening_record.json", {"opened": True, "lock_sha256": sha256(LOCK), "result_manifest_sha256": sha256(result_manifest_path)})
    print(f"PASS: practical_confirmatory_import experiments={len(imported)} post_open_tuning=false")


def _validate_measurement(experiment_id: str, payload: dict[str, object]) -> None:
    if payload.get("phase") != "sealed_confirmatory" or payload.get("post_open_tuning") is not False:
        raise SystemExit(f"FAIL: {experiment_id} is not a sealed confirmatory measurement")
    if payload.get("experiment_id") != experiment_id:
        raise SystemExit(f"FAIL: experiment identity mismatch for {experiment_id}")
    for field in ("effect_size", "confidence_interval_95", "adjusted_p", "n", "unit_of_analysis"):
        if field not in payload:
            raise SystemExit(f"FAIL: {experiment_id} lacks {field}")
    interval = payload["confidence_interval_95"]
    if not isinstance(interval, list) or len(interval) != 2 or float(interval[0]) > float(interval[1]):
        raise SystemExit(f"FAIL: invalid confidence interval for {experiment_id}")
    if int(payload["n"]) < 1 or not 0.0 <= float(payload["adjusted_p"]) <= 1.0:
        raise SystemExit(f"FAIL: invalid statistical record for {experiment_id}")
    if not payload.get("dataset_ids"):
        raise SystemExit(f"FAIL: {experiment_id} lacks dataset identities")


def _status(experiment_id: str, payload: dict[str, object]) -> str:
    effect = float(payload["effect_size"])
    lower = float(payload["confidence_interval_95"][0])
    adjusted_p = float(payload["adjusted_p"])
    significant = lower > 0 and adjusted_p < 0.05
    if experiment_id == "H3-P1":
        return "supported" if significant and effect >= 0.15 and float(payload.get("false_block_rate", 1.0)) <= 0.01 else "not_supported"
    if experiment_id == "H3-P2":
        return "supported" if significant and effect >= 0.05 else "not_supported"
    if experiment_id in {"H3-P3", "H3-P4", "H7-B"}:
        return "supported" if significant and payload.get("frozen_primary_endpoint_met") is True else "not_supported"
    if experiment_id == "H5-A":
        return (
            "supported"
            if significant
            and payload.get("frozen_primary_endpoint_met") is True
            and float(payload.get("fault_detection_f1", 0.0)) >= 0.95
            and float(payload.get("false_certification", 1.0)) <= 0.01
            and float(payload.get("source_localization", 0.0)) >= 0.90
            and len(payload.get("natural_failure_types", [])) >= 1
            else "not_supported"
        )
    if experiment_id == "H6-A":
        return "supported" if significant and payload.get("detectability_region_defined") is True else "not_supported"
    if experiment_id == "H6-B":
        return (
            "supported"
            if significant
            and payload.get("frozen_primary_endpoint_met") is True
            and len(set(payload.get("dataset_ids", []))) >= 2
            else "not_supported"
        )
    if experiment_id == "H7-A":
        return "supported" if float(payload.get("exact_source_hash_rate", 0.0)) == 1.0 else "not_supported"
    if experiment_id == "H8":
        return "supported" if float(payload.get("action_agreement", 0.0)) >= 0.95 and float(payload.get("representation_agreement", 0.0)) >= 0.90 else "not_supported"
    if experiment_id == "H9":
        return "supported" if int(payload.get("maximum_objects", 0)) >= 2_000_000 and float(payload.get("scaling_exponent", 99.0)) <= 1.10 else "not_supported"
    return "not_supported"


if __name__ == "__main__":
    main()
