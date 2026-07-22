from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from baselines.h10 import (
    AnomalyDetectorBaseline,
    HashVersionBaseline,
    IndependentRulesBaseline,
    SchemaOnlyBaseline,
    SimpleOrBaseline,
    TypedRouteBaseline,
    UntypedGraphBaseline,
)
from fuzzyxai.audit_h10 import H10Auditor

from .audit_methodology import audit as audit_methodology
from .common import ARTIFACT_ROOT, PRIVATE_ROOT, ROOT, git_commit, load_yaml, read_json, sha256_file, write_csv, write_json
from .freeze_protocol import _tree_hash
from .metrics import evaluate_method
from .serialization import route_from_dict, truth_from_dict
from .vault import open_vault


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_auditor(config: dict) -> H10Auditor:
    rows = _jsonl(ARTIFACT_ROOT / "routes" / "train_development_routes.jsonl")
    held_out = set(config["held_out_leaves"])
    samples = []
    for row in rows:
        truth = truth_from_dict(row["truth"])
        if len(truth.leaf_types) == 1 and truth.leaf_types[0] not in held_out and truth.route_status == "invalid":
            samples.append((route_from_dict(row["route"]), truth.leaf_types[0]))
    thresholds = config["thresholds"]
    return H10Auditor.create(
        threshold_known=float(thresholds["known"]),
        threshold_anomaly=float(thresholds["anomaly"]),
        leaf_threshold=float(thresholds["leaf"]),
    ).fit(samples)


def _assert_scoring_not_previously_opened(*markers: Path) -> None:
    existing = [str(path) for path in markers if path.exists()]
    if existing:
        raise RuntimeError(f"H10 confirmatory vault has already been opened or invalidated: {existing}")


def run(config_path: Path) -> None:
    methodology = audit_methodology()
    if methodology["status"] != "PASS":
        raise RuntimeError(f"H10 v19 methodology audit failed before opening: {methodology}")
    lock_path = ARTIFACT_ROOT / "lock" / "protocol_lock.json"
    opening_path = ARTIFACT_ROOT / "opening" / "opening_record.json"
    invalid_path = ARTIFACT_ROOT / "opening" / "invalid_marker.json"
    methodology_invalid_path = ARTIFACT_ROOT / "opening" / "confirmatory_invalid_marker.json"
    completion_path = ARTIFACT_ROOT / "opening" / "completion_marker.json"
    _assert_scoring_not_previously_opened(opening_path, completion_path, invalid_path, methodology_invalid_path)
    lock = read_json(lock_path)
    config = load_yaml(config_path)
    checks = {
        "protocol_sha256": sha256_file(config_path),
        "code_tree_sha256": _tree_hash(),
        "dataset_manifest_sha256": sha256_file(ARTIFACT_ROOT / "data" / "dataset_manifest.json"),
        "split_identity_hashes_sha256": sha256_file(ARTIFACT_ROOT / "data" / "split_identity_hashes.json"),
        "sealed_routes_sha256": sha256_file(ARTIFACT_ROOT / "routes" / "sealed_routes.jsonl"),
        "clean_routes_sha256": sha256_file(ARTIFACT_ROOT / "routes" / "clean_routes.jsonl"),
        "methodology_audit_sha256": sha256_file(ARTIFACT_ROOT / "closure" / "confirmatory_methodology_audit.json"),
        "vault_sha256": sha256_file(PRIVATE_ROOT / "h10_v19_label_vault.enc"),
    }
    mismatches = {key: (lock[key], value) for key, value in checks.items() if lock.get(key) != value}
    if mismatches:
        raise RuntimeError(f"H10 lock integrity failure: {mismatches}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip():
        raise RuntimeError("tracked worktree must be clean before H10 scoring")
    write_json(
        opening_path,
        {
            "study_id": config["study_id"],
            "opening_count": 1,
            "purpose": "scoring_only",
            "commit": git_commit(),
            "protocol_sha256": checks["protocol_sha256"],
            "post_lock_tuning": False,
            "methodology_audit_status": methodology["status"],
        },
    )
    try:
        key = (PRIVATE_ROOT / "h10_v19_vault.key").read_bytes()
        truths_payload = open_vault((PRIVATE_ROOT / "h10_v19_label_vault.enc").read_bytes(), key)
        truths = [truth_from_dict(item) for item in json.loads(truths_payload)]
        routes = [route_from_dict(item) for item in _jsonl(ARTIFACT_ROOT / "routes" / "sealed_routes.jsonl")]
        truth_by_id = {truth.case_id: truth for truth in truths}
        cases = [(route, truth_by_id[route.route_id]) for route in routes]
        auditor = _load_auditor(config)
        methods = {
            "schema_only": SchemaOnlyBaseline().diagnose,
            "hash_version": HashVersionBaseline().diagnose,
            "simple_or": SimpleOrBaseline().diagnose,
            "independent_if_else": IndependentRulesBaseline().diagnose,
            "untyped_graph": UntypedGraphBaseline().diagnose,
            "anomaly_detector": AnomalyDetectorBaseline().diagnose,
            "typed_route": TypedRouteBaseline().diagnose,
            "full_h10": auditor.diagnose,
        }
        rows = []
        for name, method in methods.items():
            rows.extend(evaluate_method(name, method, cases))
        trace_matches = [auditor.diagnose(route).trace == auditor.diagnose(route).trace for route in routes]
        write_csv(ARTIFACT_ROOT / "confirmatory" / "raw_results.csv", rows)
        write_json(
            ARTIFACT_ROOT / "confirmatory" / "run_summary.json",
            {
                "study_id": config["study_id"],
                "datasets": sorted({route.dataset_id for route in routes}),
                "cases": len(cases),
                "methods": list(methods),
                "trace_repetitions": len(trace_matches),
                "byte_identical_trace_rate": sum(trace_matches) / len(trace_matches),
                "sealed_scoring_openings": 1,
                "post_lock_tuning": False,
            },
        )
        post = {
            "status": "PASS",
            "opening_count": 1,
            "purpose": "scoring_only",
            "labels_exported": False,
            "model_changed": False,
            "thresholds_changed": False,
            "baseline_changed": False,
            "post_lock_tuning": False,
            "methodology_audit_status": methodology["status"],
            "repeat_scoring_forbidden": True,
            "raw_results_sha256": sha256_file(ARTIFACT_ROOT / "confirmatory" / "raw_results.csv"),
        }
        write_json(ARTIFACT_ROOT / "opening" / "post_scoring_leakage_audit.json", post)
        write_json(completion_path, {"status": "COMPLETE", "opening_count": 1, "commit": git_commit()})
    except Exception as error:
        write_json(invalid_path, {"status": "INVALID_AFTER_OPENING", "error_type": type(error).__name__, "message": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_v19_protocol.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
