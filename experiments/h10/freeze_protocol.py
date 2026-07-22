from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

from .audit_methodology import audit as audit_methodology
from .common import ARTIFACT_ROOT, PRIVATE_ROOT, ROOT, environment_manifest, git_commit, load_yaml, sha256_file, write_json


CODE_PATHS = (
    "framework/fuzzyxai/fuzzyxai/audit_h10",
    "baselines/h10",
    "experiments/h10",
    "config/h10_v19_protocol.yaml",
    "tests/h10",
)


def _tree_hash() -> str:
    paths: list[Path] = []
    for item in CODE_PATHS:
        path = ROOT / item
        paths.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def freeze(config_path: Path) -> None:
    config_path = config_path.resolve()
    methodology = audit_methodology()
    if methodology["status"] != "PASS":
        raise RuntimeError(f"methodology audit must pass before lock: {methodology}")
    lock_path = ARTIFACT_ROOT / "lock" / "protocol_lock.json"
    if lock_path.exists():
        raise RuntimeError("H10 protocol is already locked")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    allowed = ("?? artifacts/h10_v19/",)
    if any(line and not line.startswith(allowed) for line in status.splitlines()):
        raise RuntimeError("tracked implementation changes must be committed before H10 lock")
    config = load_yaml(config_path)
    vault_path = PRIVATE_ROOT / "h10_v19_label_vault.enc"
    required = (
        ARTIFACT_ROOT / "data" / "dataset_manifest.json",
        ARTIFACT_ROOT / "data" / "split_identity_hashes.json",
        ARTIFACT_ROOT / "routes" / "sealed_routes.jsonl",
        ARTIFACT_ROOT / "routes" / "clean_routes.jsonl",
        ARTIFACT_ROOT / "opening" / "pre_opening_leakage_audit.json",
        vault_path,
    )
    if any(not path.exists() for path in required):
        raise RuntimeError("prepare H10 data before protocol lock")
    lock = {
        "study_id": config["study_id"],
        "implementation_commit": git_commit(),
        "protocol_path": str(config_path.relative_to(ROOT)),
        "protocol_sha256": sha256_file(config_path),
        "code_tree_sha256": _tree_hash(),
        "dataset_manifest_sha256": sha256_file(required[0]),
        "split_identity_hashes_sha256": sha256_file(required[1]),
        "sealed_routes_sha256": sha256_file(required[2]),
        "clean_routes_sha256": sha256_file(required[3]),
        "pre_opening_audit_sha256": sha256_file(required[4]),
        "vault_sha256": sha256_file(vault_path),
        "methodology_audit_sha256": sha256_file(ARTIFACT_ROOT / "closure" / "confirmatory_methodology_audit.json"),
        "thresholds": config["thresholds"],
        "primary_metrics": config["primary_metrics"],
        "safety_constraints": config["safety_constraints"],
        "margins": config["practically_relevant_margins"],
        "best_baseline": config["best_baseline_selected_on_exploratory"],
        "bootstrap": config["bootstrap"],
        "environment": environment_manifest(),
        "status": "LOCKED_UNOPENED",
    }
    write_json(lock_path, lock)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "h10_v19_protocol.yaml")
    args = parser.parse_args()
    freeze(args.config)


if __name__ == "__main__":
    main()
