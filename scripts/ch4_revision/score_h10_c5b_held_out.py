#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from fuzzyxai.experiments.h10_c5b import run

from scripts.ch4_revision.h10_c5b_runtime_ops import (
    METHOD_COMMIT,
    sha256_path,
    verify_method_lock,
)

AUTHORIZATION = "AUTHORIZE_ONE_TIME_H10_C5B_HELD_OUT_SCORING"
FINAL_STATUSES = frozenset({"H10_C5B_SUPPORTED", "H10_C5B_NOT_SUPPORTED"})


def _parse_authorized_at(value: object) -> str:
    text = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError("held-out scoring approval time is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("held-out scoring approval time must include UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("held-out scoring approval time must be UTC")
    return text


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _claim_opening(status_path: Path, value: dict[str, object]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        status_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        status_path.unlink(missing_ok=True)
        raise


def score_once(
    *,
    lock_path: Path,
    approval_path: Path,
    status_path: Path,
    opening_record_path: Path,
    output_root: Path,
    repository_root: Path,
    runner: Callable[[Path, Path], dict[str, object]] = run,
) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if lock.get("status") != "HELD_OUT_SCORING_LOCKED":
        raise ValueError("held-out scoring lock is invalid")
    if lock.get("opening_count") != 0 or lock.get("held_out_scored") is not False:
        raise ValueError("held-out scoring lock is not pre-open")
    if lock.get("method_commit") != METHOD_COMMIT:
        raise ValueError("held-out scoring method commit mismatch")
    controller_path = Path(__file__).resolve()
    if sha256_path(controller_path) != lock.get("controller_sha256"):
        raise ValueError("held-out scoring controller SHA256 mismatch")

    lock_sha256 = sha256_path(lock_path)
    expected_approval = {
        "protocol_id": lock["protocol_id"],
        "method_commit": METHOD_COMMIT,
        "held_out_scoring_lock_sha256": lock_sha256,
        "enriched_manifest_sha256": lock["enriched_manifest_sha256"],
        "authorization": AUTHORIZATION,
    }
    for key, expected in expected_approval.items():
        if approval.get(key) != expected:
            raise ValueError(f"held-out scoring approval mismatch: {key}")
    if not str(approval.get("authorized_by", "")).strip():
        raise ValueError("held-out scoring approval has no owner")
    authorized_at = _parse_authorized_at(approval.get("authorized_at_utc"))
    if status_path.exists() or opening_record_path.exists():
        raise RuntimeError("official held-out scoring was already opened")

    method = verify_method_lock(
        repository_root / "protocol/h10_c5b_repository_grounded/METHOD_LOCK.json",
        repository_root,
    )
    if method["method_code_sha256"] != lock.get("method_code_sha256"):
        raise ValueError("held-out scoring method SHA256 mismatch")
    manifest_path = Path(str(lock["enriched_manifest_path"]))
    runtime_report_path = Path(str(lock["runtime_evidence_report_path"]))
    development_lock_path = Path(str(lock["development_runtime_lock_path"]))
    for path, expected in (
        (manifest_path, lock["enriched_manifest_sha256"]),
        (runtime_report_path, lock["runtime_evidence_report_sha256"]),
        (development_lock_path, lock["development_runtime_lock_sha256"]),
    ):
        if not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"held-out scoring input changed: {path.name}")

    opened_at = datetime.now(UTC).isoformat()
    opening = {
        "status": "HELD_OUT_SCORING_OPENED",
        "protocol_id": lock["protocol_id"],
        "method_commit": METHOD_COMMIT,
        "held_out_scoring_lock_sha256": lock_sha256,
        "approval_sha256": sha256_path(approval_path),
        "authorized_at_utc": authorized_at,
        "opened_at_utc": opened_at,
        "opening_count": 1,
        "held_out_scored": False,
    }
    _claim_opening(status_path, opening)
    _atomic_json(opening_record_path, opening)

    try:
        result = runner(manifest_path, output_root)
        scientific_status = str(result.get("status"))
        if scientific_status not in FINAL_STATUSES:
            raise RuntimeError(
                f"frozen classifier returned invalid status: {scientific_status}"
            )
        completed = {
            **opening,
            "status": "HELD_OUT_SCORING_COMPLETED",
            "held_out_scored": True,
            "scientific_status": scientific_status,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "final_status_sha256": sha256_path(
                output_root / "results/h10_c5b/H10_C5B_FINAL_STATUS.json"
            ),
        }
        _atomic_json(status_path, completed)
        return completed
    except BaseException as error:
        failed = {
            **opening,
            "status": "HELD_OUT_SCORING_FAILED_NO_REUSE",
            "failure_type": type(error).__name__,
            "failed_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(status_path, failed)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--approval", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--opening-record", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repository-root", default=Path("."), type=Path)
    args = parser.parse_args()
    result = score_once(
        lock_path=args.lock.resolve(),
        approval_path=args.approval.resolve(),
        status_path=args.status.resolve(),
        opening_record_path=args.opening_record.resolve(),
        output_root=args.output_root.resolve(),
        repository_root=args.repository_root.resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
