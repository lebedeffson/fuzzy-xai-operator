from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ch4_revision.h10_c5b_runtime_ops import sha256_path
from scripts.ch4_revision.score_h10_c5b_held_out import (
    AUTHORIZATION,
    score_once,
)


def _scoring_fixture(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "held-out.jsonl"
    runtime_report = tmp_path / "runtime-report.json"
    development_lock = tmp_path / "development-lock.json"
    manifest.write_text("{}\n", encoding="utf-8")
    runtime_report.write_text("{}\n", encoding="utf-8")
    development_lock.write_text("{}\n", encoding="utf-8")
    controller = Path(
        "scripts/ch4_revision/score_h10_c5b_held_out.py"
    ).resolve()

    lock = {
        "status": "HELD_OUT_SCORING_LOCKED",
        "protocol_id": "h10-c5b-repository-grounded-v1",
        "method_commit": "7aa72a19a70bdb5eedea520742f269bc6c26aeea",
        "method_code_sha256": "method-sha",
        "controller_sha256": sha256_path(controller),
        "enriched_manifest_path": str(manifest),
        "enriched_manifest_sha256": sha256_path(manifest),
        "runtime_evidence_report_path": str(runtime_report),
        "runtime_evidence_report_sha256": sha256_path(runtime_report),
        "development_runtime_lock_path": str(development_lock),
        "development_runtime_lock_sha256": sha256_path(development_lock),
        "opening_count": 0,
        "held_out_scored": False,
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(
        json.dumps(lock, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    approval = {
        "protocol_id": lock["protocol_id"],
        "method_commit": lock["method_commit"],
        "held_out_scoring_lock_sha256": sha256_path(lock_path),
        "enriched_manifest_sha256": lock["enriched_manifest_sha256"],
        "authorization": AUTHORIZATION,
        "authorized_by": "fixture-owner",
        "authorized_at_utc": "2026-07-26T12:00:00+00:00",
    }
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(approval, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return lock_path, approval_path


def test_official_scoring_records_opening_before_runner_and_blocks_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, approval = _scoring_fixture(tmp_path)
    status = tmp_path / "status.json"
    opening = tmp_path / "opening.json"
    output = tmp_path / "output"
    monkeypatch.setattr(
        "scripts.ch4_revision.score_h10_c5b_held_out.verify_method_lock",
        lambda *_: {"method_code_sha256": "method-sha"},
    )

    def runner(_: Path, output_root: Path) -> dict[str, object]:
        opened = json.loads(status.read_text(encoding="utf-8"))
        assert opened["opening_count"] == 1
        assert opened["held_out_scored"] is False
        final = output_root / "results/h10_c5b/H10_C5B_FINAL_STATUS.json"
        final.parent.mkdir(parents=True)
        final.write_text(
            '{"status":"H10_C5B_SUPPORTED"}\n',
            encoding="utf-8",
        )
        return {"status": "H10_C5B_SUPPORTED"}

    result = score_once(
        lock_path=lock,
        approval_path=approval,
        status_path=status,
        opening_record_path=opening,
        output_root=output,
        repository_root=Path.cwd(),
        runner=runner,
    )
    assert result["scientific_status"] == "H10_C5B_SUPPORTED"
    assert result["opening_count"] == 1
    with pytest.raises(RuntimeError, match="already opened"):
        score_once(
            lock_path=lock,
            approval_path=approval,
            status_path=status,
            opening_record_path=opening,
            output_root=output,
            repository_root=Path.cwd(),
            runner=runner,
        )


def test_failure_after_opening_is_irreversible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock, approval = _scoring_fixture(tmp_path)
    status = tmp_path / "status.json"
    opening = tmp_path / "opening.json"
    monkeypatch.setattr(
        "scripts.ch4_revision.score_h10_c5b_held_out.verify_method_lock",
        lambda *_: {"method_code_sha256": "method-sha"},
    )

    def failed_runner(_: Path, __: Path) -> dict[str, object]:
        raise RuntimeError("registered failure")

    with pytest.raises(RuntimeError, match="registered failure"):
        score_once(
            lock_path=lock,
            approval_path=approval,
            status_path=status,
            opening_record_path=opening,
            output_root=tmp_path / "output",
            repository_root=Path.cwd(),
            runner=failed_runner,
        )
    failed = json.loads(status.read_text(encoding="utf-8"))
    assert failed["status"] == "HELD_OUT_SCORING_FAILED_NO_REUSE"
    assert failed["opening_count"] == 1


def test_changed_controller_or_approval_fails_before_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path, approval_path = _scoring_fixture(tmp_path)
    status = tmp_path / "status.json"
    opening = tmp_path / "opening.json"
    monkeypatch.setattr(
        "scripts.ch4_revision.score_h10_c5b_held_out.verify_method_lock",
        lambda *_: {"method_code_sha256": "method-sha"},
    )

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["controller_sha256"] = "0" * 64
    lock_path.write_text(json.dumps(lock, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="controller SHA256"):
        score_once(
            lock_path=lock_path,
            approval_path=approval_path,
            status_path=status,
            opening_record_path=opening,
            output_root=tmp_path / "output",
            repository_root=Path.cwd(),
        )
    assert not status.exists()
    assert not opening.exists()

    lock_path, approval_path = _scoring_fixture(tmp_path / "approval-case")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["authorized_at_utc"] = "not-a-time"
    approval_path.write_text(
        json.dumps(approval, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="approval time"):
        score_once(
            lock_path=lock_path,
            approval_path=approval_path,
            status_path=status,
            opening_record_path=opening,
            output_root=tmp_path / "output",
            repository_root=Path.cwd(),
        )
    assert not status.exists()
    assert not opening.exists()
