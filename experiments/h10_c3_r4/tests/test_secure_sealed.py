from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import h10_c3_r4.secure_sealed as secure
from h10_c3_r4.scientific_classifier import classify_confirmatory_result
from scripts.build_h10_c3_r4_handoff import (
    scan_preopen_archive,
    scan_source_archive,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _small_design(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    design = {
        "pipeline_families": 6,
        "case_count": 6,
        "templates_per_family": 1,
        "cases_per_template": 1,
        "stratum_allocation": {"S2": 6, "S3": 0, "S4": 0, "S5": 0},
        "per_pipeline_stratum_allocation": {
            "S2": 1,
            "S3": 0,
            "S4": 0,
            "S5": 0,
        },
    }
    monkeypatch.setattr(secure, "DESIGN", design)
    return design


def _secure_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, bytes, bytes]:
    _small_design(monkeypatch)
    artifact_root = tmp_path / "artifacts"
    secure_root = artifact_root / "secure_sealed"
    secure_root.mkdir(parents=True)
    seed = bytes(range(32))
    protocol_sha = "ab" * 32
    payload = secure._private_payload(seed, REPO_ROOT)
    plaintext = secure._canonical_bytes(payload)
    container = secure._encrypt(plaintext, seed, protocol_sha)
    commitment = {
        "study_id": secure.STUDY_ID,
        "protocol_lock_sha256": protocol_sha,
        "seed_commitment_sha256": secure._sha256_bytes(
            secure.STUDY_ID.encode() + protocol_sha.encode() + seed
        ),
        "encrypted_payload_sha256": secure._sha256_bytes(container),
        "plaintext_commitment_sha256": secure._sha256_bytes(plaintext),
        "opening_count": 0,
    }
    _write_json(
        secure_root / "sealed_bank_commitment.json",
        commitment,
    )
    (secure_root / "sealed_ciphertext.bin").write_bytes(container)
    _write_json(
        secure_root / "sealed_status.json",
        {
            "status": "READY_FOR_SECURE_SEALED_SCORING",
            "opening_count": 0,
        },
    )
    approval = tmp_path / "approval.json"
    _write_json(
        approval,
        {
            "study_id": secure.STUDY_ID,
            "protocol_lock_sha256": protocol_sha,
            "encrypted_payload_sha256": commitment[
                "encrypted_payload_sha256"
            ],
            "plaintext_commitment_sha256": commitment[
                "plaintext_commitment_sha256"
            ],
            "authorization": secure.AUTHORIZATION,
            "authorized_by": "protocol_owner",
            "authorized_at_utc": "2026-07-24T00:00:00+00:00",
        },
    )
    secret_path = tmp_path / "secret.bin"
    secret_path.write_bytes(seed)
    monkeypatch.setattr(
        secure,
        "verify_secure_protocol_lock",
        lambda *args: {},
    )
    return artifact_root, approval, secret_path, seed, plaintext


def test_private_bank_is_secret_derived_and_has_registered_size() -> None:
    left = secure._private_template_bank(b"a" * 32, REPO_ROOT)
    right = secure._private_template_bank(b"b" * 32, REPO_ROOT)
    assert len(left) == 240
    assert len(right) == 240
    assert {item.canonical_hash for item in left}.isdisjoint(
        item.canonical_hash for item in right
    )
    assert {item.canonical_hash for item in left}.isdisjoint(
        secure._public_template_hashes(REPO_ROOT)
    )


def test_seed_commitment_and_plaintext_commitment_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, approval, secret_path, seed, plaintext = _secure_fixture(
        tmp_path,
        monkeypatch,
    )
    cases = secure.load_secure_sealed_cases(
        REPO_ROOT,
        artifact_root,
        approval,
        secret_path,
    )
    commitment = secure._read_json(
        artifact_root
        / "secure_sealed"
        / "sealed_bank_commitment.json"
    )
    assert len(cases) == 6
    assert commitment["plaintext_commitment_sha256"] == hashlib.sha256(
        plaintext
    ).hexdigest()
    assert commitment["seed_commitment_sha256"] == hashlib.sha256(
        secure.STUDY_ID.encode()
        + str(commitment["protocol_lock_sha256"]).encode()
        + seed
    ).hexdigest()


def test_direct_scoring_without_secret_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, approval, _, _, _ = _secure_fixture(
        tmp_path,
        monkeypatch,
    )
    with pytest.raises(PermissionError, match="secret is missing"):
        secure.load_secure_sealed_cases(
            REPO_ROOT,
            artifact_root,
            approval,
            tmp_path / "absent.bin",
        )


def test_wrong_secret_fails_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, approval, secret_path, _, _ = _secure_fixture(
        tmp_path,
        monkeypatch,
    )
    secret_path.write_bytes(b"x" * 32)
    with pytest.raises(PermissionError, match="does not match commitment"):
        secure.load_secure_sealed_cases(
            REPO_ROOT,
            artifact_root,
            approval,
            secret_path,
        )
    status = secure._read_json(
        artifact_root / "secure_sealed" / "sealed_status.json"
    )
    assert status["opening_count"] == 0


def test_changed_encrypted_payload_fails_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, approval, secret_path, _, _ = _secure_fixture(
        tmp_path,
        monkeypatch,
    )
    ciphertext = (
        artifact_root / "secure_sealed" / "sealed_ciphertext.bin"
    )
    ciphertext.write_bytes(ciphertext.read_bytes() + b"tampered")
    with pytest.raises(PermissionError, match="payload changed"):
        secure.load_secure_sealed_cases(
            REPO_ROOT,
            artifact_root,
            approval,
            secret_path,
        )


def test_changed_approval_fails_without_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, approval, secret_path, _, _ = _secure_fixture(
        tmp_path,
        monkeypatch,
    )
    payload = secure._read_json(approval)
    payload["authorization"] = "DO_NOT_AUTHORIZE"
    _write_json(approval, payload)
    with pytest.raises(PermissionError, match="does not match"):
        secure.load_secure_sealed_cases(
            REPO_ROOT,
            artifact_root,
            approval,
            secret_path,
        )


def test_opening_is_recorded_before_decryption_and_failure_is_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root, approval, secret_path, _, _ = _secure_fixture(
        tmp_path,
        monkeypatch,
    )

    def fail_after_check(*args: object) -> bytes:
        status = secure._read_json(
            artifact_root / "secure_sealed" / "sealed_status.json"
        )
        assert status["opening_count"] == 1
        raise RuntimeError("injected decryption failure")

    monkeypatch.setattr(secure, "_decrypt", fail_after_check)
    with pytest.raises(RuntimeError, match="injected"):
        secure.load_secure_sealed_cases(
            REPO_ROOT,
            artifact_root,
            approval,
            secret_path,
        )
    status = secure._read_json(
        artifact_root / "secure_sealed" / "sealed_status.json"
    )
    assert status["opening_count"] == 1
    assert status["status"] == "SEALED_SCORING_FAILED_NO_REUSE"
    with pytest.raises(PermissionError, match="cannot be repeated"):
        secure.load_secure_sealed_cases(
            REPO_ROOT,
            artifact_root,
            approval,
            secret_path,
        )


def test_secure_protocol_lock_detects_method_and_protocol_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    (artifacts / "lock").mkdir(parents=True)
    scientific = repo / "scientific.py"
    operational = repo / "operational.py"
    evidence = repo / "evidence.json"
    scientific.parent.mkdir(parents=True)
    scientific.write_text("locked = True\n", encoding="utf-8")
    operational.write_text("secure = True\n", encoding="utf-8")
    evidence.write_text("{}\n", encoding="utf-8")
    method = artifacts / "lock" / "method_lock.json"
    protocol = artifacts / "lock" / "protocol_lock.json"
    method.write_text("{}\n", encoding="utf-8")
    protocol.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        secure,
        "LEGACY_METHOD_LOCK_SHA256",
        secure._sha256_file(method),
    )
    monkeypatch.setattr(
        secure,
        "LEGACY_PROTOCOL_LOCK_SHA256",
        secure._sha256_file(protocol),
    )
    _write_json(
        artifacts / "lock" / "secure_protocol_lock.json",
        {
            "study_id": secure.STUDY_ID,
            "locked_implementation": secure.LOCKED_IMPLEMENTATION,
            "legacy_method_lock_sha256": secure._sha256_file(method),
            "legacy_protocol_lock_sha256": secure._sha256_file(protocol),
            "protected_scientific_files": {
                "scientific.py": secure._sha256_file(scientific)
            },
            "operational_files": {
                "operational.py": secure._sha256_file(operational)
            },
            "open_evidence_files": {
                "evidence.json": secure._sha256_file(evidence)
            },
        },
    )
    secure.verify_secure_protocol_lock(repo, artifacts)
    protocol.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(PermissionError, match="protocol lock changed"):
        secure.verify_secure_protocol_lock(repo, artifacts)
    protocol.write_text("{}\n", encoding="utf-8")
    method.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(PermissionError, match="method lock changed"):
        secure.verify_secure_protocol_lock(repo, artifacts)
    method.write_text("{}\n", encoding="utf-8")
    scientific.write_text("locked = False\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="scientific.py"):
        secure.verify_secure_protocol_lock(repo, artifacts)
    scientific.write_text("locked = True\n", encoding="utf-8")
    operational.write_text("secure = False\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="operational.py"):
        secure.verify_secure_protocol_lock(repo, artifacts)


def test_secret_file_is_owner_only_and_must_be_external(
    tmp_path: Path,
) -> None:
    secret = tmp_path / "owner" / "seed.bin"
    secure._write_secret(secret, b"s" * 32)
    assert secret.read_bytes() == b"s" * 32
    assert secret.stat().st_mode & 0o077 == 0
    with pytest.raises(PermissionError, match="outside"):
        secure._assert_external_secret_path(
            REPO_ROOT,
            REPO_ROOT / "forbidden-seed.bin",
        )


def test_secure_lock_rejects_plaintext_sealed_files(
    tmp_path: Path,
) -> None:
    leaked = (
        tmp_path
        / "artifacts"
        / "h10_c3_r4"
        / "sealed"
        / "data"
        / "sealed"
        / "cases.jsonl"
    )
    leaked.parent.mkdir(parents=True)
    leaked.write_text("{}\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="plaintext"):
        secure._assert_plaintext_absent(tmp_path)


def test_preopen_archive_scanner_rejects_plaintext_closed_data(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "leak.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            "EVIDENCE/secure_sealed/sealed_design.json",
            "{}",
        )
        stream.writestr(
            "EVIDENCE/secure_sealed/sealed_bank_commitment.json",
            "{}",
        )
        stream.writestr(
            "EVIDENCE/secure_sealed/sealed_ciphertext.bin",
            b"ciphertext",
        )
        stream.writestr(
            "EVIDENCE/secure_sealed/sealed_status.json",
            "{}",
        )
        stream.writestr(
            "EVIDENCE/secure_sealed/cases.jsonl",
            '{"reverse_candidate_ids": []}\n',
        )
    with pytest.raises(RuntimeError, match="plaintext"):
        scan_preopen_archive(archive)


def test_source_archive_scanner_rejects_public_sealed_templates(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr(
            "fuzzy-xai-operator/experiments/h10_c3_r4/"
            "templates/sealed/templates.jsonl",
            "{}\n",
        )
    with pytest.raises(RuntimeError, match="sealed plaintext"):
        scan_source_archive(archive)


def test_final_scientific_status_follows_locked_rules() -> None:
    passing = [
        {
            "claim": "H10-C3a",
            "effect": 0.05,
            "ci_low": 0.01,
            "p_holm": 0.01,
            "positive_pipeline_families": 5,
            "cost_regret_effect": 0.02,
            "cost_regret_ci_low": 0.001,
            "cost_regret_p_holm": 0.02,
        },
        {
            "claim": "H10-C3b",
            "effect": 0.06,
            "ci_low": 0.02,
            "p_holm": 0.01,
            "positive_pipeline_families": 6,
        },
    ]
    result = classify_confirmatory_result(
        passing,
        false_certification=0.0,
        new_critical_violations=0.0,
        runtime_p95_ms=49.0,
    )
    assert result["scientific_status"] == "SCIENTIFIC_PASS"
    passing[0]["effect"] = 0.039
    result = classify_confirmatory_result(
        passing,
        false_certification=0.0,
        new_critical_violations=0.0,
        runtime_p95_ms=49.0,
    )
    assert result["H10-C3a"] == "CONFIRMATORY_FAIL"
    assert result["scientific_status"] == "SCIENTIFIC_FAIL"
