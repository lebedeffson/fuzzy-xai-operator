from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.ch4_revision.collect_h10_c5b_runtime import (
    _apply_runtime_test_patch,
    _execution_command,
    _has_trace,
    collect,
)


def test_pytest_assertion_failure_is_runtime_trace() -> None:
    output = """
=================================== FAILURES ===================================
_______________________________ test_contract ________________________________
>       assert actual == expected
E       AssertionError: assert 1 == 2
tests/test_contract.py:42: AssertionError
=========================== short test summary info ============================
FAILED tests/test_contract.py::test_contract - AssertionError: assert 1 == 2
"""
    assert _has_trace(output)
    assert not _has_trace("tests/test_contract.py:42: warning")


def test_runtime_collection_requires_reproduced_trace(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "failure.py").write_text(
        "raise ValueError('registered failure')\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "repository": "fixture/repo",
                "repository_root": str(repository),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands = tmp_path / "commands.json"
    commands.write_text(
        json.dumps(
            {
                "incident-1": {
                    "command": [sys.executable, "failure.py"],
                    "timeout_seconds": 30,
                }
            }
        ),
        encoding="utf-8",
    )
    report = collect(manifest, commands, tmp_path / "evidence")
    assert report["status"] == "PASS"
    assert report["trace_complete_count"] == 1
    enriched = json.loads(
        (tmp_path / "evidence/H10_C5B_RUNTIME_ENRICHED_MANIFEST.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert enriched["runtime_evidence_status"] == "BUG_REPRODUCED_WITH_TRACE"
    assert report["child_environment_policy"].endswith("PYTHONNOUSERSITE_ONLY")
    assert len(report["command_registry_sha256"]) == 64
    assert len(report["evidence"][0]["runtime_command_sha256"]) == 64


def test_unregistered_runtime_command_is_incomplete(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "repository": "fixture/repo",
                "repository_root": str(tmp_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands = tmp_path / "commands.json"
    commands.write_text("{}", encoding="utf-8")
    report = collect(manifest, commands, tmp_path / "evidence")
    assert report["status"] == "INCOMPLETE"
    assert report["trace_complete_count"] == 0


def test_pytest_infrastructure_returncode_is_not_bug_reproduction(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "repository": "fixture/repo",
                "repository_root": str(repository),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands = tmp_path / "commands.json"
    commands.write_text(
        json.dumps(
            {
                "incident-1": {
                    "command": [sys.executable, "-c", "raise SystemExit(4)"],
                    "expected_failure_returncodes": [1],
                    "timeout_seconds": 30,
                }
            }
        ),
        encoding="utf-8",
    )
    report = collect(manifest, commands, tmp_path / "evidence")
    assert report["evidence"][0]["status"] == "RUNTIME_INFRASTRUCTURE_ERROR"
    assert report["trace_complete_count"] == 0


def test_runtime_timeout_is_recorded_without_retry(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "incident-1",
                "repository": "fixture/repo",
                "repository_root": str(repository),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    commands = tmp_path / "commands.json"
    commands.write_text(
        json.dumps(
            {
                "incident-1": {
                    "command": [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(1)",
                    ],
                    "expected_failure_returncodes": [1],
                    "timeout_seconds": 0.01,
                }
            }
        ),
        encoding="utf-8",
    )
    report = collect(manifest, commands, tmp_path / "evidence")
    assert report["evidence"][0]["status"] == "RUNTIME_TIMEOUT"
    assert report["trace_complete_count"] == 0


def test_container_backend_requires_digest_pinned_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned"):
        _execution_command(
            {
                "execution_backend": "container",
                "container_image": "python:latest",
            },
            ("python", "-m", "pytest"),
            tmp_path,
            tmp_path / "container.cid",
        )
    command, backend = _execution_command(
        {
            "execution_backend": "container",
            "container_image": "python@sha256:" + "a" * 64,
        },
        ("python", "-m", "pytest"),
        tmp_path,
        tmp_path / "container.cid",
    )
    assert backend == "container"
    assert command[:3] == ("docker", "run", "--rm")
    assert "--network" in command
    assert "none" in command


def test_runtime_test_patch_is_checksum_bound_and_path_safe(tmp_path: Path) -> None:
    sandbox = tmp_path / "repository"
    sandbox.mkdir()
    target = sandbox / "test_sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = tmp_path / "test.patch"
    patch.write_text(
        "diff --git a/test_sample.py b/test_sample.py\n"
        "--- a/test_sample.py\n"
        "+++ b/test_sample.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n",
        encoding="utf-8",
    )
    import hashlib

    registered = {
        "runtime_test_patch_path": str(patch),
        "runtime_test_patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    }
    assert _apply_runtime_test_patch(registered, sandbox) is None
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    registered["runtime_test_patch_sha256"] = "0" * 64
    assert "checksum mismatch" in str(
        _apply_runtime_test_patch(registered, sandbox)
    )
