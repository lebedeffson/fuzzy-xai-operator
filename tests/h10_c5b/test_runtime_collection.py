from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ch4_revision.collect_h10_c5b_runtime import (
    _apply_runtime_test_patch,
    _execution_command,
    _has_trace,
    _is_collection_failure,
    _prepare_runtime_image,
    _remove_runtime_image,
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


def test_colored_pytest_failure_is_runtime_trace() -> None:
    output = (
        "\x1b[31mFAILED\x1b[0m tests/test_contract.py::test_contract\n"
        "\x1b[1m\x1b[31mtests/test_contract.py\x1b[0m:42: ValueError\n"
        " FAILURES "
    )
    assert _has_trace(output)


def test_custom_runner_python_frame_and_exception_is_runtime_trace() -> None:
    output = (
        '  File "/workspace/sympy/tests/test_value.py", line 42, in test_value\n'
        "    assert actual == expected\n"
        "AssertionError\n"
    )
    assert _has_trace(output)
    assert not _has_trace(
        '  File "/workspace/sympy/tests/test_value.py", line 42, in test_value\n'
        "DeprecationWarning: old API\n"
    )


@pytest.mark.parametrize(
    "output",
    [
        "ImportError while loading conftest '/workspace/conftest.py'.",
        "ERROR collecting tests/test_module.py",
        "Interrupted: 1 error during collection",
        "ERROR: file or directory not found: tests/test_missing.py",
        "/usr/bin/python: No module named pytest",
    ],
)
def test_pytest_collection_failure_is_infrastructure(output: str) -> None:
    assert _is_collection_failure(output)


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


def test_collection_error_with_returncode_one_is_infrastructure(
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
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "print(\"ImportError while loading conftest "
                            "'/workspace/conftest.py'.\"); raise SystemExit(1)"
                        ),
                    ],
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
            tmp_path / "container.cid",
            tmp_path / "bootstrap.sh",
            tmp_path / "test.patch",
        )
    command, backend = _execution_command(
        {
            "execution_backend": "container",
            "container_image": "python@sha256:" + "a" * 64,
        },
        ("python", "-m", "pytest"),
        tmp_path / "container.cid",
        tmp_path / "bootstrap.sh",
        tmp_path / "test.patch",
    )
    assert backend == "container"
    assert command[:3] == ("docker", "run", "--rm")
    assert "--network" in command
    assert "none" in command
    assert "HYPOTHESIS_STORAGE_DIRECTORY=/tmp/hypothesis" in command
    assert "PYTEST_ADDOPTS=-p no:cacheprovider" in command
    assert "/testbed" in command
    assert "/workspace" not in command
    assert "--pull" in command
    assert "never" in command
    assert any("dst=/runtime-test.patch,readonly" in item for item in command)


def test_runtime_image_cleanup_removes_only_exact_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = "untagged"
        stderr = ""

    def fake_run(command: list[str], **_: object) -> Completed:
        observed.append(command)
        return Completed()

    monkeypatch.setattr(
        "scripts.ch4_revision.collect_h10_c5b_runtime.subprocess.run",
        fake_run,
    )
    image = "registry.invalid/project@sha256:" + "a" * 64
    result = _remove_runtime_image(image, {"PATH": "/usr/bin"})

    assert result["status"] == "PASS"
    assert observed == [["docker", "image", "rm", image]]


def test_runtime_image_prepare_retries_exact_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []
    outcomes = iter((1, 1, 0))

    class Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.stdout = "output"
            self.stderr = "" if returncode == 0 else "connection reset"

    def fake_run(command: list[str], **_: object) -> Completed:
        observed.append(command)
        if command[1:3] == ["image", "inspect"]:
            return Completed(1)
        return Completed(next(outcomes))

    monkeypatch.setattr(
        "scripts.ch4_revision.collect_h10_c5b_runtime.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "scripts.ch4_revision.collect_h10_c5b_runtime.time.sleep",
        lambda _: None,
    )
    image = "registry.invalid/project@sha256:" + "a" * 64
    result = _prepare_runtime_image(image, {"PATH": "/usr/bin"})

    assert result["status"] == "PASS"
    assert result["attempts"] == 3
    assert observed == [
        ["docker", "image", "inspect", image],
        ["docker", "pull", image],
        ["docker", "pull", image],
        ["docker", "pull", image],
    ]


def test_runtime_image_prepare_records_pull_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = 0

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "missing"

    def fake_run(command: list[str], **_: object) -> Completed:
        nonlocal observed
        observed += 1
        if command[1:3] == ["image", "inspect"]:
            return Completed()
        raise TimeoutError

    def timeout_run(command: list[str], **kwargs: object) -> Completed:
        try:
            return fake_run(command, **kwargs)
        except TimeoutError as error:
            raise subprocess.TimeoutExpired(command, 300) from error

    monkeypatch.setattr(
        "scripts.ch4_revision.collect_h10_c5b_runtime.subprocess.run",
        timeout_run,
    )
    monkeypatch.setattr(
        "scripts.ch4_revision.collect_h10_c5b_runtime.time.sleep",
        lambda _: None,
    )
    image = "registry.invalid/project@sha256:" + "b" * 64
    result = _prepare_runtime_image(image, {"PATH": "/usr/bin"}, attempts=1)

    assert result["status"] == "FAILED"
    assert result["returncode"] == 124
    assert result["attempts"] == 1
    assert observed == 2


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
