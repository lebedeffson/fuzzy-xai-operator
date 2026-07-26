#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

TRACEBACK_HEADER = "Traceback (most recent call last):"
PYTEST_FRAME = re.compile(r"^.+[.]py:\d+: in ", flags=re.MULTILINE)
PYTEST_FAILURE_LOCATION = re.compile(
    r"^.+[.]py:\d+: (?:[A-Za-z][A-Za-z0-9_.]*)(?:: .+)?$",
    flags=re.MULTILINE,
)
INFRASTRUCTURE_RETURNCODES = frozenset({2, 3, 4, 5})
DOCKER_INFRASTRUCTURE_RETURNCODES = frozenset({125, 126, 127})
CONTAINER_IMAGE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
SAFE_INCIDENT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
PATCH_PATH = re.compile(r"^diff --git a/(.+?) b/(.+?)$", flags=re.MULTILINE)


def _has_trace(value: str) -> bool:
    return (
        TRACEBACK_HEADER in value and 'File "' in value
    ) or PYTEST_FRAME.search(value) is not None or (
        " FAILURES " in value
        and "FAILED " in value
        and PYTEST_FAILURE_LOCATION.search(value) is not None
    )


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _runtime_environment(home: Path) -> dict[str, str]:
    allowed = {
        key: os.environ[key]
        for key in (
            "PATH",
            "LANG",
            "LC_ALL",
            "TZ",
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        )
        if key in os.environ
    }
    return {
        **allowed,
        "HOME": str(home),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }


def _execution_command(
    registered: dict[str, object],
    command: tuple[str, ...],
    sandbox: Path,
    cidfile: Path,
) -> tuple[tuple[str, ...], str]:
    backend = str(registered.get("execution_backend", "host"))
    if backend == "host":
        return command, backend
    if backend != "container":
        raise ValueError(f"unsupported runtime backend: {backend}")
    image = str(registered.get("container_image", ""))
    if not CONTAINER_IMAGE.fullmatch(image):
        raise ValueError("container image must be pinned by sha256 digest")
    return (
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--memory",
            "8g",
            "--cpus",
            "4",
            "--cidfile",
            str(cidfile),
            "--mount",
            f"type=bind,src={sandbox},dst=/workspace",
            "--workdir",
            "/workspace",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=2g",
            "--env",
            "PYTHONHASHSEED=0",
            "--env",
            "PYTHONNOUSERSITE=1",
            image,
            *command,
        ),
        backend,
    )


def _stop_timed_out_container(cidfile: Path, environment: dict[str, str]) -> None:
    if not cidfile.is_file():
        return
    container_id = cidfile.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{12,64}", container_id):
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            check=False,
            env=environment,
            timeout=30,
        )


def _apply_runtime_test_patch(
    registered: dict[str, object],
    sandbox: Path,
) -> str | None:
    raw_path = str(registered.get("runtime_test_patch_path", ""))
    expected = str(registered.get("runtime_test_patch_sha256", ""))
    if not raw_path or not re.fullmatch(r"[0-9a-f]{64}", expected):
        return "registered runtime test patch is incomplete"
    patch_path = Path(raw_path)
    if not patch_path.is_file():
        return "registered runtime test patch is missing"
    actual = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    if actual != expected:
        return "registered runtime test patch checksum mismatch"
    patch_text = patch_path.read_text(encoding="utf-8")
    paths = PATCH_PATH.findall(patch_text)
    if not paths:
        return "registered runtime test patch has no diff entries"
    for pair in paths:
        for raw in pair:
            path = PurePosixPath(raw)
            if path.is_absolute() or ".." in path.parts:
                return "registered runtime test patch contains an unsafe path"
    completed = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_path)],
        cwd=sandbox,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        return f"runtime test patch application failed: {completed.stderr.strip()}"
    return None


def collect(
    manifest_path: Path,
    command_registry_path: Path,
    output: Path,
) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    registry = json.loads(command_registry_path.read_text(encoding="utf-8"))
    incident_ids = [str(row["incident_id"]) for row in rows]
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("runtime manifest incident IDs must be unique")
    if any(not SAFE_INCIDENT_ID.fullmatch(incident_id) for incident_id in incident_ids):
        raise ValueError("runtime manifest contains an unsafe incident ID")
    output.mkdir(parents=True, exist_ok=True)
    enriched = []
    evidence_rows = []
    for row in rows:
        incident_id = str(row["incident_id"])
        registered = registry.get(incident_id)
        if not registered:
            enriched.append(
                {**row, "runtime_evidence_status": "RUNTIME_COMMAND_NOT_REGISTERED"}
            )
            evidence_rows.append(
                {
                    "incident_id": incident_id,
                    "status": "RUNTIME_COMMAND_NOT_REGISTERED",
                }
            )
            continue
        command = tuple(str(item) for item in registered["command"])
        with tempfile.TemporaryDirectory(prefix="fuzzyxai-h10-c5b-runtime-") as temporary:
            temporary_root = Path(temporary)
            sandbox = temporary_root / "repository"
            home = temporary_root / "home"
            cidfile = temporary_root / "container.cid"
            home.mkdir()
            shutil.copytree(
                Path(str(row["repository_root"])),
                sandbox,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".tox",
                    ".venv",
                    "__pycache__",
                    ".pytest_cache",
                ),
            )
            effective_command, backend = _execution_command(
                registered,
                command,
                sandbox,
                cidfile,
            )
            environment = _runtime_environment(home)
            patch_error = (
                _apply_runtime_test_patch(registered, sandbox)
                if backend == "container"
                else None
            )
            if patch_error is not None:
                stdout = ""
                stderr = patch_error
                returncode = 4
                combined = stderr
                status = "RUNTIME_INFRASTRUCTURE_ERROR"
            else:
                try:
                    completed = subprocess.run(
                        effective_command,
                        cwd=sandbox,
                        capture_output=True,
                        text=True,
                        timeout=int(registered.get("timeout_seconds", 900)),
                        check=False,
                        env=environment,
                    )
                    stdout = completed.stdout
                    stderr = completed.stderr
                    returncode: int | None = completed.returncode
                    expected_failure_codes = {
                        int(code)
                        for code in registered.get("expected_failure_returncodes", (1,))
                    }
                    combined = f"{stdout}\n{stderr}"
                    if returncode == 0:
                        status = "BUG_NOT_REPRODUCED"
                    elif returncode in INFRASTRUCTURE_RETURNCODES or (
                        backend == "container"
                        and returncode in DOCKER_INFRASTRUCTURE_RETURNCODES
                    ):
                        status = "RUNTIME_INFRASTRUCTURE_ERROR"
                    elif returncode not in expected_failure_codes:
                        status = "UNEXPECTED_FAILURE_RETURNCODE"
                    elif _has_trace(combined):
                        status = "BUG_REPRODUCED_WITH_TRACE"
                    else:
                        status = "BUG_REPRODUCED_WITHOUT_TRACE"
                except subprocess.TimeoutExpired as error:
                    if backend == "container":
                        _stop_timed_out_container(cidfile, environment)
                    stdout = _text(error.stdout)
                    stderr = _text(error.stderr)
                    returncode = None
                    combined = f"{stdout}\n{stderr}"
                    status = "RUNTIME_TIMEOUT"
                except OSError as error:
                    stdout = ""
                    stderr = str(error)
                    returncode = None
                    combined = stderr
                    status = "RUNTIME_EXECUTION_ERROR"
        incident_output = output / incident_id
        incident_output.mkdir(parents=True, exist_ok=True)
        stdout_path = incident_output / "stdout.txt"
        stderr_path = incident_output / "stderr.txt"
        traceback_path = incident_output / "traceback.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        traceback_path.write_text(combined if _has_trace(combined) else "", encoding="utf-8")
        enriched.append(
            {
                **row,
                "stdout_path": str(stdout_path.resolve()),
                "stderr_path": str(stderr_path.resolve()),
                "traceback_path": str(traceback_path.resolve()),
                "runtime_evidence_status": status,
            }
        )
        evidence_rows.append(
            {
                "incident_id": incident_id,
                "status": status,
                "returncode": returncode,
                "execution_backend": backend,
                "container_image": (
                    str(registered["container_image"])
                    if backend == "container"
                    else None
                ),
                "runtime_test_patch_sha256": (
                    str(registered["runtime_test_patch_sha256"])
                    if backend == "container"
                    else None
                ),
                "runtime_command_sha256": hashlib.sha256(
                    json.dumps(
                        registered,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(stderr.encode()).hexdigest(),
            }
        )
    enriched_path = output / "H10_C5B_RUNTIME_ENRICHED_MANIFEST.jsonl"
    enriched_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched),
        encoding="utf-8",
    )
    report = {
        "status": (
            "PASS"
            if evidence_rows
            and all(row["status"] == "BUG_REPRODUCED_WITH_TRACE" for row in evidence_rows)
            else "INCOMPLETE"
        ),
        "incident_count": len(rows),
        "trace_complete_count": sum(
            row["status"] == "BUG_REPRODUCED_WITH_TRACE" for row in evidence_rows
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "child_environment_policy": (
            "PATH_LANG_LOCALE_TZ_HOME_PYTHONHASHSEED_PYTHONNOUSERSITE_ONLY"
        ),
        "manifest": str(enriched_path),
        "input_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "command_registry_sha256": hashlib.sha256(
            command_registry_path.read_bytes()
        ).hexdigest(),
        "enriched_manifest_sha256": hashlib.sha256(
            enriched_path.read_bytes()
        ).hexdigest(),
        "evidence": evidence_rows,
    }
    (output / "RUNTIME_EVIDENCE_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--commands", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = collect(
        args.manifest.resolve(),
        args.commands.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
