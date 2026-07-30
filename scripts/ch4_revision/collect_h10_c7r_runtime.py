#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fuzzyxai.experiments.h10_c5c_runtime import (
    _assertion_difference,
    _launcher_source,
    _traceback_events,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    documents_from_graph,
)
from fuzzyxai.repository_diagnostics.importer import RepositoryIncident
from fuzzyxai.repository_diagnostics.importer_v2 import (
    EvidenceGroundedRepositoryImporter,
)
from fuzzyxai.repository_diagnostics.runtime_events import RuntimeEvent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def run(
    arguments: list[str],
    *,
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _instrumented_command(
    test_commands: list[str],
    fail_to_pass: list[str],
) -> tuple[list[str], str]:
    parsed = [shlex.split(command) for command in test_commands]
    ordered = sorted(
        parsed,
        key=lambda tokens: (
            0 if tokens and tokens[0] in {"pytest", "py.test"} else 1,
            len(tokens),
        ),
    )
    selected: list[str] | None = None
    for tokens in ordered:
        if "pytest" in tokens:
            selected = tokens
            break
        for index in range(len(tokens) - 2):
            if tokens[index : index + 3] == ["python", "-m", "pytest"]:
                selected = tokens
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError("no instrumentable pytest command")

    pytest_index = selected.index("pytest")
    prefix = selected[:pytest_index]
    argv = ["pytest", "-vv", "-x", *fail_to_pass]
    if (
        prefix[:2] in (["uv", "run"], ["poetry", "run"])
        or prefix
        and all("=" in token for token in prefix)
    ):
        python_prefix = [*prefix, "python"]
    else:
        python_prefix = ["python"]
    help_command = shlex.join([*python_prefix, "-m", "pytest", "--help"])
    normal = shlex.join([*python_prefix, "/h10/runtime_launcher.py"])
    xdist = shlex.join([*python_prefix, "/h10/runtime_launcher_xdist.py"])
    wrapper = (
        f"if {help_command} 2>/dev/null | grep -q -- '--numprocesses'; "
        f"then {xdist}; else {normal}; fi"
    )
    return argv, wrapper


def _image_digest(tag: str) -> tuple[str, int]:
    pull = run(["docker", "pull", tag], timeout=1800)
    if pull.returncode:
        raise RuntimeError(f"image pull failed: {pull.stderr[-1000:]}")
    inspection = run(
        [
            "docker",
            "image",
            "inspect",
            tag,
            "--format",
            "{{json .RepoDigests}}\t{{.Size}}",
        ],
        check=True,
    )
    digests_json, size = inspection.stdout.strip().split("\t", 1)
    digests = json.loads(digests_json)
    if not digests:
        raise ValueError(f"image has no repository digest: {tag}")
    return str(digests[0]), int(size)


def _probe_events(event_dir: Path) -> list[dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for path in sorted(event_dir.glob("probe-*.jsonl")):
        for row in read_jsonl(path):
            values[str(row["event_id"])] = row
    return [values[key] for key in sorted(values)]


def _project_python_event(event: dict[str, object]) -> bool:
    paths = [
        str(event[field])
        for field in ("source_file", "target_file")
        if event.get(field)
    ]
    if not paths:
        return False
    for value in paths:
        path = Path(value)
        if path.suffix != ".py":
            return False
        if any(
            part in {
                ".venv",
                "venv",
                "site-packages",
                "dist-packages",
                "__pycache__",
            }
            for part in path.parts
        ):
            return False
    return True


def _source_lines(root: Path) -> int:
    total = 0
    for path in root.rglob("*.py"):
        if ".git" in path.parts:
            continue
        try:
            total += len(path.read_bytes().splitlines())
        except OSError:
            continue
    return total


def _container_name(identifier: str) -> str:
    digest = hashlib.sha256(identifier.encode()).hexdigest()[:16]
    return f"h10-c7r-{digest}"


def collect_one(
    public: dict[str, Any],
    registry: dict[str, Any],
    output: Path,
    *,
    prune_image: bool,
) -> dict[str, object]:
    identifier = str(public["incident_id"])
    incident_dir = output / "incidents" / identifier
    incident_dir.mkdir(parents=True, exist_ok=True)
    events_dir = incident_dir / "events"
    events_dir.mkdir(exist_ok=True)
    for stale_event in events_dir.glob("probe-*.jsonl"):
        stale_event.unlink()
    events_dir.chmod(0o777)
    (incident_dir / "test.patch").write_text(
        str(registry["test_patch"]),
        encoding="utf-8",
    )
    fail_to_pass = [str(item) for item in registry["fail_to_pass"]]
    argv, wrapper = _instrumented_command(
        [str(item) for item in registry["test_commands"]],
        fail_to_pass,
    )
    (incident_dir / "runtime_launcher.py").write_text(
        _launcher_source(
            Path("/testbed"),
            Path("/h10/events"),
            fail_to_pass[0],
            tuple(argv),
        ),
        encoding="utf-8",
    )
    (incident_dir / "runtime_launcher_xdist.py").write_text(
        _launcher_source(
            Path("/testbed"),
            Path("/h10/events"),
            fail_to_pass[0],
            (*argv[:3], "-n", "0", *argv[3:]),
        ),
        encoding="utf-8",
    )
    image_tag = str(registry["container_image_tag"])
    image_digest, image_size = _image_digest(image_tag)
    container = _container_name(identifier)
    run(["docker", "rm", "-f", container])
    overlay = (
        "if git apply --check /h10/test.patch; then "
        "git apply /h10/test.patch; "
        "elif git apply --reverse --check /h10/test.patch; then :; "
        "else exit 86; fi; "
    )
    execution = (
        overlay
        + "set +e; "
        + wrapper
        + "; rc=$?; chmod 0644 /h10/events/probe-*.jsonl 2>/dev/null || :; "
        + "exit $rc"
    )
    command = [
        "docker",
        "run",
        "--name",
        container,
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "--memory",
        "12g",
        "--cpus",
        "4",
        "-v",
        f"{incident_dir}:/h10:rw",
        image_digest,
        "bash",
        "-lc",
        execution,
    ]
    started = time.monotonic()
    timed_out = False
    try:
        completed = run(
            command,
            timeout=int(registry.get("timeout_seconds", 900)),
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        run(["docker", "kill", container])
    runtime_seconds = time.monotonic() - started
    (incident_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    (incident_dir / "stderr.txt").write_text(stderr, encoding="utf-8")

    try:
        with tempfile.TemporaryDirectory(prefix="h10-c7r-source-") as temp:
            source_parent = Path(temp)
            copied = run(
                ["docker", "cp", f"{container}:/testbed", str(source_parent)],
                timeout=900,
            )
            if copied.returncode:
                raise RuntimeError(f"docker cp failed: {copied.stderr}")
            source_root = source_parent / "testbed"
            normalized = (stdout + "\n" + stderr).replace(
                "/testbed/",
                f"{source_root.as_posix()}/",
            )
            events = [
                event
                for event in _probe_events(events_dir)
                if _project_python_event(event)
            ]
            events.extend(
                _traceback_events(
                    normalized,
                    root=source_root,
                    test_id=fail_to_pass[0],
                )
            )
            unique = {str(event["event_id"]): event for event in events}
            events = [unique[key] for key in sorted(unique)]
            kinds = {str(event["kind"]) for event in events}
            typed_events = tuple(
                RuntimeEvent.from_mapping(event) for event in events
            )
            reproduced = returncode not in (0, 86, 124)
            status = (
                "BUG_REPRODUCED_WITH_TRACE"
                if reproduced
                and "coverage" in kinds
                and "traceback_frame" in kinds
                else "RUNTIME_EVIDENCE_INCOMPLETE"
            )
            if status != "BUG_REPRODUCED_WITH_TRACE":
                raise RuntimeError(
                    f"{status}: rc={returncode}, kinds={sorted(kinds)}"
                )
            assertion = _assertion_difference(stdout + "\n" + stderr)
            incident = RepositoryIncident(
                incident_id=identifier,
                repository=str(public["repository"]),
                buggy_revision=str(public["buggy_revision"]),
                repository_root=source_root,
                failing_tests=tuple(fail_to_pass),
                traceback=stdout + "\n" + stderr,
                stdout=stdout,
                stderr=stderr,
                assertion_difference=assertion,
            )
            graph = EvidenceGroundedRepositoryImporter().build(
                incident,
                runtime_events=typed_events,
            )
            documents = documents_from_graph(graph, typed_events)
            graph_path = incident_dir / "repository_graph.json"
            events_path = incident_dir / "runtime_events.jsonl"
            write_json(graph_path, asdict(graph))
            write_jsonl(events_path, events)
            row = {
                **public,
                "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
                "graph_path": str(graph_path.relative_to(output)),
                "runtime_events_path": str(events_path.relative_to(output)),
                "repository_symbol_count": len(documents),
                "repository_source_lines": _source_lines(source_root),
                "query": {
                    **public["query"],
                    "traceback": stdout + "\n" + stderr,
                    "assertion": assertion,
                },
            }
            write_json(incident_dir / "observable.json", row)
            write_json(
                incident_dir / "runtime_status.json",
                {
                    "container_digest": image_digest,
                    "container_image_size": image_size,
                    "network": "none",
                    "returncode": returncode,
                    "runtime_seconds": runtime_seconds,
                    "timed_out": timed_out,
                    "event_count": len(events),
                    "event_kinds": sorted(kinds),
                    "status": "BUG_REPRODUCED_WITH_TRACE",
                },
            )
            return row
    finally:
        events_dir.chmod(0o755)
        run(["docker", "rm", "-f", container])
        if prune_image:
            run(["docker", "image", "rm", image_tag])


def collect(
    selection: Path,
    registry_path: Path,
    output: Path,
    *,
    target_incidents: int,
    minimum_repositories: int,
    prune_images: bool,
) -> dict[str, object]:
    public_rows = {
        str(row["incident_id"]): row for row in read_jsonl(selection)
    }
    registry = {
        str(row["incident_id"]): row for row in read_jsonl(registry_path)
    }
    if set(public_rows) != set(registry):
        raise ValueError("selection and runtime registry differ")
    ordered = sorted(
        public_rows.values(),
        key=lambda row: (
            int(row["selection_rank"]),
            str(row["incident_id"]),
        ),
    )
    completed = []
    ledger = []
    for public in ordered:
        identifier = str(public["incident_id"])
        if len(completed) >= target_incidents:
            break
        cached = output / "incidents" / identifier / "observable.json"
        try:
            if cached.is_file():
                row = json.loads(cached.read_text(encoding="utf-8"))
            else:
                row = collect_one(
                    public,
                    registry[identifier],
                    output,
                    prune_image=prune_images,
                )
            completed.append(row)
            ledger.append(
                {
                    "incident_id": identifier,
                    "selection_rank": public["selection_rank"],
                    "selection_role": public["selection_role"],
                    "status": "BUG_REPRODUCED_WITH_TRACE",
                }
            )
        except Exception as error:  # noqa: BLE001 - preserve the runtime ledger
            ledger.append(
                {
                    "incident_id": identifier,
                    "selection_rank": public["selection_rank"],
                    "selection_role": public["selection_role"],
                    "status": "RUNTIME_INFRASTRUCTURE_OR_REPRODUCTION_FAILED",
                    "reason": str(error),
                }
            )
        write_jsonl(output / "RUNTIME_AVAILABILITY_LEDGER.jsonl", ledger)
        write_jsonl(output / "HELD_OUT_MANIFEST.jsonl", completed)

    repositories = {str(row["repository"]) for row in completed}
    complete = (
        len(completed) >= target_incidents
        and len(repositories) >= minimum_repositories
    )
    report = {
        "target_incidents": target_incidents,
        "complete_incidents": len(completed),
        "complete_repositories": len(repositories),
        "failed_candidates": sum(
            row["status"] != "BUG_REPRODUCED_WITH_TRACE" for row in ledger
        ),
        "network_during_execution": "none",
        "status": (
            "H10_C7R_RUNTIME_EVIDENCE_COMPLETE"
            if complete
            else "H10_C7R_RUNTIME_EVIDENCE_INCOMPLETE"
        ),
    }
    write_json(output / "H10_C7R_RUNTIME_REPORT.json", report)
    if not complete:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--runtime-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-incidents", type=int, default=40)
    parser.add_argument("--minimum-repositories", type=int, default=12)
    parser.add_argument("--prune-images", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.selection.resolve(),
                args.runtime_registry.resolve(),
                args.output.resolve(),
                target_incidents=args.target_incidents,
                minimum_repositories=args.minimum_repositories,
                prune_images=args.prune_images,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
