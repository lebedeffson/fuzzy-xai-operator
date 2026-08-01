#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fuzzyxai.experiments.h10_c5c_runtime import (
    _assertion_difference,
    _failure_observation_events,
    _launcher_source,
    _traceback_events,
)
from fuzzyxai.experiments.h10_c7r_r10 import (
    audit_runtime_rows,
    enrich_graph_with_source_excerpts,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    FileRetriever,
    IncidentQuery,
    R10SymbolPoolBuilder,
    documents_from_graph,
)
from fuzzyxai.repository_diagnostics.importer import RepositoryIncident
from fuzzyxai.repository_diagnostics.importer_v2 import (
    EvidenceGroundedRepositoryImporter,
)
from fuzzyxai.repository_diagnostics.runtime_events import (
    RuntimeEvent,
    normalize_runtime_event_rows,
)

COLLECTOR_CAPABILITY = "causal-chronology-tail-v6"


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
) -> tuple[list[str], list[str], str]:
    parsed = [shlex.split(command) for command in test_commands]
    ordered = sorted(
        parsed,
        key=lambda tokens: (
            0 if tokens and tokens[0] in {"pytest", "py.test"} else 1,
            0 if len(tokens) >= 3 and tokens[0] == "cd" else 1,
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

    working_directory = "/testbed"
    if len(selected) >= 4 and selected[0] == "cd" and selected[2] == "&&":
        relative_workdir = Path(selected[1])
        if relative_workdir.is_absolute() or ".." in relative_workdir.parts:
            raise ValueError("registered pytest working directory is unsafe")
        working_directory = str(Path("/testbed") / relative_workdir)
        selected = selected[3:]
    argv = ["pytest", "-vv", "-x", *fail_to_pass]
    fallback_files = list(
        dict.fromkeys(test_id.split("::", 1)[0] for test_id in fail_to_pass)
    )
    if not fallback_files or any(
        Path(path).is_absolute() or ".." in Path(path).parts
        for path in fallback_files
    ):
        raise ValueError("registered pytest fallback paths are unsafe")
    # A malformed parametrized node ID must not broaden recollection to the
    # repository's complete test suite.
    fallback_argv = ["pytest", "-vv", "-x", *fallback_files]
    wrapper = (
        f"cd {shlex.quote(working_directory)}; "
        "py=''; mode=''; "
        "for candidate in .venv/bin/python /opt/venv/bin/python "
        "/venv/bin/python python; do "
        "if \"$candidate\" -m pytest --help >/tmp/h10-pytest-help 2>/dev/null; "
        "then py=\"$candidate\"; mode='direct'; break; fi; done; "
        "if [ -z \"$mode\" ] && command -v poetry >/dev/null "
        "&& poetry run python -m pytest --help "
        ">/tmp/h10-pytest-help 2>/dev/null; then mode='poetry'; fi; "
        "if [ -z \"$mode\" ] && command -v uv >/dev/null "
        "&& uv run --offline --no-sync python -m pytest --help "
        ">/tmp/h10-pytest-help 2>/dev/null; then mode='uv'; fi; "
        "[ -n \"$mode\" ] || exit 87; "
        "launcher='/h10/runtime_launcher.py'; "
        "fallback='/h10/runtime_launcher_fallback.py'; "
        "if grep -q -- '--numprocesses' /tmp/h10-pytest-help; "
        "then launcher='/h10/runtime_launcher_xdist.py'; "
        "fallback='/h10/runtime_launcher_fallback_xdist.py'; fi; "
        "if [ \"$mode\" = 'direct' ]; then \"$py\" \"$launcher\"; "
        "elif [ \"$mode\" = 'poetry' ]; then poetry run python \"$launcher\"; "
        "else uv run --offline --no-sync python \"$launcher\"; fi; "
        "rc=$?; "
        "if [ \"$rc\" -eq 4 ]; then "
        "if [ \"$mode\" = 'direct' ]; then \"$py\" \"$fallback\"; "
        "elif [ \"$mode\" = 'poetry' ]; "
        "then poetry run python \"$fallback\"; "
        "else uv run --offline --no-sync python \"$fallback\"; fi; "
        "else (exit \"$rc\"); fi"
    )
    return argv, fallback_argv, wrapper


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


def _manifest_digest(value: str) -> str:
    return value.rsplit("@", 1)[-1]


def _verify_image_digest(actual: str, expected: str) -> None:
    if _manifest_digest(actual) != _manifest_digest(expected):
        raise RuntimeError(
            "container image digest mismatch: "
            f"expected={expected}, actual={actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_events(event_dir: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    for path in sorted(event_dir.glob("probe-*.jsonl")):
        values.extend(read_jsonl(path))
    values.sort(
        key=lambda row: (
            int(row.get("timestamp_ns", 0)) <= 0,
            int(row.get("timestamp_ns", 0)),
            int(row.get("sequence_id", -1)),
        )
    )
    return normalize_runtime_event_rows(values)


def _stamp_collector_observations(
    events: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Timestamp post-execution observations without inventing runtime order."""
    stamped = []
    for source in events:
        event = dict(source)
        if int(event.get("timestamp_ns", 0)) <= 0:
            event["timestamp_ns"] = time.monotonic_ns()
            event["timestamp_source"] = "collector_postprocess"
        event.setdefault("thread_id", 0)
        event.setdefault("call_depth", 0)
        event.setdefault("occurrence_count", 1)
        stamped.append(event)
    return stamped


def _prune_environment_trees(root: Path) -> tuple[str, ...]:
    """Remove copied environments from the temporary source projection."""
    excluded = {
        ".tox",
        ".venv",
        "__pycache__",
        "dist-packages",
        "site-packages",
        "venv",
    }
    removed: list[str] = []
    for parent, directory_names, _file_names in os.walk(root, topdown=True):
        parent_path = Path(parent)
        selected = [name for name in directory_names if name in excluded]
        directory_names[:] = [
            name for name in directory_names if name not in excluded
        ]
        for name in selected:
            path = parent_path / name
            run(["chmod", "-R", "u+rwX", str(path)])
            shutil.rmtree(path)
            removed.append(path.relative_to(root).as_posix())
    return tuple(sorted(removed))


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


def _pytest_node_event(
    root: Path,
    test_id: str,
    output: str,
) -> dict[str, object] | None:
    if test_id not in output:
        return None
    parts = test_id.split("::")
    relative = Path(parts[0])
    if relative.suffix != ".py":
        return None
    source_path = root / relative
    if not source_path.is_file():
        matches = sorted(
            path
            for path in root.glob(f"**/{relative.as_posix()}")
            if path.is_file()
        )
        if len(matches) != 1:
            return None
        relative = matches[0].relative_to(root)
    symbol_parts = [
        part.split("[", 1)[0]
        for part in parts[1:]
        if part.split("[", 1)[0]
    ]
    symbol = ".".join(symbol_parts) or None
    encoded = f"{test_id}\0traceback_frame\0{relative.as_posix()}".encode()
    return {
        "event_id": "trace-" + hashlib.sha256(encoded).hexdigest()[:24],
        "test_id": test_id,
        "kind": "traceback_frame",
        "source_file": relative.as_posix(),
        "source_symbol": symbol,
        "target_file": None,
        "target_symbol": None,
        "detail": f"pytest failing node: {test_id}",
    }


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
    expected_image_digest: str,
    output: Path,
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
    argv, fallback_argv, wrapper = _instrumented_command(
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
    (incident_dir / "runtime_launcher_fallback.py").write_text(
        _launcher_source(
            Path("/testbed"),
            Path("/h10/events"),
            fail_to_pass[0],
            tuple(fallback_argv),
        ),
        encoding="utf-8",
    )
    (incident_dir / "runtime_launcher_fallback_xdist.py").write_text(
        _launcher_source(
            Path("/testbed"),
            Path("/h10/events"),
            fail_to_pass[0],
            (*fallback_argv[:1], "-n", "0", *fallback_argv[1:]),
        ),
        encoding="utf-8",
    )
    image_tag = str(registry["container_image_tag"])
    image_digest, image_size = _image_digest(image_tag)
    _verify_image_digest(image_digest, expected_image_digest)
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
            pruned_environment_trees = _prune_environment_trees(source_root)
            normalized = (stdout + "\n" + stderr).replace(
                "/testbed/",
                f"{source_root.as_posix()}/",
            )
            events = [
                event
                for event in _probe_events(events_dir)
                if _project_python_event(event)
            ]
            traceback_events = _stamp_collector_observations(
                _traceback_events(
                    normalized,
                    root=source_root,
                    test_id=fail_to_pass[0],
                )
            )
            events.extend(traceback_events)
            events.extend(
                _stamp_collector_observations(
                    _failure_observation_events(
                        normalized,
                        test_id=fail_to_pass[0],
                        traceback_events=traceback_events,
                    )
                )
            )
            events = [
                event for event in events if _project_python_event(event)
            ]
            if not any(event["kind"] == "traceback_frame" for event in events):
                node_event = _pytest_node_event(
                    source_root,
                    fail_to_pass[0],
                    stdout + "\n" + stderr,
                )
                if node_event is not None:
                    events.extend(_stamp_collector_observations([node_event]))
            events = normalize_runtime_event_rows(events)
            readiness = audit_runtime_rows(events)
            if not readiness.ready:
                raise RuntimeError(
                    "R10_RUNTIME_RECOLLECTION_REQUIRED: "
                    + json.dumps(readiness.to_mapping(), sort_keys=True)
                )
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
            graph = enrich_graph_with_source_excerpts(graph, source_root)
            documents = documents_from_graph(graph, typed_events)
            query = IncidentQuery(
                identifier,
                str(public["query"].get("issue", "")),
                tuple(fail_to_pass),
                stdout + "\n" + stderr,
                assertion,
            )
            files = FileRetriever().rank(
                query,
                documents,
                typed_events,
                limit=20,
            )
            symbol_pool = R10SymbolPoolBuilder().build(
                query,
                files,
                documents,
                symbols_per_file=10,
                pool_limit=200,
            )
            graph_path = incident_dir / "repository_graph.json"
            events_path = incident_dir / "runtime_events.jsonl"
            write_json(graph_path, asdict(graph))
            write_jsonl(events_path, events)
            row = {
                **public,
                "collector_capability": COLLECTOR_CAPABILITY,
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
                    "max_sequence_id": readiness.max_sequence_id,
                    "occurrence_count_total": readiness.occurrence_count_total,
                    "aggregated_event_count": readiness.aggregated_event_count,
                    "full_tail_end_preserved": (
                        readiness.full_tail_end_preserved
                    ),
                    "event_schema": "R10_CAUSAL_RUNTIME_V1",
                    "chronology_preserved": readiness.ready,
                    "causal_event_kinds": sorted(
                        kinds
                        & {
                            "argument_value",
                            "assertion_operand",
                            "exception",
                            "last_writer",
                            "return_value",
                            "value_flow",
                        }
                    ),
                    "project_grounded_file_count": len(
                        {
                            str(event["source_file"])
                            for event in events
                            if event.get("source_file")
                        }
                    ),
                    "file_candidate_count": len(files),
                    "symbol_pool_size": len(symbol_pool),
                    "pruned_environment_trees": list(
                        pruned_environment_trees
                    ),
                    "runtime_evidence_sha256": _sha256(events_path),
                    "runtime_readiness": readiness.to_mapping(),
                    "status": "BUG_REPRODUCED_WITH_TRACE",
                },
            )
            return row
    finally:
        events_dir.chmod(0o755)
        run(["docker", "rm", "-f", container])


def collect(
    selection: Path,
    registry_path: Path,
    image_availability_lock: Path,
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
    availability_payload = json.loads(
        image_availability_lock.read_text(encoding="utf-8")
    )
    availability = {
        str(row["incident_id"]): row
        for row in availability_payload["entries"]
    }
    if set(public_rows) != set(availability):
        raise ValueError("selection and image availability lock differ")
    for identifier, registered in registry.items():
        if (
            availability[identifier]["container_image_tag"]
            != registered["container_image_tag"]
        ):
            raise ValueError(f"image availability tag mismatch: {identifier}")
    ordered = sorted(
        public_rows.values(),
        key=lambda row: (
            int(row["selection_rank"]),
            str(row["incident_id"]),
        ),
    )
    completed = []
    ledger = []
    retained_image: str | None = None
    try:
        for public in ordered:
            identifier = str(public["incident_id"])
            if len(completed) >= target_incidents:
                break
            cached = output / "incidents" / identifier / "observable.json"
            failure_cache = (
                output / "incidents" / identifier / "runtime_failure.json"
            )
            image_tag = str(registry[identifier]["container_image_tag"])
            newly_collected = not cached.is_file()
            cached_failure = False
            try:
                image_status = str(
                    availability[identifier]["availability_status"]
                )
                if image_status != "AVAILABLE_WITHIN_RUNNER_IMAGE_BUDGET":
                    raise RuntimeError(image_status)
                if cached.is_file():
                    row = json.loads(cached.read_text(encoding="utf-8"))
                    if row.get("collector_capability") != COLLECTOR_CAPABILITY:
                        cached.unlink()
                        row = collect_one(
                            public,
                            registry[identifier],
                            str(availability[identifier]["manifest_digest"]),
                            output,
                        )
                elif failure_cache.is_file():
                    failure = json.loads(
                        failure_cache.read_text(encoding="utf-8")
                    )
                    if failure.get("collector_capability") != COLLECTOR_CAPABILITY:
                        failure_cache.unlink()
                        row = collect_one(
                            public,
                            registry[identifier],
                            str(availability[identifier]["manifest_digest"]),
                            output,
                        )
                    else:
                        cached_failure = True
                        raise RuntimeError(str(failure["reason"]))
                else:
                    row = collect_one(
                        public,
                        registry[identifier],
                        str(availability[identifier]["manifest_digest"]),
                        output,
                    )
                if failure_cache.is_file():
                    failure_cache.unlink()
                completed.append(row)
                ledger.append(
                    {
                        "incident_id": identifier,
                        "selection_rank": public["selection_rank"],
                        "selection_role": public["selection_role"],
                        "status": "BUG_REPRODUCED_WITH_TRACE",
                    }
                )
                runtime_status = json.loads(
                    (
                        output
                        / "incidents"
                        / identifier
                        / "runtime_status.json"
                    ).read_text(encoding="utf-8")
                )
                checkpoint = {
                    "incident_id": identifier,
                    "repository_id": str(public["repository"]),
                    "runtime_readiness": "R10_RUNTIME_READY",
                    "event_count": int(runtime_status["event_count"]),
                    "causal_event_kinds": runtime_status[
                        "causal_event_kinds"
                    ],
                    "max_sequence_id": int(
                        runtime_status["max_sequence_id"]
                    ),
                    "project_grounded_file_count": int(
                        runtime_status["project_grounded_file_count"]
                    ),
                    "symbol_pool_size": int(
                        runtime_status["symbol_pool_size"]
                    ),
                    "collector_error": "",
                    "container_digest": str(
                        runtime_status["container_digest"]
                    ),
                    "evidence_sha256": str(
                        runtime_status["runtime_evidence_sha256"]
                    ),
                }
                if prune_images and newly_collected:
                    if retained_image and retained_image != image_tag:
                        run(["docker", "image", "rm", retained_image])
                    retained_image = image_tag
            except Exception as error:  # noqa: BLE001 - preserve runtime ledger
                incident_dir = output / "incidents" / identifier
                reason = str(error)
                if not cached_failure:
                    write_json(
                        incident_dir / "runtime_failure.json",
                        {
                            "collector_capability": COLLECTOR_CAPABILITY,
                            "incident_id": identifier,
                            "reason": reason,
                            "status": (
                                "RUNTIME_INFRASTRUCTURE_OR_REPRODUCTION_FAILED"
                            ),
                        },
                    )
                ledger.append(
                    {
                        "incident_id": identifier,
                        "selection_rank": public["selection_rank"],
                        "selection_role": public["selection_role"],
                        "status": "RUNTIME_INFRASTRUCTURE_OR_REPRODUCTION_FAILED",
                        "reason": (
                            f"CACHED_RUNTIME_FAILURE: {reason}"
                            if cached_failure
                            else reason
                        ),
                    }
                )
                checkpoint = {
                    "incident_id": identifier,
                    "repository_id": str(public["repository"]),
                    "runtime_readiness": "R10_RUNTIME_RECOLLECTION_REQUIRED",
                    "event_count": 0,
                    "causal_event_kinds": [],
                    "max_sequence_id": -1,
                    "project_grounded_file_count": 0,
                    "symbol_pool_size": 0,
                    "collector_error": reason,
                    "container_digest": str(
                        availability[identifier].get("manifest_digest", "")
                    ),
                    "evidence_sha256": "",
                }
                if prune_images and image_tag != retained_image:
                    run(["docker", "image", "rm", image_tag])
            checkpoint_rows = [
                row
                for row in (
                    json.loads(line)
                    for line in (
                        output / "R10_CAUSAL_RECOLLECTION_CHECKPOINT.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
                if row["incident_id"] != identifier
            ] if (
                output / "R10_CAUSAL_RECOLLECTION_CHECKPOINT.jsonl"
            ).is_file() else []
            checkpoint_rows.append(checkpoint)
            write_jsonl(
                output / "R10_CAUSAL_RECOLLECTION_CHECKPOINT.jsonl",
                checkpoint_rows,
            )
            write_jsonl(output / "RUNTIME_AVAILABILITY_LEDGER.jsonl", ledger)
            write_jsonl(output / "HELD_OUT_MANIFEST.jsonl", completed)
    finally:
        if prune_images and retained_image:
            run(["docker", "image", "rm", retained_image])

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
        "image_availability_lock": str(image_availability_lock),
        "network_during_execution": "none",
        "technical_barrier_passed": complete
        and all(
            row["runtime_readiness"] == "R10_RUNTIME_READY"
            for row in checkpoint_rows
        ),
        "lifecycle_status": (
            "H10_C7R_R10_CAUSAL_DEVELOPMENT_READY_FOR_SCORING"
            if complete and target_incidents >= 40
            else "H10_C7R_R10_CAUSAL_DEVELOPMENT_RECOLLECTION_IN_PROGRESS"
        ),
        "scientific_result": "NOT_EVALUATED",
        "development_scored": False,
        "held_out_created": False,
        "held_out_scored": False,
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
    parser.add_argument("--image-availability-lock", type=Path, required=True)
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
                args.image_availability_lock.resolve(),
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
