#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from fuzzyxai.experiments.h10_c5b import _read_sources
from fuzzyxai.gold_repository import extract_gold
from fuzzyxai.repository_diagnostics.guided_retrieval import documents_from_graph
from fuzzyxai.repository_diagnostics.importer import RepositoryIncident
from fuzzyxai.repository_diagnostics.importer_v2 import (
    EvidenceGroundedRepositoryImporter,
)
from fuzzyxai.repository_diagnostics.runtime_events import RuntimeEvent

TRACEBACK_FILE = re.compile(
    r'^\s*File "([^"]+\.py)", line \d+(?:, in ([^\s]+))?'
)
PYTEST_FRAME = re.compile(
    r"^\s*([A-Za-z0-9_./\\-]+\.py):(\d+): in ([^\s]+)"
)
PYTEST_LOCATION = re.compile(r"^\s*([A-Za-z0-9_./\\-]+\.py):\d+(?::|$)")


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def _relative_source(path_text: str, root: Path) -> str | None:
    candidate = Path(path_text.replace("\\", "/"))
    if not candidate.is_absolute():
        direct = root / candidate
        if direct.exists():
            return candidate.as_posix()
    else:
        try:
            return candidate.resolve().relative_to(root.resolve()).as_posix()
        except (OSError, ValueError):
            pass
    parts = candidate.parts
    for index in range(len(parts)):
        relative = Path(*parts[index:])
        if (root / relative).exists():
            return relative.as_posix()
    return None


def _traceback_events(
    text: str,
    *,
    root: Path,
    test_id: str,
) -> tuple[RuntimeEvent, ...]:
    values = []
    seen: set[tuple[str, str | None]] = set()
    for raw in text.splitlines():
        path_text = None
        symbol = None
        match = TRACEBACK_FILE.match(raw)
        if match:
            path_text, symbol = match.groups()
        else:
            match = PYTEST_FRAME.match(raw)
            if match:
                path_text, _line, symbol = match.groups()
            else:
                match = PYTEST_LOCATION.match(raw)
                if match:
                    path_text = match.group(1)
        if path_text is None:
            continue
        relative = _relative_source(path_text, root)
        if relative is None:
            continue
        normalized_symbol = symbol.strip() if symbol else None
        key = (relative, normalized_symbol)
        if key in seen:
            continue
        seen.add(key)
        encoded = (
            f"{test_id}\0traceback_frame\0{relative}\0"
            f"{normalized_symbol or ''}"
        ).encode()
        values.append(
            RuntimeEvent.from_mapping(
                {
                    "event_id": "trace-" + hashlib.sha256(encoded).hexdigest()[:24],
                    "test_id": test_id,
                    "kind": "traceback_frame",
                    "source_file": relative,
                    "source_symbol": normalized_symbol,
                    "target_file": None,
                    "target_symbol": None,
                    "detail": raw.strip(),
                }
            )
        )
    return tuple(values)


def _assertion_difference(text: str) -> str:
    selected = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith(("E ", "E\t")):
            selected.append(stripped[1:].strip())
        elif stripped.startswith(("AssertionError", "assert ")):
            selected.append(stripped)
    return "\n".join(dict.fromkeys(selected))


def _runtime_event_mapping(event: RuntimeEvent) -> dict[str, object]:
    return asdict(event)


def _copy_open_replay(
    base_bundle: Path,
    output: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str]]:
    incidents = []
    repositories = set()
    for row in _jsonl(base_bundle / "incidents.jsonl"):
        identifier = str(row["incident_id"])
        graph_target = output / "repository_graphs" / f"{identifier}.json"
        runtime_target = output / "runtime_events" / f"{identifier}.jsonl"
        _link_or_copy(base_bundle / str(row["graph_path"]), graph_target)
        _link_or_copy(
            base_bundle / str(row["runtime_events_path"]),
            runtime_target,
        )
        copied = dict(row)
        copied["graph_path"] = f"repository_graphs/{identifier}.json"
        copied["runtime_events_path"] = f"runtime_events/{identifier}.jsonl"
        copied["development_origin"] = "H10-C5c-open-replay"
        incidents.append(copied)
        repositories.add(str(row["repository"]))
    return incidents, list(_jsonl(base_bundle / "gold.jsonl")), repositories


def _materialize_disclosed_incident(
    row: dict[str, object],
    *,
    output: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    identifier = str(row["incident_id"])
    repository = str(row["repository"])
    root = Path(str(row["repository_root"])).resolve()
    failing_tests = tuple(str(item) for item in row["failing_tests"])
    traceback = Path(str(row["traceback_path"])).read_text(
        encoding="utf-8",
        errors="replace",
    )
    stdout = Path(str(row["stdout_path"])).read_text(
        encoding="utf-8",
        errors="replace",
    )
    stderr = Path(str(row["stderr_path"])).read_text(
        encoding="utf-8",
        errors="replace",
    )
    combined = f"{traceback}\n{stdout}\n{stderr}"
    events = _traceback_events(
        combined,
        root=root,
        test_id=failing_tests[0],
    )
    if not events:
        raise ValueError(
            f"no project-grounded traceback event for {identifier}"
        )
    public = RepositoryIncident(
        incident_id=identifier,
        repository=repository,
        buggy_revision=str(row["buggy_revision"]),
        repository_root=root,
        failing_tests=failing_tests,
        traceback=traceback,
        stdout=stdout,
        stderr=stderr,
        assertion_difference=_assertion_difference(combined),
    )
    graph = EvidenceGroundedRepositoryImporter().build(
        public,
        runtime_events=events,
    )
    graph_path = output / "repository_graphs" / f"{identifier}.json"
    runtime_path = output / "runtime_events" / f"{identifier}.jsonl"
    _write_json(graph_path, asdict(graph))
    _write_jsonl(
        runtime_path,
        [_runtime_event_mapping(event) for event in events],
    )
    documents = documents_from_graph(graph, events)
    observable = {
        "incident_id": identifier,
        "repository": repository,
        "split": "development",
        "development_origin": "H10-C5b-disclosed-held-out",
        "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
        "repository_symbol_count": len(documents),
        "graph_path": f"repository_graphs/{identifier}.json",
        "runtime_events_path": f"runtime_events/{identifier}.jsonl",
        "query": {
            "issue": f"{stderr}\n{stdout}".strip(),
            "failing_tests": list(failing_tests),
            "traceback": traceback,
            "assertion": public.assertion_difference,
        },
    }
    gold = extract_gold(
        Path(str(row["patch_path"])).read_text(
            encoding="utf-8",
            errors="replace",
        ),
        _read_sources(Path(str(row["before_sources_path"]))),
        _read_sources(Path(str(row["after_sources_path"]))),
    )
    if not gold.atoms:
        raise ValueError(f"independent Gold is empty for {identifier}")
    gold_row = {
        "incident_id": identifier,
        "atoms": [
            {
                "file_path": atom.file_path,
                "symbol": atom.symbol,
                "contract": atom.contract,
            }
            for atom in gold.atoms
        ],
        "scorer_version": gold.scorer_version,
    }
    return observable, gold_row


def _write_sha256s(output: Path) -> int:
    paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )
    return len(paths)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--h10-c5b-manifest", type=Path, required=True)
    parser.add_argument("--additional-incidents", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    incidents, gold, repositories = _copy_open_replay(
        args.base_bundle.resolve(),
        output,
    )
    disclosed = sorted(
        _jsonl(args.h10_c5b_manifest.resolve()),
        key=lambda row: str(row["selection_rank_sha256"]),
    )[: args.additional_incidents]
    for row in disclosed:
        observable, gold_row = _materialize_disclosed_incident(
            row,
            output=output,
        )
        incidents.append(observable)
        gold.append(gold_row)
        repositories.add(str(observable["repository"]))
    incident_ids = [str(row["incident_id"]) for row in incidents]
    if len(incident_ids) != len(set(incident_ids)):
        raise ValueError("combined development incident IDs are not unique")
    _write_jsonl(output / "DEVELOPMENT_MANIFEST.jsonl", incidents)
    _write_jsonl(output / "DEVELOPMENT_GOLD.jsonl", gold)
    inventory = {
        "status": (
            "H10_C7A_DEVELOPMENT_EXTENSION_READY"
            if len(incidents) >= 40 and len(repositories) >= 10
            else "H10_C7A_DEVELOPMENT_EXTENSION_INCOMPLETE"
        ),
        "incident_count": len(incidents),
        "repository_count": len(repositories),
        "base_incident_count": len(incidents) - len(disclosed),
        "added_incident_count": len(disclosed),
        "added_incidents": [str(row["incident_id"]) for row in disclosed],
        "added_repositories": sorted(
            {str(row["repository"]) for row in disclosed}
        ),
        "selection_rule": "first_by_registered_selection_rank_sha256",
        "gold_channel_separate": True,
        "runtime_reexecuted": False,
        "project_dependencies_reinstalled": False,
        "h10_c5b_reclassified_as_open_development": True,
        "confirmatory_excluded_repositories": sorted(repositories),
    }
    _write_json(output / "DEVELOPMENT_INVENTORY.json", inventory)
    file_count = _write_sha256s(output)
    inventory["sha256_file_count"] = file_count
    _write_json(output / "DEVELOPMENT_INVENTORY.json", inventory)
    _write_sha256s(output)
    print(json.dumps(inventory, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
