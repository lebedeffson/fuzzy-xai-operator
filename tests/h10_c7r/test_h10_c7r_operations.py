from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.ch4_revision import build_h10_c7r_gold as gold_builder
from scripts.ch4_revision import collect_h10_c7r_runtime as runtime
from scripts.ch4_revision import prepare_h10_c7r_held_out as preparation


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_selection_is_deterministic_and_gold_is_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = []
    for index in range(41):
        rows.append(
            {
                "repo": f"owner/repo-{index % 13}",
                "instance_id": f"owner__repo-{index % 13}-{index}",
                "base_commit": f"{index:040x}",
                "problem_statement": f"failure {index}",
                "test_patch": "diff --git a/test.py b/test.py\n",
                "patch": "diff --git a/pkg.py b/pkg.py\n",
                "test_cmds": ["pytest -rA"],
                "FAIL_TO_PASS": [f"test.py::test_{index}"],
            }
        )
    source = tmp_path / "source.parquet"
    pd.DataFrame(rows).to_parquet(source)
    monkeypatch.setattr(
        preparation,
        "SOURCE_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    exclusion = tmp_path / "exclusion.json"
    exclusion.write_text(
        json.dumps({"excluded_repositories": []}),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    first = preparation.prepare(source, exclusion, output)
    first_bytes = (output / "HELD_OUT_SELECTION.jsonl").read_bytes()
    second = preparation.prepare(source, exclusion, output)
    assert first == second
    assert first_bytes == (output / "HELD_OUT_SELECTION.jsonl").read_bytes()
    public = _read_jsonl(output / "HELD_OUT_SELECTION.jsonl")
    runtime_rows = _read_jsonl(output / "RUNTIME_REGISTRY.jsonl")
    sealed = _read_jsonl(output / "SEALED_GOLD_SOURCE.jsonl")
    assert len([row for row in public if row["selection_role"] == "PRIMARY"]) == 40
    assert len({row["repository"] for row in public[:40]}) >= 12
    assert all("patch" not in json.dumps(row).lower() for row in public)
    assert all("patch" in row and "test_patch" not in row for row in sealed)
    assert all("test_patch" in row and "patch" not in row for row in runtime_rows)


def test_instrumented_pytest_command_supports_registered_wrappers() -> None:
    fail_to_pass = ["tests/test_x.py::test_failure"]
    argv, direct = runtime._instrumented_command(["pytest -rA"], fail_to_pass)
    assert argv[-1] == fail_to_pass[0]
    assert ".venv/bin/python" in direct
    assert "/h10/runtime_launcher.py" in direct
    assert "/h10/runtime_launcher_xdist.py" in direct
    assert '\"$py\" \"$launcher\"' in direct
    _, uv = runtime._instrumented_command(
        ["uv run -p .venv pytest -rA"],
        fail_to_pass,
    )
    assert ".venv/bin/python" in uv
    assert "uv run --offline --no-sync" in uv
    _, poetry = runtime._instrumented_command(
        ["poetry run pytest tests -v"],
        fail_to_pass,
    )
    assert ".venv/bin/python" in poetry
    assert "poetry run python" in poetry
    _, preferred = runtime._instrumented_command(
        ["pytest -rA", "uv run pytest -rA"],
        fail_to_pass,
    )
    assert "uv run --offline --no-sync" in preferred


def test_runtime_event_filter_excludes_environment_and_non_python() -> None:
    assert runtime._project_python_event(
        {
            "source_file": "tests/test_x.py",
            "target_file": "pkg/module.py",
        }
    )
    assert not runtime._project_python_event(
        {"source_file": ".venv/lib/site-packages/pkg/module.py"}
    )
    assert not runtime._project_python_event(
        {"source_file": "src/propcache/_helpers_c.pyx"}
    )


def test_pytest_node_event_requires_observed_project_test(tmp_path: Path) -> None:
    test_file = tmp_path / "tests/test_module.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_failure(): pass\n", encoding="utf-8")
    test_id = "tests/test_module.py::Suite::test_failure[value]"
    event = runtime._pytest_node_event(tmp_path, test_id, f"FAILED {test_id}")
    assert event is not None
    assert event["source_file"] == "tests/test_module.py"
    assert event["source_symbol"] == "Suite.test_failure"
    assert runtime._pytest_node_event(tmp_path, test_id, "other output") is None


def test_runtime_collection_uses_one_image_as_rolling_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selection = tmp_path / "selection.jsonl"
    registry = tmp_path / "registry.jsonl"
    availability = tmp_path / "availability.json"
    public_rows = [
        {
            "incident_id": f"incident-{index}",
            "repository": f"repository-{index}",
            "selection_rank": index,
            "selection_role": "PRIMARY",
        }
        for index in range(1, 4)
    ]
    registry_rows = [
        {
            "incident_id": f"incident-{index}",
            "container_image_tag": f"image-{index}",
        }
        for index in range(1, 4)
    ]
    runtime.write_jsonl(selection, public_rows)
    runtime.write_jsonl(registry, registry_rows)
    availability.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        **row,
                        "availability_status": (
                            "AVAILABLE_WITHIN_RUNNER_IMAGE_BUDGET"
                        ),
                    }
                    for row in registry_rows
                ]
            }
        ),
        encoding="utf-8",
    )
    operations: list[str] = []

    def fake_collect_one(public, registered, output):
        operations.append(f"collect:{registered['container_image_tag']}")
        return {
            **public,
            "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
        }

    def fake_run(arguments, **_kwargs):
        if arguments[:3] == ["docker", "image", "rm"]:
            operations.append(f"remove:{arguments[3]}")
        return runtime.subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(runtime, "collect_one", fake_collect_one)
    monkeypatch.setattr(runtime, "run", fake_run)
    report = runtime.collect(
        selection,
        registry,
        availability,
        tmp_path / "runtime",
        target_incidents=3,
        minimum_repositories=3,
        prune_images=True,
    )

    assert report["status"] == "H10_C7R_RUNTIME_EVIDENCE_COMPLETE"
    assert operations == [
        "collect:image-1",
        "collect:image-2",
        "remove:image-1",
        "collect:image-3",
        "remove:image-2",
        "remove:image-3",
    ]


def test_runtime_collection_rejects_unavailable_image_before_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    selection = tmp_path / "selection.jsonl"
    registry = tmp_path / "registry.jsonl"
    availability = tmp_path / "availability.json"
    output = tmp_path / "runtime"
    public = {
        "incident_id": "incident-1",
        "repository": "repository-1",
        "selection_rank": 1,
        "selection_role": "PRIMARY",
    }
    registered = {
        "incident_id": "incident-1",
        "container_image_tag": "image-1",
    }
    runtime.write_jsonl(selection, [public])
    runtime.write_jsonl(registry, [registered])
    availability.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        **registered,
                        "availability_status": (
                            "RUNTIME_INFRASTRUCTURE_UNAVAILABLE_IMAGE_BUDGET"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    cached = output / "incidents/incident-1/observable.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(
        json.dumps(
            {
                **public,
                "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime,
        "collect_one",
        lambda *_args, **_kwargs: pytest.fail("collector must not run"),
    )

    with pytest.raises(RuntimeError):
        runtime.collect(
            selection,
            registry,
            availability,
            output,
            target_incidents=1,
            minimum_repositories=1,
            prune_images=True,
        )

    ledger = [
        json.loads(line)
        for line in (output / "RUNTIME_AVAILABILITY_LEDGER.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert ledger[0]["status"] == (
        "RUNTIME_INFRASTRUCTURE_OR_REPRODUCTION_FAILED"
    )
    assert "IMAGE_BUDGET" in ledger[0]["reason"]


def test_gold_builder_binds_hunk_to_graph_symbol(tmp_path: Path) -> None:
    graph = {
        "nodes": [
            {
                "file_path": "pkg/module.py",
                "symbol": "target",
                "attributes": {"lineno": 8, "end_lineno": 18},
            }
        ]
    }
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "incident_id": "case",
                "graph_path": "graph.json",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text(
        json.dumps(
            {
                "incident_id": "case",
                "patch": (
                    "diff --git a/pkg/module.py b/pkg/module.py\n"
                    "--- a/pkg/module.py\n"
                    "+++ b/pkg/module.py\n"
                    "@@ -12,1 +12,1 @@\n"
                    "-old\n"
                    "+new\n"
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "gold.jsonl"
    result = gold_builder.build_gold(sealed, manifest, output)
    assert result["incidents"] == 1
    atom = _read_jsonl(output)[0]["atoms"][0]
    assert atom == {
        "file_path": "pkg/module.py",
        "symbol": "target",
        "contract": "NOT_SCORED",
    }


def test_gold_builder_rejects_missing_localization(tmp_path: Path) -> None:
    assert gold_builder.changed_locations("not a patch") == []
