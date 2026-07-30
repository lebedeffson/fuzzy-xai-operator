from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

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
    assert direct == "python /h10/runtime_launcher.py"
    _, uv = runtime._instrumented_command(
        ["uv run -p .venv pytest -rA"],
        fail_to_pass,
    )
    assert uv == "uv run -p .venv python /h10/runtime_launcher.py"
    _, poetry = runtime._instrumented_command(
        ["poetry run pytest tests -v"],
        fail_to_pass,
    )
    assert poetry == "poetry run python /h10/runtime_launcher.py"


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
