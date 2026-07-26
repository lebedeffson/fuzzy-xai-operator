from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.ch4_revision.h10_c5b_runtime_ops import (
    METHOD_COMMIT,
    build_runtime_inputs,
    freeze_development_runtime,
    merge_runtime_evidence,
    plan_replacements,
    verify_method_lock,
    verify_runtime_readiness,
)


def _manifest_row(
    tmp_path: Path,
    incident_id: str = "repo__project-1",
    *,
    split: str = "development",
) -> dict[str, object]:
    return {
        "incident_id": incident_id,
        "repository": "repo/project",
        "repository_root": str(tmp_path),
        "buggy_revision": "a" * 40,
        "failing_tests": ["tests/test_bug.py::test_failure"],
        "selection_rank_sha256": "b" * 64,
        "split": split,
        "patch_path": "/private/gold.patch",
        "after_sources_path": "/private/after.json",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_runtime_inputs_strip_gold_and_build_argv(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_manifest_row(tmp_path)])
    report = build_runtime_inputs(source, tmp_path / "runtime")
    runtime_path = Path(str(report["runtime_manifest"]))
    runtime_row = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert "patch_path" not in runtime_row
    assert "after_sources_path" not in runtime_row
    commands = json.loads(
        Path(str(report["command_registry"])).read_text(encoding="utf-8")
    )
    command = commands["repo__project-1"]["command"]
    assert command[:3] == ["python", "-m", "pytest"]
    assert command[-2:] == ["-x", "-vv"]
    assert commands["repo__project-1"]["execution_backend"] == "host"


def test_container_runtime_requires_repository_image_map(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_manifest_row(tmp_path)])
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {
                "repo/project": (
                    "registry.example.invalid/project@sha256:" + "a" * 64
                )
            }
        ),
        encoding="utf-8",
    )
    source_dataset = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "instance_id": "repo__project-1",
                "test_patch": (
                    "diff --git a/tests/test_bug.py b/tests/test_bug.py\n"
                    "--- a/tests/test_bug.py\n"
                    "+++ b/tests/test_bug.py\n"
                ),
            }
        ]
    ).to_parquet(source_dataset)
    report = build_runtime_inputs(
        source,
        tmp_path / "runtime",
        container_images_path=images,
        source_dataset_path=source_dataset,
    )
    commands = json.loads(
        Path(str(report["command_registry"])).read_text(encoding="utf-8")
    )
    assert report["execution_backend"] == "container"
    assert commands["repo__project-1"]["container_image"].endswith("a" * 64)
    assert commands["repo__project-1"]["command"][:3] == [
        "/opt/miniconda3/envs/testbed/bin/python",
        "-m",
        "pytest",
    ]
    assert len(commands["repo__project-1"]["runtime_test_patch_sha256"]) == 64
    assert report["runtime_test_patch_count"] == 1


def test_container_runtime_prefers_incident_image_map(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        _manifest_row(tmp_path, "repo__project-1"),
        _manifest_row(tmp_path, "repo__project-2"),
    ]
    _write_jsonl(source, rows)
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {
                "repo__project-1": "registry.invalid/one@sha256:" + "a" * 64,
                "repo__project-2": "registry.invalid/two@sha256:" + "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    source_dataset = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "instance_id": row["incident_id"],
                "test_patch": (
                    "diff --git a/tests/test_bug.py b/tests/test_bug.py\n"
                    "--- a/tests/test_bug.py\n"
                    "+++ b/tests/test_bug.py\n"
                ),
            }
            for row in rows
        ]
    ).to_parquet(source_dataset)
    report = build_runtime_inputs(
        source,
        tmp_path / "runtime",
        container_images_path=images,
        source_dataset_path=source_dataset,
    )
    commands = json.loads(
        Path(str(report["command_registry"])).read_text(encoding="utf-8")
    )
    assert commands["repo__project-1"]["container_image"].endswith("a" * 64)
    assert commands["repo__project-2"]["container_image"].endswith("b" * 64)


def test_container_runtime_rejects_ambiguous_repository_image(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        _manifest_row(tmp_path, "repo__project-1"),
        _manifest_row(tmp_path, "repo__project-2"),
    ]
    _write_jsonl(source, rows)
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {"repo/project": "registry.invalid/project@sha256:" + "a" * 64}
        ),
        encoding="utf-8",
    )
    source_dataset = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "instance_id": row["incident_id"],
                "test_patch": (
                    "diff --git a/tests/test_bug.py b/tests/test_bug.py\n"
                    "--- a/tests/test_bug.py\n"
                    "+++ b/tests/test_bug.py\n"
                ),
            }
            for row in rows
        ]
    ).to_parquet(source_dataset)
    with pytest.raises(ValueError, match="ambiguous"):
        build_runtime_inputs(
            source,
            tmp_path / "runtime",
            container_images_path=images,
            source_dataset_path=source_dataset,
        )


def test_container_runtime_requires_registered_source_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [_manifest_row(tmp_path)])
    images = tmp_path / "images.json"
    images.write_text(
        json.dumps(
            {"repo/project": "registry.invalid/project@sha256:" + "a" * 64}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source dataset"):
        build_runtime_inputs(
            source,
            tmp_path / "runtime",
            container_images_path=images,
        )


def test_method_lock_matches_frozen_scientific_files() -> None:
    result = verify_method_lock(
        Path("protocol/h10_c5b_repository_grounded/METHOD_LOCK.json"),
        Path.cwd(),
    )
    assert result["method_commit"] == METHOD_COMMIT
    assert result["scientific_implementation_diff"] == 0
    git_object_available = (
        subprocess.run(
            ["git", "cat-file", "-e", f"{METHOD_COMMIT}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    assert result["git_object_verified"] is git_object_available


def test_merge_copies_only_runtime_evidence(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    runtime = tmp_path / "runtime.jsonl"
    output = tmp_path / "merged.jsonl"
    _write_jsonl(original, [_manifest_row(tmp_path)])
    _write_jsonl(
        runtime,
        [
            {
                **{
                    key: value
                    for key, value in _manifest_row(tmp_path).items()
                    if key not in {"patch_path", "after_sources_path"}
                },
                "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
                "traceback_path": "/runtime/traceback.txt",
                "patch_path": "/attempted/injection.patch",
            }
        ],
    )
    merge_runtime_evidence(original, runtime, output)
    merged = json.loads(output.read_text(encoding="utf-8"))
    assert merged["patch_path"] == "/private/gold.patch"
    assert merged["traceback_path"] == "/runtime/traceback.txt"


def test_replacement_is_same_repository_and_sha_ranked(tmp_path: Path) -> None:
    selected = tmp_path / "selected.jsonl"
    _write_jsonl(selected, [_manifest_row(tmp_path)])
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "evidence": [
                    {
                        "incident_id": "repo__project-1",
                        "status": "BUG_NOT_REPRODUCED",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.parquet"
    pd.DataFrame(
        [
            {"repo": "other/project", "instance_id": "other__project-1"},
            {"repo": "repo/project", "instance_id": "repo__project-3"},
            {"repo": "repo/project", "instance_id": "repo__project-2"},
        ]
    ).to_parquet(candidates)
    result = plan_replacements(
        candidates,
        selected,
        report,
        tmp_path / "replacement.json",
    )
    replacement = result["ledger"][0]
    assert replacement["repository"] == "repo/project"
    assert replacement["replacement_incident"] in {
        "repo__project-2",
        "repo__project-3",
    }
    assert replacement["gold_or_prediction_viewed"] is False


def test_runtime_readiness_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "runtime.jsonl"
    _write_jsonl(manifest, [_manifest_row(tmp_path)])
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "INCOMPLETE",
                "incident_count": 1,
                "trace_complete_count": 0,
                "evidence": [
                    {
                        "incident_id": "repo__project-1",
                        "status": "BUG_REPRODUCED_WITHOUT_TRACE",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete"):
        verify_runtime_readiness(manifest, report, "development")


def test_held_out_requires_development_lock(tmp_path: Path) -> None:
    rows = [
        _manifest_row(
            tmp_path,
            f"repo-{index // 3}__project-{index}",
            split="held_out",
        )
        for index in range(24)
    ]
    for index, row in enumerate(rows):
        row["repository"] = f"repo-{index // 3}/project"
    manifest = tmp_path / "heldout.jsonl"
    _write_jsonl(manifest, rows)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "incident_count": 24,
                "trace_complete_count": 24,
                "evidence": [
                    {
                        "incident_id": row["incident_id"],
                        "status": "BUG_REPRODUCED_WITH_TRACE",
                    }
                    for row in rows
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="development runtime lock"):
        verify_runtime_readiness(manifest, report, "held_out")
    lock = tmp_path / "development-lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "DEVELOPMENT_RUNTIME_FROZEN",
                "method_commit": METHOD_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    result = verify_runtime_readiness(
        manifest,
        report,
        "held_out",
        development_lock=lock,
    )
    assert result["repository_count"] == 8


def test_development_freeze_rejects_failed_leakage_audit(tmp_path: Path) -> None:
    manifest = tmp_path / "runtime.jsonl"
    _write_jsonl(manifest, [_manifest_row(tmp_path)])
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "incident_count": 1,
                "trace_complete_count": 1,
                "evidence": [
                    {
                        "incident_id": "repo__project-1",
                        "status": "BUG_REPRODUCED_WITH_TRACE",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "development_incidents": 1,
                "gold_leakage_audit": "FAIL",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leakage"):
        freeze_development_runtime(
            manifest,
            report,
            results,
            Path("protocol/h10_c5b_repository_grounded/METHOD_LOCK.json"),
            tmp_path / "lock.json",
            Path.cwd(),
        )


def test_ci_wires_runtime_targets_without_held_out_scoring() -> None:
    workflow = Path(
        ".github/workflows/h10-c5b-runtime-collection.yml"
    ).read_text(encoding="utf-8")
    for target in (
        "make h10-c5b-prepare ",
        "make h10-c5b-collect-runtime ",
        "make h10-c5b-run ",
        "make h9-e2e-v2 ",
    ):
        assert target in workflow
    assert "collect-held-out" in workflow
    assert "Provision operational Python 3.11" in workflow
    assert "venv --python 3.11 --seed" in workflow
    assert 'UV_CACHE_DIR="$RUNNER_TEMP/uv-cache"' in workflow
    assert (
        'UV_PYTHON_INSTALL_DIR="$H10_C5B_SOURCE_DIR/runtime-tools/uv-python"'
        in workflow
    )
    assert "Enforce no automatic held-out scoring" in workflow
    assert "${{ inputs.source_dir }}/runtime-*" not in workflow
    assert "${{ inputs.source_dir }}/runtime-development" in workflow
    assert "${{ inputs.source_dir }}/runtime-held_out" in workflow
    assert "h10-c5b-score" not in workflow
    assert "unregistered CI hardware" in workflow
    assert "does not replace the registered local microbenchmark" in workflow
    assert "H10_C5B_RUNTIME_SOURCE_DATASET=" in workflow
