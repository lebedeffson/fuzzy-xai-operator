from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.experiments.h10_c5b import (
    load_manifest,
    repository_cluster_bootstrap,
    run,
)


def _incident(root: Path, repository: str, incident_id: str) -> dict[str, object]:
    source = root / incident_id / "buggy"
    source.mkdir(parents=True)
    (source / "src").mkdir()
    (source / "src/core.py").write_text(
        "def load(value):\n    return value\n",
        encoding="utf-8",
    )
    before = {"src/core.py": "def load(value):\n    return value\n"}
    after = {
        "src/core.py": (
            "import json\n\n"
            "def load(value):\n"
            "    return json.loads(value)\n"
        )
    }
    incident = root / incident_id
    (incident / "before.json").write_text(json.dumps(before), encoding="utf-8")
    (incident / "after.json").write_text(json.dumps(after), encoding="utf-8")
    (incident / "fix.patch").write_text(
        "diff --git a/src/core.py b/src/core.py\n",
        encoding="utf-8",
    )
    (incident / "traceback.txt").write_text(
        f'File "{source}/src/core.py", line 1, in load\n'
        "json serialization failure\n",
        encoding="utf-8",
    )
    return {
        "incident_id": incident_id,
        "repository": repository,
        "buggy_revision": "buggy",
        "repository_root": str(source),
        "failing_tests": [f"test_{incident_id}"],
        "split": "held_out",
        "patch_path": str(incident / "fix.patch"),
        "before_sources_path": str(incident / "before.json"),
        "after_sources_path": str(incident / "after.json"),
        "traceback_path": str(incident / "traceback.txt"),
        "runtime_evidence_status": "BUG_REPRODUCED_WITH_TRACE",
    }


def test_runner_is_fail_closed_when_repository_design_is_too_small(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(_incident(tmp_path, "fixture/repo", "incident-1")) + "\n",
        encoding="utf-8",
    )
    result = run(manifest, tmp_path / "output")
    assert result["status"] == "H10_C5B_BLOCKED_REPOSITORY_DATA"
    assert result["gold_leakage_audit"] == "PASS"
    assert result["parent_result_modified"] is False
    assert (tmp_path / "output/results/h10_c5b/PER_INCIDENT_RESULTS.csv").is_file()


def test_development_only_run_is_not_reported_as_held_out_result(tmp_path: Path) -> None:
    row = _incident(tmp_path, "fixture/repo", "incident-development")
    row["split"] = "development"
    manifest = tmp_path / "development.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = run(manifest, tmp_path / "development-output")
    assert result["status"] == "H10_C5B_DEVELOPMENT_READY"
    assert result["held_out_scored"] is False
    assert result["development_incidents"] == 1


def test_missing_runtime_trace_blocks_development_claim(tmp_path: Path) -> None:
    row = _incident(tmp_path, "fixture/repo", "incident-no-runtime")
    row["split"] = "development"
    row["runtime_evidence_status"] = "FAILING_TEST_IDS_ONLY"
    manifest = tmp_path / "development-no-runtime.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = run(manifest, tmp_path / "blocked-output")
    assert result["status"] == "H10_C5B_BLOCKED_REPOSITORY_DATA"
    assert result["runtime_evidence_complete"] is False


def test_manifest_rejects_duplicate_incidents(tmp_path: Path) -> None:
    row = _incident(tmp_path, "fixture/repo", "incident-1")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        f"{json.dumps(row)}\n{json.dumps(row)}\n",
        encoding="utf-8",
    )
    try:
        load_manifest(manifest)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate incident must be rejected")


def test_repository_cluster_bootstrap_uses_repository_means() -> None:
    rows = []
    for repository in ("repo-a", "repo-b", "repo-c"):
        for incident in range(4):
            rows.extend(
                (
                    {
                        "repository": repository,
                        "method": "O_ROUTE",
                        "joint_file_symbol_contract_hit_at_3": 1.0,
                    },
                    {
                        "repository": repository,
                        "method": "B_GREEDY",
                        "joint_file_symbol_contract_hit_at_3": float(incident == 0),
                    },
                )
            )
    result = repository_cluster_bootstrap(rows, iterations=1_000, seed=7)
    assert result["repository_count"] == 3
    assert result["mean_difference"] == 0.75
    assert result["ci_lower"] == 0.75
