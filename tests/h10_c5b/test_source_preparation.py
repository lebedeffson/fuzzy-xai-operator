from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from scripts.ch4_revision.prepare_h10_c5b_sources import (
    _gold_snapshots,
    _rank,
    select_rows,
)


def test_selection_is_repository_balanced_and_patch_independent() -> None:
    frame = pd.DataFrame(
        [
            {
                "repo": repository,
                "instance_id": f"{repository}-{index}",
                "patch": f"secret-{index}",
            }
            for repository in ("repo/a", "repo/b")
            for index in range(4)
        ]
    )
    selected = select_rows(frame, 2)
    assert len(selected) == 4
    assert {str(row["repo"]) for row in selected} == {"repo/a", "repo/b"}
    expected = {
        repository: sorted(
            (
                row
                for row in frame.to_dict(orient="records")
                if row["repo"] == repository
            ),
            key=lambda row: (_rank(str(row["instance_id"])), str(row["instance_id"])),
        )[:2]
        for repository in ("repo/a", "repo/b")
    }
    assert {
        repository: [row for row in selected if row["repo"] == repository]
        for repository in expected
    } == expected


def test_gold_snapshots_apply_patch_outside_buggy_checkout(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Fixture"],
        cwd=repository,
        check=True,
    )
    (repository / "core.py").write_text(
        "def load(value):\n    return value\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "core.py"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "buggy"], cwd=repository, check=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        text=True,
    ).strip()
    (repository / "core.py").write_text(
        "import json\n\n"
        "def load(value):\n"
        "    return json.loads(value)\n",
        encoding="utf-8",
    )
    patch = subprocess.check_output(["git", "diff"], cwd=repository, text=True)
    subprocess.run(["git", "restore", "core.py"], cwd=repository, check=True)
    before, after = _gold_snapshots(repository, commit, patch)
    assert "json.loads" not in before["core.py"]
    assert "json.loads" in after["core.py"]
    assert "json.loads" not in (repository / "core.py").read_text(encoding="utf-8")
