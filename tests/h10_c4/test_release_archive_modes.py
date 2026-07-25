from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path("scripts/build_h10_c4_release.py").resolve()


def _load_builder():
    spec = importlib.util.spec_from_file_location("h10_c4_release_builder", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_member_preserves_executable_mode(tmp_path: Path) -> None:
    builder = _load_builder()
    source = tmp_path / "runner.py"
    source.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    archive_path = tmp_path / "release.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        builder._write_member(
            archive,
            source,
            Path("package/runner.py"),
            0o755,
        )

    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo("package/runner.py")
        assert builder._zip_mode(info) == 0o755


def test_committed_h10_c4_scripts_are_executable() -> None:
    builder = _load_builder()
    commit = builder._git("rev-parse", "HEAD")
    modes = dict(builder._git_entries(commit))

    assert modes["scripts/run_h10_c4.py"] == 0o755
    assert modes["scripts/build_h10_c4_chapter4.py"] == 0o755
    assert modes["scripts/build_h10_c4_release.py"] == 0o755
    assert modes["scripts/manuscript_claim_lint.py"] == 0o755
