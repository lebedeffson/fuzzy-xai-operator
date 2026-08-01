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


def test_h10_c4_scripts_are_executable_in_distribution() -> None:
    for path in (
        Path("scripts/run_h10_c4.py"),
        Path("scripts/build_h10_c4_chapter4.py"),
        Path("scripts/build_h10_c4_release.py"),
        Path("scripts/manuscript_claim_lint.py"),
    ):
        assert path.stat().st_mode & 0o777 == 0o755
