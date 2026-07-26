from __future__ import annotations

from pathlib import Path

from fuzzyxai.repository_diagnostics.importer import RepositoryIncident
from fuzzyxai.repository_diagnostics.importer_v2 import (
    INTENTIONAL_SYNTAX_FIXTURE,
    PARSEABLE_SOURCE,
    UNPARSEABLE_RUNTIME_TARGET,
    EvidenceGroundedRepositoryImporter,
)


def test_intentional_fixture_does_not_pollute_repository_limitations(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    (tmp_path / "src/core.py").write_text(
        "def run():\n    return 1\n",
        encoding="utf-8",
    )
    (tmp_path / "tests/fixtures/bad.py").write_text(
        "def broken(:\n",
        encoding="utf-8",
    )
    graph = EvidenceGroundedRepositoryImporter().build(
        RepositoryIncident(
            "incident",
            "fixture/repo",
            "buggy",
            tmp_path,
            ("tests/test_core.py::test_run",),
        )
    )
    core = next(node for node in graph.nodes if node.file_path == "src/core.py")
    bad = next(node for node in graph.nodes if node.file_path == "tests/fixtures/bad.py" and node.kind == "file")
    assert core.attributes["parse_status"] == PARSEABLE_SOURCE
    assert bad.attributes["parse_status"] == INTENTIONAL_SYNTAX_FIXTURE
    assert not any("syntax_error" in value for value in graph.limitations)


def test_unparseable_runtime_target_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    target = tmp_path / "src/broken.py"
    target.write_text("def broken(:\n", encoding="utf-8")
    graph = EvidenceGroundedRepositoryImporter().build(
        RepositoryIncident(
            "incident",
            "fixture/repo",
            "buggy",
            tmp_path,
            ("tests/test_core.py::test_run",),
            traceback=f'File "{target}", line 1, in broken\nSyntaxError',
        )
    )
    state = next(node for node in graph.nodes if node.kind == "source_parse_state")
    assert state.attributes["parse_status"] == UNPARSEABLE_RUNTIME_TARGET
    assert any(value.startswith("unparseable_runtime_target:") for value in graph.limitations)
