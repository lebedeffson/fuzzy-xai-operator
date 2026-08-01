from __future__ import annotations

from pathlib import Path

import pytest
from fuzzyxai.repository_diagnostics import (
    RepositoryIncident,
    RepositoryStructureImporter,
)


def _repository(root: Path) -> None:
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src/__init__.py").write_text("", encoding="utf-8")
    (root / "src/core.py").write_text(
        "import json\n\ndef transform(value):\n    return json.loads(value)\n",
        encoding="utf-8",
    )
    (root / "tests/test_core.py").write_text(
        "from src.core import transform\n\ndef test_transform():\n    assert transform('{}') == {}\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\ndependencies=['numpy>=2']\n",
        encoding="utf-8",
    )


def test_importer_builds_file_symbol_import_call_test_and_config_graph(tmp_path: Path) -> None:
    _repository(tmp_path)
    incident = RepositoryIncident(
        "incident-1",
        "fixture/repo",
        "buggy",
        tmp_path,
        ("tests/test_core.py::test_transform",),
        traceback=f'File "{tmp_path}/src/core.py", line 4, in transform\nValueError',
    )
    graph = RepositoryStructureImporter().build(incident)
    kinds = {node.kind for node in graph.nodes}
    relations = {edge.relation for edge in graph.edges}
    assert {
        "repository",
        "package",
        "module",
        "file",
        "function",
        "test",
        "configuration_key",
        "dependency",
        "serialized_artifact",
        "runtime_exception",
    } <= kinds
    assert {
        "imports",
        "calls",
        "loads",
        "configured_by",
        "depends_on",
        "fails_in",
        "produces",
    } <= relations
    transform = next(node for node in graph.nodes if node.symbol == "transform")
    assert transform.attributes["parameters"] == ("value",)
    assert graph.obligations == ("failing_test:0:tests/test_core.py::test_transform",)
    assert not graph.limitations


def test_repository_graph_changes_with_actual_repository_structure(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _repository(first)
    _repository(second)
    (second / "src/extra.py").write_text("def extra():\n    return 1\n", encoding="utf-8")
    common = {
        "incident_id": "x",
        "repository": "fixture/repo",
        "buggy_revision": "buggy",
        "failing_tests": ("test_transform",),
    }
    importer = RepositoryStructureImporter()
    one = importer.build(RepositoryIncident(repository_root=first, **common))
    two = importer.build(RepositoryIncident(repository_root=second, **common))
    assert one.trace_sha256 != two.trace_sha256
    assert len(one.nodes) != len(two.nodes)


def test_gold_fields_are_rejected_before_import(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gold fields"):
        RepositoryIncident.from_mapping(
            {
                "incident_id": "x",
                "repository": "fixture/repo",
                "buggy_revision": "buggy",
                "repository_root": str(tmp_path),
                "failing_tests": (),
                "patch": "secret",
            }
        )
