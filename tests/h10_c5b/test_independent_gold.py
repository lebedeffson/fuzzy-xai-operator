from __future__ import annotations

import ast
from pathlib import Path

from fuzzyxai.gold_repository import extract_gold


def test_gold_extracts_changed_symbol_and_contract_without_auditor_import() -> None:
    patch = "diff --git a/src/io.py b/src/io.py\n"
    before = {"src/io.py": "def load(value):\n    return value\n"}
    after = {"src/io.py": "import json\n\ndef load(value):\n    return json.loads(value)\n"}
    gold = extract_gold(patch, before, after)
    assert gold.changed_symbols == (("src/io.py", "load"),)
    assert gold.atoms[0].contract == "SERIALIZATION"
    assert gold.changed_api_calls == (("src/io.py", "json.loads"),)
    assert gold.scorer_version == "independent-ast-diff-gold-v2"
    source = Path(
        "framework/fuzzyxai/fuzzyxai/gold_repository/scorer.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("repository_diagnostics" in name for name in imports)
    assert "COMPONENT_RULES" not in source
    assert "CONTRACT_RULES" not in source


def test_gold_extracts_changed_configuration_key_without_method_dictionary() -> None:
    patch = "diff --git a/pyproject.toml b/pyproject.toml\n"
    gold = extract_gold(
        patch,
        {"pyproject.toml": "[project]\ndependencies = ['numpy>=1']\n"},
        {"pyproject.toml": "[project]\ndependencies = ['numpy>=2']\n"},
    )
    assert gold.changed_config_keys == (("pyproject.toml", "dependencies"),)
    assert gold.atoms[0].contract == "DEPENDENCY_VERSION"
