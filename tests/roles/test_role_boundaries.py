from __future__ import annotations

import ast
from pathlib import Path

from fuzzyxai.roles import ExternalPolicy, RouteAuditor


def test_role_protocols_are_distinct() -> None:
    assert RouteAuditor is not ExternalPolicy
    assert "audit" in RouteAuditor.__dict__
    assert "decide" not in RouteAuditor.__dict__
    assert "decide" in ExternalPolicy.__dict__


def test_incident_audit_has_no_external_policy_import() -> None:
    path = Path("framework/fuzzyxai/fuzzyxai/incidents/audit.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "ExternalPolicy" not in imported
