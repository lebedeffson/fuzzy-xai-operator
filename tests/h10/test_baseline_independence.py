from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN = {"TypedRouteGuard", "RepairPlanner", "RepairSetPlanner", "DiagnosticCutSolver", "H10Auditor"}


def test_baselines_do_not_import_full_auditor_components() -> None:
    root = Path("baselines/h10")
    failures = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names}
                if imported & FORBIDDEN or (node.module or "").startswith("fuzzyxai.audit_h10"):
                    failures.append(f"{path}:{node.lineno}")
    assert not failures, f"baseline independence violated: {failures}"
