from __future__ import annotations

import ast

from ..hashing import write_json
from ..paths import ARTIFACT_ROOT, PACKAGE_ROOT


FORBIDDEN = (
    "fuzzyxai.diagnostics.minimal_cut",
    "fuzzyxai.diagnostics.repair_planner",
    "fuzzyxai.diagnostics.causes",
    "fuzzyxai.audit_h10",
)


def audit_oracle() -> dict:
    files = sorted((PACKAGE_ROOT / "src" / "h10_c2" / "oracle").glob("*.py"))
    violations = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        violations.extend({"file": path.name, "import": value} for value in imports if value.startswith(FORBIDDEN))
    report = {"status": "PASS" if not violations else "ORACLE_INDEPENDENCE_FAIL", "violations": violations}
    write_json(ARTIFACT_ROOT / "audit" / "oracle_independence.json", report)
    if violations:
        raise RuntimeError("ORACLE_INDEPENDENCE_FAIL")
    return report

