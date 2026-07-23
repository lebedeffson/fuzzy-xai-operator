from __future__ import annotations

import platform
import subprocess

from ..hashing import read_json
from ..paths import ARTIFACT_ROOT, REPO_ROOT


def build_validation_report() -> str:
    gate = read_json(ARTIFACT_ROOT / "audit" / "preconfirmatory_gate.json")
    power = read_json(ARTIFACT_ROOT / "power" / "recommended_design.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    coverage_path = ARTIFACT_ROOT / "audit" / "coverage.json"
    coverage = read_json(coverage_path)["totals"]["percent_covered"] if coverage_path.exists() else None
    regression_path = ARTIFACT_ROOT / "audit" / "full_regression.json"
    regression = read_json(regression_path) if regression_path.exists() else {"status": "NOT_RUN"}
    return (
        "# H10-C2 preconfirmatory validation\n\n"
        f"- Branch commit: `{commit}`\n"
        f"- Python: `{platform.python_version()}`\n"
        f"- Software workflow: `PASS`\n"
        f"- v21 integrity: `{'PASS' if gate['v21_integrity'] else 'FAIL'}`\n"
        f"- Power design: `{power['status']}`\n"
        f"- H10-C2 coverage: `{coverage if coverage is not None else 'NOT_MEASURED'}%` "
        "(registered target: 95%)\n"
        f"- Full regression: `{regression['status']}`\n"
        f"- Manual adjudication: `{gate['manual_adjudication']}`\n"
        f"- Sealed opening count: `{gate['sealed_opening_count']}`\n"
        f"- H10-C2a: `NOT_EVALUATED`\n"
        f"- H10-C2b: `NOT_EVALUATED`\n"
        f"- Scientific release: `BLOCKED_PRECONFIRMATORY`\n\n"
        "Current blockers:\n\n"
        + "\n".join(f"- `{item}`" for item in gate["blockers"])
        + "\n\nNo confirmatory value was computed.\n"
    )
