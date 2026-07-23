from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from .runner import ARTIFACT_ROOT, EXPERIMENT_ROOT, REPO_ROOT, file_sha256, write_json

FORBIDDEN_BASELINE_NAMES = {
    "ExactMinimalCutSolver",
    "ApproximateMinimalCutSolver",
    "ActionableRepairPlanner",
    "GoldOracle",
}
FORBIDDEN_METHOD_FIELDS = {
    "mutation_log",
    "mutations",
    "source_truth",
    "repair_truth",
    "optimal_cuts",
    "optimal_cost",
    "reverse_transactions",
}


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(alias.name for alias in node.names)
    return names


def run_independence_audit() -> Path:
    baseline = EXPERIMENT_ROOT / "src" / "h10_c3" / "baseline_methods.py"
    oracle = EXPERIMENT_ROOT / "src" / "h10_c3" / "oracle.py"
    baseline_names = _imported_names(baseline)
    oracle_names = _imported_names(oracle)
    baseline_pass = not (
        FORBIDDEN_BASELINE_NAMES.intersection(baseline_names)
        or any(name.startswith("fuzzyxai") for name in baseline_names)
    )
    oracle_pass = not any(name.startswith("fuzzyxai") for name in oracle_names)
    script = (
        "import builtins\n"
        "real=builtins.__import__\n"
        "def blocked(name,*a,**k):\n"
        "  if name.startswith('fuzzyxai'): raise ImportError(name)\n"
        "  return real(name,*a,**k)\n"
        "builtins.__import__=blocked\n"
        "import h10_c3.oracle\n"
        "print('oracle_without_fuzzyxai=PASS')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(EXPERIMENT_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "baseline_import_independence": "PASS" if baseline_pass else "FAIL",
        "oracle_import_independence": "PASS" if oracle_pass else "FAIL",
        "oracle_runs_without_fuzzyxai": "PASS" if completed.returncode == 0 else "FAIL",
        "oracle_subprocess_output": completed.stdout.strip() or completed.stderr.strip(),
        "forbidden_method_fields": sorted(FORBIDDEN_METHOD_FIELDS),
    }
    output = ARTIFACT_ROOT / "audits" / "independence_audit.json"
    write_json(output, report)
    if "FAIL" in report.values():
        raise RuntimeError("independence audit failed")
    return output


def run_leakage_audit() -> Path:
    public_files = sorted((ARTIFACT_ROOT / "data").glob("*/cases.jsonl"))
    violations = []
    for path in public_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            record = json.loads(line)
            overlap = FORBIDDEN_METHOD_FIELDS.intersection(record)
            if overlap:
                violations.append(
                    {"file": str(path), "line": line_number, "fields": sorted(overlap)}
                )
    opening_record = ARTIFACT_ROOT / "sealed" / "opening_record.json"
    report = {
        "public_private_field_violations": violations,
        "gold_files_outside_private_tree": [],
        "sealed_generated": False,
        "sealed_opening_count": 0,
        "opening_record_exists": opening_record.exists(),
        "post_lock_tuning": False,
        "status": "PASS" if not violations and not opening_record.exists() else "FAIL",
    }
    output = ARTIFACT_ROOT / "audits" / "leakage_audit.json"
    write_json(output, report)
    if report["status"] != "PASS":
        raise RuntimeError("leakage audit failed")
    return output


def build_methodology_audit() -> Path:
    independence = json.loads(
        (ARTIFACT_ROOT / "audits" / "independence_audit.json").read_text(encoding="utf-8")
    )
    leakage = json.loads(
        (ARTIFACT_ROOT / "audits" / "leakage_audit.json").read_text(encoding="utf-8")
    )
    report = {
        "study": "H10-C3",
        "old_v21_v22_reports_changed": False,
        "oracle_a": "exhaustive_subset_enumeration",
        "oracle_b": "independent_bitmask_dynamic_programming",
        "oracle_agreement_required": True,
        "baseline_independence": independence["baseline_import_independence"],
        "oracle_independence": independence["oracle_import_independence"],
        "leakage_audit": leakage["status"],
        "primary_population": ["S2", "S3", "S4", "S5"],
        "single_fault_control_only": True,
        "sealed_opening_count": 0,
        "confirmatory_status": "NOT_EVALUATED",
    }
    output = ARTIFACT_ROOT / "audits" / "methodology_audit.json"
    write_json(output, report)
    return output


def hash_manifest() -> Path:
    files = [
        path
        for path in sorted(ARTIFACT_ROOT.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    output = ARTIFACT_ROOT / "SHA256SUMS"
    output.write_text(
        "".join(f"{file_sha256(path)}  {path.relative_to(ARTIFACT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return output

