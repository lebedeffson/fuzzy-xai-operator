from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
import re
import subprocess

import pandas as pd

from .common import ARTIFACT_ROOT, ROOT, environment_manifest, read_json, sha256_file, write_json


FORBIDDEN_IMPORTS = {"TypedRouteGuard", "RepairPlanner", "RepairSetPlanner", "DiagnosticCutSolver", "H10Auditor"}
ORACLE_FORBIDDEN_IMPORTS = FORBIDDEN_IMPORTS | {"FaultFamilyClassifier", "SourceLocalizer", "SPEC_BY_LEAF", "FIELD_TO_SPECS"}
PROTECTED_REFS = {
    "old_v16": ("docs/chapter4-operational-audit-v16", "b4af116f02d8c062420a61ed586729119314d4ed"),
    "old_v18": ("feat/h10-audit-confirmatory-v18", "59a1f45a44c40c871ea955f426a4a31b35dfe85f"),
}


def baseline_independence() -> list[str]:
    failures = []
    for path in (ROOT / "baselines" / "h10").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                if names & FORBIDDEN_IMPORTS or (node.module or "").startswith("fuzzyxai.audit_h10"):
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return failures



def oracle_independence() -> list[str]:
    path = ROOT / "experiments" / "h10" / "oracle_v19.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = {alias.name for alias in node.names}
            module = node.module or ""
            if names & ORACLE_FORBIDDEN_IMPORTS or module.startswith("fuzzyxai.audit_h10") or module.startswith("baselines.h10"):
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("fuzzyxai.audit_h10") or alias.name.startswith("baselines.h10"):
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    return failures


def protected_ref_status() -> dict[str, bool | None]:
    status: dict[str, bool | None] = {}
    for name, (ref, expected) in PROTECTED_REFS.items():
        result = subprocess.run(
            ["git", "rev-parse", "--verify", ref],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        status[f"{name}_changed"] = None if result.returncode else result.stdout.strip() != expected
    return status


def _claim_for_cell(table: Path, row: pd.Series, column: str) -> str:
    explicit = row.get("claim")
    if isinstance(explicit, str) and explicit.startswith("H10-"):
        return explicit
    if "source" in column:
        return "H10-L"
    if "repair" in column:
        return "H10-R"
    if "cut" in column or "cost_ratio" in column or "extra_nodes" in column:
        return "H10-C"
    if "unknown" in column or "abstention" in column:
        return "H10-U"
    if "trace" in column:
        return "H10-T"
    return "H10-secondary"

def build_evidence_map() -> list[dict]:
    opening = read_json(ARTIFACT_ROOT / "opening" / "opening_record.json")
    closure_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    entries = []
    methodology_path = ARTIFACT_ROOT / "closure" / "confirmatory_methodology_audit.json"
    methodology_invalid = methodology_path.exists() and read_json(methodology_path)["status"] != "PASS"
    for table in sorted((ARTIFACT_ROOT / "tables").glob("*.csv")):
        frame = pd.read_csv(table)
        digest = sha256_file(table)
        for row_index, row in frame.iterrows():
            for column, value in row.items():
                if isinstance(value, (int, float)) and pd.notna(value):
                    claim = _claim_for_cell(table, row, column)
                    entries.append(
                        {
                            "claim_id": claim,
                            "metric": column,
                            "value": float(value),
                            "dataset": row.get("dataset") if "dataset" in row else None,
                            "method": row.get("method") if "method" in row else None,
                            "source_file": str(table.relative_to(ROOT)),
                            "locator": f"row={row_index + 2},column={column}",
                            "sha256": digest,
                            "evidence_generation_commit": opening["commit"],
                            "closure_packaging_commit": closure_commit,
                            "bundle_commit": closure_commit,
                            "status": (
                                "invalid_confirmatory_cycle"
                                if methodology_invalid
                                else "confirmatory" if claim in {"H10-L", "H10-R"} else "descriptive_or_secondary"
                            ),
                        }
                    )
    write_json(ARTIFACT_ROOT / "closure" / "h10_v19_evidence_map.json", entries)
    return entries


def validate_evidence_entries(entries: list[dict]) -> list[str]:
    failures: list[str] = []
    locator_pattern = re.compile(r"^row=(\d+),column=(.+)$")
    for index, entry in enumerate(entries):
        source = ROOT / entry["source_file"]
        if not source.is_file():
            failures.append(f"entry={index}:missing:{entry['source_file']}")
            continue
        if sha256_file(source) != entry["sha256"]:
            failures.append(f"entry={index}:sha256:{entry['source_file']}")
        match = locator_pattern.match(str(entry["locator"]))
        if not match:
            failures.append(f"entry={index}:locator:{entry['locator']}")
            continue
        row_number, column = int(match.group(1)), match.group(2)
        with source.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        data_index = row_number - 2
        if data_index < 0 or data_index >= len(rows) or column not in rows[data_index]:
            failures.append(f"entry={index}:locator_out_of_range:{entry['locator']}")
            continue
        try:
            actual = float(rows[data_index][column])
        except (TypeError, ValueError):
            failures.append(f"entry={index}:non_numeric_locator:{entry['locator']}")
            continue
        if abs(actual - float(entry["value"])) > 1e-12:
            failures.append(f"entry={index}:value:{actual}!={entry['value']}")
        if entry["status"] == "confirmatory" and entry["claim_id"] not in {"H10-L", "H10-R"}:
            failures.append(f"entry={index}:unexpected_confirmatory_claim:{entry['claim_id']}")
    return failures


def validate() -> dict:
    required = (
        ARTIFACT_ROOT / "lock" / "protocol_lock.json",
        ARTIFACT_ROOT / "opening" / "pre_opening_leakage_audit.json",
        ARTIFACT_ROOT / "opening" / "opening_record.json",
        ARTIFACT_ROOT / "opening" / "post_scoring_leakage_audit.json",
        ARTIFACT_ROOT / "opening" / "completion_marker.json",
        ARTIFACT_ROOT / "confirmatory" / "raw_results.csv",
        ARTIFACT_ROOT / "confirmatory" / "statistical_tests.json",
        ARTIFACT_ROOT / "closure" / "h10_v19_claim_registry.json",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    independence = baseline_independence()
    oracle_failures = oracle_independence()
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    forbidden_tracked = [path for path in tracked if ".h10_private" in path or "label_vault" in path or path.endswith(".key")]
    evidence = build_evidence_map()
    evidence_failures = validate_evidence_entries(evidence)
    table_count = len(list((ARTIFACT_ROOT / "tables").glob("*.csv")))
    png_count = len(list((ARTIFACT_ROOT / "figures").glob("*.png")))
    pdf_count = len(list((ARTIFACT_ROOT / "figures").glob("*.pdf")))
    technical_pass = not missing and not independence and not oracle_failures and not forbidden_tracked and not evidence_failures and len(evidence) >= 188 and table_count >= 6 and png_count >= 10 and pdf_count >= 10
    methodology = read_json(ARTIFACT_ROOT / "closure" / "confirmatory_methodology_audit.json")
    protected = protected_ref_status()
    report = {
        "status": "TECHNICAL_PASS_SCIENTIFIC_INVALID" if technical_pass and methodology["status"] != "PASS" else "PASS" if technical_pass else "FAIL",
        "missing": missing,
        "baseline_independence_failures": independence,
        "oracle_independence_failures": oracle_failures,
        "forbidden_tracked_files": forbidden_tracked,
        "tables_csv": table_count,
        "figures_png": png_count,
        "figures_pdf": pdf_count,
        "evidence_entries": len(evidence),
        "evidence_integrity_failures": evidence_failures,
        "opening_count": read_json(ARTIFACT_ROOT / "opening" / "opening_record.json")["opening_count"],
        "post_lock_tuning": read_json(ARTIFACT_ROOT / "opening" / "post_scoring_leakage_audit.json")["post_lock_tuning"],
        **protected,
        "scientific_release_allowed": False if methodology["status"] != "PASS" else True,
        "methodology_audit_status": methodology["status"],
        "environment": environment_manifest(),
    }
    write_json(ARTIFACT_ROOT / "closure" / "validation_report.json", report)
    write_json(
        ARTIFACT_ROOT / "closure" / "h10_v19_leakage_audit.json",
        {
            "pre_opening": read_json(ARTIFACT_ROOT / "opening" / "pre_opening_leakage_audit.json"),
            "opening": read_json(ARTIFACT_ROOT / "opening" / "opening_record.json"),
            "post_scoring": read_json(ARTIFACT_ROOT / "opening" / "post_scoring_leakage_audit.json"),
            "methodology_invalidation_is_not_post_lock_tuning": True,
            "repeat_scoring_forbidden": True,
        },
    )
    lines = ["# H10 v19 validation report", "", f"- Status: `{report['status']}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in report.items() if key not in {"status", "environment"})
    (ARTIFACT_ROOT / "closure" / "h10_v19_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not technical_pass:
        raise RuntimeError(f"H10 evidence validation failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evidence", "validate"), default="validate", nargs="?")
    args = parser.parse_args()
    if args.command == "evidence":
        build_evidence_map()
    else:
        validate()


if __name__ == "__main__":
    main()
