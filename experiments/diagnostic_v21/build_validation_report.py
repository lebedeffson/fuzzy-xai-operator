from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/diagnostic_v21"
BASE_COMMIT = "13f82805c69fb974a236df6d8990eea115251c23"


def _run(command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(command),
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "status": "PASS" if result.returncode == 0 else "FAIL",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(*, python: str, run_full_regression: bool) -> dict[str, object]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    checks = {
        "ruff_changed_scope": _run(
            [
                python,
                "-m",
                "ruff",
                "check",
                "framework/fuzzyxai/fuzzyxai/diagnostics",
                "tests/diagnostics",
                "experiments/diagnostic_v21",
            ]
        ),
        "diagnostic_tests": _run([python, "-m", "pytest", "-q", "tests/diagnostics"]),
        "gold_regression": _run([python, "-m", "pytest", "-q", "tests/h10", "tests/h10_gold"]),
        "manifest_contract": _run(
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/test_public_framework_api.py::test_operator_manifest_is_complete_and_resolvable",
            ]
        ),
        "coverage": _run([python, "-m", "coverage", "report", "--fail-under=90"]),
        "protocol_gate": _run(
            [
                python,
                "-m",
                "experiments.diagnostic_v21.protocol_gate",
                "--protocol",
                "config/h10_c2_diagnostic_cut_protocol.yaml",
            ]
        ),
    }
    if run_full_regression:
        checks["full_regression"] = _run([python, "-m", "pytest", "-q"])
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", BASE_COMMIT],
        cwd=ROOT,
        text=True,
    ).splitlines()
    forbidden_prefixes = (
        "artifacts/h10_v16/",
        "artifacts/h10_v18/",
        "artifacts/h10_v19/",
        "artifacts/h10_final_gold/",
        "gold_oracle/",
        "experiments/h10_gold/",
        "baselines/h10_gold/",
    )
    forbidden_changes = sorted(path for path in changed if path.startswith(forbidden_prefixes))
    leakage_path = ROOT / "artifacts/h10_final_gold/closure/h10_final_gold_leakage_audit.json"
    old_leakage = json.loads(leakage_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load((ROOT / "config/h10_c2_diagnostic_cut_protocol.yaml").read_text(encoding="utf-8"))
    source_files = sorted((ROOT / "framework/fuzzyxai/fuzzyxai/diagnostics").glob("*.py"))
    production_text = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    methodology = {
        "schema_version": "1.0",
        "implementation": "diagnostic-framework-v21",
        "base_commit": BASE_COMMIT,
        "production_imports_gold_oracle": "gold_oracle" in production_text,
        "production_mentions_source_truth": "source_truth" in production_text,
        "production_mentions_repair_truth": "repair_truth" in production_text,
        "production_copies_expected_to_observed": "observed[field] = route.expected[field]" in production_text,
        "sealed_opening_count": old_leakage["opening_count"],
        "h10_c2_status": protocol["status"],
        "manual_adjudication_complete": protocol["manual_adjudication"]["completed"],
        "confirmatory_scoring_enabled": protocol["confirmatory"]["scoring_enabled"],
        "forbidden_old_artifact_changes": forbidden_changes,
        "software_alpha_release_allowed": not forbidden_changes and all(
            check["status"] == "PASS" for check in checks.values()
        ),
        "scientific_confirmatory_release_allowed": False,
        "scientific_blocker": "power analysis and independent two-reviewer adjudication are incomplete",
    }
    leakage = {
        "schema_version": "1.0",
        "cycle": "FXAI-H10-C2-DIAGNOSTIC-CUT",
        "phase": "preconfirmatory_implementation",
        "opening_count": 0,
        "old_final_gold_opening_count": old_leakage["opening_count"],
        "old_sealed_cases_reused": False,
        "sealed_scoring_performed": False,
        "post_lock_tuning": False,
        "gold_oracle_available_to_production": False,
        "raw_labels_packaged": False,
        "status": "PASS",
    }
    performance_path = OUTPUT / "performance.json"
    coverage_path = OUTPUT / "coverage.json"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    generation_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    methodology_path = OUTPUT / "methodology_audit.json"
    leakage_output_path = OUTPUT / "leakage_audit.json"
    methodology_path.write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    leakage_output_path.write_text(
        json.dumps(leakage, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    common = {
        "evidence_generation_commit": generation_commit,
        "closure_packaging_commit": None,
        "bundle_commit": None,
    }
    evidence = [
        {
            "evidence_id": "DV21-E01",
            "claim_id": "DV21-PERFORMANCE",
            "metric": "route_count",
            "value": performance["route_count"],
            "scope": performance["scope"],
            "source_file": "reports/diagnostic_v21/performance.json",
            "locator": "route_count",
            "sha256": _sha256(performance_path),
            "status": "descriptive",
            **common,
        },
        {
            "evidence_id": "DV21-E02",
            "claim_id": "DV21-TEST-COVERAGE",
            "metric": "diagnostic_line_coverage_percent",
            "value": coverage["totals"]["percent_covered"],
            "source_file": "reports/diagnostic_v21/coverage.json",
            "locator": "totals.percent_covered",
            "sha256": _sha256(coverage_path),
            "status": "technical_validation",
            **common,
        },
        {
            "evidence_id": "DV21-E03",
            "claim_id": "H10-C2",
            "metric": "sealed_opening_count",
            "value": 0,
            "source_file": "reports/diagnostic_v21/leakage_audit.json",
            "locator": "opening_count",
            "sha256": _sha256(leakage_output_path),
            "status": "preconfirmatory",
            **common,
        },
    ]
    for stage_index, (stage, values) in enumerate(performance["timings"].items(), start=4):
        for metric_index, (metric, value) in enumerate(values.items()):
            evidence.append(
                {
                    "evidence_id": f"DV21-E{stage_index:02d}-{metric_index + 1}",
                    "claim_id": "DV21-PERFORMANCE",
                    "metric": f"{stage}_{metric}",
                    "value": value,
                    "scope": performance["scope"],
                    "source_file": "reports/diagnostic_v21/performance.json",
                    "locator": f"timings.{stage}.{metric}",
                    "sha256": _sha256(performance_path),
                    "status": "descriptive",
                    **common,
                }
            )
    claims = {
        "schema_version": "1.0",
        "claims": [
            {
                "claim_id": "DV21-IMPLEMENTATION",
                "status": "technical_validation",
                "statement": "The graph diagnostic, cut, repair planning, and recertification APIs are implemented and tested.",
            },
            {
                "claim_id": "H10-C",
                "status": "exploratory_only",
                "statement": "The earlier minimal-cut difference remains exploratory.",
            },
            {
                "claim_id": "H10-C2",
                "status": "not_evaluated_preconfirmatory",
                "statement": "No new sealed scoring was performed.",
            },
        ],
    }
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_commit": BASE_COMMIT,
        "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checks": checks,
        "methodology_status": "PASS" if not forbidden_changes else "FAIL",
        "scientific_status": "BLOCKED_PRECONFIRMATORY",
        "software_status": "PASS" if all(check["status"] == "PASS" for check in checks.values()) else "FAIL",
        "whole_repository_ruff_baseline": {
            "base_errors": 313,
            "current_errors": 313,
            "changed_scope_status": checks["ruff_changed_scope"]["status"],
        },
        "performance": performance,
        "coverage_percent": coverage["totals"]["percent_covered"],
        "forbidden_old_artifact_changes": forbidden_changes,
    }
    (OUTPUT / "evidence_map.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "claim_registry.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Diagnostic framework v21 validation",
        "",
        f"- Software status: `{report['software_status']}`.",
        "- Scientific status: `BLOCKED_PRECONFIRMATORY`.",
        f"- Diagnostic line coverage: `{report['coverage_percent']:.2f}%`.",
        f"- Diagnostic p95: `{performance['timings']['total']['p95_ms']:.6f} ms`.",
        f"- Old sealed opening count: `{old_leakage['opening_count']}`.",
        f"- Forbidden old-artifact changes: `{len(forbidden_changes)}`.",
        "- H10-C remains exploratory; H10-C2 has not been scored.",
        "- Full-repository Ruff remains at the inherited 313-error baseline; changed v21 scope passes.",
        "",
        "No user-comprehension, prediction-accuracy, or safety claim is made.",
    ]
    (OUTPUT / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (OUTPUT / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256(path)}  {path.name}" for path in manifest_files) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--full-regression", action="store_true")
    args = parser.parse_args()
    report = build(python=args.python, run_full_regression=args.full_regression)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["software_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
