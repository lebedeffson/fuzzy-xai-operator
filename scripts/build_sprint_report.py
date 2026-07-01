#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "release" / "current"

SCENARIOS: dict[str, dict[str, Any]] = {
    "hybrid_xiris": {
        "action": "block",
        "diagnostic": "D_quality_source_conflict",
        "keys": {
            "gamma": 0.351,
            "delta": 0.106811,
            "r_delta": 0.3225,
            "rho": 0.800,
            "chi_crit": 1,
        },
    },
    "medical_ecg_signal": {
        "action": "defer_to_human",
        "diagnostic": "D_signal_quality",
        "keys": {},
    },
    "gd_anfis_shap": {
        "action": "audit",
        "diagnostic": "D_rule_attribution_conflict",
        "keys": {
            "alpha_rule": 0.82,
            "gamma_rule_shap": 0.685,
        },
    },
    "beacon_xai": {
        "action": "audit",
        "diagnostic": "D_counterevidence_conflict",
        "keys": {},
    },
    "gis_integro": {
        "action": "audit_report",
        "diagnostic": "D_route_context_limit",
        "keys": {
            "p": 0.67,
            "alpha_mean": 0.72,
            "s": 0.47,
            "gamma_route": 0.20,
        },
    },
}

SITE_FORBIDDEN = [
    r"\bimport\s+fuzzyxai\b",
    r"\bfrom\s+fuzzyxai\b",
    r"\bbuild_route\s*\(",
    r"\bbuild_proof_trace\s*\(",
    r"\bverify_proof_trace\s*\(",
    r"\brender_dashboard\s*\(",
    r"\bcompute_gamma\b",
    r"\bcompute_delta\b",
    r"\bcompute_rho\b",
]

APP_RUN_FORBIDDEN = [
    r"\bif\s+action\b",
    r"\baction_id\s*=",
    r"\bfinal_action\s*=",
    r"\bmake_action\s*\(",
    r"\bdiagnostic_id\s*=",
]

SOURCE_PREFIXES = (
    "framework/",
    "applications/",
    "site/",
    "registry/",
    "scripts/",
    "docs/",
    "Makefile",
    "DUBNAXAI_FULL_WORK_REPORT.md",
)
GENERATED_PREFIXES = (
    "reports/release/current/",
    "reports/audit/",
    "reports/practice",
    "reports/generated/",
    "site/dubnaxai/dist/",
)


EXTERNAL_SUMMARY = ROOT / "external_validation" / "outputs" / "external_wine_summary.json"
EXTERNAL_MODEL_KEYS = ("logistic_regression", "gradient_boosting")
EXTERNAL_ZIP = ROOT / "external_validation" / "outputs" / "external_wine_blackbox_validation.zip"


def status_path(line: str) -> str:
    if len(line) >= 3 and line[2] == " ":
        return line[3:]
    return line[2:].strip() if len(line) > 2 else line.strip()


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(scenario_id: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    return {
        "scenario_id": scenario_id,
        "path": path.relative_to(ROOT).as_posix(),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
        "sha256": sha256(path),
    }


def normalize_status(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("status")
    status = str(value or "").strip().lower()
    return "passed" if status == "pass" else status


def verifier_status(data: dict[str, Any]) -> str:
    for key in ("verification_status", "verifier_status"):
        if key in data:
            return normalize_status(data[key])
    if isinstance(data.get("verifier"), dict):
        return normalize_status(data["verifier"].get("status"))
    return ""


def diag_id(data: dict[str, Any]) -> str:
    computed = data.get("computed_result") or {}
    if computed.get("diagnostic_id"):
        return str(computed["diagnostic_id"])
    diagnostics = data.get("diagnostics") or []
    if diagnostics and isinstance(diagnostics[0], dict):
        return str(diagnostics[0].get("diagnostic_id", ""))
    return str(data.get("diagnostic_id") or data.get("diagnostic") or "")


def action_id(data: dict[str, Any]) -> str:
    computed = data.get("computed_result") or {}
    return str(
        data.get("action_id")
        or data.get("final_action")
        or data.get("action")
        or computed.get("action")
        or ""
    )


def find_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_value(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_value(item, key)
            if found is not None:
                return found
    return None


def nearly_equal(actual: Any, expected: Any) -> bool:
    try:
        return abs(float(actual) - float(expected)) < 1e-9
    except (TypeError, ValueError):
        return actual == expected


def scan_site() -> tuple[bool, list[str]]:
    issues: list[str] = []
    suffixes = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".json"}
    for path in (ROOT / "site" / "dubnaxai").rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.suffix not in suffixes:
            continue
        if any(part in rel for part in ("/node_modules/", "/dist/")):
            continue
        if path.name.endswith(".lock"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in SITE_FORBIDDEN:
            if re.search(pattern, text):
                issues.append(f"{rel}: forbidden pattern {pattern}")
    return bool(issues), issues


def scan_apps() -> tuple[bool, list[str]]:
    issues: list[str] = []
    for path in (ROOT / "applications" / "scenarios").glob("*/run.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(ROOT).as_posix()
        if "run_framework_scenario" not in text:
            issues.append(f"{rel}: does not call run_framework_scenario")
        for pattern in APP_RUN_FORBIDDEN:
            if re.search(pattern, text):
                issues.append(f"{rel}: forbidden action-selection pattern {pattern}")
    runner = ROOT / "applications" / "run_framework_scenario.py"
    if runner.exists():
        text = runner.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"\bif\s+scenario_id\b|\belif\s+scenario_id\b", text):
            issues.append("applications/run_framework_scenario.py: scenario-specific if/elif")
    return bool(issues), issues


def validate_external_framework() -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    required = [EXTERNAL_SUMMARY, EXTERNAL_ZIP]
    for model_key in EXTERNAL_MODEL_KEYS:
        required.extend(
            [
                ROOT / "external_validation" / "outputs" / f"external_wine_{model_key}_route.json",
                ROOT / "external_validation" / "outputs" / f"external_wine_{model_key}_proof_trace.json",
                ROOT / "external_validation" / "outputs" / f"external_wine_{model_key}_operator_dashboard.png",
                ROOT / "external_validation" / "outputs" / f"external_wine_{model_key}_summary.json",
            ]
        )
    exists = all(path.exists() and path.stat().st_size > 0 for path in required)
    summary: dict[str, Any] = {}
    if not exists:
        errors.append("external framework outputs are missing; run make framework-external-check")
    else:
        summary = load_json(EXTERNAL_SUMMARY)
        if summary.get("task") != "sklearn_wine_classification":
            errors.append(f"external task expected sklearn_wine_classification, got {summary.get('task')}")
        if summary.get("scenario_id") != "external_wine_classification":
            errors.append(f"external scenario_id expected external_wine_classification, got {summary.get('scenario_id')}")
        if summary.get("verifier") != "passed":
            errors.append(f"external verifier expected passed, got {summary.get('verifier')}")
        if not summary.get("source_commit"):
            errors.append("external source_commit is empty")
        validations = summary.get("validations") or []
        if len(validations) != 2:
            errors.append(f"external validations expected 2, got {len(validations)}")
        for item in validations:
            computed = item.get("computed_result") or {}
            if item.get("action") != "lower_confidence":
                errors.append(f"external {item.get('model_key')} action expected lower_confidence, got {item.get('action')}")
            if item.get("diagnostic") != "D_external_tabular_uncertainty":
                errors.append(f"external {item.get('model_key')} diagnostic expected D_external_tabular_uncertainty, got {item.get('diagnostic')}")
            for key in ("gamma", "delta", "rho"):
                value = float(computed.get(key, 0.0))
                if value <= 0.0:
                    errors.append(f"external {item.get('model_key')} {key} is zero")
            gamma = float(computed.get("gamma", 0.0))
            delta = float(computed.get("delta", 0.0))
            rho = float(computed.get("rho", 0.0))
            if not (0.10 <= gamma <= 0.60 and 0.05 <= delta <= 0.60 and 0.10 <= rho <= 0.70):
                errors.append(f"external {item.get('model_key')} values out of range: gamma={gamma}, delta={delta}, rho={rho}")
    return {
        "status": "FAIL" if errors else "PASS",
        "task": summary.get("task", ""),
        "scenario_id": summary.get("scenario_id", ""),
        "models": summary.get("models", []),
        "validations": summary.get("validations", []),
        "action": ",".join(item.get("action", "") for item in summary.get("validations", [])),
        "diagnostic": ",".join(item.get("diagnostic", "") for item in summary.get("validations", [])),
        "verifier": summary.get("verifier", ""),
        "source_commit": summary.get("source_commit", ""),
        "route_exists": all((ROOT / "external_validation" / "outputs" / f"external_wine_{key}_route.json").exists() for key in EXTERNAL_MODEL_KEYS),
        "proof_exists": all((ROOT / "external_validation" / "outputs" / f"external_wine_{key}_proof_trace.json").exists() for key in EXTERNAL_MODEL_KEYS),
        "dashboard_exists": all((ROOT / "external_validation" / "outputs" / f"external_wine_{key}_operator_dashboard.png").exists() for key in EXTERNAL_MODEL_KEYS),
    }, errors


def validate_scenarios() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    manifest = {
        "routes": [],
        "proofs": [],
        "dashboards": [],
        "site_payloads": [],
        "site_public_routes": [],
        "site_public_figures": [],
        "release_reports": [],
    }

    for scenario_id, spec in SCENARIOS.items():
        base = ROOT / "applications" / "scenarios" / scenario_id
        paths = {
            "route": base / "route" / "route.json",
            "proof": base / "proof" / "proof_trace.json",
            "dashboard": base / "figures" / "operator_dashboard.png",
            "payload": base / "site_payload" / "scenario.json",
            "site_route": ROOT / "site" / "dubnaxai" / "public" / "routes" / f"{scenario_id}_route.json",
            "site_dashboard": ROOT / "site" / "dubnaxai" / "public" / "figures" / f"{scenario_id}_operator_dashboard.png",
        }
        manifest["routes"].append(entry(scenario_id, paths["route"]))
        manifest["proofs"].append(entry(scenario_id, paths["proof"]))
        manifest["dashboards"].append(entry(scenario_id, paths["dashboard"]))
        manifest["site_payloads"].append(entry(scenario_id, paths["payload"]))
        manifest["site_public_routes"].append(entry(scenario_id, paths["site_route"]))
        manifest["site_public_figures"].append(entry(scenario_id, paths["site_dashboard"]))

        errors: list[str] = []
        warnings: list[str] = []
        for name, path in paths.items():
            if not path.exists():
                errors.append(f"missing {name}: {path.relative_to(ROOT)}")
            elif path.is_file() and path.stat().st_size == 0:
                errors.append(f"empty {name}: {path.relative_to(ROOT)}")
        if paths["dashboard"].exists() and paths["dashboard"].stat().st_size < 10_000:
            warnings.append("dashboard is smaller than recommended 10 KB")

        route: dict[str, Any] = {}
        proof: dict[str, Any] = {}
        payload: dict[str, Any] = {}
        if paths["route"].exists():
            route = load_json(paths["route"])
        if paths["proof"].exists():
            proof = load_json(paths["proof"])
        if paths["payload"].exists():
            payload = load_json(paths["payload"])

        actual_action = action_id(route)
        actual_diag = diag_id(route)
        proof_action = action_id(proof)
        proof_diag = diag_id(proof)
        payload_action = action_id(payload)
        payload_diag = diag_id(payload)
        status = verifier_status(proof)

        if route.get("scenario_id") != scenario_id:
            errors.append(f"route scenario_id expected {scenario_id}, got {route.get('scenario_id')}")
        if proof.get("scenario_id") != scenario_id:
            errors.append(f"proof scenario_id expected {scenario_id}, got {proof.get('scenario_id')}")
        if actual_action != spec["action"]:
            errors.append(f"action expected {spec['action']}, got {actual_action}")
        if actual_diag != spec["diagnostic"]:
            errors.append(f"diagnostic expected {spec['diagnostic']}, got {actual_diag}")
        if proof_action and proof_action != actual_action:
            errors.append(f"proof action mismatch: {proof_action} != {actual_action}")
        if proof_diag and proof_diag != actual_diag:
            errors.append(f"proof diagnostic mismatch: {proof_diag} != {actual_diag}")
        if payload_action and payload_action != actual_action:
            errors.append(f"payload action mismatch: {payload_action} != {actual_action}")
        if payload_diag and payload_diag != actual_diag:
            errors.append(f"payload diagnostic mismatch: {payload_diag} != {actual_diag}")
        if not payload_diag:
            errors.append("payload diagnostic_id is missing")
        if status != "passed":
            errors.append(f"verifier status expected PASS, got {status or 'missing'}")

        for field, rel_key in (("route", "route_path"), ("proof", "proof_path"), ("dashboard", "dashboard_path")):
            rel = payload.get(rel_key) or payload.get("proof_trace" if field == "proof" else "figure" if field == "dashboard" else field)
            if not rel:
                errors.append(f"payload missing {rel_key}")
            elif not (base / str(rel)).exists():
                errors.append(f"payload {rel_key} points to missing file: {rel}")

        route_result = route.get("computed_result") or {}
        proof_result = proof.get("computed_result") or {}
        for key in ("gamma", "gamma_route", "gamma_rule_shap", "delta", "rho", "chi_crit"):
            if key in route_result and key in proof_result and route_result[key] != proof_result[key]:
                errors.append(f"{key} route/proof mismatch: {route_result[key]} != {proof_result[key]}")
            elif key in route_result and key not in proof_result:
                warnings.append(f"{key} exists in route but not proof")

        for key, expected in spec["keys"].items():
            actual = find_value(route, key)
            result = "PASS" if nearly_equal(actual, expected) else "FAIL"
            if result == "FAIL":
                errors.append(f"{key} expected {expected}, got {actual}")
            key_rows.append(
                {
                    "scenario_id": scenario_id,
                    "key": key,
                    "expected": expected,
                    "actual": actual,
                    "result": result,
                }
            )

        if scenario_id == "gis_integro":
            p = find_value(route, "p")
            alpha = find_value(route, "alpha_mean")
            s = find_value(route, "s")
            gamma = find_value(route, "gamma_route")
            try:
                expected_gamma = max(abs(float(p) - float(alpha)), abs(float(p) - float(s)))
                if not nearly_equal(gamma, expected_gamma) or nearly_equal(gamma, 0.25):
                    errors.append(f"GIS gamma_route formula error: expected {expected_gamma}, got {gamma}")
            except (TypeError, ValueError):
                errors.append("GIS gamma_route formula values are incomplete")

        if scenario_id == "medical_ecg_signal":
            text = json.dumps(payload, ensure_ascii=False).lower()
            if "clinical diagnosis" in text or "клиническ" in text and "не" not in text:
                errors.append("ECG payload looks like clinical diagnosis claim")

        if scenario_id == "beacon_xai":
            text = json.dumps(route, ensure_ascii=False).lower()
            if "fuzzyxai improvement" in text or "результат fuzzyxai" in text:
                errors.append("BEACON 64->11 is claimed as FuzzyXAI result")

        rows.append(
            {
                "scenario_id": scenario_id,
                "expected_action_id": spec["action"],
                "actual_action_id": actual_action,
                "expected_diagnostic_id": spec["diagnostic"],
                "actual_diagnostic_id": actual_diag,
                "route_exists": paths["route"].exists(),
                "proof_exists": paths["proof"].exists(),
                "dashboard_exists": paths["dashboard"].exists(),
                "site_payload_exists": paths["payload"].exists(),
                "site_route_copy_exists": paths["site_route"].exists(),
                "site_dashboard_copy_exists": paths["site_dashboard"].exists(),
                "verifier_status": status,
                "status": "FAIL" if errors else "PASS",
                "warnings": warnings,
                "errors": errors,
            }
        )
    return rows, key_rows, manifest


def dirty_source_files(git_status: str) -> list[str]:
    dirty: list[str] = []
    for line in git_status.splitlines():
        path = status_path(line)
        if any(path.startswith(prefix) for prefix in GENERATED_PREFIXES):
            continue
        if path.endswith(".zip"):
            continue
        if path.startswith(SOURCE_PREFIXES):
            dirty.append(path)
    return dirty


def changed_areas(git_status: str) -> list[str]:
    areas = set()
    for line in git_status.splitlines():
        path = status_path(line)
        first = path.split("/", 1)[0]
        if first:
            areas.add(first)
    return sorted(areas)


def render_md(
    branch: str,
    commit: str,
    tag: str | None,
    rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    external: dict[str, Any],
    git_status: str,
    diff_summary: str,
) -> str:
    matrix = "\n".join(
        f"| {r['scenario_id']} | {r['actual_action_id']} | {r['actual_diagnostic_id']} | "
        f"{'yes' if r['route_exists'] else 'no'} | {'yes' if r['proof_exists'] else 'no'} | "
        f"{'yes' if r['dashboard_exists'] else 'no'} | {'yes' if r['site_payload_exists'] else 'no'} | "
        f"{r['verifier_status']} | {r['status']} |"
        for r in rows
    )
    keys = "\n".join(
        f"| {r['scenario_id']} | {r['key']} | {r['expected']} | {r['actual']} | {r['result']} |"
        for r in key_rows
    )
    checks = "\n".join(
        [
            "| fuzzyxai-framework-check | UNKNOWN |",
            f"| framework-external-check | {external['status']} |",
            "| operator-route-check | PASS |",
            "| dubnaxai-release-check | UNKNOWN |",
            f"| sprint-report | {summary['sprint_report_status']} |",
        ]
    )
    status_text = git_status if git_status.strip() else "clean"
    diff_text = diff_summary if diff_summary.strip() else "no diff"
    return f"""# Sprint Status

## Git

- branch: {branch}
- commit: {commit}
- tag: {tag or "none"}
- pushed: unknown

## Summary

DubnaXAI/FuzzyXAI has three separated layers: framework computes, applications run scenarios, site displays prepared artifacts.

## Changed Areas

{chr(10).join(f"- {area}" for area in changed_areas(git_status)) or "- none"}

## Checks

| Check | Result |
|---|---|
{checks}

## Scenario Matrix

| Scenario | Action | Diagnostic | Route | Proof | Dashboard | Site payload | Verifier | Status |
|---|---|---|---|---|---|---|---|---|
{matrix}

## Key Values

| Scenario | Key | Expected | Actual | Result |
|---|---|---:|---:|---|
{keys}

## Artifact Counts

- routes: {summary['routes_total']}
- proofs: {summary['proofs_total']}
- dashboards: {summary['dashboards_total']}
- site payloads: {summary['site_payloads_total']}
- site route copies: {summary['site_routes_total']}
- site dashboard copies: {summary['site_figures_total']}

## External Framework Validation

| Task | Result |
|---|---|
| import from /tmp | {external['status']} |
| package path | framework/fuzzyxai |
| external task | {external.get('task') or 'sklearn_wine_classification'} |
| models | {', '.join(external.get('models') or [])} |
| action | {external.get('action') or ''} |
| diagnostic | {external.get('diagnostic') or ''} |
| route | {'PASS' if external.get('route_exists') else 'FAIL'} |
| proof | {'PASS' if external.get('proof_exists') else 'FAIL'} |
| dashboard | {'PASS' if external.get('dashboard_exists') else 'FAIL'} |
| verifier | {external.get('verifier') or ''} |
| source_commit | {external.get('source_commit') or ''} |

## Site Separation

- site imports fuzzyxai: {'yes' if summary['site_computes_fuzzyxai'] else 'no'}
- site computes operator values: {'yes' if summary['site_computes_fuzzyxai'] else 'no'}

## Application Separation

- applications choose action directly: {'yes' if summary['applications_choose_actions_directly'] else 'no'}

## Dirty Working Tree

Dirty source files: {'yes' if summary['dirty_source_files'] else 'no'}

```text
{status_text}
```

Diff summary:

```text
{diff_text}
```

## Risks and Todos

See `risks_and_todos.md`.

## Next Step

{summary['next_step']}
"""


def risks_text() -> str:
    return """# Risks and TODOs

## Known Risks

- External research repositories are not yet pinned by commit.
- Adapter contracts for live external repository outputs are not fully implemented.
- Site ecosystem graph is not yet generated from registry.

## Technical TODO

- Add external payload schemas.
- Add adapter contract tests.
- Add repository-to-model/method/scenario links.

## Dissertation TODO

- Chapter 4: describe FuzzyXAI framework API and operator route.
- Chapter 5: describe DubnaXAI site as ecosystem.
- Chapter 6: describe five application scenarios with proof traces.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or "unknown"
    commit = run(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    tag = run(["git", "describe", "--tags", "--exact-match"]) or None
    git_status = run(["git", "status", "--short"])
    diff_summary = run(["git", "diff", "--stat"])

    rows, key_rows, manifest = validate_scenarios()
    external, external_errors = validate_external_framework()
    site_bad, site_issues = scan_site()
    apps_bad, app_issues = scan_apps()
    errors = [err for row in rows for err in row["errors"]] + external_errors + site_issues + app_issues
    warnings = [warn for row in rows for warn in row["warnings"]]
    dirty_sources = dirty_source_files(git_status)
    if dirty_sources:
        warnings.append(f"dirty source files: {', '.join(dirty_sources[:10])}")

    for report_name in (
        "SPRINT_STATUS.md",
        "release_summary.json",
        "check_results.json",
        "scenario_matrix.json",
        "artifact_manifest.json",
        "git_status.txt",
        "git_diff_summary.txt",
        "risks_and_todos.md",
    ):
        manifest["release_reports"].append(entry("release", OUT / report_name))

    summary = {
        "branch": branch,
        "commit": commit,
        "tag": tag,
        "sprint_report_status": "FAIL" if errors else "PASS",
        "framework_check": "UNKNOWN",
        "framework_external_check": external["status"],
        "external_task": external.get("task") or "sklearn_wine_classification",
        "external_task_action": external.get("action"),
        "external_task_verifier": external.get("verifier"),
        "external_task_models": external.get("models"),
        "package_boundary_ok": external["status"] == "PASS",
        "source_commit_filled": bool(external.get("source_commit")),
        "operator_route_check": "PASS" if not errors else "FAIL",
        "dubnaxai_release_check": "UNKNOWN",
        "scenarios_total": len(rows),
        "scenarios_passed": sum(1 for row in rows if row["status"] == "PASS"),
        "routes_total": sum(1 for item in manifest["routes"] if item["exists"]),
        "proofs_total": sum(1 for item in manifest["proofs"] if item["exists"]),
        "dashboards_total": sum(1 for item in manifest["dashboards"] if item["exists"]),
        "site_payloads_total": sum(1 for item in manifest["site_payloads"] if item["exists"]),
        "site_routes_total": sum(1 for item in manifest["site_public_routes"] if item["exists"]),
        "site_figures_total": sum(1 for item in manifest["site_public_figures"] if item["exists"]),
        "site_computes_fuzzyxai": site_bad,
        "applications_choose_actions_directly": apps_bad,
        "dirty_worktree": bool(git_status.strip()),
        "dirty_source_files": bool(dirty_sources),
        "warnings": warnings,
        "errors": errors,
        "next_step": "external payload schemas and adapter contracts",
    }

    check_results = {
        "fuzzyxai_framework_check": {
            "status": "UNKNOWN",
            "source": "not executed by sprint-report; run by dubnaxai-release-check before this target",
        },
        "framework_external_check": {
            "status": external["status"],
            "source": "external_validation/outputs/external_wine_summary.json",
        },
        "operator_route_check": {
            "status": summary["operator_route_check"],
            "source": "sprint-report artifact validation",
        },
        "dubnaxai_release_check": {
            "status": "UNKNOWN",
            "source": "not executed by sprint-report",
        },
        "sprint_report": {
            "status": summary["sprint_report_status"],
            "errors": errors,
            "warnings": warnings,
        },
    }

    (OUT / "git_status.txt").write_text(git_status + ("\n" if git_status else ""), encoding="utf-8")
    (OUT / "git_diff_summary.txt").write_text(diff_summary + ("\n" if diff_summary else ""), encoding="utf-8")
    (OUT / "risks_and_todos.md").write_text(risks_text(), encoding="utf-8")
    dump_json(OUT / "scenario_matrix.json", rows)
    dump_json(OUT / "artifact_manifest.json", manifest)
    dump_json(OUT / "release_summary.json", summary)
    dump_json(OUT / "check_results.json", check_results)
    (OUT / "SPRINT_STATUS.md").write_text(
        render_md(branch, commit, tag, rows, key_rows, summary, external, git_status, diff_summary),
        encoding="utf-8",
    )

    print("DubnaXAI Sprint Report")
    print(f"commit: {commit}")
    print(f"tag: {tag or 'none'}")
    print("\nchecks:")
    print(f"  sprint-report: {summary['sprint_report_status']}")
    print(f"  external framework check: {external['status']}")
    print("\nscenarios:")
    for row in rows:
        print(
            f"  {row['scenario_id']}: {row['actual_action_id']} / "
            f"{row['actual_diagnostic_id']} / verifier {row['verifier_status']}"
        )
    print("\nartifacts:")
    print(f"  routes: {summary['routes_total']}")
    print(f"  proofs: {summary['proofs_total']}")
    print(f"  dashboards: {summary['dashboards_total']}")
    print(f"  site payloads: {summary['site_payloads_total']}")
    print("\nseparation:")
    print(f"  site computes FuzzyXAI: {'yes' if site_bad else 'no'}")
    print(f"  applications choose actions directly: {'yes' if apps_bad else 'no'}")
    print("\ndirty:")
    print(f"  generated/source changes present: {'yes' if git_status.strip() else 'no'}")
    print(f"  dirty source files: {'yes' if dirty_sources else 'no'}")
    print(f"\nresult: {summary['sprint_report_status']}")
    if errors:
        print("\nerrors:")
        for error in errors:
            print(f"  - {error}")
    if warnings:
        print("\nwarnings:")
        for warning in warnings[:20]:
            print(f"  - {warning}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
