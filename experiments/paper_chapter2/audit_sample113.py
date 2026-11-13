"""Forensic consistency audit of chapter-2 sample_113.

No model is trained, and no archived report is rewritten. The historical
record is expected to FAIL the claimed ExplainPlan conformance check.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _addends(expr: ast.expr) -> list[ast.expr]:
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return _addends(expr.left) + _addends(expr.right)
    return [expr]


def _executor_weights(source: str) -> dict[str, float]:
    tree = ast.parse(source)
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef) or function.name != "generate_case":
            continue
        for assignment in ast.walk(function):
            if (
                not isinstance(assignment, ast.Assign)
                or not any(isinstance(target, ast.Name) and target.id == "u_m" for target in assignment.targets)
            ):
                continue
            if not isinstance(assignment.value, ast.Call) or not assignment.value.args:
                raise ValueError("Unexpected aggregation call in generator")
            pairs = {}
            for term in _addends(assignment.value.args[0]):
                if not isinstance(term, ast.BinOp) or not isinstance(term.op, ast.Mult):
                    raise ValueError("Unexpected term in uncertainty aggregation")
                lhs, rhs = term.left, term.right
                if isinstance(lhs, ast.Constant) and isinstance(rhs, ast.Name):
                    pairs[rhs.id] = float(lhs.value)
                elif isinstance(rhs, ast.Constant) and isinstance(lhs, ast.Name):
                    pairs[lhs.id] = float(rhs.value)
                else:
                    raise ValueError("Unexpected variable coefficient")
            if set(pairs) != {"u_model", "u_rules", "u_trace"}:
                raise ValueError("Unexpected uncertainty components")
            return pairs
    raise ValueError("No u_m aggregation found in generator")


def _triangular(x: float, points: list[float]) -> float:
    a, b, c = map(float, points)
    if x < a or x > c:
        return 0.0
    if x == b or (a == b and x == a) or (b == c and x == c):
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def audit(repo_root: Path = ROOT) -> dict:
    report_path = repo_root / "reports/chapter2/sample_113_report.json"
    plan_path = repo_root / "configs/explain_plan_chapter2.yaml"
    generator_path = repo_root / "experiments/chapter2_real_operator_case.py"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    generator = generator_path.read_text(encoding="utf-8")

    plan_sha = hashlib.sha256(
        json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    actual_weights = _executor_weights(generator)
    stated_weights = {key: float(value) for key, value in plan["uncertainty"]["weights"].items()}
    u = {"model": report["U_model"], "rules": report["U_rules"], "trace": report["U_trace"]}
    expected_source = sum(actual_weights["u_" + name] * u[name] for name in u)
    expected_plan = sum(stated_weights[name] * u[name] for name in u)
    observed = float(report["u_M"])

    membership = plan["terms"]["risk"]["membership"]["nodes"]
    p = float(report["P(malignant)"])
    source_mu = {key: float(report["mu_" + key]) for key in ("low", "medium", "high")}
    plan_mu = {key: _triangular(p, membership[key]) for key in ("low", "medium", "high")}
    trace_missing = sorted(set(plan["trace_required"]) - set(report["tau_fields"]))

    checks = {
        "hash_matches_stated_yaml": plan_sha == report["explain_plan_hash"],
        "source_weights_reproduce_archived_uncertainty": abs(expected_source - observed) < 1e-12,
        "stated_yaml_weights_reproduce_archived_uncertainty": abs(expected_plan - observed) < 1e-12,
        "stated_yaml_memberships_match_archive": all(abs(plan_mu[k] - source_mu[k]) < 1e-12 for k in source_mu),
        "stated_yaml_required_trace_fields_exist": not trace_missing,
    }
    return {
        "schema": "chapter2_sample113_forensic_contract_v1",
        "id": report["sample_id"],
        "plan_hash": plan_sha,
        "checks": checks,
        "contract_conformant": all(checks.values()),
        "uncertainty": {
            "observed": observed,
            "executed_source": expected_source,
            "stated_yaml": expected_plan,
            "plan_discrepancy": expected_plan - observed,
        },
        "weights": {"source": actual_weights, "yaml": stated_weights},
        "risk_membership": {"archive": source_mu, "yaml": plan_mu},
        "missing_trace_fields": trace_missing,
        "source_paths": [
            "reports/chapter2/sample_113_report.json",
            "configs/explain_plan_chapter2.yaml",
            "experiments/chapter2_real_operator_case.py",
        ],
        "limitations": [
            "No new fitting, inference or source model score validation",
            "Plan SHA verifies specified YAML bytes, not whether execution obeyed it",
            "Original report remains unchanged; an expected failed conformance is a valid forensic result",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, sort_keys=True, ensure_ascii=False))
