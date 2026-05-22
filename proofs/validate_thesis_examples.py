from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fuzzyxai import (
    ExplainPlan,
    HesitantFS,
    IntervalFS,
    MultiLevelFS,
    NeutrosophicFS,
    Rule,
    SystemOperator,
    Trace,
    build_profile,
    compose,
    interpretability_index,
    interpretability_loss,
    semantic_disagreement,
    select_minimal_sufficient,
)
from fuzzyxai.core.trust_evaluator import entropy_from_memberships
from fuzzyxai.selection.compatibility import synthesize_levels
from fuzzyxai.selection.pareto_selector import Candidate

OUT = ROOT / "reports"
OUT.mkdir(exist_ok=True)


@dataclass(frozen=True)
class CheckResult:
    name: str
    expected: Any
    actual: Any
    tolerance: float | None
    passed: bool
    note: str = ""


def _round(x: Any, digits: int = 6) -> Any:
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: _round(v, digits) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, digits) for v in x]
    return x


def _check_close(name: str, actual: float, expected: float, tolerance: float = 1e-6, note: str = "") -> CheckResult:
    return CheckResult(
        name=name,
        expected=expected,
        actual=float(actual),
        tolerance=tolerance,
        passed=abs(float(actual) - expected) <= tolerance,
        note=note,
    )


def _check_equal(name: str, actual: Any, expected: Any, note: str = "") -> CheckResult:
    return CheckResult(
        name=name,
        expected=expected,
        actual=actual,
        tolerance=None,
        passed=actual == expected,
        note=note,
    )


def _candidates() -> List[Candidate]:
    return [
        Candidate("F0", {"u_num", "u_ling"}, 0.10, 0.03, 0.42),
        Candidate("FI", {"u_num", "u_ling", "u_int"}, 0.22, 0.05, 0.30),
        Candidate("FH", {"u_num", "u_ling", "u_exp"}, 0.30, 0.81, 0.25),
        Candidate("FNsrc", {"u_num", "u_ling", "u_if", "u_conf", "u_trace"}, 0.42, 0.09, 0.18),
        Candidate("FML-user", {"u_num", "u_ling", "u_int", "u_exp", "u_conf", "u_multi"}, 0.60, 0.99, 0.08),
        Candidate("FML-audit", {"u_num", "u_ling", "u_int", "u_exp", "u_conf", "u_trace", "u_multi"}, 0.68, 1.00, 0.04),
    ]


def validate_chapter2() -> Dict[str, Any]:
    """Validate numerical example used for chapter 2.

    The checked route is: z -> memberships -> H -> d_E -> composition -> L -> I -> D_ij.
    """
    plan = ExplainPlan().with_reduction_weight(0.10)
    op = SystemOperator(plan)

    rules_model = [
        Rule("r_high_check", {"risk": "high"}, "additional_check"),
        Rule("r_medium_watch", {"risk": "medium"}, "watch"),
    ]
    rules_decision = [Rule("r_decision_high", {"risk": "high"}, "send_to_check")]

    e_model = op.explain_scalar_risk(
        0.72,
        rules_model,
        Trace("case-001-model", "v1", "2026-05-22T10:00:00", source="demo", checksum="m1"),
        model_uncertainty=0.08,
        trace_uncertainty=0.02,
    )
    e_decision = op.explain_scalar_risk(
        0.74,
        rules_decision,
        Trace("case-001-decision", "v1", "2026-05-22T10:00:01", source="demo", checksum="d1"),
        model_uncertainty=0.10,
        trace_uncertainty=0.01,
    )

    mu = e_model.metadata["memberships"]
    H = entropy_from_memberships([mu["low"], mu["medium"], mu["high"]], epsilon=plan.epsilon)
    gamma = semantic_disagreement(e_model, e_decision, plan.beta)
    composed = compose(e_model, e_decision, plan.beta)
    assert hasattr(composed, "uncertainty"), composed
    L = interpretability_loss(
        H=H,
        C=0.40,
        O=0.18,
        K=0.05,
        U=composed.uncertainty,
        weights=plan.lambda_,
        reduction_loss=composed.reduction_loss,
        lambda_delta=0.10,
    )
    I = interpretability_index(L)

    diagnostic = compose(e_model, e_decision.copy_with(terms={"allow", "deny"}), plan.beta)

    checks = [
        _check_close("chapter2.risk", e_model.metadata["risk"], 0.72),
        _check_close("chapter2.mu_low", mu["low"], 0.0),
        _check_close("chapter2.mu_medium", mu["medium"], 0.12, 1e-9),
        _check_close("chapter2.mu_high", mu["high"], 0.88, 1e-9),
        _check_close("chapter2.entropy_H", H, 0.342207, 5e-6),
        _check_close("chapter2.gamma_d_E", gamma, 0.36108, 5e-6),
        _check_close("chapter2.composed_uncertainty", composed.uncertainty, 0.449813, 5e-6),
        _check_close("chapter2.loss_L", L, 0.284404, 5e-6),
        _check_close("chapter2.index_I", I, 0.752463, 5e-6),
        _check_equal("chapter2.diagnostic_code", getattr(diagnostic, "code", None), "D_ij"),
    ]

    return {
        "checks": [c.__dict__ for c in checks],
        "computed": {
            "risk": e_model.metadata["risk"],
            "memberships": _round(mu),
            "entropy_H": round(H, 6),
            "gamma_d_E": round(gamma, 6),
            "composed_uncertainty": round(composed.uncertainty, 6),
            "loss_L": round(L, 6),
            "index_I": round(I, 6),
            "diagnostic_code": getattr(diagnostic, "code", None),
        },
    }


def validate_chapter3() -> Dict[str, Any]:
    """Validate numerical example used for chapter 3.

    The checked route is: metadata -> P_sit -> F* -> A_k^F -> Delta -> E_ext -> d_E_ext -> D_choice.
    """
    metadata = {
        "has_intervals": True,
        "num_experts": 2,
        "source_conflict": True,
        "requires_audit": True,
        "multi_level": True,
    }
    profile = build_profile(metadata)
    selected = select_minimal_sufficient(profile, _candidates(), mode="audit")

    interval = IntervalFS(lambda x: 0.68, lambda x: 0.76, policy="mid")
    hesitant = HesitantFS(lambda x: [0.61, 0.78])
    neutro = NeutrosophicFS(
        lambda x: 0.78,
        lambda x: 0.20,
        lambda x: 0.64,
        source_t="risk_model",
        source_f="expert_panel",
    )
    multi = MultiLevelFS(
        [interval, hesitant, neutro],
        gamma={("interval", "hesitant", "same_case"), ("neutrosophic", "source", "conflict")},
        weights=[0.25, 0.25, 0.50],
    )

    _, d_interval = interval.reduce_to_f0()
    _, d_hesitant = hesitant.reduce_to_f0()
    _, d_neutro = neutro.reduce_to_f0()
    _, d_multi = multi.reduce_to_f0()

    plan = ExplainPlan().with_reduction_weight(0.10)
    op = SystemOperator(plan)
    rules = [Rule("r_high_check", {"risk": "high"}, "additional_check")]
    e_ext = op.explain_scalar_risk(
        0.72,
        rules,
        Trace("case-001-ext", "v1", "2026-05-22T10:00:00", source="demo", checksum="ext"),
    )
    e_ext.representation = multi
    e_ext.reduction_loss = d_multi

    e_simple = op.explain_scalar_risk(
        0.74,
        rules,
        Trace("case-001-simple", "v1", "2026-05-22T10:00:01", source="demo", checksum="simple"),
    )
    d_ext = semantic_disagreement(e_ext, e_simple, plan.beta)
    composed = compose(e_ext, e_simple, plan.beta)

    diag = select_minimal_sufficient(set(profile) | {"u_ethical"}, _candidates(), mode="audit")
    temporal_cf_levels = synthesize_levels({"u_num", "u_ling", "u_time", "u_cf", "u_trace"})

    expected_profile = ["u_conf", "u_exp", "u_int", "u_ling", "u_multi", "u_num", "u_trace"]
    checks = [
        _check_equal("chapter3.profile", sorted(profile), expected_profile),
        _check_equal("chapter3.selected_class", getattr(selected, "name", None), "FML-audit"),
        _check_close("chapter3.delta_interval", d_interval, 0.04, 1e-9),
        _check_close("chapter3.delta_hesitant", d_hesitant, 0.085, 1e-9),
        _check_close("chapter3.delta_neutrosophic", d_neutro, 0.26, 1e-9),
        _check_close("chapter3.delta_multilevel", d_multi, 0.20125, 1e-9),
        _check_close("chapter3.extended_d_E", d_ext, 0.163045, 5e-6),
        _check_equal("chapter3.composition_result", "ExplanationObject" if hasattr(composed, "uncertainty") else getattr(composed, "code", "unknown"), "ExplanationObject"),
        _check_equal("chapter3.diagnostic_choice_code", getattr(diag, "code", None), "D_choice"),
        _check_equal("chapter3.diagnostic_missing", getattr(diag, "context", {}).get("missing"), ["u_ethical"]),
    ]

    return {
        "checks": [c.__dict__ for c in checks],
        "computed": {
            "profile": sorted(profile),
            "selected_class": getattr(selected, "name", None),
            "reductions": {
                "IntervalFS_to_F0_delta": round(d_interval, 6),
                "HesitantFS_to_F0_delta": round(d_hesitant, 6),
                "NeutrosophicFS_to_F0_delta": round(d_neutro, 6),
                "MultiLevelFS_to_F0_delta": round(d_multi, 6),
            },
            "extended_d_E_with_delta": round(d_ext, 6),
            "composition_result": "ExplanationObject" if hasattr(composed, "uncertainty") else getattr(composed, "code", "unknown"),
            "diagnostic_choice_code": getattr(diag, "code", None),
            "diagnostic_choice_context": getattr(diag, "context", None),
            "u_time_u_cf_multilevel_levels": [sorted(level) for level in temporal_cf_levels],
        },
    }


def build_validation_report() -> Dict[str, Any]:
    chapter2 = validate_chapter2()
    chapter3 = validate_chapter3()
    all_checks = chapter2["checks"] + chapter3["checks"]
    failed = [c for c in all_checks if not c["passed"]]
    report = {
        "status": "PASS" if not failed else "FAIL",
        "total_checks": len(all_checks),
        "failed_checks": len(failed),
        "chapters": {
            "chapter2": chapter2,
            "chapter3": chapter3,
        },
        "failed": failed,
    }
    return report


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Thesis numerical validation report",
        "",
        f"Status: **{report['status']}**",
        f"Total checks: {report['total_checks']}",
        f"Failed checks: {report['failed_checks']}",
        "",
        "## Checked dissertation routes",
        "",
        "- Chapter 2: `z -> mu -> H -> d_E -> composition -> L -> I -> D_ij`.",
        "- Chapter 3: `metadata -> P_sit -> F* -> A_k^F -> Delta -> E_k_ext -> d_E_ext -> D_choice`.",
        "",
        "## Checks",
        "",
        "| Check | Expected | Actual | Tolerance | Status |",
        "|---|---:|---:|---:|---|",
    ]
    checks = report["chapters"]["chapter2"]["checks"] + report["chapters"]["chapter3"]["checks"]
    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"| `{c['name']}` | `{_round(c['expected'])}` | `{_round(c['actual'])}` | `{c['tolerance']}` | {status} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report = build_validation_report()
    json_path = OUT / "thesis_validation.json"
    md_path = OUT / "thesis_validation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"status": report["status"], "total_checks": report["total_checks"], "failed_checks": report["failed_checks"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
