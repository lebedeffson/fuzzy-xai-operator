from __future__ import annotations

import ast
from pathlib import Path

from .common import ARTIFACT_ROOT, ROOT, write_json
from .oracle_v19 import CATALOG
from fuzzyxai.audit_h10.taxonomy import FAULT_SPECS


FORBIDDEN_ORACLE_MODULE_PREFIXES = ("fuzzyxai.audit_h10", "baselines.h10")
FORBIDDEN_ORACLE_NAMES = {
    "H10Auditor",
    "DiagnosticCutSolver",
    "FaultFamilyClassifier",
    "SourceLocalizer",
    "RepairSetPlanner",
    "SPEC_BY_LEAF",
    "FIELD_TO_SPECS",
    "TypedRouteGuard",
}


def _oracle_independence_failures(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    failures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if module.startswith(FORBIDDEN_ORACLE_MODULE_PREFIXES) or names & FORBIDDEN_ORACLE_NAMES:
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:{module}:{sorted(names)}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(FORBIDDEN_ORACLE_MODULE_PREFIXES):
                    failures.append(f"{path.relative_to(ROOT)}:{node.lineno}:{alias.name}")
    return failures


def audit() -> dict:
    oracle_path = ROOT / "experiments" / "h10" / "oracle_v19.py"
    mutation_source = (ROOT / "experiments" / "h10" / "mutations.py").read_text(encoding="utf-8")
    replay_source = (ROOT / "experiments" / "h10" / "run_replay.py").read_text(encoding="utf-8")
    failures = _oracle_independence_failures(oracle_path)
    oracle_sources = {(item.leaf, item.source_nodes) for item in CATALOG}
    evaluated_sources = {(item.leaf, item.source_nodes) for item in FAULT_SPECS}
    source_catalog_is_duplicated = oracle_sources == evaluated_sources
    source_truth_is_catalog_assigned = "spec.source_nodes" in mutation_source
    findings = {
        "oracle_import_independence": not failures,
        "oracle_uses_independent_exhaustive_solver": "def independent_optimal_cuts" in oracle_path.read_text(encoding="utf-8"),
        "mutation_truth_does_not_call_evaluated_cut_solver": "DiagnosticCutSolver" not in mutation_source,
        "mutation_truth_does_not_import_evaluated_taxonomy": "fuzzyxai.audit_h10.taxonomy" not in mutation_source,
        "source_truth_comes_from_mutation_log": "MutationOperation" in mutation_source and "build_truth" in mutation_source,
        "source_truth_derived_from_actual_mutated_graph_nodes": not source_truth_is_catalog_assigned,
        "source_catalog_semantically_independent": not source_catalog_is_duplicated,
        "repair_truth_independent_from_source_catalog": "repair_sets=(sources,)" not in oracle_path.read_text(encoding="utf-8"),
        "replay_loads_clean_routes": '"clean_routes.jsonl"' in replay_source,
        "replay_does_not_use_sealed_mutated_routes_as_normal": "normal_routes = routes" not in replay_source,
        "oracle_independence_failures": failures,
    }
    passed = all(value is True for key, value in findings.items() if key != "oracle_independence_failures") and not failures
    report = {
        "status": "PASS" if passed else "INVALID_FOR_PRIMARY_COMPARATIVE_CLAIMS",
        "findings": findings,
        "sealed_scoring_repeated": False,
        "post_open_tuning": False,
        "affected_claims": [] if passed else ["H10-L", "H10-R"],
        "unaffected_claims": ["H10-C", "H10-U", "H10-T"] if not passed else ["H10-L", "H10-C", "H10-R", "H10-U", "H10-T"],
        "replay_status": "valid_design" if findings["replay_loads_clean_routes"] else "invalid_design",
        "identity_anchor_reuse_disclosed": True,
    }
    write_json(ARTIFACT_ROOT / "closure" / "confirmatory_methodology_audit.json", report)
    if not passed:
        write_json(
            ARTIFACT_ROOT / "opening" / "confirmatory_invalid_marker.json",
            {
                "status": report["status"],
                "reason": "oracle or replay independence failure",
                "old_outputs_preserved": True,
                "repeat_scoring_forbidden": True,
            },
        )
    return report


if __name__ == "__main__":
    audit()
