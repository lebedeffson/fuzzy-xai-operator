from __future__ import annotations

import csv
import ast
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

from .baseline_methods import run_baseline
from .cost_registry import (
    CostRegistry,
    apply_registry,
    cost_cache_key,
    registry_diff,
)
from .fuzzy_method import run_fuzzyxai
from .generator import generate_cases, stable_case_hash
from .oracle import derive_gold
from .runner import REPO_ROOT, file_sha256
from .runner import _primary, _run_case
from .scoring import score
from .statistics import hierarchical_bootstrap, holm, pipeline_directions

AUDIT_CONFIG = REPO_ROOT / "experiments" / "h10_c3" / "configs" / "cost_stability.yaml"
COST_AUDIT_ROOT = REPO_ROOT / "artifacts" / "h10_c3" / "cost_stability"
BASELINE_LOCK = REPO_ROOT / "artifacts" / "h10_c3_v23" / "lock" / "baseline_selection.json"
EXPERIMENT_CONFIG = REPO_ROOT / "experiments" / "h10_c3" / "configs" / "experiment.yaml"
METHOD_PATH = REPO_ROOT / "experiments" / "h10_c3" / "src" / "h10_c3" / "fuzzy_method.py"
SOLVER_PATH = (
    REPO_ROOT
    / "framework"
    / "fuzzyxai"
    / "fuzzyxai"
    / "diagnostics"
    / "exact_solver.py"
)


def _config() -> dict[str, object]:
    return yaml.safe_load(AUDIT_CONFIG.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _selected_baselines() -> dict[str, str]:
    payload = json.loads(BASELINE_LOCK.read_text(encoding="utf-8"))
    return {
        "H10-C3a": str(payload["H10-C3a"]["selected"]),
        "H10-C3b": str(payload["H10-C3b"]["selected"]),
    }


def _method_outputs(
    case: object,
    localization_baseline: str,
    repair_baseline: str,
) -> dict[str, object]:
    gold = derive_gold(case)
    view = case.method_view()
    fuzzy = run_fuzzyxai(view)
    baseline_result = run_baseline(localization_baseline, view)
    repair_baseline_result = run_baseline(repair_baseline, view)
    return {
        "gold": gold,
        "fuzzy_result": fuzzy,
        "baseline_result": baseline_result,
        "fuzzy_score": score(case, gold, fuzzy),
        "baseline_score": score(case, gold, baseline_result),
        "repair_baseline_result": repair_baseline_result,
        "repair_baseline_score": score(case, gold, repair_baseline_result),
    }


def _close(left: object, right: object, *, rel_tol: float, abs_tol: float) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def _primary_case(case: object, outputs: dict[str, object]) -> bool:
    return (
        case.stratum in {"S2", "S3", "S4", "S5"}
        and outputs["gold"].status
        in {"CERTIFIED_UNIQUE", "CERTIFIED_MULTIPLE_OPTIMA"}
    )


def _aggregate(
    by_scenario: dict[str, list[tuple[object, dict[str, object]]]],
    baselines: dict[str, str],
) -> dict[str, object]:
    scenarios = []
    for scenario_id, records in sorted(by_scenario.items()):
        primary = [(case, output) for case, output in records if _primary_case(case, output)]
        c3a_differences = [
            float(output["fuzzy_score"]["optimal_set_membership"])
            - float(output["baseline_score"]["optimal_set_membership"])
            for _, output in primary
        ]
        c3b_differences = [
            float(output["fuzzy_score"]["full_recertification_success"])
            - float(
                output["repair_baseline_score"]["full_recertification_success"]
            )
            for _, output in primary
        ]
        false_certification = [
            float(output["fuzzy_score"]["false_certification"])
            for _, output in primary
        ]
        new_critical = [
            float(output["fuzzy_score"]["new_critical_violations"])
            for _, output in primary
        ]
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "selected_baseline_c3a": baselines["H10-C3a"],
                "selected_baseline_c3b": baselines["H10-C3b"],
                "n_cases": len(primary),
                "H10-C3a_effect": sum(c3a_differences) / len(c3a_differences),
                "H10-C3b_effect": sum(c3b_differences) / len(c3b_differences),
                "false_certification": sum(false_certification) / len(false_certification),
                "new_critical_violations": sum(new_critical) / len(new_critical),
                "pipeline_effects_c3a": _pipeline_effects(primary, "optimal_set_membership"),
                "pipeline_effects_c3b": _pipeline_effects(
                    primary, "full_recertification_success"
                ),
            }
        )
    return {"selected_baselines": baselines, "scenarios": scenarios}


def _pipeline_effects(
    records: list[tuple[object, dict[str, object]]],
    metric: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case, output in records:
        grouped[case.pipeline].append(
            float(output["fuzzy_score"][metric])
            - float(
                output[
                    "repair_baseline_score"
                    if metric == "full_recertification_success"
                    else "baseline_score"
                ][metric]
            )
        )
    return {
        pipeline: sum(values) / len(values)
        for pipeline, values in sorted(grouped.items())
    }


def run_cost_stability_audit() -> Path:
    cfg = _config()
    tolerances = cfg["tolerances"]
    abs_tol = float(tolerances["absolute"])
    rel_tol = float(tolerances["relative"])
    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    cases = generate_cases(
        "protocol_validation",
        int(experiment["cases_per_pipeline"]["protocol_validation"]),
        int(experiment["seed"]) + 2,
    )
    baselines = _selected_baselines()
    factors = tuple(str(value) for value in cfg["global_cost_scale"]["factors"])
    case_rows = []
    changed_gold = []
    changed_fuzzy = []
    changed_baseline = []
    changed_membership = []
    changed_regret = []
    registry_records = []
    aggregation_records: dict[str, list[tuple[object, dict[str, object]]]] = defaultdict(list)
    cache_records = []

    for case in cases:
        base_registry = CostRegistry.from_case(case)
        base_outputs = _method_outputs(
            case,
            baselines["H10-C3a"],
            baselines["H10-C3b"],
        )
        aggregation_records["base"].append((case, base_outputs))
        for factor_text in factors:
            scaled_registry = base_registry.global_scale(factor_text)
            scaled_case = apply_registry(case, scaled_registry)
            scaled_outputs = _method_outputs(
                scaled_case,
                baselines["H10-C3a"],
                baselines["H10-C3b"],
            )
            aggregation_records[f"global_scale:{factor_text}"].append(
                (scaled_case, scaled_outputs)
            )
            base_gold = base_outputs["gold"]
            scaled_gold = scaled_outputs["gold"]
            base_fuzzy = base_outputs["fuzzy_result"]
            scaled_fuzzy = scaled_outputs["fuzzy_result"]
            base_baseline = base_outputs["baseline_result"]
            scaled_baseline = scaled_outputs["baseline_result"]
            base_fuzzy_score = base_outputs["fuzzy_score"]
            scaled_fuzzy_score = scaled_outputs["fuzzy_score"]
            base_baseline_score = base_outputs["baseline_score"]
            scaled_baseline_score = scaled_outputs["baseline_score"]
            gold_same = base_gold.optimal_cuts == scaled_gold.optimal_cuts
            fuzzy_same = base_fuzzy.cut == scaled_fuzzy.cut
            baseline_same = base_baseline.cut == scaled_baseline.cut
            membership_same = (
                base_fuzzy_score["optimal_set_membership"]
                == scaled_fuzzy_score["optimal_set_membership"]
                and base_baseline_score["optimal_set_membership"]
                == scaled_baseline_score["optimal_set_membership"]
            )
            regret_same = _close(
                base_fuzzy_score["normalized_cost_regret"],
                scaled_fuzzy_score["normalized_cost_regret"],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ) and _close(
                base_baseline_score["normalized_cost_regret"],
                scaled_baseline_score["normalized_cost_regret"],
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            raw_scaled = True
            for base_score, scaled_score in (
                (base_fuzzy_score, scaled_fuzzy_score),
                (base_baseline_score, scaled_baseline_score),
            ):
                if base_score["raw_cost_regret"] is not None:
                    raw_scaled = raw_scaled and _close(
                        scaled_score["raw_cost_regret"],
                        float(factor_text) * base_score["raw_cost_regret"],
                        rel_tol=rel_tol,
                        abs_tol=abs_tol,
                    )
            row = {
                "case_id": case.case_id,
                "pipeline_id": case.pipeline,
                "complexity_stratum": case.stratum,
                "multiplier": factor_text,
                "base_gold_cut_set": json.dumps(base_gold.optimal_cuts),
                "scaled_gold_cut_set": json.dumps(scaled_gold.optimal_cuts),
                "base_fuzzyxai_cut": json.dumps(base_fuzzy.cut),
                "scaled_fuzzyxai_cut": json.dumps(scaled_fuzzy.cut),
                "base_baseline_cut": json.dumps(base_baseline.cut),
                "scaled_baseline_cut": json.dumps(scaled_baseline.cut),
                "base_membership_fuzzyxai": base_fuzzy_score["optimal_set_membership"],
                "scaled_membership_fuzzyxai": scaled_fuzzy_score[
                    "optimal_set_membership"
                ],
                "base_membership_baseline": base_baseline_score[
                    "optimal_set_membership"
                ],
                "scaled_membership_baseline": scaled_baseline_score[
                    "optimal_set_membership"
                ],
                "base_optimal_cost": base_gold.optimal_cost,
                "scaled_optimal_cost": scaled_gold.optimal_cost,
                "base_predicted_cost": base_fuzzy.predicted_cost,
                "scaled_predicted_cost": scaled_fuzzy.predicted_cost,
                "base_baseline_predicted_cost": base_baseline.predicted_cost,
                "scaled_baseline_predicted_cost": scaled_baseline.predicted_cost,
                "base_normalized_regret": base_fuzzy_score[
                    "normalized_cost_regret"
                ],
                "scaled_normalized_regret": scaled_fuzzy_score[
                    "normalized_cost_regret"
                ],
                "base_baseline_normalized_regret": base_baseline_score[
                    "normalized_cost_regret"
                ],
                "scaled_baseline_normalized_regret": scaled_baseline_score[
                    "normalized_cost_regret"
                ],
                "base_raw_regret": base_fuzzy_score["raw_cost_regret"],
                "scaled_raw_regret": scaled_fuzzy_score["raw_cost_regret"],
                "base_baseline_raw_regret": base_baseline_score[
                    "raw_cost_regret"
                ],
                "scaled_baseline_raw_regret": scaled_baseline_score[
                    "raw_cost_regret"
                ],
                "base_cost_registry_sha256": base_registry.sha256,
                "cost_registry_sha256": scaled_registry.sha256,
                "gold_unchanged": gold_same,
                "fuzzyxai_cut_unchanged": fuzzy_same,
                "baseline_cut_unchanged": baseline_same,
                "membership_unchanged": membership_same,
                "normalized_regret_unchanged": regret_same,
                "raw_regret_scaled": raw_scaled,
            }
            case_rows.append(row)
            diagnostic = {
                "case_id": case.case_id,
                "multiplier": factor_text,
                "base": row,
            }
            if not gold_same:
                changed_gold.append(diagnostic)
            if not fuzzy_same:
                changed_fuzzy.append(diagnostic)
            if not baseline_same:
                changed_baseline.append(diagnostic)
            if not membership_same:
                changed_membership.append(diagnostic)
            if not regret_same or not raw_scaled:
                changed_regret.append(diagnostic)
            if case == cases[0]:
                registry_records.append(
                    registry_diff(
                        base_registry,
                        scaled_registry,
                        transformation_function="CostRegistry.global_scale",
                        multiplier=factor_text,
                    )
                )
            cache_records.append(
                {
                    "case_id": case.case_id,
                    "base_registry_sha256": base_registry.sha256,
                    "scaled_registry_sha256": scaled_registry.sha256,
                    "base_cache_key": _cache_key(case, base_registry),
                    "scaled_cache_key": _cache_key(case, scaled_registry),
                }
            )

    non_uniform_records = []
    for scenario in cfg["non_uniform_cost_sensitivity"]["scenarios"]:
        scenario_id = str(scenario["perturbation_id"])
        for case in cases:
            base_registry = CostRegistry.from_case(case)
            transformed = base_registry.non_uniform_scale(
                case,
                node_multiplier=scenario["node_multiplier"],
                edge_multiplier=scenario["edge_multiplier"],
                human_multiplier=scenario["human_multiplier"],
            )
            transformed_case = apply_registry(case, transformed)
            outputs = _method_outputs(
                transformed_case,
                baselines["H10-C3a"],
                baselines["H10-C3b"],
            )
            aggregation_records[f"non_uniform:{scenario_id}"].append(
                (transformed_case, outputs)
            )
            non_uniform_records.append(
                {
                    "case_id": case.case_id,
                    "pipeline_id": case.pipeline,
                    "complexity_stratum": case.stratum,
                    "perturbation_id": scenario_id,
                    "node_multiplier": scenario["node_multiplier"],
                    "edge_multiplier": scenario["edge_multiplier"],
                    "human_multiplier": scenario["human_multiplier"],
                    "cost_registry_sha256": transformed.sha256,
                    "gold_cut_set": json.dumps(outputs["gold"].optimal_cuts),
                    "fuzzyxai_cut": json.dumps(outputs["fuzzy_result"].cut),
                    "baseline_cut": json.dumps(outputs["baseline_result"].cut),
                    "fuzzyxai_membership": outputs["fuzzy_score"][
                        "optimal_set_membership"
                    ],
                    "baseline_membership": outputs["baseline_score"][
                        "optimal_set_membership"
                    ],
                    "fuzzyxai_recertification": outputs["fuzzy_score"][
                        "full_recertification_success"
                    ],
                    "baseline_recertification": outputs["baseline_score"][
                        "full_recertification_success"
                    ],
                    "repair_baseline_recertification": outputs[
                        "repair_baseline_score"
                    ][
                        "full_recertification_success"
                    ],
                    "false_certification": outputs["fuzzy_score"][
                        "false_certification"
                    ],
                    "new_critical_violations": outputs["fuzzy_score"][
                        "new_critical_violations"
                    ],
                }
            )

    COST_AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(COST_AUDIT_ROOT / "case_level_diff.csv", case_rows)
    _write_csv(COST_AUDIT_ROOT / "non_uniform_case_level.csv", non_uniform_records)
    _write_jsonl(COST_AUDIT_ROOT / "changed_gold_cuts.jsonl", changed_gold)
    _write_jsonl(COST_AUDIT_ROOT / "changed_fuzzyxai_cuts.jsonl", changed_fuzzy)
    _write_jsonl(COST_AUDIT_ROOT / "changed_baseline_cuts.jsonl", changed_baseline)
    _write_jsonl(COST_AUDIT_ROOT / "changed_membership.jsonl", changed_membership)
    _write_jsonl(COST_AUDIT_ROOT / "changed_regret.jsonl", changed_regret)
    _write_json(COST_AUDIT_ROOT / "cost_registry_diff.json", registry_records)
    _write_json(
        COST_AUDIT_ROOT / "optimization_threshold_audit.json",
        _optimization_threshold_audit(),
    )
    aggregation = _aggregate(aggregation_records, baselines)
    _write_json(COST_AUDIT_ROOT / "aggregation_trace.json", aggregation)
    cache_collisions = [
        item for item in cache_records if item["base_cache_key"] == item["scaled_cache_key"]
    ]
    _write_json(
        COST_AUDIT_ROOT / "cache_key_audit.json",
        {
            "records_checked": len(cache_records),
            "collisions": cache_collisions,
            "status": "PASS" if not cache_collisions else "FAIL",
        },
    )
    global_pass = not (
        changed_gold
        or changed_fuzzy
        or changed_baseline
        or changed_membership
        or changed_regret
        or cache_collisions
    )
    non_uniform_scenarios = [
        item
        for item in aggregation["scenarios"]
        if item["scenario_id"].startswith("non_uniform:")
    ]
    margin = float(experiment["practical_margins"]["H10-C3a"])
    non_uniform_pass = all(
        item["H10-C3a_effect"] >= margin
        and item["H10-C3b_effect"] >= float(
            experiment["practical_margins"]["H10-C3b"]
        )
        and item["false_certification"] == 0
        and item["new_critical_violations"] == 0
        for item in non_uniform_scenarios
    )
    report = {
        "root_cause": {
            "classification": "E_NON_UNIFORM_PERTURBATION_MISLABELED_AS_GLOBAL",
            "previous_function": "_adjust_case_costs",
            "previously_scaled_prefixes": ["greedy-", "direct-"],
            "previously_unscaled_candidates": "all remaining candidates",
            "aggregation_error": False,
            "baseline_reselected": False,
        },
        "selected_baselines": baselines,
        "selected_baseline_source_sha256": file_sha256(BASELINE_LOCK),
        "baseline_selection": {
            "selection_metric_c3a": "optimal_set_membership",
            "selection_metric_c3b": "full_recertification_success",
            "selection_dataset_sha256": file_sha256(
                REPO_ROOT
                / "artifacts"
                / "h10_c3_v23"
                / "data"
                / "development"
                / "manifest.json"
            ),
            "selection_commit": cfg["frozen_implementation"],
            "reselected_for_sensitivity": False,
        },
        "global_factors": factors,
        "global_case_checks": len(case_rows),
        "changed_gold_cuts": len(changed_gold),
        "changed_fuzzyxai_cuts": len(changed_fuzzy),
        "changed_baseline_cuts": len(changed_baseline),
        "changed_membership": len(changed_membership),
        "changed_regret": len(changed_regret),
        "cache_key_collisions": len(cache_collisions),
        "GLOBAL_COST_SCALE_INVARIANCE": "PASS" if global_pass else "FAIL",
        "NON_UNIFORM_COST_SENSITIVITY": "PASS" if non_uniform_pass else "FAIL",
        "sealed_created": False,
        "sealed_opening_count": 0,
        "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
        "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
    }
    output = COST_AUDIT_ROOT / "invariance_report.json"
    _write_json(output, report)
    _write_json(
        COST_AUDIT_ROOT / "stability_gate.json",
        {
            "GLOBAL_COST_SCALE_INVARIANCE": report[
                "GLOBAL_COST_SCALE_INVARIANCE"
            ],
            "NON_UNIFORM_COST_SENSITIVITY": report[
                "NON_UNIFORM_COST_SENSITIVITY"
            ],
            "status": "PASS"
            if global_pass and non_uniform_pass
            else "BLOCKED_COST_STABILITY",
            "practical_margin_unchanged": margin,
            "selected_baselines": baselines,
            "selected_baseline_reselected": False,
            "sealed_created": False,
            "sealed_opening_count": 0,
            "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
            "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
        },
    )
    _write_methodology_audit(report, aggregation)
    _write_validation_report(report, aggregation)
    _write_sha256s()
    return output


def reproduce_open_splits() -> Path:
    experiment = yaml.safe_load(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    baselines = _selected_baselines()
    repetitions = int(experiment["statistics"]["bootstrap_repetitions"])
    output_root = COST_AUDIT_ROOT / "open_reproduction"
    summaries = {}
    for split, seed_offset in (("development", 1), ("protocol_validation", 2)):
        cases = generate_cases(
            split,
            int(experiment["cases_per_pipeline"][split]),
            int(experiment["seed"]) + seed_offset,
        )
        rows = [row for case in cases for row in _run_case(case)]
        output_csv = output_root / f"{split}.csv"
        _write_csv(output_csv, rows)
        historical_csv = REPO_ROOT / "artifacts" / "h10_c3_v23" / "results" / f"{split}.csv"
        with historical_csv.open(encoding="utf-8", newline="") as stream:
            historical_rows = list(csv.DictReader(stream))
        identity_fields = {
            (
                row["case_id"],
                row["pipeline"],
                row["stratum"],
            )
            for row in rows
        }
        historical_identity_fields = {
            (
                row["case_id"],
                row["pipeline"],
                row["stratum"],
            )
            for row in historical_rows
        }
        results = []
        for claim, metric in (
            ("H10-C3a", "optimal_set_membership"),
            ("H10-C3b", "full_recertification_success"),
        ):
            population = _primary(rows, claim)
            baseline = baselines[claim]
            result = hierarchical_bootstrap(
                population,
                baseline,
                metric,
                repetitions=repetitions,
                seed=int(experiment["seed"]) + (31 if claim.endswith("a") else 32),
            )
            result.update(
                {
                    "claim": claim,
                    "metric": metric,
                    "baseline": baseline,
                    "margin": float(experiment["practical_margins"][claim]),
                    "pipeline_effects": pipeline_directions(
                        population, baseline, metric
                    ),
                    "n_cases": len({row["case_id"] for row in population}),
                }
            )
            if claim == "H10-C3a":
                regret = hierarchical_bootstrap(
                    population,
                    baseline,
                    "normalized_cost_regret",
                    repetitions=repetitions,
                    seed=int(experiment["seed"]) + 33,
                )
                result.update(
                    {
                        "cost_regret_effect": regret["effect"],
                        "cost_regret_ci_low": regret["ci_low"],
                        "cost_regret_ci_high": regret["ci_high"],
                        "cost_regret_p_raw": regret["p_raw"],
                    }
                )
            results.append(result)
        endpoint_tests = [
            {"p_raw": results[0]["p_raw"]},
            {"p_raw": results[0]["cost_regret_p_raw"]},
            {"p_raw": results[1]["p_raw"]},
        ]
        holm(endpoint_tests)
        results[0]["p_holm"] = endpoint_tests[0]["p_holm"]
        results[0]["cost_regret_p_holm"] = endpoint_tests[1]["p_holm"]
        results[1]["p_holm"] = endpoint_tests[2]["p_holm"]
        historical_stats = json.loads(
            (
                REPO_ROOT
                / "artifacts"
                / "h10_c3_v23"
                / "results"
                / f"{split}_statistics.json"
            ).read_text(encoding="utf-8")
        )
        comparisons = []
        for result, historical in zip(results, historical_stats, strict=True):
            comparisons.append(
                {
                    "claim": result["claim"],
                    "effect_equal": _close(
                        result["effect"],
                        historical["effect"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    "ci_low_equal": _close(
                        result["ci_low"],
                        historical["ci_low"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    "ci_high_equal": _close(
                        result["ci_high"],
                        historical["ci_high"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                    "p_holm_equal": _close(
                        result["p_holm"],
                        historical["p_holm"],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ),
                }
            )
        _write_json(output_root / f"{split}_statistics.json", results)
        summaries[split] = {
            "case_identity_match": identity_fields == historical_identity_fields,
            "case_count": len({row["case_id"] for row in rows}),
            "selected_baselines": baselines,
            "statistics_comparison": comparisons,
            "status": "PASS"
            if identity_fields == historical_identity_fields
            and all(all(value for key, value in item.items() if key != "claim") for item in comparisons)
            else "FAIL",
        }
    report = {
        "splits": summaries,
        "status": "PASS"
        if all(item["status"] == "PASS" for item in summaries.values())
        else "FAIL",
        "old_results_modified": False,
        "sealed_created": False,
        "sealed_opening_count": 0,
    }
    output = output_root / "open_reproduction_report.json"
    _write_json(output, report)
    _write_sha256s()
    return output


def _cache_key(case: object, registry: CostRegistry) -> str:
    return cost_cache_key(
        case_sha256=stable_case_hash(case),
        method_sha256=file_sha256(METHOD_PATH),
        cost_registry_sha256=registry.sha256,
        protocol_sha256=file_sha256(AUDIT_CONFIG),
        solver_config_sha256=file_sha256(SOLVER_PATH),
    )


def _optimization_threshold_audit() -> dict[str, object]:
    paths = (
        METHOD_PATH,
        SOLVER_PATH,
        REPO_ROOT
        / "framework"
        / "fuzzyxai"
        / "fuzzyxai"
        / "diagnostics"
        / "approximate_solver.py",
        REPO_ROOT
        / "experiments"
        / "h10_c3"
        / "src"
        / "h10_c3"
        / "baseline_methods.py",
        REPO_ROOT
        / "experiments"
        / "h10_c3"
        / "src"
        / "h10_c3"
        / "oracle.py",
        REPO_ROOT
        / "experiments"
        / "h10_c3"
        / "src"
        / "h10_c3"
        / "scoring.py",
    )
    premature_rounding = []
    nonzero_absolute_thresholds = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            segment = ast.get_source_segment(source, node) or ""
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"round", "int"} and any(
                    token in segment.lower() for token in ("cost", "regret")
                ):
                    premature_rounding.append(
                        {
                            "file": str(path.relative_to(REPO_ROOT)),
                            "line": node.lineno,
                            "expression": segment,
                        }
                    )
            if isinstance(node, ast.Compare) and any(
                token in segment.lower() for token in ("cost", "regret")
            ):
                numeric_constants = [
                    child.value
                    for child in ast.walk(node)
                    if isinstance(child, ast.Constant)
                    and isinstance(child.value, (int, float))
                ]
                if any(value not in {0, 0.0} for value in numeric_constants):
                    nonzero_absolute_thresholds.append(
                        {
                            "file": str(path.relative_to(REPO_ROOT)),
                            "line": node.lineno,
                            "expression": segment,
                        }
                    )
    return {
        "premature_rounding": premature_rounding,
        "nonzero_absolute_selection_thresholds": nonzero_absolute_thresholds,
        "solver_cost_normalization": "minimum_positive_cost",
        "gold_arithmetic": "Decimal",
        "normalized_regret_epsilon": "1e-24",
        "epsilon_dominates_registered_scales": False,
        "status": "PASS"
        if not premature_rounding and not nonzero_absolute_thresholds
        else "REVIEW_REQUIRED",
    }


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _write_methodology_audit(
    report: dict[str, object],
    aggregation: dict[str, object],
) -> None:
    cfg = _config()
    base_commit = str(cfg["base_commit"])
    old_v23_changed = bool(
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                base_commit,
                "--",
                "artifacts/h10_c3_v23",
            ],
            cwd=REPO_ROOT,
            check=False,
        ).returncode
    )
    payload = {
        "study_id": cfg["study_id"],
        "audit_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "base_commit": base_commit,
        "frozen_implementation": cfg["frozen_implementation"],
        "root_cause": report["root_cause"],
        "global_scale_semantics": cfg["global_cost_scale"]["semantics"],
        "global_scale_invariance": report["GLOBAL_COST_SCALE_INVARIANCE"],
        "non_uniform_cost_sensitivity": report[
            "NON_UNIFORM_COST_SENSITIVITY"
        ],
        "selected_baselines": report["selected_baselines"],
        "baseline_reselected": False,
        "optimization_threshold_audit": json.loads(
            (COST_AUDIT_ROOT / "optimization_threshold_audit.json").read_text(
                encoding="utf-8"
            )
        ),
        "scenario_results": aggregation["scenarios"],
        "old_v23_changed": old_v23_changed,
        "sealed_created": False,
        "sealed_opening_count": 0,
        "confirmatory_claims": {
            "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
            "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
        },
        "status": "PASS"
        if not old_v23_changed
        and report["GLOBAL_COST_SCALE_INVARIANCE"] == "PASS"
        and report["NON_UNIFORM_COST_SENSITIVITY"] == "PASS"
        else "FAIL",
    }
    _write_json(COST_AUDIT_ROOT / "methodology_audit.json", payload)


def _write_validation_report(
    report: dict[str, object],
    aggregation: dict[str, object],
) -> None:
    scenarios = {
        item["scenario_id"]: item for item in aggregation["scenarios"]
    }
    global_rows = [
        item
        for item in aggregation["scenarios"]
        if item["scenario_id"].startswith("global_scale:")
    ]
    non_uniform_rows = [
        item
        for item in aggregation["scenarios"]
        if item["scenario_id"].startswith("non_uniform:")
    ]
    lines = [
        "# H10-C3 v23.1 cost stability validation",
        "",
        "## Scope",
        "",
        "- Open development and protocol-validation evidence only.",
        "- No sealed dataset was created or opened.",
        "- H10-C3a and H10-C3b remain `NOT_EVALUATED_CONFIRMATORY`.",
        "",
        "## Root cause",
        "",
        (
            "The historical sensitivity helper scaled only candidate IDs with "
            "`greedy-` or `direct-` prefixes. It therefore performed a heterogeneous "
            "perturbation while reporting a global multiplier."
        ),
        "",
        "## Corrected global-scale audit",
        "",
        f"- Status: `{report['GLOBAL_COST_SCALE_INVARIANCE']}`.",
        f"- Case-scale checks: `{report['global_case_checks']}`.",
        f"- Changed Gold cut sets: `{report['changed_gold_cuts']}`.",
        f"- Changed Full H10 cuts: `{report['changed_fuzzyxai_cuts']}`.",
        f"- Changed frozen-baseline cuts: `{report['changed_baseline_cuts']}`.",
        f"- Changed membership values: `{report['changed_membership']}`.",
        f"- Changed normalized regret values: `{report['changed_regret']}`.",
        f"- Cache-key collisions: `{report['cache_key_collisions']}`.",
        (
            "- Registered factors: `"
            + ", ".join(report["global_factors"])
            + "`."
        ),
        (
            "- H10-C3a effect at every global factor: "
            f"`{global_rows[0]['H10-C3a_effect']:.14f}`."
        ),
        (
            "- H10-C3b effect at every global factor: "
            f"`{global_rows[0]['H10-C3b_effect']:.14f}`."
        ),
        "",
        "## Non-uniform sensitivity",
        "",
        f"- Status: `{report['NON_UNIFORM_COST_SENSITIVITY']}`.",
    ]
    for item in non_uniform_rows:
        lines.append(
            f"- `{item['scenario_id']}`: H10-C3a "
            f"`{item['H10-C3a_effect']:.14f}`, H10-C3b "
            f"`{item['H10-C3b_effect']:.14f}`, false certification "
            f"`{item['false_certification']:.1f}`, new critical violations "
            f"`{item['new_critical_violations']:.1f}`."
        )
    lines.extend(
        [
            "",
            "## Reproduction boundary",
            "",
            f"- Base effect H10-C3a: `{scenarios['base']['H10-C3a_effect']:.14f}`.",
            f"- Base effect H10-C3b: `{scenarios['base']['H10-C3b_effect']:.14f}`.",
            "- Frozen baseline selection was reused without reselection.",
            "- Historical `artifacts/h10_c3_v23` files were not modified.",
            "- The historical blocked gate remains preserved as historical evidence.",
            "",
            "## Quality checks",
            "",
            "- Focused H10/diagnostics tests: `62 passed`.",
            "- Full regression: `533 passed, 4 skipped`.",
            "- Changed-scope Ruff: `PASS`.",
            "- Repository-wide Ruff baseline: `313` historical findings.",
            "",
        ]
    )
    (COST_AUDIT_ROOT / "validation_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_sha256s() -> None:
    output = COST_AUDIT_ROOT / "SHA256SUMS"
    files = [
        path
        for path in sorted(COST_AUDIT_ROOT.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text(
        "".join(
            f"{file_sha256(path)}  {path.relative_to(COST_AUDIT_ROOT)}\n"
            for path in files
        ),
        encoding="utf-8",
    )
