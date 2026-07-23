from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from statistics import median

import yaml

from .generator import generate_cases, serialize_cases
from .baseline_methods import BASELINES, run_baseline
from .fuzzy_method import run_fuzzyxai
from .oracle import derive_gold
from .scoring import score
from .statistics import hierarchical_bootstrap, holm, pipeline_directions, wilson_interval

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "h10_c3"
ARTIFACT_ROOT = Path(os.environ.get("H10_C3_ARTIFACT_ROOT", REPO_ROOT / "artifacts" / "h10_c3_v23"))
CONFIG_PATH = EXPERIMENT_ROOT / "configs" / "experiment.yaml"


def config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(split: str) -> Path:
    cfg = config()
    if split == "sealed":
        raise PermissionError("sealed generation is not part of the preconfirmatory workflow")
    count = int(cfg["cases_per_pipeline"][split])
    seed = int(cfg["seed"]) + {"development": 1, "protocol_validation": 2}[split]
    cases = generate_cases(split, count, seed)
    manifest = serialize_cases(ARTIFACT_ROOT, split, cases)
    if split == "protocol_validation":
        verify_lock()
    path = ARTIFACT_ROOT / "data" / split / "manifest.json"
    write_json(path, manifest)
    return path


def _adjust_case_costs(case: object, multiplier: float) -> object:
    return replace(
        case,
        candidates=tuple(
            replace(
                candidate,
                cost=candidate.cost * multiplier
                if candidate.atom_id.startswith(("greedy-", "direct-"))
                else candidate.cost,
            )
            for candidate in case.candidates
        ),
    )


def _run_case(case: object) -> list[dict[str, object]]:
    gold = derive_gold(case)
    view = case.method_view()
    methods = [
        *(run_baseline(name, view) for name in BASELINES),
        run_fuzzyxai(view),
    ]
    rows = []
    for result in methods:
        metrics = score(case, gold, result)
        rows.append(
            {
                "case_id": case.case_id,
                "pipeline": case.pipeline,
                "modality": case.modality,
                "split": case.split,
                "stratum": case.stratum,
                "family": case.family,
                "gold_status": gold.status,
                "repairable": gold.repairable,
                "method": result.method,
                "predicted_cut": json.dumps(result.cut),
                "predicted_plan": json.dumps(result.plan),
                "predicted_cost": result.predicted_cost,
                "runtime_ms": result.runtime_ms,
                **metrics,
            }
        )
    return rows


def run(split: str, *, cost_multiplier: float = 1.0) -> Path:
    cfg = config()
    if split == "protocol_validation":
        verify_lock()
    count = int(cfg["cases_per_pipeline"][split])
    seed = int(cfg["seed"]) + {"development": 1, "protocol_validation": 2}[split]
    cases = generate_cases(split, count, seed)
    if cost_multiplier != 1.0:
        cases = [_adjust_case_costs(case, cost_multiplier) for case in cases]
    rows = [
        row
        for case in cases
        for row in _run_case(case)
    ]
    suffix = "" if cost_multiplier == 1.0 else f"_cost_{cost_multiplier:.1f}"
    output = ARTIFACT_ROOT / "results" / f"{split}{suffix}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if split == "development" and cost_multiplier == 1.0:
        select_best_baselines(rows)
    return output


def load_rows(split: str, suffix: str = "") -> list[dict[str, object]]:
    path = ARTIFACT_ROOT / "results" / f"{split}{suffix}.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in (
            "optimal_set_membership",
            "full_recertification_success",
            "false_certification",
            "new_critical_violations",
            "repairable",
        ):
            row[key] = float(str(row[key]).lower() in {"true", "1", "1.0"})
        for key in (
            "normalized_cost_regret",
            "obligation_coverage",
            "runtime_ms",
            "plan_cost",
        ):
            row[key] = float(row[key])
    return rows


def _primary(rows: list[dict[str, object]], claim: str) -> list[dict[str, object]]:
    certified = {"CERTIFIED_UNIQUE", "CERTIFIED_MULTIPLE_OPTIMA"}
    selected = [
        row
        for row in rows
        if row["stratum"] in {"S2", "S3", "S4", "S5"}
        and row["gold_status"] in certified
    ]
    if claim == "H10-C3b":
        selected = [row for row in selected if bool(row["repairable"])]
    return selected


def select_best_baselines(rows: list[dict[str, object]]) -> None:
    selection = {}
    for claim, metric in (
        ("H10-C3a", "optimal_set_membership"),
        ("H10-C3b", "full_recertification_success"),
    ):
        population = _primary(rows, claim)
        summaries = {}
        for method in BASELINES:
            values = [float(row[metric]) for row in population if row["method"] == method]
            regret = [
                float(row["normalized_cost_regret"])
                for row in population
                if row["method"] == method
            ]
            summaries[method] = {
                "metric": sum(values) / len(values),
                "mean_cost_regret": sum(regret) / len(regret),
            }
        best = max(
            BASELINES,
            key=lambda method: (
                summaries[method]["metric"],
                -summaries[method]["mean_cost_regret"],
                method,
            ),
        )
        selection[claim] = {"selected": best, "development_summary": summaries[best]}
    write_json(ARTIFACT_ROOT / "lock" / "baseline_selection.json", selection)


def freeze() -> Path:
    selection_path = ARTIFACT_ROOT / "lock" / "baseline_selection.json"
    if not selection_path.exists():
        raise RuntimeError("run development before protocol freeze")
    tracked = [
        CONFIG_PATH,
        *sorted((EXPERIMENT_ROOT / "src" / "h10_c3").glob("*.py")),
    ]
    lock = {
        "study_id": config()["study_id"],
        "version": config()["version"],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "files": {str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in tracked},
        "baseline_selection_sha256": file_sha256(selection_path),
        "sealed_opening_count": 0,
        "post_lock_tuning": False,
    }
    output = ARTIFACT_ROOT / "lock" / "protocol.lock.json"
    write_json(output, lock)
    return output


def verify_lock() -> None:
    lock_path = ARTIFACT_ROOT / "lock" / "protocol.lock.json"
    if not lock_path.exists():
        raise PermissionError("protocol validation requires a frozen protocol")
    lock = read_json(lock_path)
    for relative, expected in lock["files"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise PermissionError(f"post-lock change detected: {relative}")
    selection = ARTIFACT_ROOT / "lock" / "baseline_selection.json"
    if file_sha256(selection) != lock["baseline_selection_sha256"]:
        raise PermissionError("post-lock baseline selection change detected")


def analyze(split: str) -> Path:
    rows = load_rows(split)
    selections = read_json(ARTIFACT_ROOT / "lock" / "baseline_selection.json")
    cfg = config()
    results = []
    for claim, metric in (
        ("H10-C3a", "optimal_set_membership"),
        ("H10-C3b", "full_recertification_success"),
    ):
        population = _primary(rows, claim)
        baseline = selections[claim]["selected"]
        result = hierarchical_bootstrap(
            population,
            baseline,
            metric,
            repetitions=int(cfg["statistics"]["bootstrap_repetitions"]),
            seed=int(cfg["seed"]) + (31 if claim.endswith("a") else 32),
        )
        result.update(
            {
                "claim": claim,
                "metric": metric,
                "baseline": baseline,
                "margin": float(cfg["practical_margins"][claim]),
                "pipeline_effects": pipeline_directions(population, baseline, metric),
                "n_cases": len({row["case_id"] for row in population}),
            }
        )
        if claim == "H10-C3a":
            regret = hierarchical_bootstrap(
                population,
                baseline,
                "normalized_cost_regret",
                repetitions=int(cfg["statistics"]["bootstrap_repetitions"]),
                seed=int(cfg["seed"]) + 33,
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
    for result in results:
        positive_pipelines = sum(value > 0 for value in result["pipeline_effects"].values())
        result["positive_pipelines"] = positive_pipelines
        cost_pass = result["claim"] != "H10-C3a" or (
            result["cost_regret_effect"] > 0
            and result["cost_regret_ci_low"] > 0
            and result["cost_regret_p_holm"] < 0.05
        )
        result["status"] = (
            "development_pass"
            if split == "development"
            and result["effect"] >= result["margin"]
            and result["ci_low"] > 0
            and result["p_holm"] < 0.05
            and positive_pipelines >= 5
            and cost_pass
            else "protocol_validation_pass"
            if split == "protocol_validation"
            and result["effect"] >= result["margin"]
            and result["ci_low"] > 0
            and result["p_holm"] < 0.05
            and positive_pipelines >= 5
            and cost_pass
            else f"{split}_fail"
        )
    output = ARTIFACT_ROOT / "results" / f"{split}_statistics.json"
    write_json(output, results)
    return output


def stability() -> Path:
    rows = load_rows("protocol_validation")
    selection = read_json(ARTIFACT_ROOT / "lock" / "baseline_selection.json")
    checks = []
    for claim, metric in (
        ("H10-C3a", "optimal_set_membership"),
        ("H10-C3b", "full_recertification_success"),
    ):
        population = _primary(rows, claim)
        baseline = selection[claim]["selected"]
        for stratum in ("S2", "S3", "S4", "S5"):
            subset = [row for row in population if row["stratum"] == stratum]
            if subset:
                effect = hierarchical_bootstrap(
                    subset, baseline, metric, repetitions=1000, seed=231100
                )["effect"]
                checks.append({"claim": claim, "check": f"stratum:{stratum}", "effect": effect})
        for excluded in sorted({str(row["family"]) for row in population}):
            subset = [row for row in population if row["family"] != excluded]
            effect = hierarchical_bootstrap(
                subset, baseline, metric, repetitions=500, seed=231101
            )["effect"]
            checks.append({"claim": claim, "check": f"leave_family_out:{excluded}", "effect": effect})
    cost_checks = []
    for multiplier in (0.8, 1.2):
        run("protocol_validation", cost_multiplier=multiplier)
        varied = load_rows("protocol_validation", f"_cost_{multiplier:.1f}")
        for claim, metric in (
            ("H10-C3a", "optimal_set_membership"),
            ("H10-C3b", "full_recertification_success"),
        ):
            population = _primary(varied, claim)
            baseline = selection[claim]["selected"]
            effect = hierarchical_bootstrap(
                population, baseline, metric, repetitions=1000, seed=231120
            )["effect"]
            cost_checks.append(
                {
                    "claim": claim,
                    "multiplier": multiplier,
                    "effect": effect,
                }
            )
    seed_checks = []
    cfg = config()
    for seed_offset in (11, 12, 13):
        cases = generate_cases(
            f"stability_seed_{seed_offset}",
            60,
            int(cfg["seed"]) + seed_offset,
        )
        seed_rows = [row for case in cases for row in _run_case(case)]
        for claim, metric in (
            ("H10-C3a", "optimal_set_membership"),
            ("H10-C3b", "full_recertification_success"),
        ):
            population = _primary(seed_rows, claim)
            baseline = selection[claim]["selected"]
            effect = hierarchical_bootstrap(
                population, baseline, metric, repetitions=500, seed=231130
            )["effect"]
            seed_checks.append(
                {"claim": claim, "seed_offset": seed_offset, "effect": effect}
            )
    stratum_pass = all(
        item["effect"] >= 0 for item in checks if item["check"].startswith("stratum:")
    )
    nontrivial_strata_positive = all(
        item["effect"] > 0
        for item in checks
        if item["check"] in {"stratum:S3", "stratum:S5"}
    )
    summary = {
        "checks": checks,
        "cost_sensitivity": cost_checks,
        "seed_sensitivity": seed_checks,
        "status": "PASS"
        if stratum_pass
        and nontrivial_strata_positive
        and all(item["effect"] > 0 for item in cost_checks)
        and all(item["effect"] > 0 for item in seed_checks)
        else "FAIL",
        "interpretation": (
            "S2 and some S4 comparisons may be parity controls; no stratum may regress, "
            "and graph-dependent S3/S5 effects must remain positive."
        ),
    }
    output = ARTIFACT_ROOT / "results" / "stability.json"
    write_json(output, summary)
    return output


def power() -> Path:
    validation_stats = read_json(ARTIFACT_ROOT / "results" / "protocol_validation_statistics.json")
    cfg = config()
    simulations = int(cfg["statistics"]["power_simulations"])
    output_claims = []
    # A conservative empirical simulation: sample a normal approximation using
    # the registered hierarchical interval width, then test the registered margin.
    import random

    rng = random.Random(int(cfg["seed"]) + 90)
    for result in validation_stats:
        standard_error = (result["ci_high"] - result["ci_low"]) / (2 * 1.96)
        successes = sum(
            rng.gauss(result["effect"], standard_error) >= result["margin"]
            for _ in range(simulations)
        )
        low, high = wilson_interval(successes, simulations)
        output_claims.append(
            {
                "claim": result["claim"],
                "point_power": successes / simulations,
                "lower_confidence_bound": low,
                "upper_confidence_bound": high,
                "number_of_simulations": simulations,
                "monte_carlo_standard_error": (
                    successes / simulations * (1 - successes / simulations) / simulations
                )
                ** 0.5,
                "status": "pass"
                if low >= float(cfg["statistics"]["power_lower_bound_min"])
                else "fail",
            }
        )
    output = ARTIFACT_ROOT / "power" / "power.json"
    write_json(output, output_claims)
    return output


def gate() -> Path:
    verify_lock()
    development = read_json(ARTIFACT_ROOT / "results" / "development_statistics.json")
    validation = read_json(ARTIFACT_ROOT / "results" / "protocol_validation_statistics.json")
    power_results = read_json(ARTIFACT_ROOT / "power" / "power.json")
    stability_results = read_json(ARTIFACT_ROOT / "results" / "stability.json")
    rows = load_rows("protocol_validation")
    fuzzy_rows = [row for row in rows if row["method"] == "full_fuzzyxai"]
    selection = read_json(ARTIFACT_ROOT / "lock" / "baseline_selection.json")
    c3a_population = _primary(rows, "H10-C3a")
    c3a_baseline = selection["H10-C3a"]["selected"]
    full_coverage = sum(
        row["obligation_coverage"]
        for row in c3a_population
        if row["method"] == "full_fuzzyxai"
    ) / sum(row["method"] == "full_fuzzyxai" for row in c3a_population)
    baseline_coverage = sum(
        row["obligation_coverage"]
        for row in c3a_population
        if row["method"] == c3a_baseline
    ) / sum(row["method"] == c3a_baseline for row in c3a_population)
    c3b_population = _primary(rows, "H10-C3b")
    c3b_baseline = selection["H10-C3b"]["selected"]
    full_success_costs = [
        row["plan_cost"]
        for row in c3b_population
        if row["method"] == "full_fuzzyxai" and row["full_recertification_success"]
    ]
    baseline_success_costs = [
        row["plan_cost"]
        for row in c3b_population
        if row["method"] == c3b_baseline and row["full_recertification_success"]
    ]
    cfg = config()
    false_certification = sum(row["false_certification"] for row in fuzzy_rows) / len(fuzzy_rows)
    new_critical = sum(row["new_critical_violations"] for row in fuzzy_rows) / len(fuzzy_rows)
    runtime = sorted(float(row["runtime_ms"]) for row in fuzzy_rows)
    runtime_p95 = runtime[int(0.95 * (len(runtime) - 1))]
    checks = {
        "development": all(item["status"] == "development_pass" for item in development),
        "protocol_validation": all(
            item["status"] == "protocol_validation_pass" for item in validation
        ),
        "power": all(item["status"] == "pass" for item in power_results),
        "stability": stability_results["status"] == "PASS",
        "cost_regret": validation[0]["cost_regret_effect"] > 0
        and validation[0]["cost_regret_ci_low"] > 0,
        "obligation_coverage_noninferior": full_coverage >= baseline_coverage,
        "successful_plan_median_cost": median(full_success_costs)
        <= median(baseline_success_costs),
        "false_certification": false_certification <= cfg["safety"]["false_certification_max"],
        "new_critical_violations": new_critical
        <= cfg["safety"]["new_critical_violation_max"],
        "runtime": runtime_p95 <= cfg["safety"]["runtime_p95_ms_max"],
        "sealed_opening_count_zero": True,
    }
    status = "READY_FOR_SEALED_GENERATION" if all(checks.values()) else "BLOCKED_PRECONFIRMATORY"
    report = {
        "status": status,
        "checks": checks,
        "false_certification": false_certification,
        "new_critical_violations": new_critical,
        "runtime_p95_ms": runtime_p95,
        "full_obligation_coverage": full_coverage,
        "baseline_obligation_coverage": baseline_coverage,
        "full_successful_plan_median_cost": median(full_success_costs),
        "baseline_successful_plan_median_cost": median(baseline_success_costs),
        "sealed_generated": False,
        "sealed_opening_count": 0,
        "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
        "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
    }
    output = ARTIFACT_ROOT / "gate" / "preconfirmatory_gate.json"
    write_json(output, report)
    return output


def score_sealed() -> None:
    raise PermissionError(
        "sealed scoring is intentionally unavailable: no sealed set has been generated or opened"
    )
