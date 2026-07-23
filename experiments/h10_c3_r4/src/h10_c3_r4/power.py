from __future__ import annotations

import math
import random
from statistics import pstdev

from .statistics import (
    paired_template_differences,
    wilson_interval,
)

DESIGN_GRID = (
    (4, 20, 1),
    (5, 20, 1),
    (6, 20, 1),
    (6, 30, 1),
    (6, 40, 1),
    (6, 60, 1),
    (6, 80, 1),
    (6, 110, 1),
    (6, 40, 2),
    (6, 40, 4),
)


def _normal_quantile_holm_two_sided() -> float:
    # Three registered endpoints use Holm; the first two-sided alpha is 0.05/3.
    return 2.3939797998185104


def simulate_design(
    rows: list[dict[str, object]],
    *,
    claim: str,
    baseline: str,
    metric: str,
    pipeline_families: int,
    templates_per_family: int,
    cases_per_template: int,
    simulations: int,
    seed: int,
    margin: float,
    intracluster_correlation: float,
    eligible_template_fraction: float = 1.0,
) -> dict[str, object]:
    grouped = paired_template_differences(rows, baseline, metric)
    empirical_pipeline_values = {
        pipeline: [
            sum(values) / len(values)
            for values in templates.values()
        ]
        for pipeline, templates in grouped.items()
    }
    available = sorted(empirical_pipeline_values)
    if pipeline_families > len(available):
        raise ValueError("design requests unavailable pipeline families")
    all_values = [
        value
        for values in empirical_pipeline_values.values()
        for value in values
    ]
    heterogeneity = max(pstdev(all_values), 0.05)
    within_sd = max(
        heterogeneity
        * math.sqrt(
            max(
                0.0,
                (1 - intracluster_correlation)
                / max(intracluster_correlation, 1e-9),
            )
        ),
        0.02,
    )
    rng = random.Random(seed)
    critical_z = _normal_quantile_holm_two_sided()
    successes = 0
    for _ in range(simulations):
        sampled_pipelines = rng.sample(
            available,
            pipeline_families,
        )
        template_means = []
        for pipeline in sampled_pipelines:
            source = empirical_pipeline_values[pipeline]
            for _ in range(templates_per_family):
                if rng.random() > eligible_template_fraction:
                    continue
                latent = rng.choice(source)
                case_values = [
                    latent + rng.gauss(0.0, within_sd)
                    for _ in range(cases_per_template)
                ]
                template_means.append(
                    sum(case_values) / len(case_values)
                )
        if len(template_means) < 2:
            continue
        effect = sum(template_means) / len(template_means)
        standard_error = (
            pstdev(template_means)
            / math.sqrt(len(template_means))
            if len(template_means) > 1
            else float("inf")
        )
        lower = effect - critical_z * standard_error
        if effect >= margin and lower > 0:
            successes += 1
    point = successes / simulations
    low, high = wilson_interval(successes, simulations)
    design_effect = 1 + (
        cases_per_template - 1
    ) * intracluster_correlation
    return {
        "claim": claim,
        "baseline": baseline,
        "pipeline_families": pipeline_families,
        "templates_per_family": templates_per_family,
        "cases_per_template": cases_per_template,
        "effective_independent_units": (
            pipeline_families
            * templates_per_family
            * cases_per_template
            / design_effect
        ),
        "intracluster_correlation": intracluster_correlation,
        "eligible_template_fraction": eligible_template_fraction,
        "holm_family_size": 3,
        "best_baseline_selected_on_development": True,
        "point_power": point,
        "lower_confidence_bound": low,
        "upper_confidence_bound": high,
        "number_of_simulations": simulations,
        "monte_carlo_standard_error": math.sqrt(
            point * (1 - point) / simulations
        ),
        "status": "pass" if low >= 0.80 else "fail",
    }


def design_power_analysis(
    rows: list[dict[str, object]],
    *,
    selections: dict[str, str],
    simulations: int,
    seed: int,
    margins: dict[str, float],
    intracluster_correlation: float,
) -> dict[str, object]:
    claims = (
        ("H10-C3a", "optimal_set_membership"),
        ("H10-C3b", "full_recertification_success"),
    )
    results = []
    selected = []
    for claim_index, (claim, metric) in enumerate(claims):
        certified = {
            "CERTIFIED_UNIQUE",
            "CERTIFIED_MULTIPLE_OPTIMA",
        }
        full_rows = [
            row
            for row in rows
            if row["stratum"] in {"S2", "S3", "S4", "S5"}
        ]
        eligible_rows = [
            row
            for row in full_rows
            if row["gold_status"] in certified
            and (
                claim != "H10-C3b"
                or bool(row["repairable"])
            )
        ]
        total_templates = {
            (row["pipeline_family"], row["template_hash"])
            for row in full_rows
            if row["method"] == "full_h10"
        }
        eligible_templates = {
            (row["pipeline_family"], row["template_hash"])
            for row in eligible_rows
            if row["method"] == "full_h10"
        }
        eligible_fraction = len(eligible_templates) / len(
            total_templates
        )
        claim_results = [
            simulate_design(
                eligible_rows,
                claim=claim,
                baseline=selections[claim],
                metric=metric,
                pipeline_families=pipelines,
                templates_per_family=templates,
                cases_per_template=cases,
                simulations=simulations,
                seed=seed
                + claim_index * 100_000
                + templates * 100
                + cases,
                margin=margins[claim],
                intracluster_correlation=intracluster_correlation,
                eligible_template_fraction=eligible_fraction,
            )
            for pipelines, templates, cases in DESIGN_GRID
        ]
        results.extend(claim_results)
        passing = [
            item
            for item in claim_results
            if item["status"] == "pass"
            and item["pipeline_families"] == len(
                empirical_pipeline_families(eligible_rows)
            )
        ]
        selected.append(
            min(
                passing,
                key=lambda item: (
                    item["effective_independent_units"],
                    item["cases_per_template"],
                ),
            )
            if passing
            else None
        )
    return {
        "design_grid": results,
        "selected_designs": selected,
        "status": (
            "PASS"
            if all(item is not None for item in selected)
            else "FAIL"
        ),
        "unit_of_analysis": (
            "pipeline_family -> route_template -> generated_case"
        ),
        "case_id_as_independent_unit": False,
        "holm_family_size": 3,
        "design_varies_pipeline_families": True,
        "design_varies_independent_templates": True,
        "design_varies_cases_per_template": True,
    }


def empirical_pipeline_families(
    rows: list[dict[str, object]],
) -> set[str]:
    return {
        str(row["pipeline_family"])
        for row in rows
        if row["method"] == "full_h10"
    }
