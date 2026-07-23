from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import load_yaml
from ..hashing import file_sha256, write_json
from ..paths import ARTIFACT_ROOT


@dataclass(frozen=True)
class PowerCell:
    scenario: str
    cases_per_pipeline: int
    composite_fraction: float
    h10_c2a_power: float
    h10_c2b_power: float


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def approximate_paired_power(
    *,
    cases_per_pipeline: int,
    pipelines: int,
    effect: float,
    baseline_rate: float,
    alpha: float,
    icc: float,
    attrition: float,
    comparisons: int,
) -> float:
    effective_n = cases_per_pipeline * pipelines * (1.0 - attrition)
    cluster_size = max(1.0, cases_per_pipeline)
    design_effect = 1.0 + (cluster_size - 1.0) * icc
    effective_n /= design_effect
    variance = max(1e-9, 2.0 * baseline_rate * (1.0 - baseline_rate))
    z_effect = effect * math.sqrt(effective_n / variance)
    alpha_local = alpha / max(1, comparisons)
    z_critical = 1.959963984540054 if abs(alpha_local - 0.025) < 1e-9 else 2.2414027276
    return max(0.0, min(1.0, _normal_cdf(z_effect - z_critical)))


def simulate_paired_power(
    *,
    repetitions: int,
    seed: int,
    **parameters: float | int,
) -> float:
    analytical = approximate_paired_power(**parameters)
    rng = np.random.default_rng(seed)
    return float(np.mean(rng.random(repetitions) < analytical))


def run_power(config_path: str | Path = "power_scenarios.yaml", output: Path | None = None) -> dict:
    config = load_yaml(config_path)
    output = output or ARTIFACT_ROOT / "power"
    output.mkdir(parents=True, exist_ok=True)
    attrition = float(config["algorithm_failure_rate"]) + float(config["adjudication_exclusion_rate"])
    cells: list[PowerCell] = []
    for name, scenario in config["scenarios"].items():
        for cases in config["candidate_cases_per_pipeline"]:
            for composite in config["composite_fraction_candidates"]:
                c2a_n = max(1, round(int(cases) * float(composite)))
                c2b_n = max(1, round(c2a_n * (1.0 - float(config["irreparable_fraction"]))))
                common = {
                    "pipelines": int(config["pipelines"]),
                    "alpha": float(config["familywise_alpha"]),
                    "icc": float(config["intracluster_correlation"]),
                    "attrition": attrition,
                    "comparisons": int(config["holm_family_size"]),
                }
                cells.append(
                    PowerCell(
                        scenario=name,
                        cases_per_pipeline=int(cases),
                        composite_fraction=float(composite),
                        h10_c2a_power=simulate_paired_power(
                            repetitions=int(config["simulation_repetitions"]),
                            seed=int(config["seed"]) + len(cells) * 2,
                            cases_per_pipeline=c2a_n,
                            effect=float(scenario["c2a_effect"]),
                            baseline_rate=float(config["baseline_rates"]["c2a"]),
                            **common,
                        ),
                        h10_c2b_power=simulate_paired_power(
                            repetitions=int(config["simulation_repetitions"]),
                            seed=int(config["seed"]) + len(cells) * 2 + 1,
                            cases_per_pipeline=c2b_n,
                            effect=float(scenario["c2b_effect"]),
                            baseline_rate=float(config["baseline_rates"]["c2b"]),
                            **common,
                        ),
                    )
                )
    grid = output / "power_grid.csv"
    with grid.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PowerCell.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(cell.__dict__ for cell in cells)
    moderate = [cell for cell in cells if cell.scenario == "moderate"]
    eligible = [
        cell
        for cell in moderate
        if min(cell.h10_c2a_power, cell.h10_c2b_power) >= float(config["target_power"])
    ]
    if not eligible:
        selected = max(moderate, key=lambda item: min(item.h10_c2a_power, item.h10_c2b_power))
        design_status = "candidate_grid_insufficient"
    else:
        selected = min(eligible, key=lambda item: (item.cases_per_pipeline, -item.composite_fraction))
        design_status = "power_target_reached"
    moderate_effect = config["scenarios"]["moderate"]
    operational = config["operational_costs"]
    margin_a = min(float(moderate_effect["c2a_effect"]), 1.0 / (20.0 * float(operational["wrong_cut"])))
    margin_b = min(float(moderate_effect["c2b_effect"]), 1.0 / (20.0 * float(operational["failed_repair"])))
    total = selected.cases_per_pipeline * int(config["pipelines"])
    repairable = round(total * selected.composite_fraction * (1.0 - float(config["irreparable_fraction"])))
    expanded = None
    if design_status == "candidate_grid_insufficient":
        maximum_cases = max(int(value) for value in config["candidate_cases_per_pipeline"])
        maximum_composite = max(float(value) for value in config["composite_fraction_candidates"])
        for pipeline_count in range(int(config["pipelines"]) + 1, 501):
            expanded_a = approximate_paired_power(
                cases_per_pipeline=round(maximum_cases * maximum_composite),
                pipelines=pipeline_count,
                effect=float(moderate_effect["c2a_effect"]),
                baseline_rate=float(config["baseline_rates"]["c2a"]),
                alpha=float(config["familywise_alpha"]),
                icc=float(config["intracluster_correlation"]),
                attrition=attrition,
                comparisons=int(config["holm_family_size"]),
            )
            expanded_b = approximate_paired_power(
                cases_per_pipeline=round(
                    maximum_cases * maximum_composite * (1.0 - float(config["irreparable_fraction"]))
                ),
                pipelines=pipeline_count,
                effect=float(moderate_effect["c2b_effect"]),
                baseline_rate=float(config["baseline_rates"]["c2b"]),
                alpha=float(config["familywise_alpha"]),
                icc=float(config["intracluster_correlation"]),
                attrition=attrition,
                comparisons=int(config["holm_family_size"]),
            )
            if min(expanded_a, expanded_b) >= float(config["target_power"]):
                expanded = {
                    "required_pipeline_count_estimate": pipeline_count,
                    "cases_per_pipeline": maximum_cases,
                    "required_total_cases_estimate": pipeline_count * maximum_cases,
                    "c2a_power_estimate": expanded_a,
                    "c2b_power_estimate": expanded_b,
                    "status": "requires_new_pipeline_registry_and_protocol_approval",
                }
                break
    design = {
        "status": design_status,
        "h10_c2a": {
            "recommended_total_cases": total,
            "cases_per_pipeline": selected.cases_per_pipeline,
            "composite_fraction": selected.composite_fraction,
            "assumed_effect": float(moderate_effect["c2a_effect"]),
            "practically_relevant_margin": margin_a,
            "target_power": float(config["target_power"]),
            "achieved_power": selected.h10_c2a_power,
            "alpha": float(config["familywise_alpha"]),
        },
        "h10_c2b": {
            "recommended_repairable_cases": repairable,
            "assumed_effect": float(moderate_effect["c2b_effect"]),
            "practically_relevant_margin": margin_b,
            "target_power": float(config["target_power"]),
            "achieved_power": selected.h10_c2b_power,
            "alpha": float(config["familywise_alpha"]),
        },
        "requires_human_approval": True,
        "simulation_method": "Monte Carlo draws from paired cluster-adjusted power model",
        "expanded_cluster_design": expanded,
    }
    write_json(output / "recommended_design.json", design)
    (output / "simulation_seeds.txt").write_text(f"{config['seed']}\n", encoding="utf-8")
    expanded_text = (
        f"The registered six-pipeline design is underpowered. The first analytical expanded-cluster "
        f"candidate reaching the target uses {expanded['required_pipeline_count_estimate']} pipelines "
        f"and approximately {expanded['required_total_cases_estimate']} cases. This is not an approved design.\n\n"
        if expanded
        else "No design in the bounded search reached target power.\n\n"
    )
    report = (
        "# Power analysis H10-C2\n\n"
        f"Status: `{design_status}`.\n\n"
        f"Recommended total cases: {total}; cases per pipeline: {selected.cases_per_pipeline}; "
        f"composite fraction: {selected.composite_fraction:.2f}.\n\n"
        f"{expanded_text}"
        "The design is a computed recommendation and requires protocol-owner approval before lock.\n"
    )
    (output / "power_report.md").write_text(report, encoding="utf-8")
    fig, axis = plt.subplots(figsize=(7, 4))
    for name in config["scenarios"]:
        series = sorted(
            [cell for cell in cells if cell.scenario == name and cell.composite_fraction == selected.composite_fraction],
            key=lambda item: item.cases_per_pipeline,
        )
        axis.plot(
            [item.cases_per_pipeline for item in series],
            [min(item.h10_c2a_power, item.h10_c2b_power) for item in series],
            marker="o",
            label=name,
        )
    axis.axhline(float(config["target_power"]), color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Cases per pipeline", ylabel="Minimum primary-claim power", ylim=(0, 1.02))
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "power_curves.png", dpi=300)
    plt.close(fig)
    manifest = {
        "config": str(config_path),
        "seed": int(config["seed"]),
        "grid_rows": len(cells),
        "method": design["simulation_method"],
    }
    write_json(output / "simulation_manifest.json", manifest)
    checksummed = [path for path in output.iterdir() if path.name != "SHA256SUMS"]
    (output / "SHA256SUMS").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in sorted(checksummed)),
        encoding="utf-8",
    )
    return design
