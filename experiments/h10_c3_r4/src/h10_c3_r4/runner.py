from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import yaml

from gold_oracle.h10_c3_r4_oracle import derive_gold

from .generator import build_cases, write_cases
from .methods import (
    BASELINES,
    public_candidates,
    run_baseline,
    run_full_h10,
)
from .models import R4Gold
from .power import design_power_analysis
from .scoring import score
from .scientific_classifier import classify_confirmatory_result
from .secure_sealed import (
    create_secure_protocol_lock,
    create_secure_sealed,
    load_secure_sealed_cases,
    mark_scoring_complete,
    mark_scoring_failed,
)
from .statistics import (
    hierarchical_bootstrap,
    holm,
    pipeline_effects,
)
from .templates import (
    audit_banks,
    build_template_bank,
    write_template_bank,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT_ROOT = REPO_ROOT / "experiments" / "h10_c3_r4"
ARTIFACT_ROOT = Path(
    os.environ.get(
        "H10_C3_R4_ARTIFACT_ROOT",
        REPO_ROOT / "artifacts" / "h10_c3_r4",
    )
)
CONFIG_PATH = EXPERIMENT_ROOT / "config.yaml"
TEMPLATE_ROOT = EXPERIMENT_ROOT / "templates"


def config() -> dict[str, object]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def template_audit() -> Path:
    banks = {
        split: build_template_bank(split)
        for split in (
            "development",
            "protocol_validation",
            "sealed",
        )
    }
    report = audit_banks(banks)
    for split, templates in banks.items():
        write_template_bank(
            TEMPLATE_ROOT / split / "templates.jsonl",
            templates,
        )
        manifest = {
            "split": split,
            "template_count": len(templates),
            "canonical_hashes": [
                template.canonical_hash for template in templates
            ],
            "graph_hashes": [
                template.graph_hash for template in templates
            ],
            "mutation_hashes": [
                template.mutation_hash for template in templates
            ],
            "coverage_hashes": [
                template.coverage_hash for template in templates
            ],
            "repair_dependency_hashes": [
                template.repair_dependency_hash
                for template in templates
            ],
            "templates_sha256": file_sha256(
                TEMPLATE_ROOT / split / "templates.jsonl"
            ),
        }
        write_json(
            TEMPLATE_ROOT / split / "manifest.json",
            manifest,
        )
    output = ARTIFACT_ROOT / "template_audit.json"
    write_json(output, report)
    return output


def _bank(split: str):
    return build_template_bank(split)


def generate(split: str) -> Path:
    if split == "sealed":
        raise PermissionError(
            "use the fail-closed generate_sealed command after the R4 gate"
        )
    if split == "protocol_validation":
        verify_method_lock()
    cases = build_cases(
        _bank(split),
        cases_per_template=int(
            config()["cases_per_template"][split]
        ),
    )
    manifest = write_cases(ARTIFACT_ROOT, split, cases)
    output = ARTIFACT_ROOT / "data" / split / "manifest.json"
    write_json(output, manifest)
    return output


def _gold(case: object, candidates: tuple[object, ...]) -> R4Gold:
    obligations = tuple(
        sorted(
            {
                obligation
                for candidate in candidates
                for obligation in candidate.covers
            }
        )
    )
    payload = derive_gold(
        private_record=case.private_record(),
        public_candidates=tuple(
            {
                "candidate_id": candidate.candidate_id,
                "covers": candidate.covers,
                "cost": candidate.cost,
            }
            for candidate in candidates
        ),
        obligations=obligations,
        repairable=case.repairable,
    )
    return R4Gold(case_id=case.case_id, **payload)


def _run_case(case: object) -> list[dict[str, object]]:
    candidates = public_candidates(case.mutated_graph)
    gold = _gold(case, candidates)
    results = [
        *(run_baseline(name, case.mutated_graph) for name in BASELINES),
        run_full_h10(case.mutated_graph),
    ]
    rows = []
    for result in results:
        metrics = score(
            case.mutated_graph,
            candidates,
            gold,
            result,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "template_id": case.template_id,
                "template_hash": case.template_hash,
                "pipeline_family": case.pipeline_family,
                "modality": case.modality,
                "split": case.split,
                "stratum": case.stratum,
                "gold_status": gold.status,
                "repairable": gold.repairable,
                "method": result.method,
                "predicted_cut": json.dumps(result.cut),
                "predicted_cost": result.predicted_cost,
                "runtime_ms": result.runtime_ms,
                **metrics,
            }
        )
    return rows


def run(split: str) -> Path:
    if split == "protocol_validation":
        verify_method_lock()
    cases = build_cases(
        _bank(split),
        cases_per_template=int(
            config()["cases_per_template"][split]
        ),
    )
    rows = [
        row
        for case in cases
        for row in _run_case(case)
    ]
    output = ARTIFACT_ROOT / "results" / f"{split}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)
    if split == "development":
        select_baselines(rows)
    analyze(split, rows)
    return output


def load_rows(split: str) -> list[dict[str, object]]:
    with (
        ARTIFACT_ROOT / "results" / f"{split}.csv"
    ).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    boolean_fields = (
        "repairable",
        "optimal_set_membership",
        "full_recertification_success",
        "false_certification",
        "all_required_postconditions_verified",
        "graph_changed",
    )
    numeric_fields = (
        "predicted_cost",
        "runtime_ms",
        "raw_cost_regret",
        "normalized_cost_regret",
        "obligation_coverage",
        "completed_steps",
        "failed_steps",
        "plan_cost",
        "remaining_critical_issue_count",
        "new_critical_violation_count",
    )
    for row in rows:
        for field in boolean_fields:
            row[field] = str(row[field]).lower() in {
                "true",
                "1",
                "1.0",
            }
        for field in numeric_fields:
            row[field] = (
                None
                if row[field] in {"", "None"}
                else float(row[field])
            )
    return rows


def _primary(
    rows: list[dict[str, object]],
    claim: str,
) -> list[dict[str, object]]:
    certified = {
        "CERTIFIED_UNIQUE",
        "CERTIFIED_MULTIPLE_OPTIMA",
    }
    selected = [
        row
        for row in rows
        if row["stratum"] in config()["primary_population"]
        and row["gold_status"] in certified
    ]
    if claim == "H10-C3b":
        selected = [
            row for row in selected if bool(row["repairable"])
        ]
    return selected


def select_baselines(rows: list[dict[str, object]]) -> Path:
    selection = {}
    for claim, metric in (
        ("H10-C3a", "optimal_set_membership"),
        ("H10-C3b", "full_recertification_success"),
    ):
        population = _primary(rows, claim)
        summaries = {}
        for method in BASELINES:
            method_rows = [
                row for row in population if row["method"] == method
            ]
            summaries[method] = {
                "metric": sum(
                    float(row[metric]) for row in method_rows
                )
                / len(method_rows),
                "mean_cost_regret": sum(
                    float(row["normalized_cost_regret"])
                    for row in method_rows
                )
                / len(method_rows),
            }
        best = max(
            BASELINES,
            key=lambda method: (
                summaries[method]["metric"],
                -summaries[method]["mean_cost_regret"],
                method,
            ),
        )
        selection[claim] = {
            "selected": best,
            "development_summary": summaries[best],
            "all_baselines": summaries,
        }
    output = ARTIFACT_ROOT / "lock" / "baseline_selection.json"
    write_json(output, selection)
    return output


def _selected_baselines() -> dict[str, str]:
    payload = read_json(
        ARTIFACT_ROOT / "lock" / "baseline_selection.json"
    )
    return {
        claim: str(payload[claim]["selected"])
        for claim in ("H10-C3a", "H10-C3b")
    }


def analyze(
    split: str,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    rows = rows or load_rows(split)
    selections = _selected_baselines()
    cfg = config()
    results = []
    for claim, metric in (
        ("H10-C3a", "optimal_set_membership"),
        ("H10-C3b", "full_recertification_success"),
    ):
        population = _primary(rows, claim)
        baseline = selections[claim]
        result = hierarchical_bootstrap(
            population,
            baseline,
            metric,
            repetitions=int(
                cfg["statistics"]["bootstrap_repetitions"]
            ),
            seed=int(cfg["seeds"]["bootstrap"]) + (
                1 if claim.endswith("a") else 2
            ),
        )
        result.update(
            {
                "claim": claim,
                "metric": metric,
                "baseline": baseline,
                "margin": float(
                    cfg["practical_margins"][claim]
                ),
                "pipeline_effects": pipeline_effects(
                    population,
                    baseline,
                    metric,
                ),
            }
        )
        if claim == "H10-C3a":
            regret = hierarchical_bootstrap(
                population,
                baseline,
                "normalized_cost_regret",
                repetitions=int(
                    cfg["statistics"][
                        "bootstrap_repetitions"
                    ]
                ),
                seed=int(cfg["seeds"]["bootstrap"]) + 3,
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
    results[0]["cost_regret_p_holm"] = endpoint_tests[1][
        "p_holm"
    ]
    results[1]["p_holm"] = endpoint_tests[2]["p_holm"]
    for result in results:
        positive_pipelines = sum(
            effect > 0
            for effect in result["pipeline_effects"].values()
        )
        cost_pass = (
            result["claim"] != "H10-C3a"
            or (
                result["cost_regret_effect"] > 0
                and result["cost_regret_ci_low"] > 0
                and result["cost_regret_p_holm"] < 0.05
            )
        )
        result["positive_pipeline_families"] = positive_pipelines
        result["status"] = (
            f"{split}_pass"
            if result["effect"] >= result["margin"]
            and result["ci_low"] > 0
            and result["p_holm"] < 0.05
            and positive_pipelines >= 5
            and cost_pass
            else f"{split}_fail"
        )
    output = (
        ARTIFACT_ROOT
        / "results"
        / f"{split}_statistics.json"
    )
    write_json(output, results)
    return output


def _tracked_method_files() -> list[Path]:
    return [
        CONFIG_PATH,
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "requirements.lock",
        REPO_ROOT / "framework" / "fuzzyxai" / "pyproject.toml",
        *sorted(
            (
                EXPERIMENT_ROOT / "src" / "h10_c3_r4"
            ).glob("*.py")
        ),
        REPO_ROOT / "gold_oracle" / "h10_c3_r4_oracle.py",
        *sorted(
            (
                REPO_ROOT
                / "framework"
                / "fuzzyxai"
                / "fuzzyxai"
                / "diagnostics"
            ).glob("*.py")
        ),
        *sorted(TEMPLATE_ROOT.rglob("*.json*")),
    ]


def freeze_method() -> Path:
    baseline = (
        ARTIFACT_ROOT / "lock" / "baseline_selection.json"
    )
    template_report = ARTIFACT_ROOT / "template_audit.json"
    if not baseline.is_file() or not template_report.is_file():
        raise RuntimeError(
            "development and template audit must precede method lock"
        )
    lock = {
        "study_id": config()["study_id"],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "files": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for path in _tracked_method_files()
        },
        "baseline_selection_sha256": file_sha256(baseline),
        "template_audit_sha256": file_sha256(template_report),
        "post_lock_tuning": False,
        "sealed_created": False,
        "sealed_opening_count": 0,
    }
    output = ARTIFACT_ROOT / "lock" / "method_lock.json"
    write_json(output, lock)
    return output


def verify_method_lock() -> None:
    path = ARTIFACT_ROOT / "lock" / "method_lock.json"
    if not path.is_file():
        raise PermissionError("R4 method lock is missing")
    lock = read_json(path)
    for relative, expected in lock["files"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise PermissionError(
                f"post-lock method change: {relative}"
            )
    baseline = (
        ARTIFACT_ROOT / "lock" / "baseline_selection.json"
    )
    if file_sha256(baseline) != lock[
        "baseline_selection_sha256"
    ]:
        raise PermissionError("post-lock baseline change")


def stability() -> Path:
    rows = load_rows("protocol_validation")
    selections = _selected_baselines()
    checks = []
    for claim, metric in (
        ("H10-C3a", "optimal_set_membership"),
        ("H10-C3b", "full_recertification_success"),
    ):
        population = _primary(rows, claim)
        effects = pipeline_effects(
            population,
            selections[claim],
            metric,
        )
        checks.append(
            {
                "claim": claim,
                "pipeline_effects": effects,
                "positive_pipeline_families": sum(
                    value > 0 for value in effects.values()
                ),
                "status": (
                    "PASS"
                    if sum(value > 0 for value in effects.values())
                    >= 5
                    and all(value >= 0 for value in effects.values())
                    else "FAIL"
                ),
            }
        )
    output = ARTIFACT_ROOT / "results" / "stability.json"
    write_json(
        output,
        {
            "checks": checks,
            "status": (
                "PASS"
                if all(item["status"] == "PASS" for item in checks)
                else "FAIL"
            ),
        },
    )
    return output


def power() -> Path:
    rows = load_rows("protocol_validation")
    cfg = config()
    payload = design_power_analysis(
        rows,
        selections=_selected_baselines(),
        simulations=int(
            cfg["statistics"]["power_simulations"]
        ),
        seed=int(cfg["seeds"]["power"]),
        margins={
            claim: float(value)
            for claim, value in cfg["practical_margins"].items()
        },
        intracluster_correlation=float(
            cfg["statistics"]["intracluster_correlation"]
        ),
    )
    output = ARTIFACT_ROOT / "power" / "power.json"
    write_json(output, payload)
    return output


def _environment_manifest() -> Path:
    packages = {}
    for name in ("pytest", "pyyaml", "ruff"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    output = ARTIFACT_ROOT / "lock" / "environment_manifest.json"
    write_json(
        output,
        {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": packages,
        },
    )
    return output


def _independence_audit() -> Path:
    gold = (
        REPO_ROOT / "gold_oracle" / "h10_c3_r4_oracle.py"
    ).read_text(encoding="utf-8")
    methods = (
        EXPERIMENT_ROOT / "src" / "h10_c3_r4" / "methods.py"
    ).read_text(encoding="utf-8")
    forbidden_gold = (
        "fuzzyxai",
        "MinimalDiagnosticCutFinder",
        "ActionableRepairPlanner",
    )
    forbidden_baseline = (
        "gold_oracle",
        "mutation_log",
        "reverse_candidate",
    )
    checks = {
        "gold_independent": not any(
            token in gold for token in forbidden_gold
        ),
        "baselines_do_not_read_gold": not any(
            token in methods for token in forbidden_baseline
        ),
        "seven_registered_baselines": len(BASELINES) == 7,
    }
    output = ARTIFACT_ROOT / "audit" / "independence.json"
    write_json(
        output,
        {
            "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL",
        },
    )
    return output


def _build_protocol_lock() -> Path:
    power_path = ARTIFACT_ROOT / "power" / "power.json"
    selected = read_json(power_path)["selected_designs"]
    if not selected or any(item is None for item in selected):
        raise RuntimeError("power analysis has no passing R4 design")
    environment = _environment_manifest()
    independence = _independence_audit()
    tracked = _tracked_method_files()
    tracked.extend(
        (
            ARTIFACT_ROOT / "lock" / "baseline_selection.json",
            ARTIFACT_ROOT / "lock" / "method_lock.json",
            ARTIFACT_ROOT / "template_audit.json",
            power_path,
            environment,
            independence,
        )
    )
    lock = {
        "study_id": config()["study_id"],
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "files": {
            str(path.relative_to(REPO_ROOT)): file_sha256(path)
            for path in tracked
        },
        "selected_designs": selected,
        "metrics": config()["primary_metrics"],
        "margins": config()["practical_margins"],
        "statistics": config()["statistics"],
        "seeds": config()["seeds"],
        "sealed_design": {
            "pipeline_families": 6,
            "templates_per_family": max(
                int(item["templates_per_family"])
                for item in selected
            ),
            "cases_per_template": max(
                int(item["cases_per_template"])
                for item in selected
            ),
            "stratum_allocation": {
                stratum: max(
                    int(item["stratum_allocation"][stratum])
                    for item in selected
                )
                for stratum in ("S2", "S3", "S4", "S5")
            },
        },
        "report_structure": [
            "template_audit.json",
            "development_statistics.json",
            "protocol_validation_statistics.json",
            "stability.json",
            "power.json",
            "preconfirmatory_gate.json",
            "sealed_status.json",
        ],
        "exclusions": [
            "UNCERTIFIED_SOLVER_DISAGREEMENT",
            "INSUFFICIENT_FORMAL_SPECIFICATION",
            "NON_REPAIRABLE for H10-C3b",
        ],
        "sealed_opening_count": 0,
        "post_lock_tuning": False,
    }
    output = ARTIFACT_ROOT / "lock" / "protocol_lock.json"
    write_json(output, lock)
    return output


def gate() -> Path:
    verify_method_lock()
    independence_path = _independence_audit()
    template_report = read_json(
        ARTIFACT_ROOT / "template_audit.json"
    )
    development = read_json(
        ARTIFACT_ROOT
        / "results"
        / "development_statistics.json"
    )
    validation = read_json(
        ARTIFACT_ROOT
        / "results"
        / "protocol_validation_statistics.json"
    )
    stability_report = read_json(
        ARTIFACT_ROOT / "results" / "stability.json"
    )
    power_report = read_json(
        ARTIFACT_ROOT / "power" / "power.json"
    )
    rows = load_rows("protocol_validation")
    full = [row for row in rows if row["method"] == "full_h10"]
    runtime = sorted(float(row["runtime_ms"]) for row in full)
    false_certification = sum(
        bool(row["false_certification"]) for row in full
    ) / len(full)
    new_critical = sum(
        int(row["new_critical_violation_count"]) > 0 for row in full
    ) / len(full)
    runtime_p95 = runtime[int(0.95 * (len(runtime) - 1))]
    cfg = config()
    checks = {
        "template_independence": template_report["status"] == "PASS",
        "oracle_and_baseline_independence": (
            read_json(independence_path)["status"] == "PASS"
        ),
        "frozen_predecessor_unchanged": (
            subprocess.check_output(
                [
                    "git",
                    "rev-parse",
                    "fix/h10-c3-cost-stability-v23.1",
                ],
                cwd=REPO_ROOT,
                text=True,
            ).strip()
            == config()["frozen_predecessor"]
        ),
        "development": all(
            item["status"] == "development_pass"
            for item in development
        ),
        "protocol_validation": all(
            item["status"] == "protocol_validation_pass"
            for item in validation
        ),
        "stability": stability_report["status"] == "PASS",
        "power": power_report["status"] == "PASS",
        "false_certification": (
            false_certification
            <= float(cfg["safety"]["false_certification_max"])
        ),
        "new_critical_violations": (
            new_critical
            <= float(
                cfg["safety"]["new_critical_violation_max"]
            )
        ),
        "runtime": (
            runtime_p95
            <= float(cfg["safety"]["runtime_p95_ms_max"])
        ),
        "sealed_absent": not (
            ARTIFACT_ROOT / "sealed"
        ).exists(),
    }
    status = (
        "READY_FOR_SEALED_GENERATION"
        if all(checks.values())
        else "BLOCKED_PRECONFIRMATORY"
    )
    report = {
        "status": status,
        "checks": checks,
        "false_certification": false_certification,
        "new_critical_violations": new_critical,
        "runtime_p95_ms": runtime_p95,
        "sealed_created": False,
        "sealed_opening_count": 0,
        "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
        "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
        "human_adjudication": (
            "NOT_REQUIRED_FOR_ALGORITHMIC_SCOPE"
        ),
        "human_factors_validation": "NOT_CONDUCTED",
    }
    output = ARTIFACT_ROOT / "gate" / "preconfirmatory_gate.json"
    write_json(output, report)
    if status == "READY_FOR_SEALED_GENERATION":
        _build_protocol_lock()
    return output


def _verify_protocol_lock() -> dict[str, object]:
    path = ARTIFACT_ROOT / "lock" / "protocol_lock.json"
    if not path.is_file():
        raise PermissionError("final R4 protocol lock is missing")
    lock = read_json(path)
    for relative, expected in lock["files"].items():
        if file_sha256(REPO_ROOT / relative) != expected:
            raise PermissionError(
                f"post-protocol-lock change: {relative}"
            )
    return lock


def freeze_secure_protocol() -> Path:
    return create_secure_protocol_lock(REPO_ROOT, ARTIFACT_ROOT)


def generate_sealed(secret_path: str | None = None) -> Path:
    if not secret_path:
        raise PermissionError(
            "secure sealed generation requires SEALED_SECRET as an external path"
        )
    return create_secure_sealed(
        REPO_ROOT,
        ARTIFACT_ROOT,
        Path(secret_path),
    )


def _sealed_safety(rows: list[dict[str, object]]) -> dict[str, float]:
    full = [row for row in rows if row["method"] == "full_h10"]
    runtime = sorted(float(row["runtime_ms"]) for row in full)
    return {
        "false_certification": sum(
            bool(row["false_certification"]) for row in full
        )
        / len(full),
        "new_critical_violations": sum(
            int(row["new_critical_violation_count"]) > 0 for row in full
        )
        / len(full),
        "runtime_p95_ms": runtime[int(0.95 * (len(runtime) - 1))],
    }


def score_sealed(
    approval_path: str | None,
    secret_path: str | None = None,
) -> Path:
    if not approval_path or not secret_path:
        raise PermissionError(
            "sealed scoring requires APPROVAL and SEALED_SECRET paths"
        )
    cases = load_secure_sealed_cases(
        REPO_ROOT,
        ARTIFACT_ROOT,
        Path(approval_path),
        Path(secret_path),
    )
    try:
        rows = [
            row
            for case in cases
            for row in _run_case(case)
        ]
        output = ARTIFACT_ROOT / "results" / "sealed.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)
        statistics_path = analyze("sealed", rows)
        safety = _sealed_safety(rows)
        classification = classify_confirmatory_result(
            read_json(statistics_path),
            **safety,
        )
        mark_scoring_complete(
            ARTIFACT_ROOT,
            results_sha256=file_sha256(output),
            classification=classification,
        )
    except Exception as exc:
        mark_scoring_failed(ARTIFACT_ROOT, exc)
        raise
    return output
