from __future__ import annotations

import csv
import datetime as dt
import json
import os
import subprocess
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
            seed=730_000 + (1 if claim.endswith("a") else 2),
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
                seed=730_003,
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
        seed=740_000,
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


def _build_protocol_lock() -> Path:
    power_path = ARTIFACT_ROOT / "power" / "power.json"
    selected = read_json(power_path)["selected_designs"]
    if not selected or any(item is None for item in selected):
        raise RuntimeError("power analysis has no passing R4 design")
    tracked = _tracked_method_files()
    tracked.extend(
        (
            ARTIFACT_ROOT / "lock" / "baseline_selection.json",
            ARTIFACT_ROOT / "lock" / "method_lock.json",
            ARTIFACT_ROOT / "template_audit.json",
            power_path,
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


def _sealed_cases(lock: dict[str, object]) -> tuple[object, ...]:
    designs = {
        item["claim"]: item
        for item in lock["selected_designs"]
    }
    templates_per_family = max(
        int(item["templates_per_family"])
        for item in designs.values()
    )
    cases_per_template = max(
        int(item["cases_per_template"])
        for item in designs.values()
    )
    allocations = {
        stratum: max(
            int(item["stratum_allocation"][stratum])
            for item in designs.values()
        )
        for stratum in ("S2", "S3", "S4", "S5")
    }
    bank = _bank("sealed")
    selected = []
    for pipeline in sorted(
        {item.pipeline_family for item in bank}
    ):
        for stratum, count in allocations.items():
            pipeline_templates = sorted(
                (
                    item
                    for item in bank
                    if item.pipeline_family == pipeline
                    and item.stratum == stratum
                ),
                key=lambda item: item.canonical_hash,
            )
            if len(pipeline_templates) < count:
                raise RuntimeError(
                    f"sealed bank lacks {pipeline}/{stratum} templates"
                )
            selected.extend(pipeline_templates[:count])
    expected = templates_per_family * len(
        {item.pipeline_family for item in bank}
    )
    if len(selected) != expected:
        raise AssertionError("sealed stratum allocation is inconsistent")
    return build_cases(
        tuple(selected),
        cases_per_template=cases_per_template,
    )


def generate_sealed() -> Path:
    gate_report = read_json(
        ARTIFACT_ROOT / "gate" / "preconfirmatory_gate.json"
    )
    if gate_report["status"] != "READY_FOR_SEALED_GENERATION":
        raise PermissionError("R4 gate does not permit sealed generation")
    lock = _verify_protocol_lock()
    sealed_root = ARTIFACT_ROOT / "sealed"
    if sealed_root.exists():
        raise FileExistsError("sealed R4 set already exists")
    cases = _sealed_cases(lock)
    manifest = write_cases(sealed_root, "sealed", cases)
    status = {
        "status": "READY_FOR_SEALED_SCORING",
        "sealed_created": True,
        "sealed_opening_count": 0,
        "manifest": manifest,
        "protocol_lock_sha256": file_sha256(
            ARTIFACT_ROOT / "lock" / "protocol_lock.json"
        ),
        "H10-C3a": "NOT_EVALUATED_CONFIRMATORY",
        "H10-C3b": "NOT_EVALUATED_CONFIRMATORY",
    }
    output = sealed_root / "sealed_status.json"
    write_json(output, status)
    return output


def score_sealed(approval_path: str | None) -> Path:
    if not approval_path:
        raise PermissionError(
            "sealed scoring requires an explicit approval file"
        )
    approval = Path(approval_path)
    if not approval.is_file():
        raise PermissionError("sealed approval file is missing")
    status_path = (
        ARTIFACT_ROOT / "sealed" / "sealed_status.json"
    )
    status = read_json(status_path)
    if int(status["sealed_opening_count"]) != 0:
        raise PermissionError("sealed R4 scoring cannot be repeated")
    lock = _verify_protocol_lock()
    manifest_path = (
        ARTIFACT_ROOT
        / "sealed"
        / "data"
        / "sealed"
        / "manifest.json"
    )
    approval_payload = read_json(approval)
    required = {
        "study_id": lock["study_id"],
        "protocol_lock_sha256": file_sha256(
            ARTIFACT_ROOT / "lock" / "protocol_lock.json"
        ),
        "sealed_manifest_sha256": file_sha256(manifest_path),
        "authorization": "AUTHORIZE_ONE_TIME_SEALED_SCORING",
    }
    if any(
        approval_payload.get(key) != value
        for key, value in required.items()
    ):
        raise PermissionError("sealed approval does not match the R4 lock")
    opening_path = ARTIFACT_ROOT / "sealed" / "opening_record.json"
    if opening_path.exists():
        raise PermissionError("sealed R4 opening record already exists")
    opening = {
        "study_id": lock["study_id"],
        "opening_count_before": 0,
        "opening_count_after": 1,
        "opened_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip(),
        "protocol_lock_sha256": required["protocol_lock_sha256"],
        "sealed_manifest_sha256": required["sealed_manifest_sha256"],
        "approval_sha256": file_sha256(approval),
        "purpose": "one_time_confirmatory_scoring",
    }
    # Record opening before reconstructing private outcomes.
    write_json(opening_path, opening)
    status.update(
        {
            "status": "SEALED_SCORING_IN_PROGRESS",
            "sealed_opening_count": 1,
        }
    )
    write_json(status_path, status)
    try:
        rows = [
            row
            for case in _sealed_cases(lock)
            for row in _run_case(case)
        ]
        output = ARTIFACT_ROOT / "results" / "sealed.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)
        analyze("sealed", rows)
    except Exception as exc:
        status.update(
            {
                "status": "SEALED_SCORING_FAILED_NO_REUSE",
                "error_type": type(exc).__name__,
            }
        )
        write_json(status_path, status)
        raise
    status.update(
        {
            "status": "SEALED_SCORED",
            "results_sha256": file_sha256(output),
            "H10-C3a": "EVALUATED_CONFIRMATORY",
            "H10-C3b": "EVALUATED_CONFIRMATORY",
        }
    )
    write_json(status_path, status)
    return output
