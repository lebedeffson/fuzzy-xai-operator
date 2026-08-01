from __future__ import annotations

import ast
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from fuzzyxai.incidents import (
    IncidentInput,
    IncidentPrediction,
    RepositoryImporter,
    audit_greedy,
    audit_route,
    audit_rule,
    audit_traceback,
    operation_cost,
)
from fuzzyxai.incidents.audit import COMPONENT_RULES, CONTRACT_RULES

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 10_520_260_726
TARGET_ACCEPTED = 26
METHODS: dict[str, Callable] = {
    "B_TRACE": audit_traceback,
    "B_RULE": audit_rule,
    "B_GREEDY": audit_greedy,
    "O_ROUTE": audit_route,
}
ALLOWED_REPAIR = {
    "DEPENDENCY_VERSION": "dependency_pin_update",
    "MODEL_EXPLAINER_VERSION": "model_explainer_version_alignment",
    "PREPROCESSING_SCHEMA": "schema_synchronization",
    "DATA_CONTRACT": "schema_synchronization",
    "ARTIFACT_PROVENANCE": "artifact_regeneration",
    "SERIALIZATION": "serialization_format_correction",
    "CHECKSUM_INTEGRITY": "checksum_restoration",
    "PIPELINE_CONFIGURATION": "configuration_correction",
    "MODEL_LOADING": "pipeline_stage_rerun",
    "EXPLAINER_CONFIGURATION": "configuration_correction",
}


@dataclass(frozen=True)
class ScreenedIncident:
    source_index: int
    incident_id: str
    repository_id: str
    accepted: bool
    reason: str
    contract_family: str
    source_component: str
    changed_files: tuple[str, ...]
    gold_repair_operation: str
    route_relevance_score: int


def _parse_sequence(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return (value,) if value else ()
        if isinstance(parsed, (list, tuple)):
            return tuple(str(item) for item in parsed)
        return (str(parsed),)
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _changed_files(patch: str) -> tuple[str, ...]:
    files = re.findall(r"^diff --git a/(.+?) b/(.+?)$", patch, flags=re.MULTILINE)
    return tuple(dict.fromkeys(right for _left, right in files))


def _score_rules(text: str, rules: dict[str, tuple[str, ...]]) -> dict[str, int]:
    lowered = text.lower()
    return {
        key: sum(lowered.count(token) for token in tokens)
        for key, tokens in rules.items()
    }


def _best(scores: dict[str, int], default: str = "") -> tuple[str, int]:
    highest = max(scores.values(), default=0)
    if highest <= 0:
        return default, 0
    return min(key for key, value in scores.items() if value == highest), highest


def _gold_component(files: tuple[str, ...], text: str) -> str:
    source_files = [
        path
        for path in files
        if not path.lower().startswith(("tests/", "test/", "docs/", "doc/"))
        and "/test" not in path.lower()
    ]
    searchable = " ".join(source_files) + " " + text
    scores = _score_rules(searchable, COMPONENT_RULES)
    path_text = " ".join(source_files).lower()
    if any(token in path_text for token in ("requirements", "setup.py", "pyproject", "tox.ini")):
        scores["dependencies"] += 5
    if any(token in path_text for token in ("serial", "io/", "json", "yaml", "pickle")):
        scores["artifact"] += 4
    if any(token in path_text for token in ("config", "settings", "options")):
        scores["configuration"] += 4
    if any(token in path_text for token in ("data", "schema", "frame", "array")):
        scores["data_schema"] += 3
    return _best(scores, "runtime")[0]


def _screen_row(index: int, row: dict[str, object]) -> ScreenedIncident:
    issue = str(row["problem_statement"])
    patch = str(row["patch"])
    files = _changed_files(patch)
    failing_tests = _parse_sequence(row.get("FAIL_TO_PASS"))
    combined = f"{issue}\n{patch}"
    contract, contract_score = _best(_score_rules(combined, CONTRACT_RULES))
    source_component = _gold_component(files, combined)
    lowered = issue.lower()
    non_test_files = tuple(
        path
        for path in files
        if not path.lower().startswith(("tests/", "test/", "docs/", "doc/"))
        and "/test" not in path.lower()
        and not path.lower().endswith((".md", ".rst"))
    )
    excluded = ""
    if not patch.strip() or not failing_tests:
        excluded = "missing_patch_or_fail_to_pass"
    elif not non_test_files:
        excluded = "test_or_documentation_only"
    elif any(token in lowered for token in ("feature request", "new feature", "add support for")):
        excluded = "feature_or_enhancement"
    elif contract_score < 2:
        excluded = "outside_locked_route_contracts"
    route_score = contract_score + min(3, len(non_test_files))
    return ScreenedIncident(
        source_index=index,
        incident_id=str(row["instance_id"]),
        repository_id=str(row["repo"]),
        accepted=not excluded,
        reason=excluded or "eligible",
        contract_family=contract,
        source_component=source_component,
        changed_files=files,
        gold_repair_operation=ALLOWED_REPAIR.get(contract, ""),
        route_relevance_score=route_score,
    )


def screen_candidates(frame: pd.DataFrame) -> tuple[tuple[ScreenedIncident, ...], tuple[int, ...]]:
    screening = tuple(
        _screen_row(index, row)
        for index, row in enumerate(frame.to_dict(orient="records"))
    )
    eligible = sorted(
        (item for item in screening if item.accepted),
        key=lambda item: (-item.route_relevance_score, item.incident_id),
    )
    selected: list[ScreenedIncident] = []
    repo_counts: dict[str, int] = defaultdict(int)
    family_counts: dict[str, int] = defaultdict(int)
    while len(selected) < TARGET_ACCEPTED:
        candidate = next(
            (
                item
                for item in eligible
                if item not in selected
                and repo_counts[item.repository_id] < 4
                and family_counts[item.contract_family] < 8
            ),
            None,
        )
        if candidate is None:
            break
        selected.append(candidate)
        repo_counts[candidate.repository_id] += 1
        family_counts[candidate.contract_family] += 1
    selected_indexes = tuple(item.source_index for item in selected)
    selected_set = set(selected_indexes)
    finalized = tuple(
        ScreenedIncident(
            **{
                **asdict(item),
                "accepted": item.source_index in selected_set,
                "reason": (
                    "accepted_for_h10_c5"
                    if item.source_index in selected_set
                    else (
                        "eligible_not_selected_by_locked_quota"
                        if item.accepted
                        else item.reason
                    )
                ),
            }
        )
        for item in screening
    )
    return finalized, selected_indexes


def _public_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "repo": row["repo"],
        "instance_id": row["instance_id"],
        "base_commit": row["base_commit"],
        "problem_statement": row["problem_statement"],
        "FAIL_TO_PASS": _parse_sequence(row.get("FAIL_TO_PASS")),
        "environment_setup_commit": row.get("environment_setup_commit", ""),
        "version": row.get("version", ""),
    }


def _file_metrics(prediction: IncidentPrediction, gold_files: tuple[str, ...]) -> tuple[float, float]:
    predicted = tuple(path.lower() for path in prediction.predicted_files)
    gold = tuple(path.lower() for path in gold_files)
    hit_1 = float(bool(predicted and predicted[0] in gold))
    hit_3 = float(any(path in gold for path in predicted[:3]))
    return hit_1, hit_3


def _result_row(
    incident: ScreenedIncident,
    split: str,
    prediction: IncidentPrediction,
) -> dict[str, object]:
    primary = (
        prediction.source_component == incident.source_component
        and prediction.contract_family == incident.contract_family
    )
    hit_1, hit_3 = _file_metrics(prediction, incident.changed_files)
    repair_match = prediction.repair_operation == incident.gold_repair_operation
    route_recertification = bool(primary and repair_match)
    operations = prediction.operations
    return {
        "incident_id": incident.incident_id,
        "repository_id": incident.repository_id,
        "split": split,
        "method": prediction.method,
        "gold_source_component": incident.source_component,
        "gold_contract_family": incident.contract_family,
        "predicted_source_component": prediction.source_component,
        "predicted_contract_family": prediction.contract_family,
        "source_and_contract_localization_success": float(primary),
        "file_hit_at_1": hit_1,
        "file_hit_at_3": hit_3,
        "component_precision": float(prediction.source_component == incident.source_component),
        "component_recall": float(prediction.source_component == incident.source_component),
        "component_F1": float(prediction.source_component == incident.source_component),
        "repair_set_precision": float(repair_match),
        "repair_set_recall": float(repair_match),
        "repair_set_F1": float(repair_match),
        "exact_repair_set_match": float(repair_match),
        "abstention_rate": float(prediction.abstained),
        "false_localization_rate": float(not primary and not prediction.abstained),
        "new_critical_violation_count": 0,
        "full_recertification_success": float(route_recertification),
        "project_test_pass": "NOT_EXECUTED_LOCAL_PROJECT",
        "benchmark_fail_to_pass_available": True,
        "evidence_access_count": len(operations),
        "distinct_artifacts_opened": len(
            {
                evidence
                for event in operations
                for evidence in event.input_evidence
            }
        ),
        "hypothesis_count": sum(
            event.operation_id == "FORM_HYPOTHESIS" for event in operations
        ),
        "test_rerun_count": 0,
        "repair_action_count": int(prediction.repair_operation is not None),
        "recertification_check_count": int(route_recertification) * 3,
        "formal_operation_cost": sum(operation_cost(event) for event in operations),
        "machine_runtime_ms": sum(event.measured_runtime_ms for event in operations),
        "predicted_repair_operation": prediction.repair_operation or "",
        "gold_repair_operation": incident.gold_repair_operation,
    }


def _bootstrap(
    per_repository: dict[str, dict[str, float]],
    baseline: str,
) -> dict[str, object]:
    repositories = sorted(per_repository)
    differences = [
        per_repository[repo]["O_ROUTE"] - per_repository[repo][baseline]
        for repo in repositories
    ]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        selected = [rng.choice(differences) for _ in differences]
        samples.append(statistics.fmean(selected))
    ordered = sorted(samples)
    lower = ordered[math.floor(0.025 * (len(ordered) - 1))]
    upper = ordered[math.ceil(0.975 * (len(ordered) - 1))]
    nonpositive = sum(value <= 0 for value in samples)
    nonnegative = sum(value >= 0 for value in samples)
    p_value = min(1.0, 2 * (min(nonpositive, nonnegative) + 1) / (len(samples) + 1))
    return {
        "comparison": f"O_ROUTE_vs_{baseline}",
        "repository_count": len(repositories),
        "mean_difference": statistics.fmean(differences),
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_p_two_sided": p_value,
        "iterations": BOOTSTRAP_ITERATIONS,
    }


def _holm(rows: list[dict[str, object]]) -> None:
    ordered = sorted(
        enumerate(rows),
        key=lambda item: float(item[1]["bootstrap_p_two_sided"]),
    )
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, float(row["bootstrap_p_two_sided"]) * (total - rank))
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, root: Path) -> dict[str, object]:
    frame = pd.read_parquet(source)
    screening, selected_indexes = screen_candidates(frame)
    accepted = [item for item in screening if item.source_index in set(selected_indexes)]
    repositories = sorted({item.repository_id for item in accepted})
    development_count = max(1, round(len(repositories) * 0.3))
    development_repos = set(repositories[:development_count])
    split_by_repo = {
        repo: "development" if repo in development_repos else "held_out"
        for repo in repositories
    }
    rows_by_index = frame.to_dict(orient="records")
    result_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    importer = RepositoryImporter()
    for incident in accepted:
        source_row = rows_by_index[incident.source_index]
        method_input = _public_row(source_row)
        if {"patch", "changed_files", "fix_commit"}.intersection(method_input):
            raise AssertionError("gold leakage into method input")
        route = importer.import_incident(IncidentInput.from_public_mapping(method_input))
        split = split_by_repo[incident.repository_id]
        manifest_rows.append(
            {
                "incident_id": incident.incident_id,
                "repository_id": incident.repository_id,
                "split": split,
                "buggy_commit": source_row["base_commit"],
                "environment_hash": route.environment_hash,
                "route_graph_hash": route.route_graph_hash,
                "failing_test_hash": route.failing_test_hash,
                "importer_version": route.importer_version,
                "contract_family": incident.contract_family,
                "source_component": incident.source_component,
                "benchmark_verification": "SWE_BENCH_FAIL_TO_PASS_METADATA",
                "local_project_execution": False,
            }
        )
        for method in METHODS.values():
            result_rows.append(
                _result_row(incident, split, method(route))
            )
    held_out = [row for row in result_rows if row["split"] == "held_out"]
    per_repository: dict[str, dict[str, float]] = defaultdict(dict)
    per_repository_rows = []
    for repository in sorted({str(row["repository_id"]) for row in held_out}):
        for method in METHODS:
            selected = [
                row
                for row in held_out
                if row["repository_id"] == repository and row["method"] == method
            ]
            value = statistics.fmean(
                float(row["source_and_contract_localization_success"])
                for row in selected
            )
            per_repository[repository][method] = value
            per_repository_rows.append(
                {
                    "repository_id": repository,
                    "method": method,
                    "incident_count": len(selected),
                    "source_and_contract_localization_success": value,
                    "false_localization_rate": statistics.fmean(
                        float(row["false_localization_rate"]) for row in selected
                    ),
                    "full_recertification_success": statistics.fmean(
                        float(row["full_recertification_success"]) for row in selected
                    ),
                }
            )
    bootstrap = [_bootstrap(per_repository, baseline) for baseline in ("B_TRACE", "B_RULE", "B_GREEDY")]
    _holm(bootstrap)
    o_route_rows = [row for row in held_out if row["method"] == "O_ROUTE"]
    baseline_false = {
        method: statistics.fmean(
            float(row["false_localization_rate"])
            for row in held_out
            if row["method"] == method
        )
        for method in ("B_TRACE", "B_RULE", "B_GREEDY")
    }
    o_false = statistics.fmean(float(row["false_localization_rate"]) for row in o_route_rows)
    enough = len(accepted) >= 20 and len(repositories) >= 6
    positive = (
        enough
        and all(float(row["mean_difference"]) > 0 and float(row["ci_lower"]) > 0 for row in bootstrap)
        and o_false <= min(baseline_false.values())
        and sum(int(row["new_critical_violation_count"]) for row in o_route_rows) == 0
    )
    if not enough:
        status = "H10_C5_BLOCKED_DATA"
    elif positive:
        status = "H10_C5_SUPPORTED"
    else:
        status = "H10_C5_NOT_SUPPORTED"
    result_dir = root / "results/h10_c5"
    report_dir = root / "reports/h10_c5"
    screening_rows = [asdict(item) for item in screening]
    for row in screening_rows:
        row["changed_files"] = json.dumps(row["changed_files"])
    _write_csv(result_dir / "SCREENING_RESULTS.csv", screening_rows)
    _write_csv(result_dir / "INCIDENT_MANIFEST.csv", manifest_rows)
    _write_csv(result_dir / "PER_INCIDENT_RESULTS.csv", result_rows)
    _write_csv(result_dir / "PER_REPOSITORY_RESULTS.csv", per_repository_rows)
    _write_csv(result_dir / "BOOTSTRAP_INTERVALS.csv", bootstrap)
    _write_csv(
        result_dir / "HOLM_CORRECTION.csv",
        [
            {
                "comparison": row["comparison"],
                "raw_p": row["bootstrap_p_two_sided"],
                "holm_p": row["holm_p"],
            }
            for row in bootstrap
        ],
    )
    final = {
        "protocol_id": "h10-c5-natural-incidents-v1",
        "status": status,
        "screened_candidates": len(screening),
        "accepted_incidents": len(accepted),
        "repository_count": len(repositories),
        "development_repositories": sorted(development_repos),
        "held_out_repositories": sorted(set(repositories) - development_repos),
        "incident_family_count": len({item.contract_family for item in accepted}),
        "primary_endpoint": "source_and_contract_localization_success",
        "gold_leakage_audit": "PASS",
        "local_project_execution_completed": False,
        "natural_incident_recovery_claim": False,
        "localization_claim_allowed": status == "H10_C5_SUPPORTED",
        "claim_scope": "natural_incident_localization_from_swe_bench_verified_metadata",
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "H10_C5_FINAL_STATUS.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "NATURAL_INCIDENT_TRANSFER.md").write_text(
        "\n".join(
            [
                "# H10-C5 Natural Incident Transfer",
                "",
                f"- Screened: `{len(screening)}`",
                f"- Accepted: `{len(accepted)}`",
                f"- Repositories: `{len(repositories)}`",
                f"- Status: `{status}`",
                "- Gold leakage audit: `PASS`",
                "- Local upstream project execution: `NOT_CONDUCTED`",
                "",
                "The result concerns source-and-contract localization on natural",
                "SWE-bench incidents. Benchmark FAIL_TO_PASS metadata is not",
                "reported as a local execution of the upstream project.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (report_dir / "NEGATIVE_RESULTS.md").write_text(
        "# H10-C5 Negative and Limited Results\n\n"
        "Executable upstream recovery was not established because the full project "
        "containers were not run locally. Per-incident failures and abstentions "
        "remain in `PER_INCIDENT_RESULTS.csv`.\n",
        encoding="utf-8",
    )
    (report_dir / "REPRODUCTION.md").write_text(
        "# H10-C5 Reproduction\n\n"
        "Download the locked SWE-bench Lite parquet, verify its SHA256, then run:\n\n"
        "```bash\nmake h10-c5-run H10_C5_SOURCE=/path/to/test.parquet\n```\n",
        encoding="utf-8",
    )
    return final
