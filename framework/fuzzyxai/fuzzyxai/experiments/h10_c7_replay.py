from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import (
    DevelopmentIncident,
    GoldAtom,
    GoldLocalization,
    _graph,
    _metrics,
    _reject_gold,
    _row,
)
from fuzzyxai.repository_diagnostics.contract_inference_v2 import (
    evaluation_contract_family,
)
from fuzzyxai.repository_diagnostics.executed_slice import ExecutedSliceBuilder
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedDiagnosis,
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import IncidentQuery
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events

SOURCE_ARTIFACT_SHA256 = (
    "7b7bd0bba2eb9eef3955d2b5e313ecf4bccd02703427117f741c382e7658db09"
)
STRUCTURAL_VARIANTS = ("R0", "R1", "R3", "R5", "R6")
BASELINE_TARGETS = {
    "candidate_recall_at_10": 0.5666666666666667,
    "candidate_recall_at_20": 0.5666666666666667,
    "contract_accuracy": 0.13333333333333333,
    "coverage": 0.8,
    "joint_hit_at_3": 0.0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for value in values:
            target.write(json.dumps(value, sort_keys=True) + "\n")


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def verify_h10_c5c_baseline(
    per_incident_path: Path,
    status_path: Path,
) -> dict[str, object]:
    rows = list(csv.DictReader(per_incident_path.open(encoding="utf-8")))
    status = _json(status_path)
    selected = [row for row in rows if row["method"] == "O_ROUTE"]
    repositories = {row["repository"] for row in selected}
    recalculated = {
        "candidate_recall_at_10": _mean(selected, "candidate_recall_at_10"),
        "candidate_recall_at_20": _mean(selected, "candidate_recall_at_20"),
        "contract_accuracy": _mean(selected, "contract_accuracy"),
        "coverage_from_final_per_incident_status": _mean(selected, "coverage"),
        "joint_hit_at_3": _mean(
            selected,
            "joint_file_symbol_contract_hit_at_3",
        ),
    }
    published = dict(status["metrics"]["O_ROUTE"])
    checks = {
        "incident_count_30": len(selected) == 30,
        "repository_count_8": len(repositories) == 8,
        "recall_at_10_exact": (
            recalculated["candidate_recall_at_10"]
            == BASELINE_TARGETS["candidate_recall_at_10"]
        ),
        "recall_at_20_exact": (
            recalculated["candidate_recall_at_20"]
            == BASELINE_TARGETS["candidate_recall_at_20"]
        ),
        "contract_accuracy_exact": (
            recalculated["contract_accuracy"]
            == BASELINE_TARGETS["contract_accuracy"]
        ),
        "joint_hit_at_3_exact": (
            recalculated["joint_hit_at_3"]
            == BASELINE_TARGETS["joint_hit_at_3"]
        ),
        "published_coverage_exact": (
            float(published["coverage"]) == BASELINE_TARGETS["coverage"]
        ),
        "scientific_result_not_evaluated": (
            status["scientific_result"] == "NOT_EVALUATED"
        ),
        "held_out_not_created_or_scored": (
            not status["held_out_created"] and not status["held_out_scored"]
        ),
    }
    return {
        "status": (
            "H10_C7_OPEN_REPLAY_BASELINE_PASS"
            if all(checks.values())
            else "H10_C7_OPEN_REPLAY_BASELINE_FAIL"
        ),
        "checks": checks,
        "published_metrics": {
            key: published[key]
            for key in (
                "candidate_recall_at_10",
                "candidate_recall_at_20",
                "contract_accuracy",
                "coverage",
                "joint_hit_at_3",
            )
        },
        "recalculated_from_final_per_incident_records": recalculated,
        "coverage_representation_note": (
            "The locked status reports pre-threshold coverage 0.8. The final "
            "per-incident CSV records post-threshold coverage 0.7666666667. "
            "Both immutable representations are retained; neither is rewritten."
        ),
        "per_incident_sha256": _sha256(per_incident_path),
        "status_sha256": _sha256(status_path),
    }


def build_open_replay_bundle(
    *,
    source_artifact: Path,
    payload_root: Path,
    prepared_manifest: Path,
    prepared_gold: Path,
    repository_root: Path,
    output: Path,
) -> dict[str, object]:
    artifact_sha256 = _sha256(source_artifact)
    if artifact_sha256 != SOURCE_ARTIFACT_SHA256:
        raise ValueError(
            f"unexpected H10-C5c artifact SHA256: {artifact_sha256}"
        )
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    source_manifest_path = (
        payload_root / "runtime/H10_C5C_DEVELOPMENT_RUNTIME_ENRICHED.jsonl"
    )
    source_records = {
        str(item["incident_id"]): item for item in _jsonl(source_manifest_path)
    }
    prepared = _jsonl(prepared_manifest)
    gold = _jsonl(prepared_gold)
    if len(prepared) != 30 or len(gold) != 30:
        raise ValueError("open replay requires exactly 30 H10-C5c incidents")
    if {str(item["incident_id"]) for item in prepared} != set(source_records):
        raise ValueError("prepared and runtime incident sets differ")

    compact: list[dict[str, object]] = []
    slices = ExecutedSliceBuilder()
    for item in prepared:
        incident_id = str(item["incident_id"])
        source = source_records[incident_id]
        graph = item["graph"]
        graph_path = output / "repository_graphs" / f"{incident_id}.json"
        _write_json(graph_path, graph)

        runtime_path = _resolve(source_manifest_path.parent, source["runtime_events_path"])
        copied_runtime = output / "runtime_events" / f"{incident_id}.jsonl"
        copied_runtime.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime_path, copied_runtime)
        executed = slices.build(load_runtime_events(runtime_path))
        _write_json(
            output / "executed_slices" / f"{incident_id}.json",
            [asdict(value) | {"sha256": value.sha256} for value in executed],
        )

        assertion_path = _resolve(
            source_manifest_path.parent,
            source["assertion_difference_path"],
        )
        assertion_target = output / "assertion_diffs" / f"{incident_id}.txt"
        assertion_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assertion_path, assertion_target)

        before_path = _resolve(source_manifest_path.parent, source["before_sources_path"])
        after_path = _resolve(source_manifest_path.parent, source["after_sources_path"])
        patch_path = _resolve(source_manifest_path.parent, source["patch_path"])
        observable_snapshot = (
            output / "source_snapshots" / "observable" / incident_id
        )
        gold_snapshot = output / "source_snapshots" / "gold" / incident_id
        observable_snapshot.mkdir(parents=True, exist_ok=True)
        gold_snapshot.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(before_path, observable_snapshot / "before_sources.json")
        shutil.copyfile(after_path, gold_snapshot / "after_sources.json")
        shutil.copyfile(patch_path, gold_snapshot / "fix.patch")

        record = dict(item)
        record.pop("graph")
        record.update(
            {
                "graph_path": f"repository_graphs/{incident_id}.json",
                "runtime_events_path": f"runtime_events/{incident_id}.jsonl",
                "executed_slice_path": f"executed_slices/{incident_id}.json",
                "assertion_difference_path": f"assertion_diffs/{incident_id}.txt",
                "before_sources_path": (
                    f"source_snapshots/observable/{incident_id}/before_sources.json"
                ),
            }
        )
        compact.append(record)

    _write_jsonl(output / "incidents.jsonl", compact)
    shutil.copyfile(prepared_gold, output / "gold.jsonl")
    shutil.copyfile(
        repository_root / "results/h10_c5c/DEVELOPMENT_PER_INCIDENT.csv",
        output / "development_per_incident.csv",
    )
    shutil.copyfile(
        repository_root / "results/h10_c5c/EXECUTED_SLICE_MANIFEST.csv",
        output / "executed_slice_manifest.csv",
    )
    shutil.copyfile(
        repository_root
        / "results/h10_c5c_posthoc/ORACLE_DECOMPOSITION_PER_INCIDENT.csv",
        output / "oracle_decomposition.csv",
    )
    shutil.copyfile(
        repository_root / "results/h10_c5c/H10_C5C_DEVELOPMENT_STATUS.json",
        output / "h10_c5c_development_status.json",
    )
    baseline = verify_h10_c5c_baseline(
        output / "development_per_incident.csv",
        output / "h10_c5c_development_status.json",
    )
    _write_json(output / "BASELINE_REPLAY_STATUS.json", baseline)
    if baseline["status"] != "H10_C7_OPEN_REPLAY_BASELINE_PASS":
        raise ValueError("H10-C5c baseline replay verification failed")

    identity = {
        "status": "H10_C7_OPEN_REPLAY_BUNDLE_READY",
        "scientific_result": "NOT_EVALUATED",
        "source_artifact_sha256": artifact_sha256,
        "incident_count": 30,
        "repository_count": len({str(item["repository"]) for item in compact}),
        "network_required": False,
        "project_setup_required": False,
        "failing_tests_reexecuted": False,
        "held_out_created": False,
        "held_out_scored": False,
    }
    _write_json(output / "BUNDLE_IDENTITY.json", identity)
    (output / "README.md").write_text(
        "# H10-C7 open replay bundle\n\n"
        "This bundle replays the 30 disclosed H10-C5c development incidents "
        "without network access, repository setup, or failing-test execution. "
        "Gold is development-only and physically separated from observable "
        "records. The bundle is not a scientific result.\n",
        encoding="utf-8",
    )
    _write_sha256sums(output)
    return identity


def _write_sha256sums(root: Path) -> None:
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    content = "".join(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in files
    )
    (root / "SHA256SUMS").write_text(content, encoding="utf-8")


def _candidate_rank(
    diagnosis: GuidedDiagnosis,
    gold_atoms: tuple[object, ...],
) -> int | None:
    for index, candidate in enumerate(diagnosis.candidates, start=1):
        if any(
            candidate.file_path == atom.file_path
            and candidate.symbol == atom.symbol
            for atom in gold_atoms
        ):
            return index
    return None


def _error_card(
    *,
    incident: object,
    diagnosis: GuidedDiagnosis,
    gold: object,
    available_runtime_signals: list[str],
) -> dict[str, object]:
    rank = _candidate_rank(diagnosis, gold.atoms)
    predicted_contract = (
        evaluation_contract_family(diagnosis.candidates[0].contract.family)
        if diagnosis.candidates
        else "UNKNOWN_CONTRACT"
    )
    gold_contracts = sorted({atom.contract for atom in gold.atoms})
    contract_supported = predicted_contract in gold_contracts
    if rank is None:
        reason = "CANDIDATE_ABSENT_FROM_POOL"
    elif rank > 10:
        reason = "CANDIDATE_LOW_RANKED"
    elif not contract_supported:
        reason = "CONTRACT_INFERENCE_MISS"
    elif rank > 3:
        reason = "JOINT_LOCALIZATION_MISS"
    else:
        reason = "MATCH"
    selected = diagnosis.candidates[rank - 1] if rank else None
    return {
        "incident_id": incident.incident_id,
        "repository": incident.repository,
        "variant": diagnosis.variant,
        "gold_atoms": [asdict(atom) for atom in gold.atoms],
        "gold_rank": rank,
        "gold_in_top_20": bool(rank),
        "gold_in_top_10": bool(rank and rank <= 10),
        "predicted_contract": predicted_contract,
        "gold_contracts": gold_contracts,
        "contract_supported": contract_supported,
        "error_class": reason,
        "available_runtime_signals_for_gold": available_runtime_signals,
        "gold_candidate_rank_sources": (
            list(selected.rank_sources) if selected is not None else []
        ),
        "top_20": [
            {
                "file_path": candidate.file_path,
                "symbol": candidate.symbol,
                "score": candidate.score,
                "rank_sources": list(candidate.rank_sources),
                "contracts": [
                    {
                        "family": item.family,
                        "evaluation_family": evaluation_contract_family(
                            item.family
                        ),
                        "confidence": item.confidence,
                        "evidence": list(item.evidence),
                    }
                    for item in candidate.contract_hypotheses
                ],
            }
            for candidate in diagnosis.candidates
        ],
    }


def _gold_runtime_signals(incident: object, gold: object) -> list[str]:
    gold_nodes = {
        node.node_id
        for node in incident.graph.nodes
        if any(
            node.file_path == atom.file_path and node.symbol == atom.symbol
            for atom in gold.atoms
        )
    }
    relations = {
        edge.relation
        for edge in incident.graph.edges
        if edge.source in gold_nodes or edge.target in gold_nodes
    }
    signals = []
    if "executes" in relations or "tested_by" in relations:
        signals.append("executed_in_failing_path")
    if "produces" in relations:
        signals.append("traceback_frame")
    if "runtime_calls" in relations:
        signals.append("dynamic_call")
    return signals


def _replay_records(
    bundle: Path,
) -> tuple[
    tuple[dict[str, object], ...],
    dict[str, GoldLocalization],
]:
    observable = _jsonl(bundle / "incidents.jsonl")
    for index, value in enumerate(observable):
        _reject_gold(value, f"$[{index}]")
    gold = {}
    for value in _jsonl(bundle / "gold.jsonl"):
        identifier = str(value["incident_id"])
        gold[identifier] = GoldLocalization(
            identifier,
            tuple(
                GoldAtom(
                    str(atom["file_path"]),
                    str(atom["symbol"]) if atom.get("symbol") is not None else None,
                    str(atom["contract"]),
                )
                for atom in value["atoms"]
            ),
        )
    ids = {str(item["incident_id"]) for item in observable}
    if len(observable) != 30 or len({str(item["repository"]) for item in observable}) != 8:
        raise ValueError("open replay requires 30 incidents from 8 repositories")
    if ids != set(gold):
        raise ValueError("open replay observable and Gold incident sets differ")
    return observable, gold


def run_open_replay_tournament(
    *,
    bundle: Path,
    output: Path,
    engine: GuidedNaturalDiagnosisEngine,
) -> dict[str, object]:
    baseline = _json(bundle / "BASELINE_REPLAY_STATUS.json")
    if baseline["status"] != "H10_C7_OPEN_REPLAY_BASELINE_PASS":
        raise ValueError("structural replay is blocked by baseline mismatch")
    records, gold = _replay_records(bundle)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cards: list[dict[str, object]] = []
    for value in records:
        graph_path = (bundle / str(value["graph_path"])).resolve()
        query = value["query"]
        incident = DevelopmentIncident(
            str(value["incident_id"]),
            str(value["repository"]),
            IncidentQuery(
                str(value["incident_id"]),
                str(query.get("issue", "")),
                tuple(str(item) for item in query.get("failing_tests", [])),
                str(query.get("traceback", "")),
                str(query.get("assertion", "")),
            ),
            _graph(_json(graph_path)),
            int(value["repository_symbol_count"]),
        )
        available_runtime_signals = _gold_runtime_signals(
            incident,
            gold[incident.incident_id],
        )
        for variant in STRUCTURAL_VARIANTS:
            diagnosis = engine.diagnose(
                incident.graph,
                incident.query,
                variant,
            )
            rows.append(_row(incident, diagnosis, gold[incident.incident_id], 0.0))
            cards.append(
                _error_card(
                    incident=incident,
                    diagnosis=diagnosis,
                    gold=gold[incident.incident_id],
                    available_runtime_signals=available_runtime_signals,
                )
            )

    fieldnames = list(rows[0])
    with (output / "OPEN_REPLAY_MODEL_MATRIX.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_jsonl(output / "ERROR_CARDS.jsonl", cards)

    metrics = {
        variant: _metrics([row for row in rows if row["variant"] == variant])
        for variant in STRUCTURAL_VARIANTS
    }
    old_rows = list(
        csv.DictReader(
            (bundle / "development_per_incident.csv").open(encoding="utf-8")
        )
    )
    old_route = [row for row in old_rows if row["method"] == "O_ROUTE"]
    old_by_repo = {
        repository: statistics.fmean(
            float(row["candidate_recall_at_10"])
            for row in old_route
            if row["repository"] == repository
        )
        for repository in {row["repository"] for row in old_route}
    }
    repository_improvements: dict[str, int] = {}
    for variant in STRUCTURAL_VARIANTS:
        variant_rows = [row for row in rows if row["variant"] == variant]
        repository_improvements[variant] = sum(
            statistics.fmean(
                float(row["candidate_recall_at_10"])
                for row in variant_rows
                if row["repository"] == repository
            )
            > old_value
            for repository, old_value in old_by_repo.items()
        )
    baseline_greedy = _json(
        bundle / "h10_c5c_development_status.json"
    )["metrics"]["B_GREEDY"]
    checks_by_variant = {}
    for variant, values in metrics.items():
        checks_by_variant[variant] = {
            "recall_at_10_at_least_0_70": values["recall_at_10"] >= 0.70,
            "recall_at_20_at_least_0_80": values["recall_at_20"] >= 0.80,
            "contract_macro_f1_at_least_0_40": (
                values["contract_macro_f1"] >= 0.40
            ),
            "coverage_at_least_0_80": values["coverage"] >= 0.80,
            "false_localization_not_worse_than_b_greedy": (
                values["false_localization"]
                <= float(baseline_greedy["false_localization"])
            ),
            "improved_in_at_least_6_repositories": (
                repository_improvements[variant] >= 6
            ),
            "recall_gain_at_least_0_10": (
                values["recall_at_10"]
                - BASELINE_TARGETS["candidate_recall_at_10"]
                >= 0.10
            ),
            "contract_gain_at_least_0_15": (
                values["contract_macro_f1"]
                - BASELINE_TARGETS["contract_accuracy"]
                >= 0.15
            ),
        }
    passing = [
        variant
        for variant, checks in checks_by_variant.items()
        if all(checks.values())
    ]
    winner = max(
        STRUCTURAL_VARIANTS,
        key=lambda variant: (
            metrics[variant]["recall_at_10"],
            metrics[variant]["recall_at_20"],
            metrics[variant]["contract_macro_f1"],
            metrics[variant]["joint_hit_at_3"],
            -metrics[variant]["false_localization"],
        ),
    )
    error_summary = {
        variant: dict(
            sorted(
                Counter(
                    card["error_class"]
                    for card in cards
                    if card["variant"] == variant
                ).items()
            )
        )
        for variant in STRUCTURAL_VARIANTS
    }
    result = {
        "status": (
            "H10_C7_OPEN_REPLAY_GO"
            if passing
            else "H10_C7_OPEN_REPLAY_NO_GO"
        ),
        "scientific_result": "NOT_EVALUATED",
        "execution_profile": "STRUCTURAL_ONLY_OFFLINE",
        "variants": list(STRUCTURAL_VARIANTS),
        "best_structural_variant": winner,
        "go_variants": passing,
        "metrics": metrics,
        "checks": checks_by_variant,
        "repository_improvements": repository_improvements,
        "error_summary": error_summary,
        "development_incidents": len(records),
        "development_repositories": len(
            {str(item["repository"]) for item in records}
        ),
        "new_development_data_collected": False,
        "project_environments_installed": False,
        "neural_models_executed": False,
        "held_out_created": False,
        "held_out_scored": False,
    }
    _write_json(output / "OPEN_REPLAY_STATUS.json", result)
    _write_json(output / "ERROR_SUMMARY.json", error_summary)
    return result
