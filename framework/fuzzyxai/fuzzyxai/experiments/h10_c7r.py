from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import (
    GoldAtom,
    GoldLocalization,
    _graph,
    _reject_gold,
)
from fuzzyxai.experiments.h10_c7a import (
    BudgetCase,
    FrozenBudgetRankingEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    IncidentQuery,
    RankedSymbol,
)
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events

R5_BUDGET = 20
BASELINE_BUDGET = 160
BOOTSTRAP_ITERATIONS = 20000
BOOTSTRAP_SEED = 7102026


@dataclass(frozen=True)
class HeldOutInputs:
    cases: tuple[BudgetCase, ...]
    gold: dict[str, GoldLocalization]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _resolve(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def load_held_out_inputs(
    manifest_path: Path,
    gold_path: Path,
    exclusion_lock_path: Path,
    *,
    minimum_incidents: int = 40,
    minimum_repositories: int = 12,
) -> HeldOutInputs:
    exclusion = json.loads(exclusion_lock_path.read_text(encoding="utf-8"))
    excluded = set(exclusion["excluded_repositories"])
    observable = read_jsonl(manifest_path)
    cases = []
    for index, value in enumerate(observable):
        _reject_gold(value, f"$[{index}]")
        if value.get("split") != "held_out":
            raise ValueError("H10-C7R accepts held_out records only")
        if value.get("runtime_evidence_status") != "BUG_REPRODUCED_WITH_TRACE":
            raise ValueError("H10-C7R requires reproduced project traceback")
        repository = str(value["repository"])
        if repository in excluded:
            raise ValueError(f"excluded repository in held-out: {repository}")
        query = value["query"]
        if not isinstance(query, dict):
            raise TypeError("query must be a mapping")
        graph_value = json.loads(
            _resolve(manifest_path.parent, value["graph_path"]).read_text(
                encoding="utf-8"
            )
        )
        graph = _graph(graph_value)
        cases.append(
            BudgetCase(
                incident_id=str(value["incident_id"]),
                repository=repository,
                query=IncidentQuery(
                    str(value["incident_id"]),
                    str(query.get("issue", "")),
                    tuple(str(item) for item in query.get("failing_tests", ())),
                    str(query.get("traceback", "")),
                    str(query.get("assertion", "")),
                ),
                graph=graph,
                runtime_events=load_runtime_events(
                    _resolve(
                        manifest_path.parent,
                        value["runtime_events_path"],
                    )
                ),
                repository_symbol_count=int(value["repository_symbol_count"]),
                repository_source_lines=int(
                    value.get("repository_source_lines", 0)
                ),
            )
        )
    gold = {}
    for value in read_jsonl(gold_path):
        identifier = str(value["incident_id"])
        atoms = tuple(
            GoldAtom(
                str(atom["file_path"]),
                str(atom["symbol"]) if atom.get("symbol") is not None else None,
                str(atom.get("contract", "NOT_SCORED")),
            )
            for atom in value["atoms"]
        )
        if not atoms:
            raise ValueError(f"empty held-out Gold: {identifier}")
        gold[identifier] = GoldLocalization(identifier, atoms)
    identifiers = [case.incident_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("held-out incident IDs must be unique")
    if set(identifiers) != set(gold):
        raise ValueError("held-out observable and Gold sets differ")
    repositories = {case.repository for case in cases}
    if len(cases) < minimum_incidents:
        raise ValueError("H10-C7R requires at least 40 held-out incidents")
    if len(repositories) < minimum_repositories:
        raise ValueError("H10-C7R requires at least 12 held-out repositories")
    return HeldOutInputs(tuple(cases), gold)


def _rank(
    candidates: Sequence[RankedSymbol],
    gold: GoldLocalization,
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if any(
            candidate.file_path == atom.file_path
            and candidate.symbol == atom.symbol
            for atom in gold.atoms
        ):
            return rank
    return None


def _repository_source_lines(case: BudgetCase) -> int:
    if case.repository_source_lines > 0:
        return case.repository_source_lines
    maxima: dict[str, int] = {}
    for node in case.graph.nodes:
        if not node.file_path:
            continue
        end = int(node.attributes.get("end_lineno", 0) or 0)
        maxima[node.file_path] = max(maxima.get(node.file_path, 0), end)
    return sum(maxima.values())


def score_incidents(
    inputs: HeldOutInputs,
    engine: FrozenBudgetRankingEngine,
) -> list[dict[str, object]]:
    rows = []
    for case in inputs.cases:
        r5 = engine.rank(case, "R5")
        baseline = engine.rank(case, "B_BM25")
        r5_candidates = r5.ranking[:R5_BUDGET]
        baseline_candidates = baseline.ranking[:BASELINE_BUDGET]
        gold = inputs.gold[case.incident_id]
        r5_rank = _rank(r5_candidates, gold)
        baseline_rank = _rank(baseline_candidates, gold)
        r5_count = len(r5_candidates)
        baseline_count = len(baseline_candidates)
        symbol_count = max(case.repository_symbol_count, 1)
        source_lines = max(_repository_source_lines(case), 1)
        r5_lines = sum(item.line_count for item in r5_candidates)
        baseline_lines = sum(item.line_count for item in baseline_candidates)
        r5_reduction = 1.0 - r5_count / symbol_count
        baseline_reduction = 1.0 - baseline_count / symbol_count
        rows.append(
            {
                "incident_id": case.incident_id,
                "repository": case.repository,
                "r5_rank": r5_rank or 0,
                "r5_hit_at_5": float(bool(r5_rank and r5_rank <= 5)),
                "r5_hit_at_10": float(bool(r5_rank and r5_rank <= 10)),
                "r5_hit_at_20": float(bool(r5_rank and r5_rank <= 20)),
                "r5_reciprocal_rank_at_20": (
                    1.0 / r5_rank if r5_rank else 0.0
                ),
                "baseline_rank": baseline_rank or 0,
                "baseline_hit_at_160": float(bool(baseline_rank)),
                "r5_candidate_count": r5_count,
                "baseline_candidate_count": baseline_count,
                "repository_symbol_count": symbol_count,
                "r5_search_space_reduction": r5_reduction,
                "baseline_search_space_reduction": baseline_reduction,
                "delta_reduction": r5_reduction - baseline_reduction,
                "r5_context_lines": r5_lines,
                "baseline_context_lines": baseline_lines,
                "repository_source_lines": source_lines,
                "r5_context_reduction": 1.0 - r5_lines / source_lines,
                "baseline_context_reduction": 1.0
                - baseline_lines / source_lines,
                "r5_runtime_ms": r5.runtime_ms,
                "baseline_runtime_ms": baseline.runtime_ms,
                "r5_available": not bool(r5.unavailable_reason),
                "baseline_available": not bool(baseline.unavailable_reason),
                "r5_top_20_sha256": hashlib.sha256(
                    json.dumps(
                        [item.node_id for item in r5_candidates],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
        )
    return rows


def repository_rows(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    values = []
    repositories = sorted({str(row["repository"]) for row in rows})
    for repository in repositories:
        selected = [row for row in rows if row["repository"] == repository]
        values.append(
            {
                "repository": repository,
                "incident_count": len(selected),
                "r5_recall_at_20": statistics.fmean(
                    float(row["r5_hit_at_20"]) for row in selected
                ),
                "baseline_recall_at_160": statistics.fmean(
                    float(row["baseline_hit_at_160"]) for row in selected
                ),
                "mean_delta_reduction": statistics.fmean(
                    float(row["delta_reduction"]) for row in selected
                ),
            }
        )
    return values


def repository_cluster_bootstrap(
    rows: Sequence[dict[str, object]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    by_repository: dict[str, list[float]] = {}
    for row in rows:
        by_repository.setdefault(str(row["repository"]), []).append(
            float(row["delta_reduction"])
        )
    repository_effects = {
        repository: statistics.fmean(values)
        for repository, values in by_repository.items()
    }
    repositories = sorted(repository_effects)
    if not repositories:
        raise ValueError("repository bootstrap requires data")
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected = [
            repository_effects[rng.choice(repositories)]
            for _ in repositories
        ]
        samples.append(statistics.fmean(selected))
    samples.sort()
    lower = samples[int(0.025 * iterations)]
    upper = samples[min(int(0.975 * iterations), iterations - 1)]
    return {
        "iterations": iterations,
        "seed": seed,
        "repository_count": len(repositories),
        "mean_difference": statistics.fmean(repository_effects.values()),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def final_status(
    rows: Sequence[dict[str, object]],
    bootstrap: dict[str, float | int],
    *,
    gold_leakage: int,
    method_signature_passed: bool,
    budget_signature_passed: bool,
    single_official_scoring: bool,
) -> dict[str, object]:
    incident_count = len(rows)
    repository_count = len({str(row["repository"]) for row in rows})
    r5_recall = statistics.fmean(
        float(row["r5_hit_at_20"]) for row in rows
    )
    baseline_recall = statistics.fmean(
        float(row["baseline_hit_at_160"]) for row in rows
    )
    coverage = statistics.fmean(
        float(bool(row["r5_candidate_count"])) for row in rows
    )
    checks = {
        "minimum_40_incidents": incident_count >= 40,
        "minimum_12_repositories": repository_count >= 12,
        "r5_recall_at_20_at_least_0_80": r5_recall >= 0.80,
        "r5_coverage_at_least_0_80": coverage >= 0.80,
        "r5_recall_not_worse_than_baseline_by_more_than_0_05": (
            r5_recall >= baseline_recall - 0.05
        ),
        "repository_cluster_ci_delta_reduction_lower_positive": (
            float(bootstrap["ci_lower"]) > 0.0
        ),
        "gold_leakage_zero": gold_leakage == 0,
        "frozen_method_signature_pass": method_signature_passed,
        "frozen_budget_signature_pass": budget_signature_passed,
        "single_official_scoring": single_official_scoring,
    }
    supported = all(checks.values())
    return {
        "protocol_id": "H10-C7R-v1",
        "status": (
            "H10_C7R_SUPPORTED"
            if supported
            else "H10_C7R_NOT_SUPPORTED"
        ),
        "scientific_result": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "incident_count": incident_count,
        "repository_count": repository_count,
        "r5_recall_at_20": r5_recall,
        "baseline_recall_at_160": baseline_recall,
        "r5_coverage": coverage,
        "mean_r5_search_space_reduction": statistics.fmean(
            float(row["r5_search_space_reduction"]) for row in rows
        ),
        "mean_baseline_search_space_reduction": statistics.fmean(
            float(row["baseline_search_space_reduction"]) for row in rows
        ),
        "bootstrap": bootstrap,
        "checks": checks,
        "contract_macro_f1_is_gate": False,
        "opening_count": 1,
        "held_out_scored": True,
    }
