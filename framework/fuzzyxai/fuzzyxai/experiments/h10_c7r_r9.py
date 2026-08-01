from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fuzzyxai.experiments.h10_c7 import GoldLocalization
from fuzzyxai.experiments.h10_c7a import BudgetCase
from fuzzyxai.experiments.h10_c7r import HeldOutInputs
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    R9_SOURCE_KINDS,
    R9CandidateCompressor,
    RankedSymbol,
    SymbolDocument,
    documents_from_graph,
)

R9_VARIANTS = ("R9A", "R9B", "R9C")
AVAILABLE_STRUCTURAL_VARIANTS = ("R9A", "R9B")


@dataclass(frozen=True)
class R9DevelopmentSummary:
    variant: str
    incident_count: int
    repository_count: int
    recall_at_20: float
    repository_recall_lower_quartile: float
    schema_gap_count: int
    contract_reordering_count: int
    maximum_exact_candidates: int


@dataclass(frozen=True)
class R9FeatureCase:
    case: BudgetCase
    gold: GoldLocalization
    documents: tuple[SymbolDocument, ...]
    channels: dict[str, tuple[RankedSymbol, ...]]
    feature_rows: dict[str, dict[str, float]]
    exact_candidate_count: int
    schema_gap_count: int
    auxiliary_gold_atom_count: int


@dataclass(frozen=True)
class R9LambdamartModel:
    feature_names: tuple[str, ...]
    estimator: Any
    model_sha256: str
    feature_importances: dict[str, int]

    def score(
        self,
        rows: Mapping[str, Mapping[str, float]],
    ) -> dict[str, float]:
        node_ids = sorted(rows)
        matrix = [
            [rows[node_id].get(name, 0.0) for name in self.feature_names]
            for node_id in node_ids
        ]
        predictions = self.estimator.predict(
            pd.DataFrame(matrix, columns=self.feature_names)
        )
        return {
            node_id: float(score)
            for node_id, score in zip(node_ids, predictions, strict=True)
        }


LAMBDAMART_MAX_CANDIDATES_PER_INCIDENT = 500
LAMBDAMART_PARAMETERS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 120,
    "learning_rate": 0.04,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 5,
    "reg_lambda": 0.25,
    "random_state": 1729,
    "n_jobs": 1,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def gold_rank(
    candidates: Sequence[object],
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


def schema_gap_count(
    case: BudgetCase,
    gold: GoldLocalization,
    documents: Sequence[SymbolDocument] | None = None,
) -> int:
    available = documents or documents_from_graph(
        case.graph, case.runtime_events, source_kinds=R9_SOURCE_KINDS
    )
    return int(
        not any(
            document.file_path == atom.file_path
            and document.symbol == atom.symbol
            for document in available
            for atom in gold.atoms
        )
    )


def auxiliary_gold_atom_count(
    case: BudgetCase,
    gold: GoldLocalization,
    documents: Sequence[SymbolDocument] | None = None,
) -> int:
    available = documents or documents_from_graph(
        case.graph, case.runtime_events, source_kinds=R9_SOURCE_KINDS
    )
    return sum(
        not any(
            document.file_path == atom.file_path
            and document.symbol == atom.symbol
            for document in available
        )
        for atom in gold.atoms
    )


def build_feature_cases(
    inputs: HeldOutInputs,
    *,
    engine: GuidedNaturalDiagnosisEngine,
) -> tuple[R9FeatureCase, ...]:
    values: list[R9FeatureCase] = []
    for case in inputs.cases:
        gold = inputs.gold[case.incident_id]
        documents = documents_from_graph(
            case.graph,
            case.runtime_events,
            source_kinds=R9_SOURCE_KINDS,
        )
        channels = engine._r9_channels(
            case.graph,
            case.query,
            documents,
        )
        values.append(
            R9FeatureCase(
                case=case,
                gold=gold,
                documents=documents,
                channels=channels,
                feature_rows=engine.r9_compressor.feature_rows(
                    channels, documents
                ),
                exact_candidate_count=len(channels["strict_identifier"]),
                schema_gap_count=schema_gap_count(case, gold, documents),
                auxiliary_gold_atom_count=auxiliary_gold_atom_count(
                    case, gold, documents
                ),
            )
        )
    return tuple(values)


def fit_lambdamart(
    cases: Sequence[R9FeatureCase],
    *,
    compressor: R9CandidateCompressor,
) -> R9LambdamartModel:
    """Fit a compact LambdaMART model on repository-separated training data."""
    try:
        from lightgbm import LGBMRanker
    except ImportError as error:  # pragma: no cover - optional backend
        raise RuntimeError(
            "R9B development requires the registered lightgbm extra"
        ) from error

    feature_names = tuple(sorted(compressor.DEFAULT_FEATURE_WEIGHTS))
    matrix: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []
    for payload in sorted(cases, key=lambda item: item.case.incident_id):
        by_id = {item.node_id: item for item in payload.documents}
        positives = {
            node_id
            for node_id in payload.feature_rows
            if _candidate_is_gold(by_id[node_id], payload.gold)
        }
        if not positives:
            continue
        ordered = sorted(
            payload.feature_rows,
            key=lambda node_id: (
                -_linear_score(
                    payload.feature_rows[node_id],
                    compressor.DEFAULT_FEATURE_WEIGHTS,
                ),
                node_id,
            ),
        )
        selected = sorted(positives)
        selected_ids = set(selected)
        for node_id in ordered:
            if node_id in selected_ids:
                continue
            selected.append(node_id)
            selected_ids.add(node_id)
            if len(selected) >= LAMBDAMART_MAX_CANDIDATES_PER_INCIDENT:
                break
        groups.append(len(selected))
        for node_id in selected:
            matrix.append(
                [
                    payload.feature_rows[node_id].get(name, 0.0)
                    for name in feature_names
                ]
            )
            labels.append(int(node_id in positives))
    if not groups:
        raise ValueError("LambdaMART R9B training has no positive groups")

    estimator = LGBMRanker(**LAMBDAMART_PARAMETERS)
    estimator.fit(
        pd.DataFrame(matrix, columns=feature_names),
        labels,
        group=groups,
        eval_at=[20],
    )
    model_text = estimator.booster_.model_to_string()
    importances = {
        name: int(value)
        for name, value in zip(
            feature_names,
            estimator.feature_importances_,
            strict=True,
        )
    }
    return R9LambdamartModel(
        feature_names,
        estimator,
        hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
        importances,
    )


def score_loro_lambdamart(
    feature_cases: Sequence[R9FeatureCase],
    *,
    compressor: R9CandidateCompressor,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Score R9A and LambdaMART R9B with outer repository holdouts."""
    repositories = sorted(
        {payload.case.repository for payload in feature_cases}
    )
    rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for held_repository in repositories:
        training = [
            payload
            for payload in feature_cases
            if payload.case.repository != held_repository
        ]
        held = [
            payload
            for payload in feature_cases
            if payload.case.repository == held_repository
        ]
        model = fit_lambdamart(training, compressor=compressor)
        training_scores = {
            "R9A": _score_cases(
                training,
                compressor=compressor,
                variant="R9A",
            ),
            "R9B": _score_cases(
                training,
                compressor=compressor,
                variant="R9B",
                score_model=model,
            ),
        }
        selected_variant = max(
            ("R9A", "R9B"),
            key=lambda variant: (training_scores[variant], variant),
        )
        held_rows: list[dict[str, object]] = []
        for payload in held:
            for variant in ("R9A", "R9B"):
                ranking = compressor.rank(
                    payload.channels,
                    payload.documents,
                    hierarchical=variant == "R9B",
                    feature_rows=payload.feature_rows,
                    score_overrides=(
                        model.score(payload.feature_rows)
                        if variant == "R9B"
                        else None
                    ),
                )
                row = _result_row(payload, ranking, variant)
                row["held_out_repository"] = held_repository
                row["training_repositories"] = [
                    repository
                    for repository in repositories
                    if repository != held_repository
                ]
                rows.append(row)
                if variant == selected_variant:
                    held_rows.append(row)
                    selected_rows.append(row)
        folds.append(
            {
                "held_repository": held_repository,
                "selected_variant": selected_variant,
                "training_repositories": "|".join(
                    repository
                    for repository in repositories
                    if repository != held_repository
                ),
                "training_recall_r9a": training_scores["R9A"],
                "training_recall_r9b": training_scores["R9B"],
                "test_incidents": len(held_rows),
                "test_recall_at_20": statistics.fmean(
                    float(row["hit_at_20"]) for row in held_rows
                ),
                "ranker": "lightgbm_lambdamart",
                "model_sha256": model.model_sha256,
                "feature_importances": json.dumps(
                    model.feature_importances,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    return rows, selected_rows, folds


def score_structural_variants(
    inputs: HeldOutInputs,
    *,
    engine: GuidedNaturalDiagnosisEngine,
    variants: Sequence[str] = AVAILABLE_STRUCTURAL_VARIANTS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in inputs.cases:
        gold = inputs.gold[case.incident_id]
        documents = documents_from_graph(
            case.graph,
            case.runtime_events,
            source_kinds=R9_SOURCE_KINDS,
        )
        channels = engine._r9_channels(
            case.graph,
            case.query,
            documents,
        )
        exact_count = len(channels["strict_identifier"])
        gap_count = schema_gap_count(case, gold, documents)
        auxiliary_count = auxiliary_gold_atom_count(case, gold, documents)
        for variant in variants:
            candidates = engine.r9_compressor.rank(
                channels,
                documents,
                hierarchical=variant == "R9B",
            )
            rank = gold_rank(candidates, gold)
            rows.append(
                {
                    "incident_id": case.incident_id,
                    "repository": case.repository,
                    "variant": variant,
                    "available": True,
                    "unavailable_reason": "",
                    "rank": rank or 0,
                    "hit_at_20": float(bool(rank and rank <= 20)),
                    "candidate_count": len(candidates),
                    "schema_gap_count": gap_count,
                    "auxiliary_gold_atom_count": auxiliary_count,
                    "contract_reordered": 0,
                    "exact_candidate_count": exact_count,
                    "channel_gold_ranks": {
                        channel: gold_rank(ranking, gold) or 0
                        for channel, ranking in channels.items()
                    },
                    "broad_union_hit": any(
                        gold_rank(ranking, gold)
                        for ranking in channels.values()
                    ),
                    "top_20": [item.node_id for item in candidates],
                }
            )
    return rows


def _candidate_is_gold(
    candidate: SymbolDocument,
    gold: GoldLocalization,
) -> bool:
    return any(
        candidate.file_path == atom.file_path
        and candidate.symbol == atom.symbol
        for atom in gold.atoms
    )


def _linear_score(
    features: Mapping[str, float],
    weights: Mapping[str, float],
) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def _score_cases(
    cases: Sequence[R9FeatureCase],
    *,
    compressor: R9CandidateCompressor,
    variant: str,
    score_model: R9LambdamartModel | None = None,
) -> float:
    return statistics.fmean(
        bool(
            gold_rank(
                compressor.rank(
                    payload.channels,
                    payload.documents,
                    hierarchical=variant == "R9B",
                    feature_rows=payload.feature_rows,
                    score_overrides=(
                        score_model.score(payload.feature_rows)
                        if score_model is not None
                        else None
                    ),
                ),
                payload.gold,
            )
        )
        for payload in cases
    )


def _result_row(
    payload: R9FeatureCase,
    ranking: Sequence[RankedSymbol],
    variant: str,
) -> dict[str, object]:
    rank = gold_rank(ranking, payload.gold)
    return {
        "incident_id": payload.case.incident_id,
        "repository": payload.case.repository,
        "variant": variant,
        "available": True,
        "unavailable_reason": "",
        "rank": rank or 0,
        "hit_at_20": float(bool(rank and rank <= 20)),
        "candidate_count": len(ranking),
        "schema_gap_count": payload.schema_gap_count,
        "auxiliary_gold_atom_count": payload.auxiliary_gold_atom_count,
        "contract_reordered": 0,
        "exact_candidate_count": payload.exact_candidate_count,
        "channel_gold_ranks": {
            channel: gold_rank(channel_ranking, payload.gold) or 0
            for channel, channel_ranking in payload.channels.items()
        },
        "broad_union_hit": any(
            gold_rank(channel_ranking, payload.gold)
            for channel_ranking in payload.channels.values()
        ),
        "top_20": [item.node_id for item in ranking],
    }


def leave_one_repository_out(
    rows: Sequence[dict[str, object]],
    *,
    variants: Sequence[str] = AVAILABLE_STRUCTURAL_VARIANTS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    repositories = sorted({str(row["repository"]) for row in rows})
    selected_rows: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for held_repository in repositories:
        training = [
            row for row in rows if row["repository"] != held_repository
        ]
        training_scores = {
            variant: statistics.fmean(
                float(row["hit_at_20"])
                for row in training
                if row["variant"] == variant
            )
            for variant in variants
        }
        selected = max(
            variants,
            key=lambda variant: (training_scores[variant], variant),
        )
        held_rows = [
            dict(row)
            for row in rows
            if row["repository"] == held_repository
            and row["variant"] == selected
        ]
        selected_rows.extend(held_rows)
        folds.append(
            {
                "held_repository": held_repository,
                "selected_variant": selected,
                "training_recall_at_20": training_scores[selected],
                "test_incidents": len(held_rows),
                "test_recall_at_20": statistics.fmean(
                    float(row["hit_at_20"]) for row in held_rows
                ),
            }
        )
    return selected_rows, folds


def summarize(
    rows: Sequence[dict[str, object]],
    *,
    variant: str,
) -> R9DevelopmentSummary:
    selected = [row for row in rows if row["variant"] == variant]
    repositories = sorted({str(row["repository"]) for row in selected})
    per_repository = [
        statistics.fmean(
            float(row["hit_at_20"])
            for row in selected
            if row["repository"] == repository
        )
        for repository in repositories
    ]
    return R9DevelopmentSummary(
        variant=variant,
        incident_count=len(selected),
        repository_count=len(repositories),
        recall_at_20=statistics.fmean(
            float(row["hit_at_20"]) for row in selected
        ),
        repository_recall_lower_quartile=_lower_quartile(per_repository),
        schema_gap_count=sum(
            int(row["schema_gap_count"]) for row in selected
        ),
        contract_reordering_count=sum(
            int(row["contract_reordered"]) for row in selected
        ),
        maximum_exact_candidates=max(
            int(row["exact_candidate_count"]) for row in selected
        ),
    )


def selected_summary(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    repositories = sorted({str(row["repository"]) for row in rows})
    per_repository = [
        statistics.fmean(
            float(row["hit_at_20"])
            for row in rows
            if row["repository"] == repository
        )
        for repository in repositories
    ]
    return {
        "incident_count": len(rows),
        "repository_count": len(repositories),
        "recall_at_20": statistics.fmean(
            float(row["hit_at_20"]) for row in rows
        ),
        "repository_recall_lower_quartile": _lower_quartile(
            per_repository
        ),
        "schema_gap_count": sum(int(row["schema_gap_count"]) for row in rows),
        "contract_reordering_count": sum(
            int(row["contract_reordered"]) for row in rows
        ),
        "maximum_exact_candidates": max(
            int(row["exact_candidate_count"]) for row in rows
        ),
        "selected_variants": dict(
            sorted(Counter(str(row["variant"]) for row in rows).items())
        ),
    }


def load_published_v1_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(rows) != 40:
        raise ValueError("published H10-C7R-v1 result must contain 40 rows")
    return rows


def _lower_quartile(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, (len(ordered) - 1) // 4)
    return ordered[index]
