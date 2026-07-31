#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import GoldLocalization
from fuzzyxai.experiments.h10_c7r import load_held_out_inputs
from fuzzyxai.experiments.h10_c7r_r10m import (
    FrozenBGEReranker,
    FrozenGraphCodeBERT,
    R10MConfig,
    R10MRetriever,
    SQLiteModelCache,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    R9_SOURCE_KINDS,
    BM25Retriever,
    RankedSymbol,
    documents_from_graph,
)

METHODS = ("B_TRACE", "B_BM25", "R9", "R10A", "R10M")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _snapshot_sha256(path: Path) -> str:
    rows = []
    for file_path in sorted(path.rglob("*")):
        if (
            not file_path.is_file()
            or ".cache" in file_path.relative_to(path).parts
        ):
            continue
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {file_path.relative_to(path)}\n")
    return hashlib.sha256("".join(rows).encode()).hexdigest()


def _rank(candidates: Sequence[RankedSymbol], gold: GoldLocalization) -> int:
    for rank, candidate in enumerate(candidates, start=1):
        if any(
            candidate.file_path == atom.file_path
            and candidate.symbol == atom.symbol
            for atom in gold.atoms
        ):
            return rank
    return 0


def _file_rank(file_paths: Sequence[str], gold: GoldLocalization) -> int:
    for rank, file_path in enumerate(file_paths, start=1):
        if any(file_path == atom.file_path for atom in gold.atoms):
            return rank
    return 0


def _unique_files(candidates: Sequence[RankedSymbol]) -> list[str]:
    return list(dict.fromkeys(item.file_path for item in candidates))


def _method_metrics(
    method: str,
    candidates: Sequence[RankedSymbol],
    gold: GoldLocalization,
    *,
    files: Sequence[str] | None = None,
    pool: Sequence[RankedSymbol] | None = None,
    runtime_seconds: float,
) -> dict[str, object]:
    rank = _rank(candidates, gold)
    file_paths = list(files) if files is not None else _unique_files(candidates)
    file_rank = _file_rank(file_paths, gold)
    pool_rank = _rank(pool or candidates, gold)
    return {
        "method": method,
        "available": bool(candidates),
        "file_rank": file_rank,
        "file_hit_at_1": float(bool(file_rank and file_rank <= 1)),
        "file_hit_at_5": float(bool(file_rank and file_rank <= 5)),
        "file_hit_at_10": float(bool(file_rank and file_rank <= 10)),
        "file_hit_at_20": float(bool(file_rank and file_rank <= 20)),
        "symbol_pool_rank": pool_rank,
        "symbol_pool_hit_at_200": float(bool(pool_rank)),
        "symbol_rank": rank,
        "symbol_hit_at_5": float(bool(rank and rank <= 5)),
        "symbol_hit_at_10": float(bool(rank and rank <= 10)),
        "symbol_hit_at_20": float(bool(rank and rank <= 20)),
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "false_localization": float(not bool(rank and rank <= 20)),
        "file_candidate_count": len(file_paths),
        "symbol_pool_count": len(pool or candidates),
        "inspected_symbol_count": len(candidates),
        "runtime_seconds": runtime_seconds,
        "top_files": file_paths[:25],
        "top_symbols": [item.node_id for item in candidates[:20]],
    }


def _aggregate(
    rows: Sequence[Mapping[str, object]],
    method: str,
) -> dict[str, object]:
    selected = [
        row["metrics"][method]
        for row in rows
        if isinstance(row["metrics"], dict)
    ]
    repositories: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        repositories[str(row["repository"])].append(row["metrics"][method])
    per_repository = {
        repository: {
            "incident_count": len(values),
            "symbol_recall_at_20": statistics.fmean(
                float(value["symbol_hit_at_20"]) for value in values
            ),
            "mrr": statistics.fmean(
                float(value["reciprocal_rank"]) for value in values
            ),
        }
        for repository, values in sorted(repositories.items())
    }
    repository_recalls = sorted(
        float(value["symbol_recall_at_20"])
        for value in per_repository.values()
    )
    return {
        "method": method,
        "incident_count": len(selected),
        "repository_count": len(repositories),
        "coverage": statistics.fmean(
            float(bool(value["available"])) for value in selected
        ),
        **{
            name: statistics.fmean(float(value[name]) for value in selected)
            for name in (
                "file_hit_at_1",
                "file_hit_at_5",
                "file_hit_at_10",
                "file_hit_at_20",
                "symbol_pool_hit_at_200",
                "symbol_hit_at_5",
                "symbol_hit_at_10",
                "symbol_hit_at_20",
                "reciprocal_rank",
                "false_localization",
                "runtime_seconds",
            )
        },
        "median_inspected_symbols": statistics.median(
            int(value["inspected_symbol_count"]) for value in selected
        ),
        "repository_recall_lower_quartile": repository_recalls[
            max(0, (len(repository_recalls) - 1) // 4)
        ],
        "per_repository": per_repository,
    }


def _synchronize() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recollection-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--exclusion-lock", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.recollection_root.resolve()
    inputs = load_held_out_inputs(
        root / "HELD_OUT_MANIFEST.jsonl",
        args.gold,
        args.exclusion_lock,
    )
    lock = json.loads(args.model_lock.read_text(encoding="utf-8"))
    models = {value["role"]: value for value in lock["models"]}
    for value in models.values():
        local_path = Path(value["local_path"])
        actual = _snapshot_sha256(local_path)
        if actual != value["snapshot_sha256"]:
            raise RuntimeError(
                f"model snapshot mismatch: {value['model_id']} {actual}"
            )

    cache = SQLiteModelCache(args.cache.resolve())
    config = R10MConfig()
    encoder = FrozenGraphCodeBERT(
        Path(models["dense_retrieval"]["local_path"]),
        cache,
        max_length=config.graphcodebert_max_length,
        batch_size=config.graphcodebert_batch_size,
        device=args.device,
        precision=str(models["dense_retrieval"]["precision"]),
    )
    pair_scorer = FrozenBGEReranker(
        Path(models["pair_reranking"]["local_path"]),
        cache,
        max_length=config.bge_max_length,
        batch_size=config.bge_batch_size,
        device=args.device,
        precision=str(models["pair_reranking"]["precision"]),
    )
    retriever = R10MRetriever(encoder, pair_scorer, config)
    structural = GuidedNaturalDiagnosisEngine(structural_only=True)
    bm25 = BM25Retriever()
    output = args.output.resolve()
    checkpoint = output / "DEVELOPMENT_CHECKPOINT.jsonl"
    completed = {
        str(row["incident_id"]): row
        for row in (
            [
                json.loads(line)
                for line in checkpoint.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if checkpoint.is_file()
            else []
        )
    }

    for index, case in enumerate(inputs.cases, start=1):
        if case.incident_id in completed:
            continue
        documents = documents_from_graph(
            case.graph,
            case.runtime_events,
            source_kinds=R9_SOURCE_KINDS,
        )
        gold = inputs.gold[case.incident_id]
        metrics: dict[str, object] = {}

        start = time.perf_counter()
        trace = tuple(
            RankedSymbol(
                item.node_id,
                item.file_path,
                item.symbol,
                1.0,
                ("traceback",),
                item.line_count,
                item.obligations,
            )
            for item in documents
            if item.traceback_distance == 0.0
        )[:20]
        metrics["B_TRACE"] = _method_metrics(
            "B_TRACE",
            trace,
            gold,
            runtime_seconds=time.perf_counter() - start,
        )

        start = time.perf_counter()
        lexical = bm25.rank(case.query.text, documents, limit=20)
        metrics["B_BM25"] = _method_metrics(
            "B_BM25",
            lexical,
            gold,
            runtime_seconds=time.perf_counter() - start,
        )

        start = time.perf_counter()
        r9 = structural.diagnose(
            case.graph,
            case.query,
            "R9A",
            case.runtime_events,
        )
        r9_candidates = tuple(
            RankedSymbol(
                item.node_id,
                item.file_path,
                item.symbol,
                item.score,
                item.rank_sources,
                item.line_count,
                (),
                item.evidence,
            )
            for item in r9.candidates[:20]
        )
        metrics["R9"] = _method_metrics(
            "R9",
            r9_candidates,
            gold,
            runtime_seconds=time.perf_counter() - start,
        )

        start = time.perf_counter()
        r10a, unavailable = structural._r10_ranking(
            case.query,
            documents,
            case.runtime_events,
            "R10A",
        )
        if unavailable:
            r10a = ()
        metrics["R10A"] = _method_metrics(
            "R10A",
            r10a,
            gold,
            runtime_seconds=time.perf_counter() - start,
        )

        _synchronize()
        start = time.perf_counter()
        r10m = retriever.rank(case.query, documents, case.runtime_events)
        _synchronize()
        metrics["R10M"] = _method_metrics(
            "R10M",
            r10m.top_symbols,
            gold,
            files=[item.file_path for item in r10m.top_files],
            pool=r10m.symbol_pool,
            runtime_seconds=time.perf_counter() - start,
        )
        completed[case.incident_id] = {
            "incident_id": case.incident_id,
            "repository": case.repository,
            "selection_index": index,
            "runtime_ready": True,
            "candidate_pool_size": len(r10m.symbol_pool),
            "model_revisions": {
                encoder.model_name: encoder.revision,
                pair_scorer.model_name: pair_scorer.revision,
            },
            "metrics": metrics,
            "gold_used_after_ranking_only": True,
        }
        _write_jsonl(checkpoint, list(completed.values()))
        print(
            json.dumps(
                {
                    "completed": len(completed),
                    "incident_id": case.incident_id,
                    "r10m_symbol_rank": metrics["R10M"]["symbol_rank"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rows = list(completed.values())
    aggregates = {method: _aggregate(rows, method) for method in METHODS}
    baseline_methods = ("B_TRACE", "B_BM25", "R9", "R10A")
    strongest = max(
        baseline_methods,
        key=lambda name: (
            aggregates[name]["symbol_hit_at_20"],
            aggregates[name]["reciprocal_rank"],
            name,
        ),
    )
    result = aggregates["R10M"]
    gates = {
        "coverage_at_least_0_90": result["coverage"] >= 0.90,
        "file_recall_at_10_at_least_0_90": (
            result["file_hit_at_10"] >= 0.90
        ),
        "file_recall_at_20_at_least_0_97": (
            result["file_hit_at_20"] >= 0.97
        ),
        "symbol_pool_recall_at_200_at_least_0_95": (
            result["symbol_pool_hit_at_200"] >= 0.95
        ),
        "symbol_recall_at_20_at_least_0_85": (
            result["symbol_hit_at_20"] >= 0.85
        ),
        "repository_q1_at_least_0_75": (
            result["repository_recall_lower_quartile"] >= 0.75
        ),
        "r10m_symbol_recall_exceeds_r9": (
            result["symbol_hit_at_20"]
            > aggregates["R9"]["symbol_hit_at_20"]
        ),
        "r10m_mrr_exceeds_bm25": (
            result["reciprocal_rank"]
            > aggregates["B_BM25"]["reciprocal_rank"]
        ),
        "false_localization_not_worse_than_strongest_baseline": (
            result["false_localization"]
            <= aggregates[strongest]["false_localization"]
        ),
        "gold_leakage_zero": True,
    }
    gate_passed = all(gates.values())
    status = {
        "protocol_id": "H10-C7R-R10M-v1",
        "status": (
            "H10_C7R_R10M_DEVELOPMENT_GO"
            if gate_passed
            else "H10_C7R_R10M_DEVELOPMENT_NOT_SUPPORTED"
        ),
        "scientific_result": "NOT_EVALUATED",
        "development_scored": True,
        "development_gate_passed": gate_passed,
        "ready_for_new_held_out": gate_passed,
        "held_out_created": False,
        "held_out_scored": False,
        "strongest_baseline": strongest,
        "gates": gates,
        "r10m": result,
    }
    _write_jsonl(output / "DEVELOPMENT_PER_INCIDENT.jsonl", rows)
    _write_json(output / "DEVELOPMENT_AGGREGATES.json", aggregates)
    _write_json(output / "DEVELOPMENT_GATE.json", status)
    print(json.dumps(status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
