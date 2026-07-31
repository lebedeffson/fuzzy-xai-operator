#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    args = parser.parse_args()

    results = args.results.resolve()
    protocol = args.protocol.resolve()
    reports = args.reports.resolve()
    reports.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(results / "DEVELOPMENT_PER_INCIDENT.jsonl")
    aggregates = json.loads(
        (results / "DEVELOPMENT_AGGREGATES.json").read_text()
    )
    gate = json.loads((results / "DEVELOPMENT_GATE.json").read_text())
    model_lock = json.loads(
        (protocol / "R10M_MODEL_LOCK.json").read_text()
    )
    input_audit = json.loads(
        (results.parent / "inputs" / "DEVELOPMENT_INPUT_AUDIT.json").read_text()
    )

    categories = Counter()
    details = []
    overlap = Counter()
    for row in rows:
        r10m = row["metrics"]["R10M"]
        r9 = row["metrics"]["R9"]
        if r10m["symbol_hit_at_20"]:
            category = "TOP20_HIT"
        elif not r10m["symbol_pool_hit_at_200"]:
            category = "SYMBOL_POOL_MISS"
        else:
            category = "FINAL_RERANK_MISS"
        categories[category] += 1
        r10m_hit = bool(r10m["symbol_hit_at_20"])
        r9_hit = bool(r9["symbol_hit_at_20"])
        overlap[
            (
                "R10M_HIT" if r10m_hit else "R10M_MISS",
                "R9_HIT" if r9_hit else "R9_MISS",
            )
        ] += 1
        details.append(
            {
                "incident_id": row["incident_id"],
                "repository": row["repository"],
                "category": category,
                "file_rank": r10m["file_rank"],
                "pool_rank": r10m["symbol_pool_rank"],
                "symbol_rank": r10m["symbol_rank"],
                "r9_symbol_rank": r9["symbol_rank"],
            }
        )

    analysis = {
        "protocol_id": "H10-C7R-R10M-v1",
        "status": "H10_C7R_R10M_DEVELOPMENT_NOT_SUPPORTED",
        "categories": dict(sorted(categories.items())),
        "file_retrieval_miss_at_20": sum(
            not row["metrics"]["R10M"]["file_hit_at_20"] for row in rows
        ),
        "r10m_r9_overlap": {
            f"{first}|{second}": count
            for (first, second), count in sorted(overlap.items())
        },
        "incidents": details,
        "interpretation": (
            "File localization transferred, but the frozen symbol pool and "
            "final four-rank RRF did not reach the registered symbol endpoint."
        ),
    }
    _write_json(results / "ERROR_ANALYSIS.json", analysis)

    final_status = {
        "protocol_id": "H10-C7R-R10M-v1",
        "status": "H10_C7R_R10M_DEVELOPMENT_NOT_SUPPORTED",
        "scientific_result": "NOT_EVALUATED",
        "development_scored": True,
        "development_gate_passed": False,
        "ready_for_new_held_out": False,
        "held_out_created": False,
        "held_out_scored": False,
        "official_held_out_scoring_count": 0,
        "collector_status": "40/40 R10_RUNTIME_READY",
        "method_modified_after_scoring": False,
        "model_revisions": {
            value["model_id"]: value["revision"]
            for value in model_lock["models"]
        },
        "development_gate": gate["gates"],
        "development_metrics": gate["r10m"],
        "strongest_baseline": gate["strongest_baseline"],
        "chapter_docx_modified": False,
    }
    _write_json(results.parent / "FINAL_STATUS.json", final_status)

    header = (
        "| Method | File R@10 | File R@20 | Pool R@200 | Symbol R@20 | "
        "MRR | False localization | Runtime, s |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    table = "".join(
        f"| {method} | {value['file_hit_at_10']:.4f} | "
        f"{value['file_hit_at_20']:.4f} | "
        f"{value['symbol_pool_hit_at_200']:.4f} | "
        f"{value['symbol_hit_at_20']:.4f} | "
        f"{value['reciprocal_rank']:.4f} | "
        f"{value['false_localization']:.4f} | "
        f"{value['runtime_seconds']:.3f} |\n"
        for method, value in aggregates.items()
    )
    (reports / "MODEL_REPORT.md").write_text(
        "# H10-C7R R10M model report\n\n"
        "Both neural components used frozen upstream weights without "
        "fine-tuning. Inference was local-only after snapshot acquisition.\n\n"
        + "\n".join(
            f"- `{value['model_id']}` revision `{value['revision']}`, "
            f"snapshot SHA256 `{value['snapshot_sha256']}`, "
            f"weights SHA256 `{value['weights_sha256']}`."
            for value in model_lock["models"]
        )
        + "\n\nRuntime: `"
        + json.dumps(model_lock["runtime"], sort_keys=True)
        + "`.\n",
        encoding="utf-8",
    )
    (reports / "ERROR_ANALYSIS.md").write_text(
        "# H10-C7R R10M error analysis\n\n"
        "## Boundary\n\n"
        "This is development-only analysis on 40 disclosed incidents. No "
        "new held-out set was created or scored.\n\n"
        "## Stage decomposition\n\n"
        "| Stage | Incidents |\n|---|---:|\n"
        + "".join(
            f"| {name} | {count} |\n"
            for name, count in sorted(categories.items())
        )
        + "\n## Interpretation\n\n"
        "R10M retained the correct file at rank 10 in 38/40 incidents, but "
        "the symbol pool retained a Gold symbol in only 35/40 and the final "
        "top-20 retained it in 21/40. The model contour therefore improved "
        "file retrieval but did not solve repository-independent symbol "
        "compression. No threshold, model, channel, or budget was changed "
        "after scoring.\n",
        encoding="utf-8",
    )
    (reports / "FINAL_REPORT.md").write_text(
        "# H10-C7R R10M final development report\n\n"
        "Status: `H10_C7R_R10M_DEVELOPMENT_NOT_SUPPORTED`.\n\n"
        "Scientific result: `NOT_EVALUATED`.\n\n"
        + header
        + table
        + "\nThe frozen development gate did not pass. Coverage and file "
        "Recall@10 passed, while file Recall@20, pool Recall@200, symbol "
        "Recall@20, repository Q1, baseline superiority, and MRR superiority "
        "did not all pass. In accordance with the protocol, no confirmatory "
        "held-out was created and no further R10 variant is opened.\n\n"
        f"Observable leakage audit: `{input_audit['status']}`, "
        f"Gold leakage `{input_audit['observable_gold_leakage']}`.\n",
        encoding="utf-8",
    )
    (reports / "REPRODUCTION.md").write_text(
        "# H10-C7R R10M reproduction\n\n"
        "1. Extract the locked causal recollection artifact.\n"
        "2. Mount the two snapshots at the paths recorded in "
        "`R10M_MODEL_LOCK.json`.\n"
        "3. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.\n"
        "4. Run `make h10-c7r-r10m-test`.\n"
        "5. Run `make h10-c7r-r10m-development` with the extracted root and "
        "a writable model-score cache.\n\n"
        "The scorer checkpoints each incident and never passes Gold to a "
        "retrieval or model feature channel.\n",
        encoding="utf-8",
    )
    print(json.dumps(final_status, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
