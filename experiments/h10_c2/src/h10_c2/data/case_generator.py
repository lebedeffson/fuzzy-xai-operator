from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Iterable

from ..config import load_yaml
from ..hashing import object_sha256, write_json
from ..models import Case
from ..oracle import derive_gold, validate_gold
from ..paths import ARTIFACT_ROOT
from .fault_generator import derive_obligations, mutate_route
from .route_generator import build_clean_route


def _case_type(index: int, mix: dict[str, float]) -> str:
    scale = 100
    marker = index % scale
    clean_end = round(scale * mix["clean"])
    single_end = clean_end + round(scale * mix["single"])
    composite_end = single_end + round(scale * mix["composite"])
    if marker < clean_end:
        return "clean"
    if marker < single_end:
        return "single"
    if marker < composite_end:
        return "composite"
    return "unknown_or_irreparable"


def generate_cases(split: str, total_cases: int, *, seed: int) -> list[Case]:
    experiment = load_yaml("experiment.yaml")
    pipelines = load_yaml("pipelines.yaml")["pipelines"]
    rng = random.Random(seed)
    cases: list[Case] = []
    for index in range(total_cases):
        pipeline = pipelines[index % len(pipelines)]
        kind = _case_type(index, experiment["case_mix"])
        clean = build_clean_route(pipeline["id"], int(pipeline["width"]) + rng.randrange(0, 3), index)
        mutation_count = 0 if kind == "clean" else (1 if kind == "single" else rng.choice((2, 3, 4)))
        observed, transactions = mutate_route(
            clean,
            rng=rng,
            mutation_count=mutation_count,
            unknown_or_irreparable=kind == "unknown_or_irreparable",
        )
        obligations, costs = derive_obligations(clean, transactions, equivalent=index % 5 == 0)
        public = {
            "case_id": f"h10-c2:{split}:{index:06d}",
            "pipeline": pipeline["id"],
            "observed_route": observed,
            "obligations": obligations,
            "costs": costs,
        }
        cases.append(
            Case(
                case_id=public["case_id"],
                pipeline=pipeline["id"],
                modality=pipeline["modality"],
                split=split,
                case_type=kind,
                clean_route=clean,
                observed_route=observed,
                public_obligations=obligations,
                repair_costs=costs,
                case_hash=object_sha256(public),
                transactions=transactions,
            )
        )
    return cases


def write_split(split: str, total_cases: int, seed: int, *, include_private_gold: bool) -> dict:
    cases = generate_cases(split, total_cases, seed=seed)
    split_dir = ARTIFACT_ROOT / "data" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    public_path = split_dir / "cases.jsonl"
    public_path.write_text(
        "".join(json.dumps(case.public_dict(), ensure_ascii=False, sort_keys=True) + "\n" for case in cases),
        encoding="utf-8",
    )
    gold_records = []
    for case in cases:
        gold = derive_gold(case)
        validate_gold(case, gold)
        gold_records.append(asdict(gold))
    if include_private_gold:
        private = ARTIFACT_ROOT / "private" / split
        private.mkdir(parents=True, exist_ok=True)
        (private / "gold.jsonl").write_text(
            "".join(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n" for value in gold_records),
            encoding="utf-8",
        )
        (private / "transactions.jsonl").write_text(
            "".join(
                json.dumps(
                    {"case_id": case.case_id, "transactions": [asdict(item) for item in case.transactions]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for case in cases
            ),
            encoding="utf-8",
        )
    manifest = {
        "split": split,
        "case_count": len(cases),
        "case_hashes": [case.case_hash for case in cases],
        "private_gold_generated": include_private_gold,
        "gold_exposed_to_methods": False,
    }
    write_json(split_dir / "manifest.json", manifest)
    return manifest


def assert_disjoint(current_hashes: Iterable[str], old_hashes: Iterable[str]) -> None:
    overlap = set(current_hashes).intersection(old_hashes)
    if overlap:
        raise ValueError(f"new H10-C2 cases overlap previous H10 Gold: {len(overlap)}")

