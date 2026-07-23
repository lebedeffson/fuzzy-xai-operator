from __future__ import annotations

import random
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .models import Candidate, Case, Mutation

FAULT_FAMILIES = (
    "missing_provenance",
    "version_mismatch",
    "schema_incompatibility",
    "preprocessing_mismatch",
    "missing_calibration",
    "expired_calibration",
    "checksum_corruption",
    "missing_edge",
    "invalid_edge_contract",
    "dictionary_mismatch",
    "reduction_loss_violation",
    "missing_explanation",
    "dependency_cycle",
    "non_repairable_component",
    "unknown_relation",
    "composite_cross_layer_failure",
)

PIPELINES = (
    ("tabular-credit", "tabular"),
    ("tabular-energy", "tabular"),
    ("text-news", "text"),
    ("text-reviews", "text"),
    ("image-quality", "image"),
    ("timeseries-sensors", "time_series"),
)

STRATUM_CYCLE = ("S1", "S2", "S2", "S3", "S3", "S3", "S4", "S4", "S5", "S5")


def _candidate(
    case_id: str,
    name: str,
    covers: tuple[str, ...],
    cost: float,
    *,
    edge: bool = False,
    **kwargs: object,
) -> Candidate:
    prefix = "edge" if edge else "node"
    subject_id = f"{prefix}-{case_id[-4:]}-{name}"
    return Candidate(
        atom_id=name,
        subject_kind="edge" if edge else "node",
        subject_id=subject_id,
        field="relation" if edge else "registered_version",
        violation_code="contract_unsatisfied",
        covers=covers,
        cost=cost,
        **kwargs,
    )


def _build_candidates(
    case_id: str,
    stratum: str,
    obligations: tuple[str, ...],
    *,
    deep_dependencies: bool,
) -> tuple[Candidate, ...]:
    if stratum == "S1":
        return (_candidate(case_id, "direct-0", obligations, 1.0),)
    if stratum == "S2":
        singles = tuple(
            _candidate(case_id, f"direct-{index}", (obligation,), 1.0 + 0.05 * index)
            for index, obligation in enumerate(obligations)
        )
        return (*singles, _candidate(case_id, "expensive-global", obligations, len(obligations) + 2.0))

    # Registered set-cover counterexample: greedy and per-obligation rules select
    # a valid but non-minimal cover; the optimum is the pair of pair repairs.
    base = (
        _candidate(
            case_id,
            "greedy-global-023",
            (obligations[0], obligations[2], obligations[3]),
            2.5,
            dependencies=("preflight-route",),
        ),
        _candidate(
            case_id,
            "greedy-single-1",
            (obligations[1],),
            2.0,
            dependencies=("preflight-route",),
        ),
        _candidate(
            case_id,
            "optimal-pair-01",
            (obligations[0], obligations[1]),
            2.0,
            dependencies=("preflight-route",),
        ),
        _candidate(
            case_id,
            "optimal-pair-23",
            (obligations[2], obligations[3]),
            2.0,
            dependencies=("preflight-route",),
            edge=True,
        ),
        _candidate(
            case_id,
            "preflight-route",
            (),
            0.2,
            dependencies=("authorize-route",) if deep_dependencies else (),
        ),
        _candidate(case_id, "authorize-route", (), 0.1),
    )
    if stratum == "S3":
        return base
    if stratum == "S4":
        return (
            *base,
            _candidate(
                case_id,
                "alternative-global-a",
                obligations,
                4.0,
                dependencies=("preflight-route",),
            ),
            _candidate(
                case_id,
                "alternative-global-b",
                obligations,
                4.0,
                dependencies=("preflight-route",),
                edge=True,
            ),
        )
    unknown = _candidate(
        case_id,
        "unknown-relation-probe",
        (obligations[-1],),
        0.4,
        edge=True,
        repairable=False,
        executable=False,
        provider_status="unavailable",
    )
    fallback = _candidate(
        case_id,
        "fallback-known-4",
        (obligations[-1],),
        2.0,
        dependencies=("preflight-route",),
        edge=True,
    )
    return (*base, unknown, fallback)


def generate_cases(split: str, cases_per_pipeline: int, seed: int) -> list[Case]:
    rng = random.Random(seed)
    cases: list[Case] = []
    for pipeline_index, (pipeline, modality) in enumerate(PIPELINES):
        offset = rng.randrange(len(STRATUM_CYCLE))
        s5_seen = 0
        for local_index in range(cases_per_pipeline):
            case_index = pipeline_index * cases_per_pipeline + local_index
            case_id = f"h10-c3:{split}:{pipeline}:{case_index:05d}"
            stratum = STRATUM_CYCLE[(local_index + offset) % len(STRATUM_CYCLE)]
            obligation_count = {"S1": 1, "S2": 4, "S3": 4, "S4": 4, "S5": 5}[stratum]
            obligations = tuple(f"obligation-{index}" for index in range(obligation_count))
            candidates = _build_candidates(
                case_id,
                stratum,
                obligations,
                deep_dependencies=stratum == "S5"
                or (stratum == "S3" and local_index % 2 == 0),
            )
            repairable = not (stratum == "S5" and s5_seen % 5 == 0)
            if stratum == "S5":
                s5_seen += 1
            if not repairable:
                candidates = tuple(
                    replace(candidate, repairable=False, executable=False)
                    if candidate.atom_id == "fallback-known-4"
                    else candidate
                    for candidate in candidates
                )
            node_ids = {candidate.subject_id for candidate in candidates if candidate.subject_kind == "node"}
            node_ids.update({"route-input", "route-output"})
            edge_ids = {
                candidate.subject_id for candidate in candidates if candidate.subject_kind == "edge"
            }
            edges = tuple(
                (edge_id, "route-input", "route-output")
                for edge_id in sorted(edge_ids)
            )
            valid_inverse = tuple(
                candidate.atom_id
                for candidate in candidates
                if candidate.repairable
                and candidate.executable
                and candidate.provider_status == "healthy"
                and candidate.covers
            )
            family = FAULT_FAMILIES[(case_index + pipeline_index) % len(FAULT_FAMILIES)]
            mutations = (
                Mutation(
                    operation_id=f"mutation-{case_index}",
                    changed_nodes=tuple(sorted(node_ids)[:2]),
                    changed_edges=tuple(sorted(edge_ids)),
                    broken_obligations=obligations,
                    allowed_inverse_ids=valid_inverse if repairable else (),
                ),
            )
            cases.append(
                Case(
                    case_id=case_id,
                    split=split,
                    pipeline=pipeline,
                    modality=modality,
                    stratum=stratum,
                    family=family,
                    obligations=obligations,
                    nodes=tuple(sorted(node_ids)),
                    edges=edges,
                    candidates=candidates,
                    mutations=mutations,
                    repairable=repairable,
                )
            )
    return cases


def stable_case_hash(case: Case) -> str:
    import json

    payload = json.dumps(case.method_view(), sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def serialize_cases(root: Path, split: str, cases: list[Case]) -> dict[str, object]:
    public_path = root / "data" / split / "cases.jsonl"
    private_path = root / "private" / split / "transactions.jsonl"
    write_jsonl(public_path, [case.method_view() for case in cases])
    write_jsonl(private_path, [case.private_record() for case in cases])
    counts = {stratum: sum(case.stratum == stratum for case in cases) for stratum in STRATUM_CYCLE}
    return {
        "split": split,
        "case_count": len(cases),
        "pipelines": len({case.pipeline for case in cases}),
        "strata": counts,
        "composite_fraction": sum(case.stratum != "S1" for case in cases) / len(cases),
        "case_hashes": [stable_case_hash(case) for case in cases],
        "private_files_exposed_to_methods": False,
    }
