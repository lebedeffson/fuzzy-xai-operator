from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuzzyxai.repository_diagnostics.contract_inference_v2 import (
    evaluation_contract_family,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedCandidate,
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import IncidentQuery
from fuzzyxai.repository_diagnostics.runtime_events import (
    RuntimeEvent,
    load_runtime_events,
)

from .h10_c7 import DevelopmentIncident, GoldLocalization, _graph
from .h10_c7_replay import _json, _replay_records

EXPECTED_ALIGNMENT = {
    "correct_top_1_pairs": 11,
    "correct_pairs_in_top_3": 14,
    "misaligned_old_positive_labels": 3,
}
CHANNEL_WEIGHTS = {
    "exact_symbol": 1.5,
    "bm25": 1.0,
    "repograph": 0.9,
    "executed_slice": 1.35,
    "h10_c5c_retriever": 1.25,
}
VALUE_PROVENANCE_KINDS = frozenset(
    {
        "argument_value",
        "assertion_operand",
        "last_writer",
        "return_value",
        "value_flow",
    }
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def pair_target(
    candidate: GuidedCandidate,
    gold: GoldLocalization,
) -> int:
    return int(
        any(
            candidate.file_path == atom.file_path
            and candidate.symbol == atom.symbol
            and evaluation_contract_family(candidate.contract.family)
            == atom.contract
            for atom in gold.atoms
        )
    )


def old_joint_target(
    candidates: tuple[GuidedCandidate, ...],
    gold: GoldLocalization,
) -> int:
    if not candidates:
        return 0
    contract_hit = evaluation_contract_family(
        candidates[0].contract.family
    ) in {atom.contract for atom in gold.atoms}
    symbol_hit = any(
        candidate.file_path == atom.file_path
        and candidate.symbol == atom.symbol
        for candidate in candidates[:3]
        for atom in gold.atoms
    )
    return int(contract_hit and symbol_hit)


def _channel_ranks(candidate: GuidedCandidate) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for evidence in candidate.evidence:
        if not evidence.startswith("channel_rank:"):
            continue
        _, channel, raw_rank = evidence.split(":", 2)
        ranks[channel] = int(raw_rank)
    return ranks


def _symbol_matches(
    event_file: str | None,
    event_symbol: str | None,
    candidate: GuidedCandidate,
) -> bool:
    if not event_file or event_file.replace("\\", "/") != candidate.file_path:
        return False
    if event_symbol is None:
        return candidate.symbol is None
    return bool(
        candidate.symbol == event_symbol
        or (candidate.symbol or "").rsplit(".", 1)[-1] == event_symbol
    )


def _candidate_runtime(
    candidate: GuidedCandidate,
    events: tuple[RuntimeEvent, ...],
) -> dict[str, object]:
    source_events = tuple(
        event
        for event in events
        if _symbol_matches(event.source_file, event.source_symbol, candidate)
    )
    target_events = tuple(
        event
        for event in events
        if _symbol_matches(event.target_file, event.target_symbol, candidate)
    )
    traceback = tuple(
        event for event in source_events if event.kind == "traceback_frame"
    )
    calls_in = tuple(
        event for event in target_events if event.kind == "call"
    )
    calls_out = tuple(
        event for event in source_events if event.kind == "call"
    )
    coverage = tuple(
        event for event in source_events if event.kind == "coverage"
    )
    value_events = tuple(
        event
        for event in (*source_events, *target_events)
        if event.kind in VALUE_PROVENANCE_KINDS
    )
    exception_origin = tuple(
        event for event in source_events if event.kind == "exception"
    )
    return {
        "candidate_executed": bool(coverage or calls_in or calls_out),
        "candidate_in_traceback": bool(traceback),
        "exact_traceback_frame": bool(traceback),
        "execution_count": len(coverage),
        "incoming_call_count": len(calls_in),
        "outgoing_call_count": len(calls_out),
        "exception_origin": bool(exception_origin),
        "value_provenance_available": bool(value_events),
        "value_provenance_event_count": len(value_events),
    }


def _load_r5_records(
    bundle: Path,
) -> tuple[
    tuple[
        tuple[
            DevelopmentIncident,
            GoldLocalization,
            tuple[GuidedCandidate, ...],
            tuple[RuntimeEvent, ...],
        ],
        ...,
    ],
    dict[str, object],
]:
    records, gold = _replay_records(bundle)
    engine = GuidedNaturalDiagnosisEngine(structural_only=True)
    values = []
    top_10 = {}
    top_20 = {}
    for value in records:
        query_value = value["query"]
        query = IncidentQuery(
            str(value["incident_id"]),
            str(query_value.get("issue", "")),
            tuple(str(item) for item in query_value.get("failing_tests", [])),
            str(query_value.get("traceback", "")),
            str(query_value.get("assertion", "")),
        )
        incident = DevelopmentIncident(
            str(value["incident_id"]),
            str(value["repository"]),
            query,
            _graph(_json((bundle / str(value["graph_path"])).resolve())),
            int(value["repository_symbol_count"]),
        )
        events = load_runtime_events(
            (bundle / str(value["runtime_events_path"])).resolve()
        )
        diagnosis = engine.diagnose(
            incident.graph,
            query,
            "R5",
            events,
        )
        candidates = diagnosis.candidates
        top_10[incident.incident_id] = [
            (item.node_id, item.file_path, item.symbol)
            for item in candidates[:10]
        ]
        top_20[incident.incident_id] = [
            (item.node_id, item.file_path, item.symbol)
            for item in candidates[:20]
        ]
        values.append((incident, gold[incident.incident_id], candidates, events))
    return tuple(values), {"top_10": top_10, "top_20": top_20}


def run_pair_verification_audits(
    *,
    bundle: Path,
    output: Path,
) -> dict[str, object]:
    records, ranking_signatures = _load_r5_records(bundle)
    alignment_rows = []
    specificity_rows = []
    contribution_rows = []
    event_kinds: Counter[str] = Counter()
    top_1_correct = 0
    top_3_correct = 0
    mismatched_old_labels = []

    for incident, gold, candidates, events in records:
        event_kinds.update(event.kind for event in events)
        pair_targets = [pair_target(item, gold) for item in candidates[:3]]
        top_1 = pair_targets[0] if pair_targets else 0
        top_3 = int(any(pair_targets))
        old_target = old_joint_target(candidates, gold)
        top_1_correct += top_1
        top_3_correct += top_3
        if old_target != top_1:
            mismatched_old_labels.append(incident.incident_id)
        alignment_rows.append(
            {
                "incident_id": incident.incident_id,
                "repository": incident.repository,
                "old_target": old_target,
                "top_1_pair_target": top_1,
                "top_3_pair_hit": top_3,
                "pair_targets": pair_targets,
                "top_3_pairs": [
                    {
                        "rank": index,
                        "file_path": candidate.file_path,
                        "symbol": candidate.symbol,
                        "contract": evaluation_contract_family(
                            candidate.contract.family
                        ),
                        "target": pair_targets[index - 1],
                    }
                    for index, candidate in enumerate(candidates[:3], start=1)
                ],
            }
        )
        for rank, candidate in enumerate(candidates[:3], start=1):
            runtime = _candidate_runtime(candidate, events)
            channel_ranks = _channel_ranks(candidate)
            origins = []
            if runtime["candidate_executed"] or runtime["candidate_in_traceback"]:
                origins.append("DYNAMIC_EXECUTION")
            if runtime["value_provenance_available"]:
                origins.append("VALUE_PROVENANCE")
            if "repograph" in channel_ranks:
                origins.append("STATIC_STRUCTURE")
            if {"bm25", "exact_symbol"}.intersection(channel_ranks):
                origins.append("TEXTUAL_MATCH")
            if any(
                evidence.startswith(
                    ("candidate_compatibility:", "direct_observation:")
                )
                for hypothesis in candidate.contract_hypotheses
                for evidence in hypothesis.evidence
            ):
                origins.append("CONTRACT_OBSERVATION")
            specificity_rows.append(
                {
                    "incident_id": incident.incident_id,
                    "repository": incident.repository,
                    "candidate_rank": rank,
                    "file_path": candidate.file_path,
                    "symbol": candidate.symbol,
                    "pair_target": pair_target(candidate, gold),
                    "candidate_specific": {
                        **runtime,
                        "channel_ranks": channel_ranks,
                        "candidate_contract_compatibility": any(
                            evidence.startswith("candidate_compatibility:")
                            for hypothesis in candidate.contract_hypotheses
                            for evidence in hypothesis.evidence
                        ),
                    },
                    "incident_general_not_candidate_evidence": {
                        "assertion_text_present": bool(
                            incident.query.assertion.strip()
                        ),
                        "exception_text_present": bool(
                            incident.query.traceback.strip()
                        ),
                        "contract_direct_observation_unbound": any(
                            evidence.startswith("direct_observation:")
                            for hypothesis in candidate.contract_hypotheses
                            for evidence in hypothesis.evidence
                        ),
                    },
                    "evidence_origins": sorted(set(origins)),
                }
            )
            contributions = {
                channel: CHANNEL_WEIGHTS[channel] / (rank_value**0.5)
                for channel, rank_value in channel_ranks.items()
                if channel in CHANNEL_WEIGHTS
            }
            contribution_rows.append(
                {
                    "incident_id": incident.incident_id,
                    "candidate_rank": rank,
                    "file_path": candidate.file_path,
                    "symbol": candidate.symbol,
                    "pair_target": pair_target(candidate, gold),
                    "channel_ranks": channel_ranks,
                    "channel_contributions": contributions,
                    "channel_consensus": len(contributions),
                    "reservoir_contribution_sum": sum(
                        contributions.values()
                    ),
                    "stored_score": candidate.score,
                }
            )

    observed = {
        "correct_top_1_pairs": top_1_correct,
        "correct_pairs_in_top_3": top_3_correct,
        "misaligned_old_positive_labels": len(mismatched_old_labels),
    }
    checks = {
        key: observed[key] == expected
        for key, expected in EXPECTED_ALIGNMENT.items()
    }
    alignment = {
        "status": (
            "H10_C7_R5V_TARGET_ALIGNMENT_PASS"
            if all(checks.values())
            else "H10_C7_R5V_TARGET_ALIGNMENT_FAIL"
        ),
        "expected": EXPECTED_ALIGNMENT,
        "observed": observed,
        "checks": checks,
        "misaligned_incident_ids": sorted(mismatched_old_labels),
        "rows": alignment_rows,
    }
    event_availability = {
        "event_kind_counts": dict(sorted(event_kinds.items())),
        "value_provenance_event_kinds": sorted(VALUE_PROVENANCE_KINDS),
        "value_provenance_events_present": any(
            event_kinds[kind] for kind in VALUE_PROVENANCE_KINDS
        ),
        "candidate_specific_value_provenance_available_count": sum(
            bool(
                row["candidate_specific"][
                    "value_provenance_available"
                ]
            )
            for row in specificity_rows
        ),
        "candidate_pair_count": len(specificity_rows),
    }
    specificity = {
        "status": "H10_C7_R5V_CANDIDATE_SPECIFICITY_AUDIT_COMPLETE",
        "feature_policy": {
            "candidate_specific": [
                "candidate execution/call/traceback matches",
                "per-candidate channel ranks",
                "candidate-contract compatibility",
            ],
            "incident_general_not_candidate_evidence": [
                "assertion text presence",
                "exception text presence",
                "unbound direct contract observations",
            ],
            "origins": [
                "DYNAMIC_EXECUTION",
                "VALUE_PROVENANCE",
                "STATIC_STRUCTURE",
                "TEXTUAL_MATCH",
                "CONTRACT_OBSERVATION",
                "ACTIVE_INTERVENTION",
            ],
        },
        "availability": event_availability,
        "rows": specificity_rows,
    }
    contribution = {
        "status": "H10_C7_R5V_CHANNEL_CONTRIBUTION_AUDIT_COMPLETE",
        "frozen_channel_weights": CHANNEL_WEIGHTS,
        "ablation_requirement": (
            "recompute candidate ordering from stored per-channel "
            "contributions; multiplying aggregate score is prohibited"
        ),
        "rows": contribution_rows,
    }

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "TARGET_ALIGNMENT_AUDIT.json", alignment)
    _write_json(output / "CANDIDATE_SPECIFICITY_AUDIT.json", specificity)
    _write_json(output / "CHANNEL_CONTRIBUTION_AUDIT.json", contribution)
    _write_json(output / "R5_RANKING_SIGNATURES.json", ranking_signatures)
    blocked_reasons = []
    if alignment["status"] != "H10_C7_R5V_TARGET_ALIGNMENT_PASS":
        blocked_reasons.append("TARGET_ALIGNMENT_AUDIT_FAILED")
    if not event_availability["value_provenance_events_present"]:
        blocked_reasons.append("VALUE_PROVENANCE_UNAVAILABLE")
    status = {
        "protocol_id": "H10-C7-R5V",
        "status": (
            "H10_C7_R5V_AUDIT_PASS"
            if not blocked_reasons
            else "H10_C7_R5V_BLOCKED_AUDIT"
        ),
        "scientific_result": "NOT_EVALUATED",
        "target_alignment": observed,
        "target_alignment_expected": EXPECTED_ALIGNMENT,
        "blocked_reasons": blocked_reasons,
        "v0_executed": False,
        "v1_executed": False,
        "model_threshold_selected": False,
        "retrieval_modified": False,
        "held_out_created": False,
        "held_out_scored": False,
    }
    _write_json(output / "R5V_STATUS.json", status)
    if alignment["status"] != "H10_C7_R5V_TARGET_ALIGNMENT_PASS":
        raise ValueError("R5V target alignment audit failed")
    return {
        "target_alignment": alignment["status"],
        "candidate_specificity": specificity["status"],
        "channel_contribution": contribution["status"],
        "observed": observed,
        "value_provenance_available": event_availability[
            "value_provenance_events_present"
        ],
    }
