from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import pvariance

from .contract_inference_v2 import evaluation_contract_family
from .guided_diagnosis import GuidedCandidate
from .runtime_events import RuntimeEvent

CHANNEL_WEIGHTS = {
    "exact_symbol": 1.5,
    "bm25": 1.0,
    "repograph": 0.9,
    "executed_slice": 1.35,
    "h10_c5c_retriever": 1.25,
}
VALUE_EVENT_KINDS = frozenset(
    {
        "argument_value",
        "assertion_operand",
        "last_writer",
        "return_value",
        "value_flow",
    }
)


class EvidenceOrigin(StrEnum):
    DYNAMIC_EXECUTION = "DYNAMIC_EXECUTION"
    VALUE_PROVENANCE = "VALUE_PROVENANCE"
    STATIC_STRUCTURE = "STATIC_STRUCTURE"
    TEXTUAL_MATCH = "TEXTUAL_MATCH"
    CONTRACT_OBSERVATION = "CONTRACT_OBSERVATION"
    ACTIVE_INTERVENTION = "ACTIVE_INTERVENTION"


@dataclass(frozen=True)
class PairVerificationFeatures:
    candidate_rank: int
    candidate_score: float
    pair_score: float
    pair_margin: float
    candidate_in_traceback: bool
    exact_traceback_frame: bool
    candidate_executed: bool
    execution_count: int
    incoming_call_count: int
    outgoing_call_count: int
    exception_origin: bool
    value_provenance_available: bool
    produced_assertion_value: bool
    consumed_assertion_value: bool
    last_writer_of_failed_value: bool
    bm25_rank: int | None
    legacy_rank: int | None
    repograph_rank: int | None
    runtime_rank: int | None
    exact_symbol_rank: int | None
    channel_consensus: int
    channel_rank_variance: float
    number_of_top3_votes: int
    contract_family: str
    contract_score: float
    contract_margin: float
    candidate_contract_compatibility: bool
    contract_direct_observation: bool
    evidence_origins: tuple[str, ...]
    rank_stability: float
    production_symbol: bool
    test_symbol: bool
    active_intervention: bool = False


@dataclass(frozen=True)
class PairVerificationDecision:
    status: str
    pair_probability: float
    reasons: tuple[str, ...]
    rejected_reasons: tuple[str, ...]


def _is_test_symbol(candidate: GuidedCandidate) -> bool:
    path = candidate.file_path.replace("\\", "/").lower()
    name = (candidate.symbol or "").lower()
    return bool(
        path.startswith(("test/", "tests/"))
        or "/test/" in path
        or "/tests/" in path
        or path.endswith(("_test.py", "/conftest.py"))
        or name.startswith("test_")
    )


def _matches(
    file_path: str | None,
    symbol: str | None,
    candidate: GuidedCandidate,
) -> bool:
    if not file_path or file_path.replace("\\", "/") != candidate.file_path:
        return False
    if symbol is None:
        return candidate.symbol is None
    return bool(
        candidate.symbol == symbol
        or (candidate.symbol or "").rsplit(".", 1)[-1] == symbol
    )


def channel_ranks(candidate: GuidedCandidate) -> dict[str, int]:
    ranks = {}
    for item in candidate.evidence:
        if item.startswith("channel_rank:"):
            _, channel, raw_rank = item.split(":", 2)
            ranks[channel] = int(raw_rank)
    return ranks


def _channel_contributions(candidate: GuidedCandidate) -> dict[str, float]:
    return {
        channel: CHANNEL_WEIGHTS[channel] / math.sqrt(rank)
        for channel, rank in channel_ranks(candidate).items()
        if channel in CHANNEL_WEIGHTS
    }


def channel_ablation_stability(
    candidates: tuple[GuidedCandidate, ...],
    selected: GuidedCandidate,
) -> float:
    """Recompute ordering after removing each stored channel contribution."""
    if not candidates:
        return 0.0
    contributions = {
        candidate.node_id: _channel_contributions(candidate)
        for candidate in candidates
    }
    stable = 0
    channels = tuple(CHANNEL_WEIGHTS)
    for channel in channels:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -(
                    candidate.score
                    - contributions[candidate.node_id].get(channel, 0.0)
                ),
                candidate.file_path,
                candidate.symbol or "",
            ),
        )
        if selected.node_id in {item.node_id for item in ordered[:3]}:
            stable += 1
    return stable / len(channels)


def extract_pair_features(
    candidates: tuple[GuidedCandidate, ...],
    candidate_index: int,
    runtime_events: tuple[RuntimeEvent, ...],
    *,
    active_intervention: bool = False,
) -> PairVerificationFeatures:
    if not 0 <= candidate_index < min(3, len(candidates)):
        raise IndexError("pair verifier accepts only an existing top-3 pair")
    candidate = candidates[candidate_index]
    source_events = tuple(
        event
        for event in runtime_events
        if _matches(event.source_file, event.source_symbol, candidate)
    )
    target_events = tuple(
        event
        for event in runtime_events
        if _matches(event.target_file, event.target_symbol, candidate)
    )
    coverage = tuple(
        event for event in source_events if event.kind == "coverage"
    )
    incoming = tuple(
        event for event in target_events if event.kind == "call"
    )
    outgoing = tuple(
        event for event in source_events if event.kind == "call"
    )
    traceback = tuple(
        event for event in source_events if event.kind == "traceback_frame"
    )
    exception = tuple(
        event for event in source_events if event.kind == "exception"
    )
    value_events = tuple(
        event
        for event in (*source_events, *target_events)
        if event.kind in VALUE_EVENT_KINDS
    )
    ranks = channel_ranks(candidate)
    rank_values = tuple(ranks.values())
    hypotheses = candidate.contract_hypotheses
    contract_score = hypotheses[0].confidence if hypotheses else 0.0
    second_contract = hypotheses[1].confidence if len(hypotheses) > 1 else 0.0
    alternative_scores = [
        item.score
        + (
            item.contract_hypotheses[0].confidence
            if item.contract_hypotheses
            else 0.0
        )
        for index, item in enumerate(candidates[:3])
        if index != candidate_index
    ]
    pair_score = candidate.score + contract_score
    origins = set()
    if coverage or incoming or outgoing or traceback or exception:
        origins.add(EvidenceOrigin.DYNAMIC_EXECUTION)
    if value_events:
        origins.add(EvidenceOrigin.VALUE_PROVENANCE)
    if "repograph" in ranks:
        origins.add(EvidenceOrigin.STATIC_STRUCTURE)
    if {"bm25", "exact_symbol"}.intersection(ranks):
        origins.add(EvidenceOrigin.TEXTUAL_MATCH)
    compatibility = any(
        evidence.startswith("candidate_compatibility:")
        for hypothesis in hypotheses
        for evidence in hypothesis.evidence
    )
    direct_contract = any(
        evidence.startswith("direct_observation:")
        for hypothesis in hypotheses
        for evidence in hypothesis.evidence
    )
    if compatibility:
        origins.add(EvidenceOrigin.CONTRACT_OBSERVATION)
    if active_intervention:
        origins.add(EvidenceOrigin.ACTIVE_INTERVENTION)
    test_symbol = _is_test_symbol(candidate)
    return PairVerificationFeatures(
        candidate_rank=candidate_index + 1,
        candidate_score=candidate.score,
        pair_score=pair_score,
        pair_margin=pair_score - max(alternative_scores, default=0.0),
        candidate_in_traceback=bool(traceback),
        exact_traceback_frame=bool(traceback),
        candidate_executed=bool(coverage or incoming or outgoing),
        execution_count=len(coverage),
        incoming_call_count=len(incoming),
        outgoing_call_count=len(outgoing),
        exception_origin=bool(exception),
        value_provenance_available=bool(value_events),
        produced_assertion_value=any(
            event.kind == "return_value" for event in source_events
        ),
        consumed_assertion_value=any(
            event.kind in {"argument_value", "assertion_operand"}
            for event in (*source_events, *target_events)
        ),
        last_writer_of_failed_value=any(
            event.kind == "last_writer"
            for event in (*source_events, *target_events)
        ),
        bm25_rank=ranks.get("bm25"),
        legacy_rank=ranks.get("h10_c5c_retriever"),
        repograph_rank=ranks.get("repograph"),
        runtime_rank=ranks.get("executed_slice"),
        exact_symbol_rank=ranks.get("exact_symbol"),
        channel_consensus=len(
            set(ranks).intersection(CHANNEL_WEIGHTS)
        ),
        channel_rank_variance=(
            pvariance(rank_values) if len(rank_values) > 1 else 0.0
        ),
        number_of_top3_votes=sum(rank <= 3 for rank in rank_values),
        contract_family=evaluation_contract_family(candidate.contract.family),
        contract_score=contract_score,
        contract_margin=contract_score - second_contract,
        candidate_contract_compatibility=compatibility,
        contract_direct_observation=direct_contract,
        evidence_origins=tuple(sorted(origin.value for origin in origins)),
        rank_stability=channel_ablation_stability(candidates, candidate),
        production_symbol=not test_symbol,
        test_symbol=test_symbol,
        active_intervention=active_intervention,
    )


class FormalPairVerifier:
    """V0 requires candidate-bound evidence from distinct origins."""

    @staticmethod
    def decide(
        features: PairVerificationFeatures,
        *,
        stronger_conflicting_pair: bool = False,
    ) -> PairVerificationDecision:
        reasons = []
        rejected = []
        execution_link = bool(
            features.candidate_executed
            or features.candidate_in_traceback
            or features.exception_origin
        )
        failure_link = bool(
            features.value_provenance_available
            or features.exact_traceback_frame
            or features.exception_origin
        )
        channel_support = features.channel_consensus >= 2
        contract_support = bool(
            features.contract_family != "UNKNOWN_CONTRACT"
            and features.candidate_contract_compatibility
        )
        checks = {
            "candidate_specific_execution": execution_link,
            "candidate_specific_failure_link": failure_link,
            "multi_channel_support": channel_support,
            "candidate_contract_compatibility": contract_support,
            "production_symbol": features.production_symbol,
            "no_stronger_conflicting_pair": not stronger_conflicting_pair,
        }
        for name, passed in checks.items():
            (reasons if passed else rejected).append(name)
        status = (
            "PAIR_VERIFIED"
            if all(checks.values())
            else "PAIR_NOT_VERIFIED"
        )
        probability = sum(checks.values()) / len(checks)
        return PairVerificationDecision(
            status,
            probability,
            tuple(reasons),
            tuple(rejected),
        )
