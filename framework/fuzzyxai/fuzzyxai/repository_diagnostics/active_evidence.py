from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .guided_retrieval import RankedSymbol


@dataclass(frozen=True)
class EvidenceRequest:
    command: tuple[str, ...]
    expected_observation: str
    distinguished_hypotheses: tuple[str, ...]
    estimated_cost: float
    timeout_seconds: int
    safety_level: str
    utility: float


class ActiveEvidenceRequestPlanner:
    """Select one bounded read-only probe from observable candidate gaps."""

    def plan(
        self,
        failing_test: str,
        candidates: Sequence[RankedSymbol],
    ) -> tuple[EvidenceRequest, ...]:
        if len(candidates) < 2:
            return ()
        top = candidates[:3]
        probabilities = _softmax([item.score for item in top])
        entropy = -sum(
            value * math.log(value)
            for value in probabilities
            if value > 0
        )
        if entropy < 0.15:
            return ()
        hypotheses = tuple(item.node_id for item in top)
        requests = [
            EvidenceRequest(
                (
                    "python",
                    "-m",
                    "pytest",
                    failing_test,
                    "-x",
                    "-vv",
                    "--showlocals",
                ),
                "runtime types and values at the top candidate frames",
                hypotheses,
                1.0,
                300,
                "READ_ONLY_TEST_EXECUTION",
                entropy * len(hypotheses),
            ),
            EvidenceRequest(
                (
                    "python",
                    "-m",
                    "pytest",
                    failing_test,
                    "-x",
                    "-vv",
                    "--trace-config",
                ),
                "loaded plugin and dependency versions",
                hypotheses,
                1.5,
                300,
                "READ_ONLY_TEST_EXECUTION",
                entropy * len(hypotheses) / 1.5,
            ),
        ]
        return tuple(
            sorted(
                requests,
                key=lambda item: (-item.utility, item.command),
            )
        )


class R10TargetedProbePlanner:
    """Request at most two read-only observations that separate top symbols."""

    def plan(
        self,
        failing_test: str,
        candidates: Sequence[RankedSymbol],
    ) -> tuple[EvidenceRequest, ...]:
        if len(candidates) < 2:
            return ()
        top = candidates[:3]
        hypotheses = tuple(item.node_id for item in top)
        targets = ",".join(
            f"{item.file_path}:{item.symbol or ''}" for item in top
        )
        return (
            EvidenceRequest(
                (
                    "env",
                    f"FUZZYXAI_R10_TARGETS={targets}",
                    "python",
                    "-m",
                    "pytest",
                    failing_test,
                    "-x",
                    "-vv",
                    "--showlocals",
                ),
                "candidate-specific arguments, returns, and assertion operands",
                hypotheses,
                1.0,
                300,
                "READ_ONLY_TEST_EXECUTION",
                float(len(hypotheses)),
            ),
            EvidenceRequest(
                (
                    "env",
                    f"FUZZYXAI_R10_TARGETS={targets}",
                    "FUZZYXAI_R10_VALUE_FLOW_DEPTH=3",
                    "python",
                    "-m",
                    "pytest",
                    failing_test,
                    "-x",
                    "-vv",
                    "--full-trace",
                ),
                "last writer and bounded value flow between competing candidates",
                hypotheses,
                1.5,
                300,
                "READ_ONLY_TEST_EXECUTION",
                float(len(hypotheses)) / 1.5,
            ),
        )


def apply_probe_observation(
    candidates: Sequence[RankedSymbol],
    observed_node_ids: Sequence[str],
    *,
    confidence: float,
) -> tuple[RankedSymbol, ...]:
    """Rerank from an observed probe result, never from a Gold identifier."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("probe confidence must be in [0, 1]")
    observed = set(observed_node_ids)
    values = [
        RankedSymbol(
            item.node_id,
            item.file_path,
            item.symbol,
            item.score + (confidence if item.node_id in observed else 0.0),
            (
                *item.rank_sources,
                "registered_probe_observation",
            ),
            item.line_count,
            item.obligations,
            item.evidence,
        )
        for item in candidates
    ]
    return tuple(
        sorted(
            values,
            key=lambda item: (-item.score, item.file_path, item.symbol or ""),
        )
    )


def _softmax(values: Sequence[float]) -> tuple[float, ...]:
    maximum = max(values, default=0.0)
    scaled = [math.exp(value - maximum) for value in values]
    total = sum(scaled)
    return tuple(value / total for value in scaled) if total else ()
