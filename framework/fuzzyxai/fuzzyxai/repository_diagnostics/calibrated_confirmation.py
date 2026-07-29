from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .contract_inference_v2 import evaluation_contract_family
from .guided_diagnosis import GuidedCandidate, GuidedDiagnosis
from .guided_retrieval import IncidentNormalizer, IncidentQuery

CONFIRMATION_STATUSES = (
    "DIAGNOSIS_CONFIRMED",
    "DIAGNOSIS_PROBABLE",
    "DIAGNOSIS_CANDIDATES",
    "INSUFFICIENT_EVIDENCE",
)


class EvidenceGroup(StrEnum):
    ASSERTION = "ASSERTION_EVIDENCE"
    EXCEPTION = "EXCEPTION_EVIDENCE"
    TRACEBACK = "TRACEBACK_EVIDENCE"
    EXECUTED_SLICE = "EXECUTED_SLICE_EVIDENCE"
    DYNAMIC_CALL = "DYNAMIC_CALL_EVIDENCE"
    EXACT_SYMBOL = "EXACT_SYMBOL_EVIDENCE"
    REPOGRAPH = "REPOGRAPH_EVIDENCE"
    LEXICAL_RETRIEVAL = "LEXICAL_RETRIEVAL_EVIDENCE"
    LEGACY_RETRIEVAL = "LEGACY_RETRIEVAL_EVIDENCE"
    CONTRACT_DIRECT = "CONTRACT_DIRECT_EVIDENCE"
    ACTIVE_PROBE = "ACTIVE_PROBE_EVIDENCE"


@dataclass(frozen=True)
class ConfirmationFeatures:
    candidate_rank: int
    candidate_score: float
    candidate_margin: float
    candidate_margin_normalized: float
    contract_score: float
    contract_margin: float
    contract_margin_normalized: float
    joint_score: float
    joint_margin: float
    joint_margin_normalized: float
    independent_evidence_groups: int
    runtime_evidence_groups: int
    retrieval_channels: int
    has_assertion_evidence: bool
    has_exception_evidence: bool
    has_traceback_evidence: bool
    has_executed_slice_evidence: bool
    has_dynamic_call_evidence: bool
    has_exact_symbol_evidence: bool
    has_repograph_evidence: bool
    has_lexical_retrieval_evidence: bool
    has_legacy_retrieval_evidence: bool
    has_contract_direct_evidence: bool
    has_active_probe_evidence: bool
    production_symbol: bool
    test_symbol: bool
    rank_stability: float
    contract_stability: float


@dataclass(frozen=True)
class ConfirmationDecision:
    status: str
    calibrated_probability: float
    threshold: float
    reasons: tuple[str, ...]
    rejected_reasons: tuple[str, ...]


FEATURE_NAMES = (
    "candidate_margin_normalized",
    "contract_margin_normalized",
    "joint_margin_normalized",
    "independent_evidence_groups",
    "runtime_evidence_groups",
    "retrieval_channels",
    "rank_stability",
    "contract_stability",
    "production_symbol",
    "has_assertion_evidence",
    "has_traceback_evidence",
    "has_active_probe_evidence",
)

FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "changed_files",
        "changed_symbols",
        "fix_commit",
        "gold",
        "gold_contract",
        "gold_file",
        "gold_patch",
        "gold_symbol",
        "incident_id",
        "repository",
        "repository_name",
    }
)

_RUNTIME_GROUPS = frozenset(
    {
        EvidenceGroup.ASSERTION,
        EvidenceGroup.EXCEPTION,
        EvidenceGroup.TRACEBACK,
        EvidenceGroup.EXECUTED_SLICE,
        EvidenceGroup.DYNAMIC_CALL,
        EvidenceGroup.ACTIVE_PROBE,
    }
)
_RETRIEVAL_GROUPS = frozenset(
    {
        EvidenceGroup.EXACT_SYMBOL,
        EvidenceGroup.REPOGRAPH,
        EvidenceGroup.LEXICAL_RETRIEVAL,
        EvidenceGroup.LEGACY_RETRIEVAL,
    }
)
_CORE_GROUPS = frozenset(
    {
        EvidenceGroup.ASSERTION,
        EvidenceGroup.EXCEPTION,
        EvidenceGroup.TRACEBACK,
        EvidenceGroup.EXECUTED_SLICE,
    }
)
_CHANNEL_SOURCES = {
    "bm25": frozenset({"bm25"}),
    "legacy": frozenset({"h10_c5c_retriever", "legacy_r0"}),
    "repograph": frozenset({"repograph"}),
    "runtime": frozenset(
        {
            "directed_callee_distance",
            "directed_caller_distance",
            "dynamic_call_distance",
            "executed_slice",
            "traceback",
        }
    ),
    "exact_symbol": frozenset({"exact_symbol"}),
}
_TEST_PATH_PARTS = frozenset(
    {"conftest.py", "fixture", "fixtures", "test", "testing", "tests"}
)


def _normalized_margin(first: float, second: float) -> float:
    return (first - second) / max(abs(first) + abs(second), 1e-12)


def _is_test_symbol(candidate: GuidedCandidate) -> bool:
    parts = {
        part.lower()
        for part in candidate.file_path.replace("\\", "/").split("/")
        if part
    }
    filename = candidate.file_path.replace("\\", "/").rsplit("/", 1)[-1]
    symbol = (candidate.symbol or "").lower()
    return bool(
        parts.intersection(_TEST_PATH_PARTS)
        or filename == "conftest.py"
        or filename.startswith("test_")
        or symbol.startswith("test_")
    )


def _evidence_groups(
    query: IncidentQuery,
    candidate: GuidedCandidate,
    *,
    active_probe: bool,
) -> frozenset[EvidenceGroup]:
    groups: set[EvidenceGroup] = set()
    sources = set(candidate.rank_sources)
    evidence = {
        item.lower()
        for hypothesis in candidate.contract_hypotheses
        for item in hypothesis.evidence
    }
    normalized = IncidentNormalizer().normalize(query)
    if query.assertion.strip():
        groups.add(EvidenceGroup.ASSERTION)
    if normalized.exception:
        groups.add(EvidenceGroup.EXCEPTION)
    if "traceback" in sources:
        groups.add(EvidenceGroup.TRACEBACK)
    if "executed_slice" in sources:
        groups.add(EvidenceGroup.EXECUTED_SLICE)
    if {
        "directed_callee_distance",
        "directed_caller_distance",
        "dynamic_call_distance",
    }.intersection(sources):
        groups.add(EvidenceGroup.DYNAMIC_CALL)
    if "exact_symbol" in sources:
        groups.add(EvidenceGroup.EXACT_SYMBOL)
    if "repograph" in sources:
        groups.add(EvidenceGroup.REPOGRAPH)
    if "bm25" in sources:
        groups.add(EvidenceGroup.LEXICAL_RETRIEVAL)
    if {"h10_c5c_retriever", "legacy_r0"}.intersection(sources):
        groups.add(EvidenceGroup.LEGACY_RETRIEVAL)
    if any(item.startswith("direct_observation:") for item in evidence):
        groups.add(EvidenceGroup.CONTRACT_DIRECT)
    if active_probe:
        groups.add(EvidenceGroup.ACTIVE_PROBE)
    return frozenset(groups)


def _rank_stability(
    candidates: Sequence[GuidedCandidate],
    selected: GuidedCandidate,
) -> float:
    if not candidates:
        return 0.0
    stable = 0
    for sources in _CHANNEL_SOURCES.values():
        ordered = sorted(
            candidates,
            key=lambda item: (
                -(
                    item.score
                    * (
                        0.76
                        if sources.intersection(item.rank_sources)
                        else 1.0
                    )
                ),
                item.file_path,
                item.symbol or "",
            ),
        )
        if selected.node_id in {item.node_id for item in ordered[:3]}:
            stable += 1
    return stable / len(_CHANNEL_SOURCES)


def _contract_stability(candidate: GuidedCandidate) -> float:
    hypotheses = candidate.contract_hypotheses
    if not hypotheses:
        return 0.0
    selected = candidate.contract.family
    removable_prefixes = (
        "candidate_compatibility:",
        "direct_observation:",
        "observed:",
        "node_kind:",
        "incident_level_hypothesis",
    )
    stable = 0
    for prefix in removable_prefixes:
        scores = []
        for hypothesis in hypotheses:
            affected = any(
                item.startswith(prefix) for item in hypothesis.evidence
            )
            score = hypothesis.confidence * (0.55 if affected else 1.0)
            scores.append((score, hypothesis.family))
        scores.sort(reverse=True)
        if scores and scores[0][1] == selected:
            stable += 1
    return stable / len(removable_prefixes)


def extract_confirmation_features(
    diagnosis: GuidedDiagnosis,
    query: IncidentQuery,
    *,
    candidate_index: int = 0,
    active_probe: bool = False,
) -> ConfirmationFeatures:
    if not 0 <= candidate_index < len(diagnosis.candidates):
        raise IndexError("confirmation candidate is outside the ranking")
    candidates = diagnosis.candidates
    candidate = candidates[candidate_index]
    next_candidate_score = (
        candidates[candidate_index + 1].score
        if candidate_index + 1 < len(candidates)
        else 0.0
    )
    contract_scores = [
        item.confidence for item in candidate.contract_hypotheses
    ]
    contract_score = contract_scores[0] if contract_scores else 0.0
    second_contract_score = (
        contract_scores[1] if len(contract_scores) > 1 else 0.0
    )
    candidate_margin = candidate.score - next_candidate_score
    contract_margin = contract_score - second_contract_score
    joint_score = candidate.score + contract_score
    alternative_joint_scores = [
        item.score
        + (
            item.contract_hypotheses[0].confidence
            if item.contract_hypotheses
            else 0.0
        )
        for index, item in enumerate(candidates[:3])
        if index != candidate_index
    ]
    second_joint_score = max(alternative_joint_scores, default=0.0)
    joint_margin = joint_score - second_joint_score
    groups = _evidence_groups(
        query,
        candidate,
        active_probe=active_probe,
    )
    test_symbol = _is_test_symbol(candidate)
    return ConfirmationFeatures(
        candidate_rank=candidate_index + 1,
        candidate_score=candidate.score,
        candidate_margin=candidate_margin,
        candidate_margin_normalized=_normalized_margin(
            candidate.score,
            next_candidate_score,
        ),
        contract_score=contract_score,
        contract_margin=contract_margin,
        contract_margin_normalized=_normalized_margin(
            contract_score,
            second_contract_score,
        ),
        joint_score=joint_score,
        joint_margin=joint_margin,
        joint_margin_normalized=_normalized_margin(
            joint_score,
            second_joint_score,
        ),
        independent_evidence_groups=len(groups),
        runtime_evidence_groups=len(groups.intersection(_RUNTIME_GROUPS)),
        retrieval_channels=len(groups.intersection(_RETRIEVAL_GROUPS)),
        has_assertion_evidence=EvidenceGroup.ASSERTION in groups,
        has_exception_evidence=EvidenceGroup.EXCEPTION in groups,
        has_traceback_evidence=EvidenceGroup.TRACEBACK in groups,
        has_executed_slice_evidence=EvidenceGroup.EXECUTED_SLICE in groups,
        has_dynamic_call_evidence=EvidenceGroup.DYNAMIC_CALL in groups,
        has_exact_symbol_evidence=EvidenceGroup.EXACT_SYMBOL in groups,
        has_repograph_evidence=EvidenceGroup.REPOGRAPH in groups,
        has_lexical_retrieval_evidence=(
            EvidenceGroup.LEXICAL_RETRIEVAL in groups
        ),
        has_legacy_retrieval_evidence=(
            EvidenceGroup.LEGACY_RETRIEVAL in groups
        ),
        has_contract_direct_evidence=(
            EvidenceGroup.CONTRACT_DIRECT in groups
        ),
        has_active_probe_evidence=EvidenceGroup.ACTIVE_PROBE in groups,
        production_symbol=not test_symbol,
        test_symbol=test_symbol,
        rank_stability=_rank_stability(candidates, candidate),
        contract_stability=_contract_stability(candidate),
    )


def feature_vector(features: ConfirmationFeatures) -> tuple[float, ...]:
    values = asdict(features)
    if FORBIDDEN_FEATURE_NAMES.intersection(values):
        raise AssertionError("forbidden identity or Gold feature detected")
    return tuple(float(values[name]) for name in FEATURE_NAMES)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 30.0), -30.0)))


class DeterministicConfirmationModel:
    model_id = "C0"

    def fit(
        self,
        features: Sequence[ConfirmationFeatures],
        labels: Sequence[int],
    ) -> DeterministicConfirmationModel:
        if len(features) != len(labels):
            raise ValueError("confirmation features and labels differ")
        return self

    def predict_probability(self, features: ConfirmationFeatures) -> float:
        value = (
            -3.40
            + 0.38 * features.independent_evidence_groups
            + 0.24 * features.runtime_evidence_groups
            + 0.16 * features.retrieval_channels
            + 0.90 * features.rank_stability
            + 0.55 * features.contract_stability
            + 0.55 * features.candidate_margin_normalized
            + 0.65 * features.contract_margin_normalized
            + 0.75 * features.joint_margin_normalized
            + 0.35 * features.has_assertion_evidence
            + 0.30 * features.has_traceback_evidence
            + 0.40 * features.has_active_probe_evidence
            + 0.25 * features.production_symbol
        )
        return _sigmoid(value)

    def parameters(self) -> dict[str, object]:
        return {"model_id": self.model_id, "fixed_rule_version": "1"}


class LogisticConfirmationModel:
    model_id = "C1"

    def __init__(self, *, seed: int = 1707) -> None:
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        )
        self._constant: float | None = None

    def fit(
        self,
        features: Sequence[ConfirmationFeatures],
        labels: Sequence[int],
    ) -> LogisticConfirmationModel:
        matrix = np.asarray([feature_vector(item) for item in features])
        target = np.asarray(labels, dtype=int)
        if matrix.shape[0] != target.shape[0]:
            raise ValueError("confirmation features and labels differ")
        unique = np.unique(target)
        if len(unique) == 1:
            self._constant = float(unique[0])
            return self
        scaled = self.scaler.fit_transform(matrix)
        self.model.fit(scaled, target)
        return self

    def predict_probability(self, features: ConfirmationFeatures) -> float:
        if self._constant is not None:
            return self._constant
        matrix = np.asarray([feature_vector(features)])
        return float(self.model.predict_proba(self.scaler.transform(matrix))[0, 1])

    def parameters(self) -> dict[str, object]:
        if self._constant is not None:
            return {
                "model_id": self.model_id,
                "seed": self.seed,
                "constant_probability": self._constant,
            }
        return {
            "model_id": self.model_id,
            "seed": self.seed,
            "feature_names": list(FEATURE_NAMES),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "coefficients": self.model.coef_[0].tolist(),
            "intercept": self.model.intercept_.tolist(),
        }


def confirmation_eligible(
    features: ConfirmationFeatures,
    contract_family: str,
    *,
    candidate_margin_threshold: float,
    joint_margin_threshold: float,
) -> tuple[bool, tuple[str, ...]]:
    rejected = []
    if features.candidate_rank > 3:
        rejected.append("candidate_rank_above_3")
    if contract_family == "UNKNOWN_CONTRACT":
        rejected.append("unknown_contract")
    if features.independent_evidence_groups < 3:
        rejected.append("fewer_than_3_independent_evidence_groups")
    if not any(
        (
            features.has_assertion_evidence,
            features.has_exception_evidence,
            features.has_traceback_evidence,
            features.has_executed_slice_evidence,
        )
    ):
        rejected.append("no_core_runtime_observation")
    if not features.production_symbol:
        rejected.append("test_or_service_symbol")
    stable_or_separated = any(
        (
            features.rank_stability >= 0.60,
            features.candidate_margin_normalized
            >= candidate_margin_threshold,
            features.joint_margin_normalized >= joint_margin_threshold,
            features.has_active_probe_evidence,
        )
    )
    if not stable_or_separated:
        rejected.append("unstable_and_unseparated_candidate")
    return not rejected, tuple(rejected)


class CalibratedDiagnosisConfirmer:
    def decide(
        self,
        *,
        features: ConfirmationFeatures,
        contract_family: str,
        probability: float,
        threshold: float,
        candidate_margin_threshold: float = 0.0,
        joint_margin_threshold: float = 0.0,
    ) -> ConfirmationDecision:
        eligible, rejected = confirmation_eligible(
            features,
            contract_family,
            candidate_margin_threshold=candidate_margin_threshold,
            joint_margin_threshold=joint_margin_threshold,
        )
        reasons = (
            f"independent_evidence_groups={features.independent_evidence_groups}",
            f"rank_stability={features.rank_stability:.6f}",
            f"contract_stability={features.contract_stability:.6f}",
            f"calibrated_probability={probability:.6f}",
        )
        if eligible and probability >= threshold:
            status = "DIAGNOSIS_CONFIRMED"
        elif (
            features.candidate_rank <= 5
            and features.independent_evidence_groups >= 2
            and contract_family != "UNKNOWN_CONTRACT"
        ):
            status = "DIAGNOSIS_PROBABLE"
        elif contract_family != "UNKNOWN_CONTRACT":
            status = "DIAGNOSIS_CANDIDATES"
        else:
            status = "INSUFFICIENT_EVIDENCE"
        return ConfirmationDecision(
            status,
            probability,
            threshold,
            reasons,
            rejected,
        )


def evaluation_family(candidate: GuidedCandidate) -> str:
    return evaluation_contract_family(candidate.contract.family)


def mean_feature(
    features: Iterable[ConfirmationFeatures],
    name: str,
) -> float:
    values = [float(getattr(item, name)) for item in features]
    return sum(values) / len(values) if values else 0.0
