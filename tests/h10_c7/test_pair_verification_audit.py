from __future__ import annotations

from fuzzyxai.experiments.h10_c7 import GoldAtom, GoldLocalization
from fuzzyxai.experiments.h10_c7_pair_verification import (
    old_joint_target,
    pair_target,
)
from fuzzyxai.repository_diagnostics.contract_inference_v2 import (
    ContractPrediction,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import GuidedCandidate
from fuzzyxai.repository_diagnostics.pair_verification import (
    FormalPairVerifier,
    channel_ablation_stability,
    extract_pair_features,
)
from fuzzyxai.repository_diagnostics.runtime_events import RuntimeEvent


def _candidate(
    file_path: str,
    symbol: str,
    contract: str,
) -> GuidedCandidate:
    prediction = ContractPrediction(contract, "CONFIGURATION", 0.8, ())
    return GuidedCandidate(
        file_path + "::" + symbol,
        file_path,
        symbol,
        1.0,
        prediction,
        (prediction,),
        ("bm25",),
        1,
        (),
    )


def test_pair_target_never_combines_different_candidates() -> None:
    gold = GoldLocalization(
        "incident",
        (GoldAtom("src/right.py", "right", "CONFIGURATION"),),
    )
    wrong = _candidate("src/wrong.py", "wrong", "OUTPUT_FORMAT")
    right_symbol_wrong_contract = _candidate(
        "src/right.py",
        "right",
        "DATA_SCHEMA",
    )
    assert pair_target(wrong, gold) == 0
    assert pair_target(right_symbol_wrong_contract, gold) == 0
    assert old_joint_target((wrong, right_symbol_wrong_contract), gold) == 1


def test_pair_target_requires_file_symbol_and_evaluation_contract() -> None:
    gold = GoldLocalization(
        "incident",
        (GoldAtom("src/right.py", "right", "CONFIGURATION"),),
    )
    candidate = _candidate("src/right.py", "right", "OUTPUT_FORMAT")
    assert pair_target(candidate, gold) == 1


def test_candidate_specific_runtime_does_not_leak_to_another_pair() -> None:
    first = _candidate("src/right.py", "right", "OUTPUT_FORMAT")
    second = _candidate("src/other.py", "other", "OUTPUT_FORMAT")
    event = RuntimeEvent(
        "event",
        "test",
        "traceback_frame",
        "src/right.py",
        "right",
    )
    first_features = extract_pair_features((first, second), 0, (event,))
    second_features = extract_pair_features((first, second), 1, (event,))
    assert first_features.exact_traceback_frame
    assert not second_features.exact_traceback_frame


def test_channel_ablation_uses_stored_contributions() -> None:
    prediction = ContractPrediction(
        "OUTPUT_FORMAT",
        "CONFIGURATION",
        0.8,
        ("candidate_compatibility:output",),
    )
    selected = GuidedCandidate(
        "selected",
        "src/right.py",
        "right",
        2.6,
        prediction,
        (prediction,),
        ("bm25", "exact_symbol"),
        1,
        (),
        ("channel_rank:bm25:1", "channel_rank:exact_symbol:1"),
    )
    alternatives = tuple(
        GuidedCandidate(
            f"alternative-{index}",
            f"src/other_{index}.py",
            f"other_{index}",
            2.0,
            prediction,
            (prediction,),
            ("repograph",),
            1,
            (),
            ("channel_rank:repograph:1",),
        )
        for index in range(3)
    )
    assert (
        channel_ablation_stability((selected, *alternatives), selected) < 1.0
    )


def test_formal_verifier_requires_candidate_bound_failure_evidence() -> None:
    first = _candidate("src/right.py", "right", "OUTPUT_FORMAT")
    event = RuntimeEvent(
        "event",
        "test",
        "coverage",
        "src/right.py",
        "right",
    )
    features = extract_pair_features((first,), 0, (event,))
    decision = FormalPairVerifier().decide(features)
    assert decision.status == "PAIR_NOT_VERIFIED"
    assert "candidate_specific_failure_link" in decision.rejected_reasons
