"""Export P19 system cases; the public FuzzyXAI result owns all operators."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzyxai import AlignmentTransform, FuzzyXAI, ObservationContext, SystemObservation
from fuzzyxai.core.explain_plan import (
    AlignmentPolicy,
    ExplainPlan,
    MembershipPolicy,
    MembershipTerm,
    ReductionPolicy,
    UncertaintyPolicy,
    UncertaintyRepresentationPolicy,
)
from fuzzyxai.core.explanation_object import Trace
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent


def _risk_policy() -> MembershipPolicy:
    return MembershipPolicy("malignant_risk", (0.0, 1.0), (
        MembershipTerm("low", "triangular", (0.0, 0.0, 0.5)),
        MembershipTerm("medium", "triangular", (0.2, 0.5, 0.8)),
        MembershipTerm("high", "triangular", (0.5, 1.0, 1.0)),
    ), "preset", "p19-v1", "P19 risk-interface contract")


def _transform() -> AlignmentTransform:
    return AlignmentTransform(
        "class-probability-to-risk-partition-v2", "class_probability", "risk_membership",
        {"class_probability:class:0": "risk:high", "class_probability:class:1": "risk:low"},
        {"source_class_0": "target_high", "source_class_1": "target_low"},
        "risk_membership_partition",
        parameters={"triangles": {"low": [0.0, 0.0, 0.5], "medium": [0.2, 0.5, 0.8], "high": [0.5, 1.0, 1.0]}},
        source_refs=("RandomForestClassifier.predict", "risk_policy:triangular_partition"),
        limitations=("The binary source has no independent medium-vote term; the declared continuous partition supplies it.",),
    )


def _system_plan(transform: AlignmentTransform, membership_policy: MembershipPolicy) -> ExplainPlan:
    """One policy contract shared by accept, conflict, and reduction cases."""

    return ExplainPlan(
        gamma_critical=0.60,
        rho_accept=0.35,
        rho_warning=0.60,
        rho_audit=0.85,
        rho_critical=0.95,
        alignment_policy=AlignmentPolicy(
            applicable=True,
            source="P19 registered transform",
            transform=transform.to_dict(),
        ),
        reduction_policy=ReductionPolicy(
            applicable=True,
            method="F_int_to_F0_midpoint",
            source="P19 interval route",
        ),
        uncertainty_policy=UncertaintyPolicy(
            method="ensemble_vote_standard_deviation",
            source="native per-tree predictions",
        ),
        uncertainty_representation_policy=UncertaintyRepresentationPolicy(
            method="vote_probability_plus_minus_dispersion",
            scale=1.0,
            clip=(0.0, 1.0),
            source="P19 declared heuristic interval; not calibrated",
        ),
        membership_policies={"system_risk": membership_policy},
        metadata={
            "system_risk_weights": {
                "w_p": 0.30,
                "w_u": 0.25,
                "w_I": 0.20,
                "w_Delta": 0.15,
                "w_R": 0.10,
            },
            "system_action_policy": {
                "theta_2_to_theta_3": "request_more_data",
                "theta_3_to_theta_4": "defer_to_human",
            },
            "system_target_consequents": {
                "low": "accept",
                "medium": "review",
                "high": "defer_to_human",
            },
        },
    )


def generate(case: str) -> dict[str, object]:
    data = load_breast_cancer()
    features = [str(name) for name in data.feature_names]
    frame = pd.DataFrame(data.data, columns=features)
    X_train, X_test, y_train, y_test = train_test_split(frame, data.target, test_size=0.2, random_state=0, stratify=data.target)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0).fit(X_train, y_train)
    risk_probabilities = model.predict_proba(X_test)[:, list(model.classes_).index(0)]
    if case == "reduction":
        predictions = model.predict(X_test)
        non_degenerate = np.flatnonzero((risk_probabilities > 0.0) & (risk_probabilities < 1.0) & (predictions == y_test))
        if not len(non_degenerate):
            raise RuntimeError("no correctly predicted real test object has non-degenerate RandomForest votes")
        index = int(non_degenerate[np.argmin(np.abs(risk_probabilities[non_degenerate] - 0.5))])
    else:
        index = int(np.argmin(risk_probabilities))
    object_id = f"p19-{case}-object"
    stamp = datetime.now(UTC).isoformat()
    model_trace = Trace(object_id, "rf-vote-v1", stamp, source="RandomForestClassifier.estimators_", checksum=f"rf:{case}:200")
    target_trace = model_trace if case != "conflict" else Trace(object_id, "risk-policy-unverifiable", stamp, source="missing-checkpoint", checksum="missing")
    transform = _transform()
    membership_policy = _risk_policy()
    plan = _system_plan(transform, membership_policy)
    plan.domain_language = {
        "features": {name: {"label": name} for name in features},
        "classes": {
            "0": {"label": "злокачественное образование", "meaning": "malignant", "domain_defined": True},
            "1": {"label": "доброкачественное образование", "meaning": "benign", "domain_defined": True},
        },
        "actions": {
            "accept": "принять по демонстрационной политике маршрута",
            "lower_confidence": "снизить уверенность",
            "request_more_data": "запросить дополнительные данные",
            "defer_to_human": "передать специалисту",
            "block": "заблокировать автоматическое применение",
        },
    }
    context = ObservationContext(
        reference_data=X_train, reference_labels=y_train, dataset_version="Breast Cancer Wisconsin",
        system_observation=SystemObservation(
            transform,
            membership_policy,
            0,
            target_trace,
            model_trace=model_trace,
            trace_complete=case != "conflict",
            trace_verification_source="P19 controlled runtime trace verifier",
        ),
    )
    result = FuzzyXAI.wrap(model, explain_plan=plan, observation_context=context).explain_one(X_test.iloc[index], object_id=object_id, include_similar_cases=True)
    output = ROOT / f"golden_system_{case}"
    output.mkdir(exist_ok=True)
    (output / "full_report_reader_ru.txt").write_text(result.full_report(level="reader"), encoding="utf-8")
    (output / "full_report_audit_ru.txt").write_text(result.full_report(level="audit"), encoding="utf-8")
    result.export_json(output / "result.json", detail="audit")
    (output / "audit.json").write_text(json.dumps(result.audit(), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(result.inspect("action").to_dict(), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    result.visualize(view="provenance", backend="matplotlib", output=output / "provenance_action.png", selector="action")
    scenario_note = (
        "Controlled fault-injection scenario: ObservationContext intentionally contains an unavailable/unverifiable trace element to verify fail-closed behavior.\n"
        if case == "conflict" else (
            "Real near-boundary object selected from the held-out split to exercise non-degenerate uncertainty reduction.\n"
            if case == "reduction" else "Natural complete-trace acceptance scenario.\n"
        )
    )
    (output / "limitations.txt").write_text(
        scenario_note + "All system quantities are exported from ModelExplanationResult.system; no generator-side scientific calculation occurs.\n",
        encoding="utf-8",
    )
    assert result.system is not None
    return {"object_id": object_id, "test_index": index, "prediction": result.prediction.predictions, "true_label": int(y_test[index]), **result.system.audit_dict(), "action": result.action}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("accept", "conflict", "reduction"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    values = generate(args.case)
    if args.json:
        print(json.dumps(values, ensure_ascii=False, default=str))
