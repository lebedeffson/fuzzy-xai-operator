from __future__ import annotations

from math import prod
from typing import Any

from .contracts import ReductionArtifact, RepresentationArtifact, UncertaintyProfile


def triangular(x: float, a: float, b: float, c: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    if (a == b and x == a) or (b == c and x == c):
        return 1.0
    if x == b:
        return 1.0
    if x <= a or x >= c:
        return 0.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def memberships(probability: float) -> dict[str, float]:
    return {
        "low": triangular(probability, 0.0, 0.0, 0.5),
        "medium": triangular(probability, 0.25, 0.5, 0.75),
        "high": triangular(probability, 0.5, 1.0, 1.0),
    }


def product_tnorm(*values: float) -> float:
    return float(prod(values))


def probabilistic_sum(*values: float) -> float:
    return 1.0 - float(prod(1.0 - value for value in values))


def select_representation(profile: UncertaintyProfile, probability: float) -> RepresentationArtifact:
    kinds = set(profile.present_types)
    base = memberships(probability)
    if {"u_int", "u_conf", "u_trace"}.issubset(kinds) or len(kinds) >= 6:
        return RepresentationArtifact(
            "F_ML",
            {"level_0": base, "interval_width": profile.values.get("u_int", 0.0), "conflict": profile.conflict},
            tuple(sorted(kinds)), 5.0, 4.0,
            "interval, conflict, and trace uncertainty require a multilevel representation",
        )
    if "u_conf" in kinds or "u_rule" in kinds:
        truth = base["high"]
        falsity = base["low"]
        indeterminacy = min(1.0, profile.aggregate + profile.conflict)
        return RepresentationArtifact(
            "NAS", {"truth": truth, "indeterminacy": indeterminacy, "falsity": falsity},
            tuple(sorted(kinds)), 3.0, 2.0, "registered evidence sources disagree",
        )
    if "u_int" in kinds:
        width = profile.values.get("u_int", 0.1)
        interval = {key: [max(0.0, value - width), min(1.0, value + width)] for key, value in base.items()}
        return RepresentationArtifact(
            "F_int", interval, tuple(sorted(kinds)), 2.0, 2.0,
            "registered feature intervals require interval membership",
        )
    return RepresentationArtifact(
        "F0", base, tuple(sorted(kinds)), 1.0, 1.0,
        "point evidence is complete and has no registered conflict",
    )


def reduce_representation(
    representation: RepresentationArtifact,
    *,
    maximum_loss: float,
    stress: bool = False,
) -> ReductionArtifact:
    membership = representation.membership
    if representation.representation_id == "F0":
        reduced = {key: float(value) for key, value in membership.items()}
        loss = 0.0
    elif representation.representation_id == "F_int":
        reduced = {key: (float(value[0]) + float(value[1])) / 2.0 for key, value in membership.items()}
        loss = max((float(value[1]) - float(value[0])) / 2.0 for value in membership.values())
    elif representation.representation_id == "NAS":
        reduced = {"low": float(membership["falsity"]), "medium": float(membership["indeterminacy"]), "high": float(membership["truth"])}
        loss = float(membership["indeterminacy"]) * 0.5
    else:
        reduced = dict(membership["level_0"])
        loss = max(float(membership.get("interval_width", 0.0)), float(membership.get("conflict", 0.0)))
    if stress:
        loss = max(loss, maximum_loss + 0.15)
    loss = min(1.0, loss)
    return ReductionArtifact(
        representation.representation_id, "F0", reduced, loss,
        loss <= maximum_loss, maximum_loss,
    )


def uncertainty_profile(*, probability: float, controls: dict[str, Any], trace_complete: bool) -> UncertaintyProfile:
    values = {
        "u_num": 1.0 - abs(probability - 0.5) * 2.0,
        "u_ling": 0.05,
    }
    present = {"u_num", "u_ling"}
    if controls.get("feature_interval"):
        values["u_int"] = min(0.4, float(controls.get("feature_interval_width", 0.12)))
        present.add("u_int")
    conflict = 0.65 if controls.get("rule_counter_evidence") else 0.0
    if conflict:
        values.update({"u_conf": conflict, "u_rule": conflict})
        present.update({"u_conf", "u_rule"})
    if not trace_complete or controls.get("trace_complexity"):
        values["u_trace"] = 0.5
        present.add("u_trace")
    if controls.get("distribution_shift"):
        values["u_shift"] = 0.5
        present.add("u_shift")
    weights = {"u_num": 0.40, "u_rule": 0.25, "u_ling": 0.20, "u_trace": 0.15}
    weighted = sum(values.get(key, 0.0) * weight for key, weight in weights.items())
    return UncertaintyProfile(tuple(sorted(present)), values, min(1.0, weighted), conflict, 1.0 if trace_complete else 0.0)
