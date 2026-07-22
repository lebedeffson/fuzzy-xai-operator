"""Expected action losses learned from development review outcomes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionCostProfile:
    prediction_error: float = 8.0
    uncertified_route: float = 10.0
    unstable_explanation: float = 2.0
    deployment_shift: float = 4.0
    short_review: float = 0.5
    full_review: float = 1.5
    false_block: float = 6.0
    short_review_residual: float = 0.45
    full_review_residual: float = 0.10

    def __post_init__(self) -> None:
        if min(self.__dict__.values()) < 0.0:
            raise ValueError("action costs and residual risks cannot be negative")
        if not 0.0 <= self.full_review_residual <= self.short_review_residual <= 1.0:
            raise ValueError("review residuals must satisfy full <= short <= 1")


@dataclass(frozen=True)
class ExpectedActionLosses:
    accept: float
    short_review: float
    full_review: float
    block: float

    def as_dict(self) -> dict[str, float]:
        return {"accept": self.accept, "short_review": self.short_review, "full_review": self.full_review, "block": self.block}


def expected_action_losses(
    predictive_risk: float,
    route_risk: float,
    explanation_risk: float,
    shift_risk: float,
    *,
    hard_fault_probability: float,
    costs: ActionCostProfile,
) -> ExpectedActionLosses:
    base = (
        costs.prediction_error * predictive_risk
        + costs.uncertified_route * route_risk
        + costs.unstable_explanation * explanation_risk
        + costs.deployment_shift * shift_risk
    )
    return ExpectedActionLosses(
        accept=float(base),
        short_review=float(costs.short_review + costs.short_review_residual * base),
        full_review=float(costs.full_review + costs.full_review_residual * base),
        block=float(costs.false_block * (1.0 - hard_fault_probability)),
    )
