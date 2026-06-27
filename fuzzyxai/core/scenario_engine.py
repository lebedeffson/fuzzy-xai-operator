from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .alignment import compute_alignment, compute_gamma_route
from .diagnostics import DiagnosticType
from .reduction import compute_reduction
from .risk_observer import observe_risk


@dataclass(frozen=True)
class HybridXirisInput:
    image_quality: float = 0.31
    segmentation_quality: float = 0.27
    model_match_signal: float = 0.88
    alpha_accept: float = 0.82
    alpha_block: float = 0.91


@dataclass(frozen=True)
class HybridXirisPlan:
    gamma_components: dict[str, float]
    gamma_weights: dict[str, float]
    gamma_max: float
    delta_t: float
    delta_max: float
    reduction_components: dict[str, float]
    reduction_weights: dict[str, float]
    risk_weights: dict[str, float]
    thresholds: dict[str, float]


@dataclass(frozen=True)
class HybridXirisResult:
    gamma: float
    delta: float
    rho: float
    chi_r: int
    chi_r_crit: int
    action: str
    diagnostic_id: str
    diagnostic_type: str
    occurrences: list[str]


DEFAULT_HYBRID_PLAN = HybridXirisPlan(
    gamma_components={"d_mu": 0.39, "d_R": 0.40, "d_u": 0.5675, "d_tau": 0.0},
    gamma_weights={"d_mu": 0.25, "d_R": 0.35, "d_u": 0.20, "d_tau": 0.20},
    gamma_max=0.40,
    delta_t=0.08,
    delta_max=0.35,
    reduction_components={"source_loss": 0.08, "trace_loss": 0.10, "action_loss": 0.1320275},
    reduction_weights={"source_loss": 0.30, "trace_loss": 0.30, "action_loss": 0.40},
    risk_weights={"model_signal": 0.35, "block_rule": 0.25, "source_conflict": 0.20, "reduction_component": 0.20},
    thresholds={"theta_1": 0.25, "theta_2": 0.45, "theta_3": 0.65, "theta_4": 0.78},
)


def compute_hybrid_xiris(input_values: HybridXirisInput | None = None, plan: HybridXirisPlan = DEFAULT_HYBRID_PLAN) -> HybridXirisResult:
    input_values = input_values or HybridXirisInput()
    alignment = compute_alignment(
        plan.gamma_components,
        plan.gamma_weights,
        gamma_max=plan.gamma_max,
        delta_t=plan.delta_t,
        delta_max=plan.delta_max,
    )
    reduction = compute_reduction(plan.reduction_components, plan.reduction_weights, delta_max=plan.delta_max)
    chi_r = int(input_values.alpha_block > input_values.alpha_accept and input_values.segmentation_quality < 0.35)
    chi_r_crit = int(chi_r == 1 and input_values.model_match_signal >= 0.80)
    risk = observe_risk(
        {
            "model_signal": input_values.model_match_signal,
            "block_rule": input_values.alpha_block,
            "source_conflict": float(chi_r),
            "reduction_component": 0.3225,
        },
        plan.risk_weights,
        plan.thresholds,
        chi_r_crit=chi_r_crit,
    )
    return HybridXirisResult(
        gamma=alignment.gamma,
        delta=reduction.delta,
        rho=risk.rho,
        chi_r=chi_r,
        chi_r_crit=chi_r_crit,
        action=risk.action,
        diagnostic_id="D_source_conflict",
        diagnostic_type=DiagnosticType.QUALITY_SOURCE_CONFLICT.value,
        occurrences=["alignment", "risk_observer", "action"],
    )


def hybrid_xiris_engine_payload() -> dict[str, Any]:
    return {
        "input_values": asdict(HybridXirisInput()),
        "explain_plan": asdict(DEFAULT_HYBRID_PLAN),
        "computed_result": asdict(compute_hybrid_xiris()),
    }


def compute_gis_route_control() -> float:
    return compute_gamma_route(p=0.67, alpha_mean=0.72, feature_support=0.47)
