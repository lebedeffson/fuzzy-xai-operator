from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from fuzzyxai.risk.decision_policy import RiskPolicy
from fuzzyxai.risk.risk_function import compute_application_risk


@dataclass
class StudioExplainPlan:
    w_p: float = 0.70
    w_u: float = 0.05
    w_I: float = 0.05
    w_Delta: float = 0.05
    w_R: float = 0.15
    theta_1: float = 0.10
    theta_2: float = 0.25
    theta_3: float = 0.50
    theta_4: float = 0.75
    i_min: float = 0.65
    delta_max: float = 0.15
    uncertainty_max: float = 0.45
    gamma_max: float = 0.45
    epsilon_int: float = 0.10
    reduction_strategy: str = 'balance'
    baseline_access: str = 'native'
    mode: str = 'audit'
    use_trace: bool = True
    use_delta: bool = True
    use_critical_rupture: bool = True
    use_topos: bool = True

    def risk_weights(self) -> dict[str, float]:
        return {
            'predicted_risk': float(self.w_p),
            'uncertainty': float(self.w_u),
            'interpretability_gap': float(self.w_I),
            'reduction_loss': float(self.w_Delta),
            'diagnostic': float(self.w_R),
        }


@dataclass
class WhatIfOverrides:
    predicted_risk: float | None = None
    uncertainty: float | None = None
    i_pre: float | None = None
    reduction_loss: float | None = None
    chi_r: int | None = None
    chi_r_crit: int | None = None
    chi_auto: bool | None = None
    trace_valid: bool | None = None
    source_conflict: bool | None = None


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y', 'allow', 'allowed'}
    return bool(value)


def _action_reason(action: str, chi_r_crit: int, chi_auto: bool, rho: float) -> str:
    if chi_r_crit:
        return 'critical rupture detected'
    if not chi_auto:
        return 'context forbids automatic action'
    if action == 'accept':
        return 'all safety constraints satisfied'
    if action == 'lower_confidence':
        return 'low risk but not enough for full accept'
    if action == 'request_more_data':
        return 'insufficient confidence/interpretability/trace quality'
    if action == 'defer_to_human':
        return 'human review required'
    if action == 'block':
        return f'high risk ({rho:.3f})'
    return 'policy decision'


def recompute_case_state(
    case_state: dict[str, Any],
    plan: StudioExplainPlan,
    overrides: WhatIfOverrides | None = None,
) -> dict[str, Any]:
    st = deepcopy(case_state)
    ov = overrides or WhatIfOverrides()

    pred = float(ov.predicted_risk if ov.predicted_risk is not None else st['model'].get('p_malignant', st['model'].get('predicted_risk', 0.5)))
    unc = float(ov.uncertainty if ov.uncertainty is not None else st['model'].get('uncertainty', 0.0))
    i_pre = float(ov.i_pre if ov.i_pre is not None else st['explanation'].get('I_pre', 1.0))
    delta_raw = float(ov.reduction_loss if ov.reduction_loss is not None else st['uncertainty'].get('delta', 0.0))
    delta = 0.0 if not plan.use_delta else delta_raw

    chi_r = int(ov.chi_r if ov.chi_r is not None else st['risk'].get('chi_R', 0))
    chi_r_crit = int(ov.chi_r_crit if ov.chi_r_crit is not None else st['risk'].get('chi_R_crit', 0))
    if not plan.use_critical_rupture:
        chi_r_crit = 0
    if chi_r_crit:
        chi_r = 1

    if ov.source_conflict is True:
        chi_r = 1
    trace_valid_default = True
    trace_valid = bool(ov.trace_valid if ov.trace_valid is not None else trace_valid_default)

    auto_e_action = st.get('contexts', {}).get('AutoAccept', {}).get('E_action', True)
    chi_auto_raw = bool(ov.chi_auto if ov.chi_auto is not None else _as_bool(auto_e_action))
    chi_auto = bool(True if not plan.use_topos else chi_auto_raw)

    diagnostics: list[str] = []
    if chi_r:
        diagnostics.append('rupture')
    if chi_r_crit:
        diagnostics.append('critical_rupture')
    if not trace_valid and plan.use_trace:
        diagnostics.append('trace_gap')

    weights = plan.risk_weights()
    breakdown = compute_application_risk(
        predicted_risk=pred,
        uncertainty=unc,
        pre_interpretability=i_pre,
        reduction_loss=delta,
        diagnostics=diagnostics,
        weights=weights,
    )

    policy = RiskPolicy(
        theta_1=float(plan.theta_1),
        theta_2=float(plan.theta_2),
        theta_3=float(plan.theta_3),
        theta_4=float(plan.theta_4),
        i_min=float(plan.i_min),
        delta_max=float(plan.delta_max),
        uncertainty_max=float(plan.uncertainty_max),
        risk_weights=weights,
    )
    decision = policy.choose_from_risk(
        application_risk=float(breakdown.rho),
        uncertainty=unc,
        predicted_risk=pred,
        pre_interpretability=i_pre,
        reduction_loss=delta,
        diagnostics=diagnostics,
        chi_r=chi_r,
        chi_r_crit=chi_r_crit,
        chi_auto=chi_auto,
        trace_valid=(trace_valid if plan.use_trace else True),
    )

    contributions = {
        'predicted_risk': weights['predicted_risk'] * breakdown.predicted_risk,
        'uncertainty': weights['uncertainty'] * breakdown.uncertainty,
        'interpretability_gap': weights['interpretability_gap'] * breakdown.interpretability_gap,
        'reduction_loss': weights['reduction_loss'] * breakdown.reduction_loss,
        'chi_R': weights['diagnostic'] * (1.0 if diagnostics else 0.0),
    }

    st['model']['p_malignant'] = _clip01(pred)
    st['model']['predicted_risk'] = _clip01(pred)
    st['model']['uncertainty'] = _clip01(unc)
    st['explanation']['I_pre'] = _clip01(i_pre)
    st['uncertainty']['delta'] = _clip01(delta)

    st['risk']['rho'] = _clip01(breakdown.rho)
    st['risk']['action'] = str(decision.action.value)
    st['risk']['chi_R'] = int(chi_r)
    st['risk']['chi_R_crit'] = int(chi_r_crit)
    st['risk']['reason'] = _action_reason(str(decision.action.value), int(chi_r_crit), bool(chi_auto), float(breakdown.rho))
    st['risk']['thresholds'] = (float(plan.theta_1), float(plan.theta_2), float(plan.theta_3), float(plan.theta_4))
    st['risk']['weights'] = dict(weights)
    st['risk']['components'] = {
        'predicted_risk': float(breakdown.predicted_risk),
        'uncertainty': float(breakdown.uncertainty),
        'interpretability_gap': float(breakdown.interpretability_gap),
        'reduction_loss': float(breakdown.reduction_loss),
        'chi_R': int(chi_r),
        'chi_R_crit': int(chi_r_crit),
    }
    st['risk']['breakdown'] = {
        'rho_p': float(breakdown.predicted_risk),
        'u_M': float(breakdown.uncertainty),
        'interpretability_gap': float(breakdown.interpretability_gap),
        'reduction_loss': float(breakdown.reduction_loss),
        'weights': dict(weights),
        'contributions': contributions,
        'threshold': float(plan.theta_4),
        'chi_auto': bool(chi_auto),
        'trace_valid': bool(trace_valid),
    }

    st.setdefault('contexts', {}).setdefault('AutoAccept', {})['E_action'] = bool(chi_auto)
    st.setdefault('composition', {})['gamma_max'] = float(plan.gamma_max)
    st.setdefault('composition', {})['ruptures'] = diagnostics
    st.setdefault('route_header', {})['Action'] = str(decision.action.value)
    st.setdefault('route_header', {})['Expl'] = 'rupture' if chi_r else 'morphism'
    st['plan'] = asdict(plan)
    return st
