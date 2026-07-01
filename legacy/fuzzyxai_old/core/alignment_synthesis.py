from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from fuzzyxai.core.diagnostics import DiagnosticState
from fuzzyxai.core.explanation_object import ExplanationObject


@dataclass(frozen=True)
class AlignmentCandidate:
    name: str
    term_map: dict[str, str]
    rule_map: dict[str, str]
    activation_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentResult:
    candidate: AlignmentCandidate
    gamma: float
    Delta_T: float
    J: float

    def to_dict(self) -> dict[str, Any]:
        return {'T_ij': self.candidate.to_dict(), 'gamma': self.gamma, 'Delta_T': self.Delta_T, 'J(T)': self.J}


def _plan_dict(explain_plan: Any) -> dict[str, Any]:
    if hasattr(explain_plan, 'to_dict'):
        return dict(explain_plan.to_dict())
    if isinstance(explain_plan, Mapping):
        return dict(explain_plan)
    return {}


def _alignment_space(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    synthesis = plan.get('alignment_synthesis') or plan.get('metadata', {}).get('alignment_synthesis')
    if not isinstance(synthesis, Mapping):
        return []
    candidates = synthesis.get('candidates')
    if not isinstance(candidates, list):
        return []
    return [dict(c) for c in candidates]


def enumerate_alignment_candidates(E_i: ExplanationObject, E_j: ExplanationObject, explain_plan: Any) -> list[AlignmentCandidate] | DiagnosticState:
    plan = _plan_dict(explain_plan)
    raw_candidates = _alignment_space(plan)
    if not raw_candidates:
        return DiagnosticState(
            code='alignment_space_not_finite',
            reason='Rho/Sigma/Pi/Theta are not given as finite alignment candidates in ExplainPlan',
            severity='warning',
            context={'required_key': 'alignment_synthesis.candidates'},
        )
    out: list[AlignmentCandidate] = []
    for idx, raw in enumerate(raw_candidates):
        term_map = {str(k): str(v) for k, v in dict(raw.get('term_map', {})).items()}
        rule_map = {str(k): str(v) for k, v in dict(raw.get('rule_map', {})).items()}
        if set(term_map).issubset(E_i.terms) and set(term_map.values()).issubset(E_j.terms):
            out.append(AlignmentCandidate(str(raw.get('name', f'T_{idx}')), term_map, rule_map, float(raw.get('activation_scale', 1.0))))
    if not out:
        return DiagnosticState('no_admissible_alignment_candidate', 'finite candidate set exists but no candidate matches E_i/E_j terms', 'warning')
    return out


def score_alignment_candidate(candidate: AlignmentCandidate, E_i: ExplanationObject, E_j: ExplanationObject, explain_plan: Any) -> dict[str, float]:
    matched_terms = sum(1 for src, dst in candidate.term_map.items() if src in E_i.terms and dst in E_j.terms)
    term_total = max(1, len(E_i.terms | set(candidate.term_map.keys())))
    term_gap = 1.0 - matched_terms / term_total

    src_rules = {r.name for r in E_i.rules}
    dst_rules = {r.name for r in E_j.rules}
    matched_rules = sum(1 for src, dst in candidate.rule_map.items() if src in src_rules and dst in dst_rules)
    rule_total = max(1, len(src_rules | set(candidate.rule_map.keys())))
    rule_gap = 1.0 - matched_rules / rule_total

    act_gap = 0.0
    compared = 0
    for src, dst in candidate.rule_map.items():
        if src in E_i.activations and dst in E_j.activations:
            act_gap += abs(float(E_i.activations[src]) * candidate.activation_scale - float(E_j.activations[dst]))
            compared += 1
    gamma = min(1.0, (act_gap / compared if compared else 0.0) + 0.5 * term_gap + 0.3 * rule_gap)
    Delta_T = min(1.0, abs(float(E_i.reduction_loss) - float(E_j.reduction_loss)) + 0.25 * term_gap)
    plan = _plan_dict(explain_plan)
    lam = float(plan.get('alignment_synthesis', {}).get('lambda_delta', plan.get('metadata', {}).get('alignment_synthesis', {}).get('lambda_delta', 0.2)))
    J = gamma + lam * Delta_T
    return {'gamma': round(gamma, 6), 'Delta_T': round(Delta_T, 6), 'J(T)': round(J, 6)}


def synthesize_alignment(E_i: ExplanationObject, E_j: ExplanationObject, explain_plan: Any) -> AlignmentResult | DiagnosticState:
    candidates = enumerate_alignment_candidates(E_i, E_j, explain_plan)
    if isinstance(candidates, DiagnosticState):
        return candidates
    scored = [(candidate, score_alignment_candidate(candidate, E_i, E_j, explain_plan)) for candidate in candidates]
    candidate, score = min(scored, key=lambda item: item[1]['J(T)'])
    plan = _plan_dict(explain_plan)
    gamma_max = float(plan.get('alignment_synthesis', {}).get('gamma_max', plan.get('composition', {}).get('gamma_max', plan.get('metadata', {}).get('gamma_max', 0.4))))
    if score['J(T)'] > gamma_max:
        return DiagnosticState(
            code='alignment_cost_exceeds_gamma_max',
            reason='min J(T) is greater than gamma_max',
            severity='critical',
            context={'gamma_max': gamma_max, **score, 'candidate': candidate.to_dict()},
        )
    return AlignmentResult(candidate=candidate, gamma=score['gamma'], Delta_T=score['Delta_T'], J=score['J(T)'])


def write_alignment_report(path: str | Path, E_i: ExplanationObject, E_j: ExplanationObject, explain_plan: Any) -> dict[str, Any]:
    result = synthesize_alignment(E_i, E_j, explain_plan)
    payload = result.to_dict() if isinstance(result, AlignmentResult) else {'DiagnosticState': asdict(result)}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload
