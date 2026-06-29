from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

from fuzzyxai import Candidate, F0, HesitantFS, IntervalFS, MultiLevelFS, NeutrosophicFS, select_minimal_sufficient
from fuzzyxai.hierarchy.meta_reducer import MetaReducer
from .reduction_graph import F_CORE, F_EXT


COGNITIVE_RAW: dict[str, float] = {
    'F0': 0.10,
    'F_int': 0.22,
    'F_H': 0.30,
    'F_N_src': 0.42,
    'F_ML': 0.68,
    'F_IF': 0.50,
    'F_2': 0.55,
    'F_RF': 0.60,
}
_C_MAX_CORE = max(COGNITIVE_RAW[k] for k in F_CORE)


@dataclass(frozen=True)
class RepresentationSelection:
    profile: set[str]
    selected_class: str
    representation_class: str
    reduction_loss: float
    reduction_policy: str
    reason: str
    representation: object
    cognitive_complexity_normalized: float = 0.0
    complexity_note: str = ''

    def as_dict(self) -> dict[str, object]:
        return {
            'profile': sorted(self.profile),
            'selected_class': self.selected_class,
            'representation_class': self.representation_class,
            'reduction_loss': self.reduction_loss,
            'reduction_policy': self.reduction_policy,
            'reason': self.reason,
            'cognitive_complexity_normalized': self.cognitive_complexity_normalized,
            'complexity_note': self.complexity_note,
        }


def _canonical_class_name(selected_name: str) -> str:
    mapping = {
        'FI': 'F_int',
        'FH': 'F_H',
        'FNsrc': 'F_N_src',
        'FML-user': 'F_ML',
        'FML-audit': 'F_ML',
    }
    return mapping.get(selected_name, selected_name)


def normalized_cognitive_complexity(selected_name: str) -> tuple[float, str]:
    canonical = _canonical_class_name(selected_name)
    raw = COGNITIVE_RAW.get(canonical, 0.0)
    if canonical in F_CORE:
        return float(raw / _C_MAX_CORE), 'core scale'
    return float(raw / _C_MAX_CORE), 'extended class, normalized outside core scale'


def default_representation_candidates() -> list[Candidate]:
    return [
        Candidate('F0', {'u_num', 'u_ling'}, 0.10, 0.03, 0.42),
        Candidate('FI', {'u_num', 'u_ling', 'u_int'}, 0.22, 0.05, 0.30),
        Candidate('FH', {'u_num', 'u_ling', 'u_exp'}, 0.30, 0.81, 0.25),
        Candidate('FNsrc', {'u_num', 'u_ling', 'u_if', 'u_conf', 'u_trace'}, 0.42, 0.09, 0.18),
        Candidate('FML-user', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_multi'}, 0.60, 0.99, 0.08),
        Candidate('FML-audit', {'u_num', 'u_ling', 'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_multi'}, 0.68, 1.00, 0.04),
    ]


def profile_from_dataset_profile(dataset_profile) -> set[str]:
    profile: set[str] = set(getattr(dataset_profile, 'suggested_uncertainty_types', []) or [])
    if 'u_expert' in profile:
        profile.discard('u_expert')
        profile.add('u_exp')
    if getattr(dataset_profile, 'numeric_columns', []):
        profile.add('u_num')
    if getattr(dataset_profile, 'categorical_columns', []):
        profile.add('u_ling')
    else:
        profile.add('u_ling')
    if getattr(dataset_profile, 'has_intervals', False):
        profile.add('u_int')
    if getattr(dataset_profile, 'has_multiple_experts', False):
        profile.add('u_exp')
    if getattr(dataset_profile, 'has_source_conflict', False):
        profile.add('u_conf')
    if getattr(dataset_profile, 'requires_audit', False):
        profile.add('u_trace')
    complex_types = {'u_int', 'u_exp', 'u_conf', 'u_trace'}
    if len(profile & complex_types) >= 2:
        profile.add('u_multi')
    return profile


def select_risk_representation(risk: float, profile: Iterable[str], mode: str = 'audit') -> RepresentationSelection:
    profile_set = set(profile)
    selected = select_minimal_sufficient(profile_set, default_representation_candidates(), mode=mode)
    selected_name = getattr(selected, 'name', getattr(selected, 'code', 'D_choice'))
    representation = build_representation_for_risk(risk, profile_set, selected_name)
    reduction = MetaReducer(goal=mode).reduce(representation)
    reason = _selection_reason(profile_set, selected_name)
    c_norm, c_note = normalized_cognitive_complexity(selected_name)
    return RepresentationSelection(
        profile=profile_set,
        selected_class=selected_name,
        representation_class=getattr(representation, 'class_name', type(representation).__name__),
        reduction_loss=float(reduction.delta),
        reduction_policy=reduction.policy,
        reason=reason,
        representation=representation,
        cognitive_complexity_normalized=c_norm,
        complexity_note=c_note,
    )


def build_representation_for_risk(risk: float, profile: Set[str], selected_class: str):
    z = max(0.0, min(1.0, float(risk)))
    if selected_class.startswith('FML') or 'u_multi' in profile:
        levels = []
        if 'u_int' in profile or selected_class.startswith('FML'):
            levels.append(IntervalFS(lambda x: max(0.0, z - 0.04), lambda x: min(1.0, z + 0.04)))
        if 'u_exp' in profile or selected_class.startswith('FML'):
            levels.append(HesitantFS(lambda x: [max(0.0, z - 0.08), min(1.0, z + 0.06)]))
        if 'u_conf' in profile or 'u_trace' in profile or selected_class.startswith('FML'):
            levels.append(NeutrosophicFS(lambda x: z, lambda x: 0.15 if 'u_conf' in profile else 0.05, lambda x: max(0.0, 1.0 - z - 0.04), source_t='model', source_f='observer'))
        if not levels:
            levels.append(F0(lambda x: z, 'risk'))
        return MultiLevelFS(levels, gamma={('dataset_profile', 'risk', 'same_case')} if len(levels) > 1 else set())
    if selected_class == 'FNsrc' or 'u_conf' in profile or 'u_trace' in profile:
        return NeutrosophicFS(lambda x: z, lambda x: 0.15 if 'u_conf' in profile else 0.05, lambda x: max(0.0, 1.0 - z - 0.04), source_t='model', source_f='observer')
    if selected_class == 'FH' or 'u_exp' in profile:
        return HesitantFS(lambda x: [max(0.0, z - 0.08), min(1.0, z + 0.06)])
    if selected_class == 'FI' or 'u_int' in profile:
        return IntervalFS(lambda x: max(0.0, z - 0.04), lambda x: min(1.0, z + 0.04))
    return F0(lambda x: z, 'risk')


def _selection_reason(profile: set[str], selected_class: str) -> str:
    if selected_class == 'D_choice':
        return 'no candidate covers the inferred uncertainty profile'
    if selected_class.startswith('FML'):
        return 'multi-level/audit profile requires preserving several uncertainty types'
    if selected_class == 'FNsrc':
        return 'trace or source-conflict uncertainty requires source-aware representation'
    if selected_class == 'FH':
        return 'multiple expert estimates require hesitant representation'
    if selected_class == 'FI':
        return 'interval-valued features require interval representation'
    return 'numeric/linguistic uncertainty is covered by classical fuzzy representation'
