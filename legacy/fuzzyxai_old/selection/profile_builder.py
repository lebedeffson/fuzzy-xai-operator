from typing import Mapping, Set

def build_profile(metadata: Mapping[str, object]) -> Set[str]:
    profile = {'u_num', 'u_ling'}
    if metadata.get('has_intervals'):
        profile.add('u_int')
    if int(metadata.get('num_experts', 1)) > 1:
        profile.add('u_exp')
    if metadata.get('source_conflict'):
        profile.add('u_conf')
    if metadata.get('requires_audit'):
        profile.add('u_trace')
    if metadata.get('dataset_shift'):
        profile.add('u_shift')
    if metadata.get('counterfactual_instability'):
        profile.add('u_cf')
    if metadata.get('multi_level'):
        profile.add('u_multi')
    return profile
