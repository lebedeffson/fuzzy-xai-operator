from typing import Iterable, Set

def compatible_types(types: Iterable[str]) -> bool:
    t = set(types)
    incompatible = [
        {'u_exp', 'u_conf'}, {'u_cf', 'u_num'}, {'u_time', 'u_num'},
        {'u_trace', 'u_exp'}, {'u_shift', 'u_ling'}
    ]
    return not any(pair <= t for pair in incompatible)

def synthesize_levels(profile: Set[str]):
    levels = []
    for typ in sorted(profile):
        for level in levels:
            if compatible_types(level | {typ}):
                level.add(typ)
                break
        else:
            levels.append({typ})
    return levels
