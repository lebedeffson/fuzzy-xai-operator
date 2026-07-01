from __future__ import annotations

from collections import deque

F_CORE = ['F0', 'F_int', 'F_H', 'F_N_src', 'F_ML']
F_EXT = ['F_IF', 'F_2', 'F_RF']

REDUCTION_EDGES: dict[str, tuple[str, ...]] = {
    'F_ML': ('F_int', 'F_H', 'F_N_src'),
    'F_int': ('F0',),
    'F_H': ('F0',),
    'F_N_src': ('F0',),
    'F0': tuple(),
}


def is_acyclic() -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return False
        if node in visited:
            return True
        visiting.add(node)
        for nxt in REDUCTION_EDGES.get(node, tuple()):
            if not dfs(nxt):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(dfs(n) for n in REDUCTION_EDGES)


def max_depth(start: str = 'F_ML') -> int:
    q = deque([(start, 0)])
    seen: set[tuple[str, int]] = set()
    depth = 0
    while q:
        node, d = q.popleft()
        if (node, d) in seen:
            continue
        seen.add((node, d))
        depth = max(depth, d)
        for nxt in REDUCTION_EDGES.get(node, tuple()):
            q.append((nxt, d + 1))
    return depth


def reduce_path(start: str, target: str = 'F0') -> list[str]:
    if start == target:
        return [start]
    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft()
        node = path[-1]
        for nxt in REDUCTION_EDGES.get(node, tuple()):
            if nxt == target:
                return path + [nxt]
            if nxt not in seen:
                seen.add(nxt)
                q.append(path + [nxt])
    return []


def reduction_terminates(start: str) -> bool:
    path = reduce_path(start, 'F0')
    return bool(path) and path[-1] == 'F0'
