from __future__ import annotations

from fuzzyxai.category import CertifiedEdge, CertifiedPath, rupture_path
from fuzzyxai.hott import certified_paths_equivalent, path_inhabited, rupture_inhabited


def _edge(source: str, target: str, *, gamma: float = 0.1, gamma_max: float = 0.45, passed: bool = True):
    return CertifiedEdge(source=source, target=target, alignment=f'T_{source}_{target}', gamma=gamma, gamma_max=gamma_max, passed=passed)


def test_path_inhabited_iff_certified_path_exists() -> None:
    p = CertifiedPath.from_edge(_edge('E_i', 'E_j'))
    assert path_inhabited(p) is True


def test_rupture_inhabited_when_no_certified_path() -> None:
    rp = rupture_path('E_i', 'E_j', diagnostics=['critical_missing_alignment'])
    assert rupture_inhabited(rp) is True


def test_path_equivalence_by_target_and_diagnostics() -> None:
    p1 = CertifiedPath.from_edge(_edge('A', 'B'))
    p2 = CertifiedPath.from_edge(_edge('A', 'B'))
    assert certified_paths_equivalent(p1, p2) is True
