from __future__ import annotations

from fuzzyxai.category.certified_path import CertifiedEdge, CertifiedPath, compose_certified_paths, rupture_path


def _edge(source: str, target: str, *, gamma: float = 0.1, gamma_max: float = 0.45, passed: bool = True):
    return CertifiedEdge(source=source, target=target, alignment=f'T_{source}_{target}', gamma=gamma, gamma_max=gamma_max, passed=passed)


def test_identity_is_empty_certified_path() -> None:
    pid = CertifiedPath.identity('E_model')
    assert pid.source == pid.target == 'E_model'
    assert pid.edges == tuple()
    assert pid.certified is True


def test_composition_is_associative() -> None:
    p = CertifiedPath.from_edge(_edge('A', 'B'))
    q = CertifiedPath.from_edge(_edge('B', 'C'))
    r = CertifiedPath.from_edge(_edge('C', 'D'))
    left = compose_certified_paths(compose_certified_paths(p, q), r)
    right = compose_certified_paths(p, compose_certified_paths(q, r))
    assert left.source == right.source == 'A'
    assert left.target == right.target == 'D'
    assert left.objects == right.objects
    assert left.certified and right.certified


def test_certified_edges_imply_certified_path() -> None:
    p = CertifiedPath.from_edge(_edge('E_model', 'E_risk'))
    q = CertifiedPath.from_edge(_edge('E_risk', 'E_action'))
    c = p.compose(q)
    assert c.certified is True
    assert c.rupture is False


def test_missing_edge_returns_rupture_not_bad_morphism() -> None:
    p = CertifiedPath.from_edge(_edge('A', 'B'))
    q = CertifiedPath.from_edge(_edge('C', 'D'))
    composed = p.compose(q)
    assert composed.rupture is True
    assert 'path_endpoint_mismatch' in composed.diagnostics


def test_explicit_rupture_path() -> None:
    rp = rupture_path('E_i', 'E_j', diagnostics=['critical_missing_alignment'])
    assert rp.rupture is True
    assert rp.certified is False
