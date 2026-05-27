from __future__ import annotations

from fuzzyxai.risk.reduction_graph import REDUCTION_EDGES, is_acyclic, max_depth, reduce_path, reduction_terminates


def test_reduction_graph_is_acyclic() -> None:
    assert is_acyclic() is True


def test_reduction_graph_max_depth_two() -> None:
    assert max_depth('F_ML') == 2


def test_reduction_terminates_to_f0() -> None:
    for cls in ['F_ML', 'F_int', 'F_H', 'F_N_src', 'F0']:
        assert reduction_terminates(cls) is True


def test_no_upward_edges_from_f0() -> None:
    assert REDUCTION_EDGES['F0'] == tuple()


def test_reduce_path_example() -> None:
    path = reduce_path('F_ML', 'F0')
    assert path[0] == 'F_ML'
    assert path[-1] == 'F0'
