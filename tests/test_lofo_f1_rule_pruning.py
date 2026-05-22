from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmarks.lofo_f1_rule_pruning_demo import build_lofo_f1_report
from fuzzyxai.rules import (
    bootstrap_lofo_f1_importance,
    budget_prune_importance,
    jaccard_similarity,
    lofo_f1_importance,
    select_top_rules_by_lofo_f1,
)


def test_lofo_f1_finds_rule_that_protects_f1():
    h = np.array([
        [1.0, 0.0, 0.2],
        [1.0, 0.1, 0.0],
        [0.0, 1.0, 0.1],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    theta = np.array([3.0, -3.0, 0.0])
    y = np.array([1, 1, 0, 0, 0, 0])

    rows = lofo_f1_importance(h, theta, y, bias=-0.5, rule_names=['positive_rule', 'negative_rule', 'noise'])
    assert rows[0].rule_name in {'positive_rule', 'negative_rule'}
    assert rows[0].delta_f1 > 0
    assert {idx for idx in select_top_rules_by_lofo_f1(rows, 2)} == {0, 1}


def test_bootstrap_lofo_f1_reports_stable_frequency():
    rng = np.random.default_rng(0)
    h = rng.random((80, 5))
    theta = np.array([2.5, -2.2, 0.0, 0.0, 0.0])
    y = ((-0.2 + h @ theta) >= 0).astype(int)

    rows = bootstrap_lofo_f1_importance(h, theta, y, bias=-0.2, n_bootstraps=12, top_k=2, random_state=1)
    top = {rows[0].rule_index, rows[1].rule_index}
    assert top == {0, 1}
    assert rows[0].top_frequency > 0.5


def test_budget_prune_and_jaccard_helpers():
    h = np.array([[1.0, 0.5], [0.0, 0.5]])
    theta = np.array([2.0, 1.0])
    scores = budget_prune_importance(h, theta)
    assert scores.tolist() == [1.0, 0.5]
    assert jaccard_similarity([1, 2], [2, 3]) == 1 / 3


def test_lofo_f1_demo_writes_report():
    report = build_lofo_f1_report(write=True)
    assert report['baseline']['f1_full'] > 0.7
    assert report['subset_quality']['lofo_f1_top_budget_f1'] > 0.65
    assert Path('reports/lofo_f1_rule_pruning.json').exists()
    assert Path('reports/lofo_f1_rule_pruning.md').exists()
