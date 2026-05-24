from __future__ import annotations

from experiments.chapter2_breast_cancer_demo import run


def test_chapter2_breast_cancer_demo_smoke():
    df, summary = run(seed=1, max_cases=24)
    assert len(df) == 24
    assert (df['i_pre'] >= 0).all() and (df['i_pre'] <= 1).all()
    assert summary['model_accuracy'] > 0.85
    assert summary['model_roc_auc'] > 0.9
