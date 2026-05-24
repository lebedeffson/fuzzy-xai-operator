from __future__ import annotations

from experiments.full_pipeline_demo import run


def test_full_pipeline_experiment_smoke():
    df, summary = run(seed=1, max_cases=32)
    assert len(df) == 32
    assert (df['I_pre'] >= 0).all() and (df['I_pre'] <= 1).all()
    assert (df['rho'] >= 0).all() and (df['rho'] <= 1).all()
    assert summary['model_accuracy'] > 0.85
