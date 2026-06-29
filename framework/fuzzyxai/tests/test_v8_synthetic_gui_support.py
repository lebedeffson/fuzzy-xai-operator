from fuzzyxai.demo.synthetic import (
    sample_dataframe,
    default_plan,
    build_demo_explanation,
    build_demo_composition,
)


def test_v8_sample_data_and_plan():
    df = sample_dataframe()
    assert 'target' in df.columns
    assert len(df) >= 10
    plan = default_plan()
    assert plan.metadata['n_rows'] == len(df)
    assert plan.metadata['numeric_features']


def test_v8_demo_explanation_is_ready_without_uploads():
    last = build_demo_explanation(0.72, plan=default_plan())
    assert last['report']['selected_class'] == 'FML-audit'
    assert last['object'].reduction_loss > 0
    assert last['text']


def test_v8_composition_has_default_graph_and_diagnostic_mode():
    ok = build_demo_composition(0.72, 0.74, plan=default_plan(), conflict=False)
    assert hasattr(ok['comp'], 'uncertainty')
    assert ok['edges']
    bad = build_demo_composition(0.72, 0.74, plan=default_plan(), conflict=True)
    assert getattr(bad['comp'], 'code', None) == 'D_ij'
