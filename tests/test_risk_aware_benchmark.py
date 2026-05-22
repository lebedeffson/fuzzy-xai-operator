from benchmarks.risk_aware_observer_benchmark import build_risk_aware_observer_report, render_markdown


def test_risk_aware_observer_benchmark_report():
    report = build_risk_aware_observer_report(write=False)
    metrics = report['observer_metrics']
    assert report['dataset']['name'] == 'sklearn breast_cancer'
    assert report['model']['accuracy_base'] > 0.8
    assert report['model']['roc_auc'] > 0.9
    assert metrics['risk_reduction'] > 0
    assert metrics['forced_conflict_action'] in {'block', 'defer_to_human'}


def test_risk_aware_observer_markdown():
    report = build_risk_aware_observer_report(write=False)
    md = render_markdown(report)
    assert 'Risk-Aware Observer benchmark' in md
    assert 'Risk reduction' in md
