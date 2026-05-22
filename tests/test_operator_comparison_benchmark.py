from benchmarks.operator_comparison_benchmark import build_operator_comparison_report, render_markdown


def test_operator_comparison_benchmark_report():
    report = build_operator_comparison_report(write=False)
    assert report['dataset']['name'] == 'sklearn breast_cancer'
    assert report['model']['accuracy'] > 0.8
    assert report['model']['roc_auc'] > 0.9
    assert 'semantic_gamma' in report['without_operator']['missing']
    assert report['with_operator']['conflict_detection_rate'] == 1.0
    assert report['with_operator']['mean_interpretability_index'] is not None


def test_operator_comparison_markdown_mentions_operator_value():
    report = build_operator_comparison_report(write=False)
    markdown = render_markdown(report)
    assert 'без оператора / с оператором' in markdown
    assert 'D_ij' in markdown
    assert 'gamma' in markdown
