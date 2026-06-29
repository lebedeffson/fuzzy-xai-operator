import pandas as pd

from fuzzyxai import FuzzyXAIPipeline, ExplainPlan
from fuzzyxai.hierarchy import IntervalFS, NeutrosophicFS, MultiLevelFS
from fuzzyxai.hierarchy.meta_reducer import MetaReducer
from fuzzyxai.calibration import synthetic_calibration_pairs, cross_validate_beta


def test_explain_plan_from_data_and_pipeline():
    df = pd.DataFrame({'age': [40, 50, 60, 70], 'risk': [0.2, 0.5, 0.7, 0.9], 'target': [0, 0, 1, 1]})
    pipe = FuzzyXAIPipeline.from_data(df, target='target')
    result = pipe.explain_scalar_risk(0.72, metadata={'has_intervals': True}, component_id='test')
    assert 'u_int' in result.profile
    assert result.object.metadata['memberships']['high'] > result.object.metadata['memberships']['medium']


def test_meta_reducer_for_multilevel():
    interval = IntervalFS(lambda x: 0.3, lambda x: 0.7)
    neutro = NeutrosophicFS(lambda x: 0.7, lambda x: 0.2, lambda x: 0.5)
    multi = MultiLevelFS([interval, neutro], weights=[0.4, 0.6])
    reduced = MetaReducer(goal='audit').reduce(multi)
    assert 0 <= reduced.delta <= 1
    assert reduced.policy == 'canonical'


def test_cross_validation_report_shape():
    report = cross_validate_beta(synthetic_calibration_pairs(n=20, seed=1), folds=4, seed=1)
    assert report['mean_mse_calibrated'] >= 0
    assert len(report['folds']) == 4
