from __future__ import annotations

import pandas as pd

from fuzzyxai.data import DatasetRecord, infer_dataset_profile
from fuzzyxai.risk import profile_from_dataset_profile, select_risk_representation
from fuzzyxai.pipelines import DatasetObserverPipeline


def _rich_uncertainty_df():
    return pd.DataFrame({
        'risk_min': [0.10, 0.20, 0.35, 0.50, 0.70, 0.80, 0.15, 0.25, 0.42, 0.60, 0.74, 0.90],
        'risk_max': [0.20, 0.32, 0.50, 0.70, 0.85, 0.95, 0.24, 0.38, 0.58, 0.78, 0.90, 1.00],
        'expert_a': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
        'expert_b': [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        'source_model': [0.1, 0.2, 0.4, 0.6, 0.7, 0.9, 0.2, 0.3, 0.5, 0.7, 0.8, 0.95],
        'source_expert': [0.2, 0.1, 0.5, 0.5, 0.8, 0.85, 0.1, 0.4, 0.4, 0.8, 0.75, 0.9],
        'target': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
    })


def test_dataset_profile_selects_multilevel_representation():
    profile = infer_dataset_profile(_rich_uncertainty_df())
    xai_profile = profile_from_dataset_profile(profile)
    selected = select_risk_representation(0.72, xai_profile, mode='audit')
    assert {'u_int', 'u_exp', 'u_conf', 'u_trace', 'u_multi'} <= selected.profile
    assert selected.selected_class == 'FML-audit'
    assert selected.reduction_loss >= 0.0


def test_dataset_observer_report_contains_selected_representation_and_objects():
    record = DatasetRecord('rich', 'unit-test', target_column='target')
    result = DatasetObserverPipeline(mode='audit').run(record, _rich_uncertainty_df(), case_index=0)
    obs = result.observer_result
    assert obs['selected_representation'] == 'FML-audit'
    assert obs['composition_route'] == ['E_M_ext', 'E_R', 'E_A']
    assert obs['E_model_ext']['representation_class'] == obs['representation_class']
    assert obs['E_R']['trace'].startswith('risk-module')
    assert obs['E_A']['trace'].startswith('action')


def test_plain_dataset_defaults_to_f0_representation():
    df = pd.DataFrame({
        'age': [30, 44, 52, 61, 73, 39, 48, 57, 69, 35, 42, 66],
        'pressure': [110, 126, 140, 155, 168, 118, 133, 149, 172, 121, 136, 160],
        'marker': [0.10, 0.20, 0.45, 0.71, 0.90, 0.18, 0.38, 0.63, 0.88, 0.22, 0.41, 0.80],
        'target': [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    })
    record = DatasetRecord('plain', 'unit-test', target_column='target')
    result = DatasetObserverPipeline().run(record, df, case_index=0)
    assert result.trace['mode'] == 'user'
    assert result.observer_result['selected_representation'] == 'F0'
    assert result.observer_result['representation_class'] == 'F0'
