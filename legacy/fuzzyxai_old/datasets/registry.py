from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from fuzzyxai.data.dataset_record import DatasetRecord
from .registry_mosmed_doctor_analysis import load_registry_mosmed_doctor_analysis
from .registry_programs import load_registry_programs
from .registry_steel_ir import load_registry_steel_ir


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    domain: str
    purpose: str
    loader: Callable[[], tuple[DatasetRecord, pd.DataFrame]]


def _breast_cancer() -> tuple[DatasetRecord, pd.DataFrame]:
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df['risk_target'] = (df['target'] == 0).astype(int)
    df = df.drop(columns=['target'])
    record = DatasetRecord(
        name='sklearn_breast_cancer',
        source='sklearn.datasets',
        target_column='risk_target',
        task_type='binary_classification',
        description='Medical diagnosis risk; baseline reference dataset.',
        metadata={'dataset_mode': 'breast_cancer'},
    )
    return record, df


def _diabetes_binary() -> tuple[DatasetRecord, pd.DataFrame]:
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    threshold = float(df['target'].median())
    df['risk_target'] = (df['target'] >= threshold).astype(int)
    df = df.drop(columns=['target'])
    record = DatasetRecord(
        name='sklearn_diabetes_binary',
        source='sklearn.datasets',
        target_column='risk_target',
        task_type='binary_classification',
        description='Medical uncertainty / borderline risk cases.',
        metadata={'dataset_mode': 'diabetes_binary', 'binarization': f'target>=median({threshold:.4f})'},
    )
    return record, df


def _wine_risk() -> tuple[DatasetRecord, pd.DataFrame]:
    data = load_wine(as_frame=True)
    df = data.frame.copy()
    df['risk_target'] = (df['target'] == 2).astype(int)
    df = df.drop(columns=['target'])
    record = DatasetRecord(
        name='sklearn_wine_risk',
        source='sklearn.datasets',
        target_column='risk_target',
        task_type='binary_classification',
        description='Non-medical proxy scenario for cross-domain robustness.',
        metadata={'dataset_mode': 'wine_risk', 'high_risk_class': 2},
    )
    return record, df


def _synthetic_ruptures(n: int = 900) -> tuple[DatasetRecord, pd.DataFrame]:
    rng = np.random.default_rng(42)
    f1 = rng.normal(0.0, 1.0, size=n)
    f2 = rng.normal(0.0, 1.2, size=n)
    f3 = rng.uniform(-1.0, 1.0, size=n)
    logits = 1.2 * f1 - 0.8 * f2 + 0.5 * f3
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.55).astype(int)

    df = pd.DataFrame({
        'feature_a': f1,
        'feature_b': f2,
        'feature_c': f3,
        'risk_target': y,
    })
    # Metadata columns intentionally create disagreement patterns for profile inference.
    df['expert_a'] = df['risk_target']
    df['expert_b'] = np.where(np.arange(n) % 6 == 0, 1 - df['risk_target'], df['risk_target'])
    df['source_model'] = df['risk_target']
    df['source_expert'] = np.where(np.arange(n) % 5 == 0, 1 - df['risk_target'], df['risk_target'])
    idx = np.arange(n)
    is_rule_conflict = (idx % 6 == 0)
    is_source_conflict = (idx % 5 == 0)
    is_trace_gap = (idx % 10 == 0)
    is_context_forbidden = (idx % 9 == 0)
    chi_r = is_rule_conflict | is_source_conflict | is_trace_gap | is_context_forbidden
    chi_r_crit = is_source_conflict | is_context_forbidden
    rupture_type = np.where(
        is_context_forbidden,
        'context_forbidden',
        np.where(
            is_source_conflict,
            'source_conflict',
            np.where(is_rule_conflict, 'rule_conflict', np.where(is_trace_gap, 'trace_gap', 'none')),
        ),
    )
    expected_action = np.where(
        chi_r_crit,
        'block',
        np.where(chi_r, 'request_more_data', np.where(probs < 0.30, 'accept', 'lower_confidence')),
    )
    df['rupture'] = chi_r.astype(int)
    df['critical_rupture'] = chi_r_crit.astype(int)
    df['chi_R'] = chi_r.astype(int)
    df['chi_R_crit'] = chi_r_crit.astype(int)
    df['rupture_type'] = rupture_type
    df['expected_action'] = expected_action

    record = DatasetRecord(
        name='synthetic_ruptures',
        source='fuzzyxai.synthetic',
        target_column='risk_target',
        task_type='binary_classification',
        description='Controlled diagnostic rupture scenarios for category/HoTT checks.',
        metadata={'dataset_mode': 'synthetic_ruptures'},
    )
    return record, df


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    'breast_cancer': DatasetSpec(
        key='breast_cancer',
        title='Breast Cancer',
        domain='medical',
        purpose='Medical risk baseline: probability + observer action policy.',
        loader=_breast_cancer,
    ),
    'diabetes_binary': DatasetSpec(
        key='diabetes_binary',
        title='Diabetes',
        domain='medical',
        purpose='Borderline medical uncertainty: lower_confidence/request_more_data behavior.',
        loader=_diabetes_binary,
    ),
    'wine_risk': DatasetSpec(
        key='wine_risk',
        title='Wine Risk Proxy',
        domain='general',
        purpose='Cross-domain robustness check beyond one medical dataset.',
        loader=_wine_risk,
    ),
    'synthetic_ruptures': DatasetSpec(
        key='synthetic_ruptures',
        title='Synthetic Ruptures',
        domain='controlled',
        purpose='Controlled rupture generation for Expl/HoTT and safe-action diagnostics.',
        loader=_synthetic_ruptures,
    ),
    'registry_programs': DatasetSpec(
        key='registry_programs',
        title='Registry: Programs for EVM',
        domain='tabular_text',
        purpose='Text+tabular explainability/ranking scenario from CIT registry.',
        loader=load_registry_programs,
    ),
    'registry_mosmed_doctor_analysis': DatasetSpec(
        key='registry_mosmed_doctor_analysis',
        title='Registry: MosMed Doctor Analysis',
        domain='medical_audit',
        purpose='Medical audit scenario: doctor/model decision alignment.',
        loader=load_registry_mosmed_doctor_analysis,
    ),
    'registry_steel_ir': DatasetSpec(
        key='registry_steel_ir',
        title='Registry: Steel IR',
        domain='industrial_cv',
        purpose='Industrial control scenario: transfer beyond medical domain.',
        loader=load_registry_steel_ir,
    ),
}


def list_dataset_modes() -> list[DatasetSpec]:
    return list(DATASET_REGISTRY.values())


def get_dataset_mode(key: str) -> DatasetSpec:
    if key not in DATASET_REGISTRY:
        raise KeyError(f'Unknown dataset mode: {key}. Available: {", ".join(DATASET_REGISTRY)}')
    return DATASET_REGISTRY[key]


def load_dataset_mode(key: str) -> tuple[DatasetRecord, pd.DataFrame]:
    return get_dataset_mode(key).loader()
