from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .citr_loader import load_citr_dataset
from .dataset_record import DatasetRecord
from .rikord_loader import load_rikord_dataset
from .ruccod_loader import load_ruccod_dataset


@dataclass(frozen=True)
class DatasetCard:
    key: str
    name: str
    source: str
    license: str
    domain: str
    description: str
    loader: Callable[[bool], tuple[DatasetRecord, pd.DataFrame]]


def _breast_cancer_loader(_allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df['risk_target'] = (df['target'] == 0).astype(int)
    df = df.drop(columns=['target'])
    record = DatasetRecord(
        name='sklearn_breast_cancer',
        source='sklearn.datasets',
        target_column='risk_target',
        task_type='binary_classification',
        description='Breast Cancer Wisconsin (Diagnostic).',
        metadata={'license': 'UCI/CC BY 4.0 (as distributed via sklearn)'},
    )
    return record, df


def _citr_mosmed_loader(allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    return load_citr_dataset(mode='registry_mosmed_doctor_analysis', allow_fallback=allow_fallback)


def _citr_steel_loader(allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    # Uses existing registry mode in fuzzyxai.datasets with fallback handled in loader.
    return load_citr_dataset(mode='registry_steel_ir', allow_fallback=allow_fallback)


def _rikord_loader(allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    return load_rikord_dataset(allow_fallback=allow_fallback)


def _ruccod_loader(allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    return load_ruccod_dataset(allow_fallback=allow_fallback)


def _synthetic_fraud_loader(_allow_fallback: bool = True) -> tuple[DatasetRecord, pd.DataFrame]:
    rng = np.random.default_rng(42)
    n = 2000
    amount = rng.gamma(shape=2.0, scale=150.0, size=n)
    velocity = rng.normal(0.0, 1.0, size=n)
    country_risk = rng.uniform(0.0, 1.0, size=n)
    logits = 0.002 * amount + 0.9 * velocity + 1.1 * country_risk - 1.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > 0.55).astype(int)
    df = pd.DataFrame(
        {
            'amount': amount,
            'velocity': velocity,
            'country_risk': country_risk,
            'risk_target': y,
        }
    )
    rec = DatasetRecord(
        name='synthetic_fraud_detection',
        source='fuzzyxai.synthetic',
        target_column='risk_target',
        task_type='binary_classification',
        description='Synthetic fraud dataset for finance-domain stress checks.',
        metadata={'license': 'generated', 'fallback': False},
    )
    return rec, df


DATASET_CATALOG: dict[str, DatasetCard] = {
    'breast_cancer': DatasetCard(
        key='breast_cancer',
        name='Breast Cancer Wisconsin',
        source='sklearn / UCI',
        license='UCI dataset terms (via sklearn)',
        domain='medicine',
        description='Medical baseline for risk-aware observer.',
        loader=_breast_cancer_loader,
    ),
    'rikord': DatasetCard(
        key='rikord',
        name='RIKORD (or fallback proxy)',
        source='public russian medical / local file',
        license='depends on source distribution',
        domain='medicine',
        description='ICU/sepsis-style tabular risk scenario.',
        loader=_rikord_loader,
    ),
    'ruccod': DatasetCard(
        key='ruccod',
        name='RuCCoD (or fallback proxy)',
        source='public russian medical / local file',
        license='depends on source distribution',
        domain='medicine+text',
        description='Clinical coding and text-risk alignment.',
        loader=_ruccod_loader,
    ),
    'citr_mosmed': DatasetCard(
        key='citr_mosmed',
        name='CITR MosMed doctor analysis',
        source='registry.cit.gov.ru',
        license='see dataset card',
        domain='medical_audit',
        description='Doctor-model consistency and auditability.',
        loader=_citr_mosmed_loader,
    ),
    'citr_steel_ir': DatasetCard(
        key='citr_steel_ir',
        name='CITR Steel IR',
        source='registry.cit.gov.ru',
        license='see dataset card',
        domain='industrial',
        description='Industrial CV transfer scenario.',
        loader=_citr_steel_loader,
    ),
    'synthetic_fraud': DatasetCard(
        key='synthetic_fraud',
        name='Synthetic Fraud Detection',
        source='generated',
        license='generated',
        domain='finance',
        description='Finance-like high-risk scenario for policy stress tests.',
        loader=_synthetic_fraud_loader,
    ),
}


def list_dataset_cards() -> list[DatasetCard]:
    return list(DATASET_CATALOG.values())


def get_dataset_card(key: str) -> DatasetCard:
    if key not in DATASET_CATALOG:
        raise KeyError(f'Unknown dataset key: {key}. Available: {", ".join(DATASET_CATALOG)}')
    return DATASET_CATALOG[key]


def load_dataset_by_key(key: str, *, allow_fallback: bool = True) -> tuple[DatasetCard, DatasetRecord, pd.DataFrame]:
    card = get_dataset_card(key)
    record, df = card.loader(allow_fallback)
    return card, record, df

