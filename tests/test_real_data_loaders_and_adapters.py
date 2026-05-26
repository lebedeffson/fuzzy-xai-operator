from __future__ import annotations

import pandas as pd

from fuzzyxai.adapters import (
    MedicalImageToExplanationAdapter,
    TabularToExplanationAdapter,
    TextToExplanationAdapter,
)
from fuzzyxai.data import load_citr_dataset, load_rikord_dataset, load_ruccod_dataset


def test_real_data_loaders_fallback_smoke():
    rec_citr, df_citr = load_citr_dataset(allow_fallback=True)
    assert len(df_citr) > 0
    assert rec_citr.target_column is not None

    rec_rikord, df_rikord = load_rikord_dataset(allow_fallback=True)
    assert len(df_rikord) > 0
    assert rec_rikord.target_column is not None

    rec_ruccod, df_ruccod = load_ruccod_dataset(allow_fallback=True)
    assert len(df_ruccod) > 0
    assert rec_ruccod.target_column is not None


def test_adapters_build_explanation_objects():
    tab = TabularToExplanationAdapter().fit(
        pd.DataFrame({'a': [1.0, 2.0], 'b': [0.1, 0.2], 'risk_target': [0, 1]}),
        target_column='risk_target',
    )
    e_tab = tab.adapt(
        pd.Series({'a': 1.3, 'b': 0.15}),
        sample_id='s1',
        predicted_risk=0.6,
        model_version='v1',
        source='tab',
    )
    assert 0.0 <= e_tab.uncertainty <= 1.0

    img = MedicalImageToExplanationAdapter()
    e_img = img.adapt(sample_id='s2', predicted_risk=0.4, metadata={'modality': 'XR'})
    assert 0.0 <= e_img.uncertainty <= 1.0

    txt = TextToExplanationAdapter()
    e_txt = txt.adapt(sample_id='s3', text='Acute sepsis but stable now', icd_codes=['A41.9'], predicted_risk=0.7)
    assert 0.0 <= e_txt.uncertainty <= 1.0
