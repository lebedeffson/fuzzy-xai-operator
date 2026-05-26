from __future__ import annotations

from fuzzyxai.datasets import DATASET_REGISTRY, list_dataset_modes, load_dataset_mode


def test_dataset_registry_modes_load_binary_targets() -> None:
    modes = list_dataset_modes()
    assert len(modes) >= 3
    assert set(DATASET_REGISTRY).issuperset(
        {
            'breast_cancer',
            'diabetes_binary',
            'synthetic_ruptures',
            'registry_programs',
            'registry_mosmed_doctor_analysis',
            'registry_steel_ir',
        }
    )

    for key in ['breast_cancer', 'diabetes_binary', 'wine_risk', 'synthetic_ruptures']:
        record, df = load_dataset_mode(key)
        assert record.target_column in df.columns
        values = set(df[record.target_column].unique().tolist())
        assert values.issubset({0, 1})
        assert len(values) == 2
