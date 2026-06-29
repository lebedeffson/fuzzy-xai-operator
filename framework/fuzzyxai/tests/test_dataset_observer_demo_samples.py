from __future__ import annotations

from examples.dataset_observer_demo import SAMPLE_NAMES
from fuzzyxai.datasets import load_dataset_mode


def test_dataset_observer_demo_builtin_samples_are_binary_and_target_present() -> None:
    assert 'registry_programs' in SAMPLE_NAMES
    for sample in SAMPLE_NAMES:
        try:
            record, df = load_dataset_mode(sample)
        except FileNotFoundError:
            assert sample.startswith('registry_')
            continue
        assert record.target_column in df.columns
        vals = set(df[record.target_column].unique().tolist())
        assert vals.issubset({0, 1})
        if not sample.startswith('registry_'):
            assert len(vals) == 2
