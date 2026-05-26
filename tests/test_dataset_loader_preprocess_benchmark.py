from __future__ import annotations

from experiments.dataset_benchmark import run_benchmark
from fuzzyxai.data import list_dataset_cards, load_dataset_by_key, preprocess_dataset


def test_dataset_loader_cards_and_load_smoke():
    cards = list_dataset_cards()
    keys = {c.key for c in cards}
    assert {'breast_cancer', 'rikord', 'ruccod'}.issubset(keys)

    card, record, df = load_dataset_by_key('breast_cancer', allow_fallback=True)
    assert card.key == 'breast_cancer'
    assert len(df) > 0
    assert record.target_column in df.columns


def test_preprocess_dataset_smoke():
    _, record, df = load_dataset_by_key('breast_cancer', allow_fallback=True)
    prep = preprocess_dataset(df, target_column=record.target_column)
    assert prep.n_rows == len(df)
    assert prep.n_features > 0
    assert prep.missing_after <= prep.missing_before


def test_dataset_benchmark_smoke(tmp_path):
    summary = run_benchmark('breast_cancer', out_root=tmp_path)
    assert summary['dataset'] == 'breast_cancer'
    assert summary['status'] == 'READY'
    assert float(summary['model_accuracy']) > 0.8
