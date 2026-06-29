from __future__ import annotations

from experiments.thesis_practice_tables import generate


def test_thesis_practice_tables_generation(tmp_path) -> None:
    payload = generate(out_dir=tmp_path)
    assert payload['dataset'] == 'breast_cancer'
    assert 'quantiles' in payload
    assert len(payload['end_to_end_cases']) == 3
    assert len(payload['ablations']) >= 6
    assert (tmp_path / 'thesis_practice_tables.json').exists()
    assert (tmp_path / 'thesis_practice_tables.md').exists()

