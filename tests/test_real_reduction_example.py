from __future__ import annotations

from experiments.real_reduction_example import generate_real_reduction_example


def test_real_reduction_example_generates_outputs(tmp_path) -> None:
    result = generate_real_reduction_example(out_dir=tmp_path)
    assert 'selected_class' in result
    assert result['selected_class'] in {'F0', 'F_int', 'F_H', 'F_N_src'}
    assert (tmp_path / 'breast_cancer_case.json').exists()
    assert (tmp_path / 'breast_cancer_case.md').exists()
