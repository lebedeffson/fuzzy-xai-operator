from __future__ import annotations

from experiments.baseline_comparison import run


def test_equal_guardrail_zero_missed_critical_for_all_rows(tmp_path) -> None:
    out = run('synthetic_ruptures', out_root=tmp_path, baseline_access='equal_guardrail')
    assert out['status'] == 'READY'
    assert out['baseline_access'] == 'equal_guardrail'
    for row in out['rows']:
        assert int(row['missed_critical_ruptures']) == 0
