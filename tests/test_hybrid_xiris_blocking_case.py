from __future__ import annotations

import json

from experiments.chapter5_hybrid_xiris_blocking_case import JSON_PATH, run


def test_hybrid_xiris_blocking_case_blocks() -> None:
    result = run()
    assert result['status'] == 'ok'
    row = json.loads(JSON_PATH.read_text(encoding='utf-8'))
    required = {
        'sample_id', 'image_quality_score', 'segmentation_quality_score', 'model_match_score',
        'rule_model_accept_activation', 'rule_quality_block_activation', 'source_conflict',
        'chi_R_crit', 'chi_Auto', 'rho', 'action', 'report_path', 'explain_plan_hash', 'fixture_sha256'
    }
    assert required <= set(row)
    assert row['chi_R_crit'] == 1
    assert row['chi_Auto'] is False
    assert row['action'] in {'block', 'audit_block'}
    assert row['fixture_sha256']
