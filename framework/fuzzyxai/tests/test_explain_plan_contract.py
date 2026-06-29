from __future__ import annotations

import json

import pytest

from fuzzyxai.core.explain_plan import (
    canonicalize_explain_plan,
    hash_explain_plan,
    load_explain_plan,
    validate_explain_plan,
)
from fuzzyxai.experiments.chapter2_sample113 import write_plan_hash


def test_chapter2_explain_plan_yaml_validates() -> None:
    plan = load_explain_plan('configs/explain_plan_chapter2.yaml')
    validate_explain_plan(plan)
    assert plan['name'] == 'chapter2_sample113_plan'
    assert plan['trace_required'] == ['id', 'version', 'time', 'params', 'source', 'hash']


def test_explain_plan_hash_is_deterministic() -> None:
    plan = load_explain_plan('configs/explain_plan_chapter2.yaml')
    first = hash_explain_plan(plan)
    second = hash_explain_plan(json.loads(json.dumps(plan, ensure_ascii=False)))
    assert first == second
    assert len(first) == 64
    assert canonicalize_explain_plan(plan) == canonicalize_explain_plan(plan)


def test_invalid_explain_plan_rejects_bad_thresholds() -> None:
    plan = load_explain_plan('configs/explain_plan_chapter2.yaml')
    plan['risk_observer']['thresholds'] = [0.2, 0.1, 0.5, 0.8]
    with pytest.raises(ValueError, match='thresholds'):
        validate_explain_plan(plan)


def test_write_plan_hash_artifact(tmp_path) -> None:
    payload = write_plan_hash(out_dir=tmp_path)
    path = tmp_path / 'explain_plan_hash.json'
    assert path.exists()
    loaded = json.loads(path.read_text(encoding='utf-8'))
    assert loaded['sha256'] == payload['sha256']
    assert loaded['validated'] is True
    assert 'hash' in loaded['required_trace_fields']
