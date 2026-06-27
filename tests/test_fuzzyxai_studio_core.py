from __future__ import annotations

import json

from apps.services.layered_case import LayeredCaseService, build_case_state
from fuzzyxai.studio import StudioExplainPlan, apply_named_preset, export_defense_case, recompute_case_state


def test_critical_preset_forces_block() -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    plan = StudioExplainPlan()
    tuned = apply_named_preset(base, plan, 'critical_rupture')
    assert tuned['risk']['chi_R_crit'] == 1
    assert tuned['risk']['action'] == 'block'


def test_recompute_with_overrides_changes_action() -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'safe', sample_index=0, dataset_mode='breast_cancer')
    plan = StudioExplainPlan()
    tuned = recompute_case_state(base, plan)
    assert tuned['risk']['action'] in {'accept', 'lower_confidence', 'request_more_data', 'defer_to_human', 'block'}


def test_export_defense_case_contains_required_fields(tmp_path) -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'rupture', sample_index=113, dataset_mode='breast_cancer')
    plan = StudioExplainPlan()
    tuned = recompute_case_state(base, plan)
    paths = export_defense_case(tuned, plan, out_dir=tmp_path, stem='defense_case')

    payload = json.loads((tmp_path / 'defense_case.json').read_text(encoding='utf-8'))
    assert 'observer' in payload
    assert 'representation' in payload
    assert 'topos' in payload
    assert 'action' in payload['observer']
    assert set(paths.keys()) == {'json', 'md', 'tex'}
