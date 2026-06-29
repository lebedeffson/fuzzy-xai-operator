from __future__ import annotations

import json

from apps.services.layered_case import LayeredCaseService, build_case_state
from fuzzyxai.studio import StudioExplainPlan, export_defense_case, recompute_case_state


def test_defense_export_has_required_fields(tmp_path) -> None:
    service = LayeredCaseService.create()
    base = build_case_state(service, 'rupture', sample_index=113, dataset_mode='breast_cancer')
    state = recompute_case_state(base, StudioExplainPlan())
    paths = export_defense_case(state, StudioExplainPlan(), out_dir=tmp_path, stem='case')

    payload = json.loads((tmp_path / 'case.json').read_text(encoding='utf-8'))
    assert 'dataset' in payload
    assert 'model' in payload
    assert 'explanation' in payload
    assert 'representation' in payload
    assert 'topos' in payload
    assert 'observer' in payload
    assert 'action' in payload['observer']
    assert paths['json'].endswith('case.json')
