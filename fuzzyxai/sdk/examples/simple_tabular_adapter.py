from __future__ import annotations

from typing import Any

from fuzzyxai.sdk import BaseAdapter


class SimpleTabularAdapter(BaseAdapter):
    registry_id = 'simple_tabular_example'
    adapter_class = 'SimpleTabularAdapter'
    input_types = ['tabular']
    claim_scope = 'example tabular adapter for SDK contract testing'

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate_input(payload)
        if not validation['ok']:
            return {'status': 'error', 'validation': validation}
        values = payload.get('payload', {}).get('features', {})
        return {
            'status': 'ok',
            'terms': ['low', 'medium', 'high'],
            'feature_count': len(values) if isinstance(values, dict) else 0,
            'report_path': 'reports/sdk/simple_tabular_adapter.md',
        }
