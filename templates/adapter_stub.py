from __future__ import annotations

from typing import Any

from fuzzyxai.sdk import BaseAdapter


class MODULE_CLASS_NAME(BaseAdapter):
    registry_id = 'MODULE_ID'
    adapter_class = 'MODULE_CLASS_NAME'
    input_types = ['generic']
    claim_scope = 'adapter route only, not model validation'

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate_input(payload)
        if not validation['ok']:
            return {'status': 'error', 'validation': validation}
        return {'status': 'ok', 'report_path': f'reports/sdk/{self.registry_id}.md'}
