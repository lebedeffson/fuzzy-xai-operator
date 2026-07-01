from __future__ import annotations

from typing import Any

from fuzzyxai.sdk import BaseAdapter


class MedicalImageAdapter(BaseAdapter):
    registry_id = 'medical_image_example'
    adapter_class = 'MedicalImageAdapter'
    input_types = ['image_biometric', 'medical_image']
    claim_scope = 'image artifact routing only, not clinical validation'

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        validation = self.validate_input(payload)
        if not validation['ok']:
            return {'status': 'error', 'validation': validation}
        body = payload.get('payload', {})
        return {
            'status': 'ok',
            'image_id': body.get('image_id', 'unknown'),
            'artifact_channels': ['L_k', 'mu_k', 'tau_k', 'Report'],
            'diagnostic_state': bool(body.get('source_conflict', False)),
            'report_path': 'reports/sdk/medical_image_adapter.md',
        }
