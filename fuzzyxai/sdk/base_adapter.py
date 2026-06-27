from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .contracts import AdapterMetadata, ExplanationArtifact, ValidationResult


class BaseAdapter(ABC):
    registry_id: str = 'base_adapter'
    adapter_class: str = 'BaseAdapter'
    input_types: list[str] = ['generic']
    claim_scope: str = 'adapter contract only'

    def metadata(self) -> dict[str, Any]:
        return AdapterMetadata(
            registry_id=self.registry_id,
            adapter_class=self.adapter_class,
            input_types=list(self.input_types),
            claim_scope=self.claim_scope,
        ).to_dict()

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        errors: list[str] = []
        if not isinstance(payload, dict):
            errors.append('payload must be a dictionary')
        elif 'payload' not in payload:
            errors.append('missing payload field')
        result = ValidationResult(ok=not errors, status='ok' if not errors else 'error', errors=errors)
        return result.to_dict()

    @abstractmethod
    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def to_explanation_artifact(self, raw_result: dict[str, Any]) -> ExplanationArtifact:
        return ExplanationArtifact(
            registry_id=self.registry_id,
            artifact_type='ExplanationArtifact',
            has_explanation_object=True,
            has_diagnostic_state=bool(raw_result.get('diagnostic_state', False)),
            payload=dict(raw_result),
            trace={
                'adapter': self.adapter_class,
                'registry_id': self.registry_id,
                'claim_scope': self.claim_scope,
            },
            report_path=str(raw_result.get('report_path', '')),
        )
