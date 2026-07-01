from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fuzzyxai.core.errors import AdapterValidationError
from fuzzyxai.core.types import AdaptedInput


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BaseAdapter:
    repo_id = ""
    scenario_id = ""
    adapter_id = ""
    task_type = ""
    supported_payload_schema = ""
    required_fields: tuple[str, ...] = ()

    def validate(self, payload: dict[str, Any]) -> None:
        missing = [field for field in self.required_fields if field not in payload]
        if missing:
            raise AdapterValidationError(f"{self.scenario_id}: missing fields: {', '.join(missing)}")

    def validate_payload(self, payload: dict[str, Any]) -> ValidationResult:
        try:
            self.validate(payload)
        except Exception as exc:
            return ValidationResult(valid=False, errors=[str(exc)])
        return ValidationResult(valid=True)

    def describe(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id or self.__class__.__name__,
            "repo_id": self.repo_id,
            "scenario_id": self.scenario_id,
            "task_type": self.task_type,
            "supported_payload_schema": self.supported_payload_schema,
            "required_fields": list(self.required_fields),
        }

    def to_adapted_input(self, payload: dict[str, Any]) -> AdaptedInput:
        self.validate(payload)
        return AdaptedInput(
            scenario_id=self.scenario_id,
            values=dict(payload),
            source=self.repo_id or "adapter",
            value_sources={key: "external_model_payload" for key in payload},
        )
