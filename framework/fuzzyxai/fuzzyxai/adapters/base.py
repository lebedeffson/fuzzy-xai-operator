from __future__ import annotations

from typing import Any

from fuzzyxai.core.errors import AdapterValidationError
from fuzzyxai.core.types import AdaptedInput


class BaseAdapter:
    repo_id = ""
    scenario_id = ""
    required_fields: tuple[str, ...] = ()

    def validate(self, payload: dict[str, Any]) -> None:
        missing = [field for field in self.required_fields if field not in payload]
        if missing:
            raise AdapterValidationError(f"{self.scenario_id}: missing fields: {', '.join(missing)}")

    def to_adapted_input(self, payload: dict[str, Any]) -> AdaptedInput:
        self.validate(payload)
        return AdaptedInput(
            scenario_id=self.scenario_id,
            values=dict(payload),
            source=self.repo_id or "adapter",
            value_sources={key: "external_model_payload" for key in payload},
        )
