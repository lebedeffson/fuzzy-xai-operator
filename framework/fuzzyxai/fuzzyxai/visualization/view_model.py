from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class ExplanationViewModel:
    """Backend-neutral contract shared by Python, web, and MATLAB views."""

    model: Mapping[str, Any]
    fuzzy: Mapping[str, Any]
    route: Sequence[Mapping[str, Any]]
    disagreement: Mapping[str, Any]
    risk: Mapping[str, Any]
    diagnostics: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    claims: Sequence[Mapping[str, Any]] | Mapping[str, Any] = field(default_factory=tuple)
    narrative: str = ""
    trace: Mapping[str, Any] = field(default_factory=dict)
    layers: Mapping[str, Any] = field(default_factory=dict)
    explanation_graph: Mapping[str, Any] = field(default_factory=dict)
    human_explanations: Mapping[str, Any] = field(default_factory=dict)
    quality_metrics: Mapping[str, float | None] = field(default_factory=dict)
    explanation_level: Mapping[str, Any] = field(default_factory=dict)
    visual_spec: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def export_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.to_json() + "\n", encoding="utf-8")
        return output

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExplanationViewModel":
        return cls(
            model=dict(payload.get("model", {})),
            fuzzy=dict(payload.get("fuzzy", {})),
            route=list(payload.get("route", [])),
            disagreement=dict(payload.get("disagreement", {})),
            risk=dict(payload.get("risk", {})),
            diagnostics=list(payload.get("diagnostics", [])),
            claims=(
                list(payload.get("claims", []))
                if isinstance(payload.get("claims", []), list)
                else dict(payload.get("claims", {}))
            ),
            narrative=str(payload.get("narrative", "")),
            trace=dict(payload.get("trace", {})),
            layers=dict(payload.get("layers", {})),
            explanation_graph=dict(payload.get("explanation_graph", {})),
            human_explanations=dict(payload.get("human_explanations", {})),
            quality_metrics=dict(payload.get("quality_metrics", {})),
            explanation_level=dict(payload.get("explanation_level", {})),
            visual_spec=dict(payload.get("visual_spec", {})),
            schema_version=str(payload.get("schema_version", "2.0")),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "ExplanationViewModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
