from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdaptedInput:
    scenario_id: str
    values: dict[str, Any]
    source: str = "adapter"
    value_sources: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplainableObject:
    scenario_id: str
    adapted_input: AdaptedInput
    components: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorNode:
    node_id: str
    title: str
    input_summary: str
    output_summary: str
    value: str
    threshold: str = ""
    status: str = "info"
    explanation: str = ""
    formula_ref: str = ""
    trace_ref: str = ""
    value_source: str = "computed"
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorRoute:
    scenario_id: str
    title: str
    nodes: list[OperatorNode]
    computed_result: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    final_action: str
    verifier_status: str = "UNVERIFIED"
    source_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "nodes": [node.to_dict() for node in self.nodes],
            "computed_result": self.computed_result,
            "diagnostics": self.diagnostics,
            "final_action": self.final_action,
            "verifier_status": self.verifier_status,
            "source_commit": self.source_commit,
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


@dataclass(frozen=True)
class ProofTrace:
    package_type: str
    schema_version: str
    scenario_id: str
    route: dict[str, Any]
    computed_result: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    final_action: str
    verifier_status: str = "UNVERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
