from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperatorNodeState:
    """Visible state of one FuzzyXAI operator node."""

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
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorRouteState:
    """Single source of truth for a rendered operator route."""

    scenario_id: str
    title: str
    nodes: list[OperatorNodeState]
    proof_ref: str
    verifier_status: str
    final_action: str
    source_commit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "proof_ref": self.proof_ref,
            "verifier_status": self.verifier_status,
            "final_action": self.final_action,
            "source_commit": self.source_commit,
            "nodes": [node.to_dict() for node in self.nodes],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

