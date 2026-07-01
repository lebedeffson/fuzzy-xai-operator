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
    title_ru: str = ""
    operator_type: str = ""
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    input_values: dict[str, Any] = field(default_factory=dict)
    output_values: dict[str, Any] = field(default_factory=dict)
    formula_id: str | None = None
    formula_text: str | None = None
    formula_latex: str | None = None
    components: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    status_reason_ru: str = ""
    interpretation_ru: str = ""
    next_node_ids: list[str] = field(default_factory=list)
    source_commit: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorEdge:
    edge_id: str
    source_node_id: str
    target_node_id: str
    passed_values: dict[str, Any]
    explanation_ru: str

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
    route_id: str = ""
    scenario_title_ru: str = ""
    created_at: str = ""
    edges: list[OperatorEdge] = field(default_factory=list)
    final_diagnostic_id: str = ""
    final_action_id: str = ""
    proof_ref: str = ""
    dashboard_ref: str = ""
    verification_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "scenario_id": self.scenario_id,
            "scenario_title_ru": self.scenario_title_ru,
            "title": self.title,
            "created_at": self.created_at,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "computed_result": self.computed_result,
            "diagnostics": self.diagnostics,
            "final_action": self.final_action,
            "final_action_id": self.final_action_id or self.final_action,
            "final_diagnostic_id": self.final_diagnostic_id or self.computed_result.get("diagnostic_id", ""),
            "verifier_status": self.verifier_status,
            "source_commit": self.source_commit,
            "proof_ref": self.proof_ref,
            "dashboard_ref": self.dashboard_ref,
            "verification_summary": self.verification_summary,
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
    source_commit: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path
