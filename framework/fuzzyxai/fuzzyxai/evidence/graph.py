from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import (
    ExplanationEdge,
    ExplanationEvidence,
    ExplanationGraph,
    ExplanationNode,
)


def build_explanation_graph(
    evidence: ExplanationEvidence,
    *,
    prediction: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    action: str,
) -> ExplanationGraph:
    """Compose evidence into a traceable directed graph without new metrics."""

    nodes: list[ExplanationNode] = []
    edges: list[ExplanationEdge] = []

    def add_node(node_id: str, node_type: str, label: str, payload: Mapping[str, Any], refs: Sequence[str] = ()) -> None:
        nodes.append(ExplanationNode(node_id=node_id, node_type=node_type, label=label, payload=dict(payload), evidence_refs=list(refs)))

    for item in evidence.data:
        node_id = f"data:{item.object_id}"
        add_node(node_id, "data", f"Data {item.object_id}", item.to_dict(), item.evidence_refs)
        for feature in item.anomaly_labels:
            anomaly_id = f"anomaly:{item.object_id}:{feature}"
            add_node(anomaly_id, "anomaly", f"Deviation in {feature}", {"feature": feature, "score": item.outlier_scores.get(feature)})
            edges.append(ExplanationEdge(anomaly_id, node_id, "derived_from"))

    for trace in evidence.training:
        node_id = f"training:{trace.object_id}"
        add_node(node_id, "training_event", f"Training trace {trace.object_id}", trace.to_dict())
        data_id = f"data:{trace.object_id}"
        if any(node.node_id == data_id for node in nodes):
            edges.append(ExplanationEdge(node_id, data_id, "derived_from"))

    for subgroup in evidence.subgroups:
        node_id = f"subgroup:{subgroup.subgroup_id}"
        add_node(node_id, "training_event", f"Subgroup {subgroup.subgroup_id}", subgroup.to_dict())

    for rule in evidence.rules:
        node_id = f"rule:{rule.rule_id}"
        add_node(node_id, "rule", rule.human_text, rule.to_dict(), rule.evidence_refs)
        for object_id in rule.source_objects:
            data_id = f"data:{object_id}"
            if any(node.node_id == data_id for node in nodes):
                edges.append(ExplanationEdge(node_id, data_id, "derived_from", rule.evidence_refs))

    for concept in evidence.concepts:
        node_id = f"concept:{concept.class_id}"
        add_node(node_id, "concept", concept.human_description, concept.to_dict())
        for rule_id in concept.primary_rules:
            edges.append(ExplanationEdge(node_id, f"rule:{rule_id}", "supported_by"))

    for similar in evidence.similar_cases:
        node_id = f"similar:{similar.query_object_id}:{similar.reference_object_id}"
        add_node(
            node_id,
            "similar_case",
            f"{similar.reference_object_id}: {similar.similarity_method}",
            similar.to_dict(),
        )
        edges.append(ExplanationEdge(node_id, f"data:{similar.query_object_id}", "similar_to"))

    for index, counterfactual in enumerate(evidence.counterfactuals):
        node_id = f"counterfactual:{index}"
        add_node(node_id, "counterfactual", f"Change toward {counterfactual.target_prediction}", counterfactual.to_dict(), counterfactual.evidence_refs)

    add_node("prediction", "prediction", "Model prediction", prediction)
    for node in list(nodes):
        if node.node_type in {"data", "rule", "concept", "similar_case", "counterfactual"}:
            relation = "changed_by" if node.node_type == "counterfactual" else "supported_by"
            edges.append(ExplanationEdge("prediction", node.node_id, relation, node.evidence_refs))

    for index, diagnostic in enumerate(diagnostics):
        node_id = f"diagnostic:{index}"
        add_node(node_id, "diagnostic", str(diagnostic.get("reason", diagnostic.get("code", "diagnostic"))), diagnostic)
        edges.append(ExplanationEdge(node_id, "prediction", "derived_from"))

    add_node("action", "action", action, {"action": action})
    edges.append(ExplanationEdge("action", "prediction", "derived_from"))
    for index in range(len(diagnostics)):
        edges.append(ExplanationEdge("action", f"diagnostic:{index}", "validated_by"))
    return ExplanationGraph(nodes=nodes, edges=edges, missing_evidence=list(evidence.missing))
