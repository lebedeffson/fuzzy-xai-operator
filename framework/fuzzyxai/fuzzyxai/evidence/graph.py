from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import (
    ExplanationEdge,
    ExplanationClaim,
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
    claims: Sequence[ExplanationClaim] = (),
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
            edges.append(ExplanationEdge(node_id, anomaly_id, "derived_from"))

    for trace in evidence.training:
        node_id = f"training:{trace.object_id}"
        add_node(node_id, "training_event", f"Training trace {trace.object_id}", trace.to_dict())
        data_id = f"data:{trace.object_id}"
        if any(node.node_id == data_id for node in nodes):
            edges.append(ExplanationEdge(data_id, node_id, "observed_during"))

    for subgroup in evidence.subgroups:
        node_id = f"subgroup:{subgroup.subgroup_id}"
        add_node(node_id, "training_event", f"Subgroup {subgroup.subgroup_id}", subgroup.to_dict())

    for rule in evidence.rules:
        node_id = f"rule:{rule.rule_id}"
        add_node(node_id, "rule", rule.human_text, rule.to_dict(), rule.evidence_refs)
        for object_id in rule.source_objects:
            data_id = f"data:{object_id}"
            if any(node.node_id == data_id for node in nodes):
                edges.append(ExplanationEdge(data_id, node_id, "derived_from", rule.evidence_refs))

    for concept in evidence.concepts:
        node_id = f"concept:{concept.class_id}"
        add_node(node_id, "concept", concept.human_description, concept.to_dict())
        rule_node_ids = {node.node_id for node in nodes if node.node_type == "rule"}
        for rule_id in concept.primary_rules:
            rule_node = f"rule:{rule_id}"
            if rule_node in rule_node_ids:
                edges.append(ExplanationEdge(rule_node, node_id, "derived_from"))

    for similar in evidence.similar_cases:
        node_id = f"similar:{similar.query_object_id}:{similar.reference_object_id}"
        add_node(
            node_id,
            "similar_case",
            f"{similar.reference_object_id}: {similar.similarity_method}",
            similar.to_dict(),
        )
        data_id = f"data:{similar.query_object_id}"
        if any(node.node_id == data_id for node in nodes):
            edges.append(ExplanationEdge(data_id, node_id, "derived_from"))

    for index, counterfactual in enumerate(evidence.counterfactuals):
        node_id = f"counterfactual:{index}"
        add_node(node_id, "counterfactual", f"Change toward {counterfactual.target_prediction}", counterfactual.to_dict(), counterfactual.evidence_refs)

    for activation in evidence.fuzzy_rule_activations:
        # node_id must match claims.py's `ref = f"fuzzy_rule:{object_id}:{rule_id}"`
        # exactly — that's what a fuzzy_rule claim's evidence_refs point to.
        node_id = f"fuzzy_rule:{activation.object_id}:{activation.rule_id}"
        add_node(node_id, "fuzzy_rule", f"Rule {activation.rule_id} (activation {activation.activation_strength:.2f})", activation.to_dict())
        data_id = f"data:{activation.object_id}"
        if any(node.node_id == data_id for node in nodes):
            edges.append(ExplanationEdge(data_id, node_id, "derived_from"))

    for image in evidence.image_representations:
        for region in image.regions:
            # node_id must match claims.py's `ref = f"image_region:{object_id}:{region.name}"`.
            node_id = f"image_region:{image.object_id}:{region.name}"
            add_node(
                node_id,
                "image_region",
                f"Region {region.name} ({region.pixel_count} px)",
                {"name": region.name, "pixel_count": region.pixel_count, "bounding_box": list(region.bounding_box), "direction": region.direction, "contribution": region.contribution},
            )
            data_id = f"data:{image.object_id}"
            if any(node.node_id == data_id for node in nodes):
                edges.append(ExplanationEdge(data_id, node_id, "derived_from"))

    contributions = prediction.get("contributions", {})
    if isinstance(contributions, Mapping):
        for feature, value in contributions.items():
            add_node(
                f"contribution:{feature}",
                "contribution",
                f"Contribution of {feature}",
                {"feature": str(feature), "value": float(value), "method": prediction.get("contribution_method")},
            )

    add_node("prediction", "prediction", "Model prediction", prediction)
    for node in list(nodes):
        if node.node_type in {"data", "contribution", "rule", "concept", "similar_case", "counterfactual", "fuzzy_rule", "image_region"}:
            relation = "changed_by" if node.node_type == "counterfactual" else "derived_from"
            edges.append(ExplanationEdge(node.node_id, "prediction", relation, node.evidence_refs))

    for index, diagnostic in enumerate(diagnostics):
        node_id = f"diagnostic:{index}"
        add_node(node_id, "diagnostic", str(diagnostic.get("reason", diagnostic.get("code", "diagnostic"))), diagnostic)
        edges.append(ExplanationEdge("prediction", node_id, "checked_by"))

    if evidence.missing:
        add_node(
            "trace:missing_evidence",
            "trace",
            "Unavailable evidence channels",
            {"missing": list(evidence.missing)},
        )

    add_node("action", "action", action, {"action": action})
    node_ids = {node.node_id for node in nodes}
    for claim in claims:
        claim_node = f"claim:{claim.claim_id}"
        add_node(claim_node, "claim", claim.short_statement, claim.to_dict(), claim.evidence_refs)
        for ref in claim.evidence_refs:
            if ref in node_ids:
                edges.append(ExplanationEdge(ref, claim_node, "supports_claim", [ref]))
        for ref in claim.counter_evidence_refs:
            if ref in node_ids:
                edges.append(ExplanationEdge(ref, claim_node, "contradicts_claim", [ref]))
        if claim.claim_type == "diagnostic":
            edges.append(ExplanationEdge(claim_node, "action", "constrains"))

    edges.append(ExplanationEdge("prediction", "action", "recommends"))
    for index in range(len(diagnostics)):
        edges.append(ExplanationEdge(f"diagnostic:{index}", "action", "constrains"))
    for claim in claims:
        if claim.claim_type == "recommended_action":
            edges.append(ExplanationEdge(f"claim:{claim.claim_id}", "action", "recommends"))
    return ExplanationGraph(nodes=nodes, edges=edges, claims=list(claims), missing_evidence=list(evidence.missing))
