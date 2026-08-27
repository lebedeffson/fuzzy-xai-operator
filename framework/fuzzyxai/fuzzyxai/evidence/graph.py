from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    ExplanationClaim,
    ExplanationEdge,
    ExplanationEvidence,
    ExplanationGraph,
    ExplanationNode,
)

# P18 item 10: provenance node labels are display strings, read directly by
# the provenance renderer without further translation — an English label
# here (a raw diagnostic ``reason``, a raw action code) leaks straight into
# an otherwise-Russian picture. Keyed by diagnostic ``code`` since the free-
# text ``reason`` may itself contain untranslated identifiers.
_ACTION_LABELS_RU = {
    "accept": "принять",
    "review": "проверить специалистом",
    "audit": "провести аудит",
    "audit_report": "провести аудит",
    "block": "не применять",
    "lower_confidence": "снизить доверие к результату",
    "request_more_data": "запросить дополнительные данные",
    "defer_to_human": "передать специалисту",
    "insufficient_evidence": "собрать недостающие данные",
}

_DIAGNOSTIC_LABELS_RU = {
    "D_ij_alignment": "Согласование Γ превышает допустимую границу",
    "D_k_alignment_missing": "Согласование Γ ожидалось планом, но не измерено",
    "D_k_alignment_not_applicable": "Согласование Γ не предусмотрено планом для этого случая",
    "D_reduction": "Потеря Δ при редукции превышает допустимую границу",
    "D_k_reduction_missing": "Редукция Δ ожидалась планом, но не измерена",
    "D_k_reduction_not_applicable": "Редукция Δ не предусмотрена планом для этого случая",
    "D_risk_critical": "Критический разрыв риска запрещает автоматическое решение",
    "D_risk_incomplete_interface": "Интерфейс риска ρ неполон: часть ожидаемых слагаемых не измерена",
    "D_k_risk_missing": "Оценка риска не была предоставлена",
    "D_surrogate_fidelity": "Суррогатное локальное объяснение имеет низкую или неизмеренную точность",
}


def _diagnostic_label_ru(diagnostic: Mapping[str, Any]) -> str:
    code = str(diagnostic.get("code", ""))
    return _DIAGNOSTIC_LABELS_RU.get(code, str(diagnostic.get("reason", diagnostic.get("code", "диагностика"))))


def build_explanation_graph(
    evidence: ExplanationEvidence,
    *,
    prediction: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    action: str,
    claims: Sequence[ExplanationClaim] = (),
    provenance: Mapping[str, Any] | None = None,
    system_evidence: Mapping[str, Any] | None = None,
) -> ExplanationGraph:
    """Compose evidence into a traceable directed graph without new metrics.

    ``provenance`` (P15.14) optionally carries ``dataset_version``,
    ``model_fingerprint``, ``model_type``, and ``adapter_id`` — when
    supplied, explicit ``dataset``/``model_artifact`` lifecycle nodes are
    added upstream of the per-object data node and downstream of the
    prediction, using only identifiers the runtime already tracks (no split/
    checkpoint nodes yet — those require tracking infrastructure this
    framework does not have, so they are not fabricated here).
    """

    nodes: list[ExplanationNode] = []
    edges: list[ExplanationEdge] = []
    lifecycle = dict(provenance or {})

    def add_node(node_id: str, node_type: str, label: str, payload: Mapping[str, Any], refs: Sequence[str] = ()) -> None:
        nodes.append(ExplanationNode(node_id=node_id, node_type=node_type, label=label, payload=dict(payload), evidence_refs=list(refs)))

    dataset_version = lifecycle.get("dataset_version")
    if dataset_version is not None:
        add_node("dataset:root", "dataset", f"Датасет {dataset_version}", {"dataset_version": dataset_version})
    model_fingerprint = lifecycle.get("model_fingerprint")
    if model_fingerprint:
        add_node(
            "model_artifact:root",
            "model_artifact",
            f"Модель ({lifecycle.get('model_type', 'unknown')})",
            {"model_fingerprint": model_fingerprint, "model_type": lifecycle.get("model_type"), "adapter_id": lifecycle.get("adapter_id")},
        )
    training_run = lifecycle.get("training_run")
    if isinstance(training_run, Mapping) and training_run.get("training_run_id"):
        add_node("training_run:root", "training_run", f"Обучение {training_run['training_run_id']}", training_run)
        if dataset_version is not None:
            edges.append(ExplanationEdge("dataset:root", "training_run:root", "derived_from"))
        if model_fingerprint:
            edges.append(ExplanationEdge("training_run:root", "model_artifact:root", "produces"))
    split = lifecycle.get("split")
    if split is not None:
        add_node("split:root", "split", "Разбиение данных", {"split": split})
        if dataset_version is not None:
            edges.append(ExplanationEdge("dataset:root", "split:root", "derived_from"))
        if isinstance(training_run, Mapping) and training_run.get("training_run_id"):
            edges.append(ExplanationEdge("split:root", "training_run:root", "derived_from"))

    for item in evidence.data:
        node_id = f"data:{item.object_id}"
        add_node(node_id, "data", f"Данные объекта {item.object_id}", item.to_dict(), item.evidence_refs)
        if dataset_version is not None:
            edges.append(ExplanationEdge("dataset:root", node_id, "derived_from"))
        for feature in item.anomaly_labels:
            anomaly_id = f"anomaly:{item.object_id}:{feature}"
            add_node(anomaly_id, "anomaly", f"Отклонение по признаку {feature}", {"feature": feature, "score": item.outlier_scores.get(feature)})
            edges.append(ExplanationEdge(node_id, anomaly_id, "derived_from"))

    for trace in evidence.training:
        node_id = f"training:{trace.object_id}"
        add_node(node_id, "training_event", f"След обучения {trace.object_id}", trace.to_dict())
        data_id = f"data:{trace.object_id}"
        if any(node.node_id == data_id for node in nodes):
            edges.append(ExplanationEdge(data_id, node_id, "observed_during"))
        if any(node.node_id == "training_run:root" for node in nodes):
            edges.append(ExplanationEdge("training_run:root", node_id, "summarized_by"))

    for subgroup in evidence.subgroups:
        node_id = f"subgroup:{subgroup.subgroup_id}"
        add_node(node_id, "training_event", f"Подгруппа {subgroup.subgroup_id}", subgroup.to_dict())

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
        add_node(node_id, "counterfactual", f"Изменение в сторону «{counterfactual.target_prediction}»", counterfactual.to_dict(), counterfactual.evidence_refs)

    for activation in evidence.fuzzy_rule_activations:
        # node_id must match claims.py's `ref = f"fuzzy_rule:{object_id}:{rule_id}"`
        # exactly — that's what a fuzzy_rule claim's evidence_refs point to.
        node_id = f"fuzzy_rule:{activation.object_id}:{activation.rule_id}"
        add_node(node_id, "fuzzy_rule", f"Правило {activation.rule_id} (активация {activation.activation_strength:.2f})", activation.to_dict())
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
                f"Область {region.name} ({region.pixel_count} px)",
                {"name": region.name, "pixel_count": region.pixel_count, "bounding_box": list(region.bounding_box), "direction": region.direction, "contribution": region.contribution},
            )
            data_id = f"data:{image.object_id}"
            if any(node.node_id == data_id for node in nodes):
                edges.append(ExplanationEdge(data_id, node_id, "derived_from"))

    for attribution in evidence.attribution_maps:
        # node_id must match claims.py's `ref = f"attribution_map:{object_id}"`.
        node_id = f"attribution_map:{attribution.object_id}"
        add_node(node_id, "attribution_map", f"Карта атрибуции ({attribution.method})", attribution.to_dict())
        data_id = f"data:{attribution.object_id}"
        if any(node.node_id == data_id for node in nodes):
            edges.append(ExplanationEdge(data_id, node_id, "derived_from"))

    for internals in evidence.model_internals:
        # node_ids must match claims.py's `ref = f"model_internals:{object_id}:..."`.
        if internals.linear_terms:
            node_id = f"model_internals:{internals.object_id}:linear_reconstruction"
            add_node(node_id, "model_internals", f"Линейное разложение ({len(internals.linear_terms)} слагаемых)", internals.to_dict())
            data_id = f"data:{internals.object_id}"
            upstream_id = data_id
            if internals.pipeline_steps:
                preprocessor_id = f"preprocessor:{internals.object_id}"
                add_node(preprocessor_id, "preprocessor", f"Предобработка ({' -> '.join(internals.pipeline_steps[:-1]) or 'тождественная'})", {"pipeline_steps": list(internals.pipeline_steps)})
                if any(node.node_id == data_id for node in nodes):
                    edges.append(ExplanationEdge(data_id, preprocessor_id, "derived_from"))
                upstream_id = preprocessor_id
            if any(node.node_id == upstream_id for node in nodes):
                edges.append(ExplanationEdge(upstream_id, node_id, "derived_from"))
        if internals.decision_path:
            node_id = f"model_internals:{internals.object_id}:decision_path"
            add_node(node_id, "model_internals", f"Путь по дереву решений ({len(internals.decision_path)} узлов)", internals.to_dict())
            data_id = f"data:{internals.object_id}"
            if any(node.node_id == data_id for node in nodes):
                edges.append(ExplanationEdge(data_id, node_id, "derived_from"))
        if internals.ensemble_votes is not None:
            node_id = f"model_internals:{internals.object_id}:ensemble_votes"
            add_node(node_id, "model_internals", f"Голосование ансамбля ({len(internals.ensemble_votes)} моделей)", internals.to_dict())
            data_id = f"data:{internals.object_id}"
            if any(node.node_id == data_id for node in nodes):
                edges.append(ExplanationEdge(data_id, node_id, "derived_from"))

    contributions = prediction.get("contributions", {})
    if isinstance(contributions, Mapping):
        # P18 item 10: a contribution genuinely derives from this object's
        # own data — without this edge, a claim's ancestry walk stopped one
        # hop too early (contribution -> claim, with nothing further back),
        # even though the real chain continues through data/preprocessing.
        contribution_data_id = f"data:{evidence.data[0].object_id}" if evidence.data else None
        for feature, value in contributions.items():
            node_id = f"contribution:{feature}"
            add_node(
                node_id,
                "contribution",
                f"Вклад признака {feature}",
                {"feature": str(feature), "value": float(value), "method": prediction.get("contribution_method")},
            )
            if contribution_data_id is not None and any(node.node_id == contribution_data_id for node in nodes):
                edges.append(ExplanationEdge(contribution_data_id, node_id, "derived_from"))

    add_node("prediction", "prediction", "Прогноз модели", prediction)
    if model_fingerprint:
        edges.append(ExplanationEdge("model_artifact:root", "prediction", "derived_from"))
    for node in nodes:
        if node.node_type in {"data", "contribution", "rule", "concept", "similar_case", "counterfactual", "fuzzy_rule", "image_region"}:
            relation = "changed_by" if node.node_type == "counterfactual" else "derived_from"
            edges.append(ExplanationEdge(node.node_id, "prediction", relation, node.evidence_refs))

    for index, diagnostic in enumerate(diagnostics):
        node_id = f"diagnostic:{index}"
        add_node(node_id, "diagnostic", _diagnostic_label_ru(diagnostic), diagnostic)
        edges.append(ExplanationEdge("prediction", node_id, "checked_by"))

    if evidence.missing:
        add_node(
            "trace:missing_evidence",
            "trace",
            "Недоступные каналы evidence",
            {"missing": list(evidence.missing)},
        )

    if system_evidence is not None:
        # P19: these are runtime facts, not a generator-side illustration.
        system_risk = system_evidence.get("risk", {})
        risk_components = system_risk.get("components", {}) if isinstance(system_risk, Mapping) else {}
        system_uncertainty = system_evidence.get("uncertainty", {})
        system_reduction = system_evidence.get("reduction") or {}
        i_pre_payload = system_evidence.get("i_pre", {})
        candidate_action = str(system_risk.get("candidate_action", "review")) if isinstance(system_risk, Mapping) else "review"
        critical_override = bool(system_risk.get("critical_override", False)) if isinstance(system_risk, Mapping) else False

        def value_label(symbol: str, value: Any) -> str:
            return f"{symbol} = {float(value):.4g}" if isinstance(value, (int, float)) else symbol

        system_nodes = (
            ("system:E_model", "explanation_object_model", "E_model", system_evidence.get("E_model", {})),
            ("system:T_ij", "alignment_transform", "T_ij", system_evidence.get("alignment_transform", {})),
            ("system:aligned_E_model", "aligned_explanation", "T_ij(E_model)", system_evidence.get("aligned_E_model", {})),
            ("system:E_target", "explanation_object_target", "E_target", system_evidence.get("E_target", {})),
            ("system:Gamma", "gamma", "Γ", system_evidence.get("gamma", {})),
            ("system:U_model", "uncertainty_model", value_label("U_model", system_uncertainty.get("u_model")), system_uncertainty),
            ("system:U_rules", "uncertainty_rules", value_label("U_rules", system_uncertainty.get("u_rules")), system_uncertainty),
            ("system:U_trace", "uncertainty_trace", value_label("U_trace", system_uncertainty.get("u_trace")), system_uncertainty),
            ("system:u_M", "uncertainty_aggregate", value_label("u_M", system_uncertainty.get("u_m")), system_uncertainty),
            ("system:representation", "uncertainty_representation", "F_int", {"representation": system_evidence.get("representation")}),
            ("system:reduction", "reduction", "Π / ι", system_evidence.get("reduction") or {"status": "not_applied"}),
            ("system:Delta", "delta", value_label("Δ", system_reduction.get("delta")), system_reduction or {"status": "not_applied"}),
            ("system:E_pre", "explanation_pre", "E_pre", system_evidence.get("E_pre", {})),
            ("system:I_pre", "interpretability_pre", value_label("I_pre", i_pre_payload.get("value") if isinstance(i_pre_payload, Mapping) else i_pre_payload), {"value": i_pre_payload}),
            ("system:rho_p", "risk_component", value_label("ρ_p", risk_components.get("rho_p")), {"value": risk_components.get("rho_p")}),
            ("system:one_minus_I_pre", "risk_component", value_label("1 - I_pre", risk_components.get("one_minus_I_pre")), {"value": risk_components.get("one_minus_I_pre")}),
            ("system:chi_R", "risk_component", value_label("χ_R", risk_components.get("chi_R")), {"value": risk_components.get("chi_R")}),
            ("system:rho", "risk", value_label("ρ", system_risk.get("rho") if isinstance(system_risk, Mapping) else None), system_risk),
            ("system:threshold_policy", "policy", "Пороговая политика ρ", {"candidate_action": candidate_action, "thresholds": system_risk.get("thresholds", {}), "reason": system_risk.get("candidate_action_reason")}),
            ("system:candidate_action", "candidate_action", f"Кандидат: {_ACTION_LABELS_RU.get(candidate_action, candidate_action)}", {"action": candidate_action, "reason": system_risk.get("candidate_action_reason")}),
            ("system:critical_override", "policy", f"Critical override: {'да' if critical_override else 'нет'}", {"critical_override": critical_override}),
            ("system:policy_resolution", "policy", "Разрешение политики", {"critical_override": critical_override, "candidate_action": candidate_action, "final_action": action, "reason": system_risk.get("final_action_reason")}),
        )
        for node_id, node_type, label, payload in system_nodes:
            add_node(node_id, node_type, label, payload if isinstance(payload, Mapping) else {})
        edges.extend((
            ExplanationEdge("prediction", "system:E_model", "derived_from"),
            ExplanationEdge("system:E_model", "system:T_ij", "transforms"),
            ExplanationEdge("system:T_ij", "system:aligned_E_model", "produces"),
            ExplanationEdge("system:aligned_E_model", "system:Gamma", "compared_by"),
            ExplanationEdge("system:E_target", "system:Gamma", "compared_by"),
            ExplanationEdge("system:U_model", "system:u_M", "aggregates"),
            ExplanationEdge("system:U_rules", "system:u_M", "aggregates"),
            ExplanationEdge("system:U_trace", "system:u_M", "aggregates"),
            ExplanationEdge("system:u_M", "system:representation", "represented_by"),
            ExplanationEdge("system:representation", "system:reduction", "reduces"),
            ExplanationEdge("system:reduction", "system:Delta", "measures"),
            ExplanationEdge("system:aligned_E_model", "system:E_pre", "composes"),
            ExplanationEdge("system:E_target", "system:E_pre", "composes"),
            ExplanationEdge("system:u_M", "system:E_pre", "composes"),
            ExplanationEdge("system:Delta", "system:E_pre", "composes"),
            ExplanationEdge("system:E_pre", "system:I_pre", "evaluated_by"),
            ExplanationEdge("prediction", "system:rho_p", "measures"),
            ExplanationEdge("system:I_pre", "system:one_minus_I_pre", "derives"),
            ExplanationEdge("system:Gamma", "system:chi_R", "evaluates_rupture"),
            ExplanationEdge("system:U_trace", "system:chi_R", "evaluates_rupture"),
            ExplanationEdge("system:rho_p", "system:rho", "aggregates"),
            ExplanationEdge("system:u_M", "system:rho", "aggregates"),
            ExplanationEdge("system:one_minus_I_pre", "system:rho", "aggregates"),
            ExplanationEdge("system:Delta", "system:rho", "aggregates"),
            ExplanationEdge("system:chi_R", "system:rho", "aggregates"),
            ExplanationEdge("system:rho", "system:threshold_policy", "evaluated_by"),
            ExplanationEdge("system:threshold_policy", "system:candidate_action", "produces"),
            ExplanationEdge("system:chi_R", "system:critical_override", "evaluated_by"),
            ExplanationEdge("system:candidate_action", "system:policy_resolution", "resolves"),
            ExplanationEdge("system:critical_override", "system:policy_resolution", "resolves"),
        ))
        if critical_override:
            add_node("system:critical", "diagnostic", "Критический разрыв", {"critical_override": True})
            edges.append(ExplanationEdge("system:critical", "system:chi_R", "sets"))
            edges.append(ExplanationEdge("system:critical", "system:critical_override", "activates"))
    add_node("action", "action", _ACTION_LABELS_RU.get(action, action), {"action": action})
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

    if system_evidence is None:
        edges.append(ExplanationEdge("prediction", "action", "recommends"))
    if system_evidence is not None:
        edges.append(ExplanationEdge("system:policy_resolution", "action", "selects"))
    for index in range(len(diagnostics)):
        edges.append(ExplanationEdge(f"diagnostic:{index}", "action", "constrains"))
    for claim in claims:
        if claim.claim_type == "recommended_action":
            edges.append(ExplanationEdge(f"claim:{claim.claim_id}", "action", "recommends"))
    return ExplanationGraph(nodes=nodes, edges=edges, claims=list(claims), missing_evidence=list(evidence.missing))
