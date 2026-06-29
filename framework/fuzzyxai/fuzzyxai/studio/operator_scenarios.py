from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from fuzzyxai.core.diagnostics import DiagnosticType
from fuzzyxai.core.scenario_engine import compute_hybrid_xiris


ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "configs" / "studio_scenarios"


STATUS_COLOR = {
    "passed": "#16a34a",
    "valid": "#16a34a",
    "warning": "#d97706",
    "blocked": "#dc2626",
    "rupture": "#dc2626",
    "not_applicable": "#64748b",
    "info": "#2563eb",
}

STATUS_LABELS = {
    "connected": "подключено",
    "verified": "проверено",
    "scenario_run_verified": "сценарный прогон",
    "source_pending": "источник уточняется",
    "source-pending": "источник уточняется",
    "fixture-certified": "проверочный стенд",
    "limited": "ограниченный вывод",
    "blocked": "блокировка",
    "passed": "пройдено",
    "warning": "предупреждение",
    "critical": "критично",
    "info": "инфо",
}

STATUS_COLORS = {
    "connected": "#2563eb",
    "verified": "#059669",
    "scenario_run_verified": "#0f766e",
    "source_pending": "#d97706",
    "source-pending": "#d97706",
    "fixture-certified": "#0f766e",
    "limited": "#7c3aed",
    "blocked": "#dc2626",
    **STATUS_COLOR,
}


def collect_unique_diagnostics(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, Any, Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for node in pipeline:
        for diagnostic in node.get("diagnostics", []):
            key = (
                diagnostic.get("diagnostic_id"),
                diagnostic.get("type"),
                diagnostic.get("source"),
                diagnostic.get("recommended_action"),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(diagnostic)
    return out


def summarize_diagnostic_occurrences(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for node in pipeline:
        node_id = node.get("node_id")
        for diagnostic in node.get("diagnostics", []):
            diagnostic_id = diagnostic.get("diagnostic_id")
            if not diagnostic_id:
                continue
            if diagnostic_id not in grouped:
                grouped[diagnostic_id] = {**diagnostic, "occurrences": [], "global": True}
            grouped[diagnostic_id]["occurrences"].append(node_id)
    return list(grouped.values())


def _node(
    node_id: str,
    title: str,
    status: str,
    operator_id: str,
    operator_name: str,
    source_chapter: str,
    formula: str,
    description: str,
    inputs: dict[str, Any],
    computed: dict[str, Any],
    output: dict[str, Any],
    diagnostics: list[dict[str, Any]] | None = None,
    effect: str = "",
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": node_id,
        "title": title,
        "status": status,
        "operator": {
            "operator_id": operator_id,
            "operator_name": operator_name,
            "source_chapter": source_chapter,
            "formula_latex": formula,
            "description": description,
        },
        "inputs": inputs,
        "computed": computed,
        "output": output,
        "diagnostics": diagnostics or [],
        "effect_on_final_action": effect,
    }


def _edge(edge_id: str, src: str, dst: str, operator_id: str, status: str, computed: dict[str, Any], reason: str = "") -> dict[str, Any]:
    return {
        "edge_id": edge_id,
        "from": src,
        "to": dst,
        "operator_id": operator_id,
        "status": status,
        "computed": computed,
        "diagnostic": {
            "has_rupture": status in {"blocked", "rupture"},
            "criticality": "high" if status in {"blocked", "rupture"} else ("medium" if status == "warning" else "low"),
            "reason": reason,
        },
    }


def _base_pipeline(final_action: str, rupture: bool, values: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diag = []
    if rupture:
        diag = [
            {
                "diagnostic_id": DiagnosticType.QUALITY_SOURCE_CONFLICT.value,
                "diagnostic_type": "quality_source_conflict",
                "legacy_id": "D_source_conflict",
                "type": "critical_rupture",
                "source": values.get("rupture_source", "source_conflict"),
                "reason": values.get("action_reason", "Источник поддержки противоречит источнику качества."),
                "criticality": "high",
                "chi_R_crit": 1,
                "recommended_action": "block",
            }
        ]
    nodes = [
        _node(
            "input_artifact",
            "Input Artifact",
            "info",
            "input",
            "Входной артефакт",
            "4",
            r"x \in U_{artifact}",
            "Фиксирует исходный объект и trace входа.",
            {"input_id": values.get("input_id", "sample_001"), "data_type": values.get("data_type", "artifact")},
            {"hash": values.get("input_hash", "auto")},
            {"artifact_ready": True},
            effect="Передаёт проверяемый вход адаптеру.",
        ),
        _node(
            "adapter",
            "Adapter",
            "passed",
            "adapter",
            "Адаптер внешнего метода",
            "4",
            r"A(x) \rightarrow channels(E_k)",
            "Переводит внешний результат в каналы объяснительного объекта.",
            values.get("adapter_inputs", {}),
            values.get("adapter_computed", {}),
            {"adapter_status": "valid", "trace_complete": True},
            effect="Создаёт проверяемый интерфейс для операторов главы 2.",
        ),
        _node(
            "explanation_object",
            "Explanation Object Eₖ",
            "passed" if not rupture else "warning",
            "build_Ek",
            "Построение объяснительного объекта Eₖ",
            "2",
            r"E_k=\langle L_k,\mu_k,R_k,\alpha_k,u_k,\tau_k\rangle",
            "Собирает термы, принадлежности, правила, активации, неопределённость и trace.",
            values.get("ek_inputs", {}),
            values.get("ek_computed", {}),
            {"object_type": "E_k", "valid": True},
            effect="Объект передан в согласование Tᵢⱼ и риск-наблюдатель.",
        ),
        _node(
            "alignment",
            "Interface Alignment Tᵢⱼ",
            "warning" if rupture else "passed",
            "T_ij",
            "Согласование интерфейсов Tᵢⱼ",
            "2",
            r"T_{ij}=(\rho_{ij},\sigma_{ij},\pi_{ij},\theta_{ij}),\quad \gamma_{ij}\leq\gamma_{max}",
            "Проверяет совместимость термов, правил и trace соседних объектов.",
            {"source_object": "E_model", "target_object": "E_quality"},
            values.get("alignment_computed", {"gamma_ij": 0.2, "gamma_max": 0.4, "delta_T": 0.08}),
            {"composition": "E_ij" if not rupture else "D_ij warning"},
            diag,
            effect="Высокое рассогласование усиливает диагностический контур.",
        ),
        _node(
            "f_selector",
            "Uncertainty Representation F",
            "passed",
            "select_F",
            "Выбор класса представления F",
            "3",
            r"F^*=\arg\min_F(c_H+c_O+c_K+\Delta_F)",
            "Выбирает F0, F_int, NAS или F_ML по покрытию и цене редукции.",
            {"P_sit": values.get("P_sit", ["u_conf", "u_trace"])},
            values.get("f_computed", {"selected_class": "NAS", "coverage": "u_conf,u_trace,source_conflict"}),
            {"selected_class": values.get("selected_class", "NAS")},
            effect="NAS/F_ML сохраняет источник поддержки и контрсвидетельства.",
        ),
        _node(
            "reduction",
            "Reduction Δ",
            "warning" if values.get("delta", 0.0) > 0.25 else "passed",
            "Delta",
            "Редукция и потеря Δ",
            "3",
            r"\Delta=\mathcal{L}(F_{source}\rightarrow F_{target})",
            "Оценивает потерю информации при упрощении представления.",
            {"source_representation": values.get("selected_class", "NAS"), "target_representation": "user_action"},
            {"delta": values.get("delta", 0.08), "delta_max": values.get("delta_max", 0.35)},
            {"reduction_allowed": values.get("delta", 0.08) <= values.get("delta_max", 0.35)},
            effect="Потеря Δ входит в риск и может запретить авто-действие.",
        ),
        _node(
            "risk_observer",
            "Risk Observer",
            "blocked" if final_action == "block" else ("warning" if final_action != "accept" else "passed"),
            "risk_observer",
            "Риск-ориентированный наблюдатель",
            "3",
            r"\rho=w_p\rho_{pred}+w_u u_M+w_I(1-I_{pre})+w_\Delta\Delta_M+w_R\chi_R",
            "Вычисляет риск, учитывает χ_R и χ_R^crit, выбирает допустимое действие.",
            values.get("risk_inputs", {}),
            values.get("risk_computed", {}),
            {"final_action": final_action},
            diag,
            effect=values.get("action_reason", "Формирует итоговое действие."),
        ),
        _node(
            "action",
            "Action",
            "blocked" if final_action == "block" else "passed",
            "action_policy",
            "Политика действия",
            "3",
            r"\chi_R^{crit}=1\Rightarrow action=block",
            "Преобразует риск и диагностический статус в действие.",
            {"rho": values.get("risk_computed", {}).get("rho", 0.0), "chi_R_crit": int(rupture)},
            {"action": final_action},
            {"action": final_action, "reason": values.get("action_reason", "")},
            diag,
            effect="Показывает, можно ли использовать результат автоматически.",
        ),
        _node(
            "report",
            "Report / Proof Package",
            "info",
            "report_export",
            "Отчёт и proof package",
            "4",
            r"Report=(operators,diagnostics,trace,action)",
            "Формирует JSON-отчёт с операторами, числами, диагностикой и trace.",
            {"operators": ["build_Ek", "T_ij", "select_F", "risk_observer"]},
            {"report_ready": True},
            {"export": "json"},
            effect="Сохраняет проверяемый след расчёта.",
        ),
    ]
    edges = [
        _edge("input_to_adapter", "input_artifact", "adapter", "adapter", "passed", {}),
        _edge("adapter_to_ek", "adapter", "explanation_object", "build_Ek", "passed", {}),
        _edge("ek_to_alignment", "explanation_object", "alignment", "T_ij", "warning" if rupture else "passed", values.get("alignment_computed", {}), values.get("action_reason", "")),
        _edge("alignment_to_f", "alignment", "f_selector", "select_F", "warning" if rupture else "passed", {}),
        _edge("f_to_reduction", "f_selector", "reduction", "Delta", "passed", {"delta": values.get("delta", 0.08)}),
        _edge("reduction_to_risk", "reduction", "risk_observer", "risk_observer", "blocked" if final_action == "block" else "passed", values.get("risk_computed", {}), values.get("action_reason", "")),
        _edge("risk_to_action", "risk_observer", "action", "action_policy", "blocked" if final_action == "block" else "passed", {"action": final_action}),
        _edge("action_to_report", "action", "report", "report_export", "passed", {}),
    ]
    return nodes, edges


def _scenario(
    scenario_id: str,
    scenario_name: str,
    domain: str,
    data_type: str,
    status: str,
    description: str,
    summary: dict[str, Any],
    values: dict[str, Any],
    final_action: str,
    rupture: bool,
    charts: dict[str, Any],
) -> dict[str, Any]:
    nodes, edges = _base_pipeline(final_action, rupture, {"data_type": data_type, **values})
    expected = values.get("expected_result", {})
    if expected:
        for node in nodes:
            if node.get("node_id") == "alignment":
                node["computed"]["gamma_ij"] = expected.get("gamma", node["computed"].get("gamma_ij"))
            elif node.get("node_id") == "reduction":
                node["computed"]["delta"] = expected.get("delta", node["computed"].get("delta"))
            elif node.get("node_id") == "risk_observer":
                node["computed"]["rho"] = expected.get("rho", node["computed"].get("rho"))
                node["computed"]["chi_R"] = expected.get("chi_R", node["computed"].get("chi_R"))
                node["computed"]["chi_R_crit"] = expected.get("chi_R_crit", node["computed"].get("chi_R_crit"))
            elif node.get("node_id") == "action":
                node["inputs"]["rho"] = expected.get("rho", node["inputs"].get("rho"))
                node["computed"]["action"] = expected.get("action", node["computed"].get("action"))
                node["output"]["action"] = expected.get("action", node["output"].get("action"))
        for edge in edges:
            if edge.get("operator_id") == "T_ij":
                edge["computed"]["gamma_ij"] = expected.get("gamma", edge.get("computed", {}).get("gamma_ij"))
            elif edge.get("operator_id") == "Delta":
                edge["computed"]["delta"] = expected.get("delta", edge.get("computed", {}).get("delta"))
            elif edge.get("operator_id") == "risk_observer":
                edge["computed"]["rho"] = expected.get("rho", edge.get("computed", {}).get("rho"))
    diagnostics = collect_unique_diagnostics(nodes)
    diagnostics_summary = summarize_diagnostic_occurrences(nodes)
    run_id = f"{scenario_id}_case_001"
    report = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "explain_plan_version": "EP-2026-01",
        "input_id": values.get("input_id", "sample_001"),
        "final_action": final_action,
        "action_reason": values.get("action_reason", ""),
        "operators_used": ["adapter", "build_Ek", "T_ij", "select_F", "Delta", "risk_observer", "action_policy"],
        "diagnostics": diagnostics_summary,
        "trace": {
            "model_version": values.get("model_version", "demo-model-v1"),
            "adapter_version": values.get("adapter_version", f"adapter-{scenario_id}-v1"),
            "timestamp": "auto",
            "hash": "auto",
        },
    }
    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "domain": domain,
        "data_type": data_type,
        "explain_plan_version": "EP-2026-01",
        "status": status,
        "evidence_level": {
            "hybrid_xiris": "full_control_run",
            "gd_anfis_shap": "operator_control_example",
            "gis_integro": "operator_control_example",
            "beacon_xai": "route_demonstration",
            "medical_ecg_signal": "operator_control_example",
        }.get(scenario_id, "extension_area"),
        "description": description,
        "summary": summary,
        "pipeline": nodes,
        "edges": edges,
        "diagnostics": diagnostics,
        "diagnostics_summary": diagnostics_summary,
        "expected_result": values.get("expected_result", {}),
        "runs": [report],
        "charts": charts,
    }


_HYBRID_RESULT = compute_hybrid_xiris()


DEFAULT_SCENARIOS: list[dict[str, Any]] = [
    _scenario(
        "hybrid_xiris",
        "HYBRID-XIRIS",
        "biometrics",
        "iris_image",
        "scenario_run_verified",
        "Конфликт между модельным сигналом и качеством сегментации радужной оболочки.",
        {"objects_total": 1000, "accept": 612, "lower_confidence": 201, "block": 187, "baseline_critical_misses": 168, "fuzzyxai_critical_misses": 0},
        {
            "input_id": "iris_case_001",
            "adapter_inputs": {"image_quality": 0.31, "segmentation_quality": 0.27, "model_match_signal": 0.88},
            "adapter_computed": {"quality_term": "low", "match_term": "high"},
            "ek_inputs": {"model_score": 0.88, "segmentation_quality": 0.27, "feature_support": "iris texture", "trace": "complete"},
            "ek_computed": {"mu_high": 0.88, "alpha_accept": 0.82, "alpha_block": 0.91, "u_k": 0.36},
            "alignment_computed": {"gamma_ij": _HYBRID_RESULT.gamma, "gamma_max": 0.40, "delta_T": 0.08},
            "P_sit": ["u_conf", "u_trace", "source_conflict"],
            "f_computed": {"selected_class": "NAS", "coverage": "u_conf,u_trace,source_conflict", "why_not_F0": "F0 теряет источник контрсвидетельства"},
            "selected_class": "NAS",
            "delta": _HYBRID_RESULT.delta,
            "risk_inputs": {"rho_pred": 0.88, "u_M": 0.36, "chi_R": 1, "chi_R_crit": 1},
            "risk_computed": {"rho": _HYBRID_RESULT.rho, "chi_R": 1, "chi_R_crit": 1, "chi_Auto": 0},
            "expected_result": {
                "gamma": _HYBRID_RESULT.gamma,
                "delta": _HYBRID_RESULT.delta,
                "rho": _HYBRID_RESULT.rho,
                "chi_R": _HYBRID_RESULT.chi_r,
                "chi_R_crit": _HYBRID_RESULT.chi_r_crit,
                "action": _HYBRID_RESULT.action,
            },
            "rupture_source": "segmentation_quality",
            "action_reason": "Модель поддерживает принятие, но низкое качество сегментации и конфликт источников дают критический разрыв.",
        },
        "block",
        True,
        {"actions": {"accept": 612, "lower_confidence": 201, "block": 187}, "misses": {"baseline": 168, "FuzzyXAI": 0}},
    ),
    _scenario(
        "beacon_xai",
        "BEACON-XAI",
        "monitoring",
        "time_series",
        "fixture-certified",
        "Контрсвидетельство BEACON переводится в диагностический аудиторский отчёт.",
        {"objects_total": 100, "counter_evidence_detected": 83, "checks_without_beacon": 64, "checks_with_beacon": 11, "audit_reports": 12},
        {
            "input_id": "beacon_signal_001",
            "adapter_inputs": {"counter_evidence_score": 0.83, "signal_window": "t-32:t"},
            "adapter_computed": {"counter_evidence_detected": 83, "audit_reports": 12},
            "ek_inputs": {"beacon_signal": 0.83, "trace": "github-head+fixture"},
            "ek_computed": {"mu_counter_evidence": 0.83, "alpha_audit": 0.79, "u_k": 0.28},
            "alignment_computed": {"gamma_ij": 0.22, "gamma_max": 0.40, "delta_T": 0.05},
            "risk_inputs": {"rho_pred": 0.54, "u_M": 0.28, "chi_R": 1, "chi_R_crit": 0},
            "risk_computed": {"rho": 0.58, "chi_R": 1, "chi_R_crit": 0, "chi_Auto": 0},
            "selected_class": "NAS",
            "delta": 0.05,
            "action_reason": "BEACON-сигнал является контрсвидетельством и переводит маршрут в audit report.",
        },
        "defer_to_human",
        False,
        {"beacon_checks": {"without_BEACON": 64, "with_BEACON": 11}, "evidence": {"counter_evidence_detected": 83, "audit_reports": 12}},
    ),
    _scenario(
        "gis_integro",
        "GIS INTEGRO",
        "geospatial",
        "geo_layer",
        "source-pending",
        "Геослой, правила и feature support объединяются в проверяемый маршрут без заявления качества исходной геомодели.",
        {"probability": 0.67, "mean_alpha_k": 0.72, "positive_feature_support": 0.47, "gamma_route": 0.20, "delta": 0.08},
        {
            "input_id": "geo_layer_001",
            "adapter_inputs": {"probability": 0.67, "layer_source": "fixture"},
            "adapter_computed": {"mean_alpha_k": 0.72, "positive_feature_support": 0.47},
            "ek_inputs": {"probability": 0.67, "rules": "geo constraints", "feature_support": 0.47},
            "ek_computed": {"mu_geo_support": 0.67, "alpha_k_mean": 0.72, "u_k": 0.31},
            "alignment_computed": {"gamma_ij": 0.20, "gamma_max": 0.40, "delta_T": 0.08},
            "risk_inputs": {"rho_pred": 0.67, "u_M": 0.31, "chi_R": 0, "chi_R_crit": 0},
            "risk_computed": {"rho": 0.41, "chi_R": 0, "chi_R_crit": 0, "chi_Auto": 0},
            "selected_class": "F_ML",
            "delta": 0.08,
            "action_reason": "Маршрут согласован, но статус source-pending оставляет результат отчётным, а не claim о качестве геомодели.",
        },
        "audit_report",
        False,
        {"gis_values": {"probability": 0.67, "mean_alpha": 0.72, "feature_support": 0.47}},
    ),
    _scenario(
        "gd_anfis_shap",
        "GD-ANFIS/SHAP",
        "tabular",
        "tabular_rules",
        "source-pending",
        "ANFIS-правила и SHAP-вклады согласуются как источники объяснительного объекта.",
        {"rules": 4, "mean_activation": 0.69, "shap_positive_support": 0.58, "diagnostic_warnings": 1},
        {
            "input_id": "tabular_case_001",
            "adapter_inputs": {"anfis_rules": 4, "shap_features": 6},
            "adapter_computed": {"mean_activation": 0.69, "shap_positive_support": 0.58},
            "ek_inputs": {"rules": "ANFIS", "alpha_k": "native", "eta_k": "SHAP"},
            "ek_computed": {"alpha_rule_1": 0.74, "alpha_rule_2": 0.61, "shap_support": 0.58, "u_k": 0.33},
            "alignment_computed": {"gamma_ij": 0.29, "gamma_max": 0.40, "delta_T": 0.10},
            "P_sit": ["u_conf", "u_num", "u_trace"],
            "f_computed": {"selected_class": "F_ML", "warning": "rule/SHAP support differs on feature x_4"},
            "selected_class": "F_ML",
            "delta": 0.10,
            "risk_inputs": {"rho_pred": 0.62, "u_M": 0.33, "chi_R": 1, "chi_R_crit": 0},
            "risk_computed": {"rho": 0.52, "chi_R": 1, "chi_R_crit": 0, "chi_Auto": 0},
            "action_reason": "Правила и SHAP частично расходятся, поэтому формируется diagnostic warning и отчётный режим.",
        },
        "defer_to_human",
        False,
        {"anfis": {"rules": 4, "mean_activation": 0.69, "shap_positive_support": 0.58, "diagnostic_warnings": 1}},
    ),
    _scenario(
        "medical_ecg_signal",
        "Medical ECG Signal",
        "medical_signal",
        "ECG signal",
        "limited",
        "Контрольный ЭКГ-сигнал: демонстрация качества сигнала, интервальной неопределённости и аудиторского действия без медицинской диагностики.",
        {"segments": 12, "noisy_segments": 3, "missing_fragments": 2, "quality_score": 0.58, "audit_actions": 1},
        {
            "input_id": "ecg_control_signal_001",
            "adapter_inputs": {"signal_quality": 0.58, "noise_level": 0.34, "missing_fragments": 2},
            "adapter_computed": {"quality_term": "borderline", "uncertainty_class": "F_int", "trace_complete": True},
            "ek_inputs": {"rr_variability": 0.21, "noise_level": 0.34, "segment_quality": 0.58},
            "ek_computed": {"mu_noisy": 0.34, "mu_quality_borderline": 0.58, "u_k": 0.42, "alpha_audit": 0.71},
            "alignment_computed": {"gamma_ij": 0.26, "gamma_max": 0.40, "delta_T": 0.06},
            "P_sit": ["u_int", "u_trace", "u_missing"],
            "f_computed": {"selected_class": "F_int", "coverage": "interval uncertainty, trace quality", "clinical_claim": "not asserted"},
            "selected_class": "F_int",
            "delta": 0.06,
            "risk_inputs": {"rho_pred": 0.48, "u_M": 0.42, "chi_R": 1, "chi_R_crit": 0},
            "risk_computed": {"rho": 0.61, "chi_R": 1, "chi_R_crit": 0, "chi_Auto": 0},
            "action_reason": "Сигнал шумный/неполный, поэтому автоматическая интерпретация запрещена: требуется аудит или повтор данных. Это не медицинская диагностика.",
        },
        "defer_to_human",
        False,
        {"ecg_quality": {"quality_score": 0.58, "noise_level": 0.34, "missing_fragments": 2, "audit_actions": 1}},
    ),
]


def ensure_scenario_json_files(directory: Path = SCENARIO_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for scenario in DEFAULT_SCENARIOS:
        path = directory / f"{scenario['scenario_id']}.json"
        tmp = directory / f".{scenario['scenario_id']}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(scenario, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[dict[str, Any]]:
    ensure_scenario_json_files(directory)
    scenarios: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            scenarios.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    order = {scenario["scenario_id"]: i for i, scenario in enumerate(DEFAULT_SCENARIOS)}
    scenarios.sort(key=lambda item: order.get(str(item.get("scenario_id")), 10_000))
    return scenarios or deepcopy(DEFAULT_SCENARIOS)


def build_report(scenario: dict[str, Any]) -> dict[str, Any]:
    report = deepcopy((scenario.get("runs") or [{}])[0])
    report["diagnostics"] = summarize_diagnostic_occurrences(scenario.get("pipeline", []))
    report["timestamp_generated"] = datetime.now(timezone.utc).isoformat()
    trace = report.setdefault("trace", {})
    trace["timestamp"] = report["timestamp_generated"]
    trace["hash"] = sha256(json.dumps(scenario, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    report["operators"] = [
        {
            "node_id": node.get("node_id"),
            "operator_id": node.get("operator", {}).get("operator_id"),
            "status": node.get("status"),
            "computed": node.get("computed", {}),
            "diagnostics": collect_unique_diagnostics([node]),
        }
        for node in scenario.get("pipeline", [])
    ]
    return report


def build_ecosystem_entities(scenarios: list[dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    scenarios = scenarios or load_scenarios()
    scenario_ids = [s["scenario_id"] for s in scenarios]
    articles = [
        {
            "id": "article_hybrid_xiris",
            "kind": "article",
            "title": "HYBRID-XIRIS: нейро-нечёткий биометрический контур",
            "subtitle": "модель + сценарий",
            "description": "Проверка конфликта между модельным сигналом и качеством сегментации радужной оболочки.",
            "tags": ["biometrics", "iris", "NAS", "risk-observer"],
            "status": "verified",
            "resultType": "model+scenario",
            "connectedModels": ["model_iris_quality", "model_iris_matcher"],
            "connectedScenarios": ["hybrid_xiris"],
            "connectedOperators": ["build_Ek", "T_ij", "select_F", "risk_observer"],
        },
        {
            "id": "article_beacon_xai",
            "kind": "article",
            "title": "BEACON-XAI: контрсвидетельства во временных сигналах",
            "subtitle": "adapter fixture",
            "description": "BEACON-сигнал подключается как источник контрсвидетельства и аудиторского отчёта.",
            "tags": ["time-series", "counter-evidence", "audit"],
            "status": "connected",
            "resultType": "method+adapter",
            "connectedModels": ["model_beacon_counter"],
            "connectedScenarios": ["beacon_xai"],
            "connectedOperators": ["build_Ek", "risk_observer"],
        },
        {
            "id": "article_gis_integro",
            "kind": "article",
            "title": "GIS INTEGRO: геослой и правиловой маршрут",
            "subtitle": "source-pending scenario",
            "description": "Геослой, признаки и правила объединяются в проверяемый объяснительный маршрут.",
            "tags": ["geo", "rules", "route"],
            "status": "limited",
            "resultType": "scenario",
            "connectedModels": ["model_gis_route"],
            "connectedScenarios": ["gis_integro"],
            "connectedOperators": ["T_ij", "Delta", "report_export"],
        },
        {
            "id": "article_gd_anfis_shap",
            "kind": "article",
            "title": "GD-ANFIS/SHAP: правила и вклады признаков",
            "subtitle": "neuro-fuzzy + SHAP",
            "description": "ANFIS-правила и SHAP-вклады согласуются как источники объяснительного объекта.",
            "tags": ["ANFIS", "SHAP", "tabular"],
            "status": "limited",
            "resultType": "model+method",
            "connectedModels": ["model_gd_anfis", "model_shap_explainer"],
            "connectedScenarios": ["gd_anfis_shap"],
            "connectedOperators": ["build_Ek", "T_ij", "select_F"],
        },
    ]
    models = [
        {
            "id": "model_iris_quality",
            "kind": "model",
            "title": "Iris segmentation quality model",
            "subtitle": "quality gate",
            "description": "Оценивает надёжность сегментации радужной оболочки.",
            "domain": "biometrics",
            "inputType": "iris image + mask",
            "outputType": "segmentation_quality",
            "adapterId": "iris_quality_adapter",
            "explainabilityChannels": ["quality_score", "quality_terms", "trace"],
            "limitations": ["не является самостоятельной биометрической моделью"],
            "status": "verified",
            "usedIn": ["hybrid_xiris"],
        },
        {
            "id": "model_iris_matcher",
            "kind": "model",
            "title": "Neuro-fuzzy iris matcher",
            "subtitle": "matching score + rules",
            "description": "Оценивает совпадение радужной оболочки и активированные правила принятия.",
            "domain": "biometrics",
            "inputType": "iris features",
            "outputType": "model_match_signal, alpha_accept, alpha_block",
            "adapterId": "iris_match_adapter",
            "explainabilityChannels": ["score", "confidence", "rule_activation", "trace"],
            "limitations": ["автоматическое действие разрешено только после риск-наблюдателя"],
            "status": "verified",
            "usedIn": ["hybrid_xiris"],
        },
        {
            "id": "model_beacon_counter",
            "kind": "model",
            "title": "BEACON counter-evidence module",
            "subtitle": "counter-evidence detector",
            "description": "Выделяет контрсвидетельства во временном сигнале.",
            "domain": "monitoring",
            "inputType": "time series",
            "outputType": "counter_evidence_score",
            "adapterId": "beacon_adapter",
            "explainabilityChannels": ["counter_evidence", "audit_trace"],
            "limitations": ["не заявляет новую метрику качества исходной модели"],
            "status": "connected",
            "usedIn": ["beacon_xai"],
        },
        {
            "id": "model_gis_route",
            "kind": "model",
            "title": "GIS route explanation profile",
            "subtitle": "geo layer + rules",
            "description": "Связывает вероятность, правила геослоя и поддержку признаков.",
            "domain": "geospatial",
            "inputType": "geo layer",
            "outputType": "probability, alpha_k, feature_support",
            "adapterId": "gis_adapter",
            "explainabilityChannels": ["layer_source", "rule_activation", "feature_support"],
            "limitations": ["source-pending; качество исходной геомодели не утверждается"],
            "status": "limited",
            "usedIn": ["gis_integro"],
        },
        {
            "id": "model_gd_anfis",
            "kind": "model",
            "title": "GD-ANFIS rule model",
            "subtitle": "native fuzzy rules",
            "description": "Нейро-нечёткая модель с нативными термами, правилами и активациями.",
            "domain": "tabular",
            "inputType": "tabular object",
            "outputType": "rules, alpha_k",
            "adapterId": "gd_anfis_adapter",
            "explainabilityChannels": ["L_k", "mu_k", "R_k", "alpha_k"],
            "limitations": ["fixture source-pending"],
            "status": "limited",
            "usedIn": ["gd_anfis_shap"],
        },
        {
            "id": "model_shap_explainer",
            "kind": "model",
            "title": "SHAP contribution explainer",
            "subtitle": "feature contributions",
            "description": "Преобразует вклады признаков в каналы поддержки и контрсвидетельства.",
            "domain": "tabular",
            "inputType": "model + tabular object",
            "outputType": "feature contributions",
            "adapterId": "shap_adapter",
            "explainabilityChannels": ["feature_support", "sign", "trace"],
            "limitations": ["согласуется с правилами через T_ij"],
            "status": "connected",
            "usedIn": ["gd_anfis_shap"],
        },
    ]
    operator_map: dict[str, dict[str, Any]] = {}
    verified_operator_ids = {"build_Ek", "T_ij", "select_F", "Delta", "risk_observer", "action_policy"}
    technical_operator_ids = {"input", "adapter", "report_export"}
    for scenario in scenarios:
        for node in scenario.get("pipeline", []):
            op = node.get("operator", {})
            op_id = str(op.get("operator_id", node.get("node_id")))
            status = "verified" if op_id in verified_operator_ids else ("connected" if op_id in technical_operator_ids else "limited")
            if op_id not in operator_map:
                operator_map[op_id] = {
                    "id": op_id,
                    "kind": "operator",
                    "title": op.get("operator_name", op_id),
                    "subtitle": f"глава {op.get('source_chapter', '')}",
                    "description": op.get("description", ""),
                    "tags": [f"chapter-{op.get('source_chapter', '')}", node.get("node_type", "")],
                    "status": status,
                    "chapter": str(op.get("source_chapter", "")),
                    "formula": op.get("formula_latex", ""),
                    "checks": [],
                    "possibleOutcomes": ["passed", "warning", "diagnostic rupture", "blocked"],
                    "usedInScenarios": [],
                    "realCases": [],
                }
            operator_map[op_id]["usedInScenarios"].append(scenario["scenario_id"])
            operator_map[op_id]["realCases"].append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_name": scenario["scenario_name"],
                    "node_title": node.get("title"),
                    "status": node.get("status"),
                    "computed": node.get("computed", {}),
                    "effect": node.get("effect_on_final_action", ""),
                }
            )
    operators = list(operator_map.values())
    model_ids_by_scenario: dict[str, list[str]] = {}
    for model in models:
        for scenario_id in model.get("usedIn", []):
            model_ids_by_scenario.setdefault(scenario_id, []).append(model["id"])

    for article in articles:
        article["links"] = (
            [{"targetId": mid, "targetKind": "model", "relation": "uses_model"} for mid in article.get("connectedModels", [])]
            + [{"targetId": sid, "targetKind": "scenario", "relation": "tested_in"} for sid in article.get("connectedScenarios", [])]
            + [{"targetId": oid, "targetKind": "operator", "relation": "uses_operator"} for oid in article.get("connectedOperators", [])]
        )

    operator_ids_by_scenario = {scenario["scenario_id"]: [] for scenario in scenarios}
    for operator in operators:
        operator["links"] = []
        for scenario_id in sorted(set(operator.get("usedInScenarios", []))):
            operator["links"].append({"targetId": scenario_id, "targetKind": "scenario", "relation": "used_in"})
            operator_ids_by_scenario.setdefault(scenario_id, []).append(operator["id"])
        for model in models:
            if any(sid in operator.get("usedInScenarios", []) for sid in model.get("usedIn", [])):
                operator["links"].append({"targetId": model["id"], "targetKind": "model", "relation": "checks_model"})

    for model in models:
        model["links"] = [{"targetId": sid, "targetKind": "scenario", "relation": "tested_in"} for sid in model.get("usedIn", [])]
        for scenario_id in model.get("usedIn", []):
            for operator_id in operator_ids_by_scenario.get(scenario_id, []):
                model["links"].append({"targetId": operator_id, "targetKind": "operator", "relation": "uses_operator"})

    for scenario in scenarios:
        scenario["id"] = scenario["scenario_id"]
        scenario["kind"] = "scenario"
        scenario["title"] = scenario["scenario_name"]
        scenario["links"] = (
            [{"targetId": mid, "targetKind": "model", "relation": "uses_model"} for mid in model_ids_by_scenario.get(scenario["scenario_id"], [])]
            + [{"targetId": oid, "targetKind": "operator", "relation": "uses_operator"} for oid in operator_ids_by_scenario.get(scenario["scenario_id"], [])]
        )

    return {
        "articles": articles,
        "models": models,
        "scenarios": scenarios,
        "operators": operators,
        "runs": [build_report(s) for s in scenarios if s.get("runs")],
        "scenario_ids": scenario_ids,
    }
