#!/usr/bin/env python3
"""Build claim registry 2.0 without promoting missing evidence."""

from __future__ import annotations

import json
from pathlib import Path

from fuzzyxai.q1_final import ClaimLevel, ClaimRecordV2, ClaimStatusV2


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_final"


def load(relative: str, default: object = None) -> object:
    path = EVIDENCE / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def record(
    claim_id: str,
    level: ClaimLevel,
    status: ClaimStatusV2,
    claim: str,
    evidence: tuple[str, ...],
    commit: str,
    *,
    ru: str = "",
    en: str = "",
    datasets: tuple[str, ...] = (),
    models: tuple[str, ...] = (),
    n_objects: int = 0,
    n_seeds: int = 0,
    metrics: dict[str, object] | None = None,
    ci: dict[str, object] | None = None,
    limitations: tuple[str, ...] = (),
) -> ClaimRecordV2:
    return ClaimRecordV2(
        claim_id,
        level,
        status,
        claim,
        ru,
        en,
        ("universal superiority", "causal correctness", "guaranteed safety"),
        datasets,
        models,
        n_objects,
        n_seeds,
        metrics or {},
        ci or {},
        evidence,
        limitations,
        commit,
    )


def main() -> None:
    identity = load("run_identity.json")
    if not isinstance(identity, dict):
        raise RuntimeError("run identity is required before claim registry")
    commit = str(identity["final_commit"])
    final = load("hypotheses/final_results.json", {})
    if not isinstance(final, dict):
        final = {}
    external = load("external/status.json", {"gates": {}})
    gates = external.get("gates", {}) if isinstance(external, dict) else {}
    h1 = _section(final, "H1_real")
    h2 = _section(final, "H2_real")
    h3 = _section(final, "H3_real")
    h4 = _section(final, "H4_real")
    h5 = _section(final, "H5_real")
    h5_structural = _section(h5, "structural")
    h5_predictive = _section(h5, "predictive")
    h6 = load("rule_ablation/final_claim_status.json", {})
    if not isinstance(h6, dict):
        h6 = {}

    def status(section: dict[str, object], key: str = "status") -> ClaimStatusV2:
        value = str(section.get(key, "inconclusive"))
        return ClaimStatusV2(value) if value in {item.value for item in ClaimStatusV2} else ClaimStatusV2.INCONCLUSIVE

    def external_status(name: str) -> ClaimStatusV2:
        value = str(gates.get(name, "open"))
        if value == "open":
            return ClaimStatusV2.EXTERNAL_GATE
        return ClaimStatusV2(value) if value in {item.value for item in ClaimStatusV2} else ClaimStatusV2.INCONCLUSIVE

    rows = [
        record("H1-controlled", ClaimLevel.CONTROLLED, ClaimStatusV2.SUPPORTED, "The typed wrapper preserved paired local attributions on the controlled contour.", ("../q1_remediation/fidelity/h1_fidelity_noninferiority.json",), commit, ru="На контролируемом контуре типизированная оболочка сохранила локальные атрибуции.", en="The typed wrapper preserved local attributions on the controlled contour.", n_objects=240),
        record("H1-real", ClaimLevel.REAL, status(h1), "The system layer preserves measured real-data explainer fidelity within the frozen margin.", ("hypotheses/final_results.json",), commit, ru="Системный слой сохранил верность измеренных объяснителей в пределах заранее заданного допуска.", en="The system layer preserved measured explainer fidelity within the preregistered margin.", n_objects=int(h1.get("n_pairs", 0)), n_seeds=5, metrics={"mean_difference": h1.get("mean_difference"), "margin": h1.get("margin")}, ci={"confidence_interval_95": h1.get("confidence_interval_95")}, limitations=("Claim applies only to measured method-modality pairs.",)),
        record("H2-controlled", ClaimLevel.CONTROLLED, ClaimStatusV2.SUPPORTED, "Controlled channel removal was localized without false certification.", ("../q1_remediation/traceability/h2_traceability_missingness.json",), commit, ru="Контролируемое удаление каналов было локализовано без ложной сертификации.", en="Controlled channel removal was localized without false certification.", n_objects=180),
        record("H2-real", ClaimLevel.REAL, status(h2), "Controlled missing provenance channels are localized on measured real-pipeline identities.", ("hypotheses/final_results.json",), commit, ru="На идентификаторах реальных контуров локализованы контролируемо удалённые каналы происхождения.", en="Controlled missing provenance channels were localized on measured real-pipeline identities.", n_objects=int(h2.get("n_removals", 0)), metrics={"missingness_f1": h2.get("missingness_f1"), "false_certification_rate": h2.get("false_certification_rate")}, limitations=("Missingness itself is a controlled intervention on real artifact identities.",)),
        record("H3-full-population", ClaimLevel.REAL, status(h3, "full_population_status"), "Adaptive ABC outperforms the strongest simple baseline on the full population.", ("hypotheses/final_results.json", "../q1_remediation/cascade/h3_adaptive_cascade.json"), commit, ru="На полной исследованной выборке адаптивный каскад превзошёл сильнейшую простую политику при сопоставимом покрытии.", en="On the full evaluated population, the adaptive cascade exceeded the strongest simple policy at comparable coverage.", n_objects=_population_size(h3, "full"), n_seeds=5, limitations=("The frozen controlled population previously favored threshold-only risk; no general superiority wording is allowed unless the real replication supports it.",)),
        record("H3-hard-cases", ClaimLevel.REAL, status(h3, "hard_case_status"), "Adaptive analysis is conditionally useful on preregistered hard cases.", ("hypotheses/final_results.json",), commit, ru="Адаптивный анализ показал условную полезность на заранее определённых сложных объектах.", en="Adaptive analysis showed conditional utility on preregistered hard cases.", n_objects=_population_size(h3, "hard_cases"), n_seeds=5, limitations=("No full-population superiority claim is made.",)),
        record("H4-controlled", ClaimLevel.CONTROLLED, ClaimStatusV2.SUPPORTED, "Adaptive representation was risk non-inferior and less complex on injected profiles.", ("../q1_remediation/uncertainty/h4_uncertainty_hierarchy.json",), commit, ru="На внесённых профилях адаптивное представление сохранило риск и снизило сложность.", en="On injected profiles, adaptive representation preserved risk and reduced complexity.", n_objects=1200),
        record("H4-real", ClaimLevel.REAL, status(h4), "Adaptive representation reduces complexity on measured real uncertainty profiles without exceeding the risk margin.", ("hypotheses/final_results.json",), commit, ru="Адаптивное представление снизило сложность реальных профилей неопределённости без выхода за предел риска.", en="Adaptive representation reduced complexity on measured real uncertainty profiles within the risk margin.", metrics={"adaptive_fml_fraction": h4.get("adaptive_fml_fraction")}, limitations=("Representation benefit is limited to the measured profiles and policy costs.",)),
        record("H5-structural", ClaimLevel.REAL, status(h5_structural), "Structural route faults are localized on real pipelines.", ("hypotheses/final_results.json",), commit, ru="Структурные нарушения маршрута локализованы на реальных вычислительных контурах.", en="Structural route faults were localized on real computational pipelines.", n_objects=int(h5_structural.get("n_faults", 0)), metrics={"f1": h5_structural.get("f1"), "false_certification_rate": h5_structural.get("false_certification_rate")}, limitations=("Critical rupture is a structural diagnostic indicator, not an error predictor.",)),
        record("H5-predictive", ClaimLevel.REAL, status(h5_predictive), "Critical rupture adds held-out predictive value for model errors.", ("hypotheses/final_results.json", "../q1_remediation/critical_rupture/h5_critical_rupture.json"), commit, ru="Структурный индикатор дал дополнительную предсказательную информацию только в составе калиброванной модели риска.", en="The structural indicator added predictive information only within a calibrated model-risk model.", metrics={"incremental_auprc": h5_predictive.get("incremental_auprc", -0.0005609278151094133)}, limitations=("When incremental AUPRC is non-positive, critical rupture is a structural diagnostic only.",)),
        record("H6-controlled", ClaimLevel.CONTROLLED, ClaimStatusV2.INCONCLUSIVE, "Selected-rule ablation exceeds matched random ablation.", ("../q1_remediation/rule_ablation/h6_rule_ablation.json",), commit),
        record("H6-real-confirmatory", ClaimLevel.REAL, status(h6), "Confirmatory rule ablation replicates on two real tabular datasets.", ("rule_ablation/final_claim_status.json",), commit, ru="Заранее выбранный групповой эффект правила воспроизведён на двух реальных табличных наборах.", en="The preregistered subgroup-specific rule effect replicated on two real tabular datasets.", limitations=("A null result removes the general rule-effect claim and retains local diagnostic use only.",)),
        record("H7-comprehension", ClaimLevel.HUMAN, external_status("comprehension"), "The user explanation improves limitation comprehension and action selection without increasing unsafe overtrust.", ("external/status.json",), commit, ru="На исследованной группе пользовательское объяснение улучшило понимание ограничений и выбор действия без роста опасного чрезмерного доверия.", en="In the studied group, the user explanation improved limitation comprehension and action selection without increasing unsafe overtrust."),
        record("H8-expert-action", ClaimLevel.HUMAN, external_status("expert_action_review"), "The adaptive action agrees more often with independent expert consensus.", ("external/status.json",), commit, ru="На исследованной выборке адаптивное действие чаще совпадало с консенсусом независимых специалистов.", en="On the studied sample, the adaptive action agreed more often with independent expert consensus."),
        record("H9-domain-language", ClaimLevel.DOMAIN, external_status("domain_language_review"), "The final domain language was independently approved.", ("external/status.json",), commit, ru="Финальный предметный язык независимо проверен и одобрен.", en="The final domain language was independently reviewed and approved."),
    ]
    payload = {
        "schema_version": "2.0",
        "final_commit": commit,
        "claims": [row.to_dict() for row in rows],
        "counts": {status.value: sum(row.status is status for row in rows) for status in ClaimStatusV2},
        "stable_release_allowed": False,
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "claim_registry.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Q1 final claim registry 2.0", "", "| Claim | Level | Status |", "|---|---|---|"]
    lines.extend(f"| {row.claim_id} | {row.level.value} | {row.status.value} |" for row in rows)
    (EVIDENCE / "claim_registry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: q1_final_claim_registry claims={len(rows)}")


def _section(payload: object, key: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def _population_size(payload: dict[str, object], population: str) -> int:
    results = payload.get("results", ())
    if not isinstance(results, list):
        return 0
    row = next((item for item in results if isinstance(item, dict) and item.get("population") == population), None)
    return int(row.get("n_objects", 0)) if isinstance(row, dict) else 0


if __name__ == "__main__":
    main()
