#!/usr/bin/env python3
"""Build claim registry 3.0 while preserving positive, null and external boundaries."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rows = [
        _claim("H1-fidelity", "supported", "Системный слой сохранил верность измеренных объяснителей в пределах frozen допуска.", "Универсальное превосходство качества объяснений.", ["release_evidence/q1_final/hypotheses/final_results.json"], "computational"),
        _claim("H2-provenance-localization", "supported", "Контролируемо недоступные каналы происхождения локализованы на измеренных вычислительных контурах.", "Предметная полнота происхождения подтверждена экспертами.", ["release_evidence/q1_final/hypotheses/final_results.json"], "computational"),
        _claim("H3-selective-policy", "not_supported", "", "Адаптивная политика превосходит сильнейший baseline на полной выборке или сложных объектах.", ["release_evidence/q1_final/claim_registry.json"], "computational"),
        _claim("H4-adaptive-hierarchy", "supported", "Адаптивное представление снизило сложность измеренных профилей без выхода за установленный риск-допуск.", "Иерархия всегда улучшает решение.", ["release_evidence/q1_final/hypotheses/final_results.json"], "computational"),
        _claim("H5-structural", "supported", "Структурные нарушения маршрута локализованы на измеренных вычислительных контурах.", "Структурный индикатор предсказывает ошибку или безопасность.", ["release_evidence/q1_final/hypotheses/final_results.json"], "structural"),
        _claim("H5-predictive", "not_supported", "", "Критический разрыв добавляет held-out predictive value для ошибок модели.", ["release_evidence/q1_final/claim_registry.json"], "predictive"),
        _claim("H6-rule-ablation", "inconclusive", "Абляция правила используется как локальная диагностика в пределах конкретного контракта.", "Важное правило всегда даёт воспроизводимый глобальный эффект.", ["release_evidence/q1_final/rule_ablation/final_claim_status.json"], "local_diagnostic"),
        _claim("H10-AI-formative", "open_external", "", "Проведена формализованная AI-предпроверка объяснений.", ["study/ai_pre_review_final/blind_batch_manifest.json"], "ai"),
        _claim("H10-AI-repeatability", "open_external", "", "AI-рецензирование является стабильным.", ["study/ai_pre_review_final/blind_batch_manifest.json"], "ai"),
        _claim("H10-AI-human-agreement", "open_external", "", "AI-предпроверка согласуется с независимыми экспертами.", ["study/ai_pre_review_final/blind_batch_manifest.json"], "human"),
        _claim("H10-critical-defect-detection", "open_external", "", "AI надёжно обнаруживает критические дефекты объяснений.", ["study/ai_pre_review_final/blind_batch_manifest.json"], "human"),
        _claim("H11-domain-language", "open_external", "", "Предметный язык независимо одобрен.", ["study/q1_final/domain_language_review"], "domain"),
        _claim("H12-comprehension", "open_external", "", "Пользователи лучше понимают объяснения.", ["study/q1_final/comprehension"], "human"),
        _claim("H13-expert-action", "open_external", "", "Предлагаемые действия согласованы с независимым экспертным консенсусом.", ["study/q1_final/expert_action_review"], "human"),
    ]
    for row in rows:
        row["final_commit"] = commit
    payload = {
        "schema_version": "3.0",
        "final_commit": commit,
        "stable_release_allowed": False,
        "claims": rows,
        "counts": {status: sum(row["status"] == status for row in rows) for status in ("supported", "not_supported", "inconclusive", "open_external", "removed")},
    }
    output = ROOT / "release_evidence/ai_pre_review_final"
    output.mkdir(parents=True, exist_ok=True)
    (output / "claim_registry_3.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Claim Registry 3.0", "", "Stable release: **BLOCKED**.", "", "| Claim | Status | Allowed wording |", "|---|---|---|"]
    lines.extend(f"| {row['claim_id']} | `{row['status']}` | {row['allowed_wording'] or 'Нет положительной формулировки'} |" for row in rows)
    (output / "claim_registry_3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"PASS: claim_registry_3 supported={payload['counts']['supported']} open_external={payload['counts']['open_external']} stable=false")


def _claim(claim_id: str, status: str, allowed: str, forbidden: str, evidence: list[str], level: str) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "status": status,
        "allowed_wording": allowed,
        "forbidden_wording": [forbidden],
        "evidence_refs": evidence,
        "confidence_interval": None,
        "dataset_scope": "frozen declared evidence only",
        "modality_scope": ["tabular", "image", "text", "timeseries"] if claim_id.startswith("H10") else ["declared in source evidence"],
        "validation_level": level,
    }


if __name__ == "__main__":
    main()
