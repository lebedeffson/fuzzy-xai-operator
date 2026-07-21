#!/usr/bin/env python3
"""Build the computational Chapter 4 package only from sealed evidence."""

from __future__ import annotations

import re
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

from common import EVIDENCE, ROOT, STUDY, load, sha256, write


OUTPUT = ROOT / "dissertation_artifacts/final_one_zip/chapter4"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"


def main() -> None:
    completion = STUDY / "confirmatory_completion_marker.json"
    statistics_path = STUDY / "confirmatory/final_statistics.json"
    claims_path = EVIDENCE / "final_claim_registry.json"
    for path in (completion, statistics_path, claims_path):
        if not path.is_file():
            raise SystemExit(f"BLOCKED: chapter 4 requires {path.relative_to(ROOT)}")
    if load(completion).get("status") != "completed_once":
        raise SystemExit("BLOCKED: confirmatory completion marker is invalid")
    statistics, claims = load(statistics_path), load(claims_path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    table_manifest = _tables(statistics, claims)
    figure_manifest = _figures(statistics)
    evidence = _evidence_ref(statistics_path)
    chapter = _chapter(statistics, claims, evidence, table_manifest, figure_manifest)
    if re.search(r"PLACEHOLDER|TBD|TODO", chapter, flags=re.IGNORECASE):
        raise RuntimeError("final chapter contains a forbidden placeholder")
    markdown = OUTPUT / "Глава_4_FuzzyXAI_final.md"
    markdown.write_text(chapter, encoding="utf-8")
    write(OUTPUT / "chapter4_values.json", {"statistics": statistics, "source": evidence})
    write(OUTPUT / "chapter4_claims.json", claims)
    write(OUTPUT / "tables_manifest.json", table_manifest)
    write(OUTPUT / "figures_manifest.json", figure_manifest)
    docx = OUTPUT / "Глава_4_FuzzyXAI_final.docx"
    subprocess.run(["pandoc", str(markdown), "-o", str(docx)], check=True)
    subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUTPUT), str(docx)], check=True)
    pdf = OUTPUT / "Глава_4_FuzzyXAI_final.pdf"
    if not docx.is_file() or not pdf.is_file():
        raise RuntimeError("DOCX/PDF rendering failed")
    print(f"PASS: chapter4_final sections=17 tables={len(table_manifest)} figures={len(figure_manifest)} placeholders=0")


def _tables(statistics: dict[str, object], claims: dict[str, object]) -> list[dict[str, str]]:
    policy = pd.read_parquet(STUDY / "confirmatory/policy_summary.parquet")
    sources = {
        "h3_policy_summary": policy,
        "h3_component_ablation": pd.DataFrame.from_dict(statistics["H3"]["component_ablation"], orient="index").reset_index(names="component"),
        "h5_route_validity": pd.DataFrame(statistics["H5-A"]["methods"]),
        "h6b_matched_controls": pd.DataFrame(statistics["H6-B"]["datasets"]),
        "final_claims": pd.DataFrame([{"claim": key, "status": value} for key, value in claims["new_claims"].items()]),
    }
    manifest = []
    for name, frame in sources.items():
        path = TABLES / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest.append(_artifact(path))
    return manifest


def _figures(statistics: dict[str, object]) -> list[dict[str, str]]:
    manifest = []
    policy = pd.read_parquet(STUDY / "confirmatory/policy_summary.parquet")
    selected = policy[policy["review_budget"] == 0.20].sort_values("invalid_automatic_actions")
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(selected["policy"], selected["invalid_automatic_actions"], color="#315c55")
    axis.set_ylabel("Недопустимые автоматические действия")
    axis.tick_params(axis="x", rotation=65)
    figure.tight_layout()
    path = FIGURES / "h3_fixed_budget.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    manifest.append(_artifact(path))
    h5 = pd.DataFrame(statistics["H5-A"]["methods"])
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(h5["method"], h5["f1"], color=["#b96b3f", "#315c55"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("F1 обнаружения нарушения")
    figure.tight_layout()
    path = FIGURES / "h5_route_validity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    manifest.append(_artifact(path))
    scaling = pd.DataFrame(statistics["H9"]["operator_only"]["measurements"])
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.loglog(scaling["n_objects"], scaling["wall_time_seconds"], marker="o", color="#315c55")
    axis.set_xlabel("Число объектов")
    axis.set_ylabel("Время операторного слоя, с")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    path = FIGURES / "h9_scaling.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    manifest.append(_artifact(path))
    return manifest


def _chapter(statistics, claims, evidence, tables, figures) -> str:
    h3 = statistics["H3"]["P1_vs_baseline"]
    h5 = next(row for row in statistics["H5-A"]["methods"] if row["method"] == "typed_route_validity")
    h6 = statistics["H6-A"]
    h7 = statistics["H7-A"]
    h9 = statistics["H9"]
    sections = [
        ("4.1 Постановка практической задачи", "Практический контур рассматривает решение модели как проверяемый маршрут от данных и объяснения к действию."),
        ("4.2 Formative и confirmatory protocol", "Настройка выполнена до однократного открытия запечатанных меток; после protocol lock изменение моделей, признаков и endpoints запрещено."),
        ("4.3 Данные, модели и разбиения", f"Исследование использует пять независимых наборов и {statistics['H3']['P1_vs_baseline']['n']} запечатанных объектов."),
        ("4.4 Архитектура практического контроллера", "Контроллер объединяет predictive risk, route risk и формальный hard guard при фиксированном бюджете проверки."),
        ("4.5 Таксономия аналогов", "Post-hoc explainers, glass-box predictors и политики действия сравниваются раздельно."),
        ("4.6 Fidelity и canonical evidence", f"Канонический hash сохранён для {h7['artifacts']} артефактов с долей {h7['canonical_hash_preservation_rate']:.6f}."),
        ("4.7 Устойчивость атрибуций и правил", "Устойчивость и fidelity рассматриваются как отдельные свойства; сокращение объяснения не приравнивается к улучшению понятности."),
        ("4.8 GAM, EBM, RuleFit, rule lists и FXAM", "Glass-box модели сопоставляются как предикторы, а не как локальные объяснители; FXAM исключён при отсутствии закреплённой воспроизводимой реализации."),
        ("4.9 Происхождение и route validity", f"Typed route validity получил F1={h5['f1']:.6f}, false certification={h5['false_certification']:.6f} и source localization={h5['source_localization']:.6f}."),
        ("4.10 Иерархия представлений", "Класс представления выбирается до действия; устойчивость проверяется только в зарегистрированном диапазоне сетки."),
        ("4.11 Практический H3", f"Относительное изменение недопустимых автоматических действий равно {h3['relative_invalid_action_reduction']:.6f}; 95% CI абсолютного эффекта: [{h3['confidence_interval_95'][0]:.6f}; {h3['confidence_interval_95'][1]:.6f}], Holm p={h3['holm_adjusted_p']:.6g}."),
        ("4.12 Planted и реальные правила", f"В eligible region доля обнаружения planted rules составила {h6['detection_rate']:.6f}, FDR={h6['false_discovery_rate']:.6f}."),
        ("4.13 Компонентная сетка", f"Confirmatory статус H8: {claims['new_claims']['H8']}."),
        ("4.14 Масштабирование, shadow и canary", f"Operator-only измерен до {h9['maximum_objects']} объектов; exponent={h9['operator_only']['empirical_scaling_exponent']:.6f}. End-to-end ограничение показано отдельно."),
        ("4.15 Формирующая AI-проверка карточек", "AI-рецензирование текста не выполнялось и не заменялось синтетическими оценками; human-perception claims отключены."),
        ("4.16 Воспроизводимость", "Каждая итоговая величина связана с protocol lock, evidence path и SHA256."),
        ("4.17 Итоговые claims и ограничения", "Отрицательные H3-original, H5-P-original и H6-general сохранены; новые claims назначены автоматически без ручного положительного override."),
    ]
    lines = ["# Глава 4. Практическая реализация и подтверждающая проверка FuzzyXAI", ""]
    for title, text in sections:
        lines.extend((f"## {title}", "", text, "", evidence, ""))
    lines.extend(("## Артефакты", "", f"Таблицы: {len(tables)}. Рисунки: {len(figures)}.", "", evidence, ""))
    return "\n".join(lines)


def _evidence_ref(path) -> str:
    return f"[evidence:{path.relative_to(ROOT).as_posix()} sha256:{sha256(path)}]"


def _artifact(path) -> dict[str, str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


if __name__ == "__main__":
    main()
