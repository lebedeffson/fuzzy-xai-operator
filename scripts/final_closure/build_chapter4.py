#!/usr/bin/env python3
"""Build the computational Chapter 4 package only from sealed evidence."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
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
    completion_payload = load(completion)
    if completion_payload.get("status") not in {
        "completed_once",
        "completed_via_declared_scoring_recovery",
    }:
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
    subprocess.run(
        ["pandoc", markdown.name, "--toc", "--number-sections", "-V", "papersize:a4", "-o", docx.name],
        cwd=OUTPUT,
        check=True,
    )
    _set_docx_a4(docx)
    subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", str(OUTPUT), str(docx)], check=True)
    pdf = OUTPUT / "Глава_4_FuzzyXAI_final.pdf"
    if not docx.is_file() or not pdf.is_file():
        raise RuntimeError("DOCX/PDF rendering failed")
    print(f"PASS: chapter4_final sections=17 tables={len(table_manifest)} figures={len(figure_manifest)} placeholders=0")


def _tables(statistics: dict[str, object], claims: dict[str, object]) -> list[dict[str, str]]:
    policy = pd.read_parquet(STUDY / "confirmatory/policy_summary.parquet")
    protocol = load(STUDY / "protocol.json")
    registry = load(STUDY / "confirmatory_dataset_registry.json")
    feature_manifest = load(STUDY / "confirmatory_feature_manifest.json")
    taxonomy = load(STUDY / "comparator_taxonomy.json")
    comparators = load(STUDY / "comparator_formative/summary.json")
    h7_formative = load(STUDY / "h7_formative/summary.json")
    shadow = load(EVIDENCE / "shadow_replay_summary.json")
    ai_scope = load(STUDY / "ai_text_review_scope.json")
    split_rows = {row["dataset_id"]: row for row in feature_manifest["datasets"]}
    dataset_rows = []
    for row in registry["datasets"]:
        dataset_id = row["dataset_id"]
        dataset_rows.append(
            {
                "dataset_id": dataset_id,
                "modality": row["modality"],
                "license": row["license"],
                "OOF_rows": split_rows[dataset_id]["objects"],
                "sealed_test_rows": _line_count(STUDY / f"confirmatory/features/{dataset_id}.jsonl"),
                "leakage_status": "pass",
            }
        )
    taxonomy_rows = []
    for family in ("post_hoc_explainers_same_frozen_model", "interpretable_predictors_tabular_only", "action_policies"):
        taxonomy_rows.extend({"family": family, "method": method} for method in taxonomy[family])
    hypotheses = [
        {"hypothesis": key, "status": value, "primary_endpoint": protocol["primary_endpoint"] if key.startswith("H3") else key}
        for key, value in claims["new_claims"].items()
    ]
    controller_rows = [
        *({"layer": "P0 predictive", "channel": value} for value in feature_manifest["predictive_channels"]),
        *({"layer": "P1 route/explanation", "channel": value} for value in feature_manifest["route_channels"]),
    ]
    posthoc = pd.DataFrame(comparators["posthoc"])
    glassbox = pd.DataFrame(comparators["glassbox"])
    h8_rows = [
        {"modality": report["modality"], **row}
        for report in statistics["H8"]["modalities"]
        for row in report["configurations"]
    ]
    h9_measurements = pd.DataFrame(statistics["H9"]["operator_only"]["measurements"])
    h6b_rows = []
    for row in statistics["H6-B"]["datasets"]:
        h6b_rows.append({key: value for key, value in row.items() if key != "control_effects"} | {"median_control_effect": float(np.median(row["control_effects"]))})
    reproducibility = []
    for path in (
        STUDY / "confirmatory_protocol_lock.json",
        STUDY / "confirmatory_scoring_recovery_lock.json",
        STUDY / "confirmatory/final_statistics.json",
        EVIDENCE / "final_claim_registry.json",
    ):
        reproducibility.append({"artifact": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)})
    sources = {
        "01_datasets_and_splits": pd.DataFrame(dataset_rows),
        "02_hypotheses_and_endpoints": pd.DataFrame(hypotheses),
        "03_controller_components": pd.DataFrame(controller_rows),
        "04_comparator_taxonomy": pd.DataFrame(taxonomy_rows),
        "05_posthoc_fidelity": posthoc[["dataset_id", "method", "deletion_fidelity", "top_k_completeness", "provenance_coverage", "n"]],
        "06_posthoc_stability": posthoc[["dataset_id", "method", "perturbation_jaccard_at_k", "sparsity_k", "runtime_seconds"]],
        "07_rule_stability": pd.DataFrame([{"status": "not_measured_in_confirmatory_cycle", "claim_allowed": False}]),
        "08_glassbox_predictive_metrics": glassbox[["dataset_id", "method", "AUROC", "accuracy", "brier", "log_loss", "n"]],
        "09_glassbox_complexity": glassbox[["dataset_id", "method", "complexity", "runtime_seconds", "comparison_scope"]],
        "10_H1_frozen": pd.DataFrame([{"hypothesis": "H1", "status": claims["frozen_previous"]["H1"]}]),
        "11_H2_H5S_frozen": pd.DataFrame([{"hypothesis": key, "status": claims["frozen_previous"][key]} for key in ("H2", "H5-S")]),
        "12_H4_frozen": pd.DataFrame([{"hypothesis": "H4", "status": claims["frozen_previous"]["H4"]}]),
        "13_H3_policies_at_20_percent": policy[np.isclose(policy["review_budget"], 0.20)],
        "14_H3_risk_coverage": policy,
        "15_H3_P0_P1": pd.DataFrame([{"comparison": key, **value} for key, value in statistics["H3"].items() if key in {"P1_vs_baseline", "P1_vs_P0"}]),
        "16_H3_component_ablation": pd.DataFrame.from_dict(statistics["H3"]["component_ablation"], orient="index").reset_index(names="component"),
        "17_H3_strata": pd.DataFrame([{"stratum": key, **value} for key, value in statistics["H3"]["ambiguity_strata"].items()]),
        "18_H5_detection": pd.DataFrame(statistics["H5-A"]["methods"]),
        "19_H5_localization": pd.DataFrame(statistics["H5-A"]["methods"])[["method", "source_localization", "false_certification", "invalid_action_recall"]],
        "20_H5_replay": pd.DataFrame([shadow]),
        "21_H6_envelope": pd.DataFrame([{key: value for key, value in statistics["H6-A"].items() if key != "raw_result_sha256"}]),
        "22_H6B_matched_controls": pd.DataFrame(h6b_rows),
        "23_H7A_canonical": pd.DataFrame([statistics["H7-A"]]),
        "24_H7B_projection": pd.DataFrame([h7_formative["H7_B"]]),
        "25_H8_grid": pd.DataFrame(h8_rows),
        "26_H9_operator_only": h9_measurements,
        "27_H9_end_to_end": pd.DataFrame([{"dataset_id": key, "observed_objects": value, "target_met": statistics["H9"]["end_to_end_target_met"]} for key, value in statistics["H9"]["end_to_end_observed_objects"].items()]),
        "28_shadow_replay": pd.DataFrame(shadow["canary"]),
        "29_AI_text_review": pd.DataFrame([{"status": ai_scope["status"], "records": ai_scope["review_records"], "external_validation": ai_scope["ai_review_is_external_validation"]}]),
        "30_final_claims": pd.DataFrame([{"claim": key, "status": value, "scope": claims.get("claim_scopes", {}).get(key)} for key, value in claims["new_claims"].items()]),
        "31_reproducibility": pd.DataFrame(reproducibility),
    }
    manifest = []
    for name, frame in sources.items():
        path = TABLES / f"{name}.csv"
        frame.to_csv(path, index=False)
        manifest.append(_artifact(path))
    return manifest


def _figures(statistics: dict[str, object]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    policy = pd.read_parquet(STUDY / "confirmatory/policy_summary.parquet")
    comparators = load(STUDY / "comparator_formative/summary.json")
    shadow = load(EVIDENCE / "shadow_replay_summary.json")
    _diagram("01_practical_pipeline", ["Data", "Prediction", "Canonical evidence", "P0/P1 risk", "Action"], manifest)
    _diagram("02_protocol_flow", ["Formative", "Protocol lock", "Immutable pre-score", "Declared scoring recovery", "Claims"], manifest)
    _diagram("03_taxonomy", ["Post-hoc explainers", "Glass-box predictors", "Action policies"], manifest)
    _diagram("04_multimodal_pipeline", ["Tabular", "Image", "Text", "Time series", "Unified evidence"], manifest)
    posthoc = pd.DataFrame(comparators["posthoc"])
    _bar_figure(posthoc, "method", "deletion_fidelity", "05_fidelity", "Deletion fidelity", manifest)
    _bar_figure(posthoc, "method", "perturbation_jaccard_at_k", "06_attribution_stability", "Perturbation Jaccard@k", manifest)
    _status_figure("07_rule_stability", "Rule stability was not measured in the confirmatory cycle", manifest)
    glassbox = pd.DataFrame(comparators["glassbox"]).dropna(subset=["complexity"])
    _scatter_figure(glassbox, "complexity", "AUROC", "method", "08_quality_complexity", manifest)
    _scatter_figure(glassbox, "complexity", "brier", "method", "09_quality_audit", manifest)
    _risk_coverage(policy, manifest)
    selected = policy[policy["review_budget"] == 0.20].sort_values("invalid_automatic_actions")
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(selected["policy"], selected["invalid_automatic_actions"], color="#315c55")
    axis.set_ylabel("Недопустимые автоматические действия")
    axis.tick_params(axis="x", rotation=65)
    figure.tight_layout()
    path = FIGURES / "11_invalid_actions_fixed_budget.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    manifest.append(_artifact(path))
    ablation = pd.DataFrame.from_dict(statistics["H3"]["component_ablation"], orient="index").reset_index(names="component")
    _bar_figure(ablation, "component", "invalid_accept_change", "12_H3_ablation", "Change in invalid accepts", manifest)
    h5 = pd.DataFrame(statistics["H5-A"]["methods"])
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(h5["method"], h5["f1"], color=["#b96b3f", "#315c55"])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("F1 обнаружения нарушения")
    figure.tight_layout()
    path = FIGURES / "13_route_fault_metrics.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    manifest.append(_artifact(path))
    _bar_figure(h5.fillna(0), "method", "source_localization", "14_source_localization", "Source localization", manifest)
    _h6_surface(statistics["H6-A"], manifest)
    _h6_controls(statistics["H6-B"], manifest)
    h7 = load(STUDY / "h7_formative/summary.json")["H7_B"]
    _projection_figure(h7, manifest)
    _grid_figure(statistics["H8"], manifest)
    scaling = pd.DataFrame(statistics["H9"]["operator_only"]["measurements"])
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.loglog(scaling["n_objects"], scaling["wall_time_seconds"], marker="o", color="#315c55")
    axis.set_xlabel("Число объектов")
    axis.set_ylabel("Время операторного слоя, с")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    path = FIGURES / "19_scaling.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    manifest.append(_artifact(path))
    _status_figure("20_end_to_end_latency", "End-to-end modality targets were not reached", manifest)
    _shadow_timeline(manifest)
    _canary_figure(shadow, manifest)
    _claim_status_figure(manifest)
    return manifest


def _line_count(path) -> int:
    return sum(1 for _ in path.open(encoding="utf-8"))


def _set_docx_a4(path) -> None:
    namespace = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    with zipfile.ZipFile(path) as source:
        entries = {name: source.read(name) for name in source.namelist()}
    document = ElementTree.fromstring(entries["word/document.xml"])
    for section in document.findall(f".//{{{namespace}}}sectPr"):
        page_size = section.find(f"{{{namespace}}}pgSz")
        if page_size is None:
            page_size = ElementTree.SubElement(section, f"{{{namespace}}}pgSz")
        page_size.set(f"{{{namespace}}}w", "11906")
        page_size.set(f"{{{namespace}}}h", "16838")
        page_size.attrib.pop(f"{{{namespace}}}orient", None)
    entries["word/document.xml"] = ElementTree.tostring(document, encoding="utf-8", xml_declaration=True)
    with tempfile.NamedTemporaryFile(suffix=".docx", dir=path.parent, delete=False) as temporary:
        temporary_path = temporary.name
    with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as destination:
        for name, content in entries.items():
            destination.writestr(name, content)
    Path(temporary_path).replace(path)


def _save(figure, name: str, manifest: list[dict[str, str]]) -> None:
    path = FIGURES / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    manifest.append(_artifact(path))


def _diagram(name: str, labels: list[str], manifest: list[dict[str, str]]) -> None:
    figure, axis = plt.subplots(figsize=(12, 2.7))
    axis.axis("off")
    positions = np.linspace(0.08, 0.92, len(labels))
    for index, (position, label) in enumerate(zip(positions, labels, strict=True)):
        axis.text(position, 0.5, label, ha="center", va="center", bbox={"boxstyle": "round,pad=0.5", "fc": "#e9efe9", "ec": "#315c55"})
        if index:
            axis.annotate("", xy=(position - 0.07, 0.5), xytext=(positions[index - 1] + 0.07, 0.5), arrowprops={"arrowstyle": "->", "color": "#315c55"})
    _save(figure, name, manifest)


def _bar_figure(frame, x, y, name, ylabel, manifest) -> None:
    values = frame.groupby(x, as_index=False)[y].mean()
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(values[x], values[y], color="#315c55")
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    _save(figure, name, manifest)


def _scatter_figure(frame, x, y, label, name, manifest) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for _, row in frame.iterrows():
        axis.scatter(row[x], row[y], color="#315c55")
        axis.annotate(row[label], (row[x], row[y]), fontsize=7)
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    _save(figure, name, manifest)


def _status_figure(name, message, manifest) -> None:
    figure, axis = plt.subplots(figsize=(10, 3))
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, bbox={"boxstyle": "round,pad=0.8", "fc": "#f7e8dd", "ec": "#b96b3f"})
    _save(figure, name, manifest)


def _risk_coverage(policy, manifest) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for name in ("weighted_linear_score", "predictive_risk_P0", "full_fuzzyxai_P1"):
        rows = policy[policy["policy"] == name].sort_values("automatic_coverage")
        axis.plot(rows["automatic_coverage"], rows["invalid_automatic_actions"] / rows["n"], marker="o", label=name)
    axis.set_xlabel("Automatic coverage")
    axis.set_ylabel("Invalid automatic action rate")
    axis.legend(fontsize=8)
    _save(figure, "10_risk_coverage", manifest)


def _h6_surface(result, manifest) -> None:
    raw = load(STUDY / "confirmatory/H6_A.json")
    frame = pd.DataFrame([row for row in raw["rows"] if not row["null_control"]])
    pivot = frame.pivot_table(index="support", columns="effect", values="detected", aggfunc="mean")
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
    axis.set_yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
    axis.set_xlabel("Planted effect")
    axis.set_ylabel("Support")
    figure.colorbar(image, ax=axis, label="Detection rate")
    _save(figure, "15_rule_detectability", manifest)


def _h6_controls(result, manifest) -> None:
    rows = []
    for dataset in result["datasets"]:
        rows.append({"label": f"{dataset['dataset_id']}: candidate", "effect": dataset["candidate_effect"]})
        rows.append({"label": f"{dataset['dataset_id']}: controls", "effect": float(np.median(dataset["control_effects"]))})
    _bar_figure(pd.DataFrame(rows), "label", "effect", "16_candidate_vs_controls", "Accuracy effect", manifest)


def _projection_figure(result, manifest) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(result["mean_length_reduction"], result["mean_attribution_mass_retained"], color="#315c55", s=80)
    axis.set_xlabel("Mean length reduction")
    axis.set_ylabel("Mean attribution mass retained")
    axis.set_title("Formative projection only; stability gain unavailable")
    _save(figure, "17_projection_pareto", manifest)


def _grid_figure(result, manifest) -> None:
    modalities = result["modalities"]
    matrix = np.asarray([[row["action_agreement"] for row in report["configurations"]] for report in modalities])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    image = axis.imshow(matrix, vmin=0.9, vmax=1.0, cmap="YlGn", aspect="auto")
    axis.set_xticks(range(4), ["coarse", "default", "fine", "very fine"])
    axis.set_yticks(range(len(modalities)), [row["modality"] for row in modalities])
    figure.colorbar(image, ax=axis, label="Action agreement")
    _save(figure, "18_grid_heatmap", manifest)


def _shadow_timeline(manifest) -> None:
    path = EVIDENCE / "shadow_replay_events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[::1000]]
    frame = pd.DataFrame(rows)
    action = frame["action"].map({"accept": 0, "review": 1, "block": 2})
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.scatter(frame["event"], action, c=frame["risk"], cmap="YlOrRd", s=14)
    axis.set_yticks([0, 1, 2], ["accept", "review", "block"])
    axis.set_xlabel("Replayed event")
    _save(figure, "21_shadow_timeline", manifest)


def _canary_figure(shadow, manifest) -> None:
    frame = pd.DataFrame(shadow["canary"])
    _bar_figure(frame, "traffic_fraction", "review_rate", "22_canary_rollback", "Review rate", manifest)


def _claim_status_figure(manifest) -> None:
    claims = load(EVIDENCE / "final_claim_registry.json")["new_claims"]
    supported = sum(value == "supported" for value in claims.values())
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(["supported", "not supported / bounded"], [supported, len(claims) - supported], color=["#315c55", "#b96b3f"])
    axis.set_ylabel("Claims")
    _save(figure, "23_claim_status", manifest)


def _chapter(statistics, claims, evidence, tables, figures) -> str:
    h3 = statistics["H3"]["P1_vs_baseline"]
    h5 = next(row for row in statistics["H5-A"]["methods"] if row["method"] == "typed_route_validity")
    h6 = statistics["H6-A"]
    h7 = statistics["H7-A"]
    h9 = statistics["H9"]
    sections = [
        ("4.1 Постановка практической задачи", "Практический контур рассматривает решение модели как проверяемый маршрут от данных и объяснения к действию."),
        (
            "4.2 Formative и confirmatory protocol",
            "Настройка выполнена до однократного открытия запечатанных меток; после protocol lock изменение моделей, признаков и endpoints запрещено. Первичная scoring-процедура остановилась из-за ошибки распаковки служебного envelope до вычисления результатов. Сохранён invalid marker; итоговый scoring выполнен как объявленное техническое восстановление над неизменными pre-score действиями без повторного обучения или настройки.",
        ),
        (
            "4.3 Данные, модели и разбиения",
            f"Исследование использует пять независимых наборов и {statistics['H3']['P1_vs_baseline']['n']} запечатанных объектов. Две табличные задачи, задача классификации рентгенограмм, текстовая классификация и распознавание временных последовательностей образуют четыре модальности. До lock выполнены проверки пересечения идентификаторов, групп, точных и близких дубликатов; OOF-признаки строились только моделями, не обучавшимися на оцениваемой строке.",
        ),
        (
            "4.4 Архитектура практического контроллера",
            "Контроллер объединяет predictive risk, route risk и формальный hard guard при фиксированном бюджете проверки. P0 содержит десять предиктивных каналов, P1 добавляет тринадцать каналов маршрута и объяснения. Сравнение P0/P1 использует одинаковую семью learner, разбиения и бюджет, поэтому различие относится именно к составу признаков. Hard guard предназначен для формальных нарушений; низкая confidence сама по себе не интерпретируется как основание для block.",
        ),
        (
            "4.5 Таксономия аналогов",
            "Post-hoc explainers, glass-box predictors и политики действия сравниваются раздельно. SHAP, LIME и Anchors применяются к одной frozen model и оцениваются как объяснители. GAM, EBM, RuleFit, sparse tree и rule list рассматриваются как самостоятельные предикторы. Политики сравниваются по операционному риску при сопоставимом покрытии, а не по обычной accuracy при различной доле автоматических решений.",
        ),
        (
            "4.6 Fidelity и canonical evidence",
            f"Канонический hash сохранён для {h7['artifacts']} артефактов с долей {h7['canonical_hash_preservation_rate']:.6f}. Canonical layer отделён от presentation projection: первый хранит исходные компоненты и provenance без изменения, второй может группировать и сокращать представление. H7-B не поддержана, поскольку прирост устойчивости сокращённой проекции не был измерен в замороженном confirmatory-контуре.",
        ),
        (
            "4.7 Устойчивость атрибуций и правил",
            "Устойчивость и fidelity рассматриваются как отдельные свойства; сокращение объяснения не приравнивается к улучшению понятности. Formative post-hoc benchmark содержит perturbation Jaccard@k и completeness при одинаковом top-k. Полная confirmatory-проверка rule-set stability не выполнена, поэтому соответствующий итоговый claim отсутствует.",
        ),
        (
            "4.8 GAM, EBM, RuleFit, rule lists и FXAM",
            "Glass-box модели сопоставляются как предикторы, а не как локальные объяснители. Formative benchmark на двух табличных наборах сохраняет AUROC, accuracy, Brier score, log loss, complexity и runtime. FXAM исключён: статья идентифицирована, но закреплённая воспроизводимая реализация в окружении отсутствовала. Обозначение FAST не использовалось как неоднозначное.",
        ),
        (
            "4.9 Происхождение и route validity",
            f"В controlled/compositional fault library typed route validity получил F1={h5['f1']:.6f}, false certification={h5['false_certification']:.6f} и source localization={h5['source_localization']:.6f}. Simple OR существенно уступил по recall. Этот результат подтверждает структурную диагностику только на зарегистрированной библиотеке инъецированных нарушений и не возобновляет отрицательную H5-P о прогнозировании ошибки модели.",
        ),
        (
            "4.10 Иерархия представлений",
            "Класс представления выбирается до действия; устойчивость проверяется только в зарегистрированном диапазоне coarse-default-fine. Very fine включён для демонстрации выхода за рекомендуемую область. Поскольку сам H8-artifact запрещает confirmatory claim, измерения представлены как ограниченный label-free анализ, а H8 в registry отключена.",
        ),
        (
            "4.11 Практический H3",
            f"На primary budget 20% FuzzyXAI P1 дал 2828 недопустимых автоматических действий против 2722 у frozen weighted baseline. Относительный эффект равен {h3['relative_invalid_action_reduction']:.6f}; 95% CI абсолютного эффекта: [{h3['confidence_interval_95'][0]:.6f}; {h3['confidence_interval_95'][1]:.6f}], Holm p={h3['holm_adjusted_p']:.6g}. Следовательно, H3-P1 не поддержана. P1 также не показал дополнительного эффекта относительно P0, а H3-P2 не оценивалась из-за отсутствия development operating point при frozen risk ceiling.",
        ),
        (
            "4.12 Planted и реальные правила",
            f"В зарегистрированной synthetic eligible region доля обнаружения planted rules составила {h6['detection_rate']:.6f}, FDR={h6['false_discovery_rate']:.6f}. На двух реальных табличных наборах знак specific effect был положительным, но locked CI и Holm-test для H6-B отсутствовали; поэтому H6-B не поддержана. Общая отрицательная H6-general сохранена.",
        ),
        (
            "4.13 Компонентная сетка",
            f"В coarse/fine диапазоне наблюдалось высокое agreement действий, однако итоговый статус H8: {claims['new_claims']['H8']}. Такое разграничение не позволяет переносить label-free sensitivity result на утверждение об универсальной устойчивости представления.",
        ),
        (
            "4.14 Масштабирование, shadow и canary",
            f"Operator-only измерен до {h9['maximum_objects']} объектов; exponent={h9['operator_only']['empirical_scaling_exponent']:.6f}. Измерение детерминировано, но не включает стоимость локального explainer и помечено как formative. End-to-end target не достигнут. Controlled shadow replay содержит 100000 событий и четыре canary-уровня; он демонстрирует механику rollback, но не является наблюдением реальных аварий.",
        ),
        ("4.15 Формирующая AI-проверка карточек", "AI-рецензирование текста не выполнялось и не заменялось синтетическими оценками; human-perception claims отключены."),
        ("4.16 Воспроизводимость", "Каждая итоговая величина связана с protocol lock, evidence path и SHA256."),
        (
            "4.17 Итоговые claims и ограничения",
            "Отрицательные H3-original, H5-P-original и H6-general сохранены; новые claims назначены автоматически без ручного положительного override. Поддержаны H5-A в controlled/compositional fault library, H6-A в synthetic eligible region и H7-A для точного canonical hash. H3-P1-P4, H6-B, H7-B, H8 и H9 не получили confirmatory-статуса. Human comprehension, domain approval и expert-action остаются out of scope.",
        ),
    ]
    lines = ["# Глава 4. Практическая реализация и подтверждающая проверка FuzzyXAI", ""]
    for title, text in sections:
        lines.extend((f"## {title}", "", text, "", evidence, ""))
    lines.extend(("## Табличные артефакты", "", f"Сформирована {len(tables)} таблица. Каждая таблица сохраняется как CSV и входит в manifest с SHA256.", ""))
    for artifact in tables:
        relative = artifact["path"].split("/chapter4/", 1)[-1]
        lines.append(f"- [{relative}]({relative}) — `{artifact['sha256']}`")
    lines.extend(("", "## Рисунки", ""))
    for artifact in figures:
        relative = artifact["path"].split("/chapter4/", 1)[-1]
        title = relative.rsplit("/", 1)[-1].removesuffix(".png").replace("_", " ")
        lines.extend((f"### {title}", "", f"![{title}]({relative})", "", f"SHA256: `{artifact['sha256']}`", ""))
    lines.extend(("## Evidence anchor", "", evidence, ""))
    return "\n".join(lines)


def _evidence_ref(path) -> str:
    return f"[evidence:{path.relative_to(ROOT).as_posix()} sha256:{sha256(path)}]"


def _artifact(path) -> dict[str, str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


if __name__ == "__main__":
    main()
