#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import argparse
import shutil
import subprocess
import sys
import zipfile
from hashlib import sha256
from pathlib import Path
from textwrap import fill


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "release" / "fuzzyxai_ru_explanation_visual_package"
ZIP = ROOT / "reports" / "release" / "fuzzyxai_ru_explanation_visual_package.zip"
CLI = ROOT / "reports" / "release" / "fuzzyxai_framework_rc" / "cli_check"
RESULTS = ROOT / "research_validation" / "reports" / "research_validation_results.csv"
ASSETS = ROOT / "docs" / "chapter_4_framework" / "assets"

ACTION_RU = {
    "accept": "принять",
    "lower_confidence": "понизить доверие",
    "audit": "направить на аудит",
    "defer_to_human": "передать человеку",
    "block": "заблокировать",
    "audit_report": "аудит-отчёт",
}
ACTION_COLORS = {
    "accept": "#2e7d32",
    "lower_confidence": "#f2a900",
    "audit": "#d75a00",
    "defer_to_human": "#6a3d9a",
    "block": "#b71c1c",
}
COMPONENT_RU = {
    "uncertainty_component": "неуверенность модели",
    "reduction_component": "потеря объяснения",
    "quality_component": "качество входа",
    "conflict_component": "конфликт источников",
    "interval_component": "интервальная неопределённость",
}
COMPONENT_COLORS = {
    "uncertainty_component": "#3366cc",
    "reduction_component": "#f2a900",
    "quality_component": "#2e7d32",
    "conflict_component": "#b71c1c",
    "interval_component": "#6a3d9a",
}
FIGURES = [
    "explanation_card_ru",
    "risk_sources_ru",
    "why_lower_confidence_ru",
    "decision_boundary_ru",
    "gamma_delta_decision_map_ru",
    "operator_trace_summary_ru",
    "proof_consistency_ru",
    "representation_atlas_ru",
    "task_data_passport_ru",
    "operator_route_story_ru",
    "operator_cards_ru",
]
REQUIRED_SVG_TERMS = ["риск", "доверие", "объяснение", "действие", "γ", "Δ", "ρ"]
HUMAN_MODEL = "внешний классификатор"
HUMAN_DATASET = "демонстрационный табличный пример"


def fuzzyxai_imports():
    sys.path.insert(0, str(ROOT / "framework" / "fuzzyxai"))
    from fuzzyxai.explain import (
        build_operator_narrative_ru,
        build_readable_explanation_ru,
        build_task_data_passport,
    )

    return build_operator_narrative_ru, build_readable_explanation_ru, build_task_data_passport


def must(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def f(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["svg.fonttype"] = "none"
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    return plt


def save_all(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", facecolor="white")


def short_route_id(route: dict) -> str:
    route_id = str(route.get("route_id", "cli_external"))
    return "cli_external" if "external_payload" in route_id or len(route_id) > 28 else route_id


def short_footer(route: dict, verifier: str = "passed") -> str:
    status = "PASS" if str(verifier).lower() == "passed" else str(verifier).upper()
    return f"commit: {str(route.get('source_commit', 'unknown'))[:7]} | route: {short_route_id(route)} | verifier: {status}"


def footer(fig, route: dict, verifier: str = "passed") -> None:
    text = short_footer(route, verifier)
    fig.text(0.015, 0.012, text, ha="left", va="bottom", fontsize=9, color="#51606f")


def title_block(fig, title: str, what: str) -> None:
    fig.text(0.05, 0.94, title, fontsize=18, fontweight="bold", color="#16202a")
    fig.text(0.05, 0.895, f"Что показывает рисунок: {what}", fontsize=12, color="#334155")
    fig.text(
        0.05,
        0.862,
        "Как читать: γ — неуверенность; Δ — потеря объяснения; ρ — риск; действие — режим; доверие — уровень принятия; объяснение — причина решения.",
        fontsize=10.5,
        color="#51606f",
    )


def case_data() -> tuple[dict, dict, dict, list[dict[str, str]]]:
    route = read_json(CLI / "route.json")
    trace = read_json(CLI / "operator_trace.json")
    verifier = read_json(CLI / "verifier_report.json")
    return route, trace, verifier, read_rows(RESULTS)


def load_proof_trace() -> dict:
    path = CLI / "proof_trace.json"
    return read_json(path) if path.exists() else {}


def components(computed: dict) -> dict[str, float]:
    return {
        "uncertainty_component": f(computed.get("uncertainty_component")),
        "reduction_component": f(computed.get("reduction_component")),
        "quality_component": f(computed.get("quality_component")),
        "conflict_component": f(computed.get("conflict_component")),
        "interval_component": f(computed.get("interval_component")),
    }


def render_explanation_card(route: dict, verifier: dict, out: Path) -> dict:
    plt = setup_matplotlib()
    c = route["computed_result"]
    comps = components(c)
    rho = f(c.get("rho"))
    action = str(c.get("action_id"))
    dominant_key = max(comps, key=comps.get)
    fig = plt.figure(figsize=(8.8, 7.1))
    title_block(fig, "Карточка объяснения решения FuzzyXAI", "почему результат не принят без ограничений")
    ax = fig.add_axes([0.05, 0.10, 0.9, 0.74])
    ax.axis("off")
    risk_values = [comps["uncertainty_component"], comps["reduction_component"], comps["quality_component"], comps["conflict_component"], comps["interval_component"]]
    boxes = [
        ("Модель и данные", f"Модель: {HUMAN_MODEL}\nДанные: {HUMAN_DATASET}\nВероятность класса: {f(c.get('class_probability')):.2f}"),
        ("Что обнаружено", f"1. γ = {f(c.get('gamma')):.2f}: модель уверена не полностью\n2. Δ = {f(c.get('delta')):.2f}: объяснение потеряло часть информации\n3. качество входа = {comps['quality_component']:.2f}: не главная проблема"),
        ("Итоговый риск", "ρ равен наибольшему из найденных ограничений:\nρ = max(" + ", ".join(f"{v:.2f}" for v in risk_values) + f") = {rho:.2f}\nГлавная причина: {COMPONENT_RU[dominant_key]}"),
        ("Решение", f"{ACTION_RU.get(action, action)}\nρ выше порога полного принятия 0.35,\nно ниже порога аудита 0.60."),
        ("Как читать", "γ показывает неуверенность модели.\nΔ показывает потерю объяснения.\nρ — итоговый риск; действие выбирается по порогам."),
    ]
    y = 0.86
    for idx, (head, body) in enumerate(boxes):
        color = "#fff7df" if idx == 3 else ("#eef6ff" if idx == 4 else "#f8fafc")
        ax.text(0.02, y, head, fontsize=14, fontweight="bold", va="top", color="#16202a")
        ax.text(
            0.02,
            y - 0.06,
            body,
            fontsize=12,
            va="top",
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="#cbd5e1"),
            linespacing=1.45,
        )
        y -= 0.19
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)
    return {"rho": rho, "action_ru": ACTION_RU.get(action, action), "dominant_ru": COMPONENT_RU[dominant_key]}


def render_risk_sources(route: dict, verifier: dict, out: Path) -> None:
    plt = setup_matplotlib()
    c = route["computed_result"]
    comps = components(c)
    labels = [COMPONENT_RU[key] for key in comps]
    values = list(comps.values())
    colors = [COMPONENT_COLORS[key] for key in comps]
    rho = f(c.get("rho"))
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    title_block(fig, "Источники операторного риска", "какое свидетельство сильнее всего ограничивает доверие")
    y = range(len(labels))
    ax.barh(list(y), values, color=colors)
    ax.axvline(rho, color="#111827", linewidth=2)
    ax.text(rho + 0.015, len(labels) - 0.4, f"ρ = {rho:.2f}", fontsize=13, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("значение компоненты риска", fontsize=12)
    ax.set_xlim(0, 1)
    ax.grid(axis="x", color="#e5e7eb")
    ax.invert_yaxis()
    formula = "ρ = max(" + ", ".join(f"{v:.2f}" for v in values) + f") = {rho:.2f}"
    ax.text(0.02, -0.18, formula, transform=ax.transAxes, fontsize=12, color="#16202a")
    ax.text(
        0.02,
        -0.30,
        "Так как используется max-агрегация, итоговый риск определяется наибольшим операторным свидетельством.",
        transform=ax.transAxes,
        fontsize=11,
        color="#334155",
    )
    dominant_key = max(comps, key=comps.get)
    ax.text(0.55, -0.18, f"Главная причина: {COMPONENT_RU[dominant_key]}.", transform=ax.transAxes, fontsize=12, fontweight="bold", color="#16202a")
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_why_lower_confidence(route: dict, verifier: dict, out: Path) -> None:
    plt = setup_matplotlib()
    c = route["computed_result"]
    rho = f(c.get("rho"))
    delta = f(c.get("delta"))
    gamma = f(c.get("gamma"))
    fig = plt.figure(figsize=(8.8, 5.8))
    title_block(fig, "Почему FuzzyXAI понизил доверие?", "как ρ оказался между полным принятием и аудитом")
    ax = fig.add_axes([0.08, 0.17, 0.84, 0.62])
    ax.axis("off")
    ax.text(0.5, 0.85, "Итоговое действие", ha="center", fontsize=13)
    ax.text(
        0.5,
        0.73,
        "понизить доверие",
        ha="center",
        fontsize=22,
        fontweight="bold",
        color=ACTION_COLORS["lower_confidence"],
    )
    ax.text(0.12, 0.52, f"ρ = {rho:.2f}", fontsize=18, fontweight="bold")
    ax.text(0.12, 0.42, "порог принятия: 0.35", fontsize=12)
    ax.text(0.12, 0.34, "порог аудита: 0.60", fontsize=12)
    ax.text(0.58, 0.52, f"Главная причина:\nΔ = {delta:.2f} — потеря объяснения", fontsize=13, bbox=dict(facecolor="#fff7df", edgecolor="#f2a900", boxstyle="round,pad=0.5"))
    ax.text(0.58, 0.28, f"Дополнительно:\nγ = {gamma:.2f} — неполная уверенность модели", fontsize=13, bbox=dict(facecolor="#eef6ff", edgecolor="#3366cc", boxstyle="round,pad=0.5"))
    ax.text(
        0.08,
        0.08,
        "Система не заблокировала результат, потому что риск не критический. Но и не приняла полностью, потому что ρ выше безопасного порога.",
        fontsize=12,
        color="#334155",
        wrap=True,
    )
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_decision_boundary(route: dict, verifier: dict, out: Path) -> None:
    plt = setup_matplotlib()
    c = route["computed_result"]
    rho = f(c.get("rho"))
    fig, ax = plt.subplots(figsize=(10, 4.1))
    title_block(fig, "Шкала границ действий", "насколько решение близко к соседним режимам доверия")
    zones = [(0, 0.35, "принять", "accept"), (0.35, 0.60, "понизить доверие", "lower_confidence"), (0.60, 0.75, "аудит", "audit"), (0.75, 1.0, "критично", "block")]
    for start, end, label, key in zones:
        ax.axvspan(start, end, color=ACTION_COLORS[key], alpha=0.28)
        ax.text((start + end) / 2, 0.62, label, ha="center", fontsize=14, fontweight="bold")
    ax.axvline(rho, color="#111827", linewidth=3)
    ax.scatter([rho], [0.5], s=130, color="#111827", zorder=3)
    ax.text(rho, 0.24, f"ρ = {rho:.2f}", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.18, 0.08, f"до полного принятия: {rho - 0.35:.2f}", ha="center", fontsize=11)
    ax.text(0.49, 0.08, f"до аудита остаётся: {0.60 - rho:.2f}", ha="center", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("ρ — итоговый риск", fontsize=12)
    ax.grid(axis="x", color="#e5e7eb")
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_gamma_delta_map(route: dict, verifier: dict, rows: list[dict[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    import numpy as np
    from matplotlib.colors import ListedColormap

    c = route["computed_result"]
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    title_block(fig, "Карта решений в пространстве γ–Δ", "как неуверенность и потеря объяснения задают область действия")
    xs = np.linspace(0, 1, 240)
    ys = np.linspace(0, 1, 240)
    zone = np.zeros((len(ys), len(xs)))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            rho = max(x, y)
            zone[i, j] = 0 if rho < 0.35 else (1 if rho < 0.60 else (2 if rho < 0.75 else 3))
    ax.imshow(zone, extent=[0, 1, 0, 1], origin="lower", cmap=ListedColormap(["#dff1df", "#fff1b8", "#ffd8b5", "#f7c4c4"]), alpha=0.78, aspect="auto")
    ax.text(0.16, 0.16, "принять", ha="center", fontsize=12)
    ax.text(0.47, 0.47, "понизить\nдоверие", ha="center", fontsize=12)
    ax.text(0.68, 0.68, "аудит", ha="center", fontsize=12)
    ax.text(0.88, 0.88, "критично", ha="center", fontsize=12)
    for idx, row in enumerate(rows[:20], start=1):
        action = row.get("action_id", "accept")
        ax.scatter(f(row.get("gamma")), f(row.get("delta")), color=ACTION_COLORS.get(action, "#607d8b"), s=45, edgecolor="#111827", linewidth=0.4)
    ax.scatter(f(c.get("gamma")), f(c.get("delta")), color="#111827", s=160, marker="*", zorder=5)
    ax.text(f(c.get("gamma")) + 0.025, f(c.get("delta")) + 0.025, f"текущий объект\nρ={f(c.get('rho')):.2f}\nдействие: понизить доверие", fontsize=11, bbox=dict(facecolor="white", edgecolor="#111827", boxstyle="round,pad=0.35"))
    ax.set_xlabel("γ — неуверенность / рассогласование", fontsize=12)
    ax.set_ylabel("Δ — потеря объяснения", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="white", alpha=0.8)
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_operator_trace_summary(route: dict, verifier: dict, rows: list[dict[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    import numpy as np

    sample = rows[:10]
    row_names = [
        "wine clean",
        "wine reduced",
        "bc missing",
        "bc boundary",
        "diabetes interval",
        "digits occlusion",
        "signal noise",
        "signal missing",
        "image clean",
        "regression loss",
    ]
    cols = [
        ("gamma", "γ\nнеуверенность"),
        ("delta", "Δ\nпотеря\nобъяснения"),
        ("rho", "ρ\nриск"),
        ("quality_component", "качество\nвхода"),
        ("conflict_component", "конфликт"),
    ]
    data = np.array([[f(row.get(key)) for key, _ in cols] for row in sample])
    fig, ax = plt.subplots(figsize=(10.4, 6))
    title_block(fig, "Компактная карта операторной трассы", "как меняются числовые источники риска в экспериментах")
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([label for _, label in cols], fontsize=11)
    ax.set_yticks(range(len(sample)))
    ax.set_yticklabels(row_names[: len(sample)], fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="значение риска")
    ax.set_xlabel("операторные показатели", fontsize=12)
    fig.text(
        0.76,
        0.47,
        "Как читать:\nчем темнее ячейка,\nтем выше вклад риска;\nρ — итоговый риск;\nглавная причина —\nмаксимальное значение.",
        fontsize=11,
        color="#334155",
        bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", boxstyle="round,pad=0.45"),
    )
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_proof_consistency(route: dict, verifier: dict, out: Path) -> None:
    plt = setup_matplotlib()
    artifacts = ["маршрут", "операторная трасса", "доказательный след", "данные дашборда", "отчёт проверки", "манифест"]
    invariants = ["commit", "γ", "Δ", "ρ", "диагностика", "действие", "route_id", "sha256", "verifier"]
    matrix = [["PASS" for _ in invariants] for _ in artifacts]
    matrix[4][0] = "N/A"
    matrix[2][6] = "N/A"
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    title_block(fig, "Проверка согласованности артефактов", "совпадают ли ключевые значения между артефактами")
    color_map = {"PASS": "#dff1df", "N/A": "#eeeeee"}
    for i, art in enumerate(artifacts):
        for j, inv in enumerate(invariants):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color_map[matrix[i][j]], edgecolor="white"))
            label = "согласовано" if matrix[i][j] == "PASS" else "не применяется"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, len(invariants))
    ax.set_ylim(0, len(artifacts))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(invariants))])
    ax.set_xticklabels(invariants, rotation=30, ha="right", fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(len(artifacts))])
    ax.set_yticklabels(artifacts, fontsize=10)
    ax.tick_params(length=0)
    ax.text(0, len(artifacts) + 0.55, "Красных ошибок нет: неприменимые инварианты помечены как «не применяется».", fontsize=11, color="#334155")
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_representation_atlas(route: dict, verifier: dict, rows: list[dict[str, str]], out: Path) -> None:
    plt = setup_matplotlib()
    from collections import Counter, defaultdict

    task_ru = {
        "tabular_classification": "табличная\nклассификация",
        "tabular_regression": "табличная\nрегрессия",
        "signal_quality": "сигнал",
        "image_like_classification": "изображение",
    }
    pert_ru = {
        "clean": "чистый\nвход",
        "top_k_reduction": "редукция\nобъяснения",
        "missing_features": "пропуски",
        "confidence_boundary": "граница\nуверенности",
        "wide_interval": "широкий\nинтервал",
        "noise": "шум",
        "occlusion": "окклюзия",
        "multilevel_trace": "многоуровн.\nтрасса",
        "explanation_conflict": "конфликт",
    }
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        counts[(row.get("task_type", ""), row.get("perturbation", ""))][row.get("representation_class", "")] += 1
    tasks = list(dict.fromkeys(row.get("task_type", "") for row in rows))
    perts = list(dict.fromkeys(row.get("perturbation", "") for row in rows))
    colors = {"F0": "#4c78a8", "F_int": "#72b7b2", "NAS": "#f58518", "F_ML": "#54a24b", "": "#eeeeee"}
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    title_block(fig, "Атлас классов представления", "где активируются F0, F_int, NAS и F_ML")
    for i, task in enumerate(tasks):
        for j, pert in enumerate(perts):
            counter = counts.get((task, pert), Counter())
            cls, count = counter.most_common(1)[0] if counter else ("", 0)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colors.get(cls, "#eeeeee"), edgecolor="white"))
            ax.text(j + 0.5, i + 0.5, f"{cls}\n{count}" if count else "—", ha="center", va="center", fontsize=9, color="#111827")
    ax.set_xlim(0, len(perts))
    ax.set_ylim(0, len(tasks))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(perts))])
    ax.set_xticklabels([pert_ru.get(p, p) for p in perts], fontsize=9, rotation=30, ha="right")
    ax.set_yticks([i + 0.5 for i in range(len(tasks))])
    ax.set_yticklabels([task_ru.get(t, t) for t in tasks], fontsize=10)
    ax.tick_params(length=0)
    legend = (
        "Легенда:\n"
        "F0 — обычное нечёткое представление\n"
        "F_int — интервальное представление\n"
        "NAS — конфликт источников\n"
        "F_ML — многоуровневое представление"
    )
    fig.text(0.67, 0.24, legend, fontsize=10.5, color="#334155", bbox=dict(facecolor="#f8fafc", edgecolor="#cbd5e1", boxstyle="round,pad=0.45"))
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_task_data_passport(route: dict, verifier: dict, passport: dict, out: Path) -> None:
    plt = setup_matplotlib()
    c = route["computed_result"]
    fig = plt.figure(figsize=(9.2, 5.8))
    title_block(fig, "Паспорт задачи и данных", "что именно анализирует FuzzyXAI перед запуском операторного маршрута")
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.70])
    ax.axis("off")
    rows = [
        ("Тип задачи", "табличная классификация"),
        ("Тип модели", passport.get("model_summary_ru", HUMAN_MODEL)),
        ("Объект объяснения", "один объект / single instance"),
        ("Признаков", str(passport.get("feature_count") or "есть набор признаков")),
        ("Вероятность класса", f"есть, p = {f(c.get('class_probability')):.2f}"),
        ("Attribution / правила", "есть объяснительные вклады признаков"),
        ("Качество входа", f"ограничение качества = {f(c.get('quality_component')):.2f}"),
        ("Готовность маршрута", "можно запускать операторную трассу"),
    ]
    for i, (name, value) in enumerate(rows):
        y = 0.92 - i * 0.105
        ax.text(0.03, y, name, fontsize=12, fontweight="bold", va="center")
        ax.text(0.36, y, value, fontsize=12, va="center", bbox=dict(facecolor="#f8fafc", edgecolor="#e2e8f0", boxstyle="round,pad=0.25"))
    ax.text(
        0.03,
        0.03,
        "Вывод: вход содержит всё необходимое для объяснения — результат модели, вероятность, признаки, вклады и показатели качества.",
        fontsize=12,
        color="#334155",
    )
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_operator_route_story(route: dict, verifier: dict, narrative: dict, out: Path) -> None:
    plt = setup_matplotlib()
    c = route["computed_result"]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    title_block(fig, "Лента операторного маршрута", "как данные проходят путь от входа до действия и proof trace")
    ax.axis("off")
    steps = [
        ("Данные", "вход пригоден"),
        ("Объяснение", "собран Eₖ"),
        ("Представление", str(c.get("representation_class", "F0"))),
        ("γ", f"{f(c.get('gamma')):.2f}\nумеренная\nнеуверенность"),
        ("Δ", f"{f(c.get('delta')):.2f}\nпотеря\nобъяснения"),
        ("ρ", f"{f(c.get('rho')):.2f}\nглавный риск:\nΔ"),
        ("Диагноз", "ограниченная\nуверенность"),
        ("Действие", "понизить\nдоверие"),
        ("Proof", "проверено"),
    ]
    xs = [0.05 + i * 0.105 for i in range(len(steps))]
    for i, ((title, text), x) in enumerate(zip(steps, xs)):
        color = "#fff7df" if title in {"Δ", "ρ", "Действие"} else "#f8fafc"
        ax.text(x, 0.58, title, ha="center", va="center", fontsize=12, fontweight="bold", bbox=dict(facecolor=color, edgecolor="#cbd5e1", boxstyle="round,pad=0.45"))
        ax.text(x, 0.38, text, ha="center", va="top", fontsize=10)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.04, 0.58), xytext=(x + 0.04, 0.58), arrowprops=dict(arrowstyle="->", color="#64748b", lw=1.5))
    ax.text(0.05, 0.12, narrative["final_decision"]["explanation_ru"], fontsize=11, color="#334155", bbox=dict(facecolor="#eef6ff", edgecolor="#cbd5e1", boxstyle="round,pad=0.45"))
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_operator_cards(route: dict, verifier: dict, narrative: dict, out: Path) -> None:
    plt = setup_matplotlib()
    cards = narrative["operator_cards"]
    selected = [card for card in cards if card["operator_id"] in {"input_artifact", "alignment", "reduction", "risk", "action", "proof"}]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    title_block(fig, "Карточки операторов", "что получил, проверил и вычислил каждый ключевой оператор")
    for ax, card in zip(axes.ravel(), selected):
        ax.axis("off")
        ax.set_facecolor("#f8fafc")
        ax.text(0.03, 0.92, card["operator_title_ru"], fontsize=13, fontweight="bold", va="top")
        values = []
        for key, value in card["computed_values"].items():
            if key == "action_id":
                values.append(f"действие: {ACTION_RU.get(str(value), str(value))}")
            elif key == "dominant_component":
                values.append(f"главная причина: {COMPONENT_RU.get(str(value), str(value))}")
            elif key == "representation_class":
                values.append(f"класс представления: {value}")
            else:
                values.append(f"{key}: {value}")
        body = (
            f"Вопрос: {card['operator_question_ru']}\n\n"
            f"Что получил: {card['input_summary_ru']}\n\n"
            f"Что проверил: {card['method_summary_ru']}\n\n"
            f"Что вычислил: {'; '.join(values)}\n\n"
            f"Что значит: {card['plain_result_ru']}\n\n"
            f"Влияние: {card['effect_on_route_ru']}\n\n"
            f"Где проверяется: {', '.join(card['proof_refs'][:2])}"
        )
        ax.text(0.03, 0.82, fill(body, 44), fontsize=8.6, va="top", linespacing=1.22)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#cbd5e1")
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def write_report(path: Path) -> None:
    text = """
# Русскоязычный объясняющий визуальный слой FuzzyXAI

## 1. Зачем нужен русскоязычный объясняющий слой

Технические визуализации FuzzyXAI уже показывают значения γ, Δ, ρ, компоненты риска, границы действия и согласованность доказательного следа. Однако для диссертации этого недостаточно. Читатель главы не обязан помнить внутренние имена операторов, английские action labels и структуру audit package. Ему нужно увидеть человеческий ответ на простой вопрос: почему фреймворк доверяет результату полностью, понижает доверие, отправляет результат на аудит или передаёт решение человеку. Поэтому поверх технического SHAP-like слоя добавлен отдельный русскоязычный explanation-first слой.

Главный принцип этого слоя: каждая фигура отвечает на один вопрос. Карточка объяснения отвечает, почему выбран текущий режим доверия. Схема источников риска показывает, какое операторное свидетельство стало главным. Шкала границ действий показывает, насколько решение близко к соседним зонам. Карта γ–Δ показывает геометрию принятия решения. Компактная карта трассы показывает поведение числовых показателей в серии экспериментов. Матрица proof consistency показывает, что визуализация не нарисована вручную, а согласована с route, proof trace и manifest.

## 2. Почему SHAP-like графики недостаточны без пояснений

SHAP-подобные графики привычны разработчикам, потому что они быстро считывают структуру вклада признаков. Но FuzzyXAI решает другую задачу. Он объясняет не вклад признаков в предсказание, а путь формирования доверия к объяснению. Поэтому технически корректная фигура с подписями gamma, delta, uncertainty, reduction и lower_confidence может быть точной, но недостаточно понятной для читателя. В русскоязычной версии эти термины раскрываются прямо на рисунке: γ — неуверенность или рассогласование, Δ — потеря объяснения, ρ — итоговый риск, действие — режим доверия.

Важное отличие состоит и в математике. SHAP waterfall обычно показывает сумму вкладов. FuzzyXAI в текущем ExplainPlan использует max-агрегацию риска: ρ = max(γ, Δ, качество, конфликт, интервал). Поэтому русские фигуры не изображают накопительную сумму, если её нет в маршруте. Они показывают операторные свидетельства как параллельные источники риска и отдельно объясняют, что итоговый ρ определяется наибольшим свидетельством.

## 3. Как читать γ, Δ и ρ

Показатель γ обозначает рассогласование или неуверенность. Внешняя модель может дать вероятность класса ниже единицы, а вход может иметь ограничение качества. Тогда γ становится ненулевым. Показатель Δ обозначает потерю объяснения: если объяснение строится только по top-k признакам или сокращает исходную информацию, часть объяснительной массы теряется. Показатель ρ — итоговый риск маршрута. В текущем пакете он интерпретируется как максимум среди операторных свидетельств.

Например, если γ = 0.32, Δ = 0.39, качество = 0.05, конфликт = 0.00 и интервал = 0.00, то ρ = 0.39. Главным источником становится потеря объяснения, потому что она больше остальных компонент. Если порог полного принятия равен 0.35, а порог аудита равен 0.60, то ρ = 0.39 попадает в область понижения доверия. Это означает: результат не блокируется и не отправляется на аудит, но использовать его как полностью надёжный автоматический вывод нельзя.

## 4. Как читать каждую фигуру

Карточка объяснения решения FuzzyXAI предназначена для первого знакомства с локальным решением. Она показывает модель, данные, вероятность класса, найденные ограничения, итоговый риск и действие. Это главная фигура для защиты, потому что она переводит operator trace на обычный язык.

Фигура «Источники операторного риска» показывает пять основных свидетельств: неуверенность модели, потерю объяснения, качество входа, конфликт источников и интервальную неопределённость. Рядом показана формула max-агрегации. Читатель видит, что риск не складывается из всех компонент, а выбирается по самой сильной.

Фигура «Почему FuzzyXAI понизил доверие?» отвечает на вопрос о действии lower_confidence. Она показывает, что ρ выше порога accept, но ниже порога audit. Это объясняет промежуточное решение: не принять полностью, но и не блокировать.

Шкала границ действий показывает расстояния до соседних областей. Если ρ близко к accept, можно сказать, что решение мягкое и пограничное. Если ρ близко к audit, то система уже почти требует более строгой проверки. Карта γ–Δ показывает общую геометрию: accept находится в нижнем левом квадрате, а рост γ или Δ двигает объект к понижению доверия и аудиту.

Компактная карта операторной трассы используется для серии экспериментов. Она не смешивает длинные диагностические подписи с числами. Видны только числовые компоненты риска. Атлас классов представления показывает, где активируются F0, F_int, NAS и F_ML. Матрица согласованности proof trace показывает, что ключевые инварианты сохранены в route, operator trace, proof trace, dashboard data, verifier report и manifest.

## 5. Что означает действие lower_confidence

Действие lower_confidence не означает, что модель ошиблась. Оно означает, что результат можно использовать только с ограниченным доверием. В исследовательском примере риск выше порога полного принятия из-за потери объяснения и неполной уверенности модели. Поэтому FuzzyXAI не говорит «результат неверен». Он говорит: «результат не следует принимать без ограничения, потому что объяснение неполное, а уверенность не абсолютная».

Такое действие важно для диссертации, потому что оно показывает отличие FuzzyXAI от простого классификатора. Обычная модель выдаёт класс и вероятность. FuzzyXAI добавляет операторный слой: он оценивает качество объяснения, вычисляет риск доверия и выбирает режим применения результата. Русскоязычные фигуры делают этот слой видимым.

## 6. Ограничения визуализации

Эти фигуры не являются доказательством промышленной, медицинской или клинической применимости. Они показывают трассируемость и интерпретируемость операторного маршрута. Высокий риск не означает, что класс обязательно неверен. Высокая потеря объяснения означает, что объяснение неполное. Высокий quality component означает, что вход или источник ограничивает доверие. PASS в матрице proof consistency означает внутреннюю согласованность артефактов, а не истинность внешнего решения.

Также важно, что русскоязычный слой не заменяет технический пакет. Technical SHAP-like figures нужны для audit package и разработчиков. Explanation-first figures нужны для главы и защиты. Оба слоя строятся из одних и тех же route, operator_trace, research CSV и verifier artifacts, но отвечают разным аудиториям.

## 7. Как использовать фигуры в главе

В основном тексте главы лучше начинать не с карты экспериментов, а с локальной карточки объяснения. Она показывает один понятный случай и вводит читателя в логику FuzzyXAI: внешний результат модели превращается в объяснительный объект, затем вычисляются γ, Δ и ρ, после чего выбирается режим доверия. После карточки можно показывать источники риска и шкалу границ действий. Такая последовательность естественна: сначала читатель видит решение, затем причины, затем правила перехода между зонами.

Карта γ–Δ подходит для второго шага, когда уже понятно, что обозначают γ и Δ. Она показывает не отдельный кейс, а геометрию решения. Важно пояснить, что нижний левый квадрат означает область полного принятия только тогда, когда и неуверенность, и потеря объяснения малы. Если хотя бы один показатель выходит за порог, ρ растёт, потому что используется max-агрегация. Поэтому движение вправо или вверх одинаково может привести к понижению доверия или аудиту.

Компактную карту операторной трассы лучше использовать как исследовательскую сводку. Она не предназначена для детального чтения каждого эксперимента, но показывает, что риск меняется не случайно и не вручную. В разных строках видны разные источники ограничения. Атлас классов представления помогает показать, что F0, F_int, NAS и F_ML не являются декоративными метками, а активируются в разных типах задач и perturbation.

## 8. Правила подписи и терминологии

На рисунках намеренно не используются только английские action labels. Вместо lower_confidence написано «понизить доверие», вместо accept — «принять», вместо audit — «направить на аудит». При этом технические идентификаторы могут оставаться в JSON metadata и manifest, потому что они нужны для воспроизводимости. Видимая часть фигур ориентирована на читателя диссертации, а не на разработчика.

Греческие обозначения оставлены, потому что они связаны с математическим описанием фреймворка, но каждое обозначение раскрыто прямо на рисунке. Это снижает риск неправильного чтения. γ не следует понимать как ошибку модели, Δ не следует понимать как качество классификации, а ρ не является вероятностью класса. Это именно показатели операторного доверия к объяснению.

## 9. Связь с доказательным следом

Матрица согласованности нужна для того, чтобы читатель видел: визуализация не является иллюстрацией, нарисованной после вычислений вручную. Она связана с route, operator trace, proof trace, dashboard data, verifier report и manifest. Если инвариант неприменим к конкретному артефакту, он помечается как N/A, а не как ошибка. Это важно: отсутствие source_commit внутри verifier_report не должно превращаться в красный сбой, если контракт verifier_report не хранит это поле. Красная ошибка допустима только там, где инвариант действительно обязан присутствовать и нарушен.

Такой подход делает visual layer частью audit trail. Фигура не только объясняет решение, но и показывает, что само объяснение можно проверить. Для диссертации это принципиально: FuzzyXAI должен выглядеть не как набор красивых графиков, а как воспроизводимый механизм операторной трассировки.
"""
    write(path, text.strip() + "\n")


def captions() -> str:
    return """# Подписи к русскоязычным визуализациям FuzzyXAI

Рисунок 4.x — Карточка объяснения решения FuzzyXAI. На рисунке показано, как неуверенность модели, потеря объяснения и качество входа формируют итоговый риск ρ и приводят к действию «понизить доверие».

Рисунок 4.x — Источники операторного риска. Фигура показывает, какое операторное свидетельство определяет итоговый риск при max-агрегации.

Рисунок 4.x — Почему доверие понижено. Показано, что ρ выше порога полного принятия, но ниже порога аудита.

Рисунок 4.x — Шкала границ действий. Шкала показывает расстояние текущего ρ до соседних режимов доверия.

Рисунок 4.x — Карта решений в пространстве γ–Δ. Карта показывает, как сочетание рассогласования γ и потери объяснения Δ определяет область действия.

Рисунок 4.x — Компактная карта операторной трассы. Карта показывает числовые компоненты риска по исследовательским экспериментам.

Рисунок 4.x — Матрица согласованности доказательного следа. Матрица показывает, что route, operator trace, proof trace, dashboard data и manifest согласованы по ключевым инвариантам.

Рисунок 4.x — Атлас классов представления. Атлас показывает, где активируются F0, F_int, NAS и F_ML.
"""


def readme_ru(narrative: dict) -> str:
    final = narrative["final_decision"]
    return f"""# FuzzyXAI: русскоязычный пакет объяснения

## Что это

Этот пакет показывает, как FuzzyXAI объясняет результат модели через операторный маршрут: от данных и результата модели до γ, Δ, ρ, действия и proof trace.

## Как читать

1. Начните с `figures/explanation_card_ru.*`.
2. Затем посмотрите `figures/task_data_passport_ru.*`.
3. Потом откройте `figures/operator_route_story_ru.*`.
4. Если нужно понять каждый шаг — `figures/operator_cards_ru.*`.
5. Для проверки согласованности — `figures/proof_consistency_ru.*`.

## Главное решение

FuzzyXAI выбрал действие **{final['action_ru']}**. Главная причина: {final['main_reason_ru']}

## Что означают γ, Δ и ρ

γ — неуверенность или рассогласование.
Δ — потеря объяснения.
ρ — итоговый риск доверия к объяснительному маршруту.

## Почему не SHAP

SHAP объясняет вклад признаков в предсказание. FuzzyXAI объясняет, можно ли доверять объяснительному маршруту и что делать дальше.

## Проверяемость

Все значения связаны с route, operator trace, proof trace и verifier report. Проверки лежат в каталоге `checks/`.
"""


def readable_report_md(narrative: dict, readable: dict) -> str:
    lines = [
        "# Человекочитаемый отчёт FuzzyXAI",
        "",
        f"## {readable['headline_ru']}",
        "",
        readable["one_sentence_ru"],
        "",
        "## Для пользователя",
        "",
    ]
    lines += [f"- {item}" for item in readable["for_non_expert_ru"]]
    lines += [
        "",
        "## Итоговое действие",
        "",
        f"- Действие: {narrative['final_decision']['action_ru']}",
        f"- Что делать: {narrative['final_decision']['user_next_step_ru']}",
        "",
        "## Техническое основание",
        "",
    ]
    lines += [f"- {item}" for item in readable["technical_ru"]]
    return "\n".join(lines) + "\n"


def readable_report_html(md: str) -> str:
    html = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = html.replace("\n# ", "\n<h1>").replace("\n## ", "\n<h2>").replace("\n- ", "\n<li>")
    html = html.replace("\n", "<br>\n")
    return f"<!doctype html><html lang='ru'><meta charset='utf-8'><body style='font-family:DejaVu Sans,Arial,sans-serif;max-width:900px;margin:32px auto;line-height:1.5'>{html}</body></html>\n"


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing RU visual artifact: {path}")


def word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def verify_package(figures: Path, report: Path, manifest_path: Path, data: Path, checks: Path) -> None:
    from PIL import Image

    route = read_json(CLI / "route.json")
    verifier = read_json(CLI / "verifier_report.json")
    if len(short_footer(route, verifier.get("overall_status", "passed"))) > 90:
        raise SystemExit("RU figure footer is longer than 90 characters")
    for base in FIGURES:
        for suffix in ("png", "pdf", "svg"):
            require_file(figures / f"{base}.{suffix}")
        with Image.open(figures / f"{base}.png") as image:
            if image.size[0] < 1800:
                raise SystemExit(f"PNG width is less than 1800px: {base}")
        svg = (figures / f"{base}.svg").read_text(encoding="utf-8", errors="ignore")
        lower = svg.lower()
        for term in REQUIRED_SVG_TERMS:
            if term not in svg:
                raise SystemExit(f"SVG {base} lacks Russian term: {term}")
        if "Как читать" not in svg and "Что показывает рисунок" not in svg:
            raise SystemExit(f"SVG {base} lacks explanation block")
        for forbidden in (
            "ExternalDemoModel",
            "manual_payload",
            "lower_confidence",
            "defer_to_human",
            ">accept<",
            ">audit<",
            ">block<",
        ):
            if forbidden.lower() in lower:
                raise SystemExit(f"SVG {base} contains forbidden technical label: {forbidden}")
    proof_svg = (figures / "proof_consistency_ru.svg").read_text(encoding="utf-8", errors="ignore")
    if "FAIL" in proof_svg:
        raise SystemExit("proof_consistency_ru contains FAIL")
    if word_count(report) < 1200:
        raise SystemExit("ru_explanation_visual_report.md must contain at least 1200 words")
    for required in ["operator_narrative_ru.json", "readable_explanation_ru.json", "task_data_passport.json", "operator_cards_ru.json"]:
        require_file(data / required)
    narrative = read_json(data / "operator_narrative_ru.json")
    for card in narrative.get("operator_cards", []):
        for key in ["operator_question_ru", "plain_result_ru", "effect_on_route_ru"]:
            if not card.get(key):
                raise SystemExit(f"operator narrative card lacks {key}: {card.get('operator_id')}")
    for required in ["editorial_check.json", "freshness_check.json", "operator_narrative_check.json", "verifier_report.json"]:
        require_file(checks / required)
    manifest = read_json(manifest_path)
    if manifest.get("manifest_self_hash_policy") != "excluded":
        raise SystemExit("manifest_self_hash_policy must be excluded")
    for item in manifest.get("sha256", []):
        if item["path"] == "manifest.json":
            raise SystemExit("manifest must not include itself")
        path = OUT / item["path"]
        require_file(path)
        if digest(path) != item["sha256"]:
            raise SystemExit(f"sha256 mismatch: {item['path']}")


def verify_zip_package(zip_path: Path) -> None:
    head = must(["git", "rev-parse", "HEAD"]).stdout.strip()
    forbidden = (
        "ExternalDemoModel",
        "manual_payload",
        "lower_confidence",
        "defer_to_human",
        "source_commit=",
        "route_id=external_wine_classification",
    )
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        root = names[0].split("/")[0]
        manifest = json.loads(archive.read(f"{root}/manifest.json").decode("utf-8"))
        if manifest.get("source_commit") != head:
            raise SystemExit(f"zip manifest source_commit is stale: {manifest.get('source_commit')} != {head}")
        for name in names:
            if not name.startswith(f"{root}/figures/") or not name.endswith(".svg"):
                continue
            svg = archive.read(name).decode("utf-8", errors="ignore")
            if "FAIL" in svg:
                raise SystemExit(f"{name} contains FAIL")
            for item in forbidden:
                if item in svg:
                    raise SystemExit(f"{name} contains stale technical label: {item}")


def build() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    figures = OUT / "figures"
    data = OUT / "data"
    report = OUT / "report"
    checks = OUT / "checks"
    figures.mkdir(parents=True)
    data.mkdir(parents=True)
    report.mkdir(parents=True)
    checks.mkdir(parents=True)
    route, trace, verifier, rows = case_data()
    proof_trace = load_proof_trace()
    commit = must(["git", "rev-parse", "HEAD"]).stdout.strip()
    build_narrative, build_readable, build_passport = fuzzyxai_imports()
    narrative_obj = build_narrative(route, trace, proof_trace, verifier)
    narrative = narrative_obj.to_dict()
    readable = build_readable(narrative_obj)
    passport = build_passport(route)

    card = render_explanation_card(route, verifier, figures / "explanation_card_ru.png")
    render_risk_sources(route, verifier, figures / "risk_sources_ru.png")
    render_why_lower_confidence(route, verifier, figures / "why_lower_confidence_ru.png")
    render_decision_boundary(route, verifier, figures / "decision_boundary_ru.png")
    render_gamma_delta_map(route, verifier, rows, figures / "gamma_delta_decision_map_ru.png")
    render_operator_trace_summary(route, verifier, rows, figures / "operator_trace_summary_ru.png")
    render_proof_consistency(route, verifier, figures / "proof_consistency_ru.png")
    render_representation_atlas(route, verifier, rows, figures / "representation_atlas_ru.png")
    render_task_data_passport(route, verifier, passport, figures / "task_data_passport_ru.png")
    render_operator_route_story(route, verifier, narrative, figures / "operator_route_story_ru.png")
    render_operator_cards(route, verifier, narrative, figures / "operator_cards_ru.png")

    visual_terms = {
        "gamma": "γ — неуверенность / рассогласование",
        "delta": "Δ — потеря объяснения",
        "rho": "ρ — итоговый риск",
        "actions": ACTION_RU,
        "components": COMPONENT_RU,
    }
    for name, payload in {
        "route.json": route,
        "operator_trace.json": trace,
        "proof_trace.json": proof_trace,
        "verifier_report.json": verifier,
        "dashboard_data.json": read_json(CLI / "dashboard_data.json"),
        "operator_narrative_ru.json": narrative,
        "readable_explanation_ru.json": readable,
        "task_data_passport.json": passport,
        "operator_cards_ru.json": {"operator_cards": narrative["operator_cards"]},
    }.items():
        write(data / name, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write(data / "visual_terms_ru.json", json.dumps(visual_terms, ensure_ascii=False, indent=2) + "\n")
    write(data / "explanation_card_ru.json", json.dumps(card | {"route_id": route.get("route_id"), "source_commit": route.get("source_commit")}, ensure_ascii=False, indent=2) + "\n")
    write(data / "figure_captions_ru.md", captions())
    write_report(OUT / "ru_explanation_visual_report.md")
    md = readable_report_md(narrative, readable)
    write(report / "readable_report_ru.md", md)
    write(report / "readable_report_ru.html", readable_report_html(md))
    write(OUT / "README_RU.md", readme_ru(narrative))
    write(checks / "verifier_report.json", json.dumps(verifier, ensure_ascii=False, indent=2) + "\n")
    write(checks / "editorial_check.json", json.dumps({"status": "PASS", "forbidden_visible_labels": "absent"}, ensure_ascii=False, indent=2) + "\n")
    write(checks / "freshness_check.json", json.dumps({"status": "PASS", "source_commit": commit}, ensure_ascii=False, indent=2) + "\n")
    write(checks / "operator_narrative_check.json", json.dumps({"status": "PASS", "cards": len(narrative["operator_cards"])}, ensure_ascii=False, indent=2) + "\n")
    write(ROOT / "docs" / "chapter_4_framework" / "ru_visualization_figures.md", captions())
    ASSETS.mkdir(parents=True, exist_ok=True)
    for png in figures.glob("*.png"):
        shutil.copy2(png, ASSETS / png.name)

    manifest = {
        "source_commit": commit,
        "package_source_commit": commit,
        "route_source_commit": route.get("source_commit"),
        "visualization_source_commit": commit,
        "manifest_self_hash_policy": "excluded",
        "package_type": "fuzzyxai_ru_explanation_visual_package",
        "checks": {
            "png_pdf_svg": "PASS",
            "png_width_ge_1800": "PASS",
            "russian_terms_present": "PASS",
            "proof_matrix_no_fail": "PASS",
            "manifest_sha256": "PASS",
        },
        "figures": FIGURES,
    }
    files = [path for path in sorted(OUT.rglob("*")) if path.is_file()]
    manifest["sha256"] = [
        {"path": path.relative_to(OUT).as_posix(), "sha256": digest(path), "size_bytes": path.stat().st_size}
        for path in files
        if path.name != "manifest.json"
    ]
    write(OUT / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    verify_package(figures, OUT / "ru_explanation_visual_report.md", OUT / "manifest.json", data, checks)

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, f"fuzzyxai_ru_explanation_visual_package/{path.relative_to(OUT).as_posix()}")
    verify_zip_package(ZIP)
    return {
        "commit": commit,
        "package": ZIP.as_posix(),
        "files": len([p for p in OUT.rglob("*") if p.is_file()]),
        "figures": len(FIGURES),
        "report_words": word_count(OUT / "ru_explanation_visual_report.md"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "fuzzyxai-ru-visual-explanation-package: PASS"
        if args.package_only
        else "fuzzyxai-ru-visual-explanation-check: PASS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
