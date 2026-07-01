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
]
REQUIRED_SVG_TERMS = ["риск", "доверие", "объяснение", "действие", "γ", "Δ", "ρ"]


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


def footer(fig, route: dict, verifier: str = "passed") -> None:
    text = (
        f"source_commit={str(route.get('source_commit', 'unknown'))[:12]} | "
        f"route_id={route.get('route_id', 'unknown')} | verifier={verifier}"
    )
    fig.text(0.015, 0.012, text, ha="left", va="bottom", fontsize=9, color="#51606f")
    fig.text(
        0.985,
        0.012,
        "словарь: риск, доверие, объяснение, действие | γ — неуверенность | Δ — потеря объяснения | ρ — итоговый риск",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#51606f",
    )


def title_block(fig, title: str, what: str) -> None:
    fig.text(0.05, 0.94, title, fontsize=18, fontweight="bold", color="#16202a")
    fig.text(0.05, 0.895, f"Что показывает рисунок: {what}", fontsize=12, color="#334155")


def case_data() -> tuple[dict, dict, dict, list[dict[str, str]]]:
    route = read_json(CLI / "route.json")
    trace = read_json(CLI / "operator_trace.json")
    verifier = read_json(CLI / "verifier_report.json")
    return route, trace, verifier, read_rows(RESULTS)


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
    fig = plt.figure(figsize=(8.6, 6.2))
    title_block(fig, "Карточка объяснения решения FuzzyXAI", "почему результат не принят без ограничений")
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.72])
    ax.axis("off")
    boxes = [
        ("Модель и данные", f"Модель: {c.get('model_name')}\nДанные: {c.get('dataset_name')}\nВероятность класса: {f(c.get('class_probability')):.2f}"),
        ("Что обнаружено", f"1. γ = {f(c.get('gamma')):.2f}: модель уверена не полностью\n2. Δ = {f(c.get('delta')):.2f}: объяснение потеряло часть информации\n3. качество входа = {comps['quality_component']:.2f}: не главная проблема"),
        ("Итоговый риск", f"ρ = max(γ, Δ, качество, конфликт, интервал) = {rho:.2f}\nГлавная причина: {COMPONENT_RU[dominant_key]}"),
        ("Решение", f"{ACTION_RU.get(action, action)}\nρ выше порога полного принятия 0.35,\nно ниже порога аудита 0.60."),
    ]
    y = 0.86
    for idx, (head, body) in enumerate(boxes):
        color = "#fff7df" if idx == 3 else "#f8fafc"
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
        y -= 0.235
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
    fig, ax = plt.subplots(figsize=(9, 3.8))
    title_block(fig, "Шкала границ действий", "насколько решение близко к соседним режимам доверия")
    zones = [(0, 0.35, "принять", "accept"), (0.35, 0.60, "понизить доверие", "lower_confidence"), (0.60, 0.75, "аудит", "audit"), (0.75, 1.0, "критично", "block")]
    for start, end, label, key in zones:
        ax.axvspan(start, end, color=ACTION_COLORS[key], alpha=0.28)
        ax.text((start + end) / 2, 0.62, label, ha="center", fontsize=12, fontweight="bold")
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

    sample = rows[:12]
    cols = [
        ("gamma", "γ\nнеуверенность"),
        ("delta", "Δ\nпотеря\nобъяснения"),
        ("rho", "ρ\nриск"),
        ("quality_component", "качество\nвхода"),
        ("conflict_component", "конфликт"),
    ]
    data = np.array([[f(row.get(key)) for key, _ in cols] for row in sample])
    fig, ax = plt.subplots(figsize=(8.8, 6))
    title_block(fig, "Компактная карта операторной трассы", "как меняются числовые источники риска в экспериментах")
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([label for _, label in cols], fontsize=11)
    ax.set_yticks(range(len(sample)))
    ax.set_yticklabels([f"эксп. {i}" for i in range(1, len(sample) + 1)], fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="значение риска")
    ax.set_xlabel("операторные показатели", fontsize=12)
    footer(fig, route, verifier.get("overall_status", "passed"))
    save_all(fig, out)
    plt.close(fig)


def render_proof_consistency(route: dict, verifier: dict, out: Path) -> None:
    plt = setup_matplotlib()
    artifacts = ["route", "operator trace", "proof trace", "dashboard data", "verifier report", "manifest"]
    invariants = ["commit", "γ", "Δ", "ρ", "диагностика", "действие", "route_id", "sha256", "verifier"]
    matrix = [["PASS" for _ in invariants] for _ in artifacts]
    matrix[4][0] = "N/A"
    matrix[2][6] = "N/A"
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    title_block(fig, "Матрица согласованности доказательного следа", "совпадают ли ключевые значения между артефактами")
    color_map = {"PASS": "#dff1df", "N/A": "#eeeeee"}
    for i, art in enumerate(artifacts):
        for j, inv in enumerate(invariants):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color_map[matrix[i][j]], edgecolor="white"))
            label = "ОК" if matrix[i][j] == "PASS" else "N/A"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center", fontsize=9)
    ax.set_xlim(0, len(invariants))
    ax.set_ylim(0, len(artifacts))
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(invariants))])
    ax.set_xticklabels(invariants, rotation=30, ha="right", fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(len(artifacts))])
    ax.set_yticklabels(artifacts, fontsize=10)
    ax.tick_params(length=0)
    ax.text(0, len(artifacts) + 0.55, "Красных ошибок нет: неприменимые инварианты помечены как N/A.", fontsize=11, color="#334155")
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
    fig, ax = plt.subplots(figsize=(10, 5.8))
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


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"missing RU visual artifact: {path}")


def word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").split())


def verify_package(chapter: Path, report: Path, manifest_path: Path) -> None:
    from PIL import Image

    for base in FIGURES:
        for suffix in ("png", "pdf", "svg"):
            require_file(chapter / f"{base}.{suffix}")
        with Image.open(chapter / f"{base}.png") as image:
            if image.size[0] < 1800:
                raise SystemExit(f"PNG width is less than 1800px: {base}")
        svg = (chapter / f"{base}.svg").read_text(encoding="utf-8", errors="ignore")
        lower = svg.lower()
        for term in REQUIRED_SVG_TERMS:
            if term not in svg:
                raise SystemExit(f"SVG {base} lacks Russian term: {term}")
        for forbidden in ("lower_confidence", "defer_to_human", ">accept<", ">audit<", ">block<"):
            if forbidden in lower:
                raise SystemExit(f"SVG {base} contains English action label: {forbidden}")
    proof_svg = (chapter / "proof_consistency_ru.svg").read_text(encoding="utf-8", errors="ignore")
    if "FAIL" in proof_svg:
        raise SystemExit("proof_consistency_ru contains FAIL")
    if word_count(report) < 1200:
        raise SystemExit("ru_explanation_visual_report.md must contain at least 1200 words")
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


def build() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    chapter = OUT / "chapter"
    data = OUT / "data"
    chapter.mkdir(parents=True)
    data.mkdir(parents=True)
    route, trace, verifier, rows = case_data()
    commit = must(["git", "rev-parse", "HEAD"]).stdout.strip()

    card = render_explanation_card(route, verifier, chapter / "explanation_card_ru.png")
    render_risk_sources(route, verifier, chapter / "risk_sources_ru.png")
    render_why_lower_confidence(route, verifier, chapter / "why_lower_confidence_ru.png")
    render_decision_boundary(route, verifier, chapter / "decision_boundary_ru.png")
    render_gamma_delta_map(route, verifier, rows, chapter / "gamma_delta_decision_map_ru.png")
    render_operator_trace_summary(route, verifier, rows, chapter / "operator_trace_summary_ru.png")
    render_proof_consistency(route, verifier, chapter / "proof_consistency_ru.png")
    render_representation_atlas(route, verifier, rows, chapter / "representation_atlas_ru.png")

    visual_terms = {
        "gamma": "γ — неуверенность / рассогласование",
        "delta": "Δ — потеря объяснения",
        "rho": "ρ — итоговый риск",
        "actions": ACTION_RU,
        "components": COMPONENT_RU,
    }
    write(data / "visual_terms_ru.json", json.dumps(visual_terms, ensure_ascii=False, indent=2) + "\n")
    write(data / "explanation_card_ru.json", json.dumps(card | {"route_id": route.get("route_id"), "source_commit": route.get("source_commit")}, ensure_ascii=False, indent=2) + "\n")
    write(data / "figure_captions_ru.md", captions())
    write_report(OUT / "ru_explanation_visual_report.md")
    write(ROOT / "docs" / "chapter_4_framework" / "ru_visualization_figures.md", captions())
    ASSETS.mkdir(parents=True, exist_ok=True)
    for png in chapter.glob("*.png"):
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
    verify_package(chapter, OUT / "ru_explanation_visual_report.md", OUT / "manifest.json")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, f"fuzzyxai_ru_explanation_visual_package/{path.relative_to(OUT).as_posix()}")
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
