#!/usr/bin/env python3
"""Generate chapter 2 methodological figures in SVG/PDF/PNG."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT = Path("figures/chapter2")
PREVIEW = OUT / "png_preview"
BLUE = "#dbeafe"
GREEN = "#dcfce7"
YELLOW = "#fef9c3"
RED = "#fee2e2"
GRAY = "#f8fafc"
EDGE = "#334155"


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
    }
)


def box(ax, xy, wh, text, fc=GRAY, lw=1.1, fs=9):
    r = Rectangle(xy, wh[0], wh[1], facecolor=fc, edgecolor=EDGE, linewidth=lw)
    ax.add_patch(r)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return r


def arrow(ax, a, b, style="-|>", ls="-", color=EDGE, lw=1.05):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, linewidth=lw, linestyle=ls, color=color))


def base_ax(figsize=(11, 6.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{name}.svg"
    pdf = OUT / f"{name}.pdf"
    png = PREVIEW / f"{name}.png"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # Normalize SVG through rsvg-convert when available; it catches malformed output early.
    if subprocess.run(["bash", "-lc", "command -v rsvg-convert >/dev/null"], check=False).returncode == 0:
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(pdf), str(svg)], check=True)


def fig_2_1():
    fig, ax = base_ax((12, 6.4))
    box(ax, (0.36, 0.39), (0.30, 0.20), "$E_k=\\langle L_k,\\mu_k,R_k,\\alpha_k,u_k,\\tau_k\\rangle$", BLUE, fs=13)
    fields = [
        ((0.72, 0.80), "$L_k$\nсловарь термов"),
        ((0.72, 0.66), "$\\mu_k$\nфункции принадлежности"),
        ((0.72, 0.52), "$R_k$\nбаза правил"),
        ((0.72, 0.38), "$\\alpha_k$\nактивации правил"),
        ((0.72, 0.24), "$u_k$\nагрегированная\nнеопределённость"),
        ((0.72, 0.10), "$\\tau_k$\nтрассируемый след"),
    ]
    sources = [
        ((0.06, 0.80), "ExplainPlan"),
        ((0.06, 0.66), "Norm$_k$"),
        ((0.06, 0.52), "Fuzz$_k$"),
        ((0.06, 0.38), "Rule$_k$"),
        ((0.06, 0.24), "$U_{model}, U_{rules}$\n$U_{trace}$"),
        ((0.06, 0.10), "Trace$_k$"),
    ]
    link_y = [0.565, 0.535, 0.505, 0.475, 0.445, 0.415]
    left_points = [(0.36, y) for y in link_y]
    right_points = [(0.66, y) for y in link_y]
    for (xy, t), p in zip(sources, left_points):
        box(ax, xy, (0.20, 0.09), t, YELLOW, fs=8.6)
        arrow(ax, (xy[0] + 0.20, xy[1] + 0.045), p)
    for (xy, t), p in zip(fields, right_points):
        box(ax, xy, (0.22, 0.09), t, GREEN, fs=8.6)
        arrow(ax, p, (xy[0], xy[1] + 0.045))
    ax.text(0.5, 0.95, "Структура нечёткого объяснительного объекта", ha="center", fontsize=12.5)
    save(fig, "fig_2_1_Ek_structure")


def fig_2_2():
    fig, ax = base_ax()
    box(ax, (0.05, 0.53), (0.16, 0.13), "$E_i$", BLUE, fs=14)
    box(ax, (0.05, 0.30), (0.16, 0.13), "$E_j$", BLUE, fs=14)
    box(ax, (0.32, 0.31), (0.34, 0.33), "$T_{ij}=(\\rho_{ij},\\sigma_{ij},\\pi_{ij},\\theta_{ij})$\n\n$\\rho_{ij}$ — переименование термов\n$\\sigma_{ij}$ — перенос носителя\n$\\pi_{ij}$ — сопоставление правил\n$\\theta_{ij}$ — согласование следа", YELLOW, fs=8.6)
    box(ax, (0.79, 0.58), (0.15, 0.12), "$E_{ij}$", GREEN, fs=14)
    box(ax, (0.79, 0.22), (0.15, 0.12), "$D_{ij}$", RED, fs=14)
    arrow(ax, (0.21, 0.59), (0.34, 0.52))
    arrow(ax, (0.21, 0.35), (0.34, 0.42))
    ax.text(0.70, 0.51, "$\\odot$", ha="center", va="center", fontsize=18)
    ax.text(0.70, 0.46, "частичная\nкомпозиция", ha="center", va="top", fontsize=8.4)
    arrow(ax, (0.66, 0.52), (0.79, 0.64))
    arrow(ax, (0.66, 0.40), (0.79, 0.28))
    ax.text(0.70, 0.69, "$\\gamma_{ij}\\leq\\gamma_{max}$\n$\\Delta_T\\leq\\Delta_{max}$", ha="center", fontsize=8.8)
    ax.text(0.70, 0.34, "иначе", ha="center", fontsize=10)
    ax.text(0.50, 0.12, "Пороговая проверка нестрогая: $\\gamma_{ij}=\\gamma_{max}$ допустимо", ha="center", fontsize=10)
    ax.text(0.5, 0.94, "Согласование и частичная композиция", ha="center", fontsize=12.5)
    save(fig, "fig_2_2_Tij_composition")


def fig_2_3():
    fig, ax = base_ax((11, 4.8))
    ax.text(0.5, 0.86, "$d_E(E_a,E_b)=\\sum_s \\beta_s d_s(E_a,E_b)$", ha="center", fontsize=15)
    xs = [0.06, 0.24, 0.42, 0.60, 0.78]
    labels = ["$\\beta_\\mu d_\\mu$\nфункции\nпринадлежности", "$\\beta_R d_R$\nсостав\nправил", "$\\beta_\\alpha d_\\alpha$\nактивации", "$\\beta_u d_u$\nнеопределённость", "$\\beta_\\tau d_\\tau$\nтрассируемый\nслед"]
    colors = [BLUE, GREEN, YELLOW, "#ede9fe", RED]
    for x, lab, c in zip(xs, labels, colors):
        box(ax, (x, 0.42), (0.16, 0.18), lab, c, fs=9)
        arrow(ax, (x + 0.08, 0.42), (0.50, 0.24), lw=1.0)
    box(ax, (0.39, 0.12), (0.22, 0.12), "$d_E$\nпрозрачная сумма\nкомпонент", GRAY, fs=10)
    box(ax, (0.73, 0.12), (0.22, 0.12), "$\\beta_s>0$\n$\\sum_s\\beta_s=1$", GRAY, fs=11)
    save(fig, "fig_2_3_dE_metric")


def fig_2_4():
    fig, ax = base_ax((12, 5.3))
    nodes = [
        ("$E_{data}$", 0.09, 0.62, 0.14),
        ("$E_{feat}$", 0.28, 0.62, 0.14),
        ("$E_{model}$", 0.47, 0.62, 0.16),
        ("$E_{risk}$", 0.67, 0.62, 0.14),
    ]
    for t, x, y, w in nodes:
        box(ax, (x - w / 2, y - 0.055), (w, 0.11), t, BLUE, fs=10.5)
    box(ax, (0.83, 0.55), (0.15, 0.14), "$E_{action}$\nитог: BLOCK / ACCEPT", GREEN, fs=8.8)
    box(ax, (0.67, 0.24), (0.15, 0.12), "$D_{ij}$\nдиагностический\nразрыв", RED, fs=8.8)
    arrow(ax, (0.16, 0.62), (0.21, 0.62))
    ax.text(0.185, 0.69, "сертифицированный\n$\\gamma\\leq\\gamma_{max}$", ha="center", fontsize=7.8)
    arrow(ax, (0.35, 0.62), (0.39, 0.62), ls="--")
    ax.text(0.37, 0.70, "приближённое\n$\\Delta_T>0$\n$\\Delta_T\\leq\\Delta_{max}$", ha="center", fontsize=7.6)
    arrow(ax, (0.55, 0.62), (0.60, 0.62))
    ax.text(0.575, 0.69, "сертифицированный", ha="center", fontsize=7.8)
    arrow(ax, (0.74, 0.62), (0.83, 0.62))
    ax.text(0.785, 0.69, "$\\gamma\\leq\\gamma_{max}$", ha="center", fontsize=8)
    arrow(ax, (0.70, 0.565), (0.72, 0.36), color="#b91c1c")
    ax.plot([0.697, 0.723], [0.47, 0.51], color="#b91c1c", lw=2)
    ax.plot([0.697, 0.723], [0.51, 0.47], color="#b91c1c", lw=2)
    ax.text(0.76, 0.43, "разрыв\n$D_{ij}$", ha="center", fontsize=8.4, color="#991b1b")
    ax.text(0.50, 0.14, "Легенда: сплошное ребро — сертифицированный переход; пунктирное ребро — приближённое согласование; крест — диагностический разрыв.", ha="center", fontsize=8.8)
    ax.text(0.5, 0.91, "$G_{Expl}$: граф сертифицированных объяснительных переходов", ha="center", fontsize=12.5)
    save(fig, "fig_2_4_GExpl_graph")


def fig_2_5():
    fig, ax = base_ax((13.2, 4.4))
    items = [
        ("$P=0{,}7036$\nвыход модели", BLUE),
        ("$\\mu=(0{,}296;\\ 0{,}186;\\ 0{,}704)$\nпринадлежности", GREEN),
        ("$u_M=0{,}3597$\nнеопределённость", YELLOW),
        ("$\\gamma=0{,}3514$\nрассогласование", "#ede9fe"),
        ("$\\chi_R^{crit}=1$\nкритический\nконфликт", RED),
        ("итог = BLOCK\nдействие наблюдателя", GREEN),
    ]
    x0, w, gap = 0.02, 0.145, 0.015
    for i, (txt, c) in enumerate(items):
        x = x0 + i * (w + gap)
        box(ax, (x, 0.47), (w, 0.24), txt, c, fs=8.5)
        if i < len(items) - 1:
            arrow(ax, (x + w, 0.59), (x + w + gap, 0.59))
    ax.text(0.5, 0.86, "Сквозной численный маршрут sample_113", ha="center", fontsize=12.5)
    save(fig, "fig_2_5_sample113_flow")


def main():
    argparse.ArgumentParser().parse_args()
    fig_2_1()
    fig_2_2()
    fig_2_3()
    fig_2_4()
    fig_2_5()
    print(f"Generated figures in {OUT}")


if __name__ == "__main__":
    main()
