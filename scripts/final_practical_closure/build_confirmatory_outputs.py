#!/usr/bin/env python3
"""Build final statistics, tables, figures and Chapter 4 source from sealed evidence."""

from __future__ import annotations

import csv
import json

from common import CONFIRMATORY, FORMATIVE, IMMUTABLE_RESULTS, ROOT, load_json, sha256, write_json


SECTIONS = (
    ("4.1", "Практическая постановка"),
    ("4.2", "Данные и sealed protocol"),
    ("4.3", "Архитектура практического контроллера"),
    ("4.4", "Корректная taxonomy аналогов"),
    ("4.5", "H1 и H2"),
    ("4.6", "H4"),
    ("4.7", "H3-original"),
    ("4.8", "H3 practical"),
    ("4.9", "H5-S и H5-A"),
    ("4.10", "H5-P"),
    ("4.11", "H6-A и H6-B"),
    ("4.12", "H7-A и H7-B"),
    ("4.13", "H8"),
    ("4.14", "H9"),
    ("4.15", "Формирующая проверка карточек"),
    ("4.16", "Воспроизводимость"),
    ("4.17", "Итоговые claims и ограничения"),
)


def main() -> None:
    summary_path = CONFIRMATORY / "summary.json"
    registry_path = FORMATIVE.parent / "claim_registry.json"
    if not summary_path.is_file() or not registry_path.is_file():
        raise SystemExit("BLOCKED: sealed confirmatory summary and claim registry are required")
    summary, registry = load_json(summary_path), load_json(registry_path)
    if summary.get("confirmatory_run_completed") is not True or registry.get("confirmatory_run_completed") is not True:
        raise SystemExit("BLOCKED: confirmatory run is incomplete")
    output = CONFIRMATORY
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rows = []
    values = {}
    for item in summary["experiments"]:
        measurement = item["measurement"]
        row = {
            "experiment_id": item["experiment_id"],
            "status": item["status"],
            "effect_size": measurement["effect_size"],
            "ci_low": measurement["confidence_interval_95"][0],
            "ci_high": measurement["confidence_interval_95"][1],
            "adjusted_p": measurement["adjusted_p"],
            "n": measurement["n"],
            "unit_of_analysis": measurement["unit_of_analysis"],
            "evidence_path": item["artifact_path"],
            "sha256": item["sha256"],
        }
        rows.append(row)
        values[item["experiment_id"]] = row
    table = tables / "confirmatory_claims.csv"
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure = _status_figure(rows, figures)
    write_json(output / "chapter4_values.json", values)
    write_json(output / "chapter4_claims.json", registry)
    chapter = output / "chapter4_final.md"
    chapter.write_text(_chapter(rows), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "phase": "sealed_confirmatory",
        "tables": [_entry(table)],
        "figures": [_entry(figure)],
        "chapter_source": _entry(chapter),
        "values": _entry(output / "chapter4_values.json"),
        "claims": _entry(output / "chapter4_claims.json"),
    }
    write_json(output / "chapter4_artifacts_manifest.json", manifest)
    print(f"PASS: practical_confirmatory_outputs claims={len(rows)} tables=1 figures=1 placeholders=0")


def _status_figure(rows, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#197278" if row["status"] == "supported" else "#c44536" for row in rows]
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    axis.barh([row["experiment_id"] for row in rows], [row["effect_size"] for row in rows], color=colors)
    axis.axvline(0, color="#222222", linewidth=0.8)
    axis.set(xlabel="Frozen primary effect size", title="Confirmatory claim status (green=supported, red=not supported)")
    fig.tight_layout()
    path = directory / "confirmatory_claim_status.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _chapter(rows) -> str:
    by_id = {row["experiment_id"]: row for row in rows}
    refs = {
        claim_id: f"[evidence:{claim_id} sha256:{row['sha256']}]"
        for claim_id, row in by_id.items()
    }
    original_ref = f"[evidence:FROZEN-RESULTS sha256:{sha256(ROOT / 'study/final_practical_closure/frozen_results.json')}]"
    lines = ["# Глава 4. Практический контур FuzzyXAI", ""]
    for number, title in SECTIONS:
        lines.extend((f"## {number} {title}", "", _section(number, by_id, refs, original_ref), ""))
    return "\n".join(lines)


def _section(number, rows, refs, original_ref):
    if number in {"4.5", "4.6", "4.7", "4.10"}:
        return f"Замороженные исходные результаты сохранены без переименования: {json.dumps(IMMUTABLE_RESULTS, ensure_ascii=False, sort_keys=True)}. {original_ref}"
    selected = {
        "4.8": ("H3-P1", "H3-P2", "H3-P3", "H3-P4"),
        "4.9": ("H5-A",),
        "4.11": ("H6-A", "H6-B"),
        "4.12": ("H7-A", "H7-B"),
        "4.13": ("H8",),
        "4.14": ("H9",),
    }.get(number)
    if selected:
        parts = []
        for claim_id in selected:
            row = rows[claim_id]
            parts.append(
                f"{claim_id}: {row['status']}; effect={row['effect_size']}; 95% CI [{row['ci_low']}, {row['ci_high']}]; adjusted p={row['adjusted_p']}; N={row['n']} ({row['unit_of_analysis']}). {refs[claim_id]}"
            )
        return " ".join(parts)
    generic = {
        "4.1": "Практическая цель определена как снижение operationally invalid automatic actions при ограниченном review budget.",
        "4.2": "Данные и splits были запечатаны до открытия test; признаки policy строились out-of-fold.",
        "4.3": "Контроллер включает hard guard, predictive risk, route risk и budget optimizer.",
        "4.4": "Post-hoc explainers, interpretable predictors и action policies сравниваются раздельно.",
        "4.15": "AI-review использовался только как автоматизированная формирующая проверка текста, не как экспертная валидация.",
        "4.16": "Артефакты содержат experiment id, evidence path, SHA256, effect size, confidence interval, adjusted p, N и unit of analysis.",
        "4.17": "Выводы ограничены вычислительным контуром; claims о понятности людям, экспертном подтверждении и предметной безопасности исключены.",
    }
    return f"{generic[number]} {original_ref}"


def _entry(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "size": path.stat().st_size}


if __name__ == "__main__":
    main()

