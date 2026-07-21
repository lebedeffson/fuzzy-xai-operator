#!/usr/bin/env python3
"""Generate claim-safe reports and chapter inserts from final evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "release_evidence/q1_final"
REPORTS = ROOT / "reports/q1_final"
CHAPTER = ROOT / "dissertation_artifacts/q1_final"


def load(relative: str) -> dict[str, object]:
    path = EVIDENCE / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHAPTER.mkdir(parents=True, exist_ok=True)
    identity = load("run_identity.json")
    real = load("real_benchmarks/combined_status.json")
    final = load("hypotheses/final_results.json")
    claims = load("claim_registry.json")
    gates = load("final_gate_matrix.json")
    h6 = load("rule_ablation/final_claim_status.json")
    scaling = load("scalability/end_to_end.json")
    _write_final_report(identity, real, final, h6, gates, claims)
    _write_reviewer_response(real, final, h6, gates)
    _write_chapter_inserts(identity, real, final, h6, gates, claims)
    _write_tables(real, final, h6, scaling, claims)
    _write_figures(final, scaling)
    print("PASS: q1_final_reports")


def _write_final_report(
    identity: dict[str, object],
    real: dict[str, object],
    final: dict[str, object],
    h6: dict[str, object],
    gates: dict[str, object],
    claims: dict[str, object],
) -> None:
    lines = [
        "# FuzzyXAI Q1 Final Closure Report",
        "",
        "## Git",
        f"- Branch: `{identity.get('branch', 'not-generated')}`",
        f"- Final commit: `{identity.get('final_commit', 'not-generated')}`",
        "",
        "## Artifact identity",
        f"- Profile: `{identity.get('profile', 'not-generated')}`",
        f"- Real benchmark status: `{real.get('status', 'NOT_RUN')}`",
        "",
        "## Real benchmarks",
        "| Modality | Dataset | Classes | Evaluation objects |",
        "|---|---|---:|---:|",
    ]
    for modality, row in real.get("modalities", {}).items():
        dataset = row["dataset"]
        lines.append(
            f"| {modality} | {dataset['dataset_id']} | {dataset['native_class_count']} | {row['evaluation_object_count']} |"
        )
    lines.extend(["", "## H1-H5", "", "```json", json.dumps(final, ensure_ascii=False, indent=2), "```"])
    lines.extend(["", "## H6", f"- Status: `{h6.get('status', 'inconclusive')}`"])
    lines.extend(["", "## External studies"])
    for name in ("domain_language", "comprehension", "expert_action"):
        lines.append(f"- {name}: `{gates.get(name, {}).get('status', 'OPEN')}`")
    counts = claims.get("counts", {})
    lines.extend(
        [
            "",
            "## Claims",
            f"- Supported: `{counts.get('supported', 0)}`",
            f"- Not supported: `{counts.get('not_supported', 0)}`",
            f"- Inconclusive: `{counts.get('inconclusive', 0)}`",
            f"- External gate: `{counts.get('external_gate', 0)}`",
            "",
            "## Release",
            f"- Stable release allowed: `{gates.get('stable_release_allowed', False)}`",
            "- A technical candidate may be built while external gates remain explicitly open.",
        ]
    )
    (REPORTS / "FuzzyXAI_Q1_FINAL_CLOSURE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reviewer_response(
    real: dict[str, object],
    final: dict[str, object],
    h6: dict[str, object],
    gates: dict[str, object],
) -> None:
    text = f"""# Ответ рецензенту: финальный экспериментальный цикл

Реальные native-multiclass контуры имеют статус `{real.get('status', 'NOT_RUN')}`. Контролируемые результаты не выданы за внешнюю проверку.

Критический разрыв разделён на структурный индикатор и предсказательную проверку. Предсказательный статус: `{final.get('H5_real', {}).get('predictive', {}).get('status', 'not_run')}`. При отсутствии прироста AUPRC разрешена только структурная интерпретация.

Confirmatory rule ablation: `{h6.get('status', 'inconclusive')}`. При нулевом результате общий эффект правила удаляется, локальная диагностика сохраняется.

Domain-language, comprehension и expert-action gate: `{gates.get('domain_language', {}).get('status', 'OPEN')}`, `{gates.get('comprehension', {}).get('status', 'OPEN')}`, `{gates.get('expert_action', {}).get('status', 'OPEN')}`. До появления реальных независимых записей утверждения о понятности и практической полезности не разрешены.
"""
    (REPORTS / "reviewer_response_ru.md").write_text(text, encoding="utf-8")
    english = f"""# Reviewer response: final empirical cycle

Native multiclass evidence status is `{real.get('status', 'NOT_RUN')}`. Controlled evidence is not presented as external validation.

Critical rupture has separate structural and predictive results. Predictive status is `{final.get('H5_real', {}).get('predictive', {}).get('status', 'not_run')}`; without incremental AUPRC only structural wording is allowed.

Confirmatory rule ablation is `{h6.get('status', 'inconclusive')}`. A null result removes the general rule-effect claim while retaining local diagnostic use.

External gate statuses remain `{gates.get('domain_language', {}).get('status', 'OPEN')}`, `{gates.get('comprehension', {}).get('status', 'OPEN')}`, and `{gates.get('expert_action', {}).get('status', 'OPEN')}`. No comprehension or practical-utility claim is allowed without genuine independent records.
"""
    (REPORTS / "reviewer_response_en.md").write_text(english, encoding="utf-8")


def _write_chapter_inserts(
    identity: dict[str, object],
    real: dict[str, object],
    final: dict[str, object],
    h6: dict[str, object],
    gates: dict[str, object],
    claims: dict[str, object],
) -> None:
    chapter3 = """# Вставка в главу 3: критический разрыв

Критический разрыв в программной реализации трактуется как структурный диагностический индикатор нарушения маршрута evidence. Предсказательная связь с ошибкой модели проверяется отдельным объектом результата и не предполагается автоматически.
"""
    (CHAPTER / "chapter3_structural_rupture_insert.md").write_text(chapter3, encoding="utf-8")
    supported = [row for row in claims.get("claims", []) if row.get("status") == "supported"]
    chapter4 = [
        "# Вставка в главу 4: финальный вычислительный контур",
        "",
        f"Артефакты относятся к commit `{identity.get('final_commit', 'not-generated')}`.",
        f"Статус native multiclass: `{real.get('status', 'NOT_RUN')}`.",
        f"Статус confirmatory rule ablation: `{h6.get('status', 'inconclusive')}`.",
        f"Stable release: `{gates.get('stable_release_allowed', False)}`.",
        "",
        "## Разрешённые формулировки",
        "",
    ]
    chapter4.extend(f"- {row['allowed_wording_ru']}" for row in supported)
    chapter4.extend(
        [
            "",
            "## Ограничение",
            "",
            "Пользовательская понятность, предметный язык и согласование действий не считаются подтверждёнными до фактического закрытия внешних ворот.",
        ]
    )
    (CHAPTER / "chapter4_final_evidence_insert.md").write_text("\n".join(chapter4) + "\n", encoding="utf-8")


def _write_tables(
    real: dict[str, object],
    final: dict[str, object],
    h6: dict[str, object],
    scaling: dict[str, object],
    claims: dict[str, object],
) -> None:
    with (CHAPTER / "native_multiclass.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("modality", "dataset", "classes", "objects", "evaluation_objects", "status"))
        for modality, row in real.get("modalities", {}).items():
            dataset = row["dataset"]
            writer.writerow((modality, dataset["dataset_id"], dataset["native_class_count"], dataset["n_objects"], row["evaluation_object_count"], real.get("status")))
    with (CHAPTER / "hypotheses_h1_h6.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("hypothesis", "status", "scope"))
        for key in ("H1_real", "H2_real", "H4_real"):
            writer.writerow((key, final.get(key, {}).get("status", "not_run"), "real"))
        writer.writerow(("H3_full", final.get("H3_real", {}).get("full_population_status", "not_run"), "real_full"))
        writer.writerow(("H3_hard", final.get("H3_real", {}).get("hard_case_status", "not_run"), "real_hard"))
        writer.writerow(("H5_structural", final.get("H5_real", {}).get("structural", {}).get("status", "not_run"), "structural"))
        writer.writerow(("H5_predictive", final.get("H5_real", {}).get("predictive", {}).get("status", "not_run"), "predictive"))
        writer.writerow(("H6", h6.get("status", "not_run"), "real_confirmatory"))
    with (CHAPTER / "scalability.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = scaling.get("measurements", [])
        fields = ("n_objects", "wall_time_seconds", "cpu_time_seconds", "peak_ram_bytes", "model_calls", "graph_nodes", "graph_edges", "serialized_bytes")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    with (CHAPTER / "claim_registry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("claim_id", "level", "status", "allowed_wording_ru"))
        writer.writerows((row["claim_id"], row["level"], row["status"], row["allowed_wording_ru"]) for row in claims.get("claims", []))
    with (CHAPTER / "explanation_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("modality", "method", "n_explained", "mean_fidelity", "status"))
        for modality, row in real.get("modalities", {}).items():
            for method in row.get("explainers", []):
                writer.writerow(
                    (
                        modality,
                        method.get("method"),
                        method.get("n_explained"),
                        method.get("mean_fidelity"),
                        method.get("status"),
                    )
                )
    gates = load("external/status.json").get("gates", {})
    with (CHAPTER / "external_gates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("gate", "status"))
        writer.writerows(sorted(gates.items()))


def _write_figures(final: dict[str, object], scaling: dict[str, object]) -> None:
    figures = CHAPTER / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    h3_rows = final.get("H3_real", {}).get("results", [])
    if h3_rows:
        fig, axis = plt.subplots(figsize=(8, 5))
        for population, marker in (("full", "o"), ("hard_cases", "s")):
            rows = [row for row in h3_rows if row.get("population") == population]
            axis.plot([row["coverage"] for row in rows], [row["risk"] for row in rows], marker, label=population)
        axis.set(xlabel="Automatic coverage", ylabel="Observed policy risk", title="Adaptive cascade: risk and coverage")
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "h3_risk_coverage.png", dpi=180)
        plt.close(fig)
    h4_rows = final.get("H4_real", {}).get("results", [])
    if h4_rows:
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.scatter([row["mean_complexity"] for row in h4_rows], [row["risk"] for row in h4_rows])
        for row in h4_rows:
            axis.annotate(row["representation_mode"], (row["mean_complexity"], row["risk"]))
        axis.set(xlabel="Mean representation complexity", ylabel="Observed policy risk", title="Representation hierarchy")
        fig.tight_layout()
        fig.savefig(figures / "h4_complexity_risk.png", dpi=180)
        plt.close(fig)
    measurements = scaling.get("measurements", [])
    if measurements:
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot([row["n_objects"] for row in measurements], [row["wall_time_seconds"] for row in measurements], marker="o")
        axis.set(xscale="log", yscale="log", xlabel="Objects", ylabel="Wall time, seconds", title="End-to-end scalability")
        fig.tight_layout()
        fig.savefig(figures / "scalability.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
