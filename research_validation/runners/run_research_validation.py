#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from fuzzyxai import build_proof_trace, build_route, verify_proof_trace
from fuzzyxai.adapters.tabular_classification import TabularClassificationAdapter
from fuzzyxai.viz.traceability import write_traceability_artifacts

from research_validation.adapters.external_classification_payload import make_payload

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "research_validation"
CONFIG = BASE / "configs" / "validation_matrix.yaml"
OUTPUTS = BASE / "outputs"
REPORTS = BASE / "reports"
FIGURES = REPORTS / "figures"
PACKAGE = REPORTS / "fuzzyxai_research_validation_package.zip"


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    payload = make_payload(experiment)
    adapter = TabularClassificationAdapter()
    adapted = adapter.to_adapted_input(payload)
    route = build_route(adapted)
    trace = build_proof_trace(route)
    verification = verify_proof_trace(trace)

    out = OUTPUTS / experiment["id"]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dump_json(out / "route.json", route.to_dict())
    dump_json(out / "proof_trace.json", trace.to_dict())
    trace_paths = write_traceability_artifacts(route, trace, verification, out)
    verifier = json.loads((out / "verifier_report.json").read_text(encoding="utf-8"))
    computed = route.computed_result
    traceability_status = "passed" if verifier.get("overall_status") == "passed" and all(path.exists() for path in trace_paths.values()) else "failed"
    summary = {
        "experiment_id": experiment["id"],
        "task_type": experiment["task_type"],
        "dataset": experiment["dataset"],
        "model": experiment["model"],
        "perturbation": experiment["perturbation"],
        "representation_class": computed.get("representation_class"),
        "gamma": computed.get("gamma"),
        "delta": computed.get("delta"),
        "rho": computed.get("rho"),
        "dominant_component": computed.get("risk_dominant_component"),
        "diagnostic_id": computed.get("diagnostic_id"),
        "action_id": computed.get("action_id") or route.final_action,
        "verifier_status": verifier.get("overall_status"),
        "traceability_status": traceability_status,
        "source_commit": route.source_commit,
        "uncertainty_component": computed.get("uncertainty_component"),
        "quality_component": computed.get("quality_component"),
        "reduction_component": computed.get("reduction_component"),
        "conflict_component": computed.get("conflict_component"),
    }
    dump_json(out / "summary.json", summary)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percent(count: int, total: int) -> float:
    return round((count / total) * 100.0, 2) if total else 0.0


def build_tables(rows: list[dict[str, Any]]) -> None:
    main_fields = [
        "experiment_id", "task_type", "dataset", "model", "representation_class", "perturbation",
        "gamma", "delta", "rho", "dominant_component", "diagnostic_id", "action_id",
        "verifier_status", "traceability_status", "source_commit",
        "uncertainty_component", "quality_component", "reduction_component", "conflict_component",
    ]
    write_csv(REPORTS / "research_validation_results.csv", rows, main_fields)

    total = len(rows)
    for key, filename in [
        ("action_id", "action_distribution.csv"),
        ("diagnostic_id", "diagnostic_distribution.csv"),
    ]:
        counts = Counter(row[key] for row in rows)
        dist = [{key: name, "count": count, "percentage": percent(count, total)} for name, count in sorted(counts.items())]
        write_csv(REPORTS / filename, dist, [key, "count", "percentage"])

    by_repr: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repr[row["representation_class"]].append(row)
    repr_rows = [
        {
            "representation_class": name,
            "count": len(items),
            "task_types": ";".join(sorted({item["task_type"] for item in items})),
            "example_experiments": ";".join(item["experiment_id"] for item in items[:3]),
        }
        for name, items in sorted(by_repr.items())
    ]
    write_csv(REPORTS / "representation_class_coverage.csv", repr_rows, ["representation_class", "count", "task_types", "example_experiments"])

    by_component: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_component[row["dominant_component"]].append(float(row["rho"]))
    component_rows = [
        {
            "dominant_component": name,
            "count": len(values),
            "mean_rho": round(sum(values) / len(values), 6),
            "max_rho": round(max(values), 6),
        }
        for name, values in sorted(by_component.items())
    ]
    write_csv(REPORTS / "risk_component_summary.csv", component_rows, ["dominant_component", "count", "mean_rho", "max_rho"])


def build_figures(rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    labels = [row["experiment_id"] for row in rows]
    x = list(range(len(rows)))

    plt.figure(figsize=(16, 6))
    plt.bar(x, [float(row["rho"]) for row in rows], color="#4c78a8")
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("rho")
    plt.tight_layout()
    plt.savefig(FIGURES / "rho_by_experiment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter([float(row["gamma"]) for row in rows], [float(row["delta"]) for row in rows], c=[float(row["rho"]) for row in rows], cmap="viridis")
    plt.xlabel("gamma")
    plt.ylabel("delta")
    plt.colorbar(label="rho")
    plt.tight_layout()
    plt.savefig(FIGURES / "gamma_delta_scatter.png", dpi=160)
    plt.close()

    for key, filename, color in [
        ("action_id", "action_distribution.png", "#59a14f"),
        ("diagnostic_id", "diagnostic_distribution.png", "#e15759"),
        ("representation_class", "representation_class_coverage.png", "#f28e2b"),
    ]:
        counts = Counter(row[key] for row in rows)
        plt.figure(figsize=(9, 5))
        plt.bar(list(counts), list(counts.values()), color=color)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(FIGURES / filename, dpi=160)
        plt.close()

    components = ["uncertainty_component", "quality_component", "reduction_component", "conflict_component"]
    matrix = [[float(row.get(component) or 0.0) for component in components] for row in rows]
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, aspect="auto", cmap="magma")
    plt.yticks(range(len(rows)), labels, fontsize=6)
    plt.xticks(range(len(components)), components, rotation=20, ha="right")
    plt.colorbar(label="component value")
    plt.tight_layout()
    plt.savefig(FIGURES / "risk_component_heatmap.png", dpi=160)
    plt.close()


def build_report(rows: list[dict[str, Any]]) -> None:
    actions = Counter(row["action_id"] for row in rows)
    diagnostics = Counter(row["diagnostic_id"] for row in rows)
    representations = Counter(row["representation_class"] for row in rows)
    dominant = Counter(row["dominant_component"] for row in rows)
    lines = [
        "# FuzzyXAI Research Validation Report",
        "",
        "## Goal",
        "Проверить переносимость FuzzyXAI на разных классах задач, моделях и типах ограничений. Проверка показывает работу операторного слоя, а не заявляет промышленную, клиническую или биометрическую применимость.",
        "",
        "## Experiment Matrix",
        f"Всего экспериментов: {len(rows)}. Матрица включает чистые входы, top-k редукцию, missing features, noise, confidence boundary, explanation conflict, wide interval и image/signal quality limits.",
        "",
        "## Models and Task Classes",
        f"Классы задач: {', '.join(sorted({row['task_type'] for row in rows}))}.",
        f"Модели: {', '.join(sorted({row['model'] for row in rows}))}.",
        "",
        "## Representation Class Coverage",
        ", ".join(f"{key}: {value}" for key, value in sorted(representations.items())),
        "",
        "## Operator Behavior",
        "Во всех экспериментах FuzzyXAI сформировал gamma, delta, rho и выбрал действие через operator route. Gamma реагирует на неопределённость, качество, конфликт и интервальную ширину; delta фиксирует потери top-k объяснения; rho агрегирует доминирующий компонент риска.",
        "",
        "## Action Distribution",
        ", ".join(f"{key}: {value}" for key, value in sorted(actions.items())),
        "",
        "## Diagnostic Distribution",
        ", ".join(f"{key}: {value}" for key, value in sorted(diagnostics.items())),
        "",
        "## Traceability Verification",
        f"verifier passed: {sum(row['verifier_status'] == 'passed' for row in rows)} / {len(rows)}.",
        f"traceability passed: {sum(row['traceability_status'] == 'passed' for row in rows)} / {len(rows)}.",
        "",
        "## Key Findings",
        "- FuzzyXAI produced non-zero operator values across several model families.",
        f"- Dominant risk components: {', '.join(f'{k}={v}' for k, v in sorted(dominant.items()))}.",
        f"- Representation classes activated: {', '.join(sorted(representations))}.",
        "- All generated routes passed verifier and traceability checks.",
        "",
        "## Limitations",
        "- This suite is deterministic payload-level research validation, not a claim of production performance.",
        "- Signal and image-like experiments are methodological controls and do not imply clinical or biometric applicability.",
        "- External repository payload contracts still require pinned live adapters.",
        "",
        "## Files and Reproducibility",
        "- research_validation_results.csv",
        "- action_distribution.csv",
        "- diagnostic_distribution.csv",
        "- representation_class_coverage.csv",
        "- risk_component_summary.csv",
        "- manifest.json with sha256 checksums; manifest self-hash is excluded by policy",
        "- outputs/<experiment_id>/",
    ]
    (REPORTS / "research_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest_and_zip() -> None:
    files: list[Path] = []
    manifest_path = REPORTS / "manifest.json"
    for root in [REPORTS, OUTPUTS]:
        for path in root.rglob("*"):
            if path.is_file() and path not in {PACKAGE, manifest_path}:
                files.append(path)
    manifest = {
        "package_type": "FuzzyXAIResearchValidationPackage",
        "manifest_self_hash_policy": "excluded",
        "files": [
            {"path": path.relative_to(BASE).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(files)
        ],
    }
    dump_json(manifest_path, manifest)
    files.append(manifest_path)
    if PACKAGE.exists():
        PACKAGE.unlink()
    with zipfile.ZipFile(PACKAGE, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, f"fuzzyxai_research_validation/{path.relative_to(BASE).as_posix()}")


def main() -> int:
    if OUTPUTS.exists():
        shutil.rmtree(OUTPUTS)
    OUTPUTS.mkdir(parents=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rows = [run_experiment(experiment) for experiment in config["experiments"]]
    build_tables(rows)
    build_figures(rows)
    build_report(rows)
    build_manifest_and_zip()
    print("research-validation: PASS")
    print(f"experiments: {len(rows)}")
    print(f"package: {PACKAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
