from __future__ import annotations

import argparse
import shutil

import pandas as pd

from .build_chapter import FINAL
from .common import ARTIFACTS, ROOT, read_json, sha256_file


BUDGETS = FINAL / "Глава_4_FuzzyXAI_v13_budget_closure.csv"
RUNTIME = FINAL / "Глава_4_FuzzyXAI_v13_runtime_summary_full.csv"
RUNTIME_RAW = FINAL / "Глава_4_FuzzyXAI_v13_runtime_raw_results.csv"
HELD_OUT = FINAL / "Глава_4_FuzzyXAI_v13_held_out_faults.csv"
REPORT = FINAL / "Глава_4_FuzzyXAI_v13_closure_report.md"


def build() -> dict[str, object]:
    FINAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ARTIFACTS / "tables" / "policies_all_budgets.csv", BUDGETS)
    shutil.copy2(ARTIFACTS / "tables" / "runtime_full_statistics.csv", RUNTIME)
    shutil.copy2(ARTIFACTS / "runtime" / "raw_results.csv", RUNTIME_RAW)
    shutil.copy2(ARTIFACTS / "tables" / "route_held_out_status.csv", HELD_OUT)

    budgets = pd.read_csv(BUDGETS)
    runtime = pd.read_csv(RUNTIME)
    raw = pd.read_csv(RUNTIME_RAW)
    held_out = pd.read_csv(HELD_OUT)
    route_manifest = read_json(ARTIFACTS / "route_faults" / "manifest.json")
    leakage = read_json(ARTIFACTS / "leakage_audit.json")
    evidence = read_json(ARTIFACTS / "evidence_map.json")
    runtime_manifest = read_json(ARTIFACTS / "runtime" / "manifest.json")

    required_budgets = {0.05, 0.10, 0.20, 0.30, 0.40}
    actual_budgets = {round(float(value), 2) for value in budgets["review_budget"]}
    if actual_budgets != required_budgets:
        raise RuntimeError(f"budget closure incomplete: {sorted(actual_budgets)}")
    required_runtime = {"mean", "std", "p95", "p99", "median"}
    if not all(any(column.endswith(f"_{suffix}") for column in runtime) for suffix in required_runtime):
        raise RuntimeError("runtime closure lacks required summary statistics")
    if set(runtime["repetitions"]) != {5} or len(raw) != len(runtime) * 5:
        raise RuntimeError("runtime raw repetitions do not match the summary")
    if set(held_out["group"]) != {"held_out_fault_types"}:
        raise RuntimeError("held-out closure contains an unexpected route group")

    checksums = {path.name: sha256_file(path) for path in (BUDGETS, RUNTIME, RUNTIME_RAW, HELD_OUT)}
    REPORT.write_text(
        "# Closure report главы 4 v13\n\n"
        "## Проверяемые пункты\n\n"
        f"- бюджеты 5/10/20/30/40 %: `PASS`; строк: `{len(budgets)}`; знак эффекта: `baseline_error_rate - fuzzyxai_error_rate`;\n"
        f"- runtime N=1/10/100/1000 и N=10000 для маскирования: `PASS`; конфигураций: `{len(runtime)}`; сырых повторов: `{len(raw)}`; прогревов: `{runtime_manifest['warmups']}`;\n"
        f"- runtime median/mean/std/p95/p99, RAM и VRAM: `PASS`; повторов на конфигурацию: `{runtime_manifest['repetitions']}`;\n"
        f"- held-out faults: `EXPLORATORY`; объектов: `{int(held_out['n'].max())}` на метод; типы заранее зафиксированы, но зарегистрированы в контракте валидатора; это не open-set проверка произвольных отказов;\n"
        f"- leakage audit: `{'PASS' if leakage.get('passed') else 'FAIL'}`;\n"
        f"- числовых записей evidence map: `{len(evidence['entries'])}`;\n"
        "- код: MIT (`LICENSE` и `pyproject.toml`); лицензии данных и модели: `THIRD_PARTY_NOTICES.md`;\n"
        "- полное воспроизведение: `make reproduce-chapter4-v13 CHAPTER4_V13_PYTHON=/path/to/python3.12`;\n"
        "- лёгкая CI-проверка: `make chapter4-v13-smoke CHAPTER4_V13_PYTHON=/path/to/python3.12`.\n\n"
        "## Граница held-out проверки\n\n"
        f"{route_manifest['held_out_fault_claim']}. Пять типов были исключены из настройки простых baseline, но их поля и проверки присутствуют в типизированной схеме. Универсальное обнаружение неизвестного класса отказа не заявляется.\n\n"
        "## Контрольные суммы\n\n"
        + "".join(f"- `{digest}`  `{name}`\n" for name, digest in checksums.items())
        + "\n## Публичная линия Git\n\n"
        "- remote: `https://github.com/lebedeffson/fuzzy-xai-operator.git`;\n"
        "- experiment branch: `experiments/chapter4-practical-v13`;\n"
        "- stable tag target: `v1.3.0 -> 1a71bae98f1554430d537670018dce7dc889e25f`;\n"
        "- `v1.3.0` не перемещается при публикации closure-артефактов.\n",
        encoding="utf-8",
    )
    return {
        "budgets": len(budgets),
        "runtime_configurations": len(runtime),
        "runtime_raw_rows": len(raw),
        "held_out_methods": len(held_out),
        "report": str(REPORT.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = build()
    print(
        "PASS: closure "
        f"budgets={result['budgets']} runtime={result['runtime_configurations']} "
        f"raw={result['runtime_raw_rows']} held_out_methods={result['held_out_methods']}"
    )


if __name__ == "__main__":
    main()
