from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a chapter-ready report only from measured artifacts")
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    required = [args.artifact_root / "aggregate_metrics.json", args.artifact_root / "selected_cases.json"]
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MISSING_EVIDENCE: cannot build report; absent {missing}")
    metrics = json.loads(required[0].read_text(encoding="utf-8"))
    cases = json.loads(required[1].read_text(encoding="utf-8"))
    text = (
        "# Результаты офтальмологического эксперимента\n\n"
        "Отчёт построен программно из измеренных артефактов. Выход модели не является клиническим диагнозом.\n\n"
        f"## Метрики модели\n\n```json\n{json.dumps(metrics, ensure_ascii=False, indent=2)}\n```\n\n"
        f"## Алгоритмически выбранные случаи\n\n```json\n{json.dumps(cases, ensure_ascii=False, indent=2)}\n```\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
