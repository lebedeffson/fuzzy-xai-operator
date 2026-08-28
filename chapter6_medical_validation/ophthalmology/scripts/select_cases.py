from __future__ import annotations

import argparse
import json
from pathlib import Path

from chapter6_medical_validation.ophthalmology.src.case_selection import select_registered_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen non-cherry-picked CH6 case rules")
    parser.add_argument("case_metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.case_metrics.read_text(encoding="utf-8"))["rows"]
    selected = select_registered_cases(rows)
    payload = {name: None if row is None else row for name, row in selected.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
