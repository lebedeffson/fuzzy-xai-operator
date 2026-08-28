from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from chapter6_medical_validation.ophthalmology.src.verbalizer import deterministic_ophthalmology_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the strict deterministic ophthalmology fallback")
    parser.add_argument("result_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result_json.read_text(encoding="utf-8"))
    prediction = SimpleNamespace(to_dict=lambda: payload["prediction"])
    system_payload = payload.get("system")
    system = None if system_payload is None else SimpleNamespace(audit_dict=lambda: system_payload)
    view_model = SimpleNamespace(claims=payload.get("claims", []))
    rendered = deterministic_ophthalmology_text(SimpleNamespace(prediction=prediction, system=system, view_model=view_model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rendered.__dict__, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
