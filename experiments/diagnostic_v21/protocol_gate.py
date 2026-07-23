from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def validate_draft(protocol: str | Path) -> dict[str, object]:
    payload = yaml.safe_load(Path(protocol).read_text(encoding="utf-8"))
    blockers = []
    if payload.get("status") != "draft_blocked_manual_adjudication":
        blockers.append("protocol status does not preserve the preconfirmatory boundary")
    if payload.get("sealed_opening_count") != 0:
        blockers.append("sealed opening count must remain zero")
    if not payload.get("manual_adjudication", {}).get("required"):
        blockers.append("two-reviewer adjudication must be required")
    if payload.get("confirmatory", {}).get("scoring_enabled"):
        blockers.append("confirmatory scoring must remain disabled")
    return {
        "status": "PASS" if not blockers else "FAIL",
        "protocol": str(protocol),
        "blockers": blockers,
        "scientific_status": "BLOCKED_PRECONFIRMATORY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="config/h10_c2_diagnostic_cut_protocol.yaml")
    args = parser.parse_args()
    result = validate_draft(args.protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
