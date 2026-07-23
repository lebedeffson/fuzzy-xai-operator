from __future__ import annotations


def safety_rates(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"false_certification": 0.0, "false_block": 0.0}
    return {
        "false_certification": sum(bool(row.get("false_certification")) for row in rows) / len(rows),
        "false_block": sum(bool(row.get("false_block")) for row in rows) / len(rows),
    }

