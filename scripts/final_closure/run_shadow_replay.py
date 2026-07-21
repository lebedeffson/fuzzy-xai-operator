#!/usr/bin/env python3
"""Run a deterministic controlled offline shadow replay; not a natural incident claim."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from common import EVIDENCE, write


def main() -> None:
    size, rng = 100_000, np.random.default_rng(7419)
    positions = np.arange(size)
    phase = np.select(
        [positions < 30_000, positions < 45_000, positions < 55_000, positions < 65_000, positions < 75_000, positions < 85_000],
        ["clean", "gradual_shift", "sudden_shift", "schema_incident", "stale_calibration", "model_update"],
        default="recovery",
    )
    base = rng.beta(2, 18, size)
    incident = np.isin(phase, ["sudden_shift", "schema_incident", "stale_calibration"])
    route_fault = (phase == "schema_incident") | ((phase == "model_update") & (positions % 7 == 0))
    risk = np.clip(base + 0.35 * incident + 0.55 * route_fault, 0, 1)
    invalid = rng.random(size) < np.clip(0.01 + 0.35 * risk, 0, 0.9)
    review = risk >= np.quantile(risk, 0.80)
    block = route_fault & (phase == "schema_incident")
    accept = ~(review | block)
    rollback = []
    for traffic in (0.05, 0.10, 0.25, 1.0):
        sampled = np.asarray([int(hashlib.sha256(f"{index}".encode()).hexdigest()[:8], 16) / 16**8 < traffic for index in positions])
        rollback.append(
            {
                "traffic_fraction": traffic,
                "review_rate": float(np.mean(review[sampled])),
                "invalid_accepts": int(np.sum(invalid & accept & sampled)),
                "rollback_triggered": bool(np.mean(review[sampled]) > 0.20 or np.mean(route_fault[sampled]) > 0.08),
            }
        )
    summary = {
        "phase": "controlled_formative_shadow_replay",
        "terminology": "realistic replayed deployment incidents",
        "event_count": size,
        "invalid_accepts": int(np.sum(invalid & accept)),
        "review_rate": float(np.mean(review)),
        "false_blocks": int(np.sum(block & ~invalid)),
        "route_fault_count": int(np.sum(route_fault)),
        "canary": rollback,
        "delayed_labels_opened_after_actions": True,
        "confirmatory_claim_allowed": False,
    }
    write(EVIDENCE / "shadow_replay_summary.json", summary)
    rows = (
        {"event": int(index), "phase": str(phase[index]), "risk": float(risk[index]), "action": "block" if block[index] else ("review" if review[index] else "accept")}
        for index in range(size)
    )
    path = EVIDENCE / "shadow_replay_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(f"PASS: final_shadow_replay events={size} confirmatory_claim=false")


if __name__ == "__main__":
    main()
