"""Shared paths and evidence helpers for the Chapter 4 build."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORMATIVE = ROOT / "release_evidence/strong_confirmatory/formative"
OUTPUT = ROOT / "dissertation_artifacts/strong_confirmatory/chapter4_formative"
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
EXPERIMENTS = (
    "H3_v2_selective",
    "H5_A_route_validity",
    "H6_A_planted_rules",
    "H7_stability",
    "H8_grid_sensitivity",
    "H9_scalability",
)


def prepare() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def load_experiment(experiment: str) -> dict[str, object]:
    path = FORMATIVE / f"{experiment}.json"
    if not path.is_file():
        raise SystemExit(f"FAIL: missing formative evidence {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_ref(experiment: str) -> dict[str, str]:
    path = FORMATIVE / f"{experiment}.json"
    return {
        "experiment_id": experiment,
        "evidence_id": f"FORMATIVE-{experiment}",
        "source_file": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
