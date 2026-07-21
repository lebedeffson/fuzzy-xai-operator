from __future__ import annotations

from pathlib import Path

from fuzzyxai.core.types import OperatorRoute, ProofTrace


def save_route_json(route: OperatorRoute, path: str | Path) -> Path:
    return route.write_json(path)


def save_proof_trace_json(trace: ProofTrace, path: str | Path) -> Path:
    return trace.write_json(path)
