from __future__ import annotations

from ..models import Case


def validate_case(case: Case) -> None:
    if not case.case_id or not case.pipeline or not case.case_hash:
        raise ValueError("case identity is incomplete")
    node_ids = [node["node_id"] for node in case.observed_route.get("nodes", ())]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate node identifiers")

