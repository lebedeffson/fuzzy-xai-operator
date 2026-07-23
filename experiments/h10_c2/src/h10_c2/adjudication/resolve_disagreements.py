from __future__ import annotations


def unresolved(case_ids: list[str], resolutions: dict[str, str]) -> list[str]:
    return [case_id for case_id in case_ids if resolutions.get(case_id) not in {"resolved", "excluded"}]

