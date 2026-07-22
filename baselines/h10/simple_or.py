from __future__ import annotations

from .common import BaselineResult, changed, missing


class SimpleOrBaseline:
    name = "simple_or"

    def diagnose(self, route: object) -> BaselineResult:
        absent = missing(route)
        differences = changed(route)
        active = tuple(dict.fromkeys(absent + differences))
        if not active:
            return BaselineResult("valid", confidence=1.0)
        return BaselineResult(
            "insufficient_evidence" if absent else "invalid",
            source_nodes=active,
            cut_nodes=active,
            repair_nodes=active,
            abstained=bool(absent),
            confidence=1.0,
        )
