from __future__ import annotations

from .common import BaselineResult, missing


class SchemaOnlyBaseline:
    name = "schema_only"

    def diagnose(self, route: object) -> BaselineResult:
        absent = missing(route)
        if absent:
            return BaselineResult("insufficient_evidence", source_nodes=absent, cut_nodes=absent, repair_nodes=absent, abstained=True)
        return BaselineResult("valid", confidence=1.0)
