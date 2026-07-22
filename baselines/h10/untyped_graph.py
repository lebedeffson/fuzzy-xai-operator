from __future__ import annotations

from .common import BaselineResult, changed, missing


class UntypedGraphBaseline:
    name = "untyped_graph"

    def diagnose(self, route: object) -> BaselineResult:
        active = tuple(dict.fromkeys(missing(route) + changed(route)))
        if not active:
            return BaselineResult("valid", confidence=1.0)
        paths = [set(path) & set(active) for path in route.dependency_paths]
        paths = [path for path in paths if path]
        selected: list[str] = []
        while paths:
            candidates = sorted(set().union(*paths))
            node = max(candidates, key=lambda item: (sum(item in path for path in paths), item))
            selected.append(node)
            paths = [path for path in paths if node not in path]
        cut = tuple(sorted(selected or active))
        return BaselineResult("invalid", source_nodes=active, cut_nodes=cut, repair_nodes=cut, abstained=True, confidence=0.5)
