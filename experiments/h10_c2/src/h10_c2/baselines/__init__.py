from . import (
    greedy_cover,
    independent_if_else,
    schema_only,
    simple_or,
    typed_without_cut,
    untyped_graph,
    weighted_greedy,
)

METHODS = {
    "schema_only": schema_only.run,
    "simple_or": simple_or.run,
    "independent_if_else": independent_if_else.run,
    "untyped_graph": untyped_graph.run,
    "typed_without_cut": typed_without_cut.run,
    "greedy_cover": greedy_cover.run,
    "weighted_greedy": weighted_greedy.run,
}

__all__ = ["METHODS"]

