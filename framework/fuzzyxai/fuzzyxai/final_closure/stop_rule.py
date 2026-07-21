"""Finite formative-iteration contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FormativeIteration:
    iteration: int
    commit: str
    config_sha256: str
    results_sha256: str
    change_reason: str
    changes_from_previous: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.iteration not in {1, 2, 3}:
            raise ValueError("at most three formative iterations are permitted")
        if self.iteration > 1 and not self.change_reason:
            raise ValueError("later formative iterations require a frozen reason")


def next_iteration(history: tuple[FormativeIteration, ...], *, reason_predeclared: bool) -> int:
    if len(history) >= 3:
        raise RuntimeError("FORMATIVE_STOP_RULE_REACHED")
    expected = len(history) + 1
    if history and not reason_predeclared:
        raise RuntimeError("FORMATIVE_CHANGE_REASON_NOT_PREDECLARED")
    if tuple(item.iteration for item in history) != tuple(range(1, expected)):
        raise ValueError("formative iteration history is not contiguous")
    return expected
