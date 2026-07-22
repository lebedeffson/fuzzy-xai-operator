"""Delayed outcome vault that cannot expose labels before their release index."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DelayedOutcome:
    event_id: str
    available_at: int
    model_error: bool


class DelayedLabelStore:
    def __init__(self) -> None:
        self._records: dict[str, DelayedOutcome] = {}

    def register(self, event_id: str, *, current_index: int, delay: int, model_error: bool) -> None:
        if delay < 1:
            raise ValueError("delayed labels require a positive delay")
        self._records[event_id] = DelayedOutcome(event_id, current_index + delay, model_error)

    def open_available(self, current_index: int) -> tuple[DelayedOutcome, ...]:
        ready = tuple(record for record in self._records.values() if record.available_at <= current_index)
        for record in ready:
            del self._records[record.event_id]
        return tuple(sorted(ready, key=lambda item: (item.available_at, item.event_id)))

    @property
    def pending(self) -> int:
        return len(self._records)
