from __future__ import annotations

from typing import Type

from fuzzyxai.adapters.base import BaseAdapter
from fuzzyxai.adapters.tabular_classification import TabularClassificationAdapter


ADAPTERS: dict[str, Type[BaseAdapter]] = {
    "tabular_classification": TabularClassificationAdapter,
    "external_wine_classification": TabularClassificationAdapter,
}


def list_adapters() -> list[dict[str, object]]:
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for _, cls in sorted(ADAPTERS.items()):
        description = cls().describe()
        adapter_id = str(description["adapter_id"])
        if adapter_id not in seen:
            rows.append(description)
            seen.add(adapter_id)
    return rows


def get_adapter(adapter_id: str, **kwargs: object) -> BaseAdapter:
    try:
        adapter_cls = ADAPTERS[adapter_id]
    except KeyError as exc:
        raise KeyError(f"unknown adapter: {adapter_id}") from exc
    return adapter_cls(**kwargs) if kwargs else adapter_cls()
