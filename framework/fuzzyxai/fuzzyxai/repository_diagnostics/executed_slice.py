from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .runtime_events import RuntimeEvent


@dataclass(frozen=True)
class SymbolRef:
    file_path: str
    symbol: str | None


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    access: str
    owner: SymbolRef


@dataclass(frozen=True)
class ConfigRef:
    path: str
    key: str | None
    owner: SymbolRef


@dataclass(frozen=True)
class ExecutedSlice:
    failing_test: str
    traceback_symbols: tuple[SymbolRef, ...]
    executed_symbols: tuple[SymbolRef, ...]
    loaded_modules: tuple[str, ...]
    accessed_artifacts: tuple[ArtifactRef, ...]
    configuration_reads: tuple[ConfigRef, ...]
    dependency_versions: tuple[tuple[str, str], ...]

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class ExecutedSliceBuilder:
    """Build one deterministic executed slice per registered failing test."""

    def build(
        self,
        events: tuple[RuntimeEvent, ...],
    ) -> tuple[ExecutedSlice, ...]:
        tests = sorted({event.test_id for event in events})
        return tuple(self._for_test(test_id, events) for test_id in tests)

    @staticmethod
    def _for_test(
        test_id: str,
        events: tuple[RuntimeEvent, ...],
    ) -> ExecutedSlice:
        selected = tuple(event for event in events if event.test_id == test_id)
        traceback_symbols = _symbols(event for event in selected if event.kind == "traceback_frame")
        executed_symbols = _symbols(event for event in selected if event.kind in {"call", "coverage", "traceback_frame"})
        loaded_modules = tuple(
            sorted(
                {str(event.target_symbol or event.target_file) for event in selected if event.kind == "import" and (event.target_symbol or event.target_file)}
            )
        )
        artifacts = tuple(
            sorted(
                (
                    ArtifactRef(
                        str(event.target_file),
                        event.kind,
                        SymbolRef(event.source_file, event.source_symbol),
                    )
                    for event in selected
                    if event.kind in {"read", "write"} and event.target_file
                ),
                key=lambda item: (item.path, item.access, item.owner.file_path),
            )
        )
        configurations = tuple(
            sorted(
                (
                    ConfigRef(
                        str(event.target_file),
                        event.target_symbol,
                        SymbolRef(event.source_file, event.source_symbol),
                    )
                    for event in selected
                    if event.kind == "config_read" and event.target_file
                ),
                key=lambda item: (item.path, item.key or "", item.owner.file_path),
            )
        )
        dependencies = tuple(
            sorted(
                {
                    (
                        str(event.target_symbol or event.target_file),
                        event.detail,
                    )
                    for event in selected
                    if event.kind == "dependency" and (event.target_symbol or event.target_file)
                }
            )
        )
        return ExecutedSlice(
            test_id,
            traceback_symbols,
            executed_symbols,
            loaded_modules,
            artifacts,
            configurations,
            dependencies,
        )


def _symbols(events: Iterable[RuntimeEvent]) -> tuple[SymbolRef, ...]:
    return tuple(
        sorted(
            {SymbolRef(event.source_file, event.source_symbol) for event in events},
            key=lambda item: (item.file_path, item.symbol or ""),
        )
    )
