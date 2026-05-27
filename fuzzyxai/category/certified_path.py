from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CertifiedEdge:
    source: str
    target: str
    alignment: str
    gamma: float
    gamma_max: float
    passed: bool
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_critical(self) -> bool:
        return any('crit' in d.lower() for d in self.diagnostics)


@dataclass(frozen=True)
class CertifiedPath:
    source: str
    target: str
    objects: tuple[str, ...]
    edges: tuple[CertifiedEdge, ...]
    rupture: bool = False
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def identity(cls, obj: str) -> 'CertifiedPath':
        return cls(source=obj, target=obj, objects=(obj,), edges=tuple(), rupture=False, diagnostics=tuple())

    @classmethod
    def from_edge(cls, edge: CertifiedEdge) -> 'CertifiedPath':
        rupture = (not edge.passed) or (edge.gamma > edge.gamma_max)
        diagnostics = tuple(edge.diagnostics)
        if edge.gamma > edge.gamma_max and 'gamma_exceeded' not in diagnostics:
            diagnostics = diagnostics + ('gamma_exceeded',)
        if rupture and not diagnostics:
            diagnostics = ('uncertified_edge',)
        return cls(
            source=edge.source,
            target=edge.target,
            objects=(edge.source, edge.target),
            edges=(edge,),
            rupture=rupture,
            diagnostics=diagnostics,
        )

    @property
    def certified(self) -> bool:
        if self.rupture:
            return False
        for edge in self.edges:
            if (not edge.passed) or (edge.gamma > edge.gamma_max):
                return False
        return True

    def compose(self, other: 'CertifiedPath') -> 'CertifiedPath':
        if self.target != other.source:
            return CertifiedPath(
                source=self.source,
                target=other.target,
                objects=self.objects + other.objects,
                edges=self.edges + other.edges,
                rupture=True,
                diagnostics=self.diagnostics + other.diagnostics + ('path_endpoint_mismatch',),
            )
        merged_objects = self.objects + other.objects[1:]
        merged_edges = self.edges + other.edges
        rupture = self.rupture or other.rupture or any((not e.passed) or (e.gamma > e.gamma_max) for e in merged_edges)
        diagnostics = self.diagnostics + other.diagnostics
        if rupture and not diagnostics:
            diagnostics = ('uncertified_path',)
        return CertifiedPath(
            source=self.source,
            target=other.target,
            objects=merged_objects,
            edges=merged_edges,
            rupture=rupture,
            diagnostics=diagnostics,
        )


def compose_certified_paths(left: CertifiedPath, right: CertifiedPath) -> CertifiedPath:
    return left.compose(right)


def rupture_path(source: str, target: str, diagnostics: list[str] | tuple[str, ...] | None = None) -> CertifiedPath:
    diag = tuple(diagnostics or ('no_certified_edge',))
    return CertifiedPath(source=source, target=target, objects=(source, target), edges=tuple(), rupture=True, diagnostics=diag)


def paths_equivalent(left: CertifiedPath, right: CertifiedPath) -> bool:
    return left.target == right.target and left.diagnostics == right.diagnostics
